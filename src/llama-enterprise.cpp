#include "llama-enterprise.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>

// Global instances with thread-safe access
static llama_request_queue * g_request_queue = nullptr;
static llama_rate_limiter * g_rate_limiter = nullptr;
static llama_health_monitor * g_health_monitor = nullptr;
static llama_audit_logger * g_audit_logger = nullptr;
static std::mutex g_enterprise_globals_mutex;

llama_request_queue * llama_get_request_queue() {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    return g_request_queue;
}
void llama_set_request_queue(llama_request_queue * q) {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    g_request_queue = q;
}

llama_rate_limiter * llama_get_rate_limiter() {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    return g_rate_limiter;
}
void llama_set_rate_limiter(llama_rate_limiter * l) {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    g_rate_limiter = l;
}

llama_health_monitor * llama_get_health_monitor() {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    return g_health_monitor;
}
void llama_set_health_monitor(llama_health_monitor * m) {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    g_health_monitor = m;
}

llama_audit_logger * llama_get_audit_logger() {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    return g_audit_logger;
}
void llama_set_audit_logger(llama_audit_logger * l) {
    std::lock_guard<std::mutex> lock(g_enterprise_globals_mutex);
    g_audit_logger = l;
}

//
// Request Queue Implementation
//

llama_request_queue::llama_request_queue() = default;

llama_request_queue::llama_request_queue(const config & cfg) : cfg_(cfg) {}

llama_request_queue::~llama_request_queue() {
    if (g_request_queue == this) g_request_queue = nullptr;
}

bool llama_request_queue::enqueue(llama_request request) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check queue size
    size_t total_pending = 0;
    for (const auto & q : priority_queues_) {
        total_pending += q.size();
    }
    if (total_pending >= cfg_.max_queue_size) {
        return false;
    }

    request.enqueue_time_us = get_time_us();

    int priority_idx = static_cast<int>(request.priority);
    priority_queues_[priority_idx].push_back(std::move(request));

    cv_.notify_one();
    return true;
}

std::optional<llama_request> llama_request_queue::dequeue(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Check concurrent limit
    if (active_requests_.size() >= cfg_.max_concurrent) {
        if (timeout_ms == 0) return std::nullopt;

        auto pred = [this]() {
            return active_requests_.size() < cfg_.max_concurrent;
        };

        if (timeout_ms < 0) {
            cv_.wait(lock, pred);
        } else {
            if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), pred)) {
                return std::nullopt;
            }
        }
    }

    // Find next request (highest priority first)
    for (int p = 3; p >= 0; --p) {
        auto & queue = priority_queues_[p];
        if (!queue.empty()) {
            llama_request req;

            if (cfg_.enable_fair_scheduling && !queue.empty()) {
                // Fair scheduling: prefer clients that haven't been served recently
                auto it = std::min_element(queue.begin(), queue.end(),
                    [this](const llama_request & a, const llama_request & b) {
                        auto a_it = client_last_served_.find(a.client_id);
                        auto b_it = client_last_served_.find(b.client_id);
                        int64_t a_time = (a_it != client_last_served_.end()) ? a_it->second : 0;
                        int64_t b_time = (b_it != client_last_served_.end()) ? b_it->second : 0;
                        return a_time < b_time;
                    });
                req = std::move(*it);
                queue.erase(it);
            } else {
                req = std::move(queue.front());
                queue.pop_front();
            }

            req.start_time_us = get_time_us();
            total_wait_time_us_ += (req.start_time_us - req.enqueue_time_us);

            active_requests_[req.request_id] = req;
            client_last_served_[req.client_id] = req.start_time_us;

            return req;
        }
    }

    return std::nullopt;
}

void llama_request_queue::complete(const std::string & request_id, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_requests_.find(request_id);
    if (it == active_requests_.end()) return;

    auto & req = it->second;
    req.end_time_us = get_time_us();
    total_processing_time_us_ += (req.end_time_us - req.start_time_us);

    if (success) {
        total_completed_++;
    } else {
        total_failed_++;
    }

    active_requests_.erase(it);
    cv_.notify_one();
}

bool llama_request_queue::cancel(const std::string & request_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check active requests
    auto it = active_requests_.find(request_id);
    if (it != active_requests_.end()) {
        active_requests_.erase(it);
        total_cancelled_++;
        cv_.notify_one();
        return true;
    }

    // Check pending queues
    for (auto & queue : priority_queues_) {
        auto qit = std::find_if(queue.begin(), queue.end(),
            [&](const llama_request & r) { return r.request_id == request_id; });
        if (qit != queue.end()) {
            queue.erase(qit);
            total_cancelled_++;
            return true;
        }
    }

    return false;
}

llama_request_queue::queue_stats llama_request_queue::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    queue_stats stats;
    for (const auto & q : priority_queues_) {
        stats.pending_requests += q.size();
    }
    stats.active_requests = active_requests_.size();
    stats.completed_requests = total_completed_.load();
    stats.failed_requests = total_failed_.load();
    stats.cancelled_requests = total_cancelled_.load();

    size_t total = stats.completed_requests + stats.failed_requests;
    if (total > 0) {
        stats.avg_wait_time_ms = (total_wait_time_us_.load() / total) / 1000.0;
        stats.avg_processing_time_ms = (total_processing_time_us_.load() / total) / 1000.0;
    }

    return stats;
}

std::vector<llama_request> llama_request_queue::get_client_requests(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<llama_request> result;

    for (const auto & q : priority_queues_) {
        for (const auto & req : q) {
            if (req.client_id == client_id) {
                result.push_back(req);
            }
        }
    }

    for (const auto & [id, req] : active_requests_) {
        if (req.client_id == client_id) {
            result.push_back(req);
        }
    }

    return result;
}

int64_t llama_request_queue::get_time_us() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

//
// Rate Limiter Implementation
//

llama_rate_limiter::llama_rate_limiter() = default;

llama_rate_limiter::llama_rate_limiter(const config & cfg) : cfg_(cfg) {}

bool llama_rate_limiter::allow_request(const std::string & client_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = get_time_ms();
    auto & state = client_states_[client_id];

    // Initialize limits if new client
    if (state.limits.requests_per_minute == 0) {
        state.limits.requests_per_minute = cfg_.requests_per_minute;
        state.limits.tokens_per_minute = cfg_.tokens_per_minute;
        state.limits.max_concurrent = cfg_.max_concurrent_requests;
    }

    reset_window_if_needed(state, now);

    // Check concurrent limit
    if (state.active_requests >= state.limits.max_concurrent) {
        return false;
    }

    // Check rate limit
    if (state.requests_this_minute >= state.limits.requests_per_minute) {
        // Check burst
        if (cfg_.enable_burst && state.requests_this_minute < state.limits.requests_per_minute + cfg_.burst_size) {
            state.requests_this_minute++;
            state.active_requests++;
            return true;
        }
        return false;
    }

    state.requests_this_minute++;
    state.active_requests++;
    return true;
}

bool llama_rate_limiter::allow_tokens(const std::string & client_id, int token_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = get_time_ms();
    auto & state = client_states_[client_id];
    reset_window_if_needed(state, now);

    return (state.tokens_this_minute + token_count) <= state.limits.tokens_per_minute;
}

void llama_rate_limiter::record_tokens(const std::string & client_id, int token_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto & state = client_states_[client_id];
    state.tokens_this_minute += token_count;

    if (state.active_requests > 0) {
        state.active_requests--;
    }
}

void llama_rate_limiter::set_client_limits(const std::string & client_id, const client_limits & limits) {
    std::lock_guard<std::mutex> lock(mutex_);
    client_states_[client_id].limits = limits;
}

llama_rate_limiter::client_usage llama_rate_limiter::get_client_usage(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    client_usage usage;
    auto it = client_states_.find(client_id);
    if (it != client_states_.end()) {
        usage.requests_this_minute = it->second.requests_this_minute;
        usage.tokens_this_minute = it->second.tokens_this_minute;
        usage.active_requests = it->second.active_requests;
        usage.window_reset_time_ms = it->second.window_start_ms + 60000;
    }
    return usage;
}

int llama_rate_limiter::get_remaining_requests(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = client_states_.find(client_id);
    if (it == client_states_.end()) {
        return cfg_.requests_per_minute;
    }

    int limit = it->second.limits.requests_per_minute;
    int used = it->second.requests_this_minute;
    return std::max(0, limit - used);
}

int llama_rate_limiter::get_remaining_tokens(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = client_states_.find(client_id);
    if (it == client_states_.end()) {
        return cfg_.tokens_per_minute;
    }

    int limit = it->second.limits.tokens_per_minute;
    int used = it->second.tokens_this_minute;
    return std::max(0, limit - used);
}

void llama_rate_limiter::reset_window_if_needed(client_state & state, int64_t now_ms) const {
    if (state.window_start_ms == 0 || (now_ms - state.window_start_ms) >= 60000) {
        state.window_start_ms = now_ms;
        state.requests_this_minute = 0;
        state.tokens_this_minute = 0;
    }
}

int64_t llama_rate_limiter::get_time_ms() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

//
// Health Monitor Implementation
//

llama_health_monitor::llama_health_monitor() {
    start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

llama_health_monitor::llama_health_monitor(const config & cfg)
    : cfg_(cfg) {
    start_time_ms_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

llama_health_monitor::~llama_health_monitor() {
    stop();
    if (g_health_monitor == this) g_health_monitor = nullptr;
}

void llama_health_monitor::start() {
    if (running_) return;
    running_ = true;
    monitor_thread_ = std::thread(&llama_health_monitor::monitor_thread, this);
}

void llama_health_monitor::stop() {
    running_ = false;
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }
}

llama_health_status llama_health_monitor::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);

    for (const auto & [name, check] : check_results_) {
        if (check.status == llama_health_status::UNHEALTHY) {
            return llama_health_status::UNHEALTHY;
        }
    }

    for (const auto & [name, check] : check_results_) {
        if (check.status == llama_health_status::DEGRADED) {
            return llama_health_status::DEGRADED;
        }
    }

    return llama_health_status::HEALTHY;
}

llama_health_monitor::health_report llama_health_monitor::get_report() const {
    std::lock_guard<std::mutex> lock(mutex_);

    health_report report;
    report.overall_status = get_status();
    report.version = "enterprise-1.0.0";

    int64_t now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    report.uptime_seconds = (now_ms - start_time_ms_) / 1000;

    for (const auto & [name, check] : check_results_) {
        report.checks.push_back(check);
    }

    return report;
}

void llama_health_monitor::register_check(const std::string & name,
                                           std::function<health_check()> check_fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    custom_checks_[name] = std::move(check_fn);
}

void llama_health_monitor::set_check_status(const std::string & name,
                                             llama_health_status status,
                                             const std::string & message) {
    std::lock_guard<std::mutex> lock(mutex_);

    health_check check;
    check.name = name;
    check.status = status;
    check.message = message;
    check.last_check_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    check_results_[name] = check;
}

void llama_health_monitor::monitor_thread() {
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            // Run custom checks
            for (const auto & [name, check_fn] : custom_checks_) {
                check_results_[name] = check_fn();
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.check_interval_ms));
    }
}

//
// Audit Logger Implementation
//

llama_audit_logger::llama_audit_logger() {
    if (cfg_.async_logging) {
        writer_thread_ = std::thread(&llama_audit_logger::writer_thread, this);
    }
}

llama_audit_logger::llama_audit_logger(const config & cfg) : cfg_(cfg) {
    if (cfg_.async_logging) {
        writer_thread_ = std::thread(&llama_audit_logger::writer_thread, this);
    }
}

llama_audit_logger::~llama_audit_logger() {
    running_ = false;
    cv_.notify_all();
    if (writer_thread_.joinable()) {
        writer_thread_.join();
    }
    if (g_audit_logger == this) g_audit_logger = nullptr;
}

void llama_audit_logger::log(const llama_audit_event & event) {
    llama_audit_event e = event;
    if (e.timestamp_ms == 0) {
        e.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    if (e.event_id.empty()) {
        e.event_id = generate_event_id();
    }

    if (cfg_.async_logging) {
        std::lock_guard<std::mutex> lock(mutex_);
        write_queue_.push_back(e);
        cv_.notify_one();
    } else {
        std::string formatted = format_event(e);
        if (cfg_.log_to_stdout) {
            std::cout << formatted << std::endl;
        }
        if (!cfg_.log_file.empty()) {
            std::ofstream file(cfg_.log_file, std::ios::app);
            if (file) {
                file << formatted << std::endl;
            }
        }
    }

    // Keep in memory buffer
    {
        std::lock_guard<std::mutex> lock(mutex_);
        event_buffer_.push_back(e);
        while (event_buffer_.size() > MAX_BUFFER_SIZE) {
            event_buffer_.pop_front();
        }
    }
}

void llama_audit_logger::log_request_received(const std::string & client_id,
                                               const std::string & request_id) {
    llama_audit_event event;
    event.type = llama_audit_event_type::REQUEST_RECEIVED;
    event.client_id = client_id;
    event.request_id = request_id;
    log(event);
}

void llama_audit_logger::log_request_completed(const std::string & client_id,
                                                const std::string & request_id,
                                                int tokens,
                                                double latency_ms) {
    llama_audit_event event;
    event.type = llama_audit_event_type::REQUEST_COMPLETED;
    event.client_id = client_id;
    event.request_id = request_id;
    event.metadata["tokens"] = std::to_string(tokens);
    event.metadata["latency_ms"] = std::to_string(latency_ms);
    log(event);
}

void llama_audit_logger::log_error(const std::string & message,
                                    const std::string & request_id) {
    llama_audit_event event;
    event.type = llama_audit_event_type::ERROR;
    event.request_id = request_id;
    event.message = message;
    log(event);
}

void llama_audit_logger::log_rate_limited(const std::string & client_id) {
    llama_audit_event event;
    event.type = llama_audit_event_type::RATE_LIMITED;
    event.client_id = client_id;
    log(event);
}

void llama_audit_logger::log_auth(const std::string & client_id, bool success) {
    llama_audit_event event;
    event.type = success ? llama_audit_event_type::AUTH_SUCCESS : llama_audit_event_type::AUTH_FAILURE;
    event.client_id = client_id;
    log(event);
}

std::vector<llama_audit_event> llama_audit_logger::query(
        int64_t start_time_ms,
        int64_t end_time_ms,
        const std::string & client_id,
        llama_audit_event_type type) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<llama_audit_event> result;
    for (const auto & e : event_buffer_) {
        if (e.timestamp_ms < start_time_ms) continue;
        if (end_time_ms > 0 && e.timestamp_ms > end_time_ms) continue;
        if (!client_id.empty() && e.client_id != client_id) continue;

        result.push_back(e);
    }
    return result;
}

void llama_audit_logger::writer_thread() {
    while (running_) {
        std::vector<llama_audit_event> events_to_write;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
                return !write_queue_.empty() || !running_;
            });

            while (!write_queue_.empty()) {
                events_to_write.push_back(std::move(write_queue_.front()));
                write_queue_.pop_front();
            }
        }

        for (const auto & e : events_to_write) {
            std::string formatted = format_event(e);
            if (cfg_.log_to_stdout) {
                std::cout << formatted << std::endl;
            }
            if (!cfg_.log_file.empty()) {
                std::ofstream file(cfg_.log_file, std::ios::app);
                if (file) {
                    file << formatted << std::endl;
                }
            }
        }
    }
}

std::string llama_audit_logger::format_event(const llama_audit_event & event) const {
    std::ostringstream oss;

    // ISO 8601 timestamp
    std::time_t time_sec = event.timestamp_ms / 1000;
    int time_ms = event.timestamp_ms % 1000;
    std::tm * tm_info = std::gmtime(&time_sec);
    char timestamp_buf[32];
    std::strftime(timestamp_buf, sizeof(timestamp_buf), "%Y-%m-%dT%H:%M:%S", tm_info);

    oss << "{";
    oss << "\"timestamp\":\"" << timestamp_buf << "." << std::setfill('0') << std::setw(3) << time_ms << "Z\",";
    oss << "\"event_id\":\"" << event.event_id << "\",";
    oss << "\"type\":\"" << static_cast<int>(event.type) << "\",";

    if (!event.client_id.empty()) {
        oss << "\"client_id\":\"" << event.client_id << "\",";
    }
    if (!event.request_id.empty()) {
        oss << "\"request_id\":\"" << event.request_id << "\",";
    }
    if (!event.model_id.empty()) {
        oss << "\"model_id\":\"" << event.model_id << "\",";
    }
    if (!event.message.empty()) {
        oss << "\"message\":\"" << event.message << "\",";
    }

    if (!event.metadata.empty()) {
        oss << "\"metadata\":{";
        bool first = true;
        for (const auto & [k, v] : event.metadata) {
            if (!first) oss << ",";
            oss << "\"" << k << "\":\"" << v << "\"";
            first = false;
        }
        oss << "},";
    }

    // Remove trailing comma and close
    std::string result = oss.str();
    if (result.back() == ',') {
        result.pop_back();
    }
    result += "}";

    return result;
}

std::string llama_audit_logger::generate_event_id() const {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << dis(gen);
    return oss.str();
}

//
// RBAC Implementation
//

llama_rbac::llama_rbac() {
    init_default_roles();
}

void llama_rbac::init_default_roles() {
    // Default user role
    llama_role user_role;
    user_role.name = "user";
    user_role.permissions = {llama_permission::INFERENCE, llama_permission::METRICS_READ};
    add_role(user_role);

    // Power user role
    llama_role power_role;
    power_role.name = "power_user";
    power_role.permissions = {
        llama_permission::INFERENCE,
        llama_permission::CONFIG_READ,
        llama_permission::METRICS_READ
    };
    power_role.rate_limit_requests = 120;
    power_role.rate_limit_tokens = 200000;
    add_role(power_role);

    // Admin role
    llama_role admin_role;
    admin_role.name = "admin";
    admin_role.permissions = {llama_permission::ADMIN};
    add_role(admin_role);
}

void llama_rbac::add_role(const llama_role & role) {
    std::lock_guard<std::mutex> lock(mutex_);
    roles_[role.name] = role;
}

void llama_rbac::remove_role(const std::string & role_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    roles_.erase(role_name);
}

std::optional<llama_role> llama_rbac::get_role(const std::string & role_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = roles_.find(role_name);
    if (it == roles_.end()) return std::nullopt;
    return it->second;
}

std::vector<llama_role> llama_rbac::get_all_roles() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<llama_role> result;
    for (const auto & [name, role] : roles_) {
        result.push_back(role);
    }
    return result;
}

void llama_rbac::assign_role(const std::string & client_id, const std::string & role_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & roles = client_roles_[client_id];
    if (std::find(roles.begin(), roles.end(), role_name) == roles.end()) {
        roles.push_back(role_name);
    }
}

void llama_rbac::revoke_role(const std::string & client_id, const std::string & role_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto & roles = client_roles_[client_id];
    roles.erase(std::remove(roles.begin(), roles.end(), role_name), roles.end());
}

std::vector<std::string> llama_rbac::get_client_roles(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = client_roles_.find(client_id);
    if (it == client_roles_.end()) return {"user"};  // Default role
    return it->second;
}

bool llama_rbac::has_permission(const std::string & client_id, llama_permission permission) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto client_it = client_roles_.find(client_id);
    std::vector<std::string> role_names = (client_it != client_roles_.end()) ?
        client_it->second : std::vector<std::string>{"user"};

    for (const auto & role_name : role_names) {
        auto role_it = roles_.find(role_name);
        if (role_it == roles_.end()) continue;

        const auto & perms = role_it->second.permissions;

        // Admin has all permissions
        if (std::find(perms.begin(), perms.end(), llama_permission::ADMIN) != perms.end()) {
            return true;
        }

        if (std::find(perms.begin(), perms.end(), permission) != perms.end()) {
            return true;
        }
    }

    return false;
}

bool llama_rbac::can_access_model(const std::string & client_id, const std::string & model_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto client_it = client_roles_.find(client_id);
    std::vector<std::string> role_names = (client_it != client_roles_.end()) ?
        client_it->second : std::vector<std::string>{"user"};

    for (const auto & role_name : role_names) {
        auto role_it = roles_.find(role_name);
        if (role_it == roles_.end()) continue;

        // Empty allowed_models means all models allowed
        if (role_it->second.allowed_models.empty()) {
            return true;
        }

        if (std::find(role_it->second.allowed_models.begin(),
                      role_it->second.allowed_models.end(),
                      model_id) != role_it->second.allowed_models.end()) {
            return true;
        }
    }

    return false;
}

int llama_rbac::get_rate_limit_requests(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    int max_limit = -1;
    auto client_it = client_roles_.find(client_id);
    std::vector<std::string> role_names = (client_it != client_roles_.end()) ?
        client_it->second : std::vector<std::string>{"user"};

    for (const auto & role_name : role_names) {
        auto role_it = roles_.find(role_name);
        if (role_it != roles_.end() && role_it->second.rate_limit_requests > max_limit) {
            max_limit = role_it->second.rate_limit_requests;
        }
    }

    return max_limit;
}

int llama_rbac::get_rate_limit_tokens(const std::string & client_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    int max_limit = -1;
    auto client_it = client_roles_.find(client_id);
    std::vector<std::string> role_names = (client_it != client_roles_.end()) ?
        client_it->second : std::vector<std::string>{"user"};

    for (const auto & role_name : role_names) {
        auto role_it = roles_.find(role_name);
        if (role_it != roles_.end() && role_it->second.rate_limit_tokens > max_limit) {
            max_limit = role_it->second.rate_limit_tokens;
        }
    }

    return max_limit;
}

//
// Content Filter Implementation
//

llama_content_filter::llama_content_filter() = default;

llama_content_filter::llama_content_filter(const config & cfg) : cfg_(cfg) {}

llama_content_filter::filter_result llama_content_filter::filter_input(const std::string & text) {
    if (!cfg_.filter_input) {
        return {true, "", {}, 0.0};
    }

    filter_result result;
    result.passed = true;

    if (contains_blocked_word(text)) {
        result.passed = !cfg_.block_on_violation;
        result.reason = "Blocked word detected";
        result.violations.push_back("blocked_word");
    }

    if (matches_blocked_pattern(text)) {
        result.passed = !cfg_.block_on_violation;
        result.reason = "Blocked pattern detected";
        result.violations.push_back("blocked_pattern");
    }

    return result;
}

llama_content_filter::filter_result llama_content_filter::filter_output(const std::string & text) {
    if (!cfg_.filter_output) {
        return {true, "", {}, 0.0};
    }

    return filter_input(text);  // Same logic for now
}

void llama_content_filter::add_blocked_pattern(const std::string & pattern) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_.blocked_patterns.push_back(pattern);
}

void llama_content_filter::add_blocked_word(const std::string & word) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_.blocked_words.push_back(word);
}

void llama_content_filter::clear_filters() {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_.blocked_patterns.clear();
    cfg_.blocked_words.clear();
}

bool llama_content_filter::contains_blocked_word(const std::string & text) const {
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);

    for (const auto & word : cfg_.blocked_words) {
        std::string lower_word = word;
        std::transform(lower_word.begin(), lower_word.end(), lower_word.begin(), ::tolower);
        if (lower_text.find(lower_word) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool llama_content_filter::matches_blocked_pattern(const std::string & text) const {
    for (const auto & pattern : cfg_.blocked_patterns) {
        try {
            std::regex re(pattern, std::regex::icase);
            if (std::regex_search(text, re)) {
                return true;
            }
        } catch (...) {
            // Invalid regex, skip
        }
    }
    return false;
}

//
// Cost Tracker Implementation
//

llama_cost_tracker::llama_cost_tracker() = default;

void llama_cost_tracker::set_pricing(const std::string & model_id, const pricing & p) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_pricing_[model_id] = p;
}

void llama_cost_tracker::record_usage(const llama_cost_record & record) {
    std::lock_guard<std::mutex> lock(mutex_);

    records_.push_back(record);
    while (records_.size() > MAX_RECORDS) {
        records_.pop_front();
    }
}

llama_cost_tracker::client_costs llama_cost_tracker::get_client_costs(
        const std::string & client_id,
        int64_t start_time_ms,
        int64_t end_time_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);

    client_costs costs;
    for (const auto & record : records_) {
        if (record.client_id != client_id) continue;
        if (start_time_ms > 0 && record.timestamp_ms < start_time_ms) continue;
        if (end_time_ms > 0 && record.timestamp_ms > end_time_ms) continue;

        costs.total_cost += record.cost_credits;
        costs.total_prompt_tokens += record.prompt_tokens;
        costs.total_completion_tokens += record.completion_tokens;
        costs.total_gpu_seconds += record.gpu_seconds;
        costs.request_count++;
    }
    return costs;
}

std::map<std::string, llama_cost_tracker::client_costs> llama_cost_tracker::get_all_costs(
        int64_t start_time_ms,
        int64_t end_time_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, client_costs> result;
    for (const auto & record : records_) {
        if (start_time_ms > 0 && record.timestamp_ms < start_time_ms) continue;
        if (end_time_ms > 0 && record.timestamp_ms > end_time_ms) continue;

        auto & costs = result[record.client_id];
        costs.total_cost += record.cost_credits;
        costs.total_prompt_tokens += record.prompt_tokens;
        costs.total_completion_tokens += record.completion_tokens;
        costs.total_gpu_seconds += record.gpu_seconds;
        costs.request_count++;
    }
    return result;
}

double llama_cost_tracker::calculate_cost(const std::string & model_id,
                                           int prompt_tokens,
                                           int completion_tokens,
                                           double gpu_seconds) const {
    std::lock_guard<std::mutex> lock(mutex_);

    pricing p;
    auto it = model_pricing_.find(model_id);
    if (it != model_pricing_.end()) {
        p = it->second;
    }

    return prompt_tokens * p.prompt_token_cost +
           completion_tokens * p.completion_token_cost +
           gpu_seconds * p.gpu_second_cost;
}

//
// SLA Monitor Implementation
//

llama_sla_monitor::llama_sla_monitor() = default;

llama_sla_monitor::llama_sla_monitor(const sla_config & cfg) : cfg_(cfg) {}

void llama_sla_monitor::record_request(double latency_ms, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);

    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    latency_samples_.emplace_back(now, latency_ms);
    success_samples_.emplace_back(now, success);

    // Prune old samples
    while (!latency_samples_.empty() &&
           (now - latency_samples_.front().first) > WINDOW_MS) {
        latency_samples_.pop_front();
    }
    while (!success_samples_.empty() &&
           (now - success_samples_.front().first) > WINDOW_MS) {
        success_samples_.pop_front();
    }

    // Limit size
    while (latency_samples_.size() > MAX_SAMPLES) {
        latency_samples_.pop_front();
    }
    while (success_samples_.size() > MAX_SAMPLES) {
        success_samples_.pop_front();
    }

    check_sla_violations();
}

llama_sla_monitor::sla_status llama_sla_monitor::get_status() const {
    std::lock_guard<std::mutex> lock(mutex_);

    sla_status status;

    // Calculate availability
    if (!success_samples_.empty()) {
        int success_count = 0;
        for (const auto & [ts, success] : success_samples_) {
            if (success) success_count++;
        }
        status.current_availability = static_cast<double>(success_count) / success_samples_.size();
        status.current_error_rate = 1.0 - status.current_availability;
    }

    // Calculate latency percentiles
    if (!latency_samples_.empty()) {
        status.current_p50_latency_ms = calculate_percentile(0.50);
        status.current_p95_latency_ms = calculate_percentile(0.95);
        status.current_p99_latency_ms = calculate_percentile(0.99);
    }

    // Check SLA compliance
    status.availability_met = status.current_availability >= cfg_.target_availability;
    status.latency_met = status.current_p99_latency_ms <= cfg_.target_p99_latency_ms;
    status.error_rate_met = status.current_error_rate <= cfg_.target_error_rate;
    status.overall_sla_met = status.availability_met && status.latency_met && status.error_rate_met;

    return status;
}

double llama_sla_monitor::get_availability(int64_t window_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (success_samples_.empty()) return 1.0;

    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();

    int total = 0, success = 0;
    for (const auto & [ts, ok] : success_samples_) {
        if ((now - ts) <= window_ms) {
            total++;
            if (ok) success++;
        }
    }

    return total > 0 ? static_cast<double>(success) / total : 1.0;
}

double llama_sla_monitor::get_error_rate(int64_t window_ms) const {
    return 1.0 - get_availability(window_ms);
}

void llama_sla_monitor::set_alert_callback(std::function<void(const std::string &)> callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    alert_callback_ = std::move(callback);
}

void llama_sla_monitor::check_sla_violations() {
    if (!alert_callback_) return;

    auto status = get_status();
    if (!status.overall_sla_met) {
        std::string alert;
        if (!status.availability_met) {
            alert += "Availability SLA violation. ";
        }
        if (!status.latency_met) {
            alert += "Latency SLA violation. ";
        }
        if (!status.error_rate_met) {
            alert += "Error rate SLA violation. ";
        }
        alert_callback_(alert);
    }
}

double llama_sla_monitor::calculate_percentile(double percentile) const {
    if (latency_samples_.empty()) return 0.0;

    std::vector<double> latencies;
    for (const auto & [ts, lat] : latency_samples_) {
        latencies.push_back(lat);
    }

    std::sort(latencies.begin(), latencies.end());

    size_t idx = static_cast<size_t>(percentile * (latencies.size() - 1));
    return latencies[idx];
}
