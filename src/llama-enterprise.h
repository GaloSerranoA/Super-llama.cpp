#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declarations
struct llama_context;

//
// Request Queue with Priority Scheduling
//

enum class llama_request_priority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2,
    CRITICAL = 3,
};

struct llama_request {
    std::string request_id;
    std::string client_id;
    std::string model_id;
    llama_request_priority priority = llama_request_priority::NORMAL;

    // Timing
    int64_t enqueue_time_us = 0;
    int64_t start_time_us = 0;
    int64_t end_time_us = 0;

    // Request data
    std::string prompt;
    int max_tokens = 512;
    float temperature = 0.7f;

    // Callback for completion
    std::function<void(const std::string & result, bool success)> on_complete;

    // Cost tracking
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double cost_credits = 0.0;
};

class llama_request_queue {
public:
    struct config {
        size_t max_queue_size = 1000;
        size_t max_concurrent = 4;
        int64_t request_timeout_ms = 300000;  // 5 minutes
        bool enable_priority = true;
        bool enable_fair_scheduling = true;
    };

    llama_request_queue();
    explicit llama_request_queue(const config & cfg);
    ~llama_request_queue();

    // Enqueue a request
    bool enqueue(llama_request request);

    // Dequeue next request (blocking)
    std::optional<llama_request> dequeue(int64_t timeout_ms = -1);

    // Mark request as completed
    void complete(const std::string & request_id, bool success);

    // Cancel a request
    bool cancel(const std::string & request_id);

    // Get queue stats
    struct queue_stats {
        size_t pending_requests = 0;
        size_t active_requests = 0;
        size_t completed_requests = 0;
        size_t failed_requests = 0;
        size_t cancelled_requests = 0;
        double avg_wait_time_ms = 0.0;
        double avg_processing_time_ms = 0.0;
    };
    queue_stats get_stats() const;

    // Get requests by client
    std::vector<llama_request> get_client_requests(const std::string & client_id) const;

    const config & get_config() const { return cfg_; }

private:
    config cfg_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Priority queues (one per priority level)
    std::array<std::deque<llama_request>, 4> priority_queues_;

    // Active requests
    std::map<std::string, llama_request> active_requests_;

    // Per-client tracking for fair scheduling
    std::map<std::string, int64_t> client_last_served_;

    // Statistics
    std::atomic<size_t> total_completed_{0};
    std::atomic<size_t> total_failed_{0};
    std::atomic<size_t> total_cancelled_{0};
    std::atomic<int64_t> total_wait_time_us_{0};
    std::atomic<int64_t> total_processing_time_us_{0};

    int64_t get_time_us() const;
};

//
// Rate Limiter
//

class llama_rate_limiter {
public:
    struct config {
        int requests_per_minute = 60;
        int tokens_per_minute = 100000;
        int max_concurrent_requests = 10;
        bool enable_burst = true;
        int burst_size = 10;
    };

    struct client_limits {
        int requests_per_minute = 60;
        int tokens_per_minute = 100000;
        int max_concurrent = 10;
    };

    llama_rate_limiter();
    explicit llama_rate_limiter(const config & cfg);

    // Check if request is allowed
    bool allow_request(const std::string & client_id);

    // Check if tokens are allowed
    bool allow_tokens(const std::string & client_id, int token_count);

    // Record token usage
    void record_tokens(const std::string & client_id, int token_count);

    // Set client-specific limits
    void set_client_limits(const std::string & client_id, const client_limits & limits);

    // Get client usage
    struct client_usage {
        int requests_this_minute = 0;
        int tokens_this_minute = 0;
        int active_requests = 0;
        int64_t window_reset_time_ms = 0;
    };
    client_usage get_client_usage(const std::string & client_id) const;

    // Check remaining quota
    int get_remaining_requests(const std::string & client_id) const;
    int get_remaining_tokens(const std::string & client_id) const;

private:
    struct client_state {
        int requests_this_minute = 0;
        int tokens_this_minute = 0;
        int active_requests = 0;
        int64_t window_start_ms = 0;
        client_limits limits;
    };

    void reset_window_if_needed(client_state & state, int64_t now_ms) const;
    int64_t get_time_ms() const;

    config cfg_;
    mutable std::mutex mutex_;
    std::map<std::string, client_state> client_states_;
};

//
// Health Monitor
//

enum class llama_health_status {
    HEALTHY,
    DEGRADED,
    UNHEALTHY,
};

class llama_health_monitor {
public:
    struct config {
        int check_interval_ms = 5000;
        double max_memory_usage = 0.95;
        int max_queue_depth = 500;
        int max_latency_ms = 30000;
        bool auto_recovery = true;
    };

    struct health_check {
        std::string name;
        llama_health_status status = llama_health_status::HEALTHY;
        std::string message;
        int64_t last_check_ms = 0;
    };

    llama_health_monitor();
    explicit llama_health_monitor(const config & cfg);
    ~llama_health_monitor();

    // Start/stop monitoring
    void start();
    void stop();

    // Get overall health
    llama_health_status get_status() const;

    // Get detailed health
    struct health_report {
        llama_health_status overall_status = llama_health_status::HEALTHY;
        std::vector<health_check> checks;
        int64_t uptime_seconds = 0;
        std::string version;
    };
    health_report get_report() const;

    // Register custom health check
    void register_check(const std::string & name,
                        std::function<health_check()> check_fn);

    // Manual health update
    void set_check_status(const std::string & name, llama_health_status status,
                          const std::string & message = "");

private:
    void monitor_thread();

    config cfg_;
    mutable std::mutex mutex_;

    std::map<std::string, std::function<health_check()>> custom_checks_;
    std::map<std::string, health_check> check_results_;

    std::thread monitor_thread_;
    std::atomic<bool> running_{false};
    int64_t start_time_ms_ = 0;
};

//
// Audit Logger
//

enum class llama_audit_event_type {
    REQUEST_RECEIVED,
    REQUEST_STARTED,
    REQUEST_COMPLETED,
    REQUEST_FAILED,
    REQUEST_CANCELLED,
    MODEL_LOADED,
    MODEL_UNLOADED,
    CONFIG_CHANGED,
    RATE_LIMITED,
    AUTH_SUCCESS,
    AUTH_FAILURE,
    ERROR,
};

struct llama_audit_event {
    int64_t timestamp_ms = 0;
    llama_audit_event_type type;
    std::string event_id;
    std::string client_id;
    std::string request_id;
    std::string model_id;
    std::string message;
    std::map<std::string, std::string> metadata;
    std::string ip_address;
    std::string user_agent;
};

class llama_audit_logger {
public:
    struct config {
        std::string log_file;
        bool log_to_stdout = false;
        bool log_prompts = false;       // Privacy: don't log by default
        bool log_completions = false;   // Privacy: don't log by default
        int max_file_size_mb = 100;
        int max_files = 10;
        bool async_logging = true;
    };

    llama_audit_logger();
    explicit llama_audit_logger(const config & cfg);
    ~llama_audit_logger();

    // Log an event
    void log(const llama_audit_event & event);

    // Convenience methods
    void log_request_received(const std::string & client_id, const std::string & request_id);
    void log_request_completed(const std::string & client_id, const std::string & request_id,
                               int tokens, double latency_ms);
    void log_error(const std::string & message, const std::string & request_id = "");
    void log_rate_limited(const std::string & client_id);
    void log_auth(const std::string & client_id, bool success);

    // Query logs (if stored in memory)
    std::vector<llama_audit_event> query(
        int64_t start_time_ms,
        int64_t end_time_ms,
        const std::string & client_id = "",
        llama_audit_event_type type = llama_audit_event_type::REQUEST_RECEIVED);

private:
    void writer_thread();
    std::string format_event(const llama_audit_event & event) const;
    std::string generate_event_id() const;

    config cfg_;
    mutable std::mutex mutex_;

    std::deque<llama_audit_event> write_queue_;
    std::condition_variable cv_;
    std::thread writer_thread_;
    std::atomic<bool> running_{true};

    // In-memory buffer for queries (limited size)
    std::deque<llama_audit_event> event_buffer_;
    static constexpr size_t MAX_BUFFER_SIZE = 10000;
};

//
// Role-Based Access Control
//

enum class llama_permission {
    INFERENCE,          // Can run inference
    MODEL_LOAD,         // Can load/unload models
    CONFIG_READ,        // Can read configuration
    CONFIG_WRITE,       // Can modify configuration
    METRICS_READ,       // Can read metrics
    AUDIT_READ,         // Can read audit logs
    ADMIN,              // Full access
};

struct llama_role {
    std::string name;
    std::vector<llama_permission> permissions;
    int rate_limit_requests = -1;   // -1 = use default
    int rate_limit_tokens = -1;
    std::vector<std::string> allowed_models;  // Empty = all models
};

class llama_rbac {
public:
    llama_rbac();

    // Role management
    void add_role(const llama_role & role);
    void remove_role(const std::string & role_name);
    std::optional<llama_role> get_role(const std::string & role_name) const;
    std::vector<llama_role> get_all_roles() const;

    // Client-role assignment
    void assign_role(const std::string & client_id, const std::string & role_name);
    void revoke_role(const std::string & client_id, const std::string & role_name);
    std::vector<std::string> get_client_roles(const std::string & client_id) const;

    // Permission checking
    bool has_permission(const std::string & client_id, llama_permission permission) const;
    bool can_access_model(const std::string & client_id, const std::string & model_id) const;

    // Get effective rate limits
    int get_rate_limit_requests(const std::string & client_id) const;
    int get_rate_limit_tokens(const std::string & client_id) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, llama_role> roles_;
    std::map<std::string, std::vector<std::string>> client_roles_;

    // Default roles
    void init_default_roles();
};

//
// Content Filter
//

class llama_content_filter {
public:
    struct config {
        bool filter_input = true;
        bool filter_output = true;
        bool block_on_violation = true;
        std::vector<std::string> blocked_patterns;
        std::vector<std::string> blocked_words;
        double toxicity_threshold = 0.8;
    };

    struct filter_result {
        bool passed = true;
        std::string reason;
        std::vector<std::string> violations;
        double toxicity_score = 0.0;
    };

    llama_content_filter();
    explicit llama_content_filter(const config & cfg);

    // Filter input
    filter_result filter_input(const std::string & text);

    // Filter output
    filter_result filter_output(const std::string & text);

    // Add blocked pattern (regex)
    void add_blocked_pattern(const std::string & pattern);

    // Add blocked word
    void add_blocked_word(const std::string & word);

    // Clear filters
    void clear_filters();

private:
    bool contains_blocked_word(const std::string & text) const;
    bool matches_blocked_pattern(const std::string & text) const;

    config cfg_;
    mutable std::mutex mutex_;
};

//
// Cost Attribution
//

struct llama_cost_record {
    std::string client_id;
    std::string request_id;
    std::string model_id;
    int64_t timestamp_ms = 0;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    double gpu_seconds = 0.0;
    double cost_credits = 0.0;
};

class llama_cost_tracker {
public:
    struct pricing {
        double prompt_token_cost = 0.0001;       // Cost per prompt token
        double completion_token_cost = 0.0002;   // Cost per completion token
        double gpu_second_cost = 0.001;          // Cost per GPU-second
    };

    llama_cost_tracker();

    // Set pricing for a model
    void set_pricing(const std::string & model_id, const pricing & p);

    // Record usage
    void record_usage(const llama_cost_record & record);

    // Get client costs
    struct client_costs {
        double total_cost = 0.0;
        int total_prompt_tokens = 0;
        int total_completion_tokens = 0;
        double total_gpu_seconds = 0.0;
        int request_count = 0;
    };
    client_costs get_client_costs(const std::string & client_id,
                                   int64_t start_time_ms = 0,
                                   int64_t end_time_ms = 0) const;

    // Get all costs for billing period
    std::map<std::string, client_costs> get_all_costs(
        int64_t start_time_ms,
        int64_t end_time_ms) const;

    // Calculate cost for a request
    double calculate_cost(const std::string & model_id,
                          int prompt_tokens,
                          int completion_tokens,
                          double gpu_seconds) const;

private:
    mutable std::mutex mutex_;
    std::map<std::string, pricing> model_pricing_;
    std::deque<llama_cost_record> records_;
    static constexpr size_t MAX_RECORDS = 100000;
};

//
// SLA Monitor
//

class llama_sla_monitor {
public:
    struct sla_config {
        double target_availability = 0.999;      // 99.9%
        double target_p50_latency_ms = 100.0;
        double target_p95_latency_ms = 500.0;
        double target_p99_latency_ms = 2000.0;
        double target_error_rate = 0.001;        // 0.1%
    };

    struct sla_status {
        double current_availability = 1.0;
        double current_p50_latency_ms = 0.0;
        double current_p95_latency_ms = 0.0;
        double current_p99_latency_ms = 0.0;
        double current_error_rate = 0.0;
        bool availability_met = true;
        bool latency_met = true;
        bool error_rate_met = true;
        bool overall_sla_met = true;
    };

    llama_sla_monitor();
    explicit llama_sla_monitor(const sla_config & cfg);

    // Record request completion
    void record_request(double latency_ms, bool success);

    // Get current SLA status
    sla_status get_status() const;

    // Get historical SLA compliance
    double get_availability(int64_t window_ms = 3600000) const;  // Default 1 hour
    double get_error_rate(int64_t window_ms = 3600000) const;

    // Alerts
    void set_alert_callback(std::function<void(const std::string &)> callback);

private:
    void check_sla_violations();
    void add_latency_sample(double latency_ms);
    double calculate_percentile(double percentile) const;

    sla_config cfg_;
    mutable std::mutex mutex_;

    // Sliding window of latencies
    std::deque<std::pair<int64_t, double>> latency_samples_;
    std::deque<std::pair<int64_t, bool>> success_samples_;

    static constexpr size_t MAX_SAMPLES = 10000;
    static constexpr int64_t WINDOW_MS = 3600000;  // 1 hour

    std::function<void(const std::string &)> alert_callback_;
};

// Global enterprise instances
llama_request_queue * llama_get_request_queue();
void llama_set_request_queue(llama_request_queue * queue);

llama_rate_limiter * llama_get_rate_limiter();
void llama_set_rate_limiter(llama_rate_limiter * limiter);

llama_health_monitor * llama_get_health_monitor();
void llama_set_health_monitor(llama_health_monitor * monitor);

llama_audit_logger * llama_get_audit_logger();
void llama_set_audit_logger(llama_audit_logger * logger);
