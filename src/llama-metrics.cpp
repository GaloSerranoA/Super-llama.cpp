#include "llama-metrics.h"
#include "llama-impl.h"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

// Global metrics logger instance
static llama_metrics_logger * g_metrics_logger = nullptr;

llama_metrics_logger * llama_get_metrics_logger() {
    return g_metrics_logger;
}

void llama_set_metrics_logger(llama_metrics_logger * logger) {
    g_metrics_logger = logger;
}

llama_metrics_logger::llama_metrics_logger()
    : cfg_{}
    , start_time_(std::chrono::steady_clock::now())
    , last_token_time_(start_time_)
    , last_log_time_(start_time_) {
}

llama_metrics_logger::llama_metrics_logger(const config & cfg)
    : cfg_(cfg)
    , start_time_(std::chrono::steady_clock::now())
    , last_token_time_(start_time_)
    , last_log_time_(start_time_) {

    if (cfg_.enabled && !cfg_.output_file.empty()) {
        output_file_.open(cfg_.output_file, std::ios::out | std::ios::app);
        if (!output_file_.is_open()) {
            LLAMA_LOG_WARN("%s: could not open metrics file '%s', using stderr\n",
                __func__, cfg_.output_file.c_str());
        }
    }
}

llama_metrics_logger::~llama_metrics_logger() {
    if (output_file_.is_open()) {
        output_file_.close();
    }
}

void llama_metrics_logger::set_token_count(int64_t prompt, int64_t generated) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.prompt_tokens = prompt;
    current_.generated_tokens = generated;
    current_.token_count = prompt + generated;
}

void llama_metrics_logger::set_layer_counts(int32_t gpu, int32_t cpu) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.gpu_layers_active = gpu;
    current_.cpu_layers_active = cpu;
}

void llama_metrics_logger::set_kv_page_counts(uint32_t gpu, uint32_t cpu) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.kv_pages_gpu = gpu;
    current_.kv_pages_cpu = cpu;
}

void llama_metrics_logger::set_memory_usage(size_t vram_used, size_t vram_total,
                                            size_t ram_used, size_t ram_total) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.vram_used_bytes = vram_used;
    current_.vram_total_bytes = vram_total;
    current_.ram_used_bytes = ram_used;
    current_.ram_total_bytes = ram_total;
}

void llama_metrics_logger::set_memory_pressure(float gpu_pressure, float cpu_pressure) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.gpu_memory_pressure = gpu_pressure;
    current_.cpu_memory_pressure = cpu_pressure;
}

void llama_metrics_logger::set_prefetch_stats(int64_t pending, int64_t completed, int64_t failed) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.prefetch_pending = pending;
    current_.prefetch_completed = completed;
    current_.prefetch_failed = failed;
}

void llama_metrics_logger::inc_layers_evicted(int64_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.layers_evicted += count;

    if (cfg_.log_on_eviction) {
        // Log immediately on eviction
        llama_metrics_snapshot snapshot = current_;
        snapshot.timestamp_ms = get_timestamp_ms();

        // Calculate throughput
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed > 0 && snapshot.generated_tokens > 0) {
            snapshot.avg_tokens_per_sec = static_cast<double>(snapshot.generated_tokens) / elapsed;
        }
        snapshot.tokens_per_sec = last_throughput_;

        std::string json = format_json(snapshot);
        write_log(json);
    }
}

void llama_metrics_logger::inc_layers_loaded(int64_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.layers_loaded += count;
}

void llama_metrics_logger::inc_kv_pages_evicted(int64_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.kv_pages_evicted += count;

    if (cfg_.log_on_eviction) {
        llama_metrics_snapshot snapshot = current_;
        snapshot.timestamp_ms = get_timestamp_ms();

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (elapsed > 0 && snapshot.generated_tokens > 0) {
            snapshot.avg_tokens_per_sec = static_cast<double>(snapshot.generated_tokens) / elapsed;
        }
        snapshot.tokens_per_sec = last_throughput_;

        std::string json = format_json(snapshot);
        write_log(json);
    }
}

void llama_metrics_logger::inc_kv_pages_prefetched(int64_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.kv_pages_prefetched += count;
}

void llama_metrics_logger::inc_tokens(int64_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_.token_count += count;
    current_.generated_tokens += count;
    tokens_since_last_calc_ += count;
}

void llama_metrics_logger::record_token_generated() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto now = std::chrono::steady_clock::now();
    current_.generated_tokens++;
    current_.token_count++;
    tokens_since_last_calc_++;

    // Calculate instantaneous throughput (over last second)
    auto elapsed_since_last = std::chrono::duration<double>(now - last_token_time_).count();
    if (elapsed_since_last >= 1.0 && tokens_since_last_calc_ > 0) {
        last_throughput_ = static_cast<double>(tokens_since_last_calc_) / elapsed_since_last;
        tokens_since_last_calc_ = 0;
        last_token_time_ = now;
    }

    if (cfg_.log_on_token) {
        llama_metrics_snapshot snapshot = current_;
        snapshot.timestamp_ms = get_timestamp_ms();
        snapshot.tokens_per_sec = last_throughput_;

        auto total_elapsed = std::chrono::duration<double>(now - start_time_).count();
        if (total_elapsed > 0) {
            snapshot.avg_tokens_per_sec = static_cast<double>(snapshot.generated_tokens) / total_elapsed;
        }

        std::string json = format_json(snapshot);
        write_log(json);
    }
}

void llama_metrics_logger::log_now() {
    if (!cfg_.enabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    llama_metrics_snapshot snapshot = current_;
    snapshot.timestamp_ms = get_timestamp_ms();
    snapshot.tokens_per_sec = last_throughput_;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed > 0 && snapshot.generated_tokens > 0) {
        snapshot.avg_tokens_per_sec = static_cast<double>(snapshot.generated_tokens) / elapsed;
    }

    std::string json = format_json(snapshot);
    write_log(json);

    last_log_time_ = now;
}

void llama_metrics_logger::maybe_log() {
    if (!cfg_.enabled) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_log_time_).count();

    if (elapsed_ms >= cfg_.log_interval_ms) {
        log_now();
    }
}

llama_metrics_snapshot llama_metrics_logger::get_snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);

    llama_metrics_snapshot snapshot = current_;
    snapshot.timestamp_ms = get_timestamp_ms();
    snapshot.tokens_per_sec = last_throughput_;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed > 0 && snapshot.generated_tokens > 0) {
        snapshot.avg_tokens_per_sec = static_cast<double>(snapshot.generated_tokens) / elapsed;
    }

    return snapshot;
}

void llama_metrics_logger::set_config(const config & cfg) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Close old file if open
    if (output_file_.is_open()) {
        output_file_.close();
    }

    cfg_ = cfg;

    // Open new file if specified
    if (cfg_.enabled && !cfg_.output_file.empty()) {
        output_file_.open(cfg_.output_file, std::ios::out | std::ios::app);
        if (!output_file_.is_open()) {
            LLAMA_LOG_WARN("%s: could not open metrics file '%s', using stderr\n",
                __func__, cfg_.output_file.c_str());
        }
    }
}

const llama_metrics_logger::config & llama_metrics_logger::get_config() const {
    return cfg_;
}

void llama_metrics_logger::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_.enabled = enabled;
}

bool llama_metrics_logger::is_enabled() const {
    return cfg_.enabled;
}

std::string llama_metrics_logger::format_json(const llama_metrics_snapshot & s) const {
    std::ostringstream oss;

    // Convert timestamp to ISO 8601 format
    std::time_t time_sec = static_cast<std::time_t>(s.timestamp_ms / 1000);
    int time_ms = static_cast<int>(s.timestamp_ms % 1000);
    std::tm * tm_info = std::gmtime(&time_sec);

    char timestamp_buf[32];
    std::strftime(timestamp_buf, sizeof(timestamp_buf), "%Y-%m-%dT%H:%M:%S", tm_info);

    // Convert bytes to MB for readability
    double vram_used_mb = static_cast<double>(s.vram_used_bytes) / (1024.0 * 1024.0);
    double vram_total_mb = static_cast<double>(s.vram_total_bytes) / (1024.0 * 1024.0);
    double ram_used_mb = static_cast<double>(s.ram_used_bytes) / (1024.0 * 1024.0);

    if (cfg_.pretty_print) {
        oss << "{\n";
        oss << "  \"timestamp\": \"" << timestamp_buf << "." << std::setfill('0') << std::setw(3) << time_ms << "Z\",\n";
        oss << "  \"token\": " << s.token_count << ",\n";
        oss << "  \"prompt_tokens\": " << s.prompt_tokens << ",\n";
        oss << "  \"generated_tokens\": " << s.generated_tokens << ",\n";
        oss << "  \"gpu_layers_active\": " << s.gpu_layers_active << ",\n";
        oss << "  \"cpu_layers_active\": " << s.cpu_layers_active << ",\n";
        oss << "  \"layers_evicted\": " << s.layers_evicted << ",\n";
        oss << "  \"layers_loaded\": " << s.layers_loaded << ",\n";
        oss << "  \"kv_pages_gpu\": " << s.kv_pages_gpu << ",\n";
        oss << "  \"kv_pages_cpu\": " << s.kv_pages_cpu << ",\n";
        oss << "  \"kv_pages_evicted\": " << s.kv_pages_evicted << ",\n";
        oss << "  \"kv_pages_prefetched\": " << s.kv_pages_prefetched << ",\n";
        oss << "  \"vram_used_mb\": " << std::fixed << std::setprecision(1) << vram_used_mb << ",\n";
        oss << "  \"vram_total_mb\": " << std::fixed << std::setprecision(1) << vram_total_mb << ",\n";
        oss << "  \"ram_used_mb\": " << std::fixed << std::setprecision(1) << ram_used_mb << ",\n";
        oss << "  \"gpu_memory_pressure\": " << std::fixed << std::setprecision(3) << s.gpu_memory_pressure << ",\n";
        oss << "  \"cpu_memory_pressure\": " << std::fixed << std::setprecision(3) << s.cpu_memory_pressure << ",\n";
        oss << "  \"tokens_per_sec\": " << std::fixed << std::setprecision(2) << s.tokens_per_sec << ",\n";
        oss << "  \"avg_tokens_per_sec\": " << std::fixed << std::setprecision(2) << s.avg_tokens_per_sec << ",\n";
        oss << "  \"prefetch_pending\": " << s.prefetch_pending << ",\n";
        oss << "  \"prefetch_completed\": " << s.prefetch_completed << ",\n";
        oss << "  \"prefetch_failed\": " << s.prefetch_failed << "\n";
        oss << "}";
    } else {
        // Compact single-line JSON
        oss << "{";
        oss << "\"timestamp\":\"" << timestamp_buf << "." << std::setfill('0') << std::setw(3) << time_ms << "Z\",";
        oss << "\"token\":" << s.token_count << ",";
        oss << "\"gpu_layers_active\":" << s.gpu_layers_active << ",";
        oss << "\"layers_evicted\":" << s.layers_evicted << ",";
        oss << "\"kv_pages_gpu\":" << s.kv_pages_gpu << ",";
        oss << "\"kv_pages_cpu\":" << s.kv_pages_cpu << ",";
        oss << "\"vram_used_mb\":" << std::fixed << std::setprecision(1) << vram_used_mb << ",";
        oss << "\"tokens_per_sec\":" << std::fixed << std::setprecision(2) << s.tokens_per_sec;
        oss << "}";
    }

    return oss.str();
}

int64_t llama_metrics_logger::get_timestamp_ms() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

void llama_metrics_logger::write_log(const std::string & json) {
    if (output_file_.is_open()) {
        output_file_ << json << "\n";
        output_file_.flush();
    } else {
        // Write to stderr with prefix
        std::cerr << "[METRICS] " << json << "\n";
    }
}
