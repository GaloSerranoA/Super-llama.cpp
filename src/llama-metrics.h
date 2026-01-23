#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

// Structured metrics snapshot for JSON logging
struct llama_metrics_snapshot {
    // Timestamp
    int64_t timestamp_ms = 0;          // Unix timestamp in milliseconds

    // Token tracking
    int64_t token_count = 0;           // Total tokens processed
    int64_t prompt_tokens = 0;         // Prompt tokens
    int64_t generated_tokens = 0;      // Generated tokens

    // Layer metrics
    int32_t gpu_layers_active = 0;     // Layers currently on GPU
    int32_t cpu_layers_active = 0;     // Layers currently on CPU
    int64_t layers_evicted = 0;        // Total layers evicted to CPU
    int64_t layers_loaded = 0;         // Total layers loaded to GPU

    // KV cache page metrics
    uint32_t kv_pages_gpu = 0;         // KV pages on GPU
    uint32_t kv_pages_cpu = 0;         // KV pages on CPU
    int64_t kv_pages_evicted = 0;      // Total KV pages evicted
    int64_t kv_pages_prefetched = 0;   // Total KV pages prefetched

    // Memory usage (bytes)
    size_t vram_used_bytes = 0;        // GPU memory used
    size_t vram_total_bytes = 0;       // GPU memory total
    size_t ram_used_bytes = 0;         // CPU memory used by model
    size_t ram_total_bytes = 0;        // System RAM total

    // Performance metrics
    double tokens_per_sec = 0.0;       // Current throughput
    double avg_tokens_per_sec = 0.0;   // Average throughput
    int64_t prefetch_pending = 0;      // Pending prefetch requests
    int64_t prefetch_completed = 0;    // Completed prefetch requests
    int64_t prefetch_failed = 0;       // Failed prefetch requests

    // Memory pressure
    float gpu_memory_pressure = 0.0f;  // GPU memory utilization (0.0-1.0)
    float cpu_memory_pressure = 0.0f;  // CPU memory utilization (0.0-1.0)

    // Migration metrics
    double avg_migration_time_ms = 0.0;  // Average migration time
    size_t total_bytes_migrated = 0;     // Total bytes migrated
    double migration_bandwidth_mbps = 0.0; // Migration bandwidth MB/s

    // Memory watermarks (high-water marks)
    size_t gpu_memory_watermark = 0;     // Peak GPU memory usage
    size_t cpu_memory_watermark = 0;     // Peak CPU memory usage
};

// Metrics logger for structured JSON output
class llama_metrics_logger {
public:
    struct config {
        bool enabled = false;              // Enable metrics logging
        std::string output_file;           // Output file path (empty = stderr)
        int64_t log_interval_ms = 1000;    // Logging interval in milliseconds
        bool log_on_token = false;         // Log on every token
        bool log_on_eviction = true;       // Log on layer/page eviction
        bool pretty_print = false;         // Pretty print JSON
    };

    llama_metrics_logger();
    explicit llama_metrics_logger(const config & cfg);
    ~llama_metrics_logger();

    // Update metrics (thread-safe)
    void set_token_count(int64_t prompt, int64_t generated);
    void set_layer_counts(int32_t gpu, int32_t cpu);
    void set_kv_page_counts(uint32_t gpu, uint32_t cpu);
    void set_memory_usage(size_t vram_used, size_t vram_total, size_t ram_used, size_t ram_total);
    void set_memory_pressure(float gpu_pressure, float cpu_pressure);
    void set_prefetch_stats(int64_t pending, int64_t completed, int64_t failed);
    void set_migration_stats(double avg_time_ms, size_t total_bytes, double bandwidth_mbps);
    void set_memory_watermarks(size_t gpu_watermark, size_t cpu_watermark);

    // Increment counters (thread-safe)
    void inc_layers_evicted(int64_t count = 1);
    void inc_layers_loaded(int64_t count = 1);
    void inc_kv_pages_evicted(int64_t count = 1);
    void inc_kv_pages_prefetched(int64_t count = 1);
    void inc_tokens(int64_t count = 1);

    // Record a token generation for throughput calculation
    void record_token_generated();

    // Force log current metrics
    void log_now();

    // Check if should log based on interval
    void maybe_log();

    // Get current snapshot
    llama_metrics_snapshot get_snapshot() const;

    // Configuration
    void set_config(const config & cfg);
    const config & get_config() const;

    // Enable/disable
    void set_enabled(bool enabled);
    bool is_enabled() const;

private:
    // Format snapshot as JSON
    std::string format_json(const llama_metrics_snapshot & snapshot) const;

    // Get current timestamp in milliseconds
    static int64_t get_timestamp_ms();

    // Write to output
    void write_log(const std::string & json);

    config cfg_;
    mutable std::mutex mutex_;

    // Current metrics state
    llama_metrics_snapshot current_;

    // Timing for throughput calculation
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_token_time_;
    std::chrono::steady_clock::time_point last_log_time_;

    // Token timing for throughput
    int64_t tokens_since_last_calc_ = 0;
    double last_throughput_ = 0.0;

    // Output file handle
    std::ofstream output_file_;
};

// Global metrics instance (optional, for convenience)
llama_metrics_logger * llama_get_metrics_logger();
void llama_set_metrics_logger(llama_metrics_logger * logger);
