#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Metric types following Prometheus conventions
enum class llama_metric_type {
    COUNTER,     // Monotonically increasing value
    GAUGE,       // Value that can go up or down
    HISTOGRAM,   // Distribution of values
    SUMMARY,     // Similar to histogram with quantiles
};

// Single metric value with labels
struct llama_metric_sample {
    std::string name;
    llama_metric_type type;
    std::string help;
    std::map<std::string, std::string> labels;
    double value = 0.0;
    int64_t timestamp_ms = 0;
};

// Histogram bucket
struct llama_histogram_bucket {
    double le;        // Less than or equal threshold
    uint64_t count;   // Number of observations <= le
};

// Histogram metric
struct llama_histogram {
    std::string name;
    std::string help;
    std::map<std::string, std::string> labels;
    std::vector<llama_histogram_bucket> buckets;
    double sum = 0.0;
    uint64_t count = 0;
};

// Summary metric with quantiles
struct llama_summary {
    std::string name;
    std::string help;
    std::map<std::string, std::string> labels;
    std::map<double, double> quantiles;  // quantile -> value
    double sum = 0.0;
    uint64_t count = 0;
};

// Prometheus metrics registry and exporter
class llama_prometheus_exporter {
public:
    struct config {
        std::string endpoint = "/metrics";   // HTTP endpoint path
        int port = 9090;                     // HTTP server port
        std::string job_name = "llama_cpp";  // Prometheus job name
        std::string instance_name;           // Instance identifier
        bool enable_process_metrics = true;  // Export process metrics
        bool enable_go_style = false;        // Use Go-style metric names
        int scrape_interval_ms = 1000;       // Metrics update interval
    };

    llama_prometheus_exporter();
    explicit llama_prometheus_exporter(const config & cfg);
    ~llama_prometheus_exporter();

    // Start/stop the metrics server
    bool start();
    void stop();
    bool is_running() const { return running_; }

    // Counter operations (monotonically increasing)
    void counter_inc(const std::string & name, double value = 1.0,
                     const std::map<std::string, std::string> & labels = {});
    void counter_set(const std::string & name, double value,
                     const std::map<std::string, std::string> & labels = {});

    // Gauge operations (can increase or decrease)
    void gauge_set(const std::string & name, double value,
                   const std::map<std::string, std::string> & labels = {});
    void gauge_inc(const std::string & name, double value = 1.0,
                   const std::map<std::string, std::string> & labels = {});
    void gauge_dec(const std::string & name, double value = 1.0,
                   const std::map<std::string, std::string> & labels = {});

    // Histogram operations
    void histogram_observe(const std::string & name, double value,
                           const std::vector<double> & buckets = {},
                           const std::map<std::string, std::string> & labels = {});

    // Summary operations
    void summary_observe(const std::string & name, double value,
                         const std::map<std::string, std::string> & labels = {});

    // Register metric with help text
    void register_metric(const std::string & name,
                         llama_metric_type type,
                         const std::string & help);

    // Get metrics in Prometheus text format
    std::string get_metrics_text() const;

    // Get metrics in OpenMetrics format
    std::string get_openmetrics_text() const;

    // Get single metric value
    double get_metric_value(const std::string & name,
                            const std::map<std::string, std::string> & labels = {}) const;

    // Configuration
    const config & get_config() const { return cfg_; }

    // Pre-defined llama.cpp metrics
    void update_inference_metrics(
        int64_t tokens_generated,
        double tokens_per_sec,
        int64_t prompt_tokens,
        double prompt_eval_time_ms);

    void update_memory_metrics(
        size_t vram_used,
        size_t vram_total,
        size_t ram_used,
        size_t ram_total);

    void update_layer_metrics(
        int gpu_layers,
        int cpu_layers,
        int64_t layers_evicted,
        int64_t layers_loaded);

    void update_kv_cache_metrics(
        uint32_t kv_pages_gpu,
        uint32_t kv_pages_cpu,
        int64_t pages_evicted,
        int64_t pages_prefetched);

    void update_request_metrics(
        int64_t requests_total,
        int64_t requests_active,
        double avg_latency_ms,
        double p99_latency_ms);

private:
    // Internal metric storage
    struct metric_entry {
        llama_metric_type type;
        std::string help;
        std::map<std::string, double> values;  // label_key -> value
    };

    std::string make_label_key(const std::map<std::string, std::string> & labels) const;
    std::string format_labels(const std::map<std::string, std::string> & labels) const;
    void add_process_metrics(std::ostringstream & oss) const;
    int64_t get_timestamp_ms() const;

    config cfg_;
    mutable std::mutex mutex_;

    std::map<std::string, metric_entry> metrics_;
    std::map<std::string, llama_histogram> histograms_;
    std::map<std::string, llama_summary> summaries_;

    // HTTP server thread (simplified - real impl would use a proper HTTP library)
    std::thread server_thread_;
    std::atomic<bool> running_{false};

    // Default histogram buckets for latency (in milliseconds)
    static const std::vector<double> DEFAULT_LATENCY_BUCKETS;
    static const std::vector<double> DEFAULT_SIZE_BUCKETS;
};

// OpenTelemetry-compatible span for distributed tracing
class llama_trace_span {
public:
    llama_trace_span(const std::string & name, const std::string & trace_id = "");
    ~llama_trace_span();

    void set_attribute(const std::string & key, const std::string & value);
    void set_attribute(const std::string & key, int64_t value);
    void set_attribute(const std::string & key, double value);

    void add_event(const std::string & name,
                   const std::map<std::string, std::string> & attributes = {});

    void set_status(bool ok, const std::string & message = "");

    void end();

    std::string get_trace_id() const { return trace_id_; }
    std::string get_span_id() const { return span_id_; }
    int64_t get_duration_us() const;

private:
    std::string name_;
    std::string trace_id_;
    std::string span_id_;
    std::string parent_span_id_;

    std::map<std::string, std::string> attributes_;
    std::vector<std::pair<int64_t, std::string>> events_;

    int64_t start_time_us_;
    int64_t end_time_us_ = 0;
    bool ended_ = false;
    bool ok_ = true;
    std::string status_message_;
};

// Global Prometheus exporter instance
llama_prometheus_exporter * llama_get_prometheus_exporter();
void llama_set_prometheus_exporter(llama_prometheus_exporter * exporter);
