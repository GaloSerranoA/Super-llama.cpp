#include "llama-prometheus.h"
#include "llama-impl.h"

#include <algorithm>
#include <ctime>
#include <iomanip>
#include <random>
#include <sstream>

// Global instance with thread-safe access
static llama_prometheus_exporter * g_prometheus_exporter = nullptr;
static std::mutex g_prometheus_exporter_mutex;

llama_prometheus_exporter * llama_get_prometheus_exporter() {
    std::lock_guard<std::mutex> lock(g_prometheus_exporter_mutex);
    return g_prometheus_exporter;
}

void llama_set_prometheus_exporter(llama_prometheus_exporter * exporter) {
    std::lock_guard<std::mutex> lock(g_prometheus_exporter_mutex);
    g_prometheus_exporter = exporter;
}

// Default buckets
const std::vector<double> llama_prometheus_exporter::DEFAULT_LATENCY_BUCKETS = {
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0
};

const std::vector<double> llama_prometheus_exporter::DEFAULT_SIZE_BUCKETS = {
    1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864
};

llama_prometheus_exporter::llama_prometheus_exporter() = default;

llama_prometheus_exporter::llama_prometheus_exporter(const config & cfg)
    : cfg_(cfg) {
    // Register default llama.cpp metrics
    register_metric("llama_tokens_generated_total", llama_metric_type::COUNTER,
                    "Total number of tokens generated");
    register_metric("llama_tokens_per_second", llama_metric_type::GAUGE,
                    "Current token generation rate");
    register_metric("llama_prompt_tokens_total", llama_metric_type::COUNTER,
                    "Total number of prompt tokens processed");
    register_metric("llama_prompt_eval_seconds", llama_metric_type::COUNTER,
                    "Total time spent evaluating prompts");

    register_metric("llama_vram_bytes_used", llama_metric_type::GAUGE,
                    "VRAM currently in use");
    register_metric("llama_vram_bytes_total", llama_metric_type::GAUGE,
                    "Total VRAM available");
    register_metric("llama_ram_bytes_used", llama_metric_type::GAUGE,
                    "RAM currently in use");
    register_metric("llama_ram_bytes_total", llama_metric_type::GAUGE,
                    "Total RAM available");

    register_metric("llama_gpu_layers_active", llama_metric_type::GAUGE,
                    "Number of layers currently on GPU");
    register_metric("llama_cpu_layers_active", llama_metric_type::GAUGE,
                    "Number of layers currently on CPU");
    register_metric("llama_layers_evicted_total", llama_metric_type::COUNTER,
                    "Total number of layer evictions");
    register_metric("llama_layers_loaded_total", llama_metric_type::COUNTER,
                    "Total number of layer loads");

    register_metric("llama_kv_pages_gpu", llama_metric_type::GAUGE,
                    "Number of KV cache pages on GPU");
    register_metric("llama_kv_pages_cpu", llama_metric_type::GAUGE,
                    "Number of KV cache pages on CPU");
    register_metric("llama_kv_pages_evicted_total", llama_metric_type::COUNTER,
                    "Total number of KV page evictions");
    register_metric("llama_kv_pages_prefetched_total", llama_metric_type::COUNTER,
                    "Total number of KV page prefetches");

    register_metric("llama_requests_total", llama_metric_type::COUNTER,
                    "Total number of inference requests");
    register_metric("llama_requests_active", llama_metric_type::GAUGE,
                    "Number of active inference requests");
    register_metric("llama_request_latency_seconds", llama_metric_type::HISTOGRAM,
                    "Request latency distribution");
}

llama_prometheus_exporter::~llama_prometheus_exporter() {
    stop();
    if (g_prometheus_exporter == this) {
        g_prometheus_exporter = nullptr;
    }
}

bool llama_prometheus_exporter::start() {
    if (running_) return true;

    running_ = true;

    // Note: In a real implementation, this would start an HTTP server
    // For now, metrics are accessed via get_metrics_text()
    LLAMA_LOG_INFO("%s: Prometheus metrics available at http://localhost:%d%s\n",
        __func__, cfg_.port, cfg_.endpoint.c_str());

    return true;
}

void llama_prometheus_exporter::stop() {
    running_ = false;
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
}

void llama_prometheus_exporter::counter_inc(
        const std::string & name,
        double value,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_label_key(labels);
    metrics_[name].values[key] += value;
}

void llama_prometheus_exporter::counter_set(
        const std::string & name,
        double value,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_label_key(labels);
    metrics_[name].values[key] = value;
}

void llama_prometheus_exporter::gauge_set(
        const std::string & name,
        double value,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_label_key(labels);
    metrics_[name].values[key] = value;
}

void llama_prometheus_exporter::gauge_inc(
        const std::string & name,
        double value,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_label_key(labels);
    metrics_[name].values[key] += value;
}

void llama_prometheus_exporter::gauge_dec(
        const std::string & name,
        double value,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key = make_label_key(labels);
    metrics_[name].values[key] -= value;
}

void llama_prometheus_exporter::histogram_observe(
        const std::string & name,
        double value,
        const std::vector<double> & buckets,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_label_key(labels);
    auto & hist = histograms_[name + key];

    if (hist.buckets.empty()) {
        const auto & bucket_bounds = buckets.empty() ? DEFAULT_LATENCY_BUCKETS : buckets;
        for (double le : bucket_bounds) {
            hist.buckets.push_back({le, 0});
        }
        hist.buckets.push_back({std::numeric_limits<double>::infinity(), 0});
    }

    for (auto & bucket : hist.buckets) {
        if (value <= bucket.le) {
            bucket.count++;
        }
    }
    hist.sum += value;
    hist.count++;
}

void llama_prometheus_exporter::summary_observe(
        const std::string & name,
        double value,
        const std::map<std::string, std::string> & labels) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string key = make_label_key(labels);
    auto & summary = summaries_[name + key];

    summary.sum += value;
    summary.count++;

    // Note: Real implementation would maintain a sliding window
    // and calculate actual quantiles
}

void llama_prometheus_exporter::register_metric(
        const std::string & name,
        llama_metric_type type,
        const std::string & help) {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_[name].type = type;
    metrics_[name].help = help;
}

std::string llama_prometheus_exporter::get_metrics_text() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;

    // Add process metrics if enabled
    if (cfg_.enable_process_metrics) {
        add_process_metrics(oss);
    }

    // Output regular metrics
    for (const auto & [name, entry] : metrics_) {
        // HELP line
        if (!entry.help.empty()) {
            oss << "# HELP " << name << " " << entry.help << "\n";
        }

        // TYPE line
        std::string type_str;
        switch (entry.type) {
            case llama_metric_type::COUNTER:   type_str = "counter"; break;
            case llama_metric_type::GAUGE:     type_str = "gauge"; break;
            case llama_metric_type::HISTOGRAM: type_str = "histogram"; break;
            case llama_metric_type::SUMMARY:   type_str = "summary"; break;
        }
        oss << "# TYPE " << name << " " << type_str << "\n";

        // Values
        for (const auto & [label_key, value] : entry.values) {
            oss << name;
            if (!label_key.empty()) {
                oss << "{" << label_key << "}";
            }
            oss << " " << std::fixed << std::setprecision(6) << value << "\n";
        }
    }

    // Output histograms
    for (const auto & [key, hist] : histograms_) {
        if (!hist.help.empty()) {
            oss << "# HELP " << hist.name << " " << hist.help << "\n";
        }
        oss << "# TYPE " << hist.name << " histogram\n";

        std::string labels = format_labels(hist.labels);
        for (const auto & bucket : hist.buckets) {
            oss << hist.name << "_bucket{" << labels;
            if (!labels.empty()) oss << ",";
            oss << "le=\"";
            if (std::isinf(bucket.le)) {
                oss << "+Inf";
            } else {
                oss << bucket.le;
            }
            oss << "\"} " << bucket.count << "\n";
        }
        oss << hist.name << "_sum{" << labels << "} " << hist.sum << "\n";
        oss << hist.name << "_count{" << labels << "} " << hist.count << "\n";
    }

    return oss.str();
}

std::string llama_prometheus_exporter::get_openmetrics_text() const {
    // OpenMetrics format is similar but with some differences
    std::string text = get_metrics_text();
    text += "# EOF\n";  // OpenMetrics requires EOF marker
    return text;
}

double llama_prometheus_exporter::get_metric_value(
        const std::string & name,
        const std::map<std::string, std::string> & labels) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = metrics_.find(name);
    if (it == metrics_.end()) {
        return 0.0;
    }

    std::string key = make_label_key(labels);
    auto val_it = it->second.values.find(key);
    if (val_it == it->second.values.end()) {
        return 0.0;
    }

    return val_it->second;
}

void llama_prometheus_exporter::update_inference_metrics(
        int64_t tokens_generated,
        double tokens_per_sec,
        int64_t prompt_tokens,
        double prompt_eval_time_ms) {
    counter_set("llama_tokens_generated_total", static_cast<double>(tokens_generated));
    gauge_set("llama_tokens_per_second", tokens_per_sec);
    counter_set("llama_prompt_tokens_total", static_cast<double>(prompt_tokens));
    counter_set("llama_prompt_eval_seconds", prompt_eval_time_ms / 1000.0);
}

void llama_prometheus_exporter::update_memory_metrics(
        size_t vram_used,
        size_t vram_total,
        size_t ram_used,
        size_t ram_total) {
    gauge_set("llama_vram_bytes_used", static_cast<double>(vram_used));
    gauge_set("llama_vram_bytes_total", static_cast<double>(vram_total));
    gauge_set("llama_ram_bytes_used", static_cast<double>(ram_used));
    gauge_set("llama_ram_bytes_total", static_cast<double>(ram_total));
}

void llama_prometheus_exporter::update_layer_metrics(
        int gpu_layers,
        int cpu_layers,
        int64_t layers_evicted,
        int64_t layers_loaded) {
    gauge_set("llama_gpu_layers_active", static_cast<double>(gpu_layers));
    gauge_set("llama_cpu_layers_active", static_cast<double>(cpu_layers));
    counter_set("llama_layers_evicted_total", static_cast<double>(layers_evicted));
    counter_set("llama_layers_loaded_total", static_cast<double>(layers_loaded));
}

void llama_prometheus_exporter::update_kv_cache_metrics(
        uint32_t kv_pages_gpu,
        uint32_t kv_pages_cpu,
        int64_t pages_evicted,
        int64_t pages_prefetched) {
    gauge_set("llama_kv_pages_gpu", static_cast<double>(kv_pages_gpu));
    gauge_set("llama_kv_pages_cpu", static_cast<double>(kv_pages_cpu));
    counter_set("llama_kv_pages_evicted_total", static_cast<double>(pages_evicted));
    counter_set("llama_kv_pages_prefetched_total", static_cast<double>(pages_prefetched));
}

void llama_prometheus_exporter::update_request_metrics(
        int64_t requests_total,
        int64_t requests_active,
        double avg_latency_ms,
        double p99_latency_ms) {
    counter_set("llama_requests_total", static_cast<double>(requests_total));
    gauge_set("llama_requests_active", static_cast<double>(requests_active));
    histogram_observe("llama_request_latency_seconds", avg_latency_ms / 1000.0);
}

std::string llama_prometheus_exporter::make_label_key(
        const std::map<std::string, std::string> & labels) const {
    if (labels.empty()) return "";

    std::ostringstream oss;
    bool first = true;
    for (const auto & [k, v] : labels) {
        if (!first) oss << ",";
        oss << k << "=\"" << v << "\"";
        first = false;
    }
    return oss.str();
}

std::string llama_prometheus_exporter::format_labels(
        const std::map<std::string, std::string> & labels) const {
    return make_label_key(labels);
}

void llama_prometheus_exporter::add_process_metrics(std::ostringstream & oss) const {
    // Add standard process metrics
    oss << "# HELP process_start_time_seconds Start time of the process since unix epoch in seconds.\n";
    oss << "# TYPE process_start_time_seconds gauge\n";
    oss << "process_start_time_seconds " << std::time(nullptr) << "\n";

    // Note: Real implementation would add more process metrics like:
    // - process_cpu_seconds_total
    // - process_resident_memory_bytes
    // - process_virtual_memory_bytes
    // - process_open_fds
}

int64_t llama_prometheus_exporter::get_timestamp_ms() const {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

//
// llama_trace_span implementation
//

static std::string generate_id() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<uint64_t> dis;

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << dis(gen);
    return oss.str();
}

llama_trace_span::llama_trace_span(const std::string & name, const std::string & trace_id)
    : name_(name)
    , trace_id_(trace_id.empty() ? generate_id() : trace_id)
    , span_id_(generate_id()) {
    auto now = std::chrono::steady_clock::now();
    start_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

llama_trace_span::~llama_trace_span() {
    if (!ended_) {
        end();
    }
}

void llama_trace_span::set_attribute(const std::string & key, const std::string & value) {
    attributes_[key] = value;
}

void llama_trace_span::set_attribute(const std::string & key, int64_t value) {
    attributes_[key] = std::to_string(value);
}

void llama_trace_span::set_attribute(const std::string & key, double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    attributes_[key] = oss.str();
}

void llama_trace_span::add_event(
        const std::string & name,
        const std::map<std::string, std::string> & attributes) {
    auto now = std::chrono::steady_clock::now();
    int64_t timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << name;
    if (!attributes.empty()) {
        oss << " {";
        bool first = true;
        for (const auto & [k, v] : attributes) {
            if (!first) oss << ", ";
            oss << k << "=" << v;
            first = false;
        }
        oss << "}";
    }
    events_.emplace_back(timestamp, oss.str());
}

void llama_trace_span::set_status(bool ok, const std::string & message) {
    ok_ = ok;
    status_message_ = message;
}

void llama_trace_span::end() {
    if (ended_) return;

    auto now = std::chrono::steady_clock::now();
    end_time_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    ended_ = true;
}

int64_t llama_trace_span::get_duration_us() const {
    if (!ended_) {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count() - start_time_us_;
    }
    return end_time_us_ - start_time_us_;
}
