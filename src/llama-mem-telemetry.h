#pragma once

#include "ggml-backend.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

// Memory statistics for a single device
struct llama_mem_stats {
    size_t total_bytes     = 0;
    size_t free_bytes      = 0;
    size_t allocated_bytes = 0;
    float  utilization_pct = 0.0f;

    // Helper to compute utilization
    void update_utilization() {
        if (total_bytes > 0) {
            utilization_pct = 100.0f * (1.0f - static_cast<float>(free_bytes) / static_cast<float>(total_bytes));
        } else {
            utilization_pct = 0.0f;
        }
    }
};

// Memory telemetry system for real-time VRAM/RAM monitoring
// Used by dynamic layer scheduler and paged KV cache for memory-aware decisions
class llama_mem_telemetry {
public:
    // Configuration
    struct config {
        float pressure_threshold = 0.85f;  // Memory pressure threshold (0.0-1.0)
        float critical_threshold = 0.95f;  // Critical memory threshold (0.0-1.0)
        int64_t refresh_interval_us = 100000;  // Refresh interval in microseconds (100ms default)
    };

    llama_mem_telemetry();
    explicit llama_mem_telemetry(const config & cfg);
    ~llama_mem_telemetry() = default;

    // Refresh memory stats for all devices
    // Returns true if stats were actually refreshed (respects refresh interval)
    bool refresh(const std::vector<ggml_backend_dev_t> & devices);

    // Check if device is under memory pressure (above pressure_threshold)
    bool is_under_pressure(ggml_backend_dev_t dev) const;

    // Check if device is in critical memory state (above critical_threshold)
    bool is_critical(ggml_backend_dev_t dev) const;

    // Get available bytes for a device
    size_t available_bytes(ggml_backend_dev_t dev) const;

    // Get total bytes for a device
    size_t total_bytes(ggml_backend_dev_t dev) const;

    // Get full stats for a device
    bool get_stats(ggml_backend_dev_t dev, llama_mem_stats & stats) const;

    // Get CPU memory stats (system RAM)
    llama_mem_stats get_cpu_stats() const;

    // Update configuration
    void set_config(const config & cfg);
    const config & get_config() const;

    // Force immediate refresh on next call
    void invalidate_cache();

private:
    // Refresh CPU memory stats
    void refresh_cpu_stats();

    // Get device key for map lookup
    static uintptr_t device_key(ggml_backend_dev_t dev);

    config cfg_;
    mutable std::mutex mutex_;

    // Per-device memory stats
    std::unordered_map<uintptr_t, llama_mem_stats> device_stats_;

    // CPU memory stats
    llama_mem_stats cpu_stats_;

    // Last refresh timestamp
    std::chrono::steady_clock::time_point last_refresh_;
    bool cache_valid_ = false;
};
