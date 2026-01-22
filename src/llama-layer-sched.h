#pragma once

#include "llama-mem-telemetry.h"
#include "llama-metrics.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_layer;
struct ggml_tensor;

// Layer location for dynamic scheduling
enum llama_layer_location {
    LLAMA_LAYER_LOC_GPU,   // Layer weights on GPU memory
    LLAMA_LAYER_LOC_CPU,   // Layer weights on CPU memory
    LLAMA_LAYER_LOC_DISK,  // Layer weights on disk (future)
};

// Information about a layer's current state
struct llama_layer_state {
    int32_t                  il           = -1;       // Layer index
    llama_layer_location     location     = LLAMA_LAYER_LOC_CPU;
    ggml_backend_dev_t       device       = nullptr;  // Current device (if on GPU)
    size_t                   size_bytes   = 0;        // Total size of layer tensors
    int64_t                  last_access_us = 0;      // Last access timestamp
    bool                     is_migrating = false;    // Currently being migrated
};

// Dynamic layer scheduler for memory-aware layer placement
// Implements AirLLM-style runtime layer migration based on memory pressure
class llama_layer_scheduler {
public:
    struct config {
        float   pressure_threshold = 0.85f;   // Memory pressure threshold
        float   critical_threshold = 0.95f;   // Critical memory threshold
        int32_t min_gpu_layers     = 0;       // Minimum layers to keep on GPU
        size_t  migration_batch    = 1;       // Number of layers to migrate at once
        bool    prefetch_enabled   = true;    // Enable prefetch hints
    };

    llama_layer_scheduler(
        llama_model & model,
        llama_mem_telemetry & telemetry,
        const config & cfg);

    ~llama_layer_scheduler() = default;

    // Prepare a layer for computation
    // Returns the device where the layer should run
    // May trigger migration if needed
    ggml_backend_dev_t prepare_layer(int32_t il);

    // Evict LRU layers from GPU to free memory
    // Returns the number of bytes freed
    size_t evict_layers(ggml_backend_dev_t dev, size_t required_bytes);

    // Hint that a layer will be needed soon (for prefetching)
    void hint_prefetch(int32_t il);

    // Check if a layer is currently on GPU
    bool is_on_gpu(int32_t il) const;

    // Get the current location of a layer
    llama_layer_location get_location(int32_t il) const;

    // Get the current state of a layer
    bool get_layer_state(int32_t il, llama_layer_state & state) const;

    // Get total GPU memory used by layers
    size_t get_gpu_memory_used() const;

    // Get total CPU memory used by layers
    size_t get_cpu_memory_used() const;

    // Get the number of layers currently on GPU
    int32_t get_gpu_layer_count() const;

    // Update configuration
    void set_config(const config & cfg);
    const config & get_config() const;

    // Enable/disable dynamic scheduling
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // Get the number of layers currently on CPU
    int32_t get_cpu_layer_count() const;

    // Get migration statistics
    int64_t get_migrations_to_gpu() const { return migrations_to_gpu_; }
    int64_t get_migrations_to_cpu() const { return migrations_to_cpu_; }

    // Set metrics logger for structured logging
    void set_metrics_logger(llama_metrics_logger * logger) { metrics_logger_ = logger; }

private:
    // Internal migration functions
    bool migrate_to_gpu(int32_t il);
    bool migrate_to_cpu(int32_t il);

    // Calculate layer size from model
    size_t calculate_layer_size(int32_t il) const;

    // Find LRU layer on GPU
    int32_t find_lru_gpu_layer() const;

    // Update access time for a layer
    void touch_layer(int32_t il);

    // Get current time in microseconds
    static int64_t get_time_us();

    llama_model & model_;
    llama_mem_telemetry & telemetry_;
    config cfg_;

    mutable std::mutex mutex_;

    // Per-layer state
    std::vector<llama_layer_state> layer_states_;

    // LRU tracking for GPU layers
    std::deque<int32_t> gpu_lru_queue_;

    // Total memory tracking
    size_t gpu_memory_used_ = 0;
    size_t cpu_memory_used_ = 0;

    // Enable flag
    bool enabled_ = true;

    // Statistics
    int64_t migrations_to_gpu_ = 0;
    int64_t migrations_to_cpu_ = 0;

    // Metrics logger (optional)
    llama_metrics_logger * metrics_logger_ = nullptr;
};
