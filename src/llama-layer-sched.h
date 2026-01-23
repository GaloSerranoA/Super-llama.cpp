#pragma once

#include "llama-mem-telemetry.h"
#include "llama-metrics.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

struct llama_model;
struct llama_layer;
struct ggml_tensor;
struct ggml_context;

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

    // Buffer for migrated tensors (owned by scheduler when migrated)
    ggml_backend_buffer_t    migration_buffer = nullptr;
};

// Dynamic layer scheduler for memory-aware layer placement
// Implements AirLLM-style runtime layer migration based on memory pressure
class llama_layer_scheduler {
public:
    struct config {
        float   pressure_threshold    = 0.85f;   // Memory pressure threshold (start evicting)
        float   pressure_low_thresh   = 0.70f;   // Hysteresis low threshold (stop evicting)
        float   critical_threshold    = 0.95f;   // Critical memory threshold
        int32_t min_gpu_layers        = 0;       // Minimum layers to keep on GPU
        size_t  migration_batch       = 4;       // Number of layers to migrate at once (batch migration)
        bool    prefetch_enabled      = true;    // Enable prefetch hints
        bool    use_pinned_memory     = true;    // Use pinned memory for faster transfers
        bool    graceful_degradation  = true;    // Continue on CPU if GPU migration fails
        std::vector<int32_t> pinned_layers;      // Layers to always keep on GPU (never evict)
    };

    llama_layer_scheduler(
        llama_model & model,
        llama_mem_telemetry & telemetry,
        const config & cfg);

    ~llama_layer_scheduler();

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

    // Get migration timing metrics
    double get_avg_migration_time_ms() const;
    size_t get_total_bytes_migrated() const { return total_bytes_migrated_; }

    // Get memory watermarks
    size_t get_gpu_memory_watermark() const { return gpu_memory_watermark_; }
    size_t get_cpu_memory_watermark() const { return cpu_memory_watermark_; }

    // Check if layer is pinned (never evicted)
    bool is_layer_pinned(int32_t il) const;

    // Set metrics logger for structured logging
    void set_metrics_logger(llama_metrics_logger * logger) { metrics_logger_ = logger; }

    // Backend synchronization
    void synchronize_device(ggml_backend_dev_t dev);
    void synchronize_all_devices();

    // Wait for a layer's migration to complete
    // Returns true if migration completed, false on timeout
    bool wait_for_migration(int32_t il, int64_t timeout_us = 0);

    // Async migration support
    bool request_async_migration(int32_t il, llama_layer_location target);
    size_t process_pending_migrations(size_t max_count = 1);
    size_t get_pending_migration_count() const;

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

    // Tensor migration helpers
    // Get list of all tensors in a layer
    std::vector<ggml_tensor *> get_layer_tensors(int32_t il);

    // Copy a single tensor to a new buffer on target device
    bool copy_tensor_to_buffer(ggml_tensor * tensor, ggml_backend_buffer_t dst_buffer, size_t & offset);

    // Free migration buffer for a layer
    void free_migration_buffer(int32_t il);

    // Batch migration helpers
    bool migrate_batch_to_gpu(const std::vector<int32_t> & layers);
    bool migrate_batch_to_cpu(const std::vector<int32_t> & layers);

    // Pinned memory management
    void * alloc_pinned_memory(size_t size);
    void free_pinned_memory(void * ptr);

    // Check hysteresis state
    bool should_evict(float current_pressure) const;
    bool should_prefetch(float current_pressure) const;

    // Update memory watermarks
    void update_watermarks();

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

    // Migration timing metrics
    int64_t total_migration_time_us_ = 0;
    int64_t migration_count_ = 0;
    size_t total_bytes_migrated_ = 0;

    // Memory watermarks (high-water marks)
    size_t gpu_memory_watermark_ = 0;
    size_t cpu_memory_watermark_ = 0;

    // Hysteresis state
    bool in_pressure_mode_ = false;  // true when actively evicting

    // Pinned memory pool (for reuse)
    std::vector<std::pair<void *, size_t>> pinned_buffers_;

    // Pending async migrations queue
    std::deque<std::pair<int32_t, llama_layer_location>> pending_migrations_;

    // Metrics logger (optional)
    llama_metrics_logger * metrics_logger_ = nullptr;
};
