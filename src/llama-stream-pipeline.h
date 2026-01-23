#pragma once

#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Forward declarations
struct ggml_tensor;
struct llama_model;

// Stream operation types
enum class llama_stream_op_type {
    COMPUTE,        // GPU compute operation
    TRANSFER_H2D,   // Host to Device transfer
    TRANSFER_D2H,   // Device to Host transfer
    TRANSFER_D2D,   // Device to Device transfer (peer)
    SYNCHRONIZE,    // Stream synchronization
    CALLBACK,       // Custom callback
};

// Stream operation
struct llama_stream_op {
    llama_stream_op_type type = llama_stream_op_type::COMPUTE;
    int stream_id = 0;
    int layer_id = -1;

    // For transfers
    void * src_ptr = nullptr;
    void * dst_ptr = nullptr;
    size_t size = 0;
    int src_device = -1;
    int dst_device = -1;

    // For compute
    ggml_tensor * tensor = nullptr;

    // For callbacks
    std::function<void()> callback;

    // Timing
    int64_t enqueue_time_us = 0;
    int64_t start_time_us = 0;
    int64_t end_time_us = 0;
};

// Stream statistics
struct llama_stream_stats {
    int64_t total_ops = 0;
    int64_t compute_ops = 0;
    int64_t transfer_ops = 0;
    int64_t total_bytes_transferred = 0;
    double avg_compute_time_ms = 0.0;
    double avg_transfer_time_ms = 0.0;
    double overlap_efficiency = 0.0;  // How well we overlap compute/transfer
};

// CUDA-style stream for pipelining operations
class llama_stream {
public:
    explicit llama_stream(int device_id = 0, int stream_id = 0);
    ~llama_stream();

    // Enqueue operations
    void enqueue_transfer_h2d(void * dst, const void * src, size_t size);
    void enqueue_transfer_d2h(void * dst, const void * src, size_t size);
    void enqueue_transfer_d2d(void * dst, const void * src, size_t size, int dst_device);
    void enqueue_compute(ggml_tensor * tensor);
    void enqueue_callback(std::function<void()> callback);

    // Synchronization
    void synchronize();
    bool is_complete() const;

    // Get stream info
    int get_device_id() const { return device_id_; }
    int get_stream_id() const { return stream_id_; }
    size_t get_pending_ops() const;

    // Statistics
    llama_stream_stats get_stats() const;

private:
    void worker_thread();
    void process_op(llama_stream_op & op);
    int64_t get_time_us();

    int device_id_;
    int stream_id_;

    std::deque<llama_stream_op> op_queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    std::thread worker_;
    std::atomic<bool> running_{true};
    std::atomic<bool> idle_{true};

    // Statistics
    llama_stream_stats stats_;
};

// Pipeline manager for overlapped execution
class llama_stream_pipeline {
public:
    struct config {
        int num_compute_streams = 2;     // Parallel compute streams
        int num_transfer_streams = 2;    // Parallel transfer streams
        int prefetch_depth = 2;          // How many layers to prefetch ahead
        bool enable_overlap = true;      // Enable compute/transfer overlap
        bool enable_peer_transfer = true; // Enable GPU-to-GPU direct transfer
    };

    llama_stream_pipeline();
    explicit llama_stream_pipeline(const config & cfg);
    ~llama_stream_pipeline();

    // Initialize pipeline for device
    bool initialize(int device_id);

    // Get streams
    llama_stream * get_compute_stream(int index = 0);
    llama_stream * get_transfer_stream(int index = 0);

    // Pipeline operations for layer processing
    void begin_layer(int layer_id);
    void end_layer(int layer_id);

    // Prefetch next layer(s) while computing current
    void prefetch_layer(int layer_id, const std::vector<ggml_tensor *> & tensors);

    // Synchronize all streams
    void synchronize_all();

    // Wait for specific layer to complete
    void wait_for_layer(int layer_id);

    // Check if layer is ready
    bool is_layer_ready(int layer_id) const;

    // Get aggregate statistics
    struct pipeline_stats {
        llama_stream_stats compute_stats;
        llama_stream_stats transfer_stats;
        double overall_overlap_ratio = 0.0;
        int64_t total_layers_processed = 0;
    };
    pipeline_stats get_stats() const;

    // Configuration
    const config & get_config() const { return cfg_; }

private:
    config cfg_;
    int device_id_ = 0;

    std::vector<std::unique_ptr<llama_stream>> compute_streams_;
    std::vector<std::unique_ptr<llama_stream>> transfer_streams_;

    // Layer completion tracking (protected by layer_mutex_)
    std::map<int, bool> layer_complete_;
    mutable std::mutex layer_mutex_;

    // Current stream indices for round-robin
    std::atomic<int> current_compute_stream_{0};
    std::atomic<int> current_transfer_stream_{0};

    // Statistics
    std::atomic<int64_t> layers_processed_{0};
};

// Global pipeline instance
llama_stream_pipeline * llama_get_stream_pipeline();
void llama_set_stream_pipeline(llama_stream_pipeline * pipeline);
