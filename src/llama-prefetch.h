#pragma once

#include "ggml-backend.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class llama_layer_scheduler;
class llama_kv_cache_paged;

// Prefetch request types
enum llama_prefetch_type {
    LLAMA_PREFETCH_LAYER,     // Prefetch a model layer
    LLAMA_PREFETCH_KV_PAGE,   // Prefetch a KV cache page
};

// Prefetch request
struct llama_prefetch_request {
    llama_prefetch_type type;
    int32_t  layer_id  = -1;      // For layer prefetch
    uint32_t page_id   = 0;       // For KV page prefetch
    int32_t  priority  = 0;       // Higher = more urgent
    std::promise<bool> promise;   // Completion promise
};

// Async prefetch system for overlapping data loading with computation
// Works with layer scheduler and paged KV cache for memory-efficient inference
class llama_prefetcher {
public:
    struct config {
        int32_t  n_workers          = 1;      // Number of worker threads
        int32_t  queue_size         = 16;     // Max pending requests
        int32_t  lookahead_layers   = 2;      // Layers to prefetch ahead
        int32_t  lookahead_pages    = 4;      // KV pages to prefetch ahead
        bool     auto_prefetch      = true;   // Enable automatic prefetching
    };

    llama_prefetcher(
        llama_layer_scheduler * layer_sched,
        llama_kv_cache_paged  * kv_cache,
        const config          & cfg);

    ~llama_prefetcher();

    // Queue a layer for prefetching
    // Returns a future that resolves when prefetch completes
    std::future<bool> prefetch_layer(int32_t il, int32_t priority = 0);

    // Queue a KV page for prefetching
    std::future<bool> prefetch_kv_page(uint32_t page_id, int32_t priority = 0);

    // Called at the start of each layer computation
    // Automatically queues prefetch for upcoming layers
    void on_layer_start(int32_t il);

    // Called when a KV page is accessed
    // Automatically queues prefetch for upcoming pages
    void on_kv_access(uint32_t page_id, int32_t il);

    // Wait for all pending prefetches to complete
    void sync();

    // Cancel all pending prefetches
    void cancel_all();

    // Get queue statistics
    size_t get_pending_count() const;
    int64_t get_completed_count() const;
    int64_t get_failed_count() const;

    // Enable/disable prefetching
    void set_enabled(bool enabled);
    bool is_enabled() const;

    // Update configuration
    void set_config(const config & cfg);
    const config & get_config() const;

private:
    // Worker thread main function
    void worker_thread();

    // Process a single prefetch request
    bool process_request(llama_prefetch_request & req);

    llama_layer_scheduler * layer_sched_;  // May be null
    llama_kv_cache_paged  * kv_cache_;     // May be null
    config cfg_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;

    // Request queue (sorted deque - higher priority at front)
    // Using deque because std::promise is move-only and priority_queue
    // has issues with move-only types in some implementations
    std::deque<llama_prefetch_request> request_queue_;

    // Insert request in priority order (higher priority = earlier position)
    void insert_request_sorted(llama_prefetch_request req);

    // Worker threads
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    std::atomic<bool> enabled_{true};

    // Statistics
    std::atomic<int64_t> completed_count_{0};
    std::atomic<int64_t> failed_count_{0};

    // Tracking last prefetched to avoid duplicates
    int32_t last_prefetched_layer_ = -1;
    uint32_t last_prefetched_page_ = 0;
};
