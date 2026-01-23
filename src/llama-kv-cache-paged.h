#pragma once

#include "llama-kv-cache.h"
#include "llama-mem-telemetry.h"
#include "llama-metrics.h"

#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

// KV page location
enum llama_kv_page_location {
    LLAMA_KV_PAGE_GPU,   // Page is on GPU memory
    LLAMA_KV_PAGE_CPU,   // Page is on CPU memory
    LLAMA_KV_PAGE_DISK,  // Page is on disk (future)
};

// A single KV cache page containing K/V data for a range of positions
struct llama_kv_page {
    uint32_t                 page_id       = 0;      // Unique page identifier
    int32_t                  il            = -1;     // Layer index
    uint32_t                 pos_start     = 0;      // Starting position in sequence
    uint32_t                 n_tokens      = 0;      // Number of tokens in this page
    llama_kv_page_location   location      = LLAMA_KV_PAGE_CPU;
    ggml_tensor            * k             = nullptr; // K tensor for this page
    ggml_tensor            * v             = nullptr; // V tensor for this page
    int64_t                  last_access_us = 0;     // Last access timestamp
    bool                     dirty         = false;  // Page has been modified
    size_t                   size_bytes    = 0;      // Total size of K+V data

    // Buffer for migrated page data (owned by page when migrated)
    ggml_backend_buffer_t    buffer        = nullptr;
};

// Paged KV cache that supports GPU/CPU spilling for memory efficiency
// Implements the llama_memory_i interface like the regular llama_kv_cache
class llama_kv_cache_paged : public llama_memory_i {
public:
    struct config {
        uint32_t page_size            = 256;   // Tokens per page
        float    pressure_thresh      = 0.85f; // GPU memory pressure threshold (start evicting)
        float    pressure_low_thresh  = 0.70f; // Hysteresis low threshold (stop evicting)
        float    critical_thresh      = 0.95f; // Critical threshold for forced eviction
        size_t   max_cpu_pages        = 0;     // Max pages in CPU (0 = unlimited)
        bool     prefetch_enabled     = true;  // Enable page prefetching
        bool     coalesce_pages       = true;  // Enable page coalescing for adjacent pages
        uint32_t coalesce_threshold   = 4;     // Minimum adjacent pages to trigger coalescing
    };

    llama_kv_cache_paged(
            const llama_model   & model,
                    ggml_type     type_k,
                    ggml_type     type_v,
                         bool     v_trans,
                         bool     offload,
                         bool     unified,
                     uint32_t     kv_size,
                     uint32_t     n_seq_max,
                     uint32_t     n_pad,
                     uint32_t     n_swa,
               llama_swa_type     swa_type,
        const layer_filter_cb   & filter,
        const layer_reuse_cb    & reuse,
        llama_mem_telemetry     * telemetry,
        const config            & cfg);

    ~llama_kv_cache_paged();

    //
    // llama_memory_i interface implementation
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // Paged KV cache specific API
    //

    // Allocate a new page for a layer starting at a position
    llama_kv_page * allocate_page(int32_t il, uint32_t pos_start);

    // Evict a page from GPU to CPU
    bool evict_page(uint32_t page_id);

    // Prefetch a page from CPU to GPU
    bool prefetch_page(uint32_t page_id);

    // Get page count statistics
    uint32_t get_gpu_page_count() const;
    uint32_t get_cpu_page_count() const;
    uint32_t get_total_page_count() const;

    // Get memory usage statistics
    size_t get_gpu_memory_used() const;
    size_t get_cpu_memory_used() const;

    // Find pages for a position range
    std::vector<llama_kv_page *> find_pages(int32_t il, uint32_t pos_start, uint32_t pos_end);

    // Access the underlying regular KV cache (for fallback/compatibility)
    llama_kv_cache * get_base_cache() { return base_cache_.get(); }
    const llama_kv_cache * get_base_cache() const { return base_cache_.get(); }

    // Get eviction/prefetch statistics
    int64_t get_pages_evicted() const { return pages_evicted_; }
    int64_t get_pages_prefetched() const { return pages_prefetched_; }
    int64_t get_pages_coalesced() const { return pages_coalesced_; }

    // Coalesce adjacent pages on the same layer
    // Returns number of pages coalesced
    uint32_t coalesce_pages(int32_t il);

    // Coalesce all layers
    uint32_t coalesce_all_pages();

    // Set metrics logger for structured logging
    void set_metrics_logger(llama_metrics_logger * logger) { metrics_logger_ = logger; }

    // Async prefetch support - queue pages for background prefetch
    bool request_prefetch(uint32_t page_id);
    bool request_prefetch_range(int32_t il, uint32_t pos_start, uint32_t pos_end);
    size_t process_pending_prefetches(size_t max_count = 1);
    size_t get_pending_prefetch_count() const;

    // Synchronization helpers
    void synchronize_page(uint32_t page_id);
    void synchronize_all_pages();

    // Check memory pressure and trigger evictions if needed
    bool check_memory_pressure();

    // Get page by ID
    llama_kv_page * get_page(uint32_t page_id);
    const llama_kv_page * get_page(uint32_t page_id) const;

private:
    // Internal helpers
    uint32_t next_page_id();
    void touch_page(uint32_t page_id);
    llama_kv_page * find_lru_gpu_page() const;
    bool ensure_gpu_memory(size_t required_bytes);
    static int64_t get_time_us();

    // Internal page count helpers (no mutex - caller must hold lock)
    uint32_t get_gpu_page_count_unlocked() const;
    uint32_t get_cpu_page_count_unlocked() const;

    // Page key: (layer_id, page_number)
    using page_key_t = std::pair<int32_t, uint32_t>;
    static page_key_t make_page_key(int32_t il, uint32_t pos_start, uint32_t page_size) {
        return {il, pos_start / page_size};
    }

    const llama_model & model_;
    llama_mem_telemetry * telemetry_;  // May be null if telemetry disabled
    config cfg_;

    mutable std::mutex mutex_;

    // Underlying traditional KV cache (used as fallback and for actual storage)
    std::unique_ptr<llama_kv_cache> base_cache_;

    // Page management
    uint32_t next_page_id_ = 1;
    std::unordered_map<uint32_t, std::unique_ptr<llama_kv_page>> pages_;  // page_id -> page

    // Page lookup by (layer, position)
    std::map<page_key_t, uint32_t> page_index_;  // (il, page_num) -> page_id

    // LRU tracking for GPU pages
    std::deque<uint32_t> gpu_lru_queue_;

    // Memory tracking
    size_t gpu_memory_used_ = 0;
    size_t cpu_memory_used_ = 0;

    // Statistics
    int64_t pages_evicted_ = 0;
    int64_t pages_prefetched_ = 0;
    int64_t pages_coalesced_ = 0;

    // Hysteresis state
    bool in_pressure_mode_ = false;

    // Pending prefetch queue
    std::deque<uint32_t> pending_prefetches_;

    // Metrics logger (optional)
    llama_metrics_logger * metrics_logger_ = nullptr;
};
