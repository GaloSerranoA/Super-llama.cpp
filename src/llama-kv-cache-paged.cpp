#include "llama-kv-cache-paged.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>

llama_kv_cache_paged::llama_kv_cache_paged(
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
    const config            & cfg)
    : model_(model)
    , telemetry_(telemetry)
    , cfg_(cfg) {

    // Create the underlying KV cache
    // The paged cache wraps the base cache and adds page management on top
    base_cache_ = std::make_unique<llama_kv_cache>(
        model, type_k, type_v, v_trans, offload, unified,
        kv_size, n_seq_max, n_pad, n_swa, swa_type, filter, reuse);

    LLAMA_LOG_INFO("%s: paged KV cache initialized (page_size=%u, pressure_thresh=%.2f)\n",
        __func__, cfg_.page_size, cfg_.pressure_thresh);
}

llama_kv_cache_paged::~llama_kv_cache_paged() {
    // Clean up pages
    pages_.clear();
}

//
// llama_memory_i interface - delegate to base cache with page management
//

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    // For now, delegate to base cache
    // Full paged implementation would manage pages here
    return base_cache_->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    return base_cache_->init_full();
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * lctx, bool optimize) {
    return base_cache_->init_update(lctx, optimize);
}

bool llama_kv_cache_paged::get_can_shift() const {
    return base_cache_->get_can_shift();
}

void llama_kv_cache_paged::clear(bool data) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Clear page management
    pages_.clear();
    page_index_.clear();
    gpu_lru_queue_.clear();
    next_page_id_ = 1;
    gpu_memory_used_ = 0;
    cpu_memory_used_ = 0;

    // Clear base cache
    base_cache_->clear(data);
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Delegate to base cache for now
    // Full implementation would also invalidate affected pages
    return base_cache_->seq_rm(seq_id, p0, p1);
}

void llama_kv_cache_paged::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    base_cache_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    base_cache_->seq_keep(seq_id);
}

void llama_kv_cache_paged::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    base_cache_->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_paged::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    base_cache_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    return base_cache_->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    return base_cache_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    return base_cache_->memory_breakdown();
}

void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    base_cache_->state_write(io, seq_id, flags);
}

void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    base_cache_->state_read(io, seq_id, flags);
}

//
// Paged KV cache specific implementation
//

llama_kv_page * llama_kv_cache_paged::allocate_page(int32_t il, uint32_t pos_start) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if page already exists
    page_key_t key = make_page_key(il, pos_start, cfg_.page_size);
    auto it = page_index_.find(key);
    if (it != page_index_.end()) {
        auto page_it = pages_.find(it->second);
        if (page_it != pages_.end()) {
            touch_page(it->second);
            return page_it->second.get();
        }
    }

    // Create new page
    auto page = std::make_unique<llama_kv_page>();
    page->page_id = next_page_id();
    page->il = il;
    page->pos_start = (pos_start / cfg_.page_size) * cfg_.page_size;  // Align to page boundary
    page->n_tokens = cfg_.page_size;
    page->location = LLAMA_KV_PAGE_CPU;  // Start on CPU
    page->last_access_us = get_time_us();

    // Calculate page size based on model dimensions
    const auto & hparams = model_.hparams;
    size_t k_size = cfg_.page_size * hparams.n_embd_head_k * hparams.n_head_kv() * ggml_type_size(GGML_TYPE_F16);
    size_t v_size = cfg_.page_size * hparams.n_embd_head_v * hparams.n_head_kv() * ggml_type_size(GGML_TYPE_F16);
    page->size_bytes = k_size + v_size;

    // Note: Actual K/V tensor allocation would happen here
    // For this implementation, we track metadata only
    // The actual data is managed by the base_cache_

    cpu_memory_used_ += page->size_bytes;

    uint32_t page_id = page->page_id;
    llama_kv_page * page_ptr = page.get();

    pages_[page_id] = std::move(page);
    page_index_[key] = page_id;

    return page_ptr;
}

bool llama_kv_cache_paged::evict_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return false;
    }

    auto & page = it->second;
    if (page->location != LLAMA_KV_PAGE_GPU) {
        return true;  // Already not on GPU
    }

    // Evict to CPU
    // In a full implementation, this would:
    // 1. Allocate CPU buffer
    // 2. Copy K/V data from GPU to CPU
    // 3. Free GPU buffer
    // 4. Update tensor pointers

    LLAMA_LOG_DEBUG("%s: evicting page %u (layer %d, pos %u) from GPU to CPU\n",
        __func__, page_id, page->il, page->pos_start);

    // Update memory tracking
    gpu_memory_used_ -= page->size_bytes;
    cpu_memory_used_ += page->size_bytes;

    page->location = LLAMA_KV_PAGE_CPU;

    // Remove from GPU LRU queue
    auto lru_it = std::find(gpu_lru_queue_.begin(), gpu_lru_queue_.end(), page_id);
    if (lru_it != gpu_lru_queue_.end()) {
        gpu_lru_queue_.erase(lru_it);
    }

    pages_evicted_++;

    // Log metrics on eviction
    if (metrics_logger_) {
        metrics_logger_->set_kv_page_counts(get_gpu_page_count(), get_cpu_page_count());
        metrics_logger_->inc_kv_pages_evicted();
    }

    return true;
}

bool llama_kv_cache_paged::prefetch_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return false;
    }

    auto & page = it->second;
    if (page->location == LLAMA_KV_PAGE_GPU) {
        return true;  // Already on GPU
    }

    // Check if we have GPU memory available
    if (telemetry_) {
        std::vector<ggml_backend_dev_t> devices = model_.devices;
        telemetry_->refresh(devices);

        // Find first GPU with available memory
        for (auto * dev : devices) {
            if (dev == nullptr) continue;

            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU &&
                dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }

            // Check if we need to evict
            if (telemetry_->is_under_pressure(dev)) {
                // Evict LRU pages until we have enough memory
                if (!ensure_gpu_memory(page->size_bytes)) {
                    return false;  // Could not free enough memory
                }
            }
            break;
        }
    }

    // Prefetch to GPU
    // In a full implementation, this would:
    // 1. Allocate GPU buffer
    // 2. Copy K/V data from CPU to GPU
    // 3. Free CPU buffer
    // 4. Update tensor pointers

    LLAMA_LOG_DEBUG("%s: prefetching page %u (layer %d, pos %u) from CPU to GPU\n",
        __func__, page_id, page->il, page->pos_start);

    // Update memory tracking
    cpu_memory_used_ -= page->size_bytes;
    gpu_memory_used_ += page->size_bytes;

    page->location = LLAMA_KV_PAGE_GPU;

    // Add to GPU LRU queue
    gpu_lru_queue_.push_back(page_id);

    pages_prefetched_++;

    // Log metrics on prefetch
    if (metrics_logger_) {
        metrics_logger_->set_kv_page_counts(get_gpu_page_count(), get_cpu_page_count());
        metrics_logger_->inc_kv_pages_prefetched();
    }

    return true;
}

uint32_t llama_kv_cache_paged::get_gpu_page_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t count = 0;
    for (const auto & [id, page] : pages_) {
        if (page->location == LLAMA_KV_PAGE_GPU) {
            count++;
        }
    }
    return count;
}

uint32_t llama_kv_cache_paged::get_cpu_page_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t count = 0;
    for (const auto & [id, page] : pages_) {
        if (page->location == LLAMA_KV_PAGE_CPU) {
            count++;
        }
    }
    return count;
}

uint32_t llama_kv_cache_paged::get_total_page_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<uint32_t>(pages_.size());
}

size_t llama_kv_cache_paged::get_gpu_memory_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu_memory_used_;
}

size_t llama_kv_cache_paged::get_cpu_memory_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cpu_memory_used_;
}

std::vector<llama_kv_page *> llama_kv_cache_paged::find_pages(int32_t il, uint32_t pos_start, uint32_t pos_end) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<llama_kv_page *> result;

    uint32_t page_start = pos_start / cfg_.page_size;
    uint32_t page_end = (pos_end + cfg_.page_size - 1) / cfg_.page_size;

    for (uint32_t page_num = page_start; page_num < page_end; ++page_num) {
        page_key_t key = {il, page_num};
        auto it = page_index_.find(key);
        if (it != page_index_.end()) {
            auto page_it = pages_.find(it->second);
            if (page_it != pages_.end()) {
                result.push_back(page_it->second.get());
            }
        }
    }

    return result;
}

//
// Internal helpers
//

uint32_t llama_kv_cache_paged::next_page_id() {
    return next_page_id_++;
}

void llama_kv_cache_paged::touch_page(uint32_t page_id) {
    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return;
    }

    it->second->last_access_us = get_time_us();

    // Update LRU position if on GPU
    if (it->second->location == LLAMA_KV_PAGE_GPU) {
        auto lru_it = std::find(gpu_lru_queue_.begin(), gpu_lru_queue_.end(), page_id);
        if (lru_it != gpu_lru_queue_.end()) {
            gpu_lru_queue_.erase(lru_it);
        }
        gpu_lru_queue_.push_back(page_id);
    }
}

llama_kv_page * llama_kv_cache_paged::find_lru_gpu_page() const {
    // Note: mutex should already be held
    if (gpu_lru_queue_.empty()) {
        return nullptr;
    }

    uint32_t lru_id = gpu_lru_queue_.front();
    auto it = pages_.find(lru_id);
    if (it == pages_.end()) {
        return nullptr;
    }

    return it->second.get();
}

bool llama_kv_cache_paged::ensure_gpu_memory(size_t required_bytes) {
    // Note: mutex should already be held
    size_t freed = 0;

    while (freed < required_bytes && !gpu_lru_queue_.empty()) {
        llama_kv_page * lru_page = find_lru_gpu_page();
        if (!lru_page) {
            break;
        }

        size_t page_size = lru_page->size_bytes;
        uint32_t page_id = lru_page->page_id;

        // Temporarily release lock for eviction (which acquires it)
        // This is safe because we're done reading gpu_lru_queue_ for this iteration
        // Actually, let's just do the eviction inline since we hold the lock

        // Update memory tracking
        gpu_memory_used_ -= page_size;
        cpu_memory_used_ += page_size;

        lru_page->location = LLAMA_KV_PAGE_CPU;

        // Remove from GPU LRU queue
        gpu_lru_queue_.pop_front();

        pages_evicted_++;
        freed += page_size;
    }

    return freed >= required_bytes;
}

int64_t llama_kv_cache_paged::get_time_us() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}
