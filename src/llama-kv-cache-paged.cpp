#include "llama-kv-cache-paged.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <chrono>
#include <set>

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
    // Clean up page buffers first
    for (auto & [id, page] : pages_) {
        if (page->buffer != nullptr) {
            ggml_backend_buffer_free(page->buffer);
            page->buffer = nullptr;
        }
    }
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

    // Allocate actual K/V tensors on CPU
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    if (cpu_buft != nullptr) {
        size_t alignment = ggml_backend_buft_get_alignment(cpu_buft);
        size_t total_size = k_size + v_size + 2 * alignment;  // Extra for alignment

        ggml_backend_buffer_t cpu_buffer = ggml_backend_buft_alloc_buffer(cpu_buft, total_size);
        if (cpu_buffer != nullptr) {
            page->buffer = cpu_buffer;

            // Create tensor context for K/V tensors
            // Note: In a full implementation, we'd create actual ggml_tensor objects here
            // For now, track the buffer allocation for migration
            LLAMA_LOG_DEBUG("%s: allocated page %u with %zu bytes for layer %d\n",
                __func__, page->page_id, total_size, il);
        } else {
            LLAMA_LOG_WARN("%s: failed to allocate CPU buffer for page %u\n",
                __func__, page->page_id);
        }
    }

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

    // Check if page has K/V tensors to evict
    if (page->k == nullptr && page->v == nullptr) {
        // No actual data, just update state
        page->location = LLAMA_KV_PAGE_CPU;
        gpu_memory_used_ -= page->size_bytes;
        cpu_memory_used_ += page->size_bytes;

        auto lru_it = std::find(gpu_lru_queue_.begin(), gpu_lru_queue_.end(), page_id);
        if (lru_it != gpu_lru_queue_.end()) {
            gpu_lru_queue_.erase(lru_it);
        }
        pages_evicted_++;
        return true;
    }

    LLAMA_LOG_DEBUG("%s: evicting page %u (layer %d, pos %u) from GPU to CPU\n",
        __func__, page_id, page->il, page->pos_start);

    // Get CPU buffer type
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    if (cpu_buft == nullptr) {
        LLAMA_LOG_WARN("%s: could not get CPU buffer type\n", __func__);
        return false;
    }

    // Allocate CPU buffer for page data
    size_t alignment = ggml_backend_buft_get_alignment(cpu_buft);
    size_t total_size = page->size_bytes;

    ggml_backend_buffer_t cpu_buffer = ggml_backend_buft_alloc_buffer(cpu_buft, total_size);
    if (cpu_buffer == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate CPU buffer for page %u\n", __func__, page_id);
        return false;
    }

    void * cpu_base = ggml_backend_buffer_get_base(cpu_buffer);
    size_t offset = 0;

    // Copy K tensor from GPU to CPU
    if (page->k != nullptr) {
        size_t k_size = ggml_nbytes(page->k);
        offset = (offset + alignment - 1) & ~(alignment - 1);
        void * k_dst = static_cast<char *>(cpu_base) + offset;

        ggml_backend_tensor_get(page->k, k_dst, 0, k_size);
        page->k->data = k_dst;
        page->k->buffer = cpu_buffer;
        offset += k_size;
    }

    // Copy V tensor from GPU to CPU
    if (page->v != nullptr) {
        size_t v_size = ggml_nbytes(page->v);
        offset = (offset + alignment - 1) & ~(alignment - 1);
        void * v_dst = static_cast<char *>(cpu_base) + offset;

        ggml_backend_tensor_get(page->v, v_dst, 0, v_size);
        page->v->data = v_dst;
        page->v->buffer = cpu_buffer;
        offset += v_size;
    }

    // Free old GPU buffer
    if (page->buffer != nullptr) {
        ggml_backend_buffer_free(page->buffer);
    }
    page->buffer = cpu_buffer;

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

    // Log metrics on eviction (use unlocked versions since we hold the lock)
    if (metrics_logger_) {
        metrics_logger_->set_kv_page_counts(get_gpu_page_count_unlocked(), get_cpu_page_count_unlocked());
        metrics_logger_->inc_kv_pages_evicted();
    }

    LLAMA_LOG_INFO("%s: page %u evicted to CPU successfully\n", __func__, page_id);
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

    // Check if page has K/V tensors to prefetch
    if (page->k == nullptr && page->v == nullptr) {
        // No actual data, just update state
        page->location = LLAMA_KV_PAGE_GPU;
        cpu_memory_used_ -= page->size_bytes;
        gpu_memory_used_ += page->size_bytes;
        gpu_lru_queue_.push_back(page_id);
        pages_prefetched_++;
        return true;
    }

    // Find a GPU device
    ggml_backend_dev_t target_dev = nullptr;
    if (telemetry_) {
        std::vector<ggml_backend_dev_t> devices = model_.devices;
        telemetry_->refresh(devices);

        for (auto * dev : devices) {
            if (dev == nullptr) continue;

            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU &&
                dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) {
                continue;
            }

            // Check if we need to evict
            if (telemetry_->is_under_pressure(dev)) {
                if (!ensure_gpu_memory(page->size_bytes)) {
                    return false;  // Could not free enough memory
                }
            }
            target_dev = dev;
            break;
        }
    } else {
        // No telemetry, try to find first GPU
        for (auto * dev : model_.devices) {
            if (dev == nullptr) continue;
            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU ||
                dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                target_dev = dev;
                break;
            }
        }
    }

    if (target_dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU device available for prefetch\n", __func__);
        return false;
    }

    LLAMA_LOG_DEBUG("%s: prefetching page %u (layer %d, pos %u) from CPU to GPU\n",
        __func__, page_id, page->il, page->pos_start);

    // Get GPU buffer type
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(target_dev);
    if (gpu_buft == nullptr) {
        LLAMA_LOG_WARN("%s: could not get GPU buffer type\n", __func__);
        return false;
    }

    // Allocate GPU buffer for page data
    size_t alignment = ggml_backend_buft_get_alignment(gpu_buft);
    size_t total_size = page->size_bytes;

    ggml_backend_buffer_t gpu_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, total_size);
    if (gpu_buffer == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate GPU buffer for page %u\n", __func__, page_id);
        return false;
    }

    void * gpu_base = ggml_backend_buffer_get_base(gpu_buffer);
    size_t offset = 0;

    // Save original K tensor pointers for rollback on failure
    void * k_orig_data = page->k ? page->k->data : nullptr;
    ggml_backend_buffer_t k_orig_buffer = page->k ? page->k->buffer : nullptr;

    // Copy K tensor from CPU to GPU
    if (page->k != nullptr) {
        size_t k_size = ggml_nbytes(page->k);
        offset = (offset + alignment - 1) & ~(alignment - 1);
        void * k_dst = static_cast<char *>(gpu_base) + offset;

        // Read from CPU, write to GPU
        void * temp = malloc(k_size);
        if (temp == nullptr) {
            ggml_backend_buffer_free(gpu_buffer);
            return false;
        }
        ggml_backend_tensor_get(page->k, temp, 0, k_size);
        page->k->data = k_dst;
        page->k->buffer = gpu_buffer;
        ggml_backend_tensor_set(page->k, temp, 0, k_size);
        free(temp);
        offset += k_size;
    }

    // Copy V tensor from CPU to GPU
    if (page->v != nullptr) {
        size_t v_size = ggml_nbytes(page->v);
        offset = (offset + alignment - 1) & ~(alignment - 1);
        void * v_dst = static_cast<char *>(gpu_base) + offset;

        void * temp = malloc(v_size);
        if (temp == nullptr) {
            // Rollback K tensor pointers before freeing buffer
            if (page->k != nullptr) {
                page->k->data = k_orig_data;
                page->k->buffer = k_orig_buffer;
            }
            ggml_backend_buffer_free(gpu_buffer);
            return false;
        }
        ggml_backend_tensor_get(page->v, temp, 0, v_size);
        page->v->data = v_dst;
        page->v->buffer = gpu_buffer;
        ggml_backend_tensor_set(page->v, temp, 0, v_size);
        free(temp);
        offset += v_size;
    }

    // Free old CPU buffer
    if (page->buffer != nullptr) {
        ggml_backend_buffer_free(page->buffer);
    }
    page->buffer = gpu_buffer;

    // Update memory tracking
    cpu_memory_used_ -= page->size_bytes;
    gpu_memory_used_ += page->size_bytes;

    page->location = LLAMA_KV_PAGE_GPU;

    // Add to GPU LRU queue
    gpu_lru_queue_.push_back(page_id);

    pages_prefetched_++;

    // Log metrics on prefetch (use unlocked versions since we hold the lock)
    if (metrics_logger_) {
        metrics_logger_->set_kv_page_counts(get_gpu_page_count_unlocked(), get_cpu_page_count_unlocked());
        metrics_logger_->inc_kv_pages_prefetched();
    }

    LLAMA_LOG_INFO("%s: page %u prefetched to GPU successfully\n", __func__, page_id);
    return true;
}

uint32_t llama_kv_cache_paged::get_gpu_page_count_unlocked() const {
    // Note: mutex should already be held by caller
    uint32_t count = 0;
    for (const auto & [id, page] : pages_) {
        if (page->location == LLAMA_KV_PAGE_GPU) {
            count++;
        }
    }
    return count;
}

uint32_t llama_kv_cache_paged::get_cpu_page_count_unlocked() const {
    // Note: mutex should already be held by caller
    uint32_t count = 0;
    for (const auto & [id, page] : pages_) {
        if (page->location == LLAMA_KV_PAGE_CPU) {
            count++;
        }
    }
    return count;
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

uint32_t llama_kv_cache_paged::coalesce_pages(int32_t il) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!cfg_.coalesce_pages) {
        return 0;
    }

    // Find all pages for this layer, sorted by position
    std::vector<llama_kv_page *> layer_pages;
    for (auto & [id, page] : pages_) {
        if (page->il == il) {
            layer_pages.push_back(page.get());
        }
    }

    if (layer_pages.size() < cfg_.coalesce_threshold) {
        return 0;  // Not enough pages to coalesce
    }

    // Sort by position
    std::sort(layer_pages.begin(), layer_pages.end(),
        [](const llama_kv_page * a, const llama_kv_page * b) {
            return a->pos_start < b->pos_start;
        });

    // Find runs of adjacent pages on the same device
    uint32_t coalesced = 0;
    size_t run_start = 0;

    while (run_start < layer_pages.size()) {
        // Find end of run (adjacent pages, same location)
        size_t run_end = run_start + 1;
        while (run_end < layer_pages.size()) {
            auto * curr = layer_pages[run_end - 1];
            auto * next = layer_pages[run_end];

            // Check adjacency: next page starts where current ends
            if (next->pos_start != curr->pos_start + curr->n_tokens) {
                break;
            }
            // Check same location
            if (next->location != curr->location) {
                break;
            }
            run_end++;
        }

        size_t run_length = run_end - run_start;
        if (run_length >= cfg_.coalesce_threshold) {
            // Coalesce these pages by keeping the first and removing others
            // Note: This is metadata-only coalescing for now
            // Full data coalescing would require buffer reallocation

            llama_kv_page * first = layer_pages[run_start];
            uint32_t total_tokens = 0;
            size_t total_bytes = 0;

            for (size_t i = run_start; i < run_end; ++i) {
                total_tokens += layer_pages[i]->n_tokens;
                total_bytes += layer_pages[i]->size_bytes;
            }

            // Update first page to span the entire range
            first->n_tokens = total_tokens;
            first->size_bytes = total_bytes;

            // Mark other pages for removal (don't actually remove during iteration)
            for (size_t i = run_start + 1; i < run_end; ++i) {
                layer_pages[i]->n_tokens = 0;  // Mark for removal
            }

            coalesced += static_cast<uint32_t>(run_length - 1);

            LLAMA_LOG_DEBUG("%s: coalesced %zu pages at layer %d pos %u-%u\n",
                __func__, run_length, il, first->pos_start, first->pos_start + total_tokens);
        }

        run_start = run_end;
    }

    // Remove marked pages
    for (auto it = pages_.begin(); it != pages_.end(); ) {
        if (it->second->n_tokens == 0) {
            // Remove from index
            page_key_t key = make_page_key(it->second->il, it->second->pos_start, cfg_.page_size);
            page_index_.erase(key);

            // Remove from LRU if on GPU
            if (it->second->location == LLAMA_KV_PAGE_GPU) {
                auto lru_it = std::find(gpu_lru_queue_.begin(), gpu_lru_queue_.end(), it->first);
                if (lru_it != gpu_lru_queue_.end()) {
                    gpu_lru_queue_.erase(lru_it);
                }
            }

            // Free buffer if any
            if (it->second->buffer != nullptr) {
                ggml_backend_buffer_free(it->second->buffer);
            }

            it = pages_.erase(it);
        } else {
            ++it;
        }
    }

    if (coalesced > 0) {
        pages_coalesced_ += coalesced;
        LLAMA_LOG_INFO("%s: coalesced %u pages for layer %d\n", __func__, coalesced, il);
    }

    return coalesced;
}

uint32_t llama_kv_cache_paged::coalesce_all_pages() {
    // Get unique layer IDs
    std::set<int32_t> layers;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto & [id, page] : pages_) {
            layers.insert(page->il);
        }
    }

    uint32_t total_coalesced = 0;
    for (int32_t il : layers) {
        total_coalesced += coalesce_pages(il);
    }

    return total_coalesced;
}

//
// Async prefetch support
//

bool llama_kv_cache_paged::request_prefetch(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return false;
    }

    // Already on GPU
    if (it->second->location == LLAMA_KV_PAGE_GPU) {
        return true;
    }

    // Check if already in queue
    auto queue_it = std::find(pending_prefetches_.begin(), pending_prefetches_.end(), page_id);
    if (queue_it != pending_prefetches_.end()) {
        return true;  // Already queued
    }

    pending_prefetches_.push_back(page_id);
    return true;
}

bool llama_kv_cache_paged::request_prefetch_range(int32_t il, uint32_t pos_start, uint32_t pos_end) {
    std::lock_guard<std::mutex> lock(mutex_);

    uint32_t page_start = pos_start / cfg_.page_size;
    uint32_t page_end = (pos_end + cfg_.page_size - 1) / cfg_.page_size;

    bool any_queued = false;
    for (uint32_t page_num = page_start; page_num < page_end; ++page_num) {
        page_key_t key = {il, page_num};
        auto it = page_index_.find(key);
        if (it != page_index_.end()) {
            auto page_it = pages_.find(it->second);
            if (page_it != pages_.end() && page_it->second->location != LLAMA_KV_PAGE_GPU) {
                // Check if not already in queue
                auto queue_it = std::find(pending_prefetches_.begin(), pending_prefetches_.end(), it->second);
                if (queue_it == pending_prefetches_.end()) {
                    pending_prefetches_.push_back(it->second);
                    any_queued = true;
                }
            }
        }
    }

    return any_queued;
}

size_t llama_kv_cache_paged::process_pending_prefetches(size_t max_count) {
    std::vector<uint32_t> to_prefetch;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t count = std::min(max_count, pending_prefetches_.size());
        for (size_t i = 0; i < count; ++i) {
            to_prefetch.push_back(pending_prefetches_.front());
            pending_prefetches_.pop_front();
        }
    }

    size_t prefetched = 0;
    for (uint32_t page_id : to_prefetch) {
        if (prefetch_page(page_id)) {
            prefetched++;
        }
    }

    return prefetched;
}

size_t llama_kv_cache_paged::get_pending_prefetch_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_prefetches_.size();
}

//
// Synchronization helpers
//

void llama_kv_cache_paged::synchronize_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return;
    }

    auto & page = it->second;
    if (page->location != LLAMA_KV_PAGE_GPU || page->buffer == nullptr) {
        return;  // Nothing to synchronize
    }

    // Synchronize the buffer's backend
    // This ensures any pending GPU operations on this page are complete
    ggml_backend_buffer_t buffer = page->buffer;
    if (buffer != nullptr) {
        // Get the buffer's backend type and synchronize if possible
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
        if (buft != nullptr) {
            // The actual synchronization happens through ggml's buffer mechanisms
            // For now, this is a marker that the page is ready
            page->last_access_us = get_time_us();
        }
    }
}

void llama_kv_cache_paged::synchronize_all_pages() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto & [id, page] : pages_) {
        if (page->location == LLAMA_KV_PAGE_GPU && page->buffer != nullptr) {
            page->last_access_us = get_time_us();
        }
    }

    // Synchronize all GPU devices we're using
    if (telemetry_) {
        std::vector<ggml_backend_dev_t> devices = model_.devices;
        for (auto * dev : devices) {
            if (dev == nullptr) continue;

            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU ||
                dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                // Create temporary backend for synchronization
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend != nullptr) {
                    ggml_backend_synchronize(backend);
                    ggml_backend_free(backend);
                }
            }
        }
    }
}

bool llama_kv_cache_paged::check_memory_pressure() {
    if (!telemetry_) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<ggml_backend_dev_t> devices = model_.devices;
    telemetry_->refresh(devices);

    bool evicted_any = false;
    for (auto * dev : devices) {
        if (dev == nullptr) continue;

        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        // Check if under pressure
        if (telemetry_->is_under_pressure(dev)) {
            if (!in_pressure_mode_) {
                in_pressure_mode_ = true;
                LLAMA_LOG_INFO("%s: entering memory pressure mode\n", __func__);
            }

            // Evict pages until below low threshold
            while (!gpu_lru_queue_.empty()) {
                uint32_t lru_id = gpu_lru_queue_.front();

                // Check if pressure is relieved
                telemetry_->refresh(devices);
                float pressure = 1.0f - (static_cast<float>(telemetry_->available_bytes(dev)) /
                                         static_cast<float>(telemetry_->total_bytes(dev)));
                if (pressure < cfg_.pressure_low_thresh) {
                    in_pressure_mode_ = false;
                    break;
                }

                // Evict the LRU page (releases lock temporarily)
                mutex_.unlock();
                bool evicted = evict_page(lru_id);
                mutex_.lock();

                if (evicted) {
                    evicted_any = true;
                }
            }
        } else if (in_pressure_mode_) {
            // Check if we can exit pressure mode
            float pressure = 1.0f - (static_cast<float>(telemetry_->available_bytes(dev)) /
                                     static_cast<float>(telemetry_->total_bytes(dev)));
            if (pressure < cfg_.pressure_low_thresh) {
                in_pressure_mode_ = false;
                LLAMA_LOG_INFO("%s: exiting memory pressure mode\n", __func__);
            }
        }
    }

    return evicted_any;
}

llama_kv_page * llama_kv_cache_paged::get_page(uint32_t page_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return nullptr;
    }
    return it->second.get();
}

const llama_kv_page * llama_kv_cache_paged::get_page(uint32_t page_id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pages_.find(page_id);
    if (it == pages_.end()) {
        return nullptr;
    }
    return it->second.get();
}
