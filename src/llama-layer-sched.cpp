#include "llama-layer-sched.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

llama_layer_scheduler::llama_layer_scheduler(
    llama_model & model,
    llama_mem_telemetry & telemetry,
    const config & cfg)
    : model_(model)
    , telemetry_(telemetry)
    , cfg_(cfg) {

    const auto & hparams = model_.hparams;
    const int32_t n_layer = static_cast<int32_t>(hparams.n_layer);

    // Initialize layer states
    layer_states_.resize(n_layer);

    for (int32_t il = 0; il < n_layer; ++il) {
        auto & state = layer_states_[il];
        state.il = il;
        state.size_bytes = calculate_layer_size(il);
        state.last_access_us = get_time_us();

        // Determine initial location based on model's device assignment
        ggml_backend_dev_t dev = model_.dev_layer(il);
        if (dev != nullptr) {
            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU ||
                dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                state.location = LLAMA_LAYER_LOC_GPU;
                state.device = dev;
                gpu_lru_queue_.push_back(il);
                gpu_memory_used_ += state.size_bytes;
            } else {
                state.location = LLAMA_LAYER_LOC_CPU;
                state.device = nullptr;
                cpu_memory_used_ += state.size_bytes;
            }
        } else {
            state.location = LLAMA_LAYER_LOC_CPU;
            state.device = nullptr;
            cpu_memory_used_ += state.size_bytes;
        }
    }

    LLAMA_LOG_INFO("%s: initialized layer scheduler with %d layers (GPU: %d, CPU: %d)\n",
        __func__, n_layer, get_gpu_layer_count(), n_layer - get_gpu_layer_count());
}

llama_layer_scheduler::~llama_layer_scheduler() {
    // Free any migration buffers we created
    for (auto & state : layer_states_) {
        if (state.migration_buffer != nullptr) {
            ggml_backend_buffer_free(state.migration_buffer);
            state.migration_buffer = nullptr;
        }
    }

    // Free pinned memory buffers
    for (auto & [ptr, size] : pinned_buffers_) {
        free_pinned_memory(ptr);
    }
    pinned_buffers_.clear();
}

ggml_backend_dev_t llama_layer_scheduler::prepare_layer(int32_t il) {
    if (!enabled_) {
        return model_.dev_layer(il);
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return nullptr;
    }

    auto & state = layer_states_[il];
    touch_layer(il);

    // If layer is already on GPU, return its device
    if (state.location == LLAMA_LAYER_LOC_GPU && state.device != nullptr) {
        return state.device;
    }

    // Layer is on CPU, check if we should migrate to GPU
    // First, refresh telemetry and check memory pressure
    std::vector<ggml_backend_dev_t> devices = model_.devices;
    if (devices.empty()) {
        // No devices available, stay on CPU
        return nullptr;
    }
    telemetry_.refresh(devices);

    // Find a GPU device with available memory
    for (auto * dev : devices) {
        if (dev == nullptr) {
            continue;
        }

        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        // Check if we have enough memory
        size_t available = telemetry_.available_bytes(dev);
        if (available >= state.size_bytes && !telemetry_.is_under_pressure(dev)) {
            // We have enough memory, migrate to GPU
            if (migrate_to_gpu(il)) {
                return state.device;
            }
        }

        // Memory pressure - try to evict LRU layers
        if (telemetry_.is_under_pressure(dev)) {
            size_t needed = state.size_bytes;
            size_t freed = evict_layers(dev, needed);
            if (freed >= needed) {
                // Retry migration after eviction
                if (migrate_to_gpu(il)) {
                    return state.device;
                }
            }
        }
    }

    // Could not migrate to GPU, return CPU device (nullptr indicates CPU)
    return nullptr;
}

size_t llama_layer_scheduler::evict_layers(ggml_backend_dev_t dev, size_t required_bytes) {
    // Note: mutex should already be held by caller
    size_t freed = 0;

    // Enter pressure mode
    in_pressure_mode_ = true;

    // Collect layers to evict (batch migration)
    std::vector<int32_t> layers_to_evict;
    size_t batch_size = 0;

    while ((freed + batch_size) < required_bytes && !gpu_lru_queue_.empty()) {
        // Find LRU layer on the specified device (skip pinned layers)
        int32_t lru_layer = -1;
        int64_t oldest_access = INT64_MAX;

        for (int32_t il : gpu_lru_queue_) {
            // Skip pinned layers
            if (is_layer_pinned(il)) {
                continue;
            }

            const auto & state = layer_states_[il];
            if (state.location == LLAMA_LAYER_LOC_GPU &&
                state.device == dev &&
                !state.is_migrating) {

                // Skip layers already queued for eviction
                if (std::find(layers_to_evict.begin(), layers_to_evict.end(), il) != layers_to_evict.end()) {
                    continue;
                }

                if (state.last_access_us < oldest_access) {
                    oldest_access = state.last_access_us;
                    lru_layer = il;
                }
            }
        }

        if (lru_layer < 0) {
            // No more evictable layers
            break;
        }

        // Don't evict below minimum GPU layers
        int32_t current_gpu_count = get_gpu_layer_count();
        int32_t pending_evictions = static_cast<int32_t>(layers_to_evict.size());
        if ((current_gpu_count - pending_evictions) <= cfg_.min_gpu_layers) {
            break;
        }

        layers_to_evict.push_back(lru_layer);
        batch_size += layer_states_[lru_layer].size_bytes;

        // Check if we have a full batch
        if (layers_to_evict.size() >= cfg_.migration_batch) {
            // Migrate the batch
            if (migrate_batch_to_cpu(layers_to_evict)) {
                freed += batch_size;
            }
            layers_to_evict.clear();
            batch_size = 0;
        }
    }

    // Migrate remaining layers (partial batch)
    if (!layers_to_evict.empty()) {
        if (migrate_batch_to_cpu(layers_to_evict)) {
            freed += batch_size;
        }
    }

    // Update watermarks
    update_watermarks();

    return freed;
}

void llama_layer_scheduler::hint_prefetch(int32_t il) {
    if (!enabled_ || !cfg_.prefetch_enabled) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return;
    }

    // Mark layer for prefetch (actual prefetch handled by llama_prefetcher)
    // This is just a hint for the prefetcher
}

bool llama_layer_scheduler::is_on_gpu(int32_t il) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return false;
    }

    return layer_states_[il].location == LLAMA_LAYER_LOC_GPU;
}

llama_layer_location llama_layer_scheduler::get_location(int32_t il) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return LLAMA_LAYER_LOC_CPU;
    }

    return layer_states_[il].location;
}

bool llama_layer_scheduler::get_layer_state(int32_t il, llama_layer_state & state) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return false;
    }

    state = layer_states_[il];
    return true;
}

size_t llama_layer_scheduler::get_gpu_memory_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return gpu_memory_used_;
}

size_t llama_layer_scheduler::get_cpu_memory_used() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cpu_memory_used_;
}

int32_t llama_layer_scheduler::get_gpu_layer_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t count = 0;
    for (const auto & state : layer_states_) {
        if (state.location == LLAMA_LAYER_LOC_GPU) {
            count++;
        }
    }
    return count;
}

int32_t llama_layer_scheduler::get_cpu_layer_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int32_t count = 0;
    for (const auto & state : layer_states_) {
        if (state.location == LLAMA_LAYER_LOC_CPU) {
            count++;
        }
    }
    return count;
}

void llama_layer_scheduler::set_config(const config & cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_ = cfg;
}

const llama_layer_scheduler::config & llama_layer_scheduler::get_config() const {
    return cfg_;
}

void llama_layer_scheduler::set_enabled(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = enabled;
}

bool llama_layer_scheduler::is_enabled() const {
    return enabled_;
}

bool llama_layer_scheduler::migrate_to_gpu(int32_t il) {
    // Note: mutex should already be held by caller
    auto & state = layer_states_[il];

    if (state.location == LLAMA_LAYER_LOC_GPU) {
        return true;  // Already on GPU
    }

    if (state.is_migrating) {
        return false;  // Already migrating
    }

    state.is_migrating = true;

    // Start timing
    int64_t start_time = get_time_us();

    // Find a suitable GPU device
    ggml_backend_dev_t target_dev = nullptr;
    if (model_.devices.empty()) {
        LLAMA_LOG_WARN("%s: no devices available for layer %d migration\n", __func__, il);
        state.is_migrating = false;
        return false;
    }
    for (auto * dev : model_.devices) {
        if (dev == nullptr) continue;
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            target_dev = dev;
            break;
        }
    }

    if (target_dev == nullptr) {
        LLAMA_LOG_WARN("%s: no GPU device available for layer %d migration\n", __func__, il);
        state.is_migrating = false;

        // Graceful degradation: continue on CPU
        if (cfg_.graceful_degradation) {
            LLAMA_LOG_INFO("%s: layer %d will run on CPU (graceful degradation)\n", __func__, il);
            return false;
        }
        return false;
    }

    // Get all tensors in this layer
    std::vector<ggml_tensor *> tensors = get_layer_tensors(il);
    if (tensors.empty()) {
        LLAMA_LOG_WARN("%s: layer %d has no tensors to migrate\n", __func__, il);
        state.is_migrating = false;
        return false;
    }

    // Get GPU buffer type
    ggml_backend_buffer_type_t gpu_buft = ggml_backend_dev_buffer_type(target_dev);
    if (gpu_buft == nullptr) {
        LLAMA_LOG_WARN("%s: could not get GPU buffer type for layer %d\n", __func__, il);
        state.is_migrating = false;
        return false;
    }

    // Calculate total size needed (with alignment)
    size_t alignment = ggml_backend_buft_get_alignment(gpu_buft);
    size_t total_size = 0;
    for (ggml_tensor * t : tensors) {
        size_t tensor_size = ggml_nbytes(t);
        // Align each tensor
        total_size = (total_size + alignment - 1) & ~(alignment - 1);
        total_size += tensor_size;
    }

    // Allocate GPU buffer
    ggml_backend_buffer_t gpu_buffer = ggml_backend_buft_alloc_buffer(gpu_buft, total_size);
    if (gpu_buffer == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate GPU buffer (%zu bytes) for layer %d\n",
            __func__, total_size, il);
        state.is_migrating = false;

        // Graceful degradation: continue on CPU
        if (cfg_.graceful_degradation) {
            LLAMA_LOG_INFO("%s: layer %d will run on CPU (GPU memory exhausted)\n", __func__, il);
        }
        return false;
    }

    LLAMA_LOG_DEBUG("%s: migrating layer %d to GPU (%zu bytes, %zu tensors)\n",
        __func__, il, total_size, tensors.size());

    // Copy each tensor to GPU
    size_t offset = 0;
    void * gpu_base = ggml_backend_buffer_get_base(gpu_buffer);
    bool success = true;

    for (ggml_tensor * t : tensors) {
        size_t tensor_size = ggml_nbytes(t);

        // Align offset
        offset = (offset + alignment - 1) & ~(alignment - 1);

        // Calculate destination address
        void * dst_addr = static_cast<char *>(gpu_base) + offset;

        // Allocate temp buffer (use pinned memory if enabled for faster transfers)
        void * temp_data = nullptr;
        if (cfg_.use_pinned_memory) {
            temp_data = alloc_pinned_memory(tensor_size);
        }
        if (temp_data == nullptr) {
            temp_data = malloc(tensor_size);
        }

        if (temp_data == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate temp buffer for tensor copy\n", __func__);
            // Clean up GPU buffer before returning
            ggml_backend_buffer_free(gpu_buffer);
            state.is_migrating = false;
            return false;
        }

        // Get data from current (CPU) location
        ggml_backend_tensor_get(t, temp_data, 0, tensor_size);

        // Update tensor to point to new GPU buffer location
        t->data = dst_addr;
        t->buffer = gpu_buffer;

        // Set data on GPU
        ggml_backend_tensor_set(t, temp_data, 0, tensor_size);

        // Free temp buffer
        if (cfg_.use_pinned_memory) {
            free_pinned_memory(temp_data);
        } else {
            free(temp_data);
        }

        offset += tensor_size;

        LLAMA_LOG_DEBUG("%s:   tensor %s: %zu bytes -> GPU\n",
            __func__, ggml_get_name(t), tensor_size);
    }

    if (!success) {
        // Rollback on failure
        ggml_backend_buffer_free(gpu_buffer);
        state.is_migrating = false;
        return false;
    }

    // Free old migration buffer if we had one (from previous CPU migration)
    free_migration_buffer(il);

    // Store the new GPU buffer
    state.migration_buffer = gpu_buffer;
    state.location = LLAMA_LAYER_LOC_GPU;
    state.device = target_dev;

    // Update memory tracking
    cpu_memory_used_ -= state.size_bytes;
    gpu_memory_used_ += state.size_bytes;

    // Add to GPU LRU queue
    gpu_lru_queue_.push_back(il);

    state.is_migrating = false;
    migrations_to_gpu_++;

    // Update timing metrics
    int64_t elapsed = get_time_us() - start_time;
    total_migration_time_us_ += elapsed;
    migration_count_++;
    total_bytes_migrated_ += total_size;

    // Update watermarks
    update_watermarks();

    // Log metrics
    if (metrics_logger_) {
        metrics_logger_->set_layer_counts(get_gpu_layer_count(), get_cpu_layer_count());
        metrics_logger_->inc_layers_loaded();
    }

    LLAMA_LOG_INFO("%s: layer %d migrated to GPU successfully (%.2f ms, %.2f MB/s)\n",
        __func__, il,
        elapsed / 1000.0,
        (total_size / (1024.0 * 1024.0)) / (elapsed / 1000000.0));
    return true;
}

bool llama_layer_scheduler::migrate_to_cpu(int32_t il) {
    // Note: mutex should already be held by caller
    auto & state = layer_states_[il];

    if (state.location == LLAMA_LAYER_LOC_CPU) {
        return true;  // Already on CPU
    }

    if (state.is_migrating) {
        return false;  // Already migrating
    }

    state.is_migrating = true;

    // Get all tensors in this layer
    std::vector<ggml_tensor *> tensors = get_layer_tensors(il);
    if (tensors.empty()) {
        LLAMA_LOG_WARN("%s: layer %d has no tensors to migrate\n", __func__, il);
        state.is_migrating = false;
        return false;
    }

    // Get CPU buffer type
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
    if (cpu_buft == nullptr) {
        LLAMA_LOG_WARN("%s: could not get CPU buffer type for layer %d\n", __func__, il);
        state.is_migrating = false;
        return false;
    }

    // Calculate total size needed (with alignment)
    size_t alignment = ggml_backend_buft_get_alignment(cpu_buft);
    size_t total_size = 0;
    for (ggml_tensor * t : tensors) {
        size_t tensor_size = ggml_nbytes(t);
        total_size = (total_size + alignment - 1) & ~(alignment - 1);
        total_size += tensor_size;
    }

    // Allocate CPU buffer
    ggml_backend_buffer_t cpu_buffer = ggml_backend_buft_alloc_buffer(cpu_buft, total_size);
    if (cpu_buffer == nullptr) {
        LLAMA_LOG_WARN("%s: failed to allocate CPU buffer (%zu bytes) for layer %d\n",
            __func__, total_size, il);
        state.is_migrating = false;
        return false;
    }

    LLAMA_LOG_DEBUG("%s: migrating layer %d to CPU (%zu bytes, %zu tensors)\n",
        __func__, il, total_size, tensors.size());

    // Copy each tensor to CPU
    size_t offset = 0;
    void * cpu_base = ggml_backend_buffer_get_base(cpu_buffer);
    bool success = true;

    for (ggml_tensor * t : tensors) {
        size_t tensor_size = ggml_nbytes(t);

        // Align offset
        offset = (offset + alignment - 1) & ~(alignment - 1);

        // Calculate destination address
        void * dst_addr = static_cast<char *>(cpu_base) + offset;

        // Copy tensor data from GPU to CPU
        // First, read tensor data from GPU into the destination (CPU buffer)
        ggml_backend_tensor_get(t, dst_addr, 0, tensor_size);

        // Update tensor to point to new CPU buffer location
        t->data = dst_addr;
        t->buffer = cpu_buffer;

        offset += tensor_size;

        LLAMA_LOG_DEBUG("%s:   tensor %s: %zu bytes -> CPU\n",
            __func__, ggml_get_name(t), tensor_size);
    }

    if (!success) {
        ggml_backend_buffer_free(cpu_buffer);
        state.is_migrating = false;
        return false;
    }

    // Free old migration buffer (the GPU buffer we created or inherited)
    free_migration_buffer(il);

    // Store the new CPU buffer
    state.migration_buffer = cpu_buffer;

    // Remove from GPU LRU queue
    auto it = std::find(gpu_lru_queue_.begin(), gpu_lru_queue_.end(), il);
    if (it != gpu_lru_queue_.end()) {
        gpu_lru_queue_.erase(it);
    }

    // Update memory tracking
    gpu_memory_used_ -= state.size_bytes;
    cpu_memory_used_ += state.size_bytes;

    state.location = LLAMA_LAYER_LOC_CPU;
    state.device = nullptr;
    state.is_migrating = false;
    migrations_to_cpu_++;

    // Log metrics on eviction
    if (metrics_logger_) {
        metrics_logger_->set_layer_counts(get_gpu_layer_count(), get_cpu_layer_count());
        metrics_logger_->inc_layers_evicted();
    }

    LLAMA_LOG_INFO("%s: layer %d migrated to CPU successfully\n", __func__, il);
    return true;
}

size_t llama_layer_scheduler::calculate_layer_size(int32_t il) const {
    if (il < 0 || il >= static_cast<int32_t>(model_.layers.size())) {
        return 0;
    }

    const auto & layer = model_.layers[il];
    size_t total_size = 0;

    // Helper lambda to add tensor size
    auto add_tensor = [&](const ggml_tensor * t) {
        if (t != nullptr) {
            total_size += ggml_nbytes(t);
        }
    };

    // Count all common layer tensors
    // Normalization
    add_tensor(layer.attn_norm);
    add_tensor(layer.attn_norm_b);
    add_tensor(layer.attn_norm_2);
    add_tensor(layer.attn_norm_2_b);
    add_tensor(layer.attn_q_norm);
    add_tensor(layer.attn_q_norm_b);
    add_tensor(layer.attn_k_norm);
    add_tensor(layer.attn_k_norm_b);

    // Attention weights
    add_tensor(layer.wq);
    add_tensor(layer.wk);
    add_tensor(layer.wv);
    add_tensor(layer.wo);
    add_tensor(layer.wqkv);
    add_tensor(layer.wq_a);
    add_tensor(layer.wq_b);
    add_tensor(layer.wkv_a_mqa);
    add_tensor(layer.wkv_b);

    // Attention biases
    add_tensor(layer.bq);
    add_tensor(layer.bk);
    add_tensor(layer.bv);
    add_tensor(layer.bo);
    add_tensor(layer.bqkv);

    // FFN normalization
    add_tensor(layer.ffn_norm);
    add_tensor(layer.ffn_norm_b);

    // FFN weights
    add_tensor(layer.ffn_gate);
    add_tensor(layer.ffn_down);
    add_tensor(layer.ffn_up);

    // MoE
    add_tensor(layer.ffn_gate_inp);
    add_tensor(layer.ffn_gate_exps);
    add_tensor(layer.ffn_down_exps);
    add_tensor(layer.ffn_up_exps);

    // SSM/Mamba
    add_tensor(layer.ssm_in);
    add_tensor(layer.ssm_x);
    add_tensor(layer.ssm_dt);
    add_tensor(layer.ssm_out);
    add_tensor(layer.ssm_conv1d);
    add_tensor(layer.ssm_a);
    add_tensor(layer.ssm_d);

    return total_size;
}

int32_t llama_layer_scheduler::find_lru_gpu_layer() const {
    // Note: mutex should already be held by caller
    int32_t lru_layer = -1;
    int64_t oldest_access = INT64_MAX;

    for (int32_t il : gpu_lru_queue_) {
        const auto & state = layer_states_[il];
        if (state.location == LLAMA_LAYER_LOC_GPU && !state.is_migrating) {
            if (state.last_access_us < oldest_access) {
                oldest_access = state.last_access_us;
                lru_layer = il;
            }
        }
    }

    return lru_layer;
}

void llama_layer_scheduler::touch_layer(int32_t il) {
    // Note: mutex should already be held by caller
    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return;
    }

    layer_states_[il].last_access_us = get_time_us();

    // Update LRU queue position
    if (layer_states_[il].location == LLAMA_LAYER_LOC_GPU) {
        auto it = std::find(gpu_lru_queue_.begin(), gpu_lru_queue_.end(), il);
        if (it != gpu_lru_queue_.end()) {
            gpu_lru_queue_.erase(it);
        }
        gpu_lru_queue_.push_back(il);  // Move to end (most recently used)
    }
}

int64_t llama_layer_scheduler::get_time_us() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

std::vector<ggml_tensor *> llama_layer_scheduler::get_layer_tensors(int32_t il) {
    std::vector<ggml_tensor *> tensors;

    if (il < 0 || il >= static_cast<int32_t>(model_.layers.size())) {
        return tensors;
    }

    const auto & layer = model_.layers[il];

    // Helper lambda to add non-null tensor
    auto add_tensor = [&](ggml_tensor * t) {
        if (t != nullptr) {
            tensors.push_back(t);
        }
    };

    // Normalization tensors
    add_tensor(layer.attn_norm);
    add_tensor(layer.attn_norm_b);
    add_tensor(layer.attn_norm_2);
    add_tensor(layer.attn_norm_2_b);
    add_tensor(layer.attn_q_norm);
    add_tensor(layer.attn_q_norm_b);
    add_tensor(layer.attn_k_norm);
    add_tensor(layer.attn_k_norm_b);
    add_tensor(layer.attn_out_norm);
    add_tensor(layer.attn_out_norm_b);
    add_tensor(layer.attn_q_a_norm);
    add_tensor(layer.attn_kv_a_norm);
    add_tensor(layer.attn_sub_norm);
    add_tensor(layer.attn_post_norm);
    add_tensor(layer.ffn_sub_norm);
    add_tensor(layer.attn_norm_cross);
    add_tensor(layer.attn_norm_enc);
    add_tensor(layer.ssm_norm);
    add_tensor(layer.ssm_dt_norm);
    add_tensor(layer.ssm_b_norm);
    add_tensor(layer.ssm_c_norm);

    // Attention weights
    add_tensor(layer.wq);
    add_tensor(layer.wk);
    add_tensor(layer.wv);
    add_tensor(layer.wo);
    add_tensor(layer.wqkv);
    add_tensor(layer.wq_a);
    add_tensor(layer.wq_b);
    add_tensor(layer.wkv_a_mqa);
    add_tensor(layer.wkv_b);
    add_tensor(layer.wk_b);
    add_tensor(layer.wv_b);
    add_tensor(layer.wq_cross);
    add_tensor(layer.wk_cross);
    add_tensor(layer.wv_cross);
    add_tensor(layer.wo_cross);
    add_tensor(layer.wq_enc);
    add_tensor(layer.wk_enc);
    add_tensor(layer.wv_enc);
    add_tensor(layer.wo_enc);
    add_tensor(layer.wqkv_gate);

    // Attention biases
    add_tensor(layer.bq);
    add_tensor(layer.bk);
    add_tensor(layer.bv);
    add_tensor(layer.bo);
    add_tensor(layer.bqkv);

    // Relative position bias
    add_tensor(layer.attn_rel_b);
    add_tensor(layer.attn_rel_b_enc);
    add_tensor(layer.attn_rel_b_cross);

    // FFN normalization
    add_tensor(layer.ffn_norm);
    add_tensor(layer.ffn_norm_b);
    add_tensor(layer.ffn_post_norm);
    add_tensor(layer.layer_out_norm);
    add_tensor(layer.layer_out_norm_b);
    add_tensor(layer.ffn_norm_exps);
    add_tensor(layer.ffn_norm_enc);

    // FFN weights
    add_tensor(layer.ffn_gate);
    add_tensor(layer.ffn_down);
    add_tensor(layer.ffn_up);
    add_tensor(layer.ffn_gate_enc);
    add_tensor(layer.ffn_down_enc);
    add_tensor(layer.ffn_up_enc);

    // MoE
    add_tensor(layer.ffn_gate_inp);
    add_tensor(layer.ffn_gate_exps);
    add_tensor(layer.ffn_down_exps);
    add_tensor(layer.ffn_up_exps);
    add_tensor(layer.ffn_gate_inp_b);
    add_tensor(layer.ffn_gate_exps_b);
    add_tensor(layer.ffn_down_exps_b);
    add_tensor(layer.ffn_up_exps_b);

    // Shared experts
    add_tensor(layer.ffn_gate_inp_shexp);
    add_tensor(layer.ffn_gate_shexp);
    add_tensor(layer.ffn_down_shexp);
    add_tensor(layer.ffn_up_shexp);

    // Adjugate experts
    add_tensor(layer.ffn_gate_chexps);
    add_tensor(layer.ffn_down_chexps);
    add_tensor(layer.ffn_up_chexps);

    // FFN biases
    add_tensor(layer.ffn_gate_b);
    add_tensor(layer.ffn_down_b);
    add_tensor(layer.ffn_up_b);
    add_tensor(layer.ffn_act);
    add_tensor(layer.ffn_exp_probs_b);

    // Mamba/SSM
    add_tensor(layer.ssm_in);
    add_tensor(layer.ssm_x);
    add_tensor(layer.ssm_dt);
    add_tensor(layer.ssm_out);
    add_tensor(layer.ssm_conv1d);
    add_tensor(layer.ssm_a);
    add_tensor(layer.ssm_d);
    add_tensor(layer.ssm_conv1d_b);
    add_tensor(layer.ssm_dt_b);
    add_tensor(layer.ssm_beta_alpha);

    // RWKV time mix
    add_tensor(layer.time_mix_w1);
    add_tensor(layer.time_mix_w2);
    add_tensor(layer.time_mix_lerp_x);
    add_tensor(layer.time_mix_lerp_w);
    add_tensor(layer.time_mix_lerp_k);
    add_tensor(layer.time_mix_lerp_v);
    add_tensor(layer.time_mix_lerp_r);
    add_tensor(layer.time_mix_lerp_g);
    add_tensor(layer.time_mix_lerp_fused);
    add_tensor(layer.time_mix_first);
    add_tensor(layer.time_mix_decay);
    add_tensor(layer.time_mix_decay_w1);
    add_tensor(layer.time_mix_decay_w2);
    add_tensor(layer.time_mix_key);
    add_tensor(layer.time_mix_key_b);
    add_tensor(layer.time_mix_value);
    add_tensor(layer.time_mix_value_b);
    add_tensor(layer.time_mix_receptance);
    add_tensor(layer.time_mix_receptance_b);
    add_tensor(layer.time_mix_gate);

    // RWKV7
    add_tensor(layer.time_mix_w0);
    add_tensor(layer.time_mix_a0);
    add_tensor(layer.time_mix_a1);
    add_tensor(layer.time_mix_a2);
    add_tensor(layer.time_mix_v0);
    add_tensor(layer.time_mix_v1);

    return tensors;
}

void llama_layer_scheduler::free_migration_buffer(int32_t il) {
    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return;
    }

    auto & state = layer_states_[il];
    if (state.migration_buffer != nullptr) {
        ggml_backend_buffer_free(state.migration_buffer);
        state.migration_buffer = nullptr;
    }
}

//
// New improvement implementations
//

bool llama_layer_scheduler::migrate_batch_to_gpu(const std::vector<int32_t> & layers) {
    if (layers.empty()) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: batch migrating %zu layers to GPU\n", __func__, layers.size());

    bool all_success = true;
    for (int32_t il : layers) {
        if (!migrate_to_gpu(il)) {
            all_success = false;
            // Continue with graceful degradation if enabled
            if (!cfg_.graceful_degradation) {
                break;
            }
        }
    }

    return all_success;
}

bool llama_layer_scheduler::migrate_batch_to_cpu(const std::vector<int32_t> & layers) {
    if (layers.empty()) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: batch migrating %zu layers to CPU\n", __func__, layers.size());

    bool all_success = true;
    for (int32_t il : layers) {
        if (!migrate_to_cpu(il)) {
            all_success = false;
            // Don't break on failure - try to evict remaining layers
        }
    }

    return all_success;
}

void * llama_layer_scheduler::alloc_pinned_memory(size_t size) {
    void * ptr = nullptr;

#ifdef _WIN32
    // Windows: Use VirtualAlloc with PAGE_READWRITE for pinned-like behavior
    // Note: For true CUDA pinned memory, would need cudaHostAlloc
    ptr = VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (ptr != nullptr) {
        // Lock pages in memory to prevent swapping
        VirtualLock(ptr, size);
    }
#else
    // POSIX: Use mmap with MAP_LOCKED for pinned memory
    ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr != MAP_FAILED) {
        // Try to lock the pages (may fail without sufficient privileges)
        if (mlock(ptr, size) != 0) {
            // mlock failed, but memory is still usable
            LLAMA_LOG_DEBUG("%s: mlock failed, using regular memory\n", __func__);
        }
    } else {
        ptr = nullptr;
    }
#endif

    if (ptr != nullptr) {
        pinned_buffers_.push_back({ptr, size});
    }

    return ptr;
}

void llama_layer_scheduler::free_pinned_memory(void * ptr) {
    if (ptr == nullptr) {
        return;
    }

    // Find size in our tracking list
    size_t size = 0;
    auto it = std::find_if(pinned_buffers_.begin(), pinned_buffers_.end(),
        [ptr](const std::pair<void *, size_t> & p) { return p.first == ptr; });

    if (it != pinned_buffers_.end()) {
        size = it->second;
        pinned_buffers_.erase(it);
    }

#ifdef _WIN32
    if (size > 0) {
        VirtualUnlock(ptr, size);
    }
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    if (size > 0) {
        munlock(ptr, size);
        munmap(ptr, size);
    }
#endif
}

bool llama_layer_scheduler::should_evict(float current_pressure) const {
    // Hysteresis logic: once in pressure mode, keep evicting until below low threshold
    if (in_pressure_mode_) {
        return current_pressure > cfg_.pressure_low_thresh;
    }
    // Not in pressure mode: start evicting only above high threshold
    return current_pressure > cfg_.pressure_threshold;
}

bool llama_layer_scheduler::should_prefetch(float current_pressure) const {
    // Only prefetch when not under pressure
    return current_pressure < cfg_.pressure_low_thresh;
}

void llama_layer_scheduler::update_watermarks() {
    // Update high-water marks
    if (gpu_memory_used_ > gpu_memory_watermark_) {
        gpu_memory_watermark_ = gpu_memory_used_;
    }
    if (cpu_memory_used_ > cpu_memory_watermark_) {
        cpu_memory_watermark_ = cpu_memory_used_;
    }
}

double llama_layer_scheduler::get_avg_migration_time_ms() const {
    if (migration_count_ == 0) {
        return 0.0;
    }
    return (total_migration_time_us_ / static_cast<double>(migration_count_)) / 1000.0;
}

bool llama_layer_scheduler::is_layer_pinned(int32_t il) const {
    return std::find(cfg_.pinned_layers.begin(), cfg_.pinned_layers.end(), il) != cfg_.pinned_layers.end();
}

bool llama_layer_scheduler::copy_tensor_to_buffer(ggml_tensor * tensor, ggml_backend_buffer_t dst_buffer, size_t & offset) {
    if (tensor == nullptr || dst_buffer == nullptr) {
        return false;
    }

    size_t tensor_size = ggml_nbytes(tensor);
    if (tensor_size == 0) {
        return true;  // Nothing to copy
    }

    // Get alignment from buffer
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(dst_buffer);
    size_t alignment = ggml_backend_buft_get_alignment(buft);

    // Align offset
    offset = (offset + alignment - 1) & ~(alignment - 1);

    // Get destination address
    void * dst_base = ggml_backend_buffer_get_base(dst_buffer);
    void * dst_addr = static_cast<char *>(dst_base) + offset;

    // Allocate temp buffer for transfer
    void * temp_data = nullptr;
    if (cfg_.use_pinned_memory) {
        temp_data = alloc_pinned_memory(tensor_size);
    }
    if (temp_data == nullptr) {
        temp_data = malloc(tensor_size);
    }
    if (temp_data == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to allocate temp buffer (%zu bytes)\n", __func__, tensor_size);
        return false;
    }

    // Read from source tensor
    ggml_backend_tensor_get(tensor, temp_data, 0, tensor_size);

    // Store original pointers for potential rollback
    void * orig_data = tensor->data;
    ggml_backend_buffer_t orig_buffer = tensor->buffer;

    // Update tensor to point to new buffer
    tensor->data = dst_addr;
    tensor->buffer = dst_buffer;

    // Write to destination
    ggml_backend_tensor_set(tensor, temp_data, 0, tensor_size);

    // Free temp buffer
    if (cfg_.use_pinned_memory) {
        free_pinned_memory(temp_data);
    } else {
        free(temp_data);
    }

    // Update offset for next tensor
    offset += tensor_size;

    LLAMA_LOG_DEBUG("%s: copied tensor %s (%zu bytes) to offset %zu\n",
        __func__, ggml_get_name(tensor), tensor_size, offset - tensor_size);

    return true;
}

//
// Backend synchronization helpers
//

void llama_layer_scheduler::synchronize_device(ggml_backend_dev_t dev) {
    if (dev == nullptr) {
        return;
    }

    // Create a temporary backend for synchronization
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend != nullptr) {
        ggml_backend_synchronize(backend);
        ggml_backend_free(backend);
    }
}

void llama_layer_scheduler::synchronize_all_devices() {
    std::set<ggml_backend_dev_t> devices_to_sync;

    // Collect all unique devices from layer states
    for (const auto & state : layer_states_) {
        if (state.device != nullptr) {
            devices_to_sync.insert(state.device);
        }
    }

    // Synchronize each device
    for (auto * dev : devices_to_sync) {
        synchronize_device(dev);
    }
}

bool llama_layer_scheduler::wait_for_migration(int32_t il, int64_t timeout_us) {
    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return false;
    }

    int64_t start_time = get_time_us();
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!layer_states_[il].is_migrating) {
                return true;  // Migration complete
            }
        }

        // Check timeout
        if (timeout_us > 0 && (get_time_us() - start_time) >= timeout_us) {
            LLAMA_LOG_WARN("%s: timeout waiting for layer %d migration\n", __func__, il);
            return false;
        }

        // Brief sleep to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

//
// Async migration support
//

bool llama_layer_scheduler::request_async_migration(int32_t il, llama_layer_location target) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (il < 0 || il >= static_cast<int32_t>(layer_states_.size())) {
        return false;
    }

    auto & state = layer_states_[il];

    // Already at target location
    if ((target == LLAMA_LAYER_LOC_GPU && state.location == LLAMA_LAYER_LOC_GPU) ||
        (target == LLAMA_LAYER_LOC_CPU && state.location == LLAMA_LAYER_LOC_CPU)) {
        return true;
    }

    // Already migrating
    if (state.is_migrating) {
        return false;
    }

    // Queue migration request
    pending_migrations_.push_back({il, target});

    return true;
}

size_t llama_layer_scheduler::process_pending_migrations(size_t max_count) {
    std::vector<std::pair<int32_t, llama_layer_location>> to_process;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t count = (std::min)(max_count, pending_migrations_.size());
        for (size_t i = 0; i < count; ++i) {
            to_process.push_back(pending_migrations_.front());
            pending_migrations_.pop_front();
        }
    }

    size_t processed = 0;
    for (const auto & [il, target] : to_process) {
        std::lock_guard<std::mutex> lock(mutex_);

        bool success = false;
        if (target == LLAMA_LAYER_LOC_GPU) {
            success = migrate_to_gpu(il);
        } else if (target == LLAMA_LAYER_LOC_CPU) {
            success = migrate_to_cpu(il);
        }

        if (success) {
            processed++;
        }
    }

    return processed;
}

size_t llama_layer_scheduler::get_pending_migration_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pending_migrations_.size();
}
