#include "llama-layer-sched.h"
#include "llama-model.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>

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

    while (freed < required_bytes && !gpu_lru_queue_.empty()) {
        // Find LRU layer on the specified device
        int32_t lru_layer = -1;
        int64_t oldest_access = INT64_MAX;

        for (int32_t il : gpu_lru_queue_) {
            const auto & state = layer_states_[il];
            if (state.location == LLAMA_LAYER_LOC_GPU &&
                state.device == dev &&
                !state.is_migrating) {

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
        if (get_gpu_layer_count() <= cfg_.min_gpu_layers) {
            break;
        }

        size_t layer_size = layer_states_[lru_layer].size_bytes;
        if (migrate_to_cpu(lru_layer)) {
            freed += layer_size;
        } else {
            // Migration failed, stop trying
            break;
        }
    }

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

    // For now, dynamic migration is a placeholder
    // Full implementation would:
    // 1. Allocate GPU buffer for layer tensors
    // 2. Copy tensors using ggml_backend_tensor_copy()
    // 3. Update tensor buffer pointers
    // 4. Free CPU buffers

    // This is a complex operation that requires careful integration
    // with the existing buffer management system

    // For the initial implementation, we'll just track the state
    // and let the existing layer device assignment handle it

    LLAMA_LOG_DEBUG("%s: layer %d migration to GPU requested (not implemented)\n", __func__, il);

    state.is_migrating = false;
    migrations_to_gpu_++;

    // Return false since actual migration is not implemented yet
    // The layer will continue to run on CPU
    return false;
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

    // Similar to migrate_to_gpu, full implementation would:
    // 1. Allocate CPU buffer for layer tensors
    // 2. Copy tensors using ggml_backend_tensor_copy()
    // 3. Update tensor buffer pointers
    // 4. Free GPU buffers

    LLAMA_LOG_DEBUG("%s: layer %d migration to CPU requested (not implemented)\n", __func__, il);

    // Update state tracking even though actual migration is not implemented
    // This helps with memory pressure calculations

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
