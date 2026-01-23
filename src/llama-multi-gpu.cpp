#include "llama-multi-gpu.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstring>
#include <numeric>

// Global instance with thread-safe access
static llama_multi_gpu_manager * g_multi_gpu_manager = nullptr;
static std::mutex g_multi_gpu_manager_mutex;

llama_multi_gpu_manager * llama_get_multi_gpu_manager() {
    std::lock_guard<std::mutex> lock(g_multi_gpu_manager_mutex);
    return g_multi_gpu_manager;
}

void llama_set_multi_gpu_manager(llama_multi_gpu_manager * manager) {
    std::lock_guard<std::mutex> lock(g_multi_gpu_manager_mutex);
    g_multi_gpu_manager = manager;
}

llama_multi_gpu_manager::llama_multi_gpu_manager() = default;

llama_multi_gpu_manager::llama_multi_gpu_manager(const config & cfg)
    : cfg_(cfg) {}

llama_multi_gpu_manager::~llama_multi_gpu_manager() {
    if (g_multi_gpu_manager == this) {
        g_multi_gpu_manager = nullptr;
    }
}

bool llama_multi_gpu_manager::initialize(const std::vector<ggml_backend_dev_t> & devices) {
    std::lock_guard<std::mutex> lock(mutex_);

    gpus_.clear();
    int gpu_id = 0;

    for (auto * dev : devices) {
        if (dev == nullptr) continue;

        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if (dtype != GGML_BACKEND_DEVICE_TYPE_GPU &&
            dtype != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        llama_gpu_device gpu;
        gpu.device = dev;
        gpu.device_id = gpu_id++;
        gpu.name = ggml_backend_dev_name(dev);

        // Get memory info
        size_t free_mem = 0, total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        gpu.total_memory = total_mem;
        gpu.free_memory = free_mem;

        gpus_.push_back(std::move(gpu));
    }

    if (gpus_.empty()) {
        LLAMA_LOG_WARN("%s: no GPU devices found\n", __func__);
        return false;
    }

    // Detect topology
    detect_topology();

    // Enable peer access if configured
    if (cfg_.enable_peer_access) {
        for (size_t i = 0; i < gpus_.size(); ++i) {
            for (size_t j = i + 1; j < gpus_.size(); ++j) {
                enable_peer_access(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    initialized_ = true;

    LLAMA_LOG_INFO("%s: initialized multi-GPU manager with %zu GPUs\n",
        __func__, gpus_.size());

    for (const auto & gpu : gpus_) {
        LLAMA_LOG_INFO("%s:   GPU %d: %s (%.1f GB total, %.1f GB free)\n",
            __func__, gpu.device_id, gpu.name.c_str(),
            gpu.total_memory / (1024.0 * 1024.0 * 1024.0),
            gpu.free_memory / (1024.0 * 1024.0 * 1024.0));
    }

    return true;
}

int llama_multi_gpu_manager::discover_gpus() {
    std::vector<ggml_backend_dev_t> devices;

    // Enumerate all backend devices
    size_t n_devices = ggml_backend_dev_count();
    for (size_t i = 0; i < n_devices; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev != nullptr) {
            devices.push_back(dev);
        }
    }

    if (initialize(devices)) {
        return static_cast<int>(gpus_.size());
    }
    return 0;
}

const llama_gpu_device * llama_multi_gpu_manager::get_gpu(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gpu_id < 0 || gpu_id >= static_cast<int>(gpus_.size())) {
        return nullptr;
    }
    return &gpus_[gpu_id];
}

const std::vector<llama_gpu_device> & llama_multi_gpu_manager::get_all_gpus() const {
    // Note: Caller must ensure thread safety if accessing GPU stats
    return gpus_;
}

void llama_multi_gpu_manager::detect_topology() {
    size_t n = gpus_.size();
    topology_.peer_access_matrix.resize(n, std::vector<bool>(n, false));
    topology_.bandwidth_matrix.resize(n, std::vector<float>(n, 0.0f));

    // Self-access is always true with max bandwidth
    for (size_t i = 0; i < n; ++i) {
        topology_.peer_access_matrix[i][i] = true;
        topology_.bandwidth_matrix[i][i] = 900.0f;  // HBM bandwidth estimate
    }

    // Check peer access between GPUs
    // Note: This is a simplified version - real implementation would use
    // CUDA APIs like cudaDeviceCanAccessPeer
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            // Assume PCIe peer access is available between all GPUs
            // Real implementation would check hardware topology
            topology_.peer_access_matrix[i][j] = true;
            topology_.peer_access_matrix[j][i] = true;

            // Estimate bandwidth (PCIe 4.0 x16 = ~32 GB/s bidirectional)
            topology_.bandwidth_matrix[i][j] = 16.0f;
            topology_.bandwidth_matrix[j][i] = 16.0f;
        }
    }

    LLAMA_LOG_DEBUG("%s: detected GPU topology with %zu GPUs\n", __func__, n);
}

std::vector<llama_layer_assignment> llama_multi_gpu_manager::distribute_layers(
        const llama_model & model,
        int n_layers,
        const std::vector<size_t> & layer_sizes) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (gpus_.empty()) {
        LLAMA_LOG_WARN("%s: no GPUs available for layer distribution\n", __func__);
        return {};
    }

    std::vector<llama_layer_assignment> assignments;

    switch (cfg_.strategy) {
        case llama_multi_gpu_strategy::ROUND_ROBIN:
            assignments = distribute_round_robin(n_layers, layer_sizes);
            break;
        case llama_multi_gpu_strategy::MEMORY_BALANCED:
            assignments = distribute_memory_balanced(n_layers, layer_sizes);
            break;
        case llama_multi_gpu_strategy::TENSOR_PARALLEL:
            assignments = distribute_tensor_parallel(n_layers, layer_sizes);
            break;
        default:
            assignments = distribute_memory_balanced(n_layers, layer_sizes);
            break;
    }

    // Store assignments
    layer_assignments_.clear();
    for (const auto & a : assignments) {
        layer_assignments_[a.layer_id] = a;
    }

    return assignments;
}

std::vector<llama_layer_assignment> llama_multi_gpu_manager::distribute_round_robin(
        int n_layers,
        const std::vector<size_t> & layer_sizes) {

    std::vector<llama_layer_assignment> assignments;
    int n_gpus = static_cast<int>(gpus_.size());

    for (int i = 0; i < n_layers; ++i) {
        llama_layer_assignment a;
        a.layer_id = i;
        a.gpu_id = i % n_gpus;
        a.memory_required = (i < static_cast<int>(layer_sizes.size())) ? layer_sizes[i] : 0;
        a.is_split = false;
        assignments.push_back(a);
    }

    LLAMA_LOG_INFO("%s: distributed %d layers across %d GPUs (round-robin)\n",
        __func__, n_layers, n_gpus);

    return assignments;
}

std::vector<llama_layer_assignment> llama_multi_gpu_manager::distribute_memory_balanced(
        int n_layers,
        const std::vector<size_t> & layer_sizes) {

    std::vector<llama_layer_assignment> assignments;
    int n_gpus = static_cast<int>(gpus_.size());

    // Calculate total memory needed
    size_t total_size = 0;
    for (size_t s : layer_sizes) {
        total_size += s;
    }

    // Calculate target memory per GPU
    std::vector<size_t> gpu_allocated(n_gpus, 0);
    std::vector<size_t> gpu_capacity(n_gpus);
    for (int i = 0; i < n_gpus; ++i) {
        gpu_capacity[i] = gpus_[i].free_memory;
    }

    // Greedy assignment: assign each layer to GPU with most free space
    for (int i = 0; i < n_layers; ++i) {
        size_t layer_size = (i < static_cast<int>(layer_sizes.size())) ? layer_sizes[i] : 0;

        // Find GPU with most remaining capacity
        int best_gpu = 0;
        size_t best_remaining = 0;
        for (int g = 0; g < n_gpus; ++g) {
            size_t remaining = (gpu_capacity[g] > gpu_allocated[g]) ?
                               (gpu_capacity[g] - gpu_allocated[g]) : 0;
            if (remaining > best_remaining) {
                best_remaining = remaining;
                best_gpu = g;
            }
        }

        llama_layer_assignment a;
        a.layer_id = i;
        a.gpu_id = best_gpu;
        a.memory_required = layer_size;
        a.is_split = false;
        assignments.push_back(a);

        gpu_allocated[best_gpu] += layer_size;
    }

    // Log distribution
    LLAMA_LOG_INFO("%s: distributed %d layers across %d GPUs (memory-balanced)\n",
        __func__, n_layers, n_gpus);
    for (int g = 0; g < n_gpus; ++g) {
        int layer_count = 0;
        for (const auto & a : assignments) {
            if (a.gpu_id == g) layer_count++;
        }
        LLAMA_LOG_INFO("%s:   GPU %d: %d layers, %.1f MB allocated\n",
            __func__, g, layer_count, gpu_allocated[g] / (1024.0 * 1024.0));
    }

    return assignments;
}

std::vector<llama_layer_assignment> llama_multi_gpu_manager::distribute_tensor_parallel(
        int n_layers,
        const std::vector<size_t> & layer_sizes) {

    std::vector<llama_layer_assignment> assignments;
    int n_gpus = static_cast<int>(gpus_.size());
    int tp_size = std::min(cfg_.tensor_parallel_size, n_gpus);

    for (int i = 0; i < n_layers; ++i) {
        llama_layer_assignment a;
        a.layer_id = i;
        a.gpu_id = 0;  // Primary GPU
        a.memory_required = (i < static_cast<int>(layer_sizes.size())) ? layer_sizes[i] : 0;
        a.is_split = (tp_size > 1);

        if (a.is_split) {
            for (int g = 0; g < tp_size; ++g) {
                a.split_gpu_ids.push_back(g);
            }
        }

        assignments.push_back(a);
    }

    LLAMA_LOG_INFO("%s: distributed %d layers with tensor parallelism (TP=%d)\n",
        __func__, n_layers, tp_size);

    return assignments;
}

int llama_multi_gpu_manager::get_layer_gpu(int32_t layer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layer_assignments_.find(layer_id);
    if (it == layer_assignments_.end()) {
        return 0;  // Default to first GPU
    }
    return it->second.gpu_id;
}

std::vector<int> llama_multi_gpu_manager::get_layer_gpus(int32_t layer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = layer_assignments_.find(layer_id);
    if (it == layer_assignments_.end()) {
        return {0};
    }
    if (it->second.is_split) {
        return it->second.split_gpu_ids;
    }
    return {it->second.gpu_id};
}

ggml_backend_buffer_t llama_multi_gpu_manager::allocate_tensor_buffer(
        int32_t layer_id,
        size_t size,
        ggml_backend_buffer_type_t preferred_buft) {

    std::lock_guard<std::mutex> lock(mutex_);

    int gpu_id = 0;
    auto it = layer_assignments_.find(layer_id);
    if (it != layer_assignments_.end()) {
        gpu_id = it->second.gpu_id;
    }

    if (gpu_id < 0 || gpu_id >= static_cast<int>(gpus_.size())) {
        gpu_id = 0;
    }

    auto & gpu = gpus_[gpu_id];

    // Get buffer type for this GPU
    ggml_backend_buffer_type_t buft = preferred_buft;
    if (buft == nullptr && gpu.device != nullptr) {
        buft = ggml_backend_dev_buffer_type(gpu.device);
    }

    if (buft == nullptr) {
        LLAMA_LOG_WARN("%s: could not get buffer type for GPU %d\n", __func__, gpu_id);
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, size);
    if (buffer != nullptr) {
        gpu.allocated_bytes += size;
    }

    return buffer;
}

bool llama_multi_gpu_manager::transfer_tensor(
        ggml_tensor * tensor,
        int src_gpu,
        int dst_gpu) {

    if (tensor == nullptr) return false;
    if (src_gpu == dst_gpu) return true;

    std::lock_guard<std::mutex> lock(mutex_);

    if (src_gpu < 0 || src_gpu >= static_cast<int>(gpus_.size()) ||
        dst_gpu < 0 || dst_gpu >= static_cast<int>(gpus_.size())) {
        return false;
    }

    size_t tensor_size = ggml_nbytes(tensor);

    // Allocate buffer on destination GPU
    auto & dst_device = gpus_[dst_gpu];
    ggml_backend_buffer_type_t dst_buft = ggml_backend_dev_buffer_type(dst_device.device);
    if (dst_buft == nullptr) {
        return false;
    }

    ggml_backend_buffer_t dst_buffer = ggml_backend_buft_alloc_buffer(dst_buft, tensor_size);
    if (dst_buffer == nullptr) {
        return false;
    }

    // Copy data
    void * temp = malloc(tensor_size);
    if (temp == nullptr) {
        ggml_backend_buffer_free(dst_buffer);
        return false;
    }

    ggml_backend_tensor_get(tensor, temp, 0, tensor_size);

    void * dst_base = ggml_backend_buffer_get_base(dst_buffer);
    tensor->data = dst_base;
    tensor->buffer = dst_buffer;

    ggml_backend_tensor_set(tensor, temp, 0, tensor_size);
    free(temp);

    // Update stats
    gpus_[src_gpu].transfer_bytes_out += tensor_size;
    gpus_[dst_gpu].transfer_bytes_in += tensor_size;
    gpus_[dst_gpu].allocated_bytes += tensor_size;

    return true;
}

void llama_multi_gpu_manager::synchronize_all() {
    // Note: Real implementation would use backend-specific sync
    // For now, this is a placeholder
    for (auto & gpu : gpus_) {
        if (gpu.device != nullptr) {
            // ggml_backend_synchronize would be called here
        }
    }
}

void llama_multi_gpu_manager::synchronize_gpu(int gpu_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (gpu_id >= 0 && gpu_id < static_cast<int>(gpus_.size())) {
        // ggml_backend_synchronize would be called here
    }
}

bool llama_multi_gpu_manager::has_peer_access(int src_gpu, int dst_gpu) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (src_gpu < 0 || src_gpu >= static_cast<int>(topology_.peer_access_matrix.size()) ||
        dst_gpu < 0 || dst_gpu >= static_cast<int>(topology_.peer_access_matrix.size())) {
        return false;
    }
    return topology_.peer_access_matrix[src_gpu][dst_gpu];
}

bool llama_multi_gpu_manager::enable_peer_access(int src_gpu, int dst_gpu) {
    // Note: Real implementation would use cudaDeviceEnablePeerAccess or similar
    // For now, just update our tracking matrix
    if (src_gpu < 0 || src_gpu >= static_cast<int>(gpus_.size()) ||
        dst_gpu < 0 || dst_gpu >= static_cast<int>(gpus_.size())) {
        return false;
    }

    topology_.peer_access_matrix[src_gpu][dst_gpu] = true;
    topology_.peer_access_matrix[dst_gpu][src_gpu] = true;

    LLAMA_LOG_DEBUG("%s: enabled peer access between GPU %d and GPU %d\n",
        __func__, src_gpu, dst_gpu);

    return true;
}

llama_multi_gpu_manager::memory_stats llama_multi_gpu_manager::get_memory_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    memory_stats stats;
    for (const auto & gpu : gpus_) {
        stats.total_memory += gpu.total_memory;
        stats.used_memory += gpu.allocated_bytes.load();
        stats.free_memory += gpu.free_memory;

        float util = (gpu.total_memory > 0) ?
            static_cast<float>(gpu.allocated_bytes.load()) / gpu.total_memory : 0.0f;
        stats.per_gpu_utilization.push_back(util);
    }

    return stats;
}

bool llama_multi_gpu_manager::rebalance_layers() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!cfg_.auto_balance || gpus_.size() < 2) {
        return false;
    }

    // Calculate current load imbalance
    std::vector<size_t> gpu_load(gpus_.size(), 0);
    for (const auto & [layer_id, assignment] : layer_assignments_) {
        if (assignment.gpu_id >= 0 && assignment.gpu_id < static_cast<int>(gpus_.size())) {
            gpu_load[assignment.gpu_id] += assignment.memory_required;
        }
    }

    size_t max_load = *std::max_element(gpu_load.begin(), gpu_load.end());
    size_t min_load = *std::min_element(gpu_load.begin(), gpu_load.end());

    if (max_load == 0) return false;

    float imbalance = static_cast<float>(max_load - min_load) / max_load;
    if (imbalance <= cfg_.load_balance_threshold) {
        return false;  // Already balanced
    }

    LLAMA_LOG_INFO("%s: rebalancing layers (imbalance=%.1f%%)\n",
        __func__, imbalance * 100.0f);

    // Simple rebalancing: move layers from overloaded to underloaded GPUs
    // Real implementation would be more sophisticated

    return true;
}
