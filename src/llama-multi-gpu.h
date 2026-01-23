#pragma once

#include "ggml-backend.h"
#include "llama-mem-telemetry.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// Forward declarations
struct llama_model;
struct ggml_tensor;

// GPU device information
struct llama_gpu_device {
    ggml_backend_dev_t device = nullptr;
    int                device_id = -1;
    std::string        name;
    size_t             total_memory = 0;
    size_t             free_memory = 0;
    float              compute_capability = 0.0f;
    bool               supports_peer_access = false;

    // Runtime stats
    std::atomic<size_t> allocated_bytes{0};
    std::atomic<int64_t> operations_count{0};
    std::atomic<int64_t> transfer_bytes_in{0};
    std::atomic<int64_t> transfer_bytes_out{0};

    // Default constructor
    llama_gpu_device() = default;

    // Move constructor (atomics need special handling)
    llama_gpu_device(llama_gpu_device && other) noexcept
        : device(other.device)
        , device_id(other.device_id)
        , name(std::move(other.name))
        , total_memory(other.total_memory)
        , free_memory(other.free_memory)
        , compute_capability(other.compute_capability)
        , supports_peer_access(other.supports_peer_access)
        , allocated_bytes(other.allocated_bytes.load())
        , operations_count(other.operations_count.load())
        , transfer_bytes_in(other.transfer_bytes_in.load())
        , transfer_bytes_out(other.transfer_bytes_out.load()) {}

    // Move assignment
    llama_gpu_device & operator=(llama_gpu_device && other) noexcept {
        if (this != &other) {
            device = other.device;
            device_id = other.device_id;
            name = std::move(other.name);
            total_memory = other.total_memory;
            free_memory = other.free_memory;
            compute_capability = other.compute_capability;
            supports_peer_access = other.supports_peer_access;
            allocated_bytes.store(other.allocated_bytes.load());
            operations_count.store(other.operations_count.load());
            transfer_bytes_in.store(other.transfer_bytes_in.load());
            transfer_bytes_out.store(other.transfer_bytes_out.load());
        }
        return *this;
    }

    // Delete copy operations
    llama_gpu_device(const llama_gpu_device &) = delete;
    llama_gpu_device & operator=(const llama_gpu_device &) = delete;
};

// Peer-to-peer topology between GPUs
struct llama_gpu_topology {
    std::vector<std::vector<bool>> peer_access_matrix;
    std::vector<std::vector<float>> bandwidth_matrix;  // GB/s between devices
    bool has_nvlink = false;
    bool has_pcie_switch = false;
};

// Layer assignment to GPU
struct llama_layer_assignment {
    int32_t layer_id = -1;
    int     gpu_id = -1;
    size_t  memory_required = 0;
    bool    is_split = false;           // For tensor parallelism
    std::vector<int> split_gpu_ids;     // GPUs this layer is split across
};

// Multi-GPU distribution strategy
enum class llama_multi_gpu_strategy {
    ROUND_ROBIN,        // Distribute layers evenly
    MEMORY_BALANCED,    // Balance by memory usage
    COMPUTE_BALANCED,   // Balance by compute capability
    PIPELINE,           // Sequential pipeline
    TENSOR_PARALLEL,    // Split tensors across GPUs
    HYBRID,             // Combination of pipeline + tensor parallel
};

// Multi-GPU manager for distributed inference
class llama_multi_gpu_manager {
public:
    struct config {
        llama_multi_gpu_strategy strategy = llama_multi_gpu_strategy::MEMORY_BALANCED;
        int    tensor_parallel_size = 1;   // Number of GPUs for tensor parallelism
        int    pipeline_parallel_size = 1; // Number of GPUs for pipeline parallelism
        bool   enable_peer_access = true;  // Enable GPU-to-GPU direct access
        bool   auto_balance = true;        // Auto-rebalance on memory pressure
        float  load_balance_threshold = 0.1f; // Rebalance if imbalance > 10%
    };

    llama_multi_gpu_manager();
    explicit llama_multi_gpu_manager(const config & cfg);
    ~llama_multi_gpu_manager();

    // Initialize with available GPUs
    bool initialize(const std::vector<ggml_backend_dev_t> & devices);

    // Discover and configure GPUs
    int discover_gpus();

    // Get GPU count
    int get_gpu_count() const { return static_cast<int>(gpus_.size()); }

    // Get GPU info
    const llama_gpu_device * get_gpu(int gpu_id) const;
    const std::vector<llama_gpu_device> & get_all_gpus() const;

    // Get topology info
    const llama_gpu_topology & get_topology() const { return topology_; }

    // Distribute model layers across GPUs
    std::vector<llama_layer_assignment> distribute_layers(
        const llama_model & model,
        int n_layers,
        const std::vector<size_t> & layer_sizes);

    // Get assigned GPU for a layer
    int get_layer_gpu(int32_t layer_id) const;

    // Get GPUs for tensor-parallel layer
    std::vector<int> get_layer_gpus(int32_t layer_id) const;

    // Allocate tensor on appropriate GPU(s)
    ggml_backend_buffer_t allocate_tensor_buffer(
        int32_t layer_id,
        size_t size,
        ggml_backend_buffer_type_t preferred_buft = nullptr);

    // Transfer tensor between GPUs
    bool transfer_tensor(
        ggml_tensor * tensor,
        int src_gpu,
        int dst_gpu);

    // Synchronize all GPUs
    void synchronize_all();

    // Synchronize specific GPU
    void synchronize_gpu(int gpu_id);

    // Check if peer access is available between GPUs
    bool has_peer_access(int src_gpu, int dst_gpu) const;

    // Enable peer access between GPUs
    bool enable_peer_access(int src_gpu, int dst_gpu);

    // Get memory stats across all GPUs
    struct memory_stats {
        size_t total_memory = 0;
        size_t used_memory = 0;
        size_t free_memory = 0;
        std::vector<float> per_gpu_utilization;
    };
    memory_stats get_memory_stats() const;

    // Rebalance layers based on current memory pressure
    bool rebalance_layers();

    // Set telemetry for memory monitoring
    void set_telemetry(llama_mem_telemetry * telemetry) { telemetry_ = telemetry; }

    // Get configuration
    const config & get_config() const { return cfg_; }

private:
    // Detect GPU topology (NVLink, PCIe, etc.)
    void detect_topology();

    // Calculate optimal layer distribution
    std::vector<llama_layer_assignment> calculate_distribution(
        int n_layers,
        const std::vector<size_t> & layer_sizes);

    // Round-robin distribution
    std::vector<llama_layer_assignment> distribute_round_robin(
        int n_layers,
        const std::vector<size_t> & layer_sizes);

    // Memory-balanced distribution
    std::vector<llama_layer_assignment> distribute_memory_balanced(
        int n_layers,
        const std::vector<size_t> & layer_sizes);

    // Tensor-parallel distribution
    std::vector<llama_layer_assignment> distribute_tensor_parallel(
        int n_layers,
        const std::vector<size_t> & layer_sizes);

    config cfg_;
    mutable std::mutex mutex_;

    std::vector<llama_gpu_device> gpus_;
    llama_gpu_topology topology_;

    // Layer to GPU assignment map
    std::map<int32_t, llama_layer_assignment> layer_assignments_;

    // Telemetry for memory monitoring
    llama_mem_telemetry * telemetry_ = nullptr;

    bool initialized_ = false;
};

// Global multi-GPU manager instance (optional singleton)
llama_multi_gpu_manager * llama_get_multi_gpu_manager();
void llama_set_multi_gpu_manager(llama_multi_gpu_manager * manager);
