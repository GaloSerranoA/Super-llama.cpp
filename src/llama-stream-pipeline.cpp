#include "llama-stream-pipeline.h"
#include "llama-impl.h"

#include <algorithm>
#include <chrono>
#include <cstring>

// Global instance with thread-safe access
static llama_stream_pipeline * g_stream_pipeline = nullptr;
static std::mutex g_stream_pipeline_mutex;

llama_stream_pipeline * llama_get_stream_pipeline() {
    std::lock_guard<std::mutex> lock(g_stream_pipeline_mutex);
    return g_stream_pipeline;
}

void llama_set_stream_pipeline(llama_stream_pipeline * pipeline) {
    std::lock_guard<std::mutex> lock(g_stream_pipeline_mutex);
    g_stream_pipeline = pipeline;
}

//
// llama_stream implementation
//

llama_stream::llama_stream(int device_id, int stream_id)
    : device_id_(device_id)
    , stream_id_(stream_id) {
    worker_ = std::thread(&llama_stream::worker_thread, this);
}

llama_stream::~llama_stream() {
    running_ = false;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void llama_stream::enqueue_transfer_h2d(void * dst, const void * src, size_t size) {
    llama_stream_op op;
    op.type = llama_stream_op_type::TRANSFER_H2D;
    op.stream_id = stream_id_;
    op.dst_ptr = dst;
    op.src_ptr = const_cast<void *>(src);
    op.size = size;
    op.enqueue_time_us = get_time_us();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        op_queue_.push_back(std::move(op));
    }
    cv_.notify_one();
}

void llama_stream::enqueue_transfer_d2h(void * dst, const void * src, size_t size) {
    llama_stream_op op;
    op.type = llama_stream_op_type::TRANSFER_D2H;
    op.stream_id = stream_id_;
    op.dst_ptr = dst;
    op.src_ptr = const_cast<void *>(src);
    op.size = size;
    op.enqueue_time_us = get_time_us();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        op_queue_.push_back(std::move(op));
    }
    cv_.notify_one();
}

void llama_stream::enqueue_transfer_d2d(void * dst, const void * src, size_t size, int dst_device) {
    llama_stream_op op;
    op.type = llama_stream_op_type::TRANSFER_D2D;
    op.stream_id = stream_id_;
    op.dst_ptr = dst;
    op.src_ptr = const_cast<void *>(src);
    op.size = size;
    op.src_device = device_id_;
    op.dst_device = dst_device;
    op.enqueue_time_us = get_time_us();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        op_queue_.push_back(std::move(op));
    }
    cv_.notify_one();
}

void llama_stream::enqueue_compute(ggml_tensor * tensor) {
    llama_stream_op op;
    op.type = llama_stream_op_type::COMPUTE;
    op.stream_id = stream_id_;
    op.tensor = tensor;
    op.enqueue_time_us = get_time_us();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        op_queue_.push_back(std::move(op));
    }
    cv_.notify_one();
}

void llama_stream::enqueue_callback(std::function<void()> callback) {
    llama_stream_op op;
    op.type = llama_stream_op_type::CALLBACK;
    op.stream_id = stream_id_;
    op.callback = std::move(callback);
    op.enqueue_time_us = get_time_us();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        op_queue_.push_back(std::move(op));
    }
    cv_.notify_one();
}

void llama_stream::synchronize() {
    // Wait for all operations to complete
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (op_queue_.empty() && idle_) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

bool llama_stream::is_complete() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return op_queue_.empty() && idle_;
}

size_t llama_stream::get_pending_ops() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return op_queue_.size();
}

llama_stream_stats llama_stream::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void llama_stream::worker_thread() {
    while (running_) {
        llama_stream_op op;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this] {
                return !op_queue_.empty() || !running_;
            });

            if (!running_ && op_queue_.empty()) {
                break;
            }

            if (op_queue_.empty()) {
                continue;
            }

            op = std::move(op_queue_.front());
            op_queue_.pop_front();
            idle_ = false;
        }

        process_op(op);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            idle_ = true;
        }
    }
}

void llama_stream::process_op(llama_stream_op & op) {
    op.start_time_us = get_time_us();

    switch (op.type) {
        case llama_stream_op_type::TRANSFER_H2D:
        case llama_stream_op_type::TRANSFER_D2H:
        case llama_stream_op_type::TRANSFER_D2D:
            // Perform memory copy
            if (op.dst_ptr != nullptr && op.src_ptr != nullptr && op.size > 0) {
                std::memcpy(op.dst_ptr, op.src_ptr, op.size);
                stats_.transfer_ops++;
                stats_.total_bytes_transferred += op.size;
            }
            break;

        case llama_stream_op_type::COMPUTE:
            // Compute operations would be handled by ggml backend
            stats_.compute_ops++;
            break;

        case llama_stream_op_type::CALLBACK:
            if (op.callback) {
                op.callback();
            }
            break;

        case llama_stream_op_type::SYNCHRONIZE:
            // No-op for synchronize
            break;
    }

    op.end_time_us = get_time_us();
    stats_.total_ops++;

    // Update timing stats
    double op_time_ms = (op.end_time_us - op.start_time_us) / 1000.0;
    if (op.type == llama_stream_op_type::COMPUTE) {
        stats_.avg_compute_time_ms =
            (stats_.avg_compute_time_ms * (stats_.compute_ops - 1) + op_time_ms) / stats_.compute_ops;
    } else if (op.type == llama_stream_op_type::TRANSFER_H2D ||
               op.type == llama_stream_op_type::TRANSFER_D2H ||
               op.type == llama_stream_op_type::TRANSFER_D2D) {
        stats_.avg_transfer_time_ms =
            (stats_.avg_transfer_time_ms * (stats_.transfer_ops - 1) + op_time_ms) / stats_.transfer_ops;
    }
}

int64_t llama_stream::get_time_us() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

//
// llama_stream_pipeline implementation
//

llama_stream_pipeline::llama_stream_pipeline() = default;

llama_stream_pipeline::llama_stream_pipeline(const config & cfg)
    : cfg_(cfg) {}

llama_stream_pipeline::~llama_stream_pipeline() {
    if (g_stream_pipeline == this) {
        g_stream_pipeline = nullptr;
    }
}

bool llama_stream_pipeline::initialize(int device_id) {
    device_id_ = device_id;

    // Create compute streams
    for (int i = 0; i < cfg_.num_compute_streams; ++i) {
        compute_streams_.push_back(
            std::make_unique<llama_stream>(device_id, i));
    }

    // Create transfer streams
    for (int i = 0; i < cfg_.num_transfer_streams; ++i) {
        transfer_streams_.push_back(
            std::make_unique<llama_stream>(device_id, 100 + i));  // Offset stream IDs
    }

    LLAMA_LOG_INFO("%s: initialized stream pipeline on device %d "
                   "(compute=%d, transfer=%d, prefetch_depth=%d)\n",
        __func__, device_id,
        cfg_.num_compute_streams, cfg_.num_transfer_streams, cfg_.prefetch_depth);

    return true;
}

llama_stream * llama_stream_pipeline::get_compute_stream(int index) {
    if (index < 0 || index >= static_cast<int>(compute_streams_.size())) {
        index = 0;
    }
    return compute_streams_[index].get();
}

llama_stream * llama_stream_pipeline::get_transfer_stream(int index) {
    if (index < 0 || index >= static_cast<int>(transfer_streams_.size())) {
        index = 0;
    }
    return transfer_streams_[index].get();
}

void llama_stream_pipeline::begin_layer(int layer_id) {
    std::lock_guard<std::mutex> lock(layer_mutex_);
    layer_complete_[layer_id] = false;
}

void llama_stream_pipeline::end_layer(int layer_id) {
    {
        std::lock_guard<std::mutex> lock(layer_mutex_);
        layer_complete_[layer_id] = true;
    }
    layers_processed_++;
}

void llama_stream_pipeline::prefetch_layer(int layer_id, const std::vector<ggml_tensor *> & tensors) {
    if (!cfg_.enable_overlap || tensors.empty()) {
        return;
    }

    // Round-robin across transfer streams
    int stream_idx = current_transfer_stream_.fetch_add(1) % cfg_.num_transfer_streams;
    auto * stream = get_transfer_stream(stream_idx);

    // Enqueue prefetch operations
    for (auto * tensor : tensors) {
        if (tensor != nullptr && tensor->data != nullptr) {
            // This would trigger actual prefetch in real implementation
            stream->enqueue_callback([layer_id, tensor]() {
                LLAMA_LOG_DEBUG("prefetched tensor %s for layer %d\n",
                    ggml_get_name(tensor), layer_id);
            });
        }
    }
}

void llama_stream_pipeline::synchronize_all() {
    for (auto & stream : compute_streams_) {
        stream->synchronize();
    }
    for (auto & stream : transfer_streams_) {
        stream->synchronize();
    }
}

void llama_stream_pipeline::wait_for_layer(int layer_id) {
    while (!is_layer_ready(layer_id)) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

bool llama_stream_pipeline::is_layer_ready(int layer_id) const {
    std::lock_guard<std::mutex> lock(layer_mutex_);
    auto it = layer_complete_.find(layer_id);
    if (it == layer_complete_.end()) {
        return false;
    }
    return it->second;
}

llama_stream_pipeline::pipeline_stats llama_stream_pipeline::get_stats() const {
    pipeline_stats stats;

    // Aggregate compute stream stats
    for (const auto & stream : compute_streams_) {
        auto s = stream->get_stats();
        stats.compute_stats.total_ops += s.total_ops;
        stats.compute_stats.compute_ops += s.compute_ops;
        stats.compute_stats.avg_compute_time_ms += s.avg_compute_time_ms;
    }
    if (!compute_streams_.empty()) {
        stats.compute_stats.avg_compute_time_ms /= compute_streams_.size();
    }

    // Aggregate transfer stream stats
    for (const auto & stream : transfer_streams_) {
        auto s = stream->get_stats();
        stats.transfer_stats.total_ops += s.total_ops;
        stats.transfer_stats.transfer_ops += s.transfer_ops;
        stats.transfer_stats.total_bytes_transferred += s.total_bytes_transferred;
        stats.transfer_stats.avg_transfer_time_ms += s.avg_transfer_time_ms;
    }
    if (!transfer_streams_.empty()) {
        stats.transfer_stats.avg_transfer_time_ms /= transfer_streams_.size();
    }

    stats.total_layers_processed = layers_processed_.load();

    // Calculate overlap ratio (simplified)
    if (stats.compute_stats.avg_compute_time_ms > 0 &&
        stats.transfer_stats.avg_transfer_time_ms > 0) {
        double total_serial = stats.compute_stats.avg_compute_time_ms +
                              stats.transfer_stats.avg_transfer_time_ms;
        double actual = std::max(stats.compute_stats.avg_compute_time_ms,
                                  stats.transfer_stats.avg_transfer_time_ms);
        stats.overall_overlap_ratio = 1.0 - (actual / total_serial);
    }

    return stats;
}
