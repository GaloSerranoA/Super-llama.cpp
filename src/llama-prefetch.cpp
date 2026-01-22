#include "llama-prefetch.h"
#include "llama-layer-sched.h"
#include "llama-kv-cache-paged.h"
#include "llama-impl.h"

#include <algorithm>

llama_prefetcher::llama_prefetcher(
    llama_layer_scheduler * layer_sched,
    llama_kv_cache_paged  * kv_cache,
    const config          & cfg)
    : layer_sched_(layer_sched)
    , kv_cache_(kv_cache)
    , cfg_(cfg) {

    // Start worker threads
    running_ = true;
    for (int32_t i = 0; i < cfg_.n_workers; ++i) {
        workers_.emplace_back(&llama_prefetcher::worker_thread, this);
    }

    LLAMA_LOG_INFO("%s: prefetcher initialized with %d workers\n",
        __func__, cfg_.n_workers);
}

llama_prefetcher::~llama_prefetcher() {
    // Signal workers to stop
    running_ = false;
    cv_.notify_all();

    // Wait for workers to finish
    for (auto & worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

std::future<bool> llama_prefetcher::prefetch_layer(int32_t il, int32_t priority) {
    std::lock_guard<std::mutex> lock(mutex_);

    llama_prefetch_request req;
    req.type = LLAMA_PREFETCH_LAYER;
    req.layer_id = il;
    req.priority = priority;

    auto future = req.promise.get_future();

    if (!enabled_ || !running_) {
        req.promise.set_value(false);
        return future;
    }

    if (request_queue_.size() >= static_cast<size_t>(cfg_.queue_size)) {
        // Queue full, complete immediately with failure
        req.promise.set_value(false);
        return future;
    }

    insert_request_sorted(std::move(req));
    cv_.notify_one();

    return future;
}

std::future<bool> llama_prefetcher::prefetch_kv_page(uint32_t page_id, int32_t priority) {
    std::lock_guard<std::mutex> lock(mutex_);

    llama_prefetch_request req;
    req.type = LLAMA_PREFETCH_KV_PAGE;
    req.page_id = page_id;
    req.priority = priority;

    auto future = req.promise.get_future();

    if (!enabled_ || !running_) {
        req.promise.set_value(false);
        return future;
    }

    if (request_queue_.size() >= static_cast<size_t>(cfg_.queue_size)) {
        req.promise.set_value(false);
        return future;
    }

    insert_request_sorted(std::move(req));
    cv_.notify_one();

    return future;
}

void llama_prefetcher::on_layer_start(int32_t il) {
    if (!enabled_ || !cfg_.auto_prefetch) {
        return;
    }

    // Prefetch upcoming layers
    for (int32_t i = 1; i <= cfg_.lookahead_layers; ++i) {
        int32_t target_layer = il + i;

        // Avoid duplicate prefetches
        if (target_layer <= last_prefetched_layer_) {
            continue;
        }

        // Queue with decreasing priority for farther layers
        prefetch_layer(target_layer, cfg_.lookahead_layers - i);
        last_prefetched_layer_ = target_layer;
    }
}

void llama_prefetcher::on_kv_access(uint32_t page_id, int32_t il) {
    if (!enabled_ || !cfg_.auto_prefetch || !kv_cache_) {
        return;
    }

    // Prefetch upcoming pages (assume sequential access pattern)
    for (int32_t i = 1; i <= cfg_.lookahead_pages; ++i) {
        uint32_t target_page = page_id + i;

        // Avoid duplicate prefetches
        if (target_page <= last_prefetched_page_) {
            continue;
        }

        prefetch_kv_page(target_page, cfg_.lookahead_pages - i);
        last_prefetched_page_ = target_page;
    }
}

void llama_prefetcher::sync() {
    // Wait for queue to empty
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (request_queue_.empty()) {
                break;
            }
        }
        std::this_thread::yield();
    }
}

void llama_prefetcher::cancel_all() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Drain the queue, setting all promises to false
    while (!request_queue_.empty()) {
        auto req = std::move(request_queue_.front());
        request_queue_.pop_front();
        req.promise.set_value(false);
    }

    // Reset tracking
    last_prefetched_layer_ = -1;
    last_prefetched_page_ = 0;
}

size_t llama_prefetcher::get_pending_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return request_queue_.size();
}

int64_t llama_prefetcher::get_completed_count() const {
    return completed_count_.load();
}

int64_t llama_prefetcher::get_failed_count() const {
    return failed_count_.load();
}

void llama_prefetcher::set_enabled(bool enabled) {
    enabled_ = enabled;
    if (!enabled) {
        cancel_all();
    }
}

bool llama_prefetcher::is_enabled() const {
    return enabled_;
}

void llama_prefetcher::set_config(const config & cfg) {
    std::lock_guard<std::mutex> lock(mutex_);
    cfg_ = cfg;
}

const llama_prefetcher::config & llama_prefetcher::get_config() const {
    return cfg_;
}

void llama_prefetcher::worker_thread() {
    while (running_) {
        llama_prefetch_request req;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait for work or shutdown
            cv_.wait(lock, [this] {
                return !running_ || !request_queue_.empty();
            });

            if (!running_ && request_queue_.empty()) {
                break;
            }

            if (request_queue_.empty()) {
                continue;
            }

            // Get highest priority request (front of deque has highest priority)
            req = std::move(request_queue_.front());
            request_queue_.pop_front();
        }

        // Process the request outside the lock
        bool success = process_request(req);
        req.promise.set_value(success);

        if (success) {
            completed_count_++;
        } else {
            failed_count_++;
        }
    }
}

void llama_prefetcher::insert_request_sorted(llama_prefetch_request req) {
    // Note: mutex should already be held by caller
    // Insert in sorted order - higher priority at front
    // For simplicity, use linear search (queue is typically small)
    auto it = request_queue_.begin();
    while (it != request_queue_.end() && it->priority >= req.priority) {
        ++it;
    }
    request_queue_.insert(it, std::move(req));
}

bool llama_prefetcher::process_request(llama_prefetch_request & req) {
    switch (req.type) {
        case LLAMA_PREFETCH_LAYER: {
            if (!layer_sched_) {
                return false;
            }

            // Hint to layer scheduler to prepare the layer
            layer_sched_->hint_prefetch(req.layer_id);

            // For now, the actual prefetch is handled by prepare_layer()
            // This is a placeholder for future async implementation
            return true;
        }

        case LLAMA_PREFETCH_KV_PAGE: {
            if (!kv_cache_) {
                return false;
            }

            // Prefetch the KV page to GPU
            return kv_cache_->prefetch_page(req.page_id);
        }

        default:
            return false;
    }
}
