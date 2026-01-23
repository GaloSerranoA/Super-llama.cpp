// test-load.cpp - Load/stress testing for Super-llama.cpp Enterprise
// Author: GALO SERRANO ABAD
//
// This test simulates production load with:
// - Multiple concurrent clients
// - Variable request sizes
// - Rate limiting verification
// - Memory pressure under load
// - SLA compliance tracking
//
// Run: test-load --model path/to/model.gguf [options]

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// Load test configuration
struct load_config {
    std::string model_path;
    int n_gpu_layers = 99;
    int n_ctx = 2048;
    int n_batch = 512;
    int n_threads = 4;

    // Load parameters
    int n_clients = 4;              // Concurrent clients
    int requests_per_client = 10;   // Requests each client sends
    int min_prompt_tokens = 32;     // Minimum prompt size
    int max_prompt_tokens = 256;    // Maximum prompt size
    int min_gen_tokens = 16;        // Minimum generation length
    int max_gen_tokens = 128;       // Maximum generation length
    double target_rps = 0.0;        // Target requests/sec (0 = unlimited)
    int duration_sec = 0;           // Test duration (0 = until all requests done)

    // SLA thresholds
    double sla_p50_ms = 500.0;      // P50 latency SLA
    double sla_p95_ms = 2000.0;     // P95 latency SLA
    double sla_p99_ms = 5000.0;     // P99 latency SLA

    bool verbose = false;
};

// Request/response tracking
struct request_stats {
    int client_id;
    int request_id;
    int prompt_tokens;
    int gen_tokens;
    double queue_time_ms;
    double process_time_ms;
    double total_time_ms;
    bool success;
    std::string error;
};

// Global statistics
struct global_stats {
    std::atomic<int> total_requests{0};
    std::atomic<int> completed_requests{0};
    std::atomic<int> failed_requests{0};
    std::atomic<int64_t> total_prompt_tokens{0};
    std::atomic<int64_t> total_gen_tokens{0};
    std::atomic<int64_t> total_latency_us{0};

    std::mutex latency_mutex;
    std::vector<double> latencies_ms;

    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;

    void record_latency(double latency_ms) {
        std::lock_guard<std::mutex> lock(latency_mutex);
        latencies_ms.push_back(latency_ms);
    }

    double percentile(double p) {
        std::lock_guard<std::mutex> lock(latency_mutex);
        if (latencies_ms.empty()) return 0.0;

        std::vector<double> sorted = latencies_ms;
        std::sort(sorted.begin(), sorted.end());

        size_t idx = static_cast<size_t>((p / 100.0) * (sorted.size() - 1));
        return sorted[idx];
    }

    double mean_latency() {
        std::lock_guard<std::mutex> lock(latency_mutex);
        if (latencies_ms.empty()) return 0.0;

        double sum = 0.0;
        for (double l : latencies_ms) sum += l;
        return sum / latencies_ms.size();
    }

    double throughput_rps() {
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        if (duration <= 0) return 0.0;
        return completed_requests.load() / duration;
    }

    double tokens_per_sec() {
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        if (duration <= 0) return 0.0;
        return (total_prompt_tokens.load() + total_gen_tokens.load()) / duration;
    }
};

// Request queue with optional rate limiting
class request_queue {
public:
    struct queued_request {
        int client_id;
        int request_id;
        int prompt_tokens;
        int gen_tokens;
        std::chrono::steady_clock::time_point enqueue_time;
    };

    request_queue(double target_rps = 0.0)
        : target_rps_(target_rps), running_(true) {

        if (target_rps_ > 0) {
            interval_us_ = static_cast<int64_t>(1000000.0 / target_rps_);
        }
    }

    void push(queued_request req) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(req);
        cv_.notify_one();
    }

    bool pop(queued_request & req, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this] { return !queue_.empty() || !running_; })) {
            return false;
        }

        if (!running_ && queue_.empty()) {
            return false;
        }

        if (queue_.empty()) {
            return false;
        }

        req = queue_.front();
        queue_.pop_front();

        // Rate limiting
        if (interval_us_ > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                now - last_dequeue_).count();

            if (elapsed < interval_us_) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(interval_us_ - elapsed));
            }
            last_dequeue_ = std::chrono::steady_clock::now();
        }

        return true;
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        cv_.notify_all();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::deque<queued_request> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    double target_rps_;
    int64_t interval_us_ = 0;
    std::chrono::steady_clock::time_point last_dequeue_;
    bool running_;
};

// Worker thread that processes requests
class worker {
public:
    worker(
        int id,
        llama_model * model,
        const load_config & cfg,
        request_queue & queue,
        global_stats & stats
    ) : id_(id), model_(model), cfg_(cfg), queue_(queue), stats_(stats) {}

    void run() {
        // Create context for this worker
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = cfg_.n_ctx;
        ctx_params.n_batch = cfg_.n_batch;
        ctx_params.n_threads = cfg_.n_threads / 2;  // Share threads among workers
        ctx_params.n_threads_batch = ctx_params.n_threads;

        llama_context * ctx = llama_init_from_model(model_, ctx_params);
        if (!ctx) {
            std::cerr << "Worker " << id_ << ": Failed to create context\n";
            return;
        }

        llama_batch batch = llama_batch_init(cfg_.n_batch, 0, 1);

        request_queue::queued_request req;
        while (queue_.pop(req)) {
            process_request(ctx, batch, req);
        }

        llama_batch_free(batch);
        llama_free(ctx);
    }

private:
    void process_request(
        llama_context * ctx,
        llama_batch & batch,
        const request_queue::queued_request & req
    ) {
        auto start = std::chrono::steady_clock::now();

        // Calculate queue time
        double queue_time_ms = std::chrono::duration<double, std::milli>(
            start - req.enqueue_time).count();

        // Clear KV cache for fresh request
        llama_kv_cache_clear(ctx);

        // Generate random tokens for prompt (simulating tokenization)
        std::random_device rd;
        std::mt19937 gen(rd());
        int n_vocab = llama_model_n_vocab(model_);
        std::uniform_int_distribution<> token_dist(1, n_vocab - 1);

        std::vector<llama_token> prompt_tokens(req.prompt_tokens);
        for (int i = 0; i < req.prompt_tokens; i++) {
            prompt_tokens[i] = token_dist(gen);
        }

        // Process prompt
        bool success = true;
        std::string error;

        llama_batch_clear(batch);
        for (int i = 0; i < req.prompt_tokens; i++) {
            llama_batch_add(batch, prompt_tokens[i], i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            success = false;
            error = "Prompt decode failed";
        }

        // Generate tokens
        int n_cur = req.prompt_tokens;
        int generated = 0;

        if (success) {
            for (int i = 0; i < req.gen_tokens && success; i++) {
                float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
                if (!logits) {
                    success = false;
                    error = "Failed to get logits";
                    break;
                }

                // Greedy sampling
                int max_idx = 0;
                float max_val = logits[0];
                for (int j = 1; j < n_vocab; j++) {
                    if (logits[j] > max_val) {
                        max_val = logits[j];
                        max_idx = j;
                    }
                }

                llama_batch_clear(batch);
                llama_batch_add(batch, max_idx, n_cur, { 0 }, true);

                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    success = false;
                    error = "Generation decode failed";
                    break;
                }

                n_cur++;
                generated++;
            }
        }

        auto end = std::chrono::steady_clock::now();
        double process_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double total_time_ms = queue_time_ms + process_time_ms;

        // Update stats
        stats_.completed_requests++;
        if (!success) {
            stats_.failed_requests++;
        }
        stats_.total_prompt_tokens += req.prompt_tokens;
        stats_.total_gen_tokens += generated;
        stats_.record_latency(total_time_ms);

        if (cfg_.verbose) {
            std::cout << "  [Worker " << id_ << "] Client " << req.client_id
                      << " Request " << req.request_id
                      << ": " << (success ? "OK" : "FAIL")
                      << " (" << req.prompt_tokens << "+" << generated << " tokens, "
                      << total_time_ms << " ms)\n";
        }
    }

    int id_;
    llama_model * model_;
    const load_config & cfg_;
    request_queue & queue_;
    global_stats & stats_;
};

// Client that generates requests
class client {
public:
    client(
        int id,
        const load_config & cfg,
        request_queue & queue,
        global_stats & stats
    ) : id_(id), cfg_(cfg), queue_(queue), stats_(stats) {}

    void run() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> prompt_dist(cfg_.min_prompt_tokens, cfg_.max_prompt_tokens);
        std::uniform_int_distribution<> gen_dist(cfg_.min_gen_tokens, cfg_.max_gen_tokens);

        for (int i = 0; i < cfg_.requests_per_client; i++) {
            request_queue::queued_request req;
            req.client_id = id_;
            req.request_id = i;
            req.prompt_tokens = prompt_dist(gen);
            req.gen_tokens = gen_dist(gen);
            req.enqueue_time = std::chrono::steady_clock::now();

            queue_.push(req);
            stats_.total_requests++;

            if (cfg_.verbose) {
                std::cout << "  [Client " << id_ << "] Queued request " << i
                          << " (" << req.prompt_tokens << "+" << req.gen_tokens << " tokens)\n";
            }

            // Small delay between requests
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    int id_;
    const load_config & cfg_;
    request_queue & queue_;
    global_stats & stats_;
};

// Print results
void print_results(const load_config & cfg, const global_stats & stats) {
    double duration = std::chrono::duration<double>(
        stats.end_time - stats.start_time).count();

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "                    LOAD TEST RESULTS\n";
    std::cout << "================================================================\n";
    std::cout << "\n";

    // Configuration
    std::cout << "Configuration:\n";
    std::cout << "  Clients: " << cfg.n_clients << "\n";
    std::cout << "  Requests/client: " << cfg.requests_per_client << "\n";
    std::cout << "  Prompt tokens: " << cfg.min_prompt_tokens << "-" << cfg.max_prompt_tokens << "\n";
    std::cout << "  Gen tokens: " << cfg.min_gen_tokens << "-" << cfg.max_gen_tokens << "\n";
    if (cfg.target_rps > 0) {
        std::cout << "  Target RPS: " << cfg.target_rps << "\n";
    }
    std::cout << "\n";

    // Request stats
    std::cout << "Request Statistics:\n";
    std::cout << "  Total requests: " << stats.total_requests.load() << "\n";
    std::cout << "  Completed: " << stats.completed_requests.load() << "\n";
    std::cout << "  Failed: " << stats.failed_requests.load() << "\n";
    std::cout << "  Success rate: "
              << std::fixed << std::setprecision(2)
              << (100.0 * stats.completed_requests.load() / stats.total_requests.load())
              << "%\n";
    std::cout << "\n";

    // Throughput
    std::cout << "Throughput:\n";
    std::cout << "  Duration: " << std::fixed << std::setprecision(2) << duration << " sec\n";
    std::cout << "  Requests/sec: " << std::fixed << std::setprecision(2)
              << (stats.completed_requests.load() / duration) << "\n";
    std::cout << "  Tokens/sec: " << std::fixed << std::setprecision(2)
              << ((stats.total_prompt_tokens.load() + stats.total_gen_tokens.load()) / duration)
              << "\n";
    std::cout << "  Total tokens: "
              << (stats.total_prompt_tokens.load() + stats.total_gen_tokens.load())
              << " (" << stats.total_prompt_tokens.load() << " prompt + "
              << stats.total_gen_tokens.load() << " gen)\n";
    std::cout << "\n";

    // Latency
    global_stats & mutable_stats = const_cast<global_stats&>(stats);
    double p50 = mutable_stats.percentile(50);
    double p95 = mutable_stats.percentile(95);
    double p99 = mutable_stats.percentile(99);
    double mean = mutable_stats.mean_latency();

    std::cout << "Latency (ms):\n";
    std::cout << "  Mean: " << std::fixed << std::setprecision(2) << mean << "\n";
    std::cout << "  P50:  " << std::fixed << std::setprecision(2) << p50 << "\n";
    std::cout << "  P95:  " << std::fixed << std::setprecision(2) << p95 << "\n";
    std::cout << "  P99:  " << std::fixed << std::setprecision(2) << p99 << "\n";
    std::cout << "\n";

    // SLA compliance
    std::cout << "SLA Compliance:\n";
    bool p50_ok = p50 <= cfg.sla_p50_ms;
    bool p95_ok = p95 <= cfg.sla_p95_ms;
    bool p99_ok = p99 <= cfg.sla_p99_ms;

    std::cout << "  P50 <= " << cfg.sla_p50_ms << " ms: "
              << (p50_ok ? "PASS" : "FAIL") << " (" << p50 << " ms)\n";
    std::cout << "  P95 <= " << cfg.sla_p95_ms << " ms: "
              << (p95_ok ? "PASS" : "FAIL") << " (" << p95 << " ms)\n";
    std::cout << "  P99 <= " << cfg.sla_p99_ms << " ms: "
              << (p99_ok ? "PASS" : "FAIL") << " (" << p99 << " ms)\n";
    std::cout << "\n";

    if (p50_ok && p95_ok && p99_ok) {
        std::cout << "  Overall: ALL SLAs MET\n";
    } else {
        std::cout << "  Overall: SLA VIOLATIONS DETECTED\n";
    }

    std::cout << "================================================================\n";
}

void print_usage(const char * prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model PATH       Path to GGUF model file (required)\n";
    std::cout << "  --ngl N            GPU layers (default: 99)\n";
    std::cout << "  --ctx N            Context size (default: 2048)\n";
    std::cout << "  --batch N          Batch size (default: 512)\n";
    std::cout << "  --threads N        CPU threads (default: 4)\n";
    std::cout << "\n";
    std::cout << "Load parameters:\n";
    std::cout << "  --clients N        Number of concurrent clients (default: 4)\n";
    std::cout << "  --requests N       Requests per client (default: 10)\n";
    std::cout << "  --min-prompt N     Minimum prompt tokens (default: 32)\n";
    std::cout << "  --max-prompt N     Maximum prompt tokens (default: 256)\n";
    std::cout << "  --min-gen N        Minimum generation tokens (default: 16)\n";
    std::cout << "  --max-gen N        Maximum generation tokens (default: 128)\n";
    std::cout << "  --rps N            Target requests/sec, 0=unlimited (default: 0)\n";
    std::cout << "\n";
    std::cout << "SLA thresholds (ms):\n";
    std::cout << "  --sla-p50 N        P50 latency SLA (default: 500)\n";
    std::cout << "  --sla-p95 N        P95 latency SLA (default: 2000)\n";
    std::cout << "  --sla-p99 N        P99 latency SLA (default: 5000)\n";
    std::cout << "\n";
    std::cout << "  --verbose          Verbose output\n";
    std::cout << "  --help             Show this help\n";
}

int main(int argc, char ** argv) {
    load_config cfg;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            cfg.model_path = argv[++i];
        } else if (arg == "--ngl" && i + 1 < argc) {
            cfg.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--ctx" && i + 1 < argc) {
            cfg.n_ctx = std::atoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            cfg.n_batch = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            cfg.n_threads = std::atoi(argv[++i]);
        } else if (arg == "--clients" && i + 1 < argc) {
            cfg.n_clients = std::atoi(argv[++i]);
        } else if (arg == "--requests" && i + 1 < argc) {
            cfg.requests_per_client = std::atoi(argv[++i]);
        } else if (arg == "--min-prompt" && i + 1 < argc) {
            cfg.min_prompt_tokens = std::atoi(argv[++i]);
        } else if (arg == "--max-prompt" && i + 1 < argc) {
            cfg.max_prompt_tokens = std::atoi(argv[++i]);
        } else if (arg == "--min-gen" && i + 1 < argc) {
            cfg.min_gen_tokens = std::atoi(argv[++i]);
        } else if (arg == "--max-gen" && i + 1 < argc) {
            cfg.max_gen_tokens = std::atoi(argv[++i]);
        } else if (arg == "--rps" && i + 1 < argc) {
            cfg.target_rps = std::atof(argv[++i]);
        } else if (arg == "--sla-p50" && i + 1 < argc) {
            cfg.sla_p50_ms = std::atof(argv[++i]);
        } else if (arg == "--sla-p95" && i + 1 < argc) {
            cfg.sla_p95_ms = std::atof(argv[++i]);
        } else if (arg == "--sla-p99" && i + 1 < argc) {
            cfg.sla_p99_ms = std::atof(argv[++i]);
        } else if (arg == "--verbose") {
            cfg.verbose = true;
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Check for env var
    if (cfg.model_path.empty()) {
        const char * env_model = std::getenv("LLAMA_TEST_MODEL");
        if (env_model) {
            cfg.model_path = env_model;
        }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Error: Model path required.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Initialize llama
    llama_backend_init();

    std::cout << "\n";
    std::cout << "================================================================\n";
    std::cout << "    Super-llama.cpp Enterprise - Load Test\n";
    std::cout << "    Author: GALO SERRANO ABAD\n";
    std::cout << "================================================================\n\n";

    std::cout << "Loading model: " << cfg.model_path << "\n";

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model\n";
        llama_backend_free();
        return 1;
    }

    std::cout << "Model loaded. Starting load test...\n\n";

    // Create queue and stats
    request_queue queue(cfg.target_rps);
    global_stats stats;

    // Determine number of workers (1 per GPU layer context is reasonable)
    int n_workers = std::min(cfg.n_clients, 4);  // Max 4 workers to avoid memory issues

    std::cout << "Configuration:\n";
    std::cout << "  Workers: " << n_workers << "\n";
    std::cout << "  Clients: " << cfg.n_clients << "\n";
    std::cout << "  Total requests: " << (cfg.n_clients * cfg.requests_per_client) << "\n";
    std::cout << "\n";

    // Start timing
    stats.start_time = std::chrono::steady_clock::now();

    // Start workers
    std::vector<std::thread> worker_threads;
    std::vector<std::unique_ptr<worker>> workers;

    for (int i = 0; i < n_workers; i++) {
        workers.push_back(std::make_unique<worker>(i, model, cfg, queue, stats));
        worker_threads.emplace_back(&worker::run, workers.back().get());
    }

    // Start clients
    std::vector<std::thread> client_threads;
    std::vector<std::unique_ptr<client>> clients;

    for (int i = 0; i < cfg.n_clients; i++) {
        clients.push_back(std::make_unique<client>(i, cfg, queue, stats));
        client_threads.emplace_back(&client::run, clients.back().get());
    }

    // Wait for clients to finish
    for (auto & t : client_threads) {
        t.join();
    }

    // Wait for queue to empty
    while (queue.size() > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Stop workers
    queue.stop();
    for (auto & t : worker_threads) {
        t.join();
    }

    // End timing
    stats.end_time = std::chrono::steady_clock::now();

    // Print results
    print_results(cfg, stats);

    // Cleanup
    llama_model_free(model);
    llama_backend_free();

    // Return code based on SLA compliance
    global_stats & mutable_stats = stats;
    bool sla_met = mutable_stats.percentile(50) <= cfg.sla_p50_ms &&
                   mutable_stats.percentile(95) <= cfg.sla_p95_ms &&
                   mutable_stats.percentile(99) <= cfg.sla_p99_ms;

    return sla_met ? 0 : 1;
}
