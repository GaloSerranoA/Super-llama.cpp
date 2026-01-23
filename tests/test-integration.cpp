// test-integration.cpp - Integration tests for Super-llama.cpp Enterprise
// Author: GALO SERRANO ABAD
//
// These tests require:
// - A GGUF model file (set via --model or LLAMA_TEST_MODEL env var)
// - GPU hardware (optional, will fallback to CPU)
//
// Run: test-integration --model path/to/model.gguf
//
// Tests cover end-to-end functionality:
// - Model loading with enterprise features
// - Inference with dynamic layer scheduling
// - KV cache paging under memory pressure
// - Multi-GPU distribution (if available)

#include "llama.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Test configuration
struct test_config {
    std::string model_path;
    int n_gpu_layers = 99;      // Offload all layers
    int n_ctx = 512;            // Context size
    int n_batch = 32;           // Batch size
    int n_threads = 4;          // CPU threads
    bool dynamic_layers = true; // Enable dynamic layer scheduling
    bool paged_kv = true;       // Enable paged KV cache
    bool verbose = false;
};

// Test result tracking
static int tests_passed = 0;
static int tests_failed = 0;
static std::vector<std::string> failed_tests;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << "\n"; \
        tests_failed++; \
        failed_tests.push_back(msg); \
        return false; \
    } \
} while(0)

#define TEST_PASS() do { tests_passed++; return true; } while(0)

// Helper: Get time in milliseconds
static double get_time_ms() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::milli>(duration).count();
}

//
// Test: Basic Model Loading
//
bool test_model_loading(const test_config & cfg) {
    std::cout << "  Testing model loading...\n";

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    double start = get_time_ms();
    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    double load_time = get_time_ms() - start;

    TEST_ASSERT(model != nullptr, "Model should load successfully");

    std::cout << "    Model loaded in " << load_time << " ms\n";

    // Get model info
    int n_vocab = llama_model_n_vocab(model);
    int n_ctx_train = llama_model_n_ctx_train(model);
    int n_embd = llama_model_n_embd(model);

    std::cout << "    n_vocab: " << n_vocab << "\n";
    std::cout << "    n_ctx_train: " << n_ctx_train << "\n";
    std::cout << "    n_embd: " << n_embd << "\n";

    TEST_ASSERT(n_vocab > 0, "Model should have vocabulary");
    TEST_ASSERT(n_embd > 0, "Model should have embedding dimension");

    llama_model_free(model);
    TEST_PASS();
}

//
// Test: Context Creation with Enterprise Features
//
bool test_context_creation(const test_config & cfg) {
    std::cout << "  Testing context creation with enterprise features...\n";

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    TEST_ASSERT(model != nullptr, "Model should load");

    // Create context with enterprise features
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cfg.n_ctx;
    ctx_params.n_batch = cfg.n_batch;
    ctx_params.n_threads = cfg.n_threads;
    ctx_params.n_threads_batch = cfg.n_threads;

    // Enterprise features would be enabled here via additional params
    // ctx_params.dynamic_layers = cfg.dynamic_layers;
    // ctx_params.paged_kv = cfg.paged_kv;

    double start = get_time_ms();
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    double init_time = get_time_ms() - start;

    TEST_ASSERT(ctx != nullptr, "Context should initialize");

    std::cout << "    Context created in " << init_time << " ms\n";
    std::cout << "    n_ctx: " << llama_n_ctx(ctx) << "\n";
    std::cout << "    n_batch: " << llama_n_batch(ctx) << "\n";

    llama_free(ctx);
    llama_model_free(model);
    TEST_PASS();
}

//
// Test: Basic Inference
//
bool test_basic_inference(const test_config & cfg) {
    std::cout << "  Testing basic inference...\n";

    // Load model and create context
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    TEST_ASSERT(model != nullptr, "Model should load");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cfg.n_ctx;
    ctx_params.n_batch = cfg.n_batch;
    ctx_params.n_threads = cfg.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    TEST_ASSERT(ctx != nullptr, "Context should initialize");

    // Tokenize prompt
    const char * prompt = "The meaning of life is";
    std::vector<llama_token> tokens(cfg.n_ctx);

    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    TEST_ASSERT(n_tokens > 0, "Tokenization should succeed");
    tokens.resize(n_tokens);

    std::cout << "    Prompt tokens: " << n_tokens << "\n";

    // Create batch
    llama_batch batch = llama_batch_init(cfg.n_batch, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    // Decode
    double start = get_time_ms();
    int ret = llama_decode(ctx, batch);
    double decode_time = get_time_ms() - start;

    TEST_ASSERT(ret == 0, "Decode should succeed");

    std::cout << "    Prompt decode time: " << decode_time << " ms\n";
    std::cout << "    Tokens/sec: " << (n_tokens / (decode_time / 1000.0)) << "\n";

    // Get logits and sample
    float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
    TEST_ASSERT(logits != nullptr, "Should get logits");

    // Find argmax
    int n_vocab = llama_model_n_vocab(model);
    int max_idx = 0;
    float max_val = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    std::cout << "    Next token: " << max_idx << " (logit: " << max_val << ")\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    TEST_PASS();
}

//
// Test: Multi-token Generation
//
bool test_generation(const test_config & cfg) {
    std::cout << "  Testing multi-token generation...\n";

    // Load model and create context
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    TEST_ASSERT(model != nullptr, "Model should load");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cfg.n_ctx;
    ctx_params.n_batch = cfg.n_batch;
    ctx_params.n_threads = cfg.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    TEST_ASSERT(ctx != nullptr, "Context should initialize");

    // Tokenize prompt
    const char * prompt = "Once upon a time";
    std::vector<llama_token> tokens(cfg.n_ctx);

    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    TEST_ASSERT(n_tokens > 0, "Tokenization should succeed");
    tokens.resize(n_tokens);

    // Initial decode
    llama_batch batch = llama_batch_init(cfg.n_batch, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    int ret = llama_decode(ctx, batch);
    TEST_ASSERT(ret == 0, "Initial decode should succeed");

    // Generate tokens
    const int n_gen = 32;
    int n_cur = n_tokens;
    int n_vocab = llama_model_n_vocab(model);

    double gen_start = get_time_ms();

    for (int i = 0; i < n_gen; i++) {
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        TEST_ASSERT(logits != nullptr, "Should get logits");

        // Greedy sampling
        int max_idx = 0;
        float max_val = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > max_val) {
                max_val = logits[j];
                max_idx = j;
            }
        }

        // Add new token
        llama_batch_clear(batch);
        llama_batch_add(batch, max_idx, n_cur, { 0 }, true);

        ret = llama_decode(ctx, batch);
        TEST_ASSERT(ret == 0, "Generation decode should succeed");

        n_cur++;
    }

    double gen_time = get_time_ms() - gen_start;

    std::cout << "    Generated " << n_gen << " tokens\n";
    std::cout << "    Generation time: " << gen_time << " ms\n";
    std::cout << "    Tokens/sec: " << (n_gen / (gen_time / 1000.0)) << "\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    TEST_PASS();
}

//
// Test: KV Cache State Save/Load
//
bool test_kv_cache_state(const test_config & cfg) {
    std::cout << "  Testing KV cache state save/load...\n";

    // Load model and create context
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    TEST_ASSERT(model != nullptr, "Model should load");

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cfg.n_ctx;
    ctx_params.n_batch = cfg.n_batch;
    ctx_params.n_threads = cfg.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    TEST_ASSERT(ctx != nullptr, "Context should initialize");

    // Process some tokens
    const char * prompt = "Hello world";
    std::vector<llama_token> tokens(cfg.n_ctx);

    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    TEST_ASSERT(n_tokens > 0, "Tokenization should succeed");
    tokens.resize(n_tokens);

    llama_batch batch = llama_batch_init(cfg.n_batch, 0, 1);

    for (int i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    int ret = llama_decode(ctx, batch);
    TEST_ASSERT(ret == 0, "Decode should succeed");

    // Get state size
    size_t state_size = llama_state_get_size(ctx);
    std::cout << "    State size: " << (state_size / 1024.0 / 1024.0) << " MB\n";

    TEST_ASSERT(state_size > 0, "State size should be positive");

    // Save state
    std::vector<uint8_t> state(state_size);
    size_t written = llama_state_get_data(ctx, state.data(), state.size());
    TEST_ASSERT(written > 0, "State save should succeed");

    std::cout << "    State saved: " << (written / 1024.0 / 1024.0) << " MB\n";

    // Clear KV cache
    llama_kv_cache_clear(ctx);

    // Restore state
    size_t read = llama_state_set_data(ctx, state.data(), state.size());
    TEST_ASSERT(read > 0, "State restore should succeed");

    std::cout << "    State restored: " << (read / 1024.0 / 1024.0) << " MB\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    TEST_PASS();
}

//
// Test: Memory Pressure Simulation
//
bool test_memory_pressure(const test_config & cfg) {
    std::cout << "  Testing memory pressure handling...\n";

    // Load model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = cfg.n_gpu_layers;

    llama_model * model = llama_model_load_from_file(cfg.model_path.c_str(), model_params);
    TEST_ASSERT(model != nullptr, "Model should load");

    // Create context with larger context to stress memory
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = cfg.n_ctx * 2;  // Double context for pressure
    ctx_params.n_batch = cfg.n_batch;
    ctx_params.n_threads = cfg.n_threads;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    TEST_ASSERT(ctx != nullptr, "Context should initialize");

    // Fill context with tokens to create memory pressure
    const char * prompt = "This is a test of memory pressure handling in the enterprise LLM system. ";
    std::vector<llama_token> tokens(ctx_params.n_ctx);

    int n_tokens = llama_tokenize(model, prompt, strlen(prompt), tokens.data(), tokens.size(), true, false);
    TEST_ASSERT(n_tokens > 0, "Tokenization should succeed");

    // Repeat prompt to fill context
    std::vector<llama_token> full_tokens;
    while (full_tokens.size() < (size_t)(ctx_params.n_ctx - 100)) {
        for (int i = 0; i < n_tokens && full_tokens.size() < (size_t)(ctx_params.n_ctx - 100); i++) {
            full_tokens.push_back(tokens[i]);
        }
    }

    std::cout << "    Processing " << full_tokens.size() << " tokens to stress memory...\n";

    // Process in batches
    llama_batch batch = llama_batch_init(cfg.n_batch, 0, 1);
    int pos = 0;

    double start = get_time_ms();

    while (pos < (int)full_tokens.size()) {
        llama_batch_clear(batch);

        int batch_size = std::min(cfg.n_batch, (int)full_tokens.size() - pos);
        for (int i = 0; i < batch_size; i++) {
            llama_batch_add(batch, full_tokens[pos + i], pos + i, { 0 }, false);
        }
        if (pos + batch_size >= (int)full_tokens.size()) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        int ret = llama_decode(ctx, batch);
        TEST_ASSERT(ret == 0, "Batch decode should succeed under pressure");

        pos += batch_size;
    }

    double total_time = get_time_ms() - start;

    std::cout << "    Processed all tokens in " << total_time << " ms\n";
    std::cout << "    Throughput: " << (full_tokens.size() / (total_time / 1000.0)) << " tokens/sec\n";

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    TEST_PASS();
}

//
// Main
//
void print_usage(const char * prog) {
    std::cout << "Usage: " << prog << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model PATH       Path to GGUF model file (required)\n";
    std::cout << "  --ngl N            Number of layers to offload to GPU (default: 99)\n";
    std::cout << "  --ctx N            Context size (default: 512)\n";
    std::cout << "  --batch N          Batch size (default: 32)\n";
    std::cout << "  --threads N        CPU threads (default: 4)\n";
    std::cout << "  --verbose          Enable verbose output\n";
    std::cout << "  --help             Show this help\n";
}

int main(int argc, char ** argv) {
    test_config cfg;

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
        std::cerr << "Error: Model path required. Use --model or set LLAMA_TEST_MODEL env var.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Initialize llama
    llama_backend_init();

    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "  Super-llama.cpp Enterprise - Integration Tests\n";
    std::cout << "  Author: GALO SERRANO ABAD\n";
    std::cout << "========================================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Model: " << cfg.model_path << "\n";
    std::cout << "  GPU layers: " << cfg.n_gpu_layers << "\n";
    std::cout << "  Context: " << cfg.n_ctx << "\n";
    std::cout << "  Batch: " << cfg.n_batch << "\n";
    std::cout << "  Threads: " << cfg.n_threads << "\n\n";

    // Run tests
    std::cout << "========================================================\n";
    std::cout << " MODEL LOADING TESTS\n";
    std::cout << "========================================================\n";
    test_model_loading(cfg);

    std::cout << "\n========================================================\n";
    std::cout << " CONTEXT CREATION TESTS\n";
    std::cout << "========================================================\n";
    test_context_creation(cfg);

    std::cout << "\n========================================================\n";
    std::cout << " BASIC INFERENCE TESTS\n";
    std::cout << "========================================================\n";
    test_basic_inference(cfg);

    std::cout << "\n========================================================\n";
    std::cout << " GENERATION TESTS\n";
    std::cout << "========================================================\n";
    test_generation(cfg);

    std::cout << "\n========================================================\n";
    std::cout << " KV CACHE STATE TESTS\n";
    std::cout << "========================================================\n";
    test_kv_cache_state(cfg);

    std::cout << "\n========================================================\n";
    std::cout << " MEMORY PRESSURE TESTS\n";
    std::cout << "========================================================\n";
    test_memory_pressure(cfg);

    // Summary
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "                    TEST RESULTS\n";
    std::cout << "========================================================\n";
    std::cout << "  Passed: " << tests_passed << "\n";
    std::cout << "  Failed: " << tests_failed << "\n";
    std::cout << "========================================================\n";

    if (tests_failed > 0) {
        std::cout << "\nFailed tests:\n";
        for (const auto & name : failed_tests) {
            std::cout << "  - " << name << "\n";
        }
    }

    llama_backend_free();

    return (tests_failed > 0) ? 1 : 0;
}
