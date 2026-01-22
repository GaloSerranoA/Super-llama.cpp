# Super-llama.cpp

A fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) with AirLLM-style memory efficiency enhancements for running large language models on memory-constrained devices.

**Modifications Author:** GALO SERRANO ABAD

---

## Overview

Super-llama.cpp extends the original llama.cpp with experimental memory efficiency features inspired by [AirLLM](https://github.com/lyogavin/airllm). These enhancements enable running larger models on devices with limited GPU memory by implementing:

- **Dynamic Layer Scheduling** - Runtime memory-aware layer placement between GPU and CPU
- **Paged KV Cache** - Spillable key-value cache with automatic GPU/CPU page management
- **Async Prefetching** - Overlapped data loading and computation for reduced latency

These features are inspired by AirLLM's memory orchestration concepts, but are implemented natively for GGUF and ggml without PyTorch or HuggingFace dependencies.

All features are **disabled by default** to preserve the original llama.cpp behavior and performance characteristics.

---

## Design Goals

Super-llama.cpp is designed with the following goals:

- Preserve llama.cpp's performance and simplicity by default
- Enable adaptive behavior only when explicitly requested
- Favor deterministic, observable behavior over opaque heuristics
- Remain compatible with upstream llama.cpp where possible
- Provide structured telemetry to support reproducible benchmarking

---

## New Features

### 1. Dynamic Layer Scheduler

The dynamic layer scheduler monitors GPU memory pressure in real-time and can migrate model layers between GPU and CPU based on available memory.

**Key capabilities:**
- Real-time memory telemetry using `ggml_backend_dev_memory()` API
- LRU-based layer eviction when GPU memory is under pressure
- Configurable memory pressure thresholds (default: 85%)
- Per-layer state tracking with access timestamps

**CLI flag:** `--dynamic-layers`

### 2. Paged KV Cache

The paged KV cache divides the key-value cache into fixed-size pages that can be individually moved between GPU and CPU memory.

**Key capabilities:**
- Configurable page size (default: 256 tokens)
- Automatic page eviction using LRU policy
- Page prefetching support
- Seamless fallback to standard KV cache

**CLI flags:**
- `--paged-kv` - Enable paged KV cache
- `--kv-page-size N` - Set page size in tokens (16-8192)

### 3. Async Prefetcher

The async prefetcher uses background worker threads to load upcoming layers and KV pages before they are needed.

**Key capabilities:**
- Configurable worker thread count
- Priority-based request queue
- Automatic lookahead prefetching
- Layer and KV page prefetch support

**CLI flag:** `--async-prefetch`

### 4. Memory Telemetry System

A cross-platform memory monitoring system that provides real-time visibility into GPU and CPU memory usage.

**Supported platforms:**
- Windows (GlobalMemoryStatusEx)
- macOS (vm_statistics64)
- Linux (/proc/meminfo)

### 5. Structured JSON Metrics Logging

Real-time structured logging for observability and performance analysis. Outputs JSON-formatted metrics to file or stderr for monitoring memory pressure, layer migrations, KV cache state, and throughput.

**Key capabilities:**
- Thread-safe metrics collection with atomic counters
- ISO 8601 timestamp formatting
- Configurable logging intervals
- Event-triggered logging on layer/page evictions
- Throughput calculation (tokens/second)
- Pretty-print or compact JSON output

**CLI flags:**
- `--metrics` - Enable metrics logging (outputs to stderr by default)
- `--metrics-file PATH` - Write metrics to specified file instead of stderr

**Output format:**
```json
{
  "timestamp": "2025-01-22T14:30:45.123Z",
  "token": 1234,
  "prompt_tokens": 128,
  "generated_tokens": 1106,
  "gpu_layers_active": 18,
  "cpu_layers_active": 14,
  "layers_evicted": 2,
  "layers_loaded": 5,
  "kv_pages_gpu": 12,
  "kv_pages_cpu": 34,
  "kv_pages_evicted": 8,
  "kv_pages_prefetched": 16,
  "vram_used_mb": 9210.5,
  "vram_total_mb": 12288.0,
  "ram_used_mb": 4096.3,
  "gpu_memory_pressure": 0.750,
  "cpu_memory_pressure": 0.312,
  "tokens_per_sec": 3.85,
  "avg_tokens_per_sec": 4.12,
  "prefetch_pending": 2,
  "prefetch_completed": 45,
  "prefetch_failed": 0
}
```

**Compact format (default):**
```json
{"timestamp":"2025-01-22T14:30:45.123Z","token":1234,"gpu_layers_active":18,"layers_evicted":2,"kv_pages_gpu":12,"kv_pages_cpu":34,"vram_used_mb":9210.5,"tokens_per_sec":3.85}
```

---

## New CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dynamic-layers` | Enable dynamic layer scheduling based on memory pressure | disabled |
| `--paged-kv` | Enable paged KV cache with GPU/CPU spilling | disabled |
| `--async-prefetch` | Enable async layer/KV prefetching | disabled |
| `--mem-pressure FLOAT` | Memory pressure threshold (0.0-1.0) | 0.85 |
| `--kv-page-size N` | KV cache page size in tokens (16-8192) | 256 |
| `--metrics` | Enable structured JSON metrics logging | disabled |
| `--metrics-file PATH` | Write metrics to file instead of stderr | stderr |

**Environment variables:**
- `LLAMA_ARG_DYNAMIC_LAYERS`
- `LLAMA_ARG_PAGED_KV`
- `LLAMA_ARG_ASYNC_PREFETCH`
- `LLAMA_ARG_MEM_PRESSURE`
- `LLAMA_ARG_KV_PAGE_SIZE`
- `LLAMA_ARG_METRICS`
- `LLAMA_ARG_METRICS_FILE`

---

## New Source Files

### Headers
- `src/llama-mem-telemetry.h` - Memory telemetry system interface
- `src/llama-layer-sched.h` - Dynamic layer scheduler interface
- `src/llama-kv-cache-paged.h` - Paged KV cache interface
- `src/llama-prefetch.h` - Async prefetcher interface
- `src/llama-metrics.h` - Structured JSON metrics logging interface

### Implementation
- `src/llama-mem-telemetry.cpp` - Platform-specific memory monitoring
- `src/llama-layer-sched.cpp` - Layer migration and LRU tracking
- `src/llama-kv-cache-paged.cpp` - Page allocation and eviction
- `src/llama-prefetch.cpp` - Worker threads and request queue
- `src/llama-metrics.cpp` - JSON formatting and throughput calculation

---

## Modified Files

| File | Changes |
|------|---------|
| `include/llama.h` | Added context params for new features |
| `src/llama-cparams.h` | Added mirrored internal params |
| `src/llama-context.h` | Added subsystem member pointers |
| `src/llama-context.cpp` | Added initialization and defaults |
| `common/common.h` | Added common_params fields |
| `common/common.cpp` | Added param passing to context |
| `common/arg.cpp` | Added CLI argument parsers |
| `src/CMakeLists.txt` | Added new source files to build |

---

## Architecture

High-level runtime architecture showing optional adaptive subsystems:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           llama_context                                   │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │mem_telemetry │  │ layer_sched  │  │ prefetcher│  │ metrics_logger  │  │
│  │              │  │              │  │           │  │                 │  │
│  │ - GPU stats  │  │ - Layer state│  │ - Workers │  │ - JSON output   │  │
│  │ - CPU stats  │  │ - LRU queue  │  │ - Queue   │  │ - Throughput    │  │
│  │ - Pressure   │  │ - Migration  │  │ - Lookahead│ │ - Timestamps    │  │
│  └──────┬───────┘  └──────┬───────┘  └─────┬─────┘  └────────┬────────┘  │
│         │                 │                │                  │           │
│         └────────┬────────┴────────────────┴──────────────────┘           │
│                  │                                                        │
│          ┌───────▼───────┐                                                │
│          │ kv_cache_paged │ (optional)                                    │
│          │               │                                                │
│          │ - Pages       │                                                │
│          │ - GPU LRU     │                                                │
│          │ - Eviction    │                                                │
│          └───────────────┘                                                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Build Instructions

Standard llama.cpp build process applies:

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release

# Run with new features
./build/bin/llama-cli -m model.gguf --dynamic-layers --async-prefetch
```

---

## Usage Examples

### Basic memory-efficient inference
```bash
llama-cli -m large-model.gguf \
    --dynamic-layers \
    --mem-pressure 0.80
```

### With paged KV cache for long contexts
```bash
llama-cli -m model.gguf \
    --paged-kv \
    --kv-page-size 512 \
    --ctx-size 32768
```

### Full memory optimization stack
```bash
llama-cli -m model.gguf \
    --dynamic-layers \
    --paged-kv \
    --async-prefetch \
    --mem-pressure 0.85 \
    --kv-page-size 256
```

### With metrics logging to file
```bash
llama-cli -m model.gguf \
    --dynamic-layers \
    --paged-kv \
    --metrics \
    --metrics-file metrics.jsonl
```

### Real-time metrics monitoring
```bash
# Terminal 1: Run inference with metrics to stderr
llama-cli -m model.gguf --dynamic-layers --metrics 2>&1 | grep "^\[METRICS\]"

# Or pipe to a file and tail
llama-cli -m model.gguf --dynamic-layers --metrics-file metrics.jsonl &
tail -f metrics.jsonl | jq .
```

---

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Memory Telemetry | ✅ Complete | Full cross-platform support |
| Dynamic Layer Scheduler | ⚠️ Partial | State tracking complete, actual migration is placeholder |
| Paged KV Cache | ⚠️ Partial | Metadata management complete, actual data movement is placeholder |
| Async Prefetcher | ✅ Complete | Full worker thread implementation |
| Structured JSON Metrics | ✅ Complete | Full logging with throughput and eviction tracking |
| CLI Integration | ✅ Complete | All flags and env vars working |

**Note:** The actual tensor migration between GPU and CPU memory requires deep integration with ggml's buffer management system. The current implementation provides the infrastructure and state tracking, with actual data movement marked as TODO for future development.

---

## Benchmarking

Benchmark results comparing Super-llama.cpp against static `--n-gpu-layers` will be published once dynamic layer migration and KV page movement are fully implemented.

The structured metrics logging system is designed to support:

- VRAM saturation stress tests
- Long-context scalability benchmarks
- Throughput under memory pressure
- Efficiency curves across VRAM budgets

---

## Additional Fixes and Improvements in This Fork

1. **Thread Safety** - Fixed missing mutex lock in `get_gpu_layer_count()`
2. **Move Semantics** - Replaced `std::priority_queue` with sorted `std::deque` for move-only `std::promise` types
3. **MSVC Compatibility** - Fixed `ggml_backend_dev_type` variable naming conflicts with function names

---

## License

Same as llama.cpp - MIT License

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov
- [AirLLM](https://github.com/lyogavin/airllm) for the memory efficiency concepts
- Anthropic Claude for implementation assistance
