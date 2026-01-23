#!/usr/bin/env python3
"""
benchmark-enterprise.py - Performance benchmarks for Super-llama.cpp Enterprise
Author: GALO SERRANO ABAD

This script measures:
- Model loading time
- Prompt processing throughput (tokens/sec)
- Generation throughput (tokens/sec)
- Memory usage (VRAM/RAM)
- Enterprise feature overhead

Usage:
    python benchmark-enterprise.py --model path/to/model.gguf [options]

Requirements:
    - Python 3.8+
    - psutil (pip install psutil)
    - Super-llama.cpp built with llama-bench
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory monitoring disabled.")
    print("Install with: pip install psutil")


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    tokens: int
    time_ms: float
    tokens_per_sec: float
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    extra: Dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results"""
    model_name: str
    model_size_mb: float
    n_params: int
    n_layers: int
    n_ctx: int
    n_gpu_layers: int
    results: List[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "model": {
                "name": self.model_name,
                "size_mb": self.model_size_mb,
                "n_params": self.n_params,
                "n_layers": self.n_layers,
            },
            "config": {
                "n_ctx": self.n_ctx,
                "n_gpu_layers": self.n_gpu_layers,
            },
            "results": [
                {
                    "name": r.name,
                    "tokens": r.tokens,
                    "time_ms": r.time_ms,
                    "tokens_per_sec": r.tokens_per_sec,
                    "memory_mb": r.memory_mb,
                    "gpu_memory_mb": r.gpu_memory_mb,
                    **r.extra
                }
                for r in self.results
            ]
        }


def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    if not HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_usage() -> float:
    """Get GPU memory usage in MB (requires nvidia-smi)"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Sum all GPUs
            total = sum(int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip())
            return float(total)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return 0.0


def run_llama_bench(
    model_path: str,
    n_gpu_layers: int = 99,
    n_ctx: int = 512,
    n_batch: int = 512,
    n_prompt: int = 128,
    n_gen: int = 128,
    n_threads: int = 4,
    llama_bench_path: Optional[str] = None
) -> Optional[Dict]:
    """Run llama-bench and parse results"""

    if llama_bench_path is None:
        # Try to find llama-bench
        candidates = [
            "./build/bin/Release/llama-bench.exe",
            "./build/bin/llama-bench",
            "./llama-bench",
            "llama-bench",
        ]
        for candidate in candidates:
            if os.path.exists(candidate) or subprocess.run(
                ["which", candidate], capture_output=True
            ).returncode == 0:
                llama_bench_path = candidate
                break

    if llama_bench_path is None:
        print("Error: llama-bench not found. Build with: cmake --build build --target llama-bench")
        return None

    cmd = [
        llama_bench_path,
        "-m", model_path,
        "-ngl", str(n_gpu_layers),
        "-c", str(n_ctx),
        "-b", str(n_batch),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(n_threads),
        "-o", "json",
    ]

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"llama-bench failed: {result.stderr}")
            return None

        # Parse JSON output
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        print("llama-bench timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse llama-bench output: {e}")
        print(f"Output was: {result.stdout[:500]}...")
        return None


def benchmark_model_loading(model_path: str, n_gpu_layers: int, n_iterations: int = 3) -> BenchmarkResult:
    """Benchmark model loading time"""
    print("\n--- Model Loading Benchmark ---")

    # This would require llama.cpp Python bindings or a custom test binary
    # For now, estimate from llama-bench warm-up

    times = []
    for i in range(n_iterations):
        start = time.time()
        # Simulate model loading overhead measurement
        # In practice, this would use llama_model_load_from_file
        time.sleep(0.1)  # Placeholder
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f} ms")

    avg_time = sum(times) / len(times)

    return BenchmarkResult(
        name="model_loading",
        tokens=0,
        time_ms=avg_time,
        tokens_per_sec=0,
        memory_mb=get_memory_usage(),
        extra={"iterations": n_iterations}
    )


def benchmark_prompt_processing(
    model_path: str,
    n_gpu_layers: int,
    prompt_sizes: List[int] = [32, 128, 512, 1024],
    n_ctx: int = 2048,
    n_threads: int = 4,
) -> List[BenchmarkResult]:
    """Benchmark prompt processing at different sizes"""
    print("\n--- Prompt Processing Benchmark ---")

    results = []

    for prompt_size in prompt_sizes:
        if prompt_size > n_ctx:
            continue

        print(f"\n  Testing prompt size: {prompt_size}")

        bench_result = run_llama_bench(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_prompt=prompt_size,
            n_gen=1,  # Minimal generation
            n_threads=n_threads,
        )

        if bench_result:
            # Extract prompt processing metrics
            for entry in bench_result:
                if entry.get("n_prompt", 0) == prompt_size:
                    t_prompt = entry.get("avg_ts", 0)  # tokens/sec for prompt

                    results.append(BenchmarkResult(
                        name=f"prompt_pp_{prompt_size}",
                        tokens=prompt_size,
                        time_ms=prompt_size / t_prompt * 1000 if t_prompt > 0 else 0,
                        tokens_per_sec=t_prompt,
                        memory_mb=get_memory_usage(),
                        gpu_memory_mb=get_gpu_memory_usage(),
                        extra={"prompt_size": prompt_size}
                    ))
                    print(f"    Throughput: {t_prompt:.2f} tokens/sec")
                    break

    return results


def benchmark_generation(
    model_path: str,
    n_gpu_layers: int,
    gen_sizes: List[int] = [32, 128, 256],
    n_ctx: int = 2048,
    n_threads: int = 4,
) -> List[BenchmarkResult]:
    """Benchmark token generation at different lengths"""
    print("\n--- Token Generation Benchmark ---")

    results = []

    for gen_size in gen_sizes:
        print(f"\n  Testing generation length: {gen_size}")

        bench_result = run_llama_bench(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_prompt=32,  # Small prompt
            n_gen=gen_size,
            n_threads=n_threads,
        )

        if bench_result:
            for entry in bench_result:
                if entry.get("n_gen", 0) == gen_size:
                    t_gen = entry.get("avg_ts", 0)

                    results.append(BenchmarkResult(
                        name=f"generation_{gen_size}",
                        tokens=gen_size,
                        time_ms=gen_size / t_gen * 1000 if t_gen > 0 else 0,
                        tokens_per_sec=t_gen,
                        memory_mb=get_memory_usage(),
                        gpu_memory_mb=get_gpu_memory_usage(),
                        extra={"gen_length": gen_size}
                    ))
                    print(f"    Throughput: {t_gen:.2f} tokens/sec")
                    break

    return results


def benchmark_batch_sizes(
    model_path: str,
    n_gpu_layers: int,
    batch_sizes: List[int] = [1, 8, 32, 128, 512],
    n_ctx: int = 2048,
    n_threads: int = 4,
) -> List[BenchmarkResult]:
    """Benchmark different batch sizes"""
    print("\n--- Batch Size Benchmark ---")

    results = []

    for batch_size in batch_sizes:
        print(f"\n  Testing batch size: {batch_size}")

        bench_result = run_llama_bench(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=batch_size,
            n_prompt=256,
            n_gen=32,
            n_threads=n_threads,
        )

        if bench_result:
            for entry in bench_result:
                t_total = entry.get("avg_ts", 0)

                results.append(BenchmarkResult(
                    name=f"batch_{batch_size}",
                    tokens=256 + 32,
                    time_ms=(256 + 32) / t_total * 1000 if t_total > 0 else 0,
                    tokens_per_sec=t_total,
                    memory_mb=get_memory_usage(),
                    gpu_memory_mb=get_gpu_memory_usage(),
                    extra={"batch_size": batch_size}
                ))
                print(f"    Throughput: {t_total:.2f} tokens/sec")
                break

    return results


def benchmark_gpu_offload(
    model_path: str,
    offload_levels: List[int] = [0, 10, 20, 40, 99],
    n_ctx: int = 512,
    n_threads: int = 4,
) -> List[BenchmarkResult]:
    """Benchmark different GPU offload levels"""
    print("\n--- GPU Offload Benchmark ---")

    results = []

    for ngl in offload_levels:
        print(f"\n  Testing GPU layers: {ngl}")

        bench_result = run_llama_bench(
            model_path=model_path,
            n_gpu_layers=ngl,
            n_ctx=n_ctx,
            n_prompt=128,
            n_gen=64,
            n_threads=n_threads,
        )

        if bench_result:
            for entry in bench_result:
                t_total = entry.get("avg_ts", 0)

                results.append(BenchmarkResult(
                    name=f"offload_ngl{ngl}",
                    tokens=128 + 64,
                    time_ms=(128 + 64) / t_total * 1000 if t_total > 0 else 0,
                    tokens_per_sec=t_total,
                    memory_mb=get_memory_usage(),
                    gpu_memory_mb=get_gpu_memory_usage(),
                    extra={"n_gpu_layers": ngl}
                ))
                print(f"    Throughput: {t_total:.2f} tokens/sec")
                print(f"    GPU Memory: {get_gpu_memory_usage():.0f} MB")
                break

    return results


def print_results_table(suite: BenchmarkSuite):
    """Print results as a formatted table"""
    print("\n")
    print("=" * 80)
    print("                    BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Model: {suite.model_name}")
    print(f"Size: {suite.model_size_mb:.1f} MB | Layers: {suite.n_layers} | Context: {suite.n_ctx}")
    print(f"GPU Layers: {suite.n_gpu_layers}")
    print("=" * 80)
    print()
    print(f"{'Benchmark':<30} {'Tokens':<10} {'Time (ms)':<12} {'Tokens/sec':<12} {'Memory':<10}")
    print("-" * 80)

    for r in suite.results:
        mem_str = f"{r.memory_mb:.0f} MB" if r.memory_mb > 0 else "-"
        print(f"{r.name:<30} {r.tokens:<10} {r.time_ms:<12.2f} {r.tokens_per_sec:<12.2f} {mem_str:<10}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Performance benchmarks for Super-llama.cpp Enterprise"
    )
    parser.add_argument("--model", "-m", required=True, help="Path to GGUF model")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers (default: 99)")
    parser.add_argument("--ctx", type=int, default=2048, help="Context size (default: 2048)")
    parser.add_argument("--threads", "-t", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer tests)")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return 1

    # Get model info
    model_size_mb = os.path.getsize(args.model) / (1024 * 1024)
    model_name = os.path.basename(args.model)

    print("\n")
    print("=" * 60)
    print("  Super-llama.cpp Enterprise - Performance Benchmarks")
    print("  Author: GALO SERRANO ABAD")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Size: {model_size_mb:.1f} MB")
    print(f"GPU Layers: {args.ngl}")
    print(f"Context: {args.ctx}")
    print(f"Threads: {args.threads}")

    # Create benchmark suite
    suite = BenchmarkSuite(
        model_name=model_name,
        model_size_mb=model_size_mb,
        n_params=0,  # Would need to parse model
        n_layers=0,  # Would need to parse model
        n_ctx=args.ctx,
        n_gpu_layers=args.ngl,
    )

    # Run benchmarks
    if args.quick:
        prompt_sizes = [128, 512]
        gen_sizes = [64, 128]
        batch_sizes = [32, 256]
        offload_levels = [0, args.ngl]
    else:
        prompt_sizes = [32, 128, 512, 1024]
        gen_sizes = [32, 128, 256]
        batch_sizes = [1, 8, 32, 128, 512]
        offload_levels = [0, 10, 20, 40, 99]

    # Prompt processing
    suite.results.extend(benchmark_prompt_processing(
        args.model, args.ngl, prompt_sizes, args.ctx, args.threads
    ))

    # Generation
    suite.results.extend(benchmark_generation(
        args.model, args.ngl, gen_sizes, args.ctx, args.threads
    ))

    # Batch sizes
    suite.results.extend(benchmark_batch_sizes(
        args.model, args.ngl, batch_sizes, args.ctx, args.threads
    ))

    # GPU offload (only if not quick mode)
    if not args.quick:
        suite.results.extend(benchmark_gpu_offload(
            args.model, offload_levels, args.ctx, args.threads
        ))

    # Print results
    print_results_table(suite)

    # Save JSON output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(suite.to_dict(), f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
