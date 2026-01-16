#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Lab 1: Benchmark Suite — Solution File
═══════════════════════════════════════════════════════════════════════════════

This file contains solutions to the exercises from lab_3_01_benchmark_suite.py.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# Import from main lab file
from lab_3_01_benchmark_suite import (
    BenchmarkResult,
    benchmark,
    generate_random_data,
    quicksort,
    merge_sort,
    compute_speedup,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1 SOLUTION: Memory Profiling
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResultWithMemory(BenchmarkResult):
    """Extended BenchmarkResult with memory tracking."""
    peak_memory_mb: float = 0.0
    memory_samples: list[float] = field(default_factory=list)
    
    @property
    def mean_memory(self) -> float:
        """Mean memory usage across runs."""
        return sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary including memory data."""
        result = super().to_dict()
        result['peak_memory_mb'] = self.peak_memory_mb
        result['mean_memory_mb'] = self.mean_memory
        return result


def benchmark_with_memory(
    func: Callable[..., Any],
    *args: Any,
    runs: int = 10,
    warmup: int = 2,
    name: str | None = None,
    n: int = 0,
    **kwargs: Any
) -> BenchmarkResultWithMemory:
    """
    Execute benchmark with memory profiling.
    
    Uses tracemalloc to measure peak memory usage during execution.
    
    Args:
        func: Function to benchmark.
        *args: Positional arguments for func.
        runs: Number of measurement runs.
        warmup: Number of warmup runs.
        name: Name for reporting.
        n: Input size for reporting.
        **kwargs: Keyword arguments for func.
    
    Returns:
        BenchmarkResultWithMemory with timing and memory statistics.
    """
    name = name or getattr(func, '__name__', 'anonymous')
    
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)
    
    gc.collect()
    gc.disable()
    
    times: list[float] = []
    memory_samples: list[float] = []
    peak_memory = 0.0
    
    try:
        for _ in range(runs):
            # Start memory tracking
            tracemalloc.start()
            
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times.append(elapsed * 1000)  # ms
            memory_mb = peak / (1024 * 1024)  # Convert to MB
            memory_samples.append(memory_mb)
            peak_memory = max(peak_memory, memory_mb)
            
    finally:
        gc.enable()
    
    return BenchmarkResultWithMemory(
        name=name,
        n=n,
        times=times,
        unit='ms',
        peak_memory_mb=peak_memory,
        memory_samples=memory_samples
    )


def demo_memory_profiling() -> None:
    """Demonstrate memory profiling capability."""
    logger.info("═" * 60)
    logger.info("  EXERCISE 1 SOLUTION: Memory Profiling")
    logger.info("═" * 60)
    
    n = 10000
    data = generate_random_data(n)
    
    result = benchmark_with_memory(merge_sort, data, runs=5, n=n, name="merge_sort")
    logger.info(f"{result}")
    logger.info(f"Peak memory: {result.peak_memory_mb:.2f} MB")
    logger.info(f"Mean memory: {result.mean_memory:.2f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 SOLUTION: Cache-Aware Matrix Multiplication
# ═══════════════════════════════════════════════════════════════════════════════

def matmul_naive(
    A: NDArray[np.float64],
    B: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Naive matrix multiplication: C[i][j] = sum(A[i][k] * B[k][j]).
    
    This version has poor cache locality because B is accessed
    column-wise, causing many cache misses.
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += A[i, k] * B[k, j]
            C[i, j] = total
    
    return C


def matmul_transposed(
    A: NDArray[np.float64],
    B: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Transposed matrix multiplication.
    
    By transposing B first, we access both matrices row-wise,
    which is cache-friendly for row-major storage.
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    B_T = B.T.copy()  # Ensure contiguous memory
    
    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += A[i, k] * B_T[j, k]  # Both row access
            C[i, j] = total
    
    return C


def matmul_blocked(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    block_size: int = 64
) -> NDArray[np.float64]:
    """
    Blocked (tiled) matrix multiplication.
    
    Processes matrices in blocks that fit in cache, maximising
    data reuse and minimising cache misses.
    
    Block size should be chosen based on L1 cache size.
    Typical L1: 32KB → block_size ≈ 64 for double precision.
    """
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                # Process block
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        total = C[i, j]
                        for k in range(kk, min(kk + block_size, n)):
                            total += A[i, k] * B[k, j]
                        C[i, j] = total
    
    return C


def demo_cache_aware_matmul() -> None:
    """Demonstrate cache-aware matrix multiplication."""
    logger.info("═" * 60)
    logger.info("  EXERCISE 2 SOLUTION: Cache-Aware Matrix Multiplication")
    logger.info("═" * 60)
    
    # Use smaller size for reasonable demo time
    n = 200
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    
    # Benchmark naive
    result_naive = benchmark(
        matmul_naive, A, B,
        runs=3, warmup=1, name="Naive", n=n
    )
    logger.info(f"Naive:      {result_naive}")
    
    # Benchmark transposed
    result_trans = benchmark(
        matmul_transposed, A, B,
        runs=3, warmup=1, name="Transposed", n=n
    )
    logger.info(f"Transposed: {result_trans}")
    logger.info(f"  Speedup: {compute_speedup(result_naive, result_trans):.2f}x")
    
    # Benchmark blocked
    result_blocked = benchmark(
        matmul_blocked, A, B,
        runs=3, warmup=1, name="Blocked", n=n
    )
    logger.info(f"Blocked:    {result_blocked}")
    logger.info(f"  Speedup: {compute_speedup(result_naive, result_blocked):.2f}x")
    
    # NumPy reference
    result_numpy = benchmark(
        np.matmul, A, B,
        runs=3, warmup=1, name="NumPy", n=n
    )
    logger.info(f"NumPy:      {result_numpy}")
    logger.info(f"  Speedup: {compute_speedup(result_naive, result_numpy):.2f}x")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 SOLUTION: Parallel Sorting Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def _sort_chunk(chunk: list[float]) -> list[float]:
    """Sort a single chunk using quicksort."""
    return sorted(chunk)


def _merge_sorted(left: list[float], right: list[float]) -> list[float]:
    """Merge two sorted lists."""
    result: list[float] = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def parallel_merge_sort(
    arr: list[float],
    num_workers: int | None = None
) -> list[float]:
    """
    Parallel merge sort using multiprocessing.
    
    Algorithm:
    1. Split array into chunks (one per worker)
    2. Sort chunks in parallel
    3. Merge sorted chunks sequentially
    
    Args:
        arr: List to sort.
        num_workers: Number of parallel workers (default: CPU count).
    
    Returns:
        Sorted list.
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    n = len(arr)
    if n < 1000 or num_workers <= 1:
        return sorted(arr)
    
    # Split into chunks
    chunk_size = (n + num_workers - 1) // num_workers
    chunks = [arr[i:i + chunk_size] for i in range(0, n, chunk_size)]
    
    # Sort chunks in parallel
    with Pool(num_workers) as pool:
        sorted_chunks = pool.map(_sort_chunk, chunks)
    
    # Merge sorted chunks
    result = sorted_chunks[0]
    for chunk in sorted_chunks[1:]:
        result = _merge_sorted(result, chunk)
    
    return result


def demo_parallel_sorting() -> None:
    """Demonstrate parallel sorting benchmark."""
    logger.info("═" * 60)
    logger.info("  EXERCISE 3 SOLUTION: Parallel Sorting Benchmark")
    logger.info("═" * 60)
    
    n = 100000
    data = generate_random_data(n)
    num_cpus = cpu_count()
    
    logger.info(f"Array size: {n:,}")
    logger.info(f"Available CPUs: {num_cpus}")
    
    # Sequential baseline
    baseline = benchmark(sorted, data, runs=5, name="Sequential", n=n)
    logger.info(f"Sequential: {baseline}")
    
    # Parallel with different worker counts
    for workers in [2, 4, num_cpus]:
        if workers > num_cpus:
            continue
        
        result = benchmark(
            parallel_merge_sort, data,
            runs=5, name=f"Parallel({workers})", n=n,
            num_workers=workers
        )
        speedup = compute_speedup(baseline, result)
        efficiency = speedup / workers
        
        logger.info(f"Parallel({workers}): {result}")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Efficiency: {efficiency:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all exercise solutions."""
    parser = argparse.ArgumentParser(description="Lab 3.1 Exercise Solutions")
    parser.add_argument("--exercise", type=int, choices=[1, 2, 3],
                        help="Run specific exercise (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all exercises")
    args = parser.parse_args()
    
    if args.exercise == 1 or args.all:
        demo_memory_profiling()
    
    if args.exercise == 2 or args.all:
        demo_cache_aware_matmul()
    
    if args.exercise == 3 or args.all:
        demo_parallel_sorting()
    
    if not args.exercise and not args.all:
        logger.info("Use --exercise N or --all to run solutions")


if __name__ == "__main__":
    main()
