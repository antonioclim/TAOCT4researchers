#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Cache Effects (Hard)
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Modern computer performance is dominated by memory hierarchy effects. The gap
between CPU speed and memory latency means that cache-friendly code can be
orders of magnitude faster than cache-hostile code with identical Big-O
complexity. This exercise develops your understanding of practical performance
optimisation beyond asymptotic analysis.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Explain the memory hierarchy and its performance implications
2. Identify cache-friendly vs cache-hostile access patterns
3. Optimise algorithms for better cache utilisation
4. Measure and quantify cache effects empirically

ESTIMATED TIME
──────────────
- Reading: 20 minutes
- Coding: 60 minutes
- Total: 80 minutes

DIFFICULTY: ⭐⭐⭐⭐⭐ (Hard)

BACKGROUND
──────────
Memory Hierarchy (typical latencies):
    - L1 cache: ~1 ns (32-64 KB)
    - L2 cache: ~4 ns (256 KB - 1 MB)
    - L3 cache: ~15 ns (2-32 MB)
    - RAM: ~100 ns
    - SSD: ~100,000 ns
    - HDD: ~10,000,000 ns

Cache Line: Typically 64 bytes. When you access one byte, the entire
64-byte cache line is loaded. Sequential access is fast; random access
is slow.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DEMONSTRATING CACHE EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════

def sum_row_major(matrix: np.ndarray) -> float:
    """
    Sum matrix elements in row-major order (cache friendly).
    
    Row-major order accesses consecutive memory locations, achieving
    excellent cache utilisation.
    
    Access pattern: matrix[0,0], matrix[0,1], ..., matrix[0,n-1],
                   matrix[1,0], matrix[1,1], ...
    
    Args:
        matrix: 2D numpy array.
        
    Returns:
        Sum of all elements.
    """
    total = 0.0
    rows, cols = matrix.shape
    
    for i in range(rows):
        for j in range(cols):
            total += matrix[i, j]
    
    return total


def sum_column_major(matrix: np.ndarray) -> float:
    """
    Sum matrix elements in column-major order (cache hostile).
    
    Column-major order jumps by `cols` elements in memory on each access,
    causing many cache misses.
    
    Access pattern: matrix[0,0], matrix[1,0], ..., matrix[n-1,0],
                   matrix[0,1], matrix[1,1], ...
    
    Args:
        matrix: 2D numpy array.
        
    Returns:
        Sum of all elements.
    """
    total = 0.0
    rows, cols = matrix.shape
    
    for j in range(cols):
        for i in range(rows):
            total += matrix[i, j]
    
    return total


def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Naive matrix multiplication (cache unfriendly).
    
    The inner loop accesses B column-wise, causing poor cache utilisation.
    
    Args:
        a: First matrix (m × n).
        b: Second matrix (n × p).
        
    Returns:
        Product matrix (m × p).
    """
    m, n = a.shape
    _, p = b.shape
    c = np.zeros((m, p))
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i, j] += a[i, k] * b[k, j]  # b[k,j] is column-major access!
    
    return c


def matmul_transposed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication with transposed B (cache friendly).
    
    By transposing B first, all inner loop accesses are row-major.
    
    Args:
        a: First matrix (m × n).
        b: Second matrix (n × p).
        
    Returns:
        Product matrix (m × p).
    """
    m, n = a.shape
    _, p = b.shape
    c = np.zeros((m, p))
    bt = b.T.copy()  # Transpose B once
    
    for i in range(m):
        for j in range(p):
            for k in range(n):
                c[i, j] += a[i, k] * bt[j, k]  # Both row-major now!
    
    return c


def matmul_blocked(
    a: np.ndarray, 
    b: np.ndarray, 
    block_size: int = 64
) -> np.ndarray:
    """
    Blocked (tiled) matrix multiplication (cache optimal).
    
    By processing the matrix in blocks that fit in cache, we achieve
    much better cache utilisation than either naive approach.
    
    Args:
        a: First matrix (m × n).
        b: Second matrix (n × p).
        block_size: Size of square blocks (tune to cache size).
        
    Returns:
        Product matrix (m × p).
    """
    m, n = a.shape
    _, p = b.shape
    c = np.zeros((m, p))
    
    for i0 in range(0, m, block_size):
        for j0 in range(0, p, block_size):
            for k0 in range(0, n, block_size):
                # Process block
                i_end = min(i0 + block_size, m)
                j_end = min(j0 + block_size, p)
                k_end = min(k0 + block_size, n)
                
                for i in range(i0, i_end):
                    for j in range(j0, j_end):
                        for k in range(k0, k_end):
                            c[i, j] += a[i, k] * b[k, j]
    
    return c


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ARRAY ACCESS PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def sequential_access(arr: np.ndarray) -> float:
    """
    Access array elements sequentially.
    
    This is the optimal access pattern for cache utilisation.
    Expected: ~1 cache miss per cache line (64 bytes / 8 bytes per float = 8 elements)
    """
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total


def strided_access(arr: np.ndarray, stride: int) -> float:
    """
    Access array elements with a stride.
    
    Larger strides cause more cache misses.
    At stride = cache_line_size / element_size, every access is a cache miss.
    """
    total = 0.0
    n = len(arr)
    indices = list(range(0, n, stride))
    
    for i in indices:
        total += arr[i]
    return total


def random_access(arr: np.ndarray, indices: np.ndarray) -> float:
    """
    Access array elements in random order.
    
    This is the worst case for cache utilisation.
    Expected: nearly 1 cache miss per access.
    """
    total = 0.0
    for i in indices:
        total += arr[i]
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: YOUR TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def measure_row_vs_column_major() -> dict[str, Any]:
    """
    EXERCISE 1: Quantify the cache effect on matrix traversal.
    
    Create matrices of various sizes and measure row-major vs column-major
    traversal times.
    
    Returns:
        Dictionary with:
        - "sizes": List of matrix sizes tested
        - "row_major_times": List of times for row-major traversal
        - "column_major_times": List of times for column-major traversal
        - "speedup": List of (column_time / row_time) ratios
        
    Test with square matrices of sizes: [256, 512, 1024, 2048, 4096]
    
    Questions to answer (in comments):
        - At what size does the cache effect become significant?
        - What is the maximum speedup you observe?
        - How does this relate to your CPU's cache sizes?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement measure_row_vs_column_major")


def measure_stride_effects() -> dict[str, Any]:
    """
    EXERCISE 2: Measure the effect of stride on access time.
    
    Create a large array (e.g., 10 million elements) and measure
    access time for different strides.
    
    Returns:
        Dictionary with:
        - "strides": List of strides tested [1, 2, 4, 8, 16, 32, 64, 128]
        - "times": List of times for each stride
        - "elements_per_second": List of throughput values
        
    Questions to answer:
        - At what stride does performance drop significantly?
        - How does this relate to cache line size (typically 64 bytes)?
        - Why does random access perform even worse than large strides?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement measure_stride_effects")


def compare_matmul_versions() -> dict[str, Any]:
    """
    EXERCISE 3: Compare matrix multiplication implementations.
    
    Compare naive, transposed and blocked matrix multiplication.
    
    Returns:
        Dictionary with:
        - "sizes": Matrix sizes tested [64, 128, 256, 512]
        - "naive_times": Times for naive version
        - "transposed_times": Times for transposed version
        - "blocked_times": Times for blocked version
        - "numpy_times": Times for np.matmul (reference)
        
    Also experiment with different block sizes for blocked version.
    Report the optimal block size you find.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compare_matmul_versions")


def implement_cache_oblivious_transpose(matrix: np.ndarray) -> np.ndarray:
    """
    EXERCISE 4: Implement cache-oblivious matrix transpose.
    
    A cache-oblivious algorithm achieves good cache utilisation without
    knowing the cache size. The recursive approach naturally achieves this.
    
    Approach:
        1. Base case: If matrix is small (e.g., 32×32), transpose directly
        2. Recursive case: Divide matrix into quadrants and recursively
           transpose each, then swap appropriate quadrants
    
    Args:
        matrix: Input matrix (must be square, size power of 2).
        
    Returns:
        Transposed matrix.
        
    Compare performance with the naive in-place transpose.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement implement_cache_oblivious_transpose")


def measure_false_sharing() -> dict[str, Any]:
    """
    EXERCISE 5: Demonstrate false sharing in multi-threaded code.
    
    False sharing occurs when threads on different cores modify variables
    that share a cache line, causing unnecessary cache invalidation.
    
    Create two scenarios:
    1. Multiple counters in adjacent memory (bad - false sharing)
    2. Multiple counters with padding between them (good - no sharing)
    
    Use Python's threading or multiprocessing to demonstrate the effect.
    
    Returns:
        Dictionary with:
        - "adjacent_time": Time for adjacent counters
        - "padded_time": Time for padded counters
        - "speedup": padded_time / adjacent_time (should be > 1)
        
    Note: Effect may be less pronounced in Python due to GIL, but
    principles still apply. Consider using numpy or multiprocessing.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement measure_false_sharing")


def implement_memory_efficient_transpose(matrix: np.ndarray) -> np.ndarray:
    """
    EXERCISE 6: Implement in-place transpose for non-square matrices.
    
    In-place transpose of a non-square matrix is tricky because elements
    move in cycles. Implement the follow-the-cycles algorithm.
    
    Args:
        matrix: Input matrix (m × n), stored as 1D array.
        
    Returns:
        Transposed matrix (n × m).
        
    Requirements:
        - O(1) auxiliary space (not counting the matrix itself)
        - Must handle non-square matrices
        
    Hint: Use the cycle-following approach. For matrix of size m×n,
    element at position p moves to position p' = (p*m) mod (m*n-1)
    (with special handling for position 0 and m*n-1).
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement implement_memory_efficient_transpose")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BONUS CHALLENGES
# ═══════════════════════════════════════════════════════════════════════════════

def implement_prefetching_simulation() -> dict[str, float]:
    """
    BONUS 1: Simulate the effect of prefetching.
    
    Modern CPUs prefetch sequential data. Simulate this by:
    1. Processing data in chunks
    2. "Warming" the next chunk while processing current
    
    Compare random vs sequential access patterns and measure
    the benefit of explicit prefetch hints (where available).
    
    Returns:
        Dictionary with timing comparisons.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement implement_prefetching_simulation")


def analyse_numa_effects() -> dict[str, Any]:
    """
    BONUS 2: Analyse NUMA (Non-Uniform Memory Access) effects.
    
    On multi-socket systems, memory access time depends on which
    socket allocated the memory. Analyse this effect if running
    on a NUMA system.
    
    Returns:
        Dictionary with NUMA analysis results.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement analyse_numa_effects")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result of a cache benchmark."""
    name: str
    size: int
    time_seconds: float
    elements_per_second: float
    estimated_cache_misses: float | None = None


def benchmark_access_pattern(
    name: str,
    arr: np.ndarray,
    access_func: Any,
    repetitions: int = 5
) -> BenchmarkResult:
    """Benchmark an array access pattern."""
    times = []
    
    for _ in range(repetitions):
        start = time.perf_counter()
        _ = access_func(arr)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    median_time = sorted(times)[len(times) // 2]
    elements = len(arr)
    
    return BenchmarkResult(
        name=name,
        size=elements,
        time_seconds=median_time,
        elements_per_second=elements / median_time,
    )


def estimate_cache_line_size() -> int:
    """
    Estimate the CPU's cache line size by measuring stride effects.
    
    Returns:
        Estimated cache line size in bytes.
    """
    arr = np.zeros(10_000_000, dtype=np.float64)  # 80 MB
    
    strides = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    times = []
    
    for stride in strides:
        n_accesses = len(arr) // stride
        indices = np.arange(0, len(arr), stride)
        
        # Warmup
        _ = arr[indices].sum()
        
        start = time.perf_counter()
        for _ in range(3):
            _ = arr[indices].sum()
        elapsed = time.perf_counter() - start
        
        times.append(elapsed / (n_accesses * 3))
    
    # Find stride where time per access increases significantly
    # This corresponds to cache line size
    for i in range(1, len(strides)):
        if times[i] > times[i-1] * 1.5:
            return strides[i] * 8  # 8 bytes per float64
    
    return 64  # Default assumption


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: VERIFICATION AND DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def verify_implementations() -> bool:
    """Verify that implementations are correct."""
    # Test matrix operations produce correct results
    a = np.random.rand(64, 64)
    b = np.random.rand(64, 64)
    
    naive_result = matmul_naive(a, b)
    transposed_result = matmul_transposed(a, b)
    blocked_result = matmul_blocked(a, b)
    numpy_result = np.matmul(a, b)
    
    assert np.allclose(naive_result, numpy_result), "Naive matmul incorrect"
    assert np.allclose(transposed_result, numpy_result), "Transposed matmul incorrect"
    assert np.allclose(blocked_result, numpy_result), "Blocked matmul incorrect"
    
    # Test sum functions
    mat = np.random.rand(100, 100)
    row_sum = sum_row_major(mat)
    col_sum = sum_column_major(mat)
    np_sum = mat.sum()
    
    assert abs(row_sum - np_sum) < 1e-6, "Row-major sum incorrect"
    assert abs(col_sum - np_sum) < 1e-6, "Column-major sum incorrect"
    
    logger.info("All verifications passed!")
    return True


def demo() -> None:
    """Demonstrate cache effects."""
    logger.info("=" * 70)
    logger.info("CACHE EFFECTS DEMONSTRATION")
    logger.info("=" * 70)
    
    # Estimate cache line size
    logger.info(f"\nEstimated cache line size: {estimate_cache_line_size()} bytes")
    
    # Demo 1: Row vs Column major
    logger.info("\nDemo 1: Row-major vs Column-major matrix traversal")
    
    for size in [512, 1024, 2048]:
        mat = np.random.rand(size, size)
        
        # Row-major
        start = time.perf_counter()
        _ = sum_row_major(mat)
        row_time = time.perf_counter() - start
        
        # Column-major
        start = time.perf_counter()
        _ = sum_column_major(mat)
        col_time = time.perf_counter() - start
        
        speedup = col_time / row_time
        logger.info(
            f"  {size}×{size}: row={row_time:.4f}s, col={col_time:.4f}s, "
            f"speedup={speedup:.2f}×"
        )
    
    # Demo 2: Stride effects
    logger.info("\nDemo 2: Array access stride effects")
    
    arr = np.random.rand(10_000_000)
    
    for stride in [1, 4, 16, 64]:
        n_accesses = len(arr) // stride
        indices = np.arange(0, len(arr), stride)
        
        start = time.perf_counter()
        total = arr[indices].sum()
        elapsed = time.perf_counter() - start
        
        throughput = n_accesses / elapsed / 1e6
        logger.info(
            f"  stride={stride:>2}: {elapsed:.4f}s, {throughput:.1f}M elem/s"
        )
    
    # Demo 3: Matrix multiplication comparison
    logger.info("\nDemo 3: Matrix multiplication comparison")
    
    for size in [128, 256]:
        a = np.random.rand(size, size)
        b = np.random.rand(size, size)
        
        start = time.perf_counter()
        _ = matmul_naive(a, b)
        naive_time = time.perf_counter() - start
        
        start = time.perf_counter()
        _ = matmul_transposed(a, b)
        trans_time = time.perf_counter() - start
        
        start = time.perf_counter()
        _ = matmul_blocked(a, b, block_size=32)
        blocked_time = time.perf_counter() - start
        
        start = time.perf_counter()
        _ = np.matmul(a, b)
        numpy_time = time.perf_counter() - start
        
        logger.info(f"  {size}×{size}:")
        logger.info(f"    naive:     {naive_time:.4f}s")
        logger.info(f"    transposed:{trans_time:.4f}s ({naive_time/trans_time:.1f}× faster)")
        logger.info(f"    blocked:   {blocked_time:.4f}s ({naive_time/blocked_time:.1f}× faster)")
        logger.info(f"    numpy:     {numpy_time:.6f}s ({naive_time/numpy_time:.0f}× faster)")
    
    logger.info("\n" + "=" * 70)
    logger.info("Complete exercises to master cache optimisation!")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cache Effects Practice Exercise"
    )
    parser.add_argument("--verify", action="store_true", help="Run verifications")
    parser.add_argument("--demo", action="store_true", help="Run demonstrations")
    args = parser.parse_args()
    
    if args.verify:
        verify_implementations()
    elif args.demo:
        demo()
    else:
        logger.info("Use --verify to test or --demo to see demonstrations")


if __name__ == "__main__":
    main()
