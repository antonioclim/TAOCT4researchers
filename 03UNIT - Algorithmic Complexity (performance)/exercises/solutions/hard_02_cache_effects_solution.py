#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Algorithmic Complexity - Cache Effects Solutions
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for exercises on memory hierarchy and cache optimisation.

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
from typing import Callable

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BENCHMARK UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CacheBenchmarkResult:
    """Result of a cache-related benchmark."""

    name: str
    elapsed_seconds: float
    elements_processed: int
    bytes_processed: int
    throughput_mb_per_sec: float
    cache_efficiency_estimate: str


def benchmark_function(
    func: Callable[[], float],
    name: str,
    elements: int,
    element_size: int = 8,
    warmup_runs: int = 2,
    measurement_runs: int = 5,
) -> CacheBenchmarkResult:
    """
    Benchmark a function with warmup and multiple measurements.

    Args:
        func: Function to benchmark (should return the sum or similar checksum).
        name: Name for the benchmark.
        elements: Number of elements processed.
        element_size: Size of each element in bytes.
        warmup_runs: Number of warmup iterations.
        measurement_runs: Number of measured iterations.

    Returns:
        CacheBenchmarkResult with timing and throughput statistics.
    """
    # Warmup
    for _ in range(warmup_runs):
        func()

    # Measure
    times = []
    for _ in range(measurement_runs):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    bytes_processed = elements * element_size
    throughput = bytes_processed / avg_time / (1024 * 1024)  # MB/s

    # Estimate cache efficiency based on throughput
    # Modern systems: L1 ~100-200 GB/s, L2 ~50-80 GB/s, L3 ~30-50 GB/s, RAM ~10-20 GB/s
    if throughput > 50000:
        efficiency = "L1 cache (excellent)"
    elif throughput > 20000:
        efficiency = "L2 cache (good)"
    elif throughput > 5000:
        efficiency = "L3 cache (moderate)"
    else:
        efficiency = "Main memory (poor)"

    return CacheBenchmarkResult(
        name=name,
        elapsed_seconds=avg_time,
        elements_processed=elements,
        bytes_processed=bytes_processed,
        throughput_mb_per_sec=throughput,
        cache_efficiency_estimate=efficiency,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXERCISE 1 SOLUTION - ROW VS COLUMN MAJOR
# ═══════════════════════════════════════════════════════════════════════════════


def sum_row_major(matrix: np.ndarray) -> float:
    """Sum matrix elements in row-major order (cache-friendly for C arrays)."""
    total = 0.0
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            total += matrix[i, j]
    return total


def sum_column_major(matrix: np.ndarray) -> float:
    """Sum matrix elements in column-major order (cache-unfriendly for C arrays)."""
    total = 0.0
    rows, cols = matrix.shape
    for j in range(cols):
        for i in range(rows):
            total += matrix[i, j]
    return total


def measure_row_vs_column_major(
    sizes: list[int] | None = None,
) -> dict[int, dict[str, CacheBenchmarkResult]]:
    """
    SOLUTION: Measure performance difference between row-major and column-major access.

    This demonstrates cache line effects: row-major access is sequential in memory
    for C-contiguous arrays (NumPy default), while column-major access jumps by
    row_size bytes between each access, causing cache misses.

    Args:
        sizes: List of matrix sizes (N×N) to test.

    Returns:
        Dictionary mapping size to benchmark results for each access pattern.

    Theory:
        - Cache lines are typically 64 bytes (8 doubles)
        - Row-major: 1 cache miss per 8 elements accessed
        - Column-major: 1 cache miss per element (if row > cache line size)
        - Expected speedup: 4-10× for large matrices
    """
    if sizes is None:
        sizes = [256, 512, 1024, 2048, 4096]

    results: dict[int, dict[str, CacheBenchmarkResult]] = {}

    for size in sizes:
        logger.info(f"Testing {size}×{size} matrix ({size * size * 8 / 1024 / 1024:.1f} MB)...")

        # Create matrix with random values
        matrix = np.random.rand(size, size)
        elements = size * size

        # Benchmark row-major
        row_result = benchmark_function(
            lambda m=matrix: sum_row_major(m),
            f"row_major_{size}",
            elements,
        )

        # Benchmark column-major
        col_result = benchmark_function(
            lambda m=matrix: sum_column_major(m),
            f"column_major_{size}",
            elements,
        )

        speedup = col_result.elapsed_seconds / row_result.elapsed_seconds

        results[size] = {
            "row_major": row_result,
            "column_major": col_result,
        }

        logger.info(f"  Row-major: {row_result.elapsed_seconds * 1000:.2f} ms")
        logger.info(f"  Column-major: {col_result.elapsed_seconds * 1000:.2f} ms")
        logger.info(f"  Speedup (row vs column): {speedup:.2f}×")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EXERCISE 2 SOLUTION - STRIDE EFFECTS
# ═══════════════════════════════════════════════════════════════════════════════


def sequential_access(arr: np.ndarray) -> float:
    """Access array elements sequentially (stride 1)."""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total


def strided_access(arr: np.ndarray, stride: int) -> float:
    """Access array elements with given stride."""
    total = 0.0
    for i in range(0, len(arr), stride):
        total += arr[i]
    return total


def random_access(arr: np.ndarray, indices: np.ndarray) -> float:
    """Access array elements in random order."""
    total = 0.0
    for idx in indices:
        total += arr[idx]
    return total


def measure_stride_effects(
    array_size: int = 10_000_000,
    strides: list[int] | None = None,
) -> dict[str, CacheBenchmarkResult]:
    """
    SOLUTION: Measure how access stride affects cache performance.

    Demonstrates that cache lines prefetch contiguous memory, so larger strides
    result in more cache misses per element accessed.

    Args:
        array_size: Size of the array to test.
        strides: List of strides to test.

    Returns:
        Dictionary mapping stride description to benchmark result.

    Theory:
        - Sequential access: 1 cache miss per 8 elements (64 bytes / 8 bytes per double)
        - Stride 8: 1 cache miss per element (exactly cache line size)
        - Stride > 8: 1 cache miss per element + potential TLB misses
        - Random: worst case, each access likely cache miss + TLB miss
    """
    if strides is None:
        strides = [1, 2, 4, 8, 16, 32, 64, 128]

    logger.info(f"Testing stride effects on {array_size:,} element array...")

    arr = np.random.rand(array_size)
    results: dict[str, CacheBenchmarkResult] = {}

    # Sequential access baseline
    result = benchmark_function(
        lambda a=arr: sequential_access(a),
        "sequential",
        array_size,
    )
    results["stride_1_sequential"] = result
    logger.info(f"  Sequential: {result.throughput_mb_per_sec:.0f} MB/s")

    # Strided access
    for stride in strides[1:]:  # Skip 1 (already done as sequential)
        elements_accessed = array_size // stride
        result = benchmark_function(
            lambda a=arr, s=stride: strided_access(a, s),
            f"stride_{stride}",
            elements_accessed,
        )
        results[f"stride_{stride}"] = result
        logger.info(f"  Stride {stride}: {result.throughput_mb_per_sec:.0f} MB/s")

    # Random access
    indices = np.random.permutation(array_size)
    elements_accessed = array_size
    result = benchmark_function(
        lambda a=arr, idx=indices: random_access(a, idx),
        "random",
        elements_accessed,
    )
    results["random"] = result
    logger.info(f"  Random: {result.throughput_mb_per_sec:.0f} MB/s")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EXERCISE 3 SOLUTION - MATRIX MULTIPLICATION COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Naive matrix multiplication (ijk order)."""
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_transposed(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication with B transposed (ikj order with B^T)."""
    n = A.shape[0]
    BT = B.T.copy()  # Transpose B for better cache locality
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * BT[j, k]
    return C


def matmul_blocked(
    A: np.ndarray,
    B: np.ndarray,
    block_size: int = 64,
) -> np.ndarray:
    """Cache-oblivious blocked matrix multiplication."""
    n = A.shape[0]
    C = np.zeros((n, n))

    # Pad to multiple of block_size if necessary
    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            for kk in range(0, n, block_size):
                # Multiply blocks
                for i in range(ii, min(ii + block_size, n)):
                    for j in range(jj, min(jj + block_size, n)):
                        for k in range(kk, min(kk + block_size, n)):
                            C[i, j] += A[i, k] * B[k, j]
    return C


def compare_matmul_versions(
    sizes: list[int] | None = None,
    block_size: int = 32,
) -> dict[int, dict[str, CacheBenchmarkResult]]:
    """
    SOLUTION: Compare different matrix multiplication implementations.

    Demonstrates how algorithm restructuring can dramatically improve
    cache performance without changing asymptotic complexity.

    Args:
        sizes: Matrix sizes to test.
        block_size: Block size for blocked multiplication.

    Returns:
        Dictionary mapping size to benchmark results for each implementation.

    Theory:
        - Naive: O(n³) operations, O(n³) cache misses (inner loop accesses B column)
        - Transposed: O(n³) operations, O(n³/B) cache misses (B = cache line size)
        - Blocked: O(n³) operations, O(n³/√M) cache misses (M = cache size)

        The blocked algorithm keeps working sets in cache, reducing misses by
        a factor proportional to cache size.
    """
    if sizes is None:
        sizes = [64, 128, 256]  # Keep small for pure Python

    results: dict[int, dict[str, CacheBenchmarkResult]] = {}

    for size in sizes:
        logger.info(f"Testing {size}×{size} matrix multiplication...")

        A = np.random.rand(size, size)
        B = np.random.rand(size, size)
        operations = size * size * size  # Approximate FLOP count

        size_results: dict[str, CacheBenchmarkResult] = {}

        # Naive
        result = benchmark_function(
            lambda a=A, b=B: matmul_naive(a, b),
            f"naive_{size}",
            operations,
            element_size=1,  # Count operations, not bytes
            warmup_runs=1,
            measurement_runs=3,
        )
        size_results["naive"] = result
        logger.info(f"  Naive: {result.elapsed_seconds * 1000:.1f} ms")

        # Transposed
        result = benchmark_function(
            lambda a=A, b=B: matmul_transposed(a, b),
            f"transposed_{size}",
            operations,
            element_size=1,
            warmup_runs=1,
            measurement_runs=3,
        )
        size_results["transposed"] = result
        logger.info(f"  Transposed: {result.elapsed_seconds * 1000:.1f} ms")

        # Blocked
        result = benchmark_function(
            lambda a=A, b=B, bs=block_size: matmul_blocked(a, b, bs),
            f"blocked_{size}",
            operations,
            element_size=1,
            warmup_runs=1,
            measurement_runs=3,
        )
        size_results["blocked"] = result
        logger.info(f"  Blocked: {result.elapsed_seconds * 1000:.1f} ms")

        # NumPy (BLAS) for reference
        result = benchmark_function(
            lambda a=A, b=B: np.matmul(a, b),
            f"numpy_{size}",
            operations,
            element_size=1,
            warmup_runs=1,
            measurement_runs=3,
        )
        size_results["numpy"] = result
        logger.info(f"  NumPy (BLAS): {result.elapsed_seconds * 1000:.1f} ms")

        results[size] = size_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXERCISE 4 SOLUTION - CACHE-OBLIVIOUS TRANSPOSE
# ═══════════════════════════════════════════════════════════════════════════════


def transpose_naive(matrix: np.ndarray) -> np.ndarray:
    """Naive in-place transpose (swap elements across diagonal)."""
    n = matrix.shape[0]
    result = matrix.copy()
    for i in range(n):
        for j in range(i + 1, n):
            result[i, j], result[j, i] = result[j, i], result[i, j]
    return result


def transpose_blocked(matrix: np.ndarray, block_size: int = 32) -> np.ndarray:
    """Blocked transpose for better cache utilisation."""
    n = matrix.shape[0]
    result = np.empty_like(matrix)

    for ii in range(0, n, block_size):
        for jj in range(0, n, block_size):
            # Transpose block
            i_end = min(ii + block_size, n)
            j_end = min(jj + block_size, n)
            for i in range(ii, i_end):
                for j in range(jj, j_end):
                    result[j, i] = matrix[i, j]

    return result


def transpose_cache_oblivious(matrix: np.ndarray, threshold: int = 32) -> np.ndarray:
    """
    SOLUTION: Cache-oblivious recursive transpose.

    Uses divide-and-conquer to automatically adapt to cache hierarchy
    without knowing cache sizes.

    Args:
        matrix: Square matrix to transpose.
        threshold: Base case size for direct transpose.

    Returns:
        Transposed matrix.

    Algorithm:
        1. If matrix fits in cache (size ≤ threshold), transpose directly
        2. Otherwise, divide into quadrants and recursively transpose
        3. Swap appropriate quadrants

    Complexity:
        - Time: O(n²)
        - Cache misses: O(n²/B) where B is cache line size
        - Works well regardless of cache size (cache-oblivious property)
    """
    n = matrix.shape[0]
    result = np.empty_like(matrix)

    def recursive_transpose(
        src: np.ndarray,
        dst: np.ndarray,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
    ) -> None:
        """Recursively transpose a submatrix."""
        rows = row_end - row_start
        cols = col_end - col_start

        # Base case: small enough to transpose directly
        if rows <= threshold and cols <= threshold:
            for i in range(row_start, row_end):
                for j in range(col_start, col_end):
                    dst[j, i] = src[i, j]
            return

        # Recursive case: divide along larger dimension
        if rows >= cols:
            mid = row_start + rows // 2
            recursive_transpose(src, dst, row_start, mid, col_start, col_end)
            recursive_transpose(src, dst, mid, row_end, col_start, col_end)
        else:
            mid = col_start + cols // 2
            recursive_transpose(src, dst, row_start, row_end, col_start, mid)
            recursive_transpose(src, dst, row_start, row_end, mid, col_end)

    recursive_transpose(matrix, result, 0, n, 0, n)
    return result


def implement_cache_oblivious_transpose(
    sizes: list[int] | None = None,
) -> dict[int, dict[str, CacheBenchmarkResult]]:
    """
    SOLUTION: Compare transpose implementations.

    Demonstrates cache-oblivious algorithm design principles.
    """
    if sizes is None:
        sizes = [256, 512, 1024, 2048]

    results: dict[int, dict[str, CacheBenchmarkResult]] = {}

    for size in sizes:
        logger.info(f"Testing {size}×{size} transpose...")

        matrix = np.random.rand(size, size)
        elements = size * size

        size_results: dict[str, CacheBenchmarkResult] = {}

        # Naive
        result = benchmark_function(
            lambda m=matrix: transpose_naive(m),
            f"naive_{size}",
            elements,
        )
        size_results["naive"] = result
        logger.info(f"  Naive: {result.elapsed_seconds * 1000:.1f} ms")

        # Blocked
        result = benchmark_function(
            lambda m=matrix: transpose_blocked(m),
            f"blocked_{size}",
            elements,
        )
        size_results["blocked"] = result
        logger.info(f"  Blocked: {result.elapsed_seconds * 1000:.1f} ms")

        # Cache-oblivious
        result = benchmark_function(
            lambda m=matrix: transpose_cache_oblivious(m),
            f"cache_oblivious_{size}",
            elements,
        )
        size_results["cache_oblivious"] = result
        logger.info(f"  Cache-oblivious: {result.elapsed_seconds * 1000:.1f} ms")

        # NumPy reference
        result = benchmark_function(
            lambda m=matrix: m.T.copy(),
            f"numpy_{size}",
            elements,
        )
        size_results["numpy"] = result
        logger.info(f"  NumPy: {result.elapsed_seconds * 1000:.1f} ms")

        results[size] = size_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EXERCISE 5 SOLUTION - FALSE SHARING
# ═══════════════════════════════════════════════════════════════════════════════


def simulate_false_sharing(
    iterations: int = 1_000_000,
    array_size: int = 16,
) -> dict[str, float]:
    """
    SOLUTION: Demonstrate false sharing effects (simulation for single-threaded Python).

    In multi-threaded scenarios, false sharing occurs when threads on different
    cores modify variables that share the same cache line, causing cache
    invalidation even though they're logically independent.

    This simulation demonstrates the concept without actual threads.

    Args:
        iterations: Number of increment operations.
        array_size: Number of counters.

    Returns:
        Dictionary with timing results for different layouts.

    Theory:
        - Cache line: 64 bytes (8 int64 values)
        - Contiguous layout: counters share cache lines
        - Padded layout: each counter on its own cache line

        In real multi-threaded code:
        - Contiguous: ~10-100× slower due to cache line bouncing
        - Padded: Full parallel speedup achieved
    """
    logger.info("Simulating false sharing effects...")

    # Contiguous counters (false sharing in multi-threaded scenario)
    contiguous = np.zeros(array_size, dtype=np.int64)

    start = time.perf_counter()
    for _ in range(iterations):
        idx = random.randint(0, array_size - 1)
        contiguous[idx] += 1
    contiguous_time = time.perf_counter() - start

    # Padded counters (8 values = 64 bytes = 1 cache line padding)
    # Shape: (array_size, 8) with only first element used
    padded = np.zeros((array_size, 8), dtype=np.int64)

    start = time.perf_counter()
    for _ in range(iterations):
        idx = random.randint(0, array_size - 1)
        padded[idx, 0] += 1
    padded_time = time.perf_counter() - start

    logger.info(f"  Contiguous: {contiguous_time * 1000:.1f} ms")
    logger.info(f"  Padded: {padded_time * 1000:.1f} ms")
    logger.info("  Note: Real speedup visible in multi-threaded scenarios")

    return {
        "contiguous_seconds": contiguous_time,
        "padded_seconds": padded_time,
        "explanation": (
            "False sharing occurs when threads modify variables on the same "
            "cache line. Each modification invalidates the cache line for all "
            "other cores, causing expensive coherence traffic. Padding ensures "
            "each variable occupies its own cache line."
        ),
    }


def measure_false_sharing() -> dict[str, float]:
    """
    SOLUTION: Wrapper for false sharing measurement.

    Provides educational context about cache coherence protocols.
    """
    results = simulate_false_sharing()

    # Add educational content
    results["cache_coherence_explanation"] = """
    Cache Coherence and False Sharing:
    
    Modern CPUs use cache coherence protocols (e.g., MESI) to keep caches
    synchronised across cores:
    
    - Modified (M): Line is dirty, owned exclusively
    - Exclusive (E): Line is clean, owned exclusively  
    - Shared (S): Line is clean, possibly in other caches
    - Invalid (I): Line is not valid
    
    False sharing forces lines to bounce between M and I states across cores,
    even when different cores access different variables. This "cache line
    ping-pong" can reduce performance by 10-100× in the worst case.
    
    Solutions:
    1. Padding: Ensure each thread's data is on separate cache lines
    2. Thread-local storage: Give each thread its own copy
    3. Reduce sharing: Redesign algorithms to minimise shared state
    """

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: EXERCISE 6 SOLUTION - MEMORY-EFFICIENT TRANSPOSE
# ═══════════════════════════════════════════════════════════════════════════════


def transpose_inplace_square(matrix: np.ndarray) -> np.ndarray:
    """
    SOLUTION: In-place transpose for square matrices.

    Only swaps elements above diagonal with those below.
    Memory efficient: O(1) extra space.
    """
    n = matrix.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j], matrix[j, i] = matrix[j, i], matrix[i, j]
    return matrix


def transpose_inplace_rectangular(matrix: np.ndarray) -> np.ndarray:
    """
    SOLUTION: In-place transpose for rectangular matrices using cycle-following.

    This is a complex algorithm that follows permutation cycles to transpose
    without extra memory. For an m×n matrix becoming n×m:
    - Element at position i in linear order moves to position (i*n) mod (m*n-1)
    - Exception: last element stays in place

    Args:
        matrix: Rectangular matrix to transpose.

    Returns:
        Transposed matrix (modifies input).

    Complexity:
        - Time: O(m×n)
        - Space: O(1) extra (only tracking visited positions)
    """
    m, n = matrix.shape

    if m == n:
        return transpose_inplace_square(matrix)

    # Flatten view for cycle-following
    flat = matrix.ravel()
    size = m * n

    # Track which positions have been moved
    visited = np.zeros(size, dtype=bool)
    visited[0] = True  # First element stays
    visited[size - 1] = True  # Last element stays

    for start in range(1, size - 1):
        if visited[start]:
            continue

        # Follow the cycle starting from 'start'
        current = start
        temp = flat[current]

        while True:
            # Calculate next position in cycle
            # Position i in m×n matrix (row i//n, col i%n) maps to
            # position in n×m matrix (row i%n, col i//n) = (i%n)*m + (i//n)
            next_pos = (current * m) % (size - 1)

            if next_pos == start:
                # Cycle complete
                flat[current] = temp
                visited[current] = True
                break

            flat[current] = flat[next_pos]
            visited[current] = True
            current = next_pos

    # Reshape to transposed dimensions
    return flat.reshape(n, m)


def implement_memory_efficient_transpose(
    test_rectangular: bool = True,
) -> dict[str, np.ndarray]:
    """
    SOLUTION: Demonstrate memory-efficient transpose implementations.

    Shows that transpose can be done with O(1) extra memory, which is
    important for very large matrices that barely fit in memory.
    """
    results: dict[str, np.ndarray] = {}

    # Square matrix test
    logger.info("Testing in-place square transpose...")
    square = np.arange(16).reshape(4, 4).astype(float)
    logger.info(f"Original:\n{square}")

    square_transposed = transpose_inplace_square(square.copy())
    results["square_original"] = np.arange(16).reshape(4, 4).astype(float)
    results["square_transposed"] = square_transposed
    logger.info(f"Transposed:\n{square_transposed}")

    if test_rectangular:
        logger.info("\nTesting in-place rectangular transpose...")
        rectangular = np.arange(12).reshape(3, 4).astype(float)
        logger.info(f"Original (3×4):\n{rectangular}")

        rect_transposed = transpose_inplace_rectangular(rectangular.copy())
        results["rectangular_original"] = np.arange(12).reshape(3, 4).astype(float)
        results["rectangular_transposed"] = rect_transposed
        logger.info(f"Transposed (4×3):\n{rect_transposed}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: BONUS EXERCISE SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def implement_prefetching_simulation(
    array_size: int = 1_000_000,
    prefetch_distance: int = 16,
) -> dict[str, CacheBenchmarkResult]:
    """
    BONUS SOLUTION: Simulate software prefetching effects.

    Software prefetching can hide memory latency by requesting data before
    it's needed. This simulation demonstrates the concept.

    Args:
        array_size: Size of array to process.
        prefetch_distance: How far ahead to "prefetch".

    Returns:
        Benchmark results showing prefetching benefit.

    Theory:
        - Memory latency: ~100 CPU cycles for main memory access
        - Prefetching: Request data early, do other work while waiting
        - Works best for predictable access patterns
    """
    logger.info("Simulating prefetching effects...")

    arr = np.random.rand(array_size)
    indices = np.random.permutation(array_size)

    # Without prefetching: just access elements
    def no_prefetch() -> float:
        total = 0.0
        for i in range(len(indices)):
            total += arr[indices[i]]
        return total

    # With simulated prefetching: access elements, but also touch future elements
    # (In real code, would use __builtin_prefetch or similar)
    def with_prefetch() -> float:
        total = 0.0
        for i in range(len(indices)):
            # Touch future element (simulates prefetch)
            if i + prefetch_distance < len(indices):
                _ = arr[indices[i + prefetch_distance]]
            total += arr[indices[i]]
        return total

    results: dict[str, CacheBenchmarkResult] = {}

    results["no_prefetch"] = benchmark_function(
        no_prefetch,
        "no_prefetch",
        array_size,
    )
    logger.info(f"  No prefetch: {results['no_prefetch'].elapsed_seconds * 1000:.1f} ms")

    results["with_prefetch"] = benchmark_function(
        with_prefetch,
        "with_prefetch",
        array_size,
    )
    logger.info(f"  With prefetch: {results['with_prefetch'].elapsed_seconds * 1000:.1f} ms")

    logger.info("  Note: Real prefetching uses CPU instructions, not extra accesses")

    return results


def analyse_numa_effects() -> dict[str, str]:
    """
    BONUS SOLUTION: Explain NUMA effects on memory access.

    NUMA (Non-Uniform Memory Access) systems have different memory latencies
    depending on which CPU socket/node the memory is attached to.

    Returns:
        Educational content about NUMA.
    """
    return {
        "overview": """
        NUMA (Non-Uniform Memory Access) Architecture:
        
        In multi-socket systems, memory is distributed across nodes:
        - Local memory: Attached to the same socket as the CPU (~100ns)
        - Remote memory: Attached to a different socket (~150-300ns)
        
        This creates a memory access hierarchy:
        1. L1 cache: ~1ns
        2. L2 cache: ~4ns  
        3. L3 cache: ~15ns
        4. Local DRAM: ~100ns
        5. Remote DRAM: ~150-300ns
        """,
        "implications": """
        Performance Implications:
        
        1. Memory allocation matters:
           - first-touch policy: memory allocated on first-writing node
           - Use numactl or libnuma to control placement
        
        2. Thread migration is expensive:
           - Moving thread to different node = remote memory access
           - Pin threads to nodes for consistent performance
        
        3. NUMA-aware data structures:
           - Replicate read-mostly data on each node
           - Partition write-heavy data by node
           - Use thread-local storage where possible
        """,
        "python_tools": """
        Python Tools for NUMA:
        
        - numpy: Generally NUMA-unaware, but fast enough
        - multiprocessing: Each process can be pinned to a node
        - py-numa: Python bindings for libnuma
        - numexpr: Can help with NUMA by processing in chunks
        
        For serious NUMA optimization, consider C/C++ with explicit
        memory placement using numa_alloc_onnode() and thread pinning.
        """,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CACHE LINE SIZE ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_cache_line_size(max_stride: int = 512) -> int:
    """
    SOLUTION: Estimate cache line size by measuring stride effects.

    The cache line size is where the performance cliff occurs when
    increasing stride - beyond this point, each access is a cache miss.

    Args:
        max_stride: Maximum stride to test.

    Returns:
        Estimated cache line size in bytes.

    Method:
        1. Create large array
        2. Measure access time for various strides
        3. Find stride where time stops increasing linearly
        4. That stride × element_size ≈ cache line size
    """
    logger.info("Estimating cache line size...")

    array_size = 64 * 1024 * 1024 // 8  # 64 MB of doubles
    arr = np.random.rand(array_size)
    strides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    strides = [s for s in strides if s <= max_stride]

    times = []
    for stride in strides:
        iterations = array_size // stride

        start = time.perf_counter()
        total = 0.0
        for i in range(0, array_size, stride):
            total += arr[i]
        elapsed = time.perf_counter() - start

        time_per_access = elapsed / iterations
        times.append(time_per_access)
        logger.info(f"  Stride {stride}: {time_per_access * 1e9:.2f} ns/access")

    # Find where time per access stops increasing significantly
    # This indicates we're now getting one cache miss per access
    ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]

    # Find first ratio close to 1.0 (within 20%)
    cache_line_elements = 8  # Default guess
    for i, ratio in enumerate(ratios):
        if ratio < 1.2:
            cache_line_elements = strides[i + 1]
            break

    cache_line_bytes = cache_line_elements * 8  # 8 bytes per double
    logger.info(f"  Estimated cache line size: {cache_line_bytes} bytes")

    return cache_line_bytes


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def run_all_solutions() -> None:
    """Run all cache effects solutions with demonstration."""
    logger.info("=" * 70)
    logger.info("CACHE EFFECTS SOLUTIONS - COMPLETE DEMONSTRATION")
    logger.info("=" * 70)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 1: Row vs Column Major Access")
    logger.info("─" * 70)
    measure_row_vs_column_major([256, 512])

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 2: Stride Effects")
    logger.info("─" * 70)
    measure_stride_effects(array_size=1_000_000)

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 3: Matrix Multiplication Comparison")
    logger.info("─" * 70)
    compare_matmul_versions([64, 128])

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 4: Cache-Oblivious Transpose")
    logger.info("─" * 70)
    implement_cache_oblivious_transpose([256, 512])

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 5: False Sharing Simulation")
    logger.info("─" * 70)
    measure_false_sharing()

    logger.info("\n" + "─" * 70)
    logger.info("Exercise 6: Memory-Efficient Transpose")
    logger.info("─" * 70)
    implement_memory_efficient_transpose()

    logger.info("\n" + "─" * 70)
    logger.info("Bonus: Cache Line Size Estimation")
    logger.info("─" * 70)
    estimate_cache_line_size()

    logger.info("\n" + "=" * 70)
    logger.info("ALL CACHE EFFECTS SOLUTIONS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cache Effects Exercise Solutions"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run full demonstration",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_all_solutions()
    else:
        logger.info("Run with --demo to see all solutions")
