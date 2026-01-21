#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTIONS: medium_01_sorting_benchmark.py
Week 3, Practice Exercise Solutions
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
import time
import statistics
import logging
from dataclasses import dataclass
from typing import Callable, Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class SortBenchmarkResult:
    """Result of sorting benchmark."""
    algorithm: str
    size: int
    times: list[float]
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.times)
    
    @property
    def median(self) -> float:
        return statistics.median(self.times)
    
    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Implement Sorting Algorithms
# ═══════════════════════════════════════════════════════════════════════════════

def bubble_sort(arr: list[int]) -> list[int]:
    """SOLUTION: Bubble sort implementation."""
    arr = arr.copy()
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def insertion_sort(arr: list[int]) -> list[int]:
    """SOLUTION: Insertion sort implementation."""
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr: list[int]) -> list[int]:
    """SOLUTION: Merge sort implementation."""
    if len(arr) <= 1:
        return arr.copy()
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    result = []
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


def quicksort(arr: list[int]) -> list[int]:
    """SOLUTION: Quicksort implementation."""
    if len(arr) <= 1:
        return arr.copy()
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Benchmarking Framework
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_sort(
    sort_func: Callable[[list[int]], list[int]],
    sizes: list[int],
    runs: int = 5,
    data_type: str = "random"
) -> list[SortBenchmarkResult]:
    """
    SOLUTION: Benchmark a sorting algorithm across sizes.
    """
    results = []
    
    for n in sizes:
        # Generate data based on type
        if data_type == "random":
            base_data = [random.randint(0, n * 10) for _ in range(n)]
        elif data_type == "sorted":
            base_data = list(range(n))
        elif data_type == "reverse":
            base_data = list(range(n, 0, -1))
        elif data_type == "nearly_sorted":
            base_data = list(range(n))
            # Swap ~5% of elements
            for _ in range(n // 20):
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                base_data[i], base_data[j] = base_data[j], base_data[i]
        else:
            base_data = [random.randint(0, n * 10) for _ in range(n)]
        
        times = []
        for _ in range(runs):
            data = base_data.copy()
            start = time.perf_counter()
            sort_func(data)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        results.append(SortBenchmarkResult(
            algorithm=sort_func.__name__,
            size=n,
            times=times
        ))
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Compare Algorithms
# ═══════════════════════════════════════════════════════════════════════════════

def compare_sorting_algorithms(
    algorithms: list[Callable[[list[int]], list[int]]],
    sizes: list[int],
    data_types: list[str] | None = None
) -> dict[str, dict[str, list[SortBenchmarkResult]]]:
    """
    SOLUTION: Compare multiple sorting algorithms.
    """
    if data_types is None:
        data_types = ["random", "sorted", "reverse"]
    
    results: dict[str, dict[str, list[SortBenchmarkResult]]] = {}
    
    for algo in algorithms:
        results[algo.__name__] = {}
        for dtype in data_types:
            results[algo.__name__][dtype] = benchmark_sort(algo, sizes, data_type=dtype)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Estimate Complexity
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_complexity_from_benchmarks(
    results: list[SortBenchmarkResult]
) -> tuple[float, str]:
    """
    SOLUTION: Estimate complexity from benchmark results using log-log analysis.
    """
    import math
    
    if len(results) < 2:
        return 0.0, "Insufficient data"
    
    # Use log-log regression
    log_sizes = [math.log(r.size) for r in results]
    log_times = [math.log(r.median) for r in results]
    
    n = len(log_sizes)
    sum_x = sum(log_sizes)
    sum_y = sum(log_times)
    sum_xy = sum(x * y for x, y in zip(log_sizes, log_times))
    sum_x2 = sum(x * x for x in log_sizes)
    
    # Slope = exponent
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    
    # Classify
    if slope < 0.5:
        complexity = "O(1) or O(log n)"
    elif slope < 1.5:
        complexity = "O(n)"
    elif slope < 1.8:
        complexity = "O(n log n)"
    elif slope < 2.5:
        complexity = "O(n²)"
    else:
        complexity = "O(n³) or worse"
    
    return slope, complexity


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Find Crossover Points
# ═══════════════════════════════════════════════════════════════════════════════

def find_crossover_point(
    algo1: Callable[[list[int]], list[int]],
    algo2: Callable[[list[int]], list[int]],
    start: int = 10,
    end: int = 1000,
    step: int = 10
) -> int | None:
    """
    SOLUTION: Find where algo2 becomes faster than algo1.
    """
    for n in range(start, end, step):
        data = [random.randint(0, n * 10) for _ in range(n)]
        
        # Time algo1
        data1 = data.copy()
        start_time = time.perf_counter()
        algo1(data1)
        time1 = time.perf_counter() - start_time
        
        # Time algo2
        data2 = data.copy()
        start_time = time.perf_counter()
        algo2(data2)
        time2 = time.perf_counter() - start_time
        
        if time2 < time1:
            return n
    
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo_solutions() -> None:
    """Demonstrate all solutions."""
    logger.info("=" * 60)
    logger.info("SORTING BENCHMARK SOLUTIONS DEMONSTRATION")
    logger.info("=" * 60)
    
    # Demo 1: Verify sorts work
    logger.info("\n1. Verify Sorting Correctness:")
    test_data = [64, 34, 25, 12, 22, 11, 90]
    expected = sorted(test_data)
    
    for algo in [bubble_sort, insertion_sort, merge_sort, quicksort]:
        result = algo(test_data)
        status = "✓" if result == expected else "✗"
        logger.info(f"   {algo.__name__}: {status}")
    
    # Demo 2: Benchmark comparison
    logger.info("\n2. Benchmark Comparison (random data):")
    sizes = [100, 500, 1000]
    
    for algo in [insertion_sort, merge_sort, quicksort, sorted]:
        results = benchmark_sort(algo, sizes)
        logger.info(f"   {algo.__name__}:")
        for r in results:
            logger.info(f"      n={r.size}: {r.median:.6f}s")
    
    # Demo 3: Complexity estimation
    logger.info("\n3. Complexity Estimation:")
    
    for algo in [insertion_sort, merge_sort]:
        # Use smaller sizes for O(n²) to finish in reasonable time
        if "insertion" in algo.__name__:
            sizes = [100, 200, 400, 800]
        else:
            sizes = [1000, 2000, 4000, 8000]
        
        results = benchmark_sort(algo, sizes)
        slope, complexity = estimate_complexity_from_benchmarks(results)
        logger.info(f"   {algo.__name__}: slope={slope:.2f}, estimated {complexity}")
    
    # Demo 4: Crossover point
    logger.info("\n4. Crossover Point (insertion vs merge):")
    crossover = find_crossover_point(insertion_sort, merge_sort, 5, 200, 5)
    if crossover:
        logger.info(f"   Merge sort becomes faster at n ≈ {crossover}")
    else:
        logger.info("   No crossover found in range")


if __name__ == "__main__":
    demo_solutions()
