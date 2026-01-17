#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Medium 01 — Sorting Algorithm Benchmark
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Sorting algorithms have different performance characteristics. This exercise
develops skills in implementing and benchmarking multiple algorithms.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Implement basic sorting algorithms correctly
2. Benchmark algorithms with proper methodology
3. Compare empirical results with theoretical complexity

ESTIMATED TIME
──────────────
40 minutes

DIFFICULTY
──────────
⭐⭐ Medium (2/3)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

import gc
import random
import time
from dataclasses import dataclass
from typing import Callable


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Implement Sorting Algorithms
# ═══════════════════════════════════════════════════════════════════════════════

def bubble_sort(data: list[int]) -> list[int]:
    """
    Sort using bubble sort algorithm.

    Time complexity: O(n²) average and worst case
    Space complexity: O(1) — in-place

    Args:
        data: List to sort (will be modified).

    Returns:
        Sorted list (same reference as input).
    """
    # TODO: Implement bubble sort
    # Repeatedly swap adjacent elements if in wrong order
    raise NotImplementedError("Implement bubble sort")


def insertion_sort(data: list[int]) -> list[int]:
    """
    Sort using insertion sort algorithm.

    Time complexity: O(n²) average and worst, O(n) best (nearly sorted)
    Space complexity: O(1) — in-place

    Args:
        data: List to sort (will be modified).

    Returns:
        Sorted list (same reference as input).
    """
    # TODO: Implement insertion sort
    # Build sorted array one element at a time
    raise NotImplementedError("Implement insertion sort")


def merge_sort(data: list[int]) -> list[int]:
    """
    Sort using merge sort algorithm.

    Time complexity: O(n log n) all cases
    Space complexity: O(n) — requires auxiliary space

    Args:
        data: List to sort.

    Returns:
        New sorted list.
    """
    # TODO: Implement merge sort
    # Divide, recursively sort, merge
    raise NotImplementedError("Implement merge sort")


def quicksort(data: list[int]) -> list[int]:
    """
    Sort using quicksort algorithm.

    Time complexity: O(n log n) average, O(n²) worst case
    Space complexity: O(log n) stack space

    Args:
        data: List to sort (will be modified).

    Returns:
        Sorted list (same reference as input).
    """
    # TODO: Implement quicksort
    # Choose pivot, partition, recursively sort
    raise NotImplementedError("Implement quicksort")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Benchmarking Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Results from benchmarking an algorithm."""

    algorithm: str
    size: int
    times: list[float]

    @property
    def mean(self) -> float:
        """Calculate mean time."""
        return sum(self.times) / len(self.times)

    @property
    def median(self) -> float:
        """Calculate median time."""
        sorted_times = sorted(self.times)
        n = len(sorted_times)
        if n % 2 == 0:
            return (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2
        return sorted_times[n // 2]


def benchmark_sort(
    sort_func: Callable[[list[int]], list[int]],
    size: int,
    runs: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """
    Benchmark a sorting algorithm.

    Args:
        sort_func: Sorting function to benchmark.
        size: Size of data to sort.
        runs: Number of measured runs.
        warmup: Number of warmup runs (discarded).

    Returns:
        BenchmarkResult with timing data.
    """
    # TODO: Implement proper benchmarking
    # 1. Run warmup iterations (discard)
    # 2. For each measured run:
    #    a. Generate fresh random data
    #    b. Disable GC
    #    c. Time the sort
    #    d. Re-enable GC
    # 3. Return BenchmarkResult
    raise NotImplementedError("Implement benchmark_sort")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Scaling Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_scaling_analysis(
    algorithms: dict[str, Callable[[list[int]], list[int]]],
    sizes: list[int],
    runs: int = 5,
) -> dict[str, list[BenchmarkResult]]:
    """
    Run scaling analysis on multiple algorithms.

    Args:
        algorithms: Dictionary mapping name to sort function.
        sizes: List of input sizes to test.
        runs: Number of runs per size.

    Returns:
        Dictionary mapping algorithm name to list of BenchmarkResults.
    """
    # TODO: For each algorithm, benchmark at each size
    raise NotImplementedError("Implement run_scaling_analysis")


def estimate_complexity(results: list[BenchmarkResult]) -> tuple[float, float]:
    """
    Estimate complexity exponent from benchmark results using log-log regression.

    Args:
        results: List of BenchmarkResults at different sizes.

    Returns:
        Tuple of (exponent, r_squared).
    """
    import math

    # TODO: Implement log-log regression
    # 1. Extract sizes and median times
    # 2. Take log of both
    # 3. Fit line using least squares
    # 4. Return slope (exponent) and R²
    raise NotImplementedError("Implement estimate_complexity")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Generate Report
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(
    all_results: dict[str, list[BenchmarkResult]],
) -> str:
    """
    Generate a markdown report from benchmark results.

    Args:
        all_results: Results from run_scaling_analysis.

    Returns:
        Markdown formatted report string.
    """
    # TODO: Generate a nice report including:
    # - Table of times for each algorithm and size
    # - Estimated complexity for each algorithm
    # - Analysis of which algorithm is best for each size range
    raise NotImplementedError("Implement generate_report")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _test_sorting_correctness() -> None:
    """Test that sorting algorithms produce correct results."""
    print("Testing sorting correctness...\n")

    test_cases = [
        [],
        [1],
        [2, 1],
        [3, 1, 2],
        [5, 2, 8, 1, 9, 3],
        list(range(10, 0, -1)),  # Reverse sorted
        [1, 1, 1, 1, 1],  # All same
    ]

    algorithms = {
        "bubble_sort": bubble_sort,
        "insertion_sort": insertion_sort,
        "merge_sort": merge_sort,
        "quicksort": quicksort,
    }

    for name, func in algorithms.items():
        print(f"  {name}:")
        all_pass = True
        for tc in test_cases:
            try:
                data = tc.copy()
                result = func(data)
                expected = sorted(tc)
                if result != expected:
                    print(f"    ✗ {tc} → {result} (expected {expected})")
                    all_pass = False
            except NotImplementedError:
                print(f"    ✗ Not implemented")
                all_pass = False
                break
            except Exception as e:
                print(f"    ✗ {tc} raised {e}")
                all_pass = False

        if all_pass:
            print(f"    ✓ All test cases passed")


def _test_benchmarking() -> None:
    """Test benchmarking infrastructure."""
    print("\nTesting benchmarking...\n")

    # Use Python's built-in sort as reference
    try:
        result = benchmark_sort(sorted, 1000, runs=3, warmup=1)
        print(f"  ✓ benchmark_sort: mean={result.mean:.6f}s, median={result.median:.6f}s")
    except NotImplementedError:
        print("  ✗ benchmark_sort: Not implemented")


def _test_scaling() -> None:
    """Test scaling analysis."""
    print("\nTesting scaling analysis...\n")

    try:
        algorithms = {"python_sorted": sorted}
        sizes = [100, 500, 1000]
        results = run_scaling_analysis(algorithms, sizes, runs=2)
        print(f"  ✓ run_scaling_analysis completed")

        for name, res_list in results.items():
            exp, r2 = estimate_complexity(res_list)
            print(f"    {name}: exponent={exp:.2f}, R²={r2:.3f}")
    except NotImplementedError:
        print("  ✗ run_scaling_analysis or estimate_complexity: Not implemented")


if __name__ == "__main__":
    _test_sorting_correctness()
    _test_benchmarking()
    _test_scaling()

    print("\n" + "=" * 60)
    print("Implement all functions, then compare your sorts to Python's!")
