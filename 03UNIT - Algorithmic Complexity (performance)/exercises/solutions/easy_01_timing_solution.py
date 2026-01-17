#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTIONS: easy_01_timing.py
Week 3, Practice Exercise Solutions
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
import statistics
from typing import Any, Callable
from contextlib import contextmanager


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Simple Timer Context Manager
# ═══════════════════════════════════════════════════════════════════════════════

@contextmanager
def simple_timer():
    """
    SOLUTION: Context manager that measures elapsed time.
    """
    start = time.perf_counter()
    elapsed = [0.0]  # Mutable container for result
    
    try:
        yield elapsed
    finally:
        elapsed[0] = time.perf_counter() - start


# Usage example:
# with simple_timer() as elapsed:
#     some_operation()
# print(f"Elapsed: {elapsed[0]} seconds")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Timing Decorator
# ═══════════════════════════════════════════════════════════════════════════════

def timing_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    SOLUTION: Decorator that prints execution time.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} executed in {elapsed:.6f} seconds")
        return result
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Multiple Timing Runs
# ═══════════════════════════════════════════════════════════════════════════════

def measure_multiple(
    func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    runs: int = 10
) -> dict[str, float]:
    """
    SOLUTION: Run function multiple times and report statistics.
    """
    if kwargs is None:
        kwargs = {}
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "runs": runs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Compare Function Performance
# ═══════════════════════════════════════════════════════════════════════════════

def compare_functions(
    functions: list[Callable[[list[int]], Any]],
    data: list[int],
    runs: int = 5
) -> dict[str, dict[str, float]]:
    """
    SOLUTION: Compare execution times of multiple functions.
    """
    results = {}
    
    for func in functions:
        times = []
        for _ in range(runs):
            data_copy = data.copy()  # Use fresh copy each time
            start = time.perf_counter()
            func(data_copy)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        results[func.__name__] = {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Timing with Warmup
# ═══════════════════════════════════════════════════════════════════════════════

def measure_with_warmup(
    func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    warmup_runs: int = 3,
    measured_runs: int = 10
) -> dict[str, Any]:
    """
    SOLUTION: Time with warmup runs (not counted in measurements).
    """
    # Warmup phase
    for _ in range(warmup_runs):
        func(*args)
    
    # Measurement phase
    times = []
    for _ in range(measured_runs):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo_solutions() -> None:
    """Demonstrate all solutions."""
    import random
    
    print("=" * 60)
    print("TIMING SOLUTIONS DEMONSTRATION")
    print("=" * 60)
    
    # Test data
    data = [random.randint(0, 10000) for _ in range(1000)]
    
    # Demo 1: Simple timer
    print("\n1. Simple Timer:")
    with simple_timer() as elapsed:
        sorted(data)
    print(f"   Sorting took {elapsed[0]:.6f} seconds")
    
    # Demo 2: Timing decorator
    print("\n2. Timing Decorator:")
    
    @timing_decorator
    def example_sort(arr: list[int]) -> list[int]:
        return sorted(arr)
    
    _ = example_sort(data)
    
    # Demo 3: Multiple runs
    print("\n3. Multiple Runs Statistics:")
    stats = measure_multiple(sorted, (data,), runs=10)
    print(f"   Mean: {stats['mean']:.6f}s, Std: {stats['std']:.6f}s")
    
    # Demo 4: Compare functions
    print("\n4. Function Comparison:")
    
    def bubble_sort(arr: list[int]) -> list[int]:
        arr = arr.copy()
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    small_data = data[:100]  # Smaller for bubble sort
    results = compare_functions([bubble_sort, sorted], small_data)
    for name, stats in results.items():
        print(f"   {name}: {stats['median']:.6f}s")
    
    # Demo 5: Warmup measurement
    print("\n5. Measurement with Warmup:")
    stats = measure_with_warmup(sorted, (data,), warmup_runs=3, measured_runs=10)
    print(f"   After {stats['warmup_runs']} warmup runs:")
    print(f"   Median: {stats['median']:.6f}s")


if __name__ == "__main__":
    demo_solutions()
