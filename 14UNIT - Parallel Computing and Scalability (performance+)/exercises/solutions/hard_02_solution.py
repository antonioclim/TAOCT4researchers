#!/usr/bin/env python3
"""
14UNIT Exercise Solution: Profiling and Optimisation (Hard)
© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar('T')


@dataclass
class ProfileResult:
    """Container for profiling results."""
    total_time: float
    total_calls: int
    hotspots: list[tuple[str, float, int]]
    
    def __str__(self) -> str:
        lines = [f"Total time: {self.total_time:.4f}s", f"Total calls: {self.total_calls:,}"]
        for func, t, calls in self.hotspots[:5]:
            lines.append(f"  {func}: {t:.4f}s ({calls:,} calls)")
        return '\n'.join(lines)


def profile_execution(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> tuple[T, ProfileResult]:
    """Profile function execution."""
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    
    # Extract stats
    total_time = 0.0
    total_calls = 0
    hotspots = []
    
    for key, value in stats.stats.items():
        filename, line, name = key
        cc, nc, tt, ct, callers = value
        total_time += tt
        total_calls += nc
        hotspots.append((f"{name} ({filename}:{line})", ct, nc))
    
    hotspots.sort(key=lambda x: x[1], reverse=True)
    
    return result, ProfileResult(total_time, total_calls, hotspots[:10])


def identify_hotspots(profile: ProfileResult, threshold_pct: float = 0.1) -> list[str]:
    """Identify functions consuming significant execution time."""
    threshold = profile.total_time * threshold_pct
    return [name for name, time_spent, _ in profile.hotspots if time_spent >= threshold]


def optimise_duplicates(items: list[int]) -> list[int]:
    """O(n) duplicate finding using set."""
    seen = set()
    duplicates = []
    duplicates_set = set()
    
    for item in items:
        if item in seen and item not in duplicates_set:
            duplicates.append(item)
            duplicates_set.add(item)
        seen.add(item)
    
    return duplicates


def optimise_matrix_ops(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Optimised matrix multiplication using NumPy."""
    return a @ b


def optimise_sum_of_squares(n: int) -> int:
    """Optimised sum of squares using formula."""
    # sum(i² for i in range(n)) = (n-1)*n*(2n-1)/6
    if n <= 0:
        return 0
    return (n - 1) * n * (2 * n - 1) // 6


@dataclass
class OptimisationReport:
    function_name: str
    slow_time: float
    optimised_time: float
    speedup: float
    results_match: bool
    
    def __str__(self) -> str:
        return (f"Optimisation: {self.function_name}\n"
                f"  Slow: {self.slow_time:.4f}s, Fast: {self.optimised_time:.4f}s\n"
                f"  Speedup: {self.speedup:.1f}x, Match: {self.results_match}")


def generate_optimisation_report(
    slow_func: Callable[..., T],
    fast_func: Callable[..., T],
    test_args: tuple[Any, ...],
    name: str = "function"
) -> OptimisationReport:
    """Compare slow and optimised implementations."""
    # Time slow
    start = time.perf_counter()
    slow_result = slow_func(*test_args)
    slow_time = time.perf_counter() - start
    
    # Time fast
    start = time.perf_counter()
    fast_result = fast_func(*test_args)
    fast_time = time.perf_counter() - start
    
    # Compare
    if isinstance(slow_result, np.ndarray):
        results_match = np.allclose(slow_result, fast_result)
    elif isinstance(slow_result, list):
        results_match = sorted(slow_result) == sorted(fast_result)
    else:
        results_match = slow_result == fast_result
    
    speedup = slow_time / max(fast_time, 1e-9)
    
    return OptimisationReport(name, slow_time, fast_time, speedup, results_match)


if __name__ == '__main__':
    data = list(range(500)) + list(range(250))
    print(optimise_duplicates(data))
