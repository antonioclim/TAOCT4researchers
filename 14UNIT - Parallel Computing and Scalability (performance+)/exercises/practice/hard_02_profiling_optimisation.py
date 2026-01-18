#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Profiling and Optimisation (Hard)
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ★★★★★ (Hard)
ESTIMATED TIME: 35 minutes
PREREQUISITES: cProfile, memory profiling, algorithmic analysis

LEARNING OBJECTIVES
───────────────────
- LO6: Profile code to identify performance bottlenecks
- LO6: Apply systematic optimisation based on profiling data

PROBLEM DESCRIPTION
───────────────────
Given inefficient implementations, use profiling tools to identify bottlenecks
and implement optimised versions. Document the improvement quantitatively.

TASKS
─────
1. Implement `profile_execution` - wrapper for cProfile analysis
2. Implement `identify_hotspots` - extract top time-consuming functions
3. Implement `optimise_duplicates` - optimise duplicate finding
4. Implement `optimise_matrix_ops` - optimise matrix operations
5. Implement `generate_optimisation_report` - document improvements

HINTS
─────
- Hint 1: Use pstats.Stats for parsing profile output
- Hint 2: Set-based lookup is O(1) vs list O(n)
- Hint 3: NumPy vectorisation beats Python loops

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
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
    hotspots: list[tuple[str, float, int]]  # (function, time, calls)
    
    def __str__(self) -> str:
        lines = [
            f"Total time: {self.total_time:.4f}s",
            f"Total calls: {self.total_calls:,}",
            "Top hotspots:"
        ]
        for func, time_spent, calls in self.hotspots[:5]:
            lines.append(f"  {func}: {time_spent:.4f}s ({calls:,} calls)")
        return '\n'.join(lines)


def profile_execution(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> tuple[T, ProfileResult]:
    """
    Profile function execution and return structured results.
    
    Args:
        func: Function to profile.
        *args: Positional arguments for function.
        **kwargs: Keyword arguments for function.
        
    Returns:
        Tuple of (function result, ProfileResult).
        
    Example:
        >>> def slow_func(n):
        ...     return sum(i**2 for i in range(n))
        >>> result, profile = profile_execution(slow_func, 100000)
        >>> profile.total_time > 0
        True
    """
    # TODO: Implement this function
    # 1. Create cProfile.Profile()
    # 2. Enable, run function, disable
    # 3. Extract stats using pstats.Stats
    # 4. Build ProfileResult with hotspots
    raise NotImplementedError("Implement profile_execution")


def identify_hotspots(
    profile: ProfileResult,
    threshold_pct: float = 0.1
) -> list[str]:
    """
    Identify functions consuming significant execution time.
    
    Args:
        profile: Profiling results.
        threshold_pct: Minimum percentage of total time (0.0-1.0).
        
    Returns:
        List of function names exceeding threshold.
        
    Example:
        >>> hotspots = identify_hotspots(profile, threshold_pct=0.05)
        >>> len(hotspots) > 0
        True
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement identify_hotspots")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

def find_duplicates_slow(items: list[int]) -> list[int]:
    """
    SLOW: Find duplicate items using nested loops.
    Time complexity: O(n²)
    """
    duplicates = []
    for i, item in enumerate(items):
        for j, other in enumerate(items):
            if i != j and item == other and item not in duplicates:
                duplicates.append(item)
    return duplicates


def optimise_duplicates(items: list[int]) -> list[int]:
    """
    OPTIMISED: Find duplicate items efficiently.
    
    Your implementation should achieve O(n) time complexity
    using appropriate data structures.
    
    Args:
        items: List of integers.
        
    Returns:
        List of items that appear more than once (in first-occurrence order).
        
    Example:
        >>> optimise_duplicates([1, 2, 3, 2, 4, 3, 5])
        [2, 3]
    """
    # TODO: Implement O(n) solution using set/dict
    raise NotImplementedError("Implement optimise_duplicates")


def matrix_multiply_slow(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    SLOW: Matrix multiplication using Python loops.
    Time complexity: O(n³) with high constant factor.
    """
    n, m = a.shape
    m2, p = b.shape
    assert m == m2
    
    result = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i, j] += a[i, k] * b[k, j]
    return result


def optimise_matrix_ops(
    a: NDArray[np.float64],
    b: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    OPTIMISED: Matrix multiplication using NumPy.
    
    Your implementation should use NumPy's optimised routines.
    
    Args:
        a: First matrix (n × m).
        b: Second matrix (m × p).
        
    Returns:
        Product matrix (n × p).
        
    Example:
        >>> a = np.array([[1, 2], [3, 4]])
        >>> b = np.array([[5, 6], [7, 8]])
        >>> optimise_matrix_ops(a, b)
        array([[19, 22],
               [43, 50]])
    """
    # TODO: Implement using NumPy
    raise NotImplementedError("Implement optimise_matrix_ops")


def sum_of_squares_slow(n: int) -> int:
    """
    SLOW: Sum of squares using loop with function call overhead.
    """
    def square(x: int) -> int:
        return x * x
    
    total = 0
    for i in range(n):
        total += square(i)
    return total


def optimise_sum_of_squares(n: int) -> int:
    """
    OPTIMISED: Sum of squares efficiently.
    
    Use mathematical formula or NumPy vectorisation.
    Formula: sum(i² for i in range(n)) = n(n-1)(2n-1)/6
    
    Args:
        n: Upper limit (exclusive).
        
    Returns:
        Sum of squares from 0 to n-1.
        
    Example:
        >>> optimise_sum_of_squares(10)
        285
    """
    # TODO: Implement using formula or NumPy
    raise NotImplementedError("Implement optimise_sum_of_squares")


@dataclass
class OptimisationReport:
    """Report comparing slow vs optimised implementations."""
    function_name: str
    slow_time: float
    optimised_time: float
    speedup: float
    results_match: bool
    
    def __str__(self) -> str:
        return (
            f"Optimisation Report: {self.function_name}\n"
            f"  Slow: {self.slow_time:.4f}s\n"
            f"  Optimised: {self.optimised_time:.4f}s\n"
            f"  Speedup: {self.speedup:.1f}x\n"
            f"  Results match: {self.results_match}"
        )


def generate_optimisation_report(
    slow_func: Callable[..., T],
    fast_func: Callable[..., T],
    test_args: tuple[Any, ...],
    name: str = "function"
) -> OptimisationReport:
    """
    Compare slow and optimised implementations.
    
    Args:
        slow_func: Original slow implementation.
        fast_func: Optimised implementation.
        test_args: Arguments for testing.
        name: Name for the report.
        
    Returns:
        OptimisationReport with timing comparison.
        
    Example:
        >>> report = generate_optimisation_report(
        ...     find_duplicates_slow,
        ...     optimise_duplicates,
        ...     ([1, 2, 3, 2, 4, 3],),
        ...     "duplicates"
        ... )
        >>> report.speedup > 1.0
        True
    """
    # TODO: Implement this function
    # 1. Time slow function
    # 2. Time fast function
    # 3. Compare results for correctness
    # 4. Calculate speedup
    raise NotImplementedError("Implement generate_optimisation_report")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_profile_execution() -> None:
    """Test profiling wrapper."""
    def sample(n: int) -> int:
        return sum(i for i in range(n))
    
    result, profile = profile_execution(sample, 10000)
    assert result == sum(range(10000))
    assert profile.total_time > 0
    print("✓ profile_execution tests passed")


def test_optimise_duplicates() -> None:
    """Test duplicate finding optimisation."""
    test_data = [1, 2, 3, 2, 4, 3, 5, 1]
    slow_result = sorted(find_duplicates_slow(test_data))
    fast_result = sorted(optimise_duplicates(test_data))
    assert slow_result == fast_result
    print("✓ optimise_duplicates tests passed")


def test_optimise_matrix_ops() -> None:
    """Test matrix multiplication optimisation."""
    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    slow_result = matrix_multiply_slow(a, b)
    fast_result = optimise_matrix_ops(a, b)
    assert np.allclose(slow_result, fast_result)
    print("✓ optimise_matrix_ops tests passed")


def test_optimise_sum_of_squares() -> None:
    """Test sum of squares optimisation."""
    for n in [10, 100, 1000]:
        slow = sum_of_squares_slow(n)
        fast = optimise_sum_of_squares(n)
        assert slow == fast, f"Mismatch for n={n}: {slow} vs {fast}"
    print("✓ optimise_sum_of_squares tests passed")


def test_generate_report() -> None:
    """Test report generation."""
    data = list(range(1000)) + list(range(500))  # Has duplicates
    report = generate_optimisation_report(
        find_duplicates_slow,
        optimise_duplicates,
        (data,),
        "duplicates"
    )
    assert report.results_match
    assert report.speedup >= 1.0
    print(f"✓ Report generated: {report.speedup:.1f}x speedup")


def main() -> None:
    """Run all tests."""
    print("Running hard_02_profiling_optimisation tests...")
    print("-" * 50)
    
    try:
        test_profile_execution()
        test_optimise_duplicates()
        test_optimise_matrix_ops()
        test_optimise_sum_of_squares()
        test_generate_report()
        print("-" * 50)
        print("All tests passed! ✓")
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}")
    except AssertionError as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    main()
