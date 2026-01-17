#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
03UNIT, Lab 01: Benchmark Suite — Profiling and Optimisation
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Performance measurement is fundamental to computational research. Whether
comparing sorting algorithms, evaluating machine learning models or optimising
simulation code, rigorous benchmarking requires statistical methodology beyond
simple timing. This lab provides a comprehensive framework for measuring,
analysing and comparing algorithm performance.

HISTORICAL MOTIVATION
─────────────────────
In 1962, Tony Hoare proposed the Quicksort algorithm. Theory predicted O(n log n)
average-case performance, yet early implementations were disappointingly slow.
The culprit? Cache misses. Modern hardware has a memory hierarchy that
profoundly affects real-world performance:

    - Registers: ~1 cycle latency
    - L1 Cache: ~4 cycles latency
    - L2 Cache: ~12 cycles latency
    - L3 Cache: ~40 cycles latency
    - RAM: ~200+ cycles latency

A theoretically optimal algorithm may perform poorly in practice if it accesses
memory in a cache-unfriendly pattern. Understanding this gap between theory and
practice is essential for computational researchers.

PREREQUISITES
─────────────
- 02UNIT: Object-oriented design, Strategy pattern
- Python: Functions, classes, decorators, context managers
- Mathematics: Logarithms, linear regression basics

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement precise timing measurements with statistical rigour
2. Profile code to identify performance bottlenecks
3. Compare algorithm implementations across multiple dimensions
4. Estimate empirical complexity using log-log regression

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 45 minutes
- Total: 75 minutes

DEPENDENCIES
────────────
- numpy>=1.24
- numba>=0.58 (optional, for JIT compilation)

MATHEMATICAL BACKGROUND
───────────────────────
For empirical complexity analysis, we use the power-law relationship:

    T(n) = c · n^k

Taking logarithms of both sides:

    log T(n) = log c + k · log n

On a log-log plot, this becomes a straight line with slope k (the exponent).
We can therefore estimate k through linear regression in log-log space:

    T(n) = c · n^k  →  log T = log c + k · log n

The slope of the regression line gives the complexity exponent.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import random
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from io import StringIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Protocol,
    Sequence,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Optional: advanced profiling
try:
    import cProfile
    import pstats
    HAS_CPROFILE = True
except ImportError:
    HAS_CPROFILE = False

# Optional: JIT compilation for performance
try:
    from numba import jit, prange
    HAS_NUMBA = True
    logger.debug("Numba JIT compilation available")
except ImportError:
    HAS_NUMBA = False
    logger.debug("Numba not available; using pure Python fallbacks")
    
    # Placeholder decorator when Numba unavailable
    def jit(*args, **kwargs):
        """Placeholder decorator when Numba is not installed."""
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range

# Type aliases
T = TypeVar('T')
SortableArray = list[float] | NDArray[np.float64]
DataGenerator = Callable[[int], list[float]]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BENCHMARK INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """
    Result of an individual benchmark measurement.
    
    Stores comprehensive statistics for rigorous analysis:
    - Mean and median for central tendency comparison
    - Standard deviation for variability assessment
    - Min/max for outlier detection
    - All individual timings for subsequent analysis
    
    Attributes:
        name: Name of the algorithm/function tested.
        n: Input size used for the benchmark.
        times: List of individual timing measurements.
        unit: Time unit ('s', 'ms', 'μs', 'ns').
    
    Example:
        >>> result = BenchmarkResult("quicksort", n=10000, times=[1.2, 1.1, 1.3])
        >>> result.median
        1.2
        >>> result.coefficient_of_variation
        0.0833...
    """
    name: str
    n: int
    times: list[float] = field(default_factory=list)
    unit: str = 'ms'
    
    @property
    def mean(self) -> float:
        """Arithmetic mean of timing measurements."""
        return statistics.mean(self.times) if self.times else 0.0
    
    @property
    def median(self) -> float:
        """Median timing (more robust to outliers than mean)."""
        return statistics.median(self.times) if self.times else 0.0
    
    @property
    def std(self) -> float:
        """Standard deviation of timing measurements."""
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0
    
    @property
    def min_time(self) -> float:
        """Minimum timing (best case observation)."""
        return min(self.times) if self.times else 0.0
    
    @property
    def max_time(self) -> float:
        """Maximum timing (worst case / possible outlier)."""
        return max(self.times) if self.times else 0.0
    
    @property
    def coefficient_of_variation(self) -> float:
        """
        Coefficient of variation (CV = std/mean).
        
        Measures relative variability; CV < 0.1 indicates stable measurements.
        """
        return self.std / self.mean if self.mean > 0 else 0.0
    
    @property
    def iqr(self) -> float:
        """Interquartile range (Q3 - Q1), robust measure of spread."""
        if len(self.times) < 4:
            return self.std
        sorted_times = sorted(self.times)
        q1_idx = len(sorted_times) // 4
        q3_idx = 3 * len(sorted_times) // 4
        return sorted_times[q3_idx] - sorted_times[q1_idx]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'name': self.name,
            'n': self.n,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'min': self.min_time,
            'max': self.max_time,
            'iqr': self.iqr,
            'cv': self.coefficient_of_variation,
            'runs': len(self.times),
            'unit': self.unit,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.name}(n={self.n:,}): "
            f"{self.median:.3f} ± {self.std:.3f} {self.unit} "
            f"(CV={self.coefficient_of_variation:.1%})"
        )


class Timer:
    """Simple timer class for use with context manager."""
    elapsed: float = 0.0


@contextmanager
def timer() -> Iterator[Timer]:
    """
    Context manager for precise time measurement.
    
    Uses time.perf_counter() which provides ~100ns resolution and
    is unaffected by system clock adjustments.
    
    Yields:
        Timer object with `elapsed` attribute set upon exit.
    
    Example:
        >>> with timer() as t:
        ...     expensive_operation()
        >>> logger.info(f"Operation took {t.elapsed:.3f}s")
    """
    t = Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    runs: int = 10,
    warmup: int = 2,
    name: str | None = None,
    n: int = 0,
    disable_gc: bool = True,
    **kwargs: Any
) -> BenchmarkResult:
    """
    Execute a benchmark on a function with statistical rigour.
    
    Protocol:
        1. Execute warmup runs (for JIT compilation, cache warming)
        2. Force garbage collection
        3. Optionally disable GC during measurement
        4. Take multiple measurements
        5. Return comprehensive statistics
    
    Args:
        func: Function to benchmark.
        *args: Positional arguments for func.
        runs: Number of measurement runs (minimum 10 recommended).
        warmup: Number of warmup runs (not measured).
        name: Name for reporting (default: func.__name__).
        n: Input size for reporting.
        disable_gc: Whether to disable GC during measurement.
        **kwargs: Keyword arguments for func.
    
    Returns:
        BenchmarkResult with comprehensive statistics.
    
    Example:
        >>> data = list(range(10000))
        >>> result = benchmark(sorted, data, runs=20, n=10000)
        >>> logger.info(f"Median: {result.median:.3f} ms")
    """
    name = name or getattr(func, '__name__', 'anonymous')
    logger.debug(f"Benchmarking {name} with n={n}, runs={runs}")
    
    # Warmup: allows JIT compilation and cache warming
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Force garbage collection for consistent measurements
    gc.collect()
    
    if disable_gc:
        gc.disable()
    
    times: list[float] = []
    try:
        for run_idx in range(runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to milliseconds
            logger.debug(f"  Run {run_idx + 1}/{runs}: {elapsed * 1000:.3f} ms")
    finally:
        if disable_gc:
            gc.enable()
    
    result = BenchmarkResult(name=name, n=n, times=times, unit='ms')
    logger.debug(f"Completed: {result}")
    return result


def benchmark_with_setup(
    func: Callable[..., Any],
    setup: Callable[[], tuple[tuple, dict]],
    runs: int = 10,
    warmup: int = 2,
    name: str | None = None,
    n: int = 0,
) -> BenchmarkResult:
    """
    Benchmark with fresh setup before each run.
    
    Useful when the function modifies its input (e.g., in-place sorting).
    
    Args:
        func: Function to benchmark.
        setup: Callable that returns (args, kwargs) for each run.
        runs: Number of measurement runs.
        warmup: Number of warmup runs.
        name: Name for reporting.
        n: Input size for reporting.
    
    Returns:
        BenchmarkResult with statistics.
    """
    name = name or getattr(func, '__name__', 'anonymous')
    logger.debug(f"Benchmarking {name} with setup, n={n}")
    
    # Warmup
    for _ in range(warmup):
        args, kwargs = setup()
        func(*args, **kwargs)
    
    gc.collect()
    gc.disable()
    
    times: list[float] = []
    try:
        for _ in range(runs):
            args, kwargs = setup()
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)
    finally:
        gc.enable()
    
    return BenchmarkResult(name=name, n=n, times=times, unit='ms')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SORTING ALGORITHMS FOR COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def bubble_sort(arr: list[float]) -> list[float]:
    """
    Bubble Sort — the simplest (and slowest) sorting algorithm.
    
    Complexity:
        Time: O(n²) worst, O(n²) average, O(n) best (already sorted)
        Space: O(1) — in-place
    
    Algorithm:
        Repeatedly traverses the list, comparing adjacent elements and
        swapping them if they are in the wrong order. The largest
        element "bubbles up" to its correct position each pass.
    
    Why it is slow:
        - Many redundant comparisons
        - Many swaps (memory writes)
        - Not cache-friendly
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list (original unchanged).
    
    Example:
        >>> bubble_sort([3, 1, 4, 1, 5])
        [1, 1, 3, 4, 5]
    """
    arr = arr.copy()  # Do not modify the original
    n = len(arr)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        
        # Optimisation: if no swap occurred, list is sorted
        if not swapped:
            break
    
    return arr


def insertion_sort(arr: list[float]) -> list[float]:
    """
    Insertion Sort — efficient for small or nearly-sorted lists.
    
    Complexity:
        Time: O(n²) worst, O(n²) average, O(n) best
        Space: O(1)
    
    Algorithm:
        Builds the sorted list one element at a time by inserting
        each element into its correct position among the already
        sorted elements.
    
    Advantages:
        - Simple to implement
        - Efficient for small lists (n < 50)
        - Adaptive: O(n) for nearly-sorted input
        - Stable (preserves order of equal elements)
        - In-place (O(1) extra memory)
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list.
    """
    arr = arr.copy()
    n = len(arr)
    
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        
        # Move elements greater than key one position to the right
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = key
    
    return arr


def merge_sort(arr: list[float]) -> list[float]:
    """
    Merge Sort — divide and conquer, guaranteed O(n log n).
    
    Complexity:
        Time: O(n log n) in ALL cases
        Space: O(n) — requires auxiliary memory
    
    Algorithm:
        1. Divide the list into two halves
        2. Recursively sort each half
        3. Merge the two sorted halves
    
    Complexity demonstration (Master Theorem):
        T(n) = 2·T(n/2) + O(n)
        a=2, b=2, f(n)=n, log_b(a)=1
        f(n) = Θ(n^log_b(a)) → Case 2 → T(n) = Θ(n log n)
    
    Advantages:
        - Guaranteed O(n log n)
        - Stable
        - Parallelisable (two halves are independent)
    
    Disadvantages:
        - O(n) extra memory
        - Not cache-friendly (many allocations)
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list.
    """
    if len(arr) <= 1:
        return arr.copy()
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return _merge(left, right)


def _merge(left: list[float], right: list[float]) -> list[float]:
    """Merge two sorted lists into a single sorted list."""
    result: list[float] = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    # Append the remainder
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result


def quicksort(arr: list[float]) -> list[float]:
    """
    Quicksort — fast on average, but O(n²) worst case.
    
    Complexity:
        Time: O(n²) worst (poor pivot), O(n log n) average
        Space: O(log n) for stack (recursion depth)
    
    Algorithm:
        1. Choose a pivot element
        2. Partition: elements < pivot | pivot | elements > pivot
        3. Recursively sort the two partitions
    
    Implemented optimisations:
        - Median-of-three pivot selection
        - Cutoff to insertion sort for small n
        - Tail call elimination (via while loop)
    
    Why it is fast in practice:
        - Sequential memory access (cache-friendly)
        - In-place (minimises allocations)
        - Parallelisable
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list.
    """
    arr = arr.copy()
    _quicksort_inplace(arr, 0, len(arr) - 1)
    return arr


def _quicksort_inplace(arr: list[float], low: int, high: int) -> None:
    """In-place implementation of Quicksort."""
    while low < high:
        # Cutoff to insertion sort for small sublists
        if high - low < 16:
            _insertion_sort_partial(arr, low, high)
            break
        
        # Median-of-three pivot selection
        mid = (low + high) // 2
        if arr[low] > arr[mid]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[low] > arr[high]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[mid] > arr[high]:
            arr[mid], arr[high] = arr[high], arr[mid]
        
        # Pivot is now median; place it in second-to-last position
        arr[mid], arr[high - 1] = arr[high - 1], arr[mid]
        pivot = arr[high - 1]
        
        # Partitioning
        i = low
        j = high - 1
        while True:
            i += 1
            while arr[i] < pivot:
                i += 1
            j -= 1
            while arr[j] > pivot:
                j -= 1
            if i >= j:
                break
            arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in its final position
        arr[i], arr[high - 1] = arr[high - 1], arr[i]
        
        # Recurse on smaller partition; loop on larger (tail call elimination)
        if i - low < high - i:
            _quicksort_inplace(arr, low, i - 1)
            low = i + 1
        else:
            _quicksort_inplace(arr, i + 1, high)
            high = i - 1


def _insertion_sort_partial(arr: list[float], low: int, high: int) -> None:
    """Insertion sort on a sublist."""
    for i in range(low + 1, high + 1):
        key = arr[i]
        j = i - 1
        while j >= low and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def heapsort(arr: list[float]) -> list[float]:
    """
    Heapsort — guaranteed O(n log n) and in-place.
    
    Complexity:
        Time: O(n log n) in ALL cases
        Space: O(1) — in-place
    
    Algorithm:
        1. Build a max-heap from the array (heapify)
        2. Repeatedly extract the maximum and place at the end
    
    Properties:
        - Not stable
        - Guaranteed O(n log n) like merge sort
        - In-place like quicksort
        - Poor cache locality (jumping through heap)
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list.
    """
    arr = arr.copy()
    n = len(arr)
    
    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    
    # Extract elements one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move max to end
        _heapify(arr, i, 0)
    
    return arr


def _heapify(arr: list[float], heap_size: int, root: int) -> None:
    """Maintain heap property for subtree rooted at index root."""
    largest = root
    left = 2 * root + 1
    right = 2 * root + 2
    
    if left < heap_size and arr[left] > arr[largest]:
        largest = left
    
    if right < heap_size and arr[right] > arr[largest]:
        largest = right
    
    if largest != root:
        arr[root], arr[largest] = arr[largest], arr[root]
        _heapify(arr, heap_size, largest)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OPTIMISED VERSIONS (NUMPY, NUMBA)
# ═══════════════════════════════════════════════════════════════════════════════

def numpy_sort(arr: list[float]) -> NDArray[np.float64]:
    """
    Sorting using NumPy.
    
    NumPy uses Introsort (hybrid quicksort/heapsort/insertion) implemented
    in C, which is substantially faster than pure Python.
    
    Overhead:
        - Conversion from list to ndarray
        - Memory allocation for result
    
    For large lists (n > 1000), the overhead is negligible compared
    to the sorting time.
    
    Args:
        arr: List to sort.
    
    Returns:
        Sorted NumPy array.
    """
    np_arr = np.array(arr, dtype=np.float64)
    return np.sort(np_arr)


if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def numba_quicksort(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Quicksort compiled with Numba JIT.
        
        Numba compiles Python code to machine code via LLVM, providing
        performance approaching that of C.
        
        Notes:
            - First run is slow (JIT compilation)
            - Subsequent runs are fast
            - Works only with NumPy arrays
        
        Args:
            arr: NumPy array to sort.
        
        Returns:
            New sorted NumPy array.
        """
        arr = arr.copy()
        _numba_qs(arr, 0, len(arr) - 1)
        return arr
    
    @jit(nopython=True, cache=True)
    def _numba_qs(arr: NDArray[np.float64], low: int, high: int) -> None:
        """Recursive helper for Numba quicksort."""
        if low < high:
            pivot_idx = _numba_partition(arr, low, high)
            _numba_qs(arr, low, pivot_idx - 1)
            _numba_qs(arr, pivot_idx + 1, high)
    
    @jit(nopython=True, cache=True)
    def _numba_partition(arr: NDArray[np.float64], low: int, high: int) -> int:
        """Partitioning for Numba quicksort."""
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
else:
    def numba_quicksort(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Fallback when Numba is not available."""
        logger.warning("Numba not available; using NumPy sort as fallback")
        return np.sort(arr)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPLEXITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_complexity(
    sizes: Sequence[int],
    times: Sequence[float]
) -> tuple[float, float, str]:
    """
    Estimate empirical complexity via linear regression in log-log space.
    
    Method:
        T(n) = c · n^k
        log(T) = log(c) + k · log(n)
        
        Perform linear regression: y = a + b·x
        where y = log(T), x = log(n), b = k (the exponent)
    
    Args:
        sizes: List of input sizes n.
        times: List of corresponding timing measurements.
    
    Returns:
        Tuple of (exponent, constant, complexity_class).
    
    Example:
        >>> sizes = [100, 1000, 10000]
        >>> times = [0.01, 1.0, 100.0]  # Grows with n²
        >>> exp, c, cls = estimate_complexity(sizes, times)
        >>> cls
        'O(n²)'
    """
    # Remove zero or negative values
    valid = [(n, t) for n, t in zip(sizes, times) if n > 0 and t > 0]
    if len(valid) < 2:
        logger.warning("Insufficient data for complexity estimation")
        return 0.0, 0.0, "insufficient data"
    
    sizes_clean, times_clean = zip(*valid)
    
    # Log-transform
    log_n = np.log(np.array(sizes_clean, dtype=np.float64))
    log_t = np.log(np.array(times_clean, dtype=np.float64))
    
    # Linear regression: log_t = a + b * log_n
    # Using direct least squares formulae
    n_points = len(log_n)
    sum_x = np.sum(log_n)
    sum_y = np.sum(log_t)
    sum_xy = np.sum(log_n * log_t)
    sum_x2 = np.sum(log_n ** 2)
    
    denominator = n_points * sum_x2 - sum_x ** 2
    if abs(denominator) < 1e-10:
        logger.warning("Degenerate data for regression")
        return 0.0, 0.0, "degenerate data"
    
    # Slope (exponent)
    b = (n_points * sum_xy - sum_x * sum_y) / denominator
    
    # Intercept
    a = (sum_y - b * sum_x) / n_points
    c = np.exp(a)
    
    # Compute R² for fit quality
    y_pred = a + b * log_n
    ss_res = np.sum((log_t - y_pred) ** 2)
    ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    logger.debug(f"Regression: exponent={b:.3f}, R²={r_squared:.4f}")
    
    # Classification
    if b < 0.1:
        complexity = "O(1)"
    elif b < 0.6:
        complexity = "O(log n)"
    elif b < 1.2:
        complexity = "O(n)"
    elif b < 1.6:
        complexity = "O(n log n)"
    elif b < 2.2:
        complexity = "O(n²)"
    elif b < 3.2:
        complexity = "O(n³)"
    else:
        complexity = f"O(n^{b:.1f})"
    
    return float(b), float(c), complexity


def compute_speedup(baseline: BenchmarkResult, optimised: BenchmarkResult) -> float:
    """
    Calculate speedup between two implementations.
    
    Speedup = T_baseline / T_optimised
    
    Speedup > 1 means the optimised version is faster.
    Speedup = 2 means twice as fast.
    
    Args:
        baseline: Reference benchmark result.
        optimised: Optimised benchmark result.
    
    Returns:
        Speedup factor.
    """
    if optimised.median <= 0:
        return float('inf')
    return baseline.median / optimised.median


def compute_efficiency(speedup: float, num_processors: int) -> float:
    """
    Calculate parallel efficiency.
    
    Efficiency = Speedup / Number of Processors
    
    Efficiency near 1.0 indicates good parallel scaling.
    
    Args:
        speedup: Observed speedup.
        num_processors: Number of parallel processors used.
    
    Returns:
        Efficiency value between 0 and 1 (ideally).
    """
    if num_processors <= 0:
        return 0.0
    return speedup / num_processors


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BENCHMARK SUITE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkSuite:
    """
    Comprehensive benchmark suite for comparing algorithms.
    
    Features:
        - Run benchmarks across multiple input sizes
        - Compare multiple algorithms
        - Export results (CSV, JSON)
        - Estimate empirical complexity
        - Generate comparison summaries
    
    Attributes:
        name: Name of the benchmark suite.
        algorithms: Dictionary mapping names to callable algorithms.
        results: Dictionary mapping names to lists of BenchmarkResult.
    
    Example:
        >>> suite = BenchmarkSuite("Sorting Comparison")
        >>> suite.add_algorithm("bubble", bubble_sort)
        >>> suite.add_algorithm("merge", merge_sort)
        >>> suite.run([100, 1000, 10000], generate_random_data)
        >>> logger.info(suite.summary())
    """
    name: str
    algorithms: dict[str, Callable] = field(default_factory=dict)
    results: dict[str, list[BenchmarkResult]] = field(default_factory=dict)
    
    def add_algorithm(self, name: str, func: Callable) -> None:
        """
        Add an algorithm to the suite.
        
        Args:
            name: Display name for the algorithm.
            func: Callable that takes a list and returns a sorted list.
        """
        self.algorithms[name] = func
        self.results[name] = []
        logger.debug(f"Added algorithm: {name}")
    
    def run(
        self,
        sizes: list[int],
        data_generator: DataGenerator,
        runs_per_size: int = 10,
        warmup: int = 2,
        verbose: bool = True
    ) -> None:
        """
        Run all algorithms on all specified sizes.
        
        Args:
            sizes: List of input sizes to test.
            data_generator: Function that generates test data given size.
            runs_per_size: Number of runs per size for statistics.
            warmup: Number of warmup runs.
            verbose: Whether to display progress.
        """
        total = len(self.algorithms) * len(sizes)
        current = 0
        
        for size in sizes:
            # Generate data once per size
            test_data = data_generator(size)
            
            for name, func in self.algorithms.items():
                current += 1
                if verbose:
                    logger.info(f"[{current}/{total}] {name} (n={size:,})...")
                
                result = benchmark(
                    func, test_data,
                    runs=runs_per_size,
                    warmup=warmup,
                    name=name,
                    n=size
                )
                self.results[name].append(result)
                
                if verbose:
                    logger.info(f"  → {result.median:.3f} ms (CV={result.coefficient_of_variation:.1%})")
    
    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            "═" * 60,
            f"  BENCHMARK SUITE: {self.name}",
            "═" * 60,
            ""
        ]
        
        for name, results in self.results.items():
            lines.append(f"▸ {name}")
            sizes = [r.n for r in results]
            times = [r.median for r in results]
            
            exp, _, complexity = estimate_complexity(sizes, times)
            lines.append(f"  Estimated complexity: {complexity} (exponent={exp:.2f})")
            
            for r in results:
                lines.append(f"    n={r.n:>8,}: {r.median:>10.3f} ms ± {r.std:>8.3f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_csv(self, filename: str | Path) -> None:
        """
        Export results to CSV format.
        
        Args:
            filename: Output file path.
        """
        filepath = Path(filename)
        
        with filepath.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'algorithm', 'n', 'mean_ms', 'median_ms', 'std_ms',
                'min_ms', 'max_ms', 'iqr_ms', 'cv', 'runs'
            ])
            writer.writeheader()
            
            for name, results in self.results.items():
                for r in results:
                    writer.writerow({
                        'algorithm': name,
                        'n': r.n,
                        'mean_ms': f"{r.mean:.6f}",
                        'median_ms': f"{r.median:.6f}",
                        'std_ms': f"{r.std:.6f}",
                        'min_ms': f"{r.min_time:.6f}",
                        'max_ms': f"{r.max_time:.6f}",
                        'iqr_ms': f"{r.iqr:.6f}",
                        'cv': f"{r.coefficient_of_variation:.4f}",
                        'runs': len(r.times)
                    })
        
        logger.info(f"Results exported to {filepath}")
    
    def export_json(self, filename: str | Path) -> None:
        """
        Export results to JSON format.
        
        Args:
            filename: Output file path.
        """
        filepath = Path(filename)
        
        data = {
            'suite_name': self.name,
            'algorithms': {}
        }
        
        for name, results in self.results.items():
            sizes = [r.n for r in results]
            times = [r.median for r in results]
            exp, _, complexity = estimate_complexity(sizes, times)
            
            data['algorithms'][name] = {
                'estimated_complexity': complexity,
                'exponent': exp,
                'results': [r.to_dict() for r in results]
            }
        
        with filepath.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")
    
    def get_fastest(self, size: int) -> tuple[str, BenchmarkResult] | None:
        """
        Get the fastest algorithm for a given input size.
        
        Args:
            size: Input size to query.
        
        Returns:
            Tuple of (algorithm_name, result) or None if not found.
        """
        best_name: str | None = None
        best_result: BenchmarkResult | None = None
        
        for name, results in self.results.items():
            for r in results:
                if r.n == size:
                    if best_result is None or r.median < best_result.median:
                        best_name = name
                        best_result = r
        
        if best_name is not None and best_result is not None:
            return best_name, best_result
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_random_data(n: int) -> list[float]:
    """Generate random data for benchmarking."""
    return [random.random() for _ in range(n)]


def generate_sorted_data(n: int) -> list[float]:
    """Generate already-sorted data (best case for some algorithms)."""
    return list(range(n))


def generate_reverse_sorted_data(n: int) -> list[float]:
    """Generate reverse-sorted data (worst case for some algorithms)."""
    return list(range(n, 0, -1))


def generate_nearly_sorted_data(n: int, swaps: int = 10) -> list[float]:
    """Generate nearly-sorted data with a few random swaps."""
    arr = list(range(n))
    for _ in range(swaps):
        i, j = random.randint(0, n - 1), random.randint(0, n - 1)
        arr[i], arr[j] = arr[j], arr[i]
    return arr


def generate_duplicate_data(n: int, unique_values: int = 10) -> list[float]:
    """Generate data with many duplicate values."""
    return [random.randint(0, unique_values - 1) for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_basic_benchmark() -> None:
    """Demonstration: basic benchmark usage."""
    logger.info("═" * 60)
    logger.info("  DEMO: Basic Benchmark")
    logger.info("═" * 60)
    
    data = generate_random_data(10000)
    
    # Individual benchmarks
    result = benchmark(sorted, data, runs=10, name="Python built-in sort", n=10000)
    logger.info(f"  {result}")
    
    result = benchmark(quicksort, data, runs=10, name="Our quicksort", n=10000)
    logger.info(f"  {result}")
    
    result = benchmark(heapsort, data, runs=10, name="Our heapsort", n=10000)
    logger.info(f"  {result}")


def demo_complexity_analysis() -> None:
    """Demonstration: empirical complexity analysis."""
    logger.info("═" * 60)
    logger.info("  DEMO: Complexity Analysis")
    logger.info("═" * 60)
    
    # Test bubble sort
    sizes = [100, 200, 500, 1000, 2000]
    times: list[float] = []
    
    logger.info("Bubble Sort:")
    for n in sizes:
        data = generate_random_data(n)
        result = benchmark(bubble_sort, data, runs=3, warmup=1, name="bubble", n=n)
        times.append(result.median)
        logger.info(f"  n={n:>5}: {result.median:.3f} ms")
    
    exp, c, complexity = estimate_complexity(sizes, times)
    logger.info(f"  → Estimated complexity: {complexity} (exponent={exp:.2f})")
    
    # Test merge sort
    sizes = [1000, 2000, 5000, 10000, 20000]
    times = []
    
    logger.info("Merge Sort:")
    for n in sizes:
        data = generate_random_data(n)
        result = benchmark(merge_sort, data, runs=5, warmup=2, name="merge", n=n)
        times.append(result.median)
        logger.info(f"  n={n:>5}: {result.median:.3f} ms")
    
    exp, c, complexity = estimate_complexity(sizes, times)
    logger.info(f"  → Estimated complexity: {complexity} (exponent={exp:.2f})")


def demo_full_suite() -> None:
    """Demonstration: complete benchmark suite."""
    logger.info("═" * 60)
    logger.info("  DEMO: Full Benchmark Suite")
    logger.info("═" * 60)
    
    suite = BenchmarkSuite("Sorting Algorithms Comparison")
    
    # Add algorithms
    suite.add_algorithm("Bubble Sort", bubble_sort)
    suite.add_algorithm("Insertion Sort", insertion_sort)
    suite.add_algorithm("Merge Sort", merge_sort)
    suite.add_algorithm("Quicksort", quicksort)
    suite.add_algorithm("Heapsort", heapsort)
    suite.add_algorithm("Python sorted()", lambda x: sorted(x))
    suite.add_algorithm("NumPy sort", numpy_sort)
    
    if HAS_NUMBA:
        suite.add_algorithm("Numba Quicksort", lambda x: numba_quicksort(np.array(x)))
    
    # Run benchmarks (smaller sizes for quick demo)
    sizes = [500, 1000, 2000, 5000]
    
    logger.info("Running benchmarks...")
    suite.run(
        sizes=sizes,
        data_generator=generate_random_data,
        runs_per_size=5,
        warmup=2,
        verbose=True
    )
    
    logger.info(suite.summary())


def demo_speedup_comparison() -> None:
    """Demonstration: speedup comparison."""
    logger.info("═" * 60)
    logger.info("  DEMO: Speedup Comparison")
    logger.info("═" * 60)
    
    n = 10000
    data = generate_random_data(n)
    np_data = np.array(data)
    
    # Baseline: pure Python
    baseline = benchmark(quicksort, data, runs=5, name="Quicksort (Python)", n=n)
    
    # NumPy
    numpy_result = benchmark(numpy_sort, data, runs=5, name="NumPy sort", n=n)
    
    logger.info(f"Baseline: {baseline}")
    logger.info(f"NumPy:    {numpy_result}")
    logger.info(f"  Speedup: {compute_speedup(baseline, numpy_result):.1f}x")
    
    # Numba (if available)
    if HAS_NUMBA:
        # Separate warmup for Numba (JIT compilation)
        _ = numba_quicksort(np_data)
        numba_result = benchmark(
            lambda x: numba_quicksort(np.array(x)),
            data, runs=5, name="Numba Quicksort", n=n
        )
        logger.info(f"Numba:    {numba_result}")
        logger.info(f"  Speedup: {compute_speedup(baseline, numba_result):.1f}x")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_basic_benchmark()
    logger.info("")
    demo_complexity_analysis()
    logger.info("")
    demo_speedup_comparison()
    logger.info("")
    # Uncomment for full suite (slower)
    # demo_full_suite()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 1: Memory Profiling                                                   ║
║                                                                               ║
║ Add memory measurement to BenchmarkResult.                                    ║
║                                                                               ║
║ Hints:                                                                        ║
║ - tracemalloc.start() / tracemalloc.get_traced_memory()                      ║
║ - memory_profiler (pip install memory-profiler)                              ║
║                                                                               ║
║ Extend BenchmarkResult with:                                                  ║
║   peak_memory: float  # Peak memory usage in MB                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 2: Cache-Aware Matrix Multiplication                                  ║
║                                                                               ║
║ Implement matrix multiplication in three variants:                            ║
║ 1. Naive: C[i][j] = Σ A[i][k] * B[k][j]                                      ║
║ 2. Transposed: B^T for sequential access                                      ║
║ 3. Blocked: process in cache-friendly block sizes                            ║
║                                                                               ║
║ Measure speedup for 500×500 and 1000×1000 matrices.                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 3: Parallel Sorting Benchmark                                         ║
║                                                                               ║
║ Use multiprocessing to parallelise merge sort:                               ║
║ - Split array into chunks                                                     ║
║ - Sort chunks in parallel                                                     ║
║ - Merge sorted chunks                                                         ║
║                                                                               ║
║ Measure efficiency = speedup / num_processors                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="03UNIT Lab: Benchmark Suite for Algorithm Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_3_01_benchmark_suite.py --demo
  python lab_3_01_benchmark_suite.py --full-suite --export results.csv
  python lab_3_01_benchmark_suite.py -v --sizes 100 1000 10000
        """
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration benchmarks"
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Run complete benchmark suite"
    )
    parser.add_argument(
        "--export",
        type=str,
        metavar="FILE",
        help="Export results to CSV file"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 5000, 10000],
        help="Input sizes to benchmark (default: 1000 5000 10000)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    logger.info("═" * 60)
    logger.info("  03UNIT LAB: BENCHMARK SUITE")
    logger.info("═" * 60)
    logger.info(f"NumPy available: True")
    logger.info(f"Numba available: {HAS_NUMBA}")
    
    if args.demo:
        run_all_demos()
    elif args.full_suite:
        demo_full_suite()
    else:
        # Default: run demos
        run_all_demos()
    
    logger.info("═" * 60)
    logger.info("Exercises to complete:")
    logger.info("  1. Add memory profiling")
    logger.info("  2. Cache-aware matrix multiplication")
    logger.info("  3. Parallel sorting benchmark")
    logger.info("═" * 60)


if __name__ == "__main__":
    main()
