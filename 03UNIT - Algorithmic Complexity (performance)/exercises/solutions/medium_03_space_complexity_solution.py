#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Solutions for medium_03_space_complexity.py
═══════════════════════════════════════════════════════════════════════════════

Complete solutions demonstrating space complexity analysis techniques.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import sys
import tracemalloc
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY: MEMORY TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MemoryMeasurement:
    """Result of a memory measurement."""

    current_bytes: int
    peak_bytes: int
    description: str

    @property
    def current_kb(self) -> float:
        """Current memory in kilobytes."""
        return self.current_bytes / 1024

    @property
    def peak_kb(self) -> float:
        """Peak memory in kilobytes."""
        return self.peak_bytes / 1024

    def __repr__(self) -> str:
        return (
            f"MemoryMeasurement(current={self.current_kb:.2f}KB, "
            f"peak={self.peak_kb:.2f}KB, desc='{self.description}')"
        )


@contextmanager
def memory_tracker(description: str = "") -> Generator[MemoryMeasurement, None, None]:
    """Context manager that tracks memory usage within its scope.

    Args:
        description: Label for this measurement.

    Yields:
        MemoryMeasurement object that is updated upon context exit.
    """
    measurement = MemoryMeasurement(
        current_bytes=0, peak_bytes=0, description=description
    )
    tracemalloc.start()
    try:
        yield measurement
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measurement.current_bytes = current
        measurement.peak_bytes = peak


# ═══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATIONS: REVERSAL ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def reverse_list_new(data: list[Any]) -> list[Any]:
    """Reverse a list by creating a new list.

    Space Complexity: O(n) - creates a complete copy of the input.

    Args:
        data: List to reverse.

    Returns:
        New list with elements in reversed order.
    """
    return data[::-1]


def reverse_list_inplace(data: list[Any]) -> list[Any]:
    """Reverse a list in place using two-pointer technique.

    Space Complexity: O(1) - only uses two index variables.

    Args:
        data: List to reverse (modified in place).

    Returns:
        The same list object, now reversed.
    """
    left, right = 0, len(data) - 1
    while left < right:
        data[left], data[right] = data[right], data[left]
        left += 1
        right -= 1
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATIONS: MATRIX TRANSPOSE
# ═══════════════════════════════════════════════════════════════════════════════


def matrix_transpose_new(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix by creating a new matrix.

    Space Complexity: O(m*n) where m and n are matrix dimensions.

    Args:
        matrix: Input matrix to transpose.

    Returns:
        New transposed matrix.
    """
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    result = [[0.0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            result[j][i] = matrix[i][j]
    return result


def matrix_transpose_inplace(matrix: list[list[float]]) -> list[list[float]]:
    """Transpose a square matrix in place.

    Space Complexity: O(1) - swaps elements without auxiliary storage.

    Note: Only works for square matrices.

    Args:
        matrix: Square matrix to transpose (modified in place).

    Returns:
        The same matrix object, now transposed.

    Raises:
        ValueError: If matrix is not square.
    """
    if not matrix:
        return matrix
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square for in-place transpose")

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATIONS: FIBONACCI VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


def fibonacci_array(n: int) -> int:
    """Compute nth Fibonacci number using array storage.

    Space Complexity: O(n) - stores all intermediate values.

    Args:
        n: Index of Fibonacci number (0-indexed).

    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    fib = [0] * (n + 1)
    fib[1] = 1
    for i in range(2, n + 1):
        fib[i] = fib[i - 1] + fib[i - 2]
    return fib[n]


def fibonacci_variables(n: int) -> int:
    """Compute nth Fibonacci number using only two variables.

    Space Complexity: O(1) - constant space regardless of n.

    Args:
        n: Index of Fibonacci number (0-indexed).

    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


# ═══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTATIONS: SUBSET GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def get_all_subsets_eager(items: list[Any]) -> list[list[Any]]:
    """Generate all subsets eagerly (storing all in memory).

    Space Complexity: O(n * 2^n) - stores all 2^n subsets.

    Args:
        items: Input list.

    Returns:
        List of all subsets.
    """
    result: list[list[Any]] = [[]]
    for item in items:
        result.extend([subset + [item] for subset in result])
    return result


def get_all_subsets_lazy(items: list[Any]) -> Iterator[list[Any]]:
    """Generate all subsets lazily using a generator.

    Space Complexity: O(n) per yielded subset - doesn't store all subsets.

    Args:
        items: Input list.

    Yields:
        Subsets one at a time.
    """
    n = len(items)
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = [items[i] for i in range(n) if mask & (1 << i)]
        yield subset


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 1: COMPARE SPACE COMPLEXITY FOR REVERSE
# ═══════════════════════════════════════════════════════════════════════════════


def compare_space_complexity_reverse(sizes: list[int]) -> dict[str, list[float]]:
    """Compare space usage between reversal implementations.

    This solution demonstrates how O(n) vs O(1) space manifests in practice.

    Args:
        sizes: List of input sizes to test.

    Returns:
        Dictionary mapping algorithm names to lists of peak memory (KB).
    """
    results: dict[str, list[float]] = {"reverse_new": [], "reverse_inplace": []}

    for size in sizes:
        # Test reverse_list_new (O(n) space)
        data = list(range(size))
        with memory_tracker("reverse_new") as mem_new:
            _ = reverse_list_new(data)
        results["reverse_new"].append(mem_new.peak_kb)

        # Test reverse_list_inplace (O(1) space)
        data = list(range(size))
        with memory_tracker("reverse_inplace") as mem_inplace:
            _ = reverse_list_inplace(data)
        results["reverse_inplace"].append(mem_inplace.peak_kb)

        logger.info(
            "Size %d: new=%.2f KB, inplace=%.2f KB",
            size,
            mem_new.peak_kb,
            mem_inplace.peak_kb,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 2: COMPARE FIBONACCI SPACE
# ═══════════════════════════════════════════════════════════════════════════════


def compare_fibonacci_space(n_values: list[int]) -> dict[str, list[float]]:
    """Compare space usage between Fibonacci implementations.

    Demonstrates O(n) vs O(1) space complexity for the same result.

    Args:
        n_values: List of Fibonacci indices to compute.

    Returns:
        Dictionary mapping algorithm names to lists of peak memory (KB).
    """
    results: dict[str, list[float]] = {"fib_array": [], "fib_variables": []}

    for n in n_values:
        # Test fibonacci_array (O(n) space)
        with memory_tracker("fib_array") as mem_array:
            result_array = fibonacci_array(n)
        results["fib_array"].append(mem_array.peak_kb)

        # Test fibonacci_variables (O(1) space)
        with memory_tracker("fib_variables") as mem_vars:
            result_vars = fibonacci_variables(n)
        results["fib_variables"].append(mem_vars.peak_kb)

        # Verify correctness
        assert result_array == result_vars, f"Mismatch at n={n}"

        logger.info(
            "F(%d)=%d: array=%.4f KB, variables=%.4f KB",
            n,
            result_array,
            mem_array.peak_kb,
            mem_vars.peak_kb,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 3: ANALYSE SUBSET MEMORY
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_subset_memory(max_size: int = 15) -> dict[str, list[float]]:
    """Analyse memory growth for eager vs lazy subset generation.

    Demonstrates exponential O(n * 2^n) vs linear O(n) space.

    Args:
        max_size: Maximum input size to test.

    Returns:
        Dictionary with memory measurements.
    """
    results: dict[str, list[float]] = {
        "eager": [],
        "lazy_iteration": [],
        "sizes": [],
    }

    for size in range(1, max_size + 1):
        items = list(range(size))

        # Test eager generation (O(n * 2^n) space)
        with memory_tracker("eager") as mem_eager:
            all_subsets = get_all_subsets_eager(items)
            subset_count_eager = len(all_subsets)
        results["eager"].append(mem_eager.peak_kb)

        # Test lazy generation (O(n) space per subset)
        with memory_tracker("lazy") as mem_lazy:
            subset_count_lazy = sum(1 for _ in get_all_subsets_lazy(items))
        results["lazy_iteration"].append(mem_lazy.peak_kb)

        results["sizes"].append(size)

        # Verify same number of subsets
        assert subset_count_eager == subset_count_lazy == (1 << size), (
            f"Subset count mismatch at size={size}"
        )

        logger.info(
            "Size %d (%d subsets): eager=%.2f KB, lazy=%.4f KB",
            size,
            1 << size,
            mem_eager.peak_kb,
            mem_lazy.peak_kb,
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 4: SLIDING WINDOW MAXIMUM
# ═══════════════════════════════════════════════════════════════════════════════


def implement_sliding_window_max(
    data: list[int], window_size: int
) -> list[int]:
    """Find the maximum in each sliding window of size k.

    Uses a monotonic deque for O(n) time and O(k) space.

    Space Complexity: O(k) where k is window_size.

    Args:
        data: Input array.
        window_size: Size of the sliding window.

    Returns:
        List of maximums for each window position.

    Example:
        >>> implement_sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)
        [3, 3, 5, 5, 6, 7]
    """
    from collections import deque

    if not data or window_size <= 0:
        return []

    if window_size > len(data):
        return [max(data)] if data else []

    # Monotonic deque storing indices
    # Front always contains index of current max
    dq: deque[int] = deque()
    result: list[int] = []

    for i, value in enumerate(data):
        # Remove indices outside current window
        while dq and dq[0] <= i - window_size:
            dq.popleft()

        # Remove indices of smaller elements (maintain monotonicity)
        while dq and data[dq[-1]] < value:
            dq.pop()

        dq.append(i)

        # Start adding results once first window is complete
        if i >= window_size - 1:
            result.append(data[dq[0]])

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 5: SPACE-EFFICIENT MATRIX MULTIPLICATION
# ═══════════════════════════════════════════════════════════════════════════════


def optimise_matrix_multiplication(
    a: list[list[float]], b: list[list[float]], block_size: int = 64
) -> list[list[float]]:
    """Multiply matrices with cache-efficient blocked approach.

    This implementation is cache-friendly but doesn't reduce asymptotic space.
    Space Complexity: O(n²) for result matrix.

    For truly space-efficient multiplication of large matrices, one would need
    disk-based or streaming approaches.

    Args:
        a: First matrix (m x k).
        b: Second matrix (k x n).
        block_size: Block size for cache efficiency.

    Returns:
        Result matrix (m x n).

    Raises:
        ValueError: If matrix dimensions are incompatible.
    """
    if not a or not b or not a[0] or not b[0]:
        return []

    m, k1 = len(a), len(a[0])
    k2, n = len(b), len(b[0])

    if k1 != k2:
        raise ValueError(f"Incompatible dimensions: {k1} vs {k2}")

    k = k1

    # Allocate result matrix
    result = [[0.0] * n for _ in range(m)]

    # Blocked multiplication for better cache utilisation
    for i0 in range(0, m, block_size):
        for j0 in range(0, n, block_size):
            for l0 in range(0, k, block_size):
                # Process one block
                for i in range(i0, min(i0 + block_size, m)):
                    for j in range(j0, min(j0 + block_size, n)):
                        total = 0.0
                        for ll in range(l0, min(l0 + block_size, k)):
                            total += a[i][ll] * b[ll][j]
                        result[i][j] += total

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 6: STRING CONCATENATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_string_concatenation(n: int) -> dict[str, float]:
    """Compare space efficiency of string building methods.

    Demonstrates why string += is O(n²) space (due to copying) while
    ''.join() is O(n) space.

    Args:
        n: Number of strings to concatenate.

    Returns:
        Dictionary with peak memory for each method.
    """
    results: dict[str, float] = {}

    # Method 1: Naive concatenation (O(n²) copies due to immutability)
    with memory_tracker("naive_concat") as mem_naive:
        result = ""
        for i in range(n):
            result = result + str(i)  # noqa: PERF401
        _ = len(result)  # Force evaluation
    results["naive_concat_kb"] = mem_naive.peak_kb

    # Method 2: List accumulation + join (O(n) space)
    with memory_tracker("list_join") as mem_join:
        parts: list[str] = []
        for i in range(n):
            parts.append(str(i))
        result = "".join(parts)
        _ = len(result)
    results["list_join_kb"] = mem_join.peak_kb

    # Method 3: Generator + join (O(1) extra space during generation)
    with memory_tracker("generator_join") as mem_gen:
        result = "".join(str(i) for i in range(n))
        _ = len(result)
    results["generator_join_kb"] = mem_gen.peak_kb

    logger.info(
        "String concat (n=%d): naive=%.2f KB, list_join=%.2f KB, gen_join=%.2f KB",
        n,
        results["naive_concat_kb"],
        results["list_join_kb"],
        results["generator_join_kb"],
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS SOLUTION 1: ITERATOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def implement_iterator_pipeline(
    data: Iterator[int], operations: list[str]
) -> Iterator[int]:
    """Create a memory-efficient iterator pipeline.

    Chains transformations without materialising intermediate results.
    Space Complexity: O(1) additional space (excludes input).

    Args:
        data: Input iterator of integers.
        operations: List of operation names: 'double', 'square', 'filter_even'.

    Yields:
        Transformed values.

    Example:
        >>> list(implement_iterator_pipeline(iter([1, 2, 3, 4]), ['double', 'filter_even']))
        [2, 4, 6, 8]
    """

    def double(x: int) -> int:
        return x * 2

    def square(x: int) -> int:
        return x * x

    current: Iterator[int] = data

    for op in operations:
        if op == "double":
            current = (double(x) for x in current)
        elif op == "square":
            current = (square(x) for x in current)
        elif op == "filter_even":
            current = (x for x in current if x % 2 == 0)
        elif op == "filter_positive":
            current = (x for x in current if x > 0)
        else:
            logger.warning("Unknown operation: %s", op)

    yield from current


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS SOLUTION 2: RECURSION STACK SPACE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def analyse_recursion_stack_space(n: int) -> dict[str, int]:
    """Analyse stack space for different recursive patterns.

    Compares linear recursion O(n) vs tail-call-optimisable O(1)
    (though Python doesn't optimise tail calls, we simulate it).

    Args:
        n: Recursion depth to test.

    Returns:
        Dictionary with max stack depths.
    """

    # Track maximum recursion depth
    max_depth_linear = 0
    max_depth_tail = 0

    def linear_recursive_sum(k: int, current_depth: int = 1) -> int:
        """Standard recursion: O(n) stack space."""
        nonlocal max_depth_linear
        max_depth_linear = max(max_depth_linear, current_depth)
        if k <= 0:
            return 0
        return k + linear_recursive_sum(k - 1, current_depth + 1)

    def tail_recursive_sum(k: int, acc: int = 0, current_depth: int = 1) -> int:
        """Tail recursion pattern: O(n) in Python (would be O(1) with TCO)."""
        nonlocal max_depth_tail
        max_depth_tail = max(max_depth_tail, current_depth)
        if k <= 0:
            return acc
        return tail_recursive_sum(k - 1, acc + k, current_depth + 1)

    def iterative_sum(k: int) -> int:
        """Iterative version: O(1) stack space."""
        total = 0
        for i in range(1, k + 1):
            total += i
        return total

    # Increase recursion limit temporarily for testing
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(n + 100, old_limit))

    try:
        result_linear = linear_recursive_sum(n)
        result_tail = tail_recursive_sum(n)
        result_iterative = iterative_sum(n)

        # Verify all produce same result
        expected = n * (n + 1) // 2
        assert result_linear == expected, f"Linear: {result_linear} != {expected}"
        assert result_tail == expected, f"Tail: {result_tail} != {expected}"
        assert result_iterative == expected, f"Iterative: {result_iterative} != {expected}"

    finally:
        sys.setrecursionlimit(old_limit)

    logger.info(
        "Stack space analysis (n=%d): linear_depth=%d, tail_depth=%d",
        n,
        max_depth_linear,
        max_depth_tail,
    )

    return {
        "linear_recursion_depth": max_depth_linear,
        "tail_recursion_depth": max_depth_tail,
        "iterative_stack_frames": 1,  # Constant
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_demonstrations() -> None:
    """Run all solution demonstrations."""
    logger.info("=" * 70)
    logger.info("SPACE COMPLEXITY SOLUTIONS DEMONSTRATION")
    logger.info("=" * 70)

    # Demo 1: Reverse comparison
    logger.info("\n--- Reverse Space Comparison ---")
    sizes = [1000, 5000, 10000, 50000]
    compare_space_complexity_reverse(sizes)

    # Demo 2: Fibonacci comparison
    logger.info("\n--- Fibonacci Space Comparison ---")
    fib_indices = [100, 500, 1000, 5000]
    compare_fibonacci_space(fib_indices)

    # Demo 3: Subset generation
    logger.info("\n--- Subset Generation Space ---")
    analyse_subset_memory(max_size=12)

    # Demo 4: Sliding window
    logger.info("\n--- Sliding Window Maximum ---")
    test_data = [1, 3, -1, -3, 5, 3, 6, 7]
    result = implement_sliding_window_max(test_data, 3)
    logger.info("Input: %s, Window: 3, Result: %s", test_data, result)

    # Demo 5: String concatenation
    logger.info("\n--- String Concatenation Space ---")
    analyse_string_concatenation(5000)

    # Demo 6: Recursion stack
    logger.info("\n--- Recursion Stack Analysis ---")
    analyse_recursion_stack_space(500)

    logger.info("\n" + "=" * 70)
    logger.info("All demonstrations completed successfully")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_demonstrations()
