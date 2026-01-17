#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Space Complexity (Medium)
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Space complexity is equally important as time complexity, particularly when
working with large datasets or memory-constrained environments. This exercise
develops your ability to analyse and optimise memory usage in algorithms,
distinguishing between auxiliary space and total space requirements.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Analyse space complexity of iterative and recursive algorithms
2. Measure actual memory usage using Python's tracemalloc
3. Optimise algorithms for reduced memory footprint

ESTIMATED TIME
──────────────
- Reading: 15 minutes
- Coding: 35 minutes
- Total: 50 minutes

DIFFICULTY: ⭐⭐⭐ (Medium)

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
from typing import Any, Generator, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ALGORITHMS WITH DIFFERENT SPACE COMPLEXITIES
# ═══════════════════════════════════════════════════════════════════════════════

def reverse_list_new(arr: list[Any]) -> list[Any]:
    """
    Reverse a list by creating a new list.
    
    Space Complexity: O(n) - creates a new list of size n
    
    Args:
        arr: Input list.
        
    Returns:
        New reversed list.
    """
    return arr[::-1]


def reverse_list_inplace(arr: list[Any]) -> list[Any]:
    """
    Reverse a list in place using swaps.
    
    Space Complexity: O(1) - only uses a few variables
    
    Args:
        arr: Input list (modified in place).
        
    Returns:
        The same list, now reversed.
    """
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
    return arr


def matrix_transpose_new(matrix: list[list[float]]) -> list[list[float]]:
    """
    Transpose a matrix by creating a new matrix.
    
    Space Complexity: O(n*m) for n×m matrix
    
    Args:
        matrix: Input n×m matrix.
        
    Returns:
        New m×n transposed matrix.
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
    """
    Transpose a square matrix in place.
    
    Space Complexity: O(1) - swaps elements without extra storage
    Note: Only works for square matrices!
    
    Args:
        matrix: Input n×n square matrix (modified in place).
        
    Returns:
        The same matrix, now transposed.
    """
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    return matrix


def fibonacci_array(n: int) -> int:
    """
    Compute nth Fibonacci using array storage.
    
    Space Complexity: O(n) - stores all values up to n
    
    Args:
        n: Index in Fibonacci sequence.
        
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
    """
    Compute nth Fibonacci using only two variables.
    
    Space Complexity: O(1) - constant space regardless of n
    
    Args:
        n: Index in Fibonacci sequence.
        
    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1


def get_all_subsets_eager(items: list[Any]) -> list[list[Any]]:
    """
    Generate all subsets eagerly (store all in memory).
    
    Space Complexity: O(n * 2^n) - 2^n subsets of average size n/2
    
    Args:
        items: List of items.
        
    Returns:
        List of all 2^n subsets.
    """
    result: list[list[Any]] = [[]]
    
    for item in items:
        result += [subset + [item] for subset in result]
    
    return result


def get_all_subsets_lazy(items: list[Any]) -> Generator[list[Any], None, None]:
    """
    Generate all subsets lazily (one at a time).
    
    Space Complexity: O(n) - only one subset in memory at a time
    
    Args:
        items: List of items.
        
    Yields:
        Each subset one at a time.
    """
    n = len(items)
    for i in range(2 ** n):
        subset = [items[j] for j in range(n) if (i >> j) & 1]
        yield subset


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: MEMORY MEASUREMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryTracker:
    """Context manager for tracking memory usage."""
    
    def __init__(self) -> None:
        """Initialise the memory tracker."""
        self.peak_memory: int = 0
        self.start_memory: int = 0
        self.end_memory: int = 0
    
    def __enter__(self) -> "MemoryTracker":
        """Start memory tracking."""
        tracemalloc.start()
        self.start_memory = tracemalloc.get_traced_memory()[0]
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Stop memory tracking."""
        self.end_memory, self.peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    
    @property
    def peak_mb(self) -> float:
        """Peak memory in megabytes."""
        return self.peak_memory / (1024 * 1024)
    
    @property
    def allocated_mb(self) -> float:
        """Total allocated memory in megabytes."""
        return (self.end_memory - self.start_memory) / (1024 * 1024)


def measure_memory(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Measure peak memory usage of a function call.
    
    Args:
        func: Function to measure.
        *args: Arguments to pass to function.
        **kwargs: Keyword arguments.
        
    Returns:
        Tuple of (result, peak_memory_mb).
    """
    with MemoryTracker() as tracker:
        result = func(*args, **kwargs)
    return result, tracker.peak_mb


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: YOUR TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def compare_space_complexity_reverse() -> dict[str, list[tuple[int, float]]]:
    """
    EXERCISE 1: Compare space complexity of list reversal methods.
    
    Measure memory usage for both reverse_list_new and reverse_list_inplace
    for increasing list sizes.
    
    Returns:
        Dictionary with:
        - "new": List of (n, memory_mb) tuples
        - "inplace": List of (n, memory_mb) tuples
        
    Test with sizes: [1000, 10000, 100000, 1000000]
    
    Questions to answer (in comments):
        - How does memory scale with input size for each method?
        - When would you prefer one over the other?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compare_space_complexity_reverse")


def compare_fibonacci_space() -> dict[str, list[tuple[int, float]]]:
    """
    EXERCISE 2: Compare Fibonacci implementations' space usage.
    
    Compare fibonacci_array vs fibonacci_variables for various n values.
    
    Returns:
        Dictionary with memory measurements for both implementations.
        
    Test with n values: [100, 1000, 10000, 100000]
    
    Questions to answer:
        - What is the memory growth rate for each?
        - At what point does the array version become problematic?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compare_fibonacci_space")


def analyse_subset_memory() -> dict[str, Any]:
    """
    EXERCISE 3: Analyse eager vs lazy subset generation.
    
    Compare get_all_subsets_eager vs get_all_subsets_lazy.
    
    Returns:
        Dictionary with:
        - "eager_memory": Memory for eager version (n=15)
        - "lazy_memory": Memory for processing lazy version (n=15)
        - "eager_count": Number of subsets
        - "lazy_count": Number of subsets yielded
        
    Caution: Don't go above n=20 for eager - it will exhaust memory!
    
    Questions to answer:
        - Why is lazy evaluation so much more memory efficient?
        - When might you still prefer the eager version?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement analyse_subset_memory")


def implement_sliding_window_max(arr: list[int], k: int) -> list[int]:
    """
    EXERCISE 4: Implement sliding window maximum with O(n) time and O(k) space.
    
    For each window of size k, find the maximum element.
    
    Args:
        arr: Input array of integers.
        k: Window size.
        
    Returns:
        List of maximum values for each window position.
        
    Example:
        >>> implement_sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)
        [3, 3, 5, 5, 6, 7]
        
    Requirements:
        - Time complexity: O(n)
        - Space complexity: O(k) auxiliary space
        
    Hint: Use a deque to track indices of potential maximum values.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement implement_sliding_window_max")


def optimise_matrix_multiplication(
    a: list[list[float]], 
    b: list[list[float]]
) -> list[list[float]]:
    """
    EXERCISE 5: Implement matrix multiplication with minimal auxiliary space.
    
    Multiply two matrices A (m×n) and B (n×p) to produce C (m×p).
    
    Args:
        a: First matrix (m×n).
        b: Second matrix (n×p).
        
    Returns:
        Result matrix (m×p).
        
    Requirements:
        - Must work correctly
        - Auxiliary space should be O(p) for the row buffer
        - Do NOT create intermediate matrices
        
    Hint: Compute one row of the result at a time.
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement optimise_matrix_multiplication")


def analyse_string_concatenation() -> dict[str, tuple[float, float]]:
    """
    EXERCISE 6: Analyse space complexity of string building strategies.
    
    Compare three approaches to building a string of n characters:
    1. Repeated concatenation: s = s + char (very inefficient!)
    2. Using a list and join: chars.append(char); ''.join(chars)
    3. Using io.StringIO
    
    Returns:
        Dictionary mapping method name to (time_seconds, memory_mb).
        
    Test with n = 100000 characters.
    
    Questions to answer:
        - Why is repeated concatenation so memory-inefficient?
        - Which method is best for memory? For speed?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement analyse_string_concatenation")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BONUS CHALLENGES
# ═══════════════════════════════════════════════════════════════════════════════

def implement_iterator_pipeline() -> Iterator[int]:
    """
    BONUS 1: Create a memory-efficient data processing pipeline.
    
    Create a pipeline that:
    1. Generates integers from 1 to 1,000,000
    2. Filters to keep only even numbers
    3. Squares each number
    4. Takes only first 100 results
    
    All with O(1) memory complexity using generators!
    
    Returns:
        Iterator yielding the first 100 squared even numbers.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement implement_iterator_pipeline")


def analyse_recursion_stack_space(n: int) -> dict[str, int]:
    """
    BONUS 2: Analyse stack space usage in recursive algorithms.
    
    Compare stack frames used by:
    1. Factorial (linear recursion)
    2. Fibonacci (tree recursion)
    3. Binary search (logarithmic recursion)
    
    Args:
        n: Input size.
        
    Returns:
        Dictionary with maximum stack depth for each algorithm.
        
    Hint: Use sys.getrecursionlimit() and a counter variable.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement analyse_recursion_stack_space")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VERIFICATION AND DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def verify_implementations() -> bool:
    """Verify that provided implementations work correctly."""
    # Test reverse functions
    arr1 = [1, 2, 3, 4, 5]
    assert reverse_list_new(arr1) == [5, 4, 3, 2, 1]
    arr2 = [1, 2, 3, 4, 5]
    assert reverse_list_inplace(arr2) == [5, 4, 3, 2, 1]
    
    # Test Fibonacci functions
    assert fibonacci_array(20) == 6765
    assert fibonacci_variables(20) == 6765
    
    # Test matrix transpose
    mat = [[1, 2, 3], [4, 5, 6]]
    transposed = matrix_transpose_new(mat)
    assert transposed == [[1, 4], [2, 5], [3, 6]]
    
    # Test subset generation
    items = [1, 2, 3]
    eager = get_all_subsets_eager(items)
    lazy = list(get_all_subsets_lazy(items))
    assert len(eager) == 8  # 2^3
    assert len(lazy) == 8
    
    logger.info("All verifications passed!")
    return True


def demo() -> None:
    """Demonstrate space complexity concepts."""
    logger.info("=" * 70)
    logger.info("SPACE COMPLEXITY DEMONSTRATION")
    logger.info("=" * 70)
    
    # Demo 1: Reverse comparison
    logger.info("\nDemo 1: List reversal memory comparison")
    for size in [10000, 100000, 1000000]:
        arr_new = list(range(size))
        arr_inplace = list(range(size))
        
        _, mem_new = measure_memory(reverse_list_new, arr_new)
        _, mem_inplace = measure_memory(reverse_list_inplace, arr_inplace)
        
        logger.info(
            f"  n={size:>7}: new={mem_new:.2f}MB, inplace={mem_inplace:.4f}MB"
        )
    
    # Demo 2: Fibonacci memory
    logger.info("\nDemo 2: Fibonacci memory comparison")
    for n in [100, 1000, 10000]:
        _, mem_array = measure_memory(fibonacci_array, n)
        _, mem_vars = measure_memory(fibonacci_variables, n)
        
        logger.info(
            f"  n={n:>5}: array={mem_array:.4f}MB, variables={mem_vars:.6f}MB"
        )
    
    # Demo 3: Subset generation
    logger.info("\nDemo 3: Subset generation comparison")
    items = list(range(15))  # 2^15 = 32768 subsets
    
    _, mem_eager = measure_memory(get_all_subsets_eager, items)
    
    # For lazy, we need to consume the generator
    def consume_lazy() -> int:
        count = 0
        for _ in get_all_subsets_lazy(items):
            count += 1
        return count
    
    _, mem_lazy = measure_memory(consume_lazy)
    
    logger.info(f"  n=15 items (32768 subsets):")
    logger.info(f"    Eager: {mem_eager:.2f}MB")
    logger.info(f"    Lazy:  {mem_lazy:.4f}MB")
    
    logger.info("\n" + "=" * 70)
    logger.info("Complete your exercises to master space complexity!")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Space Complexity Practice Exercise"
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
