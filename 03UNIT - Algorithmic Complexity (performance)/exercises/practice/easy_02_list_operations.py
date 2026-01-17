#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Easy 02 — List Operations Complexity
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Python lists are dynamic arrays with specific complexity characteristics for
different operations. Understanding these helps predict performance.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Identify the complexity of common list operations
2. Measure and verify theoretical complexity empirically
3. Understand amortised complexity for append operations

ESTIMATED TIME
──────────────
25 minutes

DIFFICULTY
──────────
⭐ Easy (1/3)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

import time
from typing import Any
import logging

logger = logging.getLogger(__name__)



# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Predict Complexity
# ═══════════════════════════════════════════════════════════════════════════════

def get_complexity_predictions() -> dict[str, str]:
    """
    Return your predictions for the time complexity of each list operation.

    Operations to analyse:
    - append: Adding to the end
    - insert_front: Inserting at index 0
    - insert_middle: Inserting at index n//2
    - pop_end: Removing from the end
    - pop_front: Removing from index 0
    - access_index: Accessing element by index
    - search_in: Using 'in' operator to check membership

    Returns:
        Dictionary mapping operation name to complexity string.
        Use format: "O(1)", "O(n)", "O(n^2)", "O(log n)", etc.

    Example:
        >>> predictions = get_complexity_predictions()
        >>> predictions['append']  # What is the amortised complexity?
        'O(1)'  # or 'O(n)' — you decide!
    """
    # TODO: Fill in your predictions
    return {
        "append": "O(?)",         # Adding to end
        "insert_front": "O(?)",   # Inserting at index 0
        "insert_middle": "O(?)",  # Inserting at n//2
        "pop_end": "O(?)",        # Removing from end
        "pop_front": "O(?)",      # Removing from index 0
        "access_index": "O(?)",   # Accessing by index
        "search_in": "O(?)",      # Using 'in' operator
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Benchmark List Operations
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_append(n: int) -> float:
    """
    Benchmark n append operations.

    Args:
        n: Number of elements to append.

    Returns:
        Total time in seconds for n append operations.
    """
    # TODO: Create empty list and time n appends
    raise NotImplementedError("Implement this function")


def benchmark_insert_front(n: int) -> float:
    """
    Benchmark n insert-at-front operations.

    Args:
        n: Number of elements to insert at front.

    Returns:
        Total time in seconds for n insert operations.
    """
    # TODO: Create empty list and time n inserts at index 0
    raise NotImplementedError("Implement this function")


def benchmark_access(lst: list[Any], accesses: int) -> float:
    """
    Benchmark random index accesses.

    Args:
        lst: List to access.
        accesses: Number of accesses to perform.

    Returns:
        Total time in seconds for all accesses.
    """
    import random

    # TODO: Time 'accesses' random index lookups
    raise NotImplementedError("Implement this function")


def benchmark_search(lst: list[Any], searches: int) -> float:
    """
    Benchmark membership tests using 'in' operator.

    Args:
        lst: List to search in.
        searches: Number of searches to perform.

    Returns:
        Total time in seconds for all searches.
    """
    import random

    # TODO: Time 'searches' membership tests
    # Search for random values that may or may not be in the list
    raise NotImplementedError("Implement this function")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Empirical Complexity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_complexity(
    sizes: list[int],
    times: list[float],
) -> tuple[float, str]:
    """
    Estimate complexity class from empirical measurements.

    Uses log-log regression: if T(n) = c * n^k, then
    log(T) = log(c) + k * log(n), which is linear with slope k.

    Args:
        sizes: List of input sizes.
        times: List of corresponding execution times.

    Returns:
        Tuple of (estimated exponent k, complexity class string).

    Example:
        >>> sizes = [100, 200, 400, 800]
        >>> times = [0.01, 0.04, 0.16, 0.64]  # Quadratic pattern
        >>> exponent, complexity = analyse_complexity(sizes, times)
        >>> 1.9 < exponent < 2.1  # Should be close to 2
        True
    """
    import math

    # TODO: Implement log-log regression
    # 1. Take log of sizes and times
    # 2. Fit a line (you can use simple least squares)
    # 3. The slope is the exponent k
    # 4. Map k to complexity class:
    #    k ≈ 0 → O(1)
    #    k ≈ 0.5 → O(√n)
    #    k ≈ 1 → O(n)
    #    k ≈ 2 → O(n²)
    raise NotImplementedError("Implement this function")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Compare Append vs Insert Front
# ═══════════════════════════════════════════════════════════════════════════════

def compare_append_vs_insert() -> dict[str, dict[str, float]]:
    """
    Compare the scaling of append vs insert_front operations.

    Run both operations for sizes [1000, 2000, 4000, 8000, 16000] and
    return timing data.

    Returns:
        Dictionary with structure:
        {
            "append": {1000: time, 2000: time, ...},
            "insert_front": {1000: time, 2000: time, ...}
        }
    """
    sizes = [1000, 2000, 4000, 8000, 16000]

    # TODO: Benchmark both operations for each size
    # Return the results in the specified format
    raise NotImplementedError("Implement this function")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _test_implementations() -> None:
    """Test all implementations."""
    logger.info("Testing list operation benchmarks...\n")

    # Test Exercise 1
    logger.info("Exercise 1: Complexity predictions")
    predictions = get_complexity_predictions()
    logger.info("  Your predictions:")
    for op, complexity in predictions.items():
        logger.info(f"    {op}: {complexity}")

    # Test Exercise 2
    logger.info("\nExercise 2: Benchmarks")
    try:
        t_append = benchmark_append(10000)
        logger.info(f"  ✓ benchmark_append(10000): {t_append:.6f}s")
    except NotImplementedError:
        logger.info("  ✗ benchmark_append: Not implemented")

    try:
        t_insert = benchmark_insert_front(5000)
        logger.info(f"  ✓ benchmark_insert_front(5000): {t_insert:.6f}s")
    except NotImplementedError:
        logger.info("  ✗ benchmark_insert_front: Not implemented")

    # Test Exercise 3
    logger.info("\nExercise 3: Complexity analysis")
    try:
        # Test with known quadratic data
        sizes = [100, 200, 400, 800]
        times = [0.01, 0.04, 0.16, 0.64]
        exp, cls = analyse_complexity(sizes, times)
        logger.info(f"  ✓ Test case (quadratic): exponent={exp:.2f}, class={cls}")
    except NotImplementedError:
        logger.info("  ✗ analyse_complexity: Not implemented")

    # Test Exercise 4
    logger.info("\nExercise 4: Compare append vs insert")
    try:
        results = compare_append_vs_insert()
        logger.info("  ✓ Comparison results:")
        for op, data in results.items():
            logger.info(f"    {op}:")
            for size, t in data.items():
                logger.info(f"      n={size}: {t:.6f}s")
    except NotImplementedError:
        logger.info("  ✗ compare_append_vs_insert: Not implemented")

    logger.info("\n" + "=" * 60)
    logger.info("After implementing, observe how insert_front scales much worse!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _test_implementations()
