#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Recursion Analysis (Medium)
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Recursive algorithms often have non-obvious complexity characteristics. Analysing
their time and space complexity requires understanding recurrence relations and
the structure of the recursion tree. This exercise develops your ability to
reason about recursive performance both theoretically and empirically.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Identify recurrence relations from recursive code
2. Solve basic recurrence relations using substitution
3. Measure actual performance to validate theoretical analysis

ESTIMATED TIME
──────────────
- Reading: 10 minutes
- Coding: 30 minutes
- Total: 40 minutes

DIFFICULTY: ⭐⭐⭐ (Medium)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RECURSIVE ALGORITHMS TO ANALYSE
# ═══════════════════════════════════════════════════════════════════════════════

def fibonacci_naive(n: int) -> int:
    """
    Compute the nth Fibonacci number using naive recursion.
    
    Args:
        n: The index in the Fibonacci sequence (0-indexed).
        
    Returns:
        The nth Fibonacci number.
        
    Recurrence Relation:
        T(n) = T(n-1) + T(n-2) + O(1)
        
    TODO: Analyse - What is the time complexity? Space complexity?
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memoised(n: int, memo: dict[int, int] | None = None) -> int:
    """
    Compute the nth Fibonacci number using memoisation.
    
    Args:
        n: The index in the Fibonacci sequence (0-indexed).
        memo: Dictionary for memoisation (created if None).
        
    Returns:
        The nth Fibonacci number.
        
    TODO: Analyse - How does memoisation change the complexity?
    """
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    result = fibonacci_memoised(n - 1, memo) + fibonacci_memoised(n - 2, memo)
    memo[n] = result
    return result


def binary_search_recursive(arr: list[int], target: int, low: int, high: int) -> int:
    """
    Find target in sorted array using recursive binary search.
    
    Args:
        arr: Sorted list of integers.
        target: Value to find.
        low: Lower bound index.
        high: Upper bound index.
        
    Returns:
        Index of target if found, -1 otherwise.
        
    Recurrence Relation:
        T(n) = T(n/2) + O(1)
        
    TODO: Solve this recurrence using the Master Theorem.
    """
    if low > high:
        return -1
    
    mid = (low + high) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, high)
    else:
        return binary_search_recursive(arr, target, low, mid - 1)


def merge_sort_recursive(arr: list[int]) -> list[int]:
    """
    Sort array using recursive merge sort.
    
    Args:
        arr: List of integers to sort.
        
    Returns:
        Sorted list.
        
    Recurrence Relation:
        T(n) = 2T(n/2) + O(n)
        
    TODO: Verify this is O(n log n) empirically.
    """
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort_recursive(arr[:mid])
    right = merge_sort_recursive(arr[mid:])
    
    return _merge(left, right)


def _merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists."""
    result: list[int] = []
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


def power_naive(base: float, exp: int) -> float:
    """
    Compute base^exp using naive recursion.
    
    Args:
        base: The base number.
        exp: The exponent (non-negative integer).
        
    Returns:
        base raised to the power of exp.
        
    Recurrence Relation:
        T(n) = T(n-1) + O(1)
        
    TODO: What is the complexity? How can we improve it?
    """
    if exp == 0:
        return 1.0
    return base * power_naive(base, exp - 1)


def power_fast(base: float, exp: int) -> float:
    """
    Compute base^exp using fast exponentiation (binary exponentiation).
    
    Args:
        base: The base number.
        exp: The exponent (non-negative integer).
        
    Returns:
        base raised to the power of exp.
        
    Recurrence Relation:
        T(n) = T(n/2) + O(1)
        
    TODO: Verify this is O(log n).
    """
    if exp == 0:
        return 1.0
    
    if exp % 2 == 0:
        half = power_fast(base, exp // 2)
        return half * half
    else:
        return base * power_fast(base, exp - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: YOUR TASKS
# ═══════════════════════════════════════════════════════════════════════════════

def count_recursive_calls(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, int]:
    """
    EXERCISE 1: Implement a function that counts recursive calls.
    
    This function should wrap a recursive function and count how many times
    it is called during execution.
    
    Args:
        func: The recursive function to wrap.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Tuple of (result, call_count).
        
    Hints:
        - You may need to modify the approach for different functions
        - Consider using a mutable container to track count
        - Alternatively, create a wrapper class
        
    Example:
        >>> result, calls = count_recursive_calls(fibonacci_naive, 10)
        >>> print(f"fib(10) = {result}, calls = {calls}")
    """
    # TODO: Implement this function
    # Hint: One approach is to use a closure with a mutable counter
    
    call_count = [0]  # Mutable container to track calls
    
    # Your implementation here
    raise NotImplementedError("Implement count_recursive_calls")


def measure_recursion_depth(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, int]:
    """
    EXERCISE 2: Measure the maximum recursion depth reached.
    
    This is related to space complexity - each recursive call uses stack space.
    
    Args:
        func: The recursive function to analyse.
        *args: Arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        Tuple of (result, max_depth).
        
    Note:
        Python's default recursion limit is 1000. You may need to adjust
        this for large inputs using sys.setrecursionlimit().
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement measure_recursion_depth")


def analyse_fibonacci_complexity() -> dict[str, list[tuple[int, float]]]:
    """
    EXERCISE 3: Empirically analyse Fibonacci implementations.
    
    Compare the time complexity of fibonacci_naive vs fibonacci_memoised
    by measuring execution time for various input sizes.
    
    Returns:
        Dictionary with:
        - "naive": List of (n, time_seconds) tuples
        - "memoised": List of (n, time_seconds) tuples
        
    Requirements:
        1. Test naive version for n in [5, 10, 15, 20, 25, 30]
        2. Test memoised version for n in [10, 50, 100, 500, 1000]
        3. Use time.perf_counter() for accurate timing
        4. Log results as you measure
        
    Questions to answer (in comments):
        - What is the growth rate of the naive version?
        - How much faster is the memoised version?
        - What would happen to naive at n=50?
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement analyse_fibonacci_complexity")


def compare_power_functions() -> dict[str, list[tuple[int, float]]]:
    """
    EXERCISE 4: Compare naive vs fast exponentiation.
    
    Returns:
        Dictionary with timing data for both implementations.
        
    Test both functions with:
        - base = 1.0001 (to avoid overflow)
        - exp in [100, 1000, 10000, 100000, 1000000]
        
    Calculate the ratio of times and verify it matches the theoretical
    complexity difference (O(n) vs O(log n)).
    """
    # TODO: Implement this function
    raise NotImplementedError("Implement compare_power_functions")


def derive_recurrence(algorithm_name: str) -> str:
    """
    EXERCISE 5: Write recurrence relations for given algorithms.
    
    For each algorithm, provide:
    1. The recurrence relation T(n) = ...
    2. The solution (closed form or Big-O)
    3. Brief justification
    
    Args:
        algorithm_name: One of "fibonacci_naive", "binary_search", 
                       "merge_sort", "power_naive", "power_fast"
                       
    Returns:
        String containing the analysis (multiline).
        
    Example output for binary_search:
        '''
        Algorithm: Binary Search
        Recurrence: T(n) = T(n/2) + O(1)
        Solution: O(log n)
        Justification: Each call halves the problem size, doing constant
        work at each level. The number of levels is log₂(n).
        '''
    """
    # TODO: Implement this function with analyses for all algorithms
    raise NotImplementedError("Implement derive_recurrence")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: BONUS CHALLENGES
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_recursion_tree(func_name: str, n: int) -> dict[str, Any]:
    """
    BONUS 1: Visualise the recursion tree structure.
    
    For a given recursive function and input n, build a representation
    of the recursion tree showing:
    - Total number of nodes (function calls)
    - Maximum depth (height of tree)
    - Number of nodes at each level
    
    Args:
        func_name: Name of function ("fibonacci", "merge_sort", etc.)
        n: Input size.
        
    Returns:
        Dictionary with tree statistics.
    """
    # TODO: Implement this bonus challenge
    raise NotImplementedError("Implement analyse_recursion_tree")


def tail_recursion_optimisation() -> None:
    """
    BONUS 2: Implement tail-recursive versions.
    
    Python doesn't optimise tail recursion, but understanding the concept
    is valuable. Implement tail-recursive versions of:
    1. Factorial
    2. Fibonacci
    3. Sum of list
    
    Compare stack depth with non-tail-recursive versions.
    """
    # TODO: Implement tail-recursive functions and compare
    raise NotImplementedError("Implement tail_recursion_optimisation")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: VERIFICATION AND DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def verify_implementations() -> bool:
    """Verify that implementations are correct."""
    # Test Fibonacci
    assert fibonacci_naive(10) == 55, "fibonacci_naive(10) should be 55"
    assert fibonacci_memoised(10) == 55, "fibonacci_memoised(10) should be 55"
    
    # Test binary search
    arr = list(range(100))
    assert binary_search_recursive(arr, 42, 0, 99) == 42
    assert binary_search_recursive(arr, 101, 0, 99) == -1
    
    # Test merge sort
    import random
    test_arr = random.sample(range(1000), 100)
    assert merge_sort_recursive(test_arr) == sorted(test_arr)
    
    # Test power functions
    assert power_naive(2, 10) == 1024
    assert power_fast(2, 10) == 1024
    assert abs(power_fast(2, 100) - 2**100) < 1e-6
    
    logger.info("All verifications passed!")
    return True


def demo() -> None:
    """Demonstrate recursion analysis concepts."""
    logger.info("=" * 70)
    logger.info("RECURSION ANALYSIS DEMONSTRATION")
    logger.info("=" * 70)
    
    # Demo 1: Timing comparison
    logger.info("\nDemo 1: Fibonacci timing comparison")
    
    for n in [20, 25, 30]:
        start = time.perf_counter()
        result = fibonacci_naive(n)
        naive_time = time.perf_counter() - start
        
        start = time.perf_counter()
        result_memo = fibonacci_memoised(n)
        memo_time = time.perf_counter() - start
        
        speedup = naive_time / memo_time if memo_time > 0 else float('inf')
        logger.info(
            f"  n={n}: naive={naive_time:.4f}s, memo={memo_time:.6f}s, "
            f"speedup={speedup:.0f}x"
        )
    
    # Demo 2: Power function comparison
    logger.info("\nDemo 2: Power function comparison")
    
    base = 1.0001
    for exp in [1000, 10000, 100000]:
        start = time.perf_counter()
        _ = power_naive(base, exp)
        naive_time = time.perf_counter() - start
        
        start = time.perf_counter()
        _ = power_fast(base, exp)
        fast_time = time.perf_counter() - start
        
        logger.info(
            f"  exp={exp}: naive={naive_time:.4f}s, fast={fast_time:.6f}s"
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("Complete your exercises to fully understand recursion analysis!")
    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Recursion Analysis Practice Exercise"
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
