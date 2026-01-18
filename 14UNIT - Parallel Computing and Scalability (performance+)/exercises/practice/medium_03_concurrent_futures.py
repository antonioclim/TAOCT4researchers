#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: concurrent.futures Patterns (Medium)
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ★★★☆☆ (Medium)
ESTIMATED TIME: 25 minutes
PREREQUISITES: Basic multiprocessing, Future concept

LEARNING OBJECTIVES
───────────────────
- LO4: Use concurrent.futures for high-level parallel execution
- LO4: Handle Future objects and exceptions properly

PROBLEM DESCRIPTION
───────────────────
Implement parallel execution patterns using the concurrent.futures module,
which provides a cleaner interface than raw multiprocessing.

TASKS
─────
1. Implement `parallel_map_executor` - map using ProcessPoolExecutor
2. Implement `process_as_completed` - handle results as they complete
3. Implement `parallel_with_timeout` - handle timeouts gracefully

HINTS
─────
- Hint 1: Use 'with ProcessPoolExecutor() as executor:' for cleanup
- Hint 2: as_completed yields futures as they finish (not input order)
- Hint 3: future.result(timeout=n) raises TimeoutError if too slow

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from concurrent.futures import (
    ProcessPoolExecutor,
    TimeoutError,
    as_completed,
)
from typing import Any, Callable, Sequence, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def slow_square(x: int) -> int:
    """Square a number with simulated delay."""
    time.sleep(0.1)
    return x * x


def parallel_map_executor(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 4
) -> list[R]:
    """
    Apply function to items in parallel using ProcessPoolExecutor.
    
    Results should be returned in the same order as input items.
    
    Args:
        func: Function to apply to each item.
        items: Input sequence.
        max_workers: Maximum number of worker processes.
        
    Returns:
        List of results in input order.
        
    Example:
        >>> parallel_map_executor(slow_square, [1, 2, 3, 4], max_workers=2)
        [1, 4, 9, 16]
    """
    # TODO: Implement this function
    # Use executor.map() which preserves order
    raise NotImplementedError("Implement parallel_map_executor")


def process_as_completed(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 4
) -> list[tuple[T, R]]:
    """
    Process items and yield results as they complete.
    
    Unlike map, as_completed returns results in completion order,
    which may differ from input order.
    
    Args:
        func: Function to apply.
        items: Input sequence.
        max_workers: Maximum workers.
        
    Returns:
        List of (input, result) tuples in completion order.
        
    Example:
        >>> results = process_as_completed(slow_square, [1, 2, 3])
        >>> sorted(results)  # Sort to compare
        [(1, 1), (2, 4), (3, 9)]
    """
    # TODO: Implement this function
    # 1. Submit all tasks, keeping track of which future maps to which input
    # 2. Use as_completed to iterate through completed futures
    # 3. Return list of (input, result) tuples
    raise NotImplementedError("Implement process_as_completed")


def parallel_with_timeout(
    func: Callable[[T], R],
    items: Sequence[T],
    timeout_per_task: float = 1.0,
    max_workers: int = 4
) -> list[R | None]:
    """
    Execute tasks with timeout handling.
    
    If a task exceeds the timeout, return None for that result
    instead of raising an exception.
    
    Args:
        func: Function to apply.
        items: Input sequence.
        timeout_per_task: Maximum seconds per task.
        max_workers: Maximum workers.
        
    Returns:
        List of results (None for timed-out tasks) in input order.
        
    Example:
        >>> def maybe_slow(x):
        ...     time.sleep(x)  # Sleep for x seconds
        ...     return x * 2
        >>> parallel_with_timeout(maybe_slow, [0.1, 5.0, 0.2], timeout_per_task=1.0)
        [0.2, None, 0.4]  # Middle task times out
    """
    # TODO: Implement this function
    # 1. Submit all tasks
    # 2. For each future (in input order), try to get result with timeout
    # 3. Catch TimeoutError and return None for that position
    raise NotImplementedError("Implement parallel_with_timeout")


def parallel_with_exceptions(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 4
) -> list[R | Exception]:
    """
    Execute tasks, capturing exceptions instead of raising.
    
    If a task raises an exception, capture it in the result list
    rather than propagating it.
    
    Args:
        func: Function to apply.
        items: Input sequence.
        max_workers: Maximum workers.
        
    Returns:
        List of results or Exception objects in input order.
        
    Example:
        >>> def risky(x):
        ...     if x == 0:
        ...         raise ValueError("Cannot process zero")
        ...     return 10 / x
        >>> results = parallel_with_exceptions(risky, [2, 0, 5])
        >>> results[0]
        5.0
        >>> isinstance(results[1], ValueError)
        True
    """
    # TODO: Implement this function
    # Use future.exception() to check for exceptions
    raise NotImplementedError("Implement parallel_with_exceptions")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_parallel_map_executor() -> None:
    """Test basic parallel map."""
    result = parallel_map_executor(slow_square, [1, 2, 3, 4], max_workers=2)
    assert result == [1, 4, 9, 16]
    print("✓ parallel_map_executor tests passed")


def test_process_as_completed() -> None:
    """Test as_completed processing."""
    results = process_as_completed(slow_square, [1, 2, 3], max_workers=2)
    # Check all items processed (order may vary)
    assert sorted(results) == [(1, 1), (2, 4), (3, 9)]
    print("✓ process_as_completed tests passed")


def test_parallel_with_timeout() -> None:
    """Test timeout handling."""
    def variable_delay(x: int) -> int:
        time.sleep(0.1 * x)
        return x * 2
    
    # With generous timeout, all succeed
    results = parallel_with_timeout(variable_delay, [1, 2, 3], timeout_per_task=2.0)
    assert results == [2, 4, 6]
    print("✓ parallel_with_timeout tests passed")


def test_parallel_with_exceptions() -> None:
    """Test exception handling."""
    def risky(x: int) -> float:
        if x == 0:
            raise ValueError("Zero!")
        return 10.0 / x
    
    results = parallel_with_exceptions(risky, [2, 0, 5], max_workers=2)
    assert results[0] == 5.0
    assert isinstance(results[1], ValueError)
    assert results[2] == 2.0
    print("✓ parallel_with_exceptions tests passed")


def main() -> None:
    """Run all tests."""
    print("Running medium_03_concurrent_futures tests...")
    print("-" * 50)
    
    try:
        test_parallel_map_executor()
        test_process_as_completed()
        test_parallel_with_timeout()
        test_parallel_with_exceptions()
        print("-" * 50)
        print("All tests passed! ✓")
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}")
    except AssertionError as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    main()
