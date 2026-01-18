#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Pool.map Usage (Easy)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Master the Pool abstraction for parallel mapping operations, the most common
pattern for embarrassingly parallel workloads.

LEARNING OUTCOMES
─────────────────
- LO2: Implement parallel mapping with Pool.map and Pool.starmap

ESTIMATED TIME: 15 minutes

INSTRUCTIONS
────────────
Complete the TODO sections. Test functions verify your implementations.

HINTS
─────
Hint 1: Pool() creates a pool of worker processes
Hint 2: Use 'with' statement for automatic cleanup
Hint 3: starmap unpacks tuples as positional arguments

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Callable, Sequence, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def square(x: int) -> int:
    """Return square of x."""
    return x * x


def add(a: int, b: int) -> int:
    """Return sum of a and b."""
    return a + b


def parallel_map_basic(
    func: Callable[[T], R],
    items: Sequence[T]
) -> list[R]:
    """
    Apply function to each item in parallel using Pool.map.
    
    TODO: Implement using Pool.map with default worker count.
    
    Args:
        func: Function to apply to each item.
        items: Sequence of input items.
        
    Returns:
        List of results in input order.
        
    Example:
        >>> parallel_map_basic(square, [1, 2, 3, 4])
        [1, 4, 9, 16]
    """
    # TODO: Create Pool using 'with' statement
    # TODO: Use pool.map() to apply func to items
    # TODO: Return results
    pass


def parallel_map_custom_workers(
    func: Callable[[T], R],
    items: Sequence[T],
    n_workers: int
) -> list[R]:
    """
    Apply function with specified number of workers.
    
    TODO: Implement with explicit worker count.
    
    Args:
        func: Function to apply.
        items: Input items.
        n_workers: Number of worker processes.
        
    Returns:
        List of results.
        
    Example:
        >>> parallel_map_custom_workers(square, range(10), 2)
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
    """
    # TODO: Create Pool with processes=n_workers
    # TODO: Map and return results
    pass


def parallel_starmap(
    func: Callable[..., R],
    args_list: Sequence[tuple]
) -> list[R]:
    """
    Apply function to argument tuples using Pool.starmap.
    
    starmap unpacks each tuple as positional arguments to func.
    
    TODO: Implement using pool.starmap().
    
    Args:
        func: Function accepting multiple arguments.
        args_list: List of argument tuples.
        
    Returns:
        List of results.
        
    Example:
        >>> parallel_starmap(add, [(1, 2), (3, 4), (5, 6)])
        [3, 7, 11]
    """
    # TODO: Create Pool
    # TODO: Use pool.starmap() instead of pool.map()
    pass


def parallel_map_with_chunksize(
    func: Callable[[T], R],
    items: Sequence[T],
    chunksize: int
) -> list[R]:
    """
    Apply function with explicit chunk size for better load balancing.
    
    Chunking groups items together, reducing communication overhead
    but potentially causing load imbalance with uneven processing times.
    
    TODO: Implement pool.map with chunksize parameter.
    
    Args:
        func: Function to apply.
        items: Input items.
        chunksize: Number of items per chunk.
        
    Returns:
        List of results.
        
    Example:
        >>> parallel_map_with_chunksize(square, range(100), 10)
        [0, 1, 4, ..., 9801]
    """
    # TODO: Use pool.map(func, items, chunksize=chunksize)
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_parallel_map_basic() -> None:
    """Test basic parallel map."""
    result = parallel_map_basic(square, [1, 2, 3, 4, 5])
    assert result == [1, 4, 9, 16, 25], f"Expected [1,4,9,16,25], got {result}"
    print("✓ parallel_map_basic works")


def test_parallel_map_custom_workers() -> None:
    """Test parallel map with custom worker count."""
    result = parallel_map_custom_workers(square, list(range(10)), 2)
    expected = [i * i for i in range(10)]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ parallel_map_custom_workers works")


def test_parallel_starmap() -> None:
    """Test starmap for multi-argument functions."""
    args = [(1, 2), (3, 4), (5, 6), (10, 20)]
    result = parallel_starmap(add, args)
    assert result == [3, 7, 11, 30], f"Expected [3,7,11,30], got {result}"
    print("✓ parallel_starmap works")


def test_parallel_map_with_chunksize() -> None:
    """Test map with explicit chunksize."""
    result = parallel_map_with_chunksize(square, list(range(20)), 5)
    expected = [i * i for i in range(20)]
    assert result == expected, f"Expected {expected}, got {result}"
    print("✓ parallel_map_with_chunksize works")


def main() -> None:
    """Run all tests."""
    print("Testing easy_02_pool_map.py")
    print("=" * 50)
    print(f"Available CPUs: {cpu_count()}")
    
    test_parallel_map_basic()
    test_parallel_map_custom_workers()
    test_parallel_starmap()
    test_parallel_map_with_chunksize()
    
    print("=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    main()
