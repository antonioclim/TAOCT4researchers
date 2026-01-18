#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Pool.map Usage - SOLUTION
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
from typing import Callable, Sequence, TypeVar

T = TypeVar('T')
R = TypeVar('R')


def square(x: int) -> int:
    return x * x


def add(a: int, b: int) -> int:
    return a + b


def parallel_map_basic(
    func: Callable[[T], R],
    items: Sequence[T]
) -> list[R]:
    """Apply function using Pool.map with default workers."""
    with Pool() as pool:
        results = pool.map(func, items)
    return results


def parallel_map_custom_workers(
    func: Callable[[T], R],
    items: Sequence[T],
    n_workers: int
) -> list[R]:
    """Apply function with specified worker count."""
    with Pool(processes=n_workers) as pool:
        results = pool.map(func, items)
    return results


def parallel_starmap(
    func: Callable[..., R],
    args_list: Sequence[tuple]
) -> list[R]:
    """Apply function to argument tuples using starmap."""
    with Pool() as pool:
        results = pool.starmap(func, args_list)
    return results


def parallel_map_with_chunksize(
    func: Callable[[T], R],
    items: Sequence[T],
    chunksize: int
) -> list[R]:
    """Apply function with explicit chunk size."""
    with Pool() as pool:
        results = pool.map(func, items, chunksize=chunksize)
    return results


def test_parallel_map_basic() -> None:
    result = parallel_map_basic(square, [1, 2, 3, 4, 5])
    assert result == [1, 4, 9, 16, 25]
    print("✓ parallel_map_basic works")


def test_parallel_map_custom_workers() -> None:
    result = parallel_map_custom_workers(square, list(range(10)), 2)
    assert result == [i * i for i in range(10)]
    print("✓ parallel_map_custom_workers works")


def test_parallel_starmap() -> None:
    result = parallel_starmap(add, [(1, 2), (3, 4), (5, 6), (10, 20)])
    assert result == [3, 7, 11, 30]
    print("✓ parallel_starmap works")


def test_parallel_map_with_chunksize() -> None:
    result = parallel_map_with_chunksize(square, list(range(20)), 5)
    assert result == [i * i for i in range(20)]
    print("✓ parallel_map_with_chunksize works")


if __name__ == '__main__':
    print("Testing easy_02_solution.py")
    print("=" * 50)
    test_parallel_map_basic()
    test_parallel_map_custom_workers()
    test_parallel_starmap()
    test_parallel_map_with_chunksize()
    print("All tests passed!")
