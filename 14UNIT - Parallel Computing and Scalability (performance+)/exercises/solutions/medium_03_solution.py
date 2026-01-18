#!/usr/bin/env python3
"""
14UNIT Exercise Solution: concurrent.futures Patterns (Medium)
© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
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
    """Apply function to items in parallel using ProcessPoolExecutor."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, items))
    return results


def process_as_completed(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 4
) -> list[tuple[T, R]]:
    """Process items and yield results as they complete."""
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(func, item): item for item in items}
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            result = future.result()
            results.append((item, result))
    return results


def parallel_with_timeout(
    func: Callable[[T], R],
    items: Sequence[T],
    timeout_per_task: float = 1.0,
    max_workers: int = 4
) -> list[R | None]:
    """Execute tasks with timeout handling."""
    results: list[R | None] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        for future in futures:
            try:
                result = future.result(timeout=timeout_per_task)
                results.append(result)
            except TimeoutError:
                results.append(None)
    return results


def parallel_with_exceptions(
    func: Callable[[T], R],
    items: Sequence[T],
    max_workers: int = 4
) -> list[R | Exception]:
    """Execute tasks, capturing exceptions instead of raising."""
    results: list[R | Exception] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        for future in futures:
            exc = future.exception()
            if exc is not None:
                results.append(exc)
            else:
                results.append(future.result())
    return results


if __name__ == '__main__':
    print(parallel_map_executor(slow_square, [1, 2, 3, 4]))
