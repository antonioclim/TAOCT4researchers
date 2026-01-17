#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Easy 01 — Basic Timing
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Accurate timing is the foundation of empirical complexity analysis. This
exercise introduces Python's timing mechanisms and common pitfalls to avoid.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Use time.perf_counter() for high-resolution timing
2. Understand the difference between wall time and CPU time
3. Implement basic warmup and multiple-run timing

ESTIMATED TIME
──────────────
20 minutes

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
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)



# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Basic Timing with perf_counter
# ═══════════════════════════════════════════════════════════════════════════════

def time_function(func: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """
    Time a single execution of a function.

    Args:
        func: The function to time.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Elapsed time in seconds.

    Example:
        >>> def slow_sum(n):
        ...     return sum(range(n))
        >>> elapsed = time_function(slow_sum, 1000000)
        >>> elapsed > 0
        True
    """
    # TODO: Implement timing using time.perf_counter()
    # 1. Record start time
    # 2. Call the function with provided arguments
    # 3. Record end time
    # 4. Return elapsed time
    raise NotImplementedError("Implement this function")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Multiple Runs with Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def time_function_multiple(
    func: Callable[..., Any],
    *args: Any,
    runs: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Time multiple executions and compute statistics.

    Args:
        func: The function to time.
        *args: Positional arguments to pass to the function.
        runs: Number of times to run the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Dictionary with 'mean', 'min', 'max' times in seconds.

    Example:
        >>> def simple_func():
        ...     return sum(range(1000))
        >>> stats = time_function_multiple(simple_func, runs=5)
        >>> 'mean' in stats and 'min' in stats and 'max' in stats
        True
        >>> stats['min'] <= stats['mean'] <= stats['max']
        True
    """
    # TODO: Implement multiple-run timing
    # 1. Create a list to store times
    # 2. Run the function 'runs' times, recording each time
    # 3. Calculate mean, min, max
    # 4. Return as dictionary
    raise NotImplementedError("Implement this function")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Timing with Warmup
# ═══════════════════════════════════════════════════════════════════════════════

def time_function_with_warmup(
    func: Callable[..., Any],
    *args: Any,
    warmup_runs: int = 3,
    measured_runs: int = 10,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Time a function with warmup runs (discarded) followed by measured runs.

    Warmup runs help account for:
    - JIT compilation (in Numba, PyPy)
    - CPU cache warming
    - Initial memory allocation

    Args:
        func: The function to time.
        *args: Positional arguments to pass to the function.
        warmup_runs: Number of warmup runs to discard.
        measured_runs: Number of runs to measure.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Dictionary with 'mean', 'min', 'max' times from measured runs only.

    Example:
        >>> def func():
        ...     return [i**2 for i in range(1000)]
        >>> stats = time_function_with_warmup(func, warmup_runs=2, measured_runs=5)
        >>> stats['min'] <= stats['mean'] <= stats['max']
        True
    """
    # TODO: Implement timing with warmup
    # 1. Run function warmup_runs times (discard results)
    # 2. Run function measured_runs times (record times)
    # 3. Return statistics from measured runs only
    raise NotImplementedError("Implement this function")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Context Manager for Timing
# ═══════════════════════════════════════════════════════════════════════════════

class Timer:
    """
    Context manager for timing code blocks.

    Example:
        >>> with Timer() as t:
        ...     result = sum(range(100000))
        >>> t.elapsed > 0
        True
    """

    def __init__(self) -> None:
        """Initialise the timer."""
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer."""
        # TODO: Record start time and return self
        raise NotImplementedError("Implement this method")

    def __exit__(self, *args: Any) -> None:
        """Stop the timer and calculate elapsed time."""
        # TODO: Calculate elapsed time
        raise NotImplementedError("Implement this method")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _test_implementations() -> None:
    """Test all implementations."""
    logger.info("Testing timing implementations...\n")

    # Test function
    def slow_sum(n: int) -> int:
        return sum(range(n))

    # Test Exercise 1
    logger.info("Exercise 1: Basic timing")
    try:
        elapsed = time_function(slow_sum, 100000)
        logger.info(f"  ✓ time_function: {elapsed:.6f}s")
    except NotImplementedError:
        logger.info("  ✗ time_function: Not implemented")

    # Test Exercise 2
    logger.info("\nExercise 2: Multiple runs")
    try:
        stats = time_function_multiple(slow_sum, 100000, runs=5)
        logger.info(f"  ✓ Mean: {stats['mean']:.6f}s, Min: {stats['min']:.6f}s, Max: {stats['max']:.6f}s")
    except NotImplementedError:
        logger.info("  ✗ time_function_multiple: Not implemented")

    # Test Exercise 3
    logger.info("\nExercise 3: Timing with warmup")
    try:
        stats = time_function_with_warmup(slow_sum, 100000, warmup_runs=2, measured_runs=5)
        logger.info(f"  ✓ Mean: {stats['mean']:.6f}s (after warmup)")
    except NotImplementedError:
        logger.info("  ✗ time_function_with_warmup: Not implemented")

    # Test Exercise 4
    logger.info("\nExercise 4: Timer context manager")
    try:
        with Timer() as t:
            _ = slow_sum(100000)
        logger.info(f"  ✓ Timer elapsed: {t.elapsed:.6f}s")
    except NotImplementedError:
        logger.info("  ✗ Timer: Not implemented")

    logger.info("\n" + "=" * 60)
    logger.info("Complete the exercises by replacing 'raise NotImplementedError'")
    logger.info("with your implementation.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _test_implementations()
