#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Process Basics (Easy)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Implement basic multiprocessing patterns using the Process class to understand
how processes are created, started and joined.

LEARNING OUTCOMES
─────────────────
- LO1: Understand process creation and lifecycle
- LO2: Implement inter-process communication with Queue

ESTIMATED TIME: 15 minutes

INSTRUCTIONS
────────────
Complete the TODO sections below. Each function has a docstring explaining
the expected behaviour and return type.

HINTS
─────
Hint 1: Process requires a callable target and args tuple
Hint 2: Use Queue for returning values from child processes
Hint 3: Always join() processes to wait for completion

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from multiprocessing import Process, Queue
from typing import Any, Callable


def compute_factorial(n: int) -> int:
    """
    Compute factorial of n iteratively.
    
    Args:
        n: Non-negative integer.
        
    Returns:
        n! (factorial of n).
        
    Example:
        >>> compute_factorial(5)
        120
    """
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def spawn_single_process(func: Callable[..., Any], args: tuple[Any, ...]) -> None:
    """
    Create and run a single process executing the given function.
    
    TODO: Implement this function to:
    1. Create a Process with the given target function and arguments
    2. Start the process
    3. Wait for the process to complete
    
    Args:
        func: Function to execute in the child process.
        args: Arguments to pass to the function.
        
    Example:
        >>> spawn_single_process(print, ("Hello from process!",))
        Hello from process!
    """
    # TODO: Create Process instance
    # TODO: Start the process
    # TODO: Wait for completion with join()
    pass


def process_with_queue_result(
    func: Callable[..., Any],
    args: tuple[Any, ...]
) -> Any:
    """
    Execute function in a separate process and return its result.
    
    Since Process cannot directly return values, use a Queue to
    communicate the result back to the parent process.
    
    TODO: Implement this function to:
    1. Create a Queue for receiving the result
    2. Create a wrapper function that puts func's result in the queue
    3. Create and start a Process running the wrapper
    4. Wait for completion and return the result from the queue
    
    Args:
        func: Function to execute.
        args: Arguments for the function.
        
    Returns:
        The result of func(*args).
        
    Example:
        >>> result = process_with_queue_result(compute_factorial, (5,))
        >>> result
        120
    """
    # TODO: Create a Queue
    # TODO: Define wrapper function that calls func and puts result in queue
    # TODO: Create and start Process with wrapper
    # TODO: Join and return result from queue
    pass


def run_parallel_processes(
    func: Callable[..., Any],
    args_list: list[tuple[Any, ...]]
) -> list[Any]:
    """
    Execute function with multiple argument sets in parallel processes.
    
    TODO: Implement this function to:
    1. Create a shared Queue for all results
    2. Create a Process for each set of arguments
    3. Start all processes
    4. Wait for all to complete
    5. Collect and return all results
    
    Note: Results may not be in the same order as args_list due to
    non-deterministic scheduling.
    
    Args:
        func: Function to execute.
        args_list: List of argument tuples, one per process.
        
    Returns:
        List of results (order may vary).
        
    Example:
        >>> results = run_parallel_processes(compute_factorial, [(3,), (4,), (5,)])
        >>> sorted(results)
        [6, 24, 120]
    """
    # TODO: Create shared Queue
    # TODO: Define wrapper that puts (args, result) in queue
    # TODO: Create all processes
    # TODO: Start all processes
    # TODO: Join all processes
    # TODO: Collect results from queue
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_spawn_single_process() -> None:
    """Test that spawn_single_process runs without error."""
    # This should execute without raising
    spawn_single_process(compute_factorial, (5,))
    print("✓ spawn_single_process works")


def test_process_with_queue_result() -> None:
    """Test that process_with_queue_result returns correct value."""
    result = process_with_queue_result(compute_factorial, (6,))
    assert result == 720, f"Expected 720, got {result}"
    print("✓ process_with_queue_result returns correct value")


def test_run_parallel_processes() -> None:
    """Test that run_parallel_processes returns all results."""
    results = run_parallel_processes(
        compute_factorial,
        [(3,), (4,), (5,), (6,)]
    )
    assert sorted(results) == [6, 24, 120, 720], f"Unexpected results: {results}"
    print("✓ run_parallel_processes returns all results")


def main() -> None:
    """Run all tests."""
    print("Testing easy_01_process_basics.py")
    print("=" * 50)
    
    test_spawn_single_process()
    test_process_with_queue_result()
    test_run_parallel_processes()
    
    print("=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    main()
