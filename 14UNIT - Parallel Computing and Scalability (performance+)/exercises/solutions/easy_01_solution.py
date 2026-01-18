#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Process Basics - SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Complete solutions with explanations for each function.

Author: Antonio Clim
Licence: Restrictive - See README.md
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from multiprocessing import Process, Queue
from typing import Any, Callable


def compute_factorial(n: int) -> int:
    """Compute factorial of n iteratively."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def spawn_single_process(func: Callable[..., Any], args: tuple[Any, ...]) -> None:
    """
    Create and run a single process executing the given function.
    
    Solution Notes:
    - Process takes target= and args= parameters
    - start() begins execution in separate process
    - join() blocks until process completes
    """
    # Create Process with target function and arguments
    process = Process(target=func, args=args)
    
    # Start the process (begins execution in child process)
    process.start()
    
    # Wait for process to complete before returning
    process.join()


def process_with_queue_result(
    func: Callable[..., Any],
    args: tuple[Any, ...]
) -> Any:
    """
    Execute function in a separate process and return its result.
    
    Solution Notes:
    - Queue enables inter-process communication
    - Wrapper function executes target and puts result in queue
    - Parent retrieves result after join() completes
    """
    # Create Queue for returning result
    result_queue: Queue[Any] = Queue()
    
    # Wrapper puts result in queue after execution
    def wrapper(f: Callable[..., Any], a: tuple[Any, ...], q: Queue[Any]) -> None:
        result = f(*a)
        q.put(result)
    
    # Create and start process with wrapper
    process = Process(target=wrapper, args=(func, args, result_queue))
    process.start()
    process.join()
    
    # Retrieve and return result
    return result_queue.get()


def run_parallel_processes(
    func: Callable[..., Any],
    args_list: list[tuple[Any, ...]]
) -> list[Any]:
    """
    Execute function with multiple argument sets in parallel processes.
    
    Solution Notes:
    - All processes share a single Queue for results
    - Processes started together run concurrently
    - Results collected after all processes complete
    - Order is non-deterministic due to scheduling
    """
    # Shared queue for all results
    result_queue: Queue[Any] = Queue()
    
    # Wrapper that puts result in shared queue
    def wrapper(f: Callable[..., Any], a: tuple[Any, ...], q: Queue[Any]) -> None:
        result = f(*a)
        q.put(result)
    
    # Create all processes
    processes = [
        Process(target=wrapper, args=(func, args, result_queue))
        for args in args_list
    ]
    
    # Start all processes (they run concurrently)
    for p in processes:
        p.start()
    
    # Wait for all to complete
    for p in processes:
        p.join()
    
    # Collect all results from queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_spawn_single_process() -> None:
    """Test that spawn_single_process runs without error."""
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
    print("Testing easy_01_solution.py")
    print("=" * 50)
    
    test_spawn_single_process()
    test_process_with_queue_result()
    test_run_parallel_processes()
    
    print("=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    main()
