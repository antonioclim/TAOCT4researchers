#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
14UNIT, Lab 01: Multiprocessing and Threading
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Modern computational research frequently encounters workloads that exceed
single-processor capabilities. This laboratory develops practical skills for
parallel computing in Python, addressing the Global Interpreter Lock (GIL)
constraint through multiprocessing and demonstrating when threading provides
genuine benefit for I/O-bound workloads.

HISTORICAL MOTIVATION
─────────────────────
In 1967, Gene Amdahl formalised the theoretical limits of parallel speedup.
His observation—that sequential portions of a program bound achievable
speedup regardless of processor count—remains foundational. If 5% of a
program is inherently sequential, no amount of parallelism can achieve
more than 20× speedup.

Python's Global Interpreter Lock, introduced for memory management simplicity,
creates a practical constraint beyond Amdahl's theoretical limit. For CPU-bound
work, the GIL restricts threads to sequential execution; multiprocessing
provides the escape route through separate interpreter processes.

PREREQUISITES
─────────────
- 03UNIT: Complexity analysis, Big-O notation
- 08UNIT: Divide-and-conquer decomposition
- Python: Functions, classes, context managers

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Distinguish CPU-bound from I/O-bound workloads and select appropriate strategies
2. Implement parallel execution using Process and Pool
3. Handle shared state safely with synchronisation primitives
4. Implement threading for I/O-bound tasks

ESTIMATED TIME
──────────────
- Reading: 20 minutes
- Coding: 35 minutes
- Total: 55 minutes

DEPENDENCIES
────────────
- numpy>=1.24
- requests>=2.31 (for I/O examples)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import (
    Array,
    Lock,
    Manager,
    Pool,
    Process,
    Queue,
    Value,
    cpu_count,
)
from queue import Empty
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
)

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Type aliases
T = TypeVar('T')
R = TypeVar('R')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROCESS FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════

def cpu_intensive_task(n: int) -> int:
    """
    Perform CPU-intensive computation (sum of squares).
    
    This function serves as a benchmark for demonstrating the difference
    between threading and multiprocessing for CPU-bound work.
    
    Args:
        n: Upper limit for summation.
        
    Returns:
        Sum of squares from 0 to n-1.
        
    Complexity:
        Time: O(n)
        Space: O(1)
        
    Example:
        >>> cpu_intensive_task(5)
        30
    """
    total = 0
    for i in range(n):
        total += i * i
    return total


def spawn_process(
    target: Callable[..., Any],
    args: tuple[Any, ...] = ()
) -> None:
    """
    Spawn a single process and wait for completion.
    
    Demonstrates basic Process creation and lifecycle management.
    
    Args:
        target: Function to execute in the child process.
        args: Arguments to pass to the target function.
        
    Example:
        >>> spawn_process(cpu_intensive_task, (1000000,))
    """
    logger.info(f"Spawning process for {target.__name__}")
    
    process = Process(target=target, args=args)
    start_time = time.perf_counter()
    
    process.start()
    logger.info(f"Process started with PID {process.pid}")
    
    process.join()  # Wait for completion
    
    elapsed = time.perf_counter() - start_time
    logger.info(f"Process completed in {elapsed:.3f}s")


def process_with_return_value(
    func: Callable[[T], R],
    arg: T
) -> R:
    """
    Execute function in separate process and retrieve result.
    
    Since Process does not directly return values, we use a Queue
    for inter-process communication.
    
    Args:
        func: Function to execute.
        arg: Argument to pass to function.
        
    Returns:
        Result from the function execution.
        
    Example:
        >>> result = process_with_return_value(cpu_intensive_task, 1000)
        >>> result
        332833500
    """
    result_queue: Queue[R] = Queue()
    
    def wrapper(f: Callable[[T], R], a: T, q: Queue[R]) -> None:
        """Wrapper that puts result in queue."""
        result = f(a)
        q.put(result)
    
    process = Process(target=wrapper, args=(func, arg, result_queue))
    process.start()
    process.join()
    
    return result_queue.get()


def demonstrate_gil() -> dict[str, float]:
    """
    Demonstrate the Global Interpreter Lock's impact on CPU-bound threading.
    
    Compares execution time for:
    1. Sequential execution
    2. Threaded execution (limited by GIL)
    3. Multiprocessing execution (true parallelism)
    
    Returns:
        Dictionary with timing results for each approach.
        
    Note:
        Threading shows no speedup (or slowdown) for CPU-bound work due to GIL.
        Multiprocessing achieves near-linear speedup for embarrassingly parallel tasks.
    """
    n = 10_000_000
    num_tasks = 4
    
    results: dict[str, float] = {}
    
    # Sequential baseline
    logger.info("Running sequential baseline...")
    start = time.perf_counter()
    for _ in range(num_tasks):
        cpu_intensive_task(n)
    results['sequential'] = time.perf_counter() - start
    
    # Threading (GIL-limited)
    logger.info("Running with threads (GIL-limited)...")
    start = time.perf_counter()
    threads = [
        threading.Thread(target=cpu_intensive_task, args=(n,))
        for _ in range(num_tasks)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    results['threading'] = time.perf_counter() - start
    
    # Multiprocessing (true parallelism)
    logger.info("Running with multiprocessing...")
    start = time.perf_counter()
    with Pool(processes=num_tasks) as pool:
        pool.map(cpu_intensive_task, [n] * num_tasks)
    results['multiprocessing'] = time.perf_counter() - start
    
    # Report results
    logger.info("=" * 50)
    logger.info("GIL Demonstration Results:")
    logger.info(f"  Sequential:      {results['sequential']:.3f}s")
    logger.info(f"  Threading:       {results['threading']:.3f}s "
                f"(speedup: {results['sequential']/results['threading']:.2f}x)")
    logger.info(f"  Multiprocessing: {results['multiprocessing']:.3f}s "
                f"(speedup: {results['sequential']/results['multiprocessing']:.2f}x)")
    logger.info("=" * 50)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PROCESS POOL PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def parallel_map(
    func: Callable[[T], R],
    iterable: Sequence[T],
    n_workers: int | None = None
) -> list[R]:
    """
    Apply function to each element in parallel using Pool.map.
    
    Pool.map provides the simplest parallel mapping interface, distributing
    work across a pool of worker processes.
    
    Args:
        func: Function to apply to each element.
        iterable: Input elements.
        n_workers: Number of worker processes (default: CPU count).
        
    Returns:
        List of results in input order.
        
    Complexity:
        Time: O(n/p) where n is input size, p is worker count (ideal case)
        Space: O(n) for results storage
        
    Example:
        >>> parallel_map(lambda x: x**2, [1, 2, 3, 4])
        [1, 4, 9, 16]
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    logger.info(f"parallel_map with {n_workers} workers on {len(iterable)} items")
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(func, iterable)
    
    return results


def parallel_starmap(
    func: Callable[..., R],
    args_list: Sequence[tuple[Any, ...]],
    n_workers: int | None = None
) -> list[R]:
    """
    Apply function to argument tuples in parallel using Pool.starmap.
    
    Unlike map, starmap unpacks each tuple as positional arguments,
    enabling functions with multiple parameters.
    
    Args:
        func: Function accepting multiple arguments.
        args_list: List of argument tuples.
        n_workers: Number of worker processes.
        
    Returns:
        List of results in input order.
        
    Example:
        >>> def add(a, b): return a + b
        >>> parallel_starmap(add, [(1, 2), (3, 4), (5, 6)])
        [3, 7, 11]
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    logger.info(f"parallel_starmap with {n_workers} workers")
    
    with Pool(processes=n_workers) as pool:
        results = pool.starmap(func, args_list)
    
    return results


@dataclass
class AsyncResult(Generic[R]):
    """Container for asynchronous computation result."""
    
    value: R | None = None
    error: Exception | None = None
    completed: bool = False
    
    @property
    def successful(self) -> bool:
        """Check if computation completed without error."""
        return self.completed and self.error is None


def parallel_async_with_callback(
    func: Callable[[T], R],
    items: Sequence[T],
    callback: Callable[[R], None] | None = None,
    n_workers: int | None = None
) -> list[R]:
    """
    Execute tasks asynchronously with optional callback.
    
    apply_async enables non-blocking submission with callbacks for
    result processing. Useful when results should be processed as
    they become available rather than waiting for all to complete.
    
    Args:
        func: Function to apply.
        items: Input elements.
        callback: Optional function called with each result.
        n_workers: Number of worker processes.
        
    Returns:
        List of results (order may not match input).
        
    Example:
        >>> results = parallel_async_with_callback(
        ...     lambda x: x**2,
        ...     [1, 2, 3],
        ...     callback=print
        ... )
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    results: list[R] = []
    
    def collect_result(result: R) -> None:
        """Collect result and optionally invoke callback."""
        results.append(result)
        if callback:
            callback(result)
    
    with Pool(processes=n_workers) as pool:
        async_results = [
            pool.apply_async(func, (item,), callback=collect_result)
            for item in items
        ]
        
        # Wait for all to complete
        for ar in async_results:
            ar.wait()
    
    return results


def chunk_and_process(
    data: Sequence[T],
    func: Callable[[Sequence[T]], R],
    chunk_size: int,
    n_workers: int | None = None
) -> list[R]:
    """
    Divide data into chunks and process each chunk in parallel.
    
    Chunking reduces per-task overhead by grouping work. Essential
    when individual items are too small to justify process communication.
    
    Args:
        data: Input sequence to process.
        func: Function applied to each chunk.
        chunk_size: Size of each chunk.
        n_workers: Number of worker processes.
        
    Returns:
        List of results from processing each chunk.
        
    Example:
        >>> def sum_chunk(chunk): return sum(chunk)
        >>> chunk_and_process(range(100), sum_chunk, chunk_size=10)
        [45, 145, 245, 345, 445, 545, 645, 745, 845, 945]
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    # Create chunks
    chunks: list[Sequence[T]] = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    
    logger.info(f"Processing {len(chunks)} chunks of size ~{chunk_size}")
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(func, chunks)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MONTE CARLO PARALLELISATION
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_pi_sequential(n_points: int) -> float:
    """
    Estimate π using Monte Carlo sampling (sequential).
    
    The dartboard method: random points in unit square, count those
    inside the inscribed quarter-circle. Ratio approximates π/4.
    
    Args:
        n_points: Number of random samples.
        
    Returns:
        Estimate of π.
        
    Note:
        Standard error decreases as 1/√n — doubling precision requires
        quadrupling samples.
        
    Example:
        >>> pi_est = estimate_pi_sequential(1_000_000)
        >>> abs(pi_est - 3.14159) < 0.01
        True
    """
    inside = 0
    
    for _ in range(n_points):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    
    return 4.0 * inside / n_points


def _worker_pi_estimation(args: tuple[int, int]) -> int:
    """
    Worker function for parallel π estimation.
    
    Args:
        args: Tuple of (n_points, seed) for this worker.
        
    Returns:
        Count of points inside the quarter-circle.
    """
    n_points, seed = args
    rng = random.Random(seed)
    
    inside = 0
    for _ in range(n_points):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    
    return inside


def estimate_pi_parallel(
    n_points: int,
    n_workers: int | None = None
) -> tuple[float, float]:
    """
    Estimate π using parallel Monte Carlo sampling.
    
    Demonstrates embarrassingly parallel computation: each worker
    generates independent samples, results combine trivially.
    
    Args:
        n_points: Total number of random samples.
        n_workers: Number of parallel workers.
        
    Returns:
        Tuple of (π estimate, execution time in seconds).
        
    Example:
        >>> pi_est, elapsed = estimate_pi_parallel(10_000_000, n_workers=4)
        >>> abs(pi_est - 3.14159) < 0.01
        True
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    # Distribute points across workers
    points_per_worker = n_points // n_workers
    remainder = n_points % n_workers
    
    # Create args with unique seeds for each worker
    worker_args: list[tuple[int, int]] = []
    for i in range(n_workers):
        points = points_per_worker + (1 if i < remainder else 0)
        seed = i * 12345 + 67890  # Deterministic but distinct seeds
        worker_args.append((points, seed))
    
    start_time = time.perf_counter()
    
    with Pool(processes=n_workers) as pool:
        inside_counts = pool.map(_worker_pi_estimation, worker_args)
    
    elapsed = time.perf_counter() - start_time
    
    total_inside = sum(inside_counts)
    pi_estimate = 4.0 * total_inside / n_points
    
    return pi_estimate, elapsed


def benchmark_scaling(
    n_points: int,
    worker_range: Sequence[int]
) -> dict[int, dict[str, float]]:
    """
    Benchmark parallel scaling across different worker counts.
    
    Demonstrates Amdahl's Law in practice: speedup is bounded by
    the sequential fraction of the computation.
    
    Args:
        n_points: Number of Monte Carlo samples.
        worker_range: Sequence of worker counts to test.
        
    Returns:
        Dictionary mapping worker count to timing results.
        
    Example:
        >>> results = benchmark_scaling(10_000_000, [1, 2, 4, 8])
        >>> results[4]['speedup'] > 3.0  # Near-linear for embarrassingly parallel
        True
    """
    results: dict[int, dict[str, float]] = {}
    
    # Sequential baseline
    logger.info("Measuring sequential baseline...")
    start = time.perf_counter()
    _ = estimate_pi_sequential(n_points)
    baseline_time = time.perf_counter() - start
    
    for n_workers in worker_range:
        logger.info(f"Testing with {n_workers} workers...")
        _, elapsed = estimate_pi_parallel(n_points, n_workers)
        
        speedup = baseline_time / elapsed
        efficiency = speedup / n_workers
        
        results[n_workers] = {
            'time': elapsed,
            'speedup': speedup,
            'efficiency': efficiency
        }
        
        logger.info(f"  Time: {elapsed:.3f}s, "
                   f"Speedup: {speedup:.2f}x, "
                   f"Efficiency: {efficiency:.1%}")
    
    return results


def _worker_bootstrap(
    args: tuple[NDArray[np.float64], int, Callable[[NDArray[np.float64]], float]]
) -> float:
    """Worker function for parallel bootstrap."""
    data, seed, statistic = args
    rng = np.random.default_rng(seed)
    sample = rng.choice(data, size=len(data), replace=True)
    return statistic(sample)


def parallel_bootstrap(
    data: NDArray[np.float64],
    statistic: Callable[[NDArray[np.float64]], float],
    n_iterations: int = 1000,
    n_workers: int | None = None,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval in parallel.
    
    Bootstrap resampling is embarrassingly parallel: each iteration
    draws an independent sample and computes the statistic.
    
    Args:
        data: Original data array.
        statistic: Function computing statistic of interest.
        n_iterations: Number of bootstrap iterations.
        n_workers: Number of parallel workers.
        confidence: Confidence level (e.g., 0.95 for 95%).
        
    Returns:
        Tuple of (point estimate, lower bound, upper bound).
        
    Example:
        >>> data = np.random.randn(1000)
        >>> mean, lower, upper = parallel_bootstrap(data, np.mean, n_iterations=5000)
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    # Create worker arguments
    worker_args = [
        (data, seed, statistic)
        for seed in range(n_iterations)
    ]
    
    with Pool(processes=n_workers) as pool:
        bootstrap_statistics = pool.map(_worker_bootstrap, worker_args)
    
    bootstrap_array = np.array(bootstrap_statistics)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    point_estimate = statistic(data)
    lower_bound = float(np.percentile(bootstrap_array, lower_percentile))
    upper_bound = float(np.percentile(bootstrap_array, upper_percentile))
    
    return point_estimate, lower_bound, upper_bound


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SHARED STATE AND SYNCHRONISATION
# ═══════════════════════════════════════════════════════════════════════════════

def shared_counter_race() -> int:
    """
    Demonstrate race condition with unsynchronised shared counter.
    
    Multiple processes incrementing a shared counter without
    synchronisation produces incorrect results.
    
    Returns:
        Final counter value (typically less than expected due to races).
        
    Warning:
        This function intentionally demonstrates a bug. Do not use
        this pattern in production code.
    """
    # Shared counter (Value provides shared memory)
    counter = Value('i', 0)
    n_increments = 100_000
    n_workers = 4
    
    def increment_without_lock(c: Any, n: int) -> None:
        """Unsafe increment without synchronisation."""
        for _ in range(n):
            c.value += 1  # Race condition!
    
    processes = [
        Process(target=increment_without_lock, args=(counter, n_increments))
        for _ in range(n_workers)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    expected = n_workers * n_increments
    actual = counter.value
    
    logger.warning(f"Race condition demo: expected {expected}, got {actual}")
    logger.warning(f"Lost {expected - actual} increments ({(expected-actual)/expected:.1%})")
    
    return actual


def shared_counter_lock() -> int:
    """
    Demonstrate correct synchronisation with Lock.
    
    Using a Lock ensures mutual exclusion, preventing race conditions.
    
    Returns:
        Final counter value (exactly as expected).
    """
    counter = Value('i', 0)
    lock = Lock()
    n_increments = 100_000
    n_workers = 4
    
    def increment_with_lock(c: Any, lk: Lock, n: int) -> None:
        """Safe increment with lock synchronisation."""
        for _ in range(n):
            with lk:  # Acquire lock, release automatically
                c.value += 1
    
    processes = [
        Process(target=increment_with_lock, args=(counter, lock, n_increments))
        for _ in range(n_workers)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    expected = n_workers * n_increments
    actual = counter.value
    
    assert actual == expected, f"Lock failed: expected {expected}, got {actual}"
    logger.info(f"With lock: expected {expected}, got {actual} ✓")
    
    return actual


def shared_array_processing(
    shape: tuple[int, ...],
    func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_workers: int | None = None
) -> NDArray[np.float64]:
    """
    Process shared array with multiple workers.
    
    Uses multiprocessing.Array for efficient shared memory access.
    Each worker processes a slice of the array.
    
    Args:
        shape: Shape of the array to create.
        func: Function applied to each slice.
        n_workers: Number of parallel workers.
        
    Returns:
        Processed array.
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    # Calculate total size
    total_size = 1
    for dim in shape:
        total_size *= dim
    
    # Create shared array
    shared_arr = Array('d', total_size)
    
    # Initialize with random data
    np_arr = np.frombuffer(shared_arr.get_obj(), dtype=np.float64)
    np_arr[:] = np.random.randn(total_size)
    
    def worker(shared: Array, start: int, end: int) -> None:
        """Process slice of shared array."""
        arr = np.frombuffer(shared.get_obj(), dtype=np.float64)
        arr[start:end] = func(arr[start:end])
    
    # Divide work among workers
    chunk_size = total_size // n_workers
    processes = []
    
    for i in range(n_workers):
        start = i * chunk_size
        end = start + chunk_size if i < n_workers - 1 else total_size
        p = Process(target=worker, args=(shared_arr, start, end))
        processes.append(p)
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    # Return as numpy array
    result = np.frombuffer(shared_arr.get_obj(), dtype=np.float64).copy()
    return result.reshape(shape)


def producer_consumer_queue(
    n_producers: int,
    n_consumers: int,
    n_items: int
) -> list[int]:
    """
    Implement producer-consumer pattern with Queue.
    
    Producers generate items, consumers process them. Queue provides
    thread/process-safe communication.
    
    Args:
        n_producers: Number of producer processes.
        n_consumers: Number of consumer processes.
        n_items: Total items to produce.
        
    Returns:
        List of processed results.
    """
    task_queue: Queue[int | None] = Queue()
    result_queue: Queue[int] = Queue()
    
    def producer(q: Queue[int | None], items: list[int]) -> None:
        """Produce items to queue."""
        for item in items:
            q.put(item)
    
    def consumer(task_q: Queue[int | None], result_q: Queue[int]) -> None:
        """Consume items from queue, compute result."""
        while True:
            try:
                item = task_q.get(timeout=1)
                if item is None:  # Poison pill
                    break
                result = item * item  # Simple computation
                result_q.put(result)
            except Empty:
                break
    
    # Distribute items among producers
    items_per_producer = n_items // n_producers
    producer_items: list[list[int]] = []
    
    for i in range(n_producers):
        start = i * items_per_producer
        end = start + items_per_producer if i < n_producers - 1 else n_items
        producer_items.append(list(range(start, end)))
    
    # Start producers
    producers = [
        Process(target=producer, args=(task_queue, items))
        for items in producer_items
    ]
    for p in producers:
        p.start()
    
    # Start consumers
    consumers = [
        Process(target=consumer, args=(task_queue, result_queue))
        for _ in range(n_consumers)
    ]
    for c in consumers:
        c.start()
    
    # Wait for producers
    for p in producers:
        p.join()
    
    # Send poison pills to consumers
    for _ in range(n_consumers):
        task_queue.put(None)
    
    # Wait for consumers
    for c in consumers:
        c.join()
    
    # Collect results
    results: list[int] = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: THREADING FOR I/O
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_io_task(url_id: int, delay: float = 0.5) -> dict[str, Any]:
    """
    Simulate an I/O-bound task (e.g., HTTP request).
    
    Args:
        url_id: Identifier for the simulated request.
        delay: Simulated network latency.
        
    Returns:
        Dictionary with request metadata.
    """
    time.sleep(delay)  # Simulate network latency
    return {
        'url_id': url_id,
        'status': 200,
        'size': random.randint(1000, 10000),
        'thread': threading.current_thread().name
    }


def download_urls_sequential(urls: Sequence[int]) -> list[dict[str, Any]]:
    """
    Fetch URLs sequentially (baseline for comparison).
    
    Args:
        urls: Sequence of URL identifiers.
        
    Returns:
        List of response dictionaries.
    """
    results: list[dict[str, Any]] = []
    
    start = time.perf_counter()
    for url in urls:
        results.append(simulate_io_task(url))
    elapsed = time.perf_counter() - start
    
    logger.info(f"Sequential: {len(urls)} requests in {elapsed:.2f}s")
    return results


def download_urls_threaded(
    urls: Sequence[int],
    max_workers: int = 10
) -> list[dict[str, Any]]:
    """
    Fetch URLs using thread pool.
    
    Threading excels for I/O-bound tasks because the GIL releases
    during I/O operations, allowing true concurrency.
    
    Args:
        urls: Sequence of URL identifiers.
        max_workers: Maximum concurrent threads.
        
    Returns:
        List of response dictionaries.
    """
    results: list[dict[str, Any]] = []
    
    start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulate_io_task, url) for url in urls]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    elapsed = time.perf_counter() - start
    
    logger.info(f"Threaded ({max_workers} workers): "
               f"{len(urls)} requests in {elapsed:.2f}s")
    return results


def concurrent_file_processing(
    paths: Sequence[str],
    func: Callable[[str], Any],
    max_workers: int = 4
) -> list[Any]:
    """
    Process files concurrently using thread pool.
    
    File I/O benefits from threading when waiting for disk or network.
    
    Args:
        paths: File paths to process.
        func: Function to apply to each file.
        max_workers: Maximum concurrent threads.
        
    Returns:
        List of processing results.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(func, paths))
    
    return results


def demonstrate_io_speedup() -> dict[str, float]:
    """
    Demonstrate threading speedup for I/O-bound tasks.
    
    Returns:
        Dictionary with timing comparisons.
    """
    n_urls = 20
    urls = list(range(n_urls))
    
    # Sequential
    start = time.perf_counter()
    _ = download_urls_sequential(urls)
    sequential_time = time.perf_counter() - start
    
    # Threaded
    start = time.perf_counter()
    _ = download_urls_threaded(urls, max_workers=10)
    threaded_time = time.perf_counter() - start
    
    speedup = sequential_time / threaded_time
    
    results = {
        'sequential': sequential_time,
        'threaded': threaded_time,
        'speedup': speedup
    }
    
    logger.info("=" * 50)
    logger.info("I/O-bound Threading Results:")
    logger.info(f"  Sequential: {sequential_time:.2f}s")
    logger.info(f"  Threaded:   {threaded_time:.2f}s")
    logger.info(f"  Speedup:    {speedup:.1f}x")
    logger.info("=" * 50)
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    """Run demonstration of all lab components."""
    logger.info("=" * 70)
    logger.info("14UNIT Lab 01: Multiprocessing and Threading - Demonstration")
    logger.info("=" * 70)
    
    # Section 1: Process Fundamentals
    logger.info("\n--- Section 1: Process Fundamentals ---")
    logger.info(f"Available CPUs: {cpu_count()}")
    
    logger.info("\nDemonstrating GIL impact...")
    gil_results = demonstrate_gil()
    
    # Section 2: Pool Patterns
    logger.info("\n--- Section 2: Process Pool Patterns ---")
    data = list(range(1, 21))
    squared = parallel_map(lambda x: x * x, data, n_workers=4)
    logger.info(f"Squared {len(data)} numbers: {squared[:5]}...")
    
    # Section 3: Monte Carlo
    logger.info("\n--- Section 3: Monte Carlo Parallelisation ---")
    n_points = 1_000_000
    pi_est, elapsed = estimate_pi_parallel(n_points, n_workers=4)
    error = abs(pi_est - math.pi)
    logger.info(f"π estimate: {pi_est:.6f} (error: {error:.6f})")
    logger.info(f"Computed in {elapsed:.3f}s")
    
    # Scaling benchmark
    logger.info("\nBenchmarking parallel scaling...")
    scaling_results = benchmark_scaling(5_000_000, [1, 2, 4])
    
    # Section 4: Synchronisation
    logger.info("\n--- Section 4: Shared State ---")
    logger.info("Demonstrating race condition...")
    race_result = shared_counter_race()
    logger.info("Demonstrating lock synchronisation...")
    lock_result = shared_counter_lock()
    
    # Section 5: I/O Threading
    logger.info("\n--- Section 5: Threading for I/O ---")
    io_results = demonstrate_io_speedup()
    
    logger.info("\n" + "=" * 70)
    logger.info("Demonstration Complete")
    logger.info("=" * 70)


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='14UNIT Lab 01: Multiprocessing and Threading'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration mode'
    )
    parser.add_argument(
        '--section',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run specific section only'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce logging output'
    )
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    if args.demo:
        run_demo()
    elif args.section:
        logger.info(f"Running Section {args.section}...")
        if args.section == 1:
            demonstrate_gil()
        elif args.section == 2:
            parallel_map(lambda x: x**2, range(100))
        elif args.section == 3:
            estimate_pi_parallel(1_000_000)
        elif args.section == 4:
            shared_counter_lock()
        elif args.section == 5:
            demonstrate_io_speedup()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
