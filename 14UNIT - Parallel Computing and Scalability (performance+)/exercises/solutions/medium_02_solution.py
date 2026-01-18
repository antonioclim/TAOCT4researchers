#!/usr/bin/env python3
"""
14UNIT Exercise Solution: Synchronisation Patterns (Medium)
© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
from multiprocessing import Lock, Process, Value
from typing import Any


def _unsafe_worker(counter: Any, n: int) -> None:
    """Worker WITHOUT lock - demonstrates race condition."""
    for _ in range(n):
        counter.value += 1


def unsafe_counter(n_processes: int, increments_per_process: int) -> int:
    """Demonstrate race condition with unsynchronised counter."""
    counter = Value('i', 0)
    
    processes = [
        Process(target=_unsafe_worker, args=(counter, increments_per_process))
        for _ in range(n_processes)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    return counter.value


def _safe_worker_lock(counter: Any, lock: Lock, n: int) -> None:
    """Worker WITH external lock."""
    for _ in range(n):
        with lock:
            counter.value += 1


def safe_counter_lock(n_processes: int, increments_per_process: int) -> int:
    """Thread-safe counter using explicit Lock."""
    counter = Value('i', 0)
    lock = Lock()
    
    processes = [
        Process(target=_safe_worker_lock, args=(counter, lock, increments_per_process))
        for _ in range(n_processes)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    return counter.value


def _safe_worker_value(counter: Any, n: int) -> None:
    """Worker using Value's built-in lock."""
    for _ in range(n):
        with counter.get_lock():
            counter.value += 1


def safe_counter_value(n_processes: int, increments_per_process: int) -> int:
    """Thread-safe counter using Value's built-in lock."""
    counter = Value('i', 0)
    
    processes = [
        Process(target=_safe_worker_value, args=(counter, increments_per_process))
        for _ in range(n_processes)
    ]
    
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    return counter.value


if __name__ == '__main__':
    print(f"Unsafe: {unsafe_counter(4, 10000)}")
    print(f"Safe (lock): {safe_counter_lock(4, 10000)}")
    print(f"Safe (value): {safe_counter_value(4, 10000)}")
