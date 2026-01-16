#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTIONS: easy_02_list_operations.py
Week 3, Practice Exercise Solutions
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Measure Append Operations
# ═══════════════════════════════════════════════════════════════════════════════

def measure_append_operations(sizes: list[int]) -> dict[str, list[tuple[int, float]]]:
    """
    SOLUTION: Measure list append vs list insert(0, x).
    """
    results: dict[str, list[tuple[int, float]]] = {
        "append": [],
        "insert_front": [],
    }
    
    for n in sizes:
        # Measure append (O(1) amortised)
        arr: list[int] = []
        start = time.perf_counter()
        for i in range(n):
            arr.append(i)
        append_time = time.perf_counter() - start
        results["append"].append((n, append_time))
        
        # Measure insert at front (O(n))
        arr = []
        start = time.perf_counter()
        for i in range(n):
            arr.insert(0, i)
        insert_time = time.perf_counter() - start
        results["insert_front"].append((n, insert_time))
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Measure Access Patterns
# ═══════════════════════════════════════════════════════════════════════════════

def measure_access_patterns(sizes: list[int]) -> dict[str, list[tuple[int, float]]]:
    """
    SOLUTION: Compare index access vs iteration.
    """
    results: dict[str, list[tuple[int, float]]] = {
        "index_access": [],
        "iteration": [],
    }
    
    for n in sizes:
        arr = list(range(n))
        
        # Index access
        start = time.perf_counter()
        total = 0
        for i in range(len(arr)):
            total += arr[i]
        index_time = time.perf_counter() - start
        results["index_access"].append((n, index_time))
        
        # Iteration
        start = time.perf_counter()
        total = 0
        for x in arr:
            total += x
        iter_time = time.perf_counter() - start
        results["iteration"].append((n, iter_time))
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: List vs Deque Operations
# ═══════════════════════════════════════════════════════════════════════════════

def compare_list_vs_deque(n: int) -> dict[str, dict[str, float]]:
    """
    SOLUTION: Compare list and deque for front operations.
    """
    from collections import deque
    
    results: dict[str, dict[str, float]] = {
        "list": {},
        "deque": {},
    }
    
    # List operations
    arr: list[int] = []
    start = time.perf_counter()
    for i in range(n):
        arr.insert(0, i)
    results["list"]["insert_front"] = time.perf_counter() - start
    
    arr = list(range(n))
    start = time.perf_counter()
    for _ in range(n):
        arr.pop(0)
    results["list"]["pop_front"] = time.perf_counter() - start
    
    # Deque operations
    dq: deque[int] = deque()
    start = time.perf_counter()
    for i in range(n):
        dq.appendleft(i)
    results["deque"]["insert_front"] = time.perf_counter() - start
    
    dq = deque(range(n))
    start = time.perf_counter()
    for _ in range(n):
        dq.popleft()
    results["deque"]["pop_front"] = time.perf_counter() - start
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Membership Testing
# ═══════════════════════════════════════════════════════════════════════════════

def compare_membership_testing(n: int, tests: int = 1000) -> dict[str, float]:
    """
    SOLUTION: Compare 'in' operator for list vs set.
    """
    import random
    
    arr = list(range(n))
    s = set(arr)
    test_values = [random.randint(0, n * 2) for _ in range(tests)]
    
    # List membership (O(n) per test)
    start = time.perf_counter()
    for v in test_values:
        _ = v in arr
    list_time = time.perf_counter() - start
    
    # Set membership (O(1) per test)
    start = time.perf_counter()
    for v in test_values:
        _ = v in s
    set_time = time.perf_counter() - start
    
    return {
        "list_time": list_time,
        "set_time": set_time,
        "speedup": list_time / set_time if set_time > 0 else float("inf"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Slice vs Loop Copy
# ═══════════════════════════════════════════════════════════════════════════════

def compare_copy_methods(sizes: list[int]) -> dict[str, list[tuple[int, float]]]:
    """
    SOLUTION: Compare different list copy methods.
    """
    results: dict[str, list[tuple[int, float]]] = {
        "slice": [],
        "list_constructor": [],
        "copy_method": [],
        "loop": [],
    }
    
    for n in sizes:
        arr = list(range(n))
        
        # Slice copy
        start = time.perf_counter()
        _ = arr[:]
        results["slice"].append((n, time.perf_counter() - start))
        
        # list() constructor
        start = time.perf_counter()
        _ = list(arr)
        results["list_constructor"].append((n, time.perf_counter() - start))
        
        # .copy() method
        start = time.perf_counter()
        _ = arr.copy()
        results["copy_method"].append((n, time.perf_counter() - start))
        
        # Loop copy
        start = time.perf_counter()
        new_arr = []
        for x in arr:
            new_arr.append(x)
        results["loop"].append((n, time.perf_counter() - start))
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo_solutions() -> None:
    """Demonstrate all solutions."""
    logger.info("=" * 60)
    logger.info("LIST OPERATIONS SOLUTIONS DEMONSTRATION")
    logger.info("=" * 60)
    
    # Demo 1: Append vs Insert
    logger.info("\n1. Append vs Insert(0):")
    results = measure_append_operations([1000, 5000, 10000])
    for n, t in results["append"]:
        insert_t = next(t2 for n2, t2 in results["insert_front"] if n2 == n)
        logger.info(f"   n={n}: append={t:.4f}s, insert(0)={insert_t:.4f}s")
    
    # Demo 2: List vs Deque
    logger.info("\n2. List vs Deque (front operations, n=10000):")
    results = compare_list_vs_deque(10000)
    logger.info(f"   List insert_front: {results['list']['insert_front']:.4f}s")
    logger.info(f"   Deque appendleft:  {results['deque']['insert_front']:.6f}s")
    
    # Demo 3: Membership testing
    logger.info("\n3. Membership Testing (n=100000, 1000 tests):")
    results = compare_membership_testing(100000, 1000)
    logger.info(f"   List: {results['list_time']:.4f}s")
    logger.info(f"   Set:  {results['set_time']:.6f}s")
    logger.info(f"   Speedup: {results['speedup']:.0f}x")
    
    # Demo 4: Copy methods
    logger.info("\n4. Copy Methods (n=100000):")
    results = compare_copy_methods([100000])
    for method, data in results.items():
        logger.info(f"   {method}: {data[0][1]:.6f}s")


if __name__ == "__main__":
    demo_solutions()
