#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Threading for I/O - SOLUTION
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Sequence


def simulate_network_request(url_id: int, latency: float = 0.2) -> dict[str, Any]:
    """Simulate a network request with given latency."""
    time.sleep(latency)
    return {
        'url_id': url_id,
        'status': 200,
        'thread': threading.current_thread().name
    }


def fetch_sequential(url_ids: Sequence[int], latency: float = 0.2) -> list[dict[str, Any]]:
    """Fetch URLs sequentially."""
    return [simulate_network_request(uid, latency) for uid in url_ids]


def fetch_threaded_basic(
    url_ids: Sequence[int],
    latency: float = 0.2
) -> list[dict[str, Any]]:
    """Fetch URLs using ThreadPoolExecutor with map."""
    # Create partial function with fixed latency
    fetch_func = partial(simulate_network_request, latency=latency)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # map returns results in input order
        results = list(executor.map(fetch_func, url_ids))
    
    return results


def fetch_threaded_as_completed(
    url_ids: Sequence[int],
    latency: float = 0.2
) -> list[dict[str, Any]]:
    """Fetch URLs and process results as they complete."""
    results: list[dict[str, Any]] = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        futures = {
            executor.submit(simulate_network_request, uid, latency): uid
            for uid in url_ids
        }
        
        # Process in completion order
        for future in as_completed(futures):
            results.append(future.result())
    
    return results


def measure_speedup(
    url_ids: Sequence[int],
    latency: float = 0.2
) -> dict[str, float]:
    """Measure and compare sequential vs threaded performance."""
    # Time sequential
    start = time.perf_counter()
    _ = fetch_sequential(url_ids, latency)
    sequential_time = time.perf_counter() - start
    
    # Time threaded
    start = time.perf_counter()
    _ = fetch_threaded_basic(url_ids, latency)
    threaded_time = time.perf_counter() - start
    
    return {
        'sequential': sequential_time,
        'threaded': threaded_time,
        'speedup': sequential_time / threaded_time
    }


def test_fetch_threaded_basic() -> None:
    results = fetch_threaded_basic([1, 2, 3, 4, 5], latency=0.05)
    assert len(results) == 5
    assert {r['url_id'] for r in results} == {1, 2, 3, 4, 5}
    print("✓ fetch_threaded_basic works")


def test_fetch_threaded_as_completed() -> None:
    results = fetch_threaded_as_completed([10, 20, 30], latency=0.05)
    assert len(results) == 3
    print("✓ fetch_threaded_as_completed works")


def test_measure_speedup() -> None:
    metrics = measure_speedup(list(range(10)), latency=0.1)
    assert metrics['speedup'] > 2.0
    print(f"✓ measure_speedup: {metrics['speedup']:.1f}x faster")


if __name__ == '__main__':
    print("Testing easy_03_solution.py")
    print("=" * 50)
    test_fetch_threaded_basic()
    test_fetch_threaded_as_completed()
    test_measure_speedup()
    print("All tests passed!")
