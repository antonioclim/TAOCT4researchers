#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Threading for I/O (Easy)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Understand when threading provides genuine speedup despite the GIL, focusing
on I/O-bound workloads where the GIL releases during wait operations.

LEARNING OUTCOMES
─────────────────
- LO3: Implement threading for I/O-bound workloads

ESTIMATED TIME: 15 minutes

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence


def simulate_network_request(url_id: int, latency: float = 0.2) -> dict[str, Any]:
    """
    Simulate a network request with given latency.
    
    Args:
        url_id: Identifier for the simulated request.
        latency: Simulated network delay in seconds.
        
    Returns:
        Dictionary with request metadata.
    """
    time.sleep(latency)  # Simulate network wait
    return {
        'url_id': url_id,
        'status': 200,
        'thread': threading.current_thread().name
    }


def fetch_sequential(url_ids: Sequence[int], latency: float = 0.2) -> list[dict[str, Any]]:
    """
    Fetch URLs sequentially (baseline for comparison).
    
    Args:
        url_ids: Sequence of URL identifiers.
        latency: Simulated latency per request.
        
    Returns:
        List of response dictionaries.
    """
    results = []
    for url_id in url_ids:
        results.append(simulate_network_request(url_id, latency))
    return results


def fetch_threaded_basic(
    url_ids: Sequence[int],
    latency: float = 0.2
) -> list[dict[str, Any]]:
    """
    Fetch URLs using ThreadPoolExecutor.
    
    TODO: Implement using ThreadPoolExecutor with executor.map().
    
    Args:
        url_ids: URL identifiers to fetch.
        latency: Simulated latency.
        
    Returns:
        List of responses (in input order).
        
    Example:
        >>> results = fetch_threaded_basic([1, 2, 3], latency=0.1)
        >>> len(results)
        3
    """
    # TODO: Create ThreadPoolExecutor with max_workers=10
    # TODO: Use executor.map() to apply simulate_network_request
    # TODO: Return list of results
    pass


def fetch_threaded_as_completed(
    url_ids: Sequence[int],
    latency: float = 0.2
) -> list[dict[str, Any]]:
    """
    Fetch URLs and process results as they complete.
    
    as_completed yields futures in completion order, enabling
    early processing of fast responses.
    
    TODO: Implement using executor.submit() and as_completed().
    
    Args:
        url_ids: URL identifiers.
        latency: Simulated latency.
        
    Returns:
        List of responses (completion order, may differ from input).
    """
    # TODO: Create ThreadPoolExecutor
    # TODO: Submit tasks with executor.submit()
    # TODO: Iterate with as_completed() and collect results
    pass


def measure_speedup(
    url_ids: Sequence[int],
    latency: float = 0.2
) -> dict[str, float]:
    """
    Measure and compare sequential vs threaded performance.
    
    TODO: Implement timing comparison.
    
    Args:
        url_ids: URL identifiers.
        latency: Simulated latency.
        
    Returns:
        Dictionary with 'sequential', 'threaded', and 'speedup' times.
    """
    # TODO: Time sequential execution
    # TODO: Time threaded execution
    # TODO: Calculate speedup
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_fetch_threaded_basic() -> None:
    """Test basic threaded fetch returns correct results."""
    results = fetch_threaded_basic([1, 2, 3, 4, 5], latency=0.05)
    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    url_ids = {r['url_id'] for r in results}
    assert url_ids == {1, 2, 3, 4, 5}, f"Missing URLs: {url_ids}"
    print("✓ fetch_threaded_basic works")


def test_fetch_threaded_as_completed() -> None:
    """Test as_completed fetch returns all results."""
    results = fetch_threaded_as_completed([10, 20, 30], latency=0.05)
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print("✓ fetch_threaded_as_completed works")


def test_measure_speedup() -> None:
    """Test that threading provides speedup for I/O."""
    metrics = measure_speedup(list(range(10)), latency=0.1)
    assert metrics['speedup'] > 2.0, f"Expected speedup > 2, got {metrics['speedup']}"
    print(f"✓ measure_speedup: {metrics['speedup']:.1f}x faster")


def main() -> None:
    """Run all tests."""
    print("Testing easy_03_threading_io.py")
    print("=" * 50)
    
    test_fetch_threaded_basic()
    test_fetch_threaded_as_completed()
    test_measure_speedup()
    
    print("=" * 50)
    print("All tests passed!")


if __name__ == '__main__':
    main()
