#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Parallel Monte Carlo (Medium)
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ★★★☆☆ (Medium)
ESTIMATED TIME: 25 minutes
PREREQUISITES: Process pools, random number generation

LEARNING OBJECTIVES
───────────────────
- LO2: Implement parallel Monte Carlo estimation
- LO6: Measure and analyse parallel speedup

PROBLEM DESCRIPTION
───────────────────
Implement parallel Monte Carlo estimation of π and measure the speedup
achieved compared to sequential execution.

TASKS
─────
1. Implement `estimate_pi_worker` - worker function for π estimation
2. Implement `parallel_monte_carlo_pi` - parallel π estimation with Pool
3. Implement `measure_speedup` - compare sequential vs parallel performance

MATHEMATICAL BACKGROUND
───────────────────────
The dartboard method: generate random points (x, y) in [0, 1]².
Points satisfying x² + y² ≤ 1 fall inside the quarter-circle.
Ratio of inside to total points approximates π/4.

HINTS
─────
- Hint 1: Each worker needs a unique random seed to avoid correlation
- Hint 2: Use Pool.map to distribute work across workers
- Hint 3: Speedup = T_sequential / T_parallel

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
import time
from multiprocessing import Pool, cpu_count
from typing import Sequence


def estimate_pi_worker(args: tuple[int, int]) -> int:
    """
    Worker function: count points inside quarter-circle.
    
    Args:
        args: Tuple of (n_points, seed) for this worker.
        
    Returns:
        Count of points where x² + y² ≤ 1.
        
    Example:
        >>> inside = estimate_pi_worker((10000, 42))
        >>> 7000 < inside < 8500  # Approximately π/4 * 10000
        True
    """
    # TODO: Implement this function
    # 1. Unpack n_points and seed from args
    # 2. Create a Random instance with the seed
    # 3. Generate n_points random (x, y) pairs
    # 4. Count how many satisfy x² + y² ≤ 1
    raise NotImplementedError("Implement estimate_pi_worker")


def parallel_monte_carlo_pi(
    n_points: int,
    n_workers: int | None = None
) -> tuple[float, float]:
    """
    Estimate π using parallel Monte Carlo sampling.
    
    Distribute work across multiple processes, each generating
    independent samples with unique random seeds.
    
    Args:
        n_points: Total number of random samples.
        n_workers: Number of parallel workers (default: CPU count).
        
    Returns:
        Tuple of (π estimate, execution time in seconds).
        
    Example:
        >>> pi_est, elapsed = parallel_monte_carlo_pi(1_000_000, 4)
        >>> abs(pi_est - 3.14159) < 0.01
        True
    """
    # TODO: Implement this function
    # 1. Determine number of workers
    # 2. Divide points among workers (handle remainder)
    # 3. Create list of (points, seed) tuples for each worker
    # 4. Use Pool.map to execute workers
    # 5. Sum results and compute π estimate
    raise NotImplementedError("Implement parallel_monte_carlo_pi")


def sequential_monte_carlo_pi(n_points: int) -> tuple[float, float]:
    """
    Sequential π estimation (baseline for comparison).
    
    Args:
        n_points: Number of random samples.
        
    Returns:
        Tuple of (π estimate, execution time).
    """
    start = time.perf_counter()
    
    inside = 0
    for _ in range(n_points):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1.0:
            inside += 1
    
    elapsed = time.perf_counter() - start
    return 4.0 * inside / n_points, elapsed


def measure_speedup(
    n_points: int,
    worker_counts: Sequence[int]
) -> dict[int, dict[str, float]]:
    """
    Measure parallel speedup across different worker counts.
    
    Args:
        n_points: Number of Monte Carlo samples.
        worker_counts: Sequence of worker counts to test.
        
    Returns:
        Dictionary mapping worker count to:
        - 'time': execution time
        - 'speedup': T_sequential / T_parallel
        - 'efficiency': speedup / n_workers
        
    Example:
        >>> results = measure_speedup(1_000_000, [1, 2, 4])
        >>> results[4]['speedup'] > 2.0  # Expect significant speedup
        True
    """
    # TODO: Implement this function
    # 1. Run sequential version to get baseline time
    # 2. For each worker count, run parallel version
    # 3. Calculate speedup and efficiency
    raise NotImplementedError("Implement measure_speedup")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_estimate_pi_worker() -> None:
    """Test the worker function."""
    inside = estimate_pi_worker((100_000, 42))
    # Should be approximately π/4 * 100000 ≈ 78540
    assert 75_000 < inside < 82_000, f"Unexpected count: {inside}"
    print("✓ estimate_pi_worker tests passed")


def test_parallel_monte_carlo_pi() -> None:
    """Test parallel π estimation."""
    pi_est, elapsed = parallel_monte_carlo_pi(500_000, n_workers=2)
    assert abs(pi_est - 3.14159) < 0.02, f"Poor estimate: {pi_est}"
    assert elapsed > 0
    print(f"✓ parallel_monte_carlo_pi: π ≈ {pi_est:.5f} in {elapsed:.3f}s")


def test_measure_speedup() -> None:
    """Test speedup measurement."""
    results = measure_speedup(200_000, [1, 2])
    assert 1 in results
    assert 2 in results
    assert results[2]['speedup'] > 1.0  # Should see some speedup
    print(f"✓ measure_speedup: 2-worker speedup = {results[2]['speedup']:.2f}x")


def main() -> None:
    """Run all tests."""
    print("Running medium_01_monte_carlo_parallel tests...")
    print("-" * 50)
    
    try:
        test_estimate_pi_worker()
        test_parallel_monte_carlo_pi()
        test_measure_speedup()
        print("-" * 50)
        print("All tests passed! ✓")
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}")
    except AssertionError as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    main()
