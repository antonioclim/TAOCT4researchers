#!/usr/bin/env python3
"""
14UNIT Exercise Solution: Parallel Monte Carlo (Medium)
© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
import random
import time
from multiprocessing import Pool, cpu_count
from typing import Sequence


def estimate_pi_worker(args: tuple[int, int]) -> int:
    """Worker function: count points inside quarter-circle."""
    n_points, seed = args
    rng = random.Random(seed)
    
    inside = 0
    for _ in range(n_points):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1.0:
            inside += 1
    return inside


def parallel_monte_carlo_pi(
    n_points: int,
    n_workers: int | None = None
) -> tuple[float, float]:
    """Estimate π using parallel Monte Carlo sampling."""
    if n_workers is None:
        n_workers = cpu_count()
    
    points_per_worker = n_points // n_workers
    remainder = n_points % n_workers
    
    worker_args = []
    for i in range(n_workers):
        points = points_per_worker + (1 if i < remainder else 0)
        seed = i * 12345 + 67890
        worker_args.append((points, seed))
    
    start = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        inside_counts = pool.map(estimate_pi_worker, worker_args)
    elapsed = time.perf_counter() - start
    
    total_inside = sum(inside_counts)
    pi_estimate = 4.0 * total_inside / n_points
    
    return pi_estimate, elapsed


def sequential_monte_carlo_pi(n_points: int) -> tuple[float, float]:
    """Sequential π estimation (baseline)."""
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
    """Measure parallel speedup across different worker counts."""
    _, baseline_time = sequential_monte_carlo_pi(n_points)
    
    results = {}
    for n_workers in worker_counts:
        _, elapsed = parallel_monte_carlo_pi(n_points, n_workers)
        speedup = baseline_time / elapsed
        efficiency = speedup / n_workers
        results[n_workers] = {
            'time': elapsed,
            'speedup': speedup,
            'efficiency': efficiency
        }
    return results


if __name__ == '__main__':
    pi, t = parallel_monte_carlo_pi(1_000_000, 4)
    print(f"π ≈ {pi:.6f} in {t:.3f}s")
