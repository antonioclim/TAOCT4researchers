#!/usr/bin/env python3
"""
14UNIT Benchmark Script
Run Monte Carlo benchmarks and generate performance data.

Usage:
    python scripts/run_benchmarks.py [--output results.csv]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Sequence

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def monte_carlo_worker(args: tuple[int, int]) -> int:
    """Worker for parallel Monte Carlo."""
    import random
    n_points, seed = args
    rng = random.Random(seed)
    inside = sum(
        1 for _ in range(n_points)
        if rng.random()**2 + rng.random()**2 <= 1
    )
    return inside


def estimate_pi_sequential(n_points: int) -> tuple[float, float]:
    """Sequential π estimation."""
    import random
    start = time.perf_counter()
    inside = sum(
        1 for _ in range(n_points)
        if random.random()**2 + random.random()**2 <= 1
    )
    elapsed = time.perf_counter() - start
    return 4.0 * inside / n_points, elapsed


def estimate_pi_parallel(n_points: int, n_workers: int) -> tuple[float, float]:
    """Parallel π estimation."""
    points_per_worker = n_points // n_workers
    worker_args = [(points_per_worker, i * 12345) for i in range(n_workers)]
    
    start = time.perf_counter()
    with Pool(n_workers) as pool:
        results = pool.map(monte_carlo_worker, worker_args)
    elapsed = time.perf_counter() - start
    
    total_inside = sum(results)
    return 4.0 * total_inside / n_points, elapsed


def run_benchmark(
    point_counts: Sequence[int],
    worker_counts: Sequence[int],
    output_path: Path | None = None
) -> list[dict]:
    """Run full benchmark suite."""
    results = []
    
    for n_points in point_counts:
        print(f"\nBenchmarking {n_points:,} points...")
        
        # Sequential baseline
        _, seq_time = estimate_pi_sequential(n_points)
        
        for n_workers in worker_counts:
            _, par_time = estimate_pi_parallel(n_points, n_workers)
            speedup = seq_time / par_time
            efficiency = speedup / n_workers
            
            result = {
                'n_points': n_points,
                'n_workers': n_workers,
                'sequential_time': round(seq_time, 4),
                'parallel_time': round(par_time, 4),
                'speedup': round(speedup, 2),
                'efficiency': round(efficiency, 2),
            }
            results.append(result)
            print(f"  {n_workers} workers: {speedup:.2f}x speedup, {efficiency:.0%} efficiency")
    
    if output_path:
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run parallel computing benchmarks')
    parser.add_argument('--output', '-o', type=Path, default=None,
                        help='Output CSV file path')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark with fewer points')
    args = parser.parse_args()
    
    if args.quick:
        point_counts = [100_000, 500_000]
        worker_counts = [1, 2, 4]
    else:
        point_counts = [100_000, 500_000, 1_000_000, 5_000_000]
        worker_counts = [1, 2, 4, min(8, cpu_count())]
    
    print("=" * 60)
    print("14UNIT Parallel Computing Benchmark")
    print(f"CPU cores available: {cpu_count()}")
    print("=" * 60)
    
    run_benchmark(point_counts, worker_counts, args.output)
    
    print("\nBenchmark complete.")


if __name__ == '__main__':
    main()
