#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
14UNIT, Lab 02: Dask and Profiling
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
When datasets exceed available memory or computations span many cores, higher-
level abstractions become essential. Dask provides a parallel computing
framework that scales from laptop to cluster, enabling out-of-core computation
and lazy evaluation. This lab develops practical skills with Dask whilst
introducing profiling tools for performance analysis.

HISTORICAL MOTIVATION
─────────────────────
The explosion of data in computational research—genomic sequences, climate
simulations, social media streams—has outpaced single-machine memory growth.
Dask emerged from the PyData environment to address this gap, providing familiar
NumPy and pandas interfaces for larger-than-memory datasets.

Effective optimisation requires measurement, not guesswork. Donald Knuth's
observation that "premature optimisation is the root of all evil" applies
particularly to parallel computing, where intuition frequently misleads.
Profiling tools transform optimisation from art to science.

PREREQUISITES
─────────────
- 03UNIT: Complexity analysis
- Lab 01: Multiprocessing fundamentals
- NumPy/Pandas: Basic operations

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Construct Dask delayed computations and visualise task graphs
2. Process larger-than-memory datasets with Dask arrays and DataFrames
3. Profile code using cProfile and memory_profiler
4. Identify and address performance bottlenecks

ESTIMATED TIME
──────────────
- Reading: 15 minutes
- Coding: 30 minutes
- Total: 45 minutes

DEPENDENCIES
────────────
- dask>=2024.1
- pandas>=2.0
- numpy>=1.24

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Sequence,
    TypeVar,
)

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Dask imports
try:
    import dask
    import dask.array as da
    import dask.dataframe as dd
    from dask import delayed
    from dask.diagnostics import ProgressBar
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    delayed = lambda x: x  # Placeholder

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


def check_dask() -> None:
    """Verify Dask is available."""
    if not HAS_DASK:
        raise ImportError(
            "Dask is required for this lab. Install with: pip install dask[complete]"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DASK DELAYED
# ═══════════════════════════════════════════════════════════════════════════════

def expensive_operation(x: float, delay: float = 0.1) -> float:
    """
    Simulate an expensive computation.
    
    Args:
        x: Input value.
        delay: Simulated computation time.
        
    Returns:
        Result of computation (x squared).
    """
    time.sleep(delay)
    return x * x


def sequential_pipeline(data: Sequence[float]) -> float:
    """
    Execute pipeline sequentially (baseline).
    
    Args:
        data: Input values.
        
    Returns:
        Sum of squared values.
    """
    results = []
    for x in data:
        results.append(expensive_operation(x))
    return sum(results)


@delayed
def delayed_expensive_operation(x: float, delay: float = 0.1) -> float:
    """Delayed version of expensive operation."""
    time.sleep(delay)
    return x * x


@delayed
def delayed_sum(values: list[float]) -> float:
    """Delayed summation."""
    return sum(values)


def delayed_pipeline(data: Sequence[float]) -> Any:
    """
    Construct delayed computation graph.
    
    Dask's delayed decorator marks functions for lazy evaluation.
    Computations build a task graph executed only when .compute() is called.
    
    Args:
        data: Input values.
        
    Returns:
        Delayed computation object (call .compute() to execute).
        
    Example:
        >>> result = delayed_pipeline([1.0, 2.0, 3.0])
        >>> result.compute()
        14.0
    """
    check_dask()
    
    # Build lazy computation graph
    delayed_results = [
        delayed_expensive_operation(x, delay=0.05)
        for x in data
    ]
    
    # Combine results (also delayed)
    total = delayed_sum(delayed_results)
    
    return total


def visualise_task_graph(computation: Any, filename: str | None = None) -> str | None:
    """
    Visualise Dask task graph.
    
    Task graphs reveal parallelisation opportunities and dependencies.
    
    Args:
        computation: Dask delayed computation.
        filename: Optional path to save visualisation.
        
    Returns:
        Path to saved file, or None if visualisation unavailable.
    """
    check_dask()
    
    try:
        if filename:
            computation.visualize(filename=filename, format='png')
            logger.info(f"Task graph saved to {filename}")
            return filename
        else:
            # Try to display inline
            computation.visualize()
            return None
    except Exception as e:
        logger.warning(f"Could not visualise task graph: {e}")
        return None


def compare_schedulers(
    computation: Any,
    schedulers: Sequence[str] = ('synchronous', 'threads', 'processes')
) -> dict[str, float]:
    """
    Compare Dask scheduler performance.
    
    Dask supports multiple schedulers:
    - synchronous: Single-threaded, useful for debugging
    - threads: Multi-threaded, good for NumPy-heavy or I/O-bound work
    - processes: Multi-process, good for pure Python CPU-bound work
    
    Args:
        computation: Dask computation to execute.
        schedulers: Schedulers to compare.
        
    Returns:
        Dictionary mapping scheduler name to execution time.
    """
    check_dask()
    
    results: dict[str, float] = {}
    
    for scheduler in schedulers:
        start = time.perf_counter()
        
        try:
            with dask.config.set(scheduler=scheduler):
                _ = computation.compute()
            elapsed = time.perf_counter() - start
            results[scheduler] = elapsed
            logger.info(f"Scheduler '{scheduler}': {elapsed:.3f}s")
        except Exception as e:
            logger.warning(f"Scheduler '{scheduler}' failed: {e}")
            results[scheduler] = float('inf')
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DASK ARRAYS AND DATAFRAMES
# ═══════════════════════════════════════════════════════════════════════════════

def create_dask_array(
    shape: tuple[int, ...],
    chunks: tuple[int, ...] | str = 'auto'
) -> da.Array:
    """
    Create a Dask array with specified chunking.
    
    Chunking determines the granularity of parallel operations.
    Smaller chunks enable more parallelism but increase overhead.
    
    Args:
        shape: Array shape.
        chunks: Chunk sizes (tuple or 'auto').
        
    Returns:
        Dask array filled with random values.
        
    Example:
        >>> arr = create_dask_array((10000, 10000), chunks=(1000, 1000))
        >>> arr.shape
        (10000, 10000)
    """
    check_dask()
    
    arr = da.random.random(shape, chunks=chunks)
    logger.info(f"Created Dask array: shape={shape}, chunks={arr.chunks}")
    
    return arr


def dask_array_operations(arr: da.Array) -> dict[str, Any]:
    """
    Demonstrate Dask array operations.
    
    Dask arrays support most NumPy operations with lazy evaluation.
    Operations build a task graph; compute() triggers execution.
    
    Args:
        arr: Input Dask array.
        
    Returns:
        Dictionary of computed statistics.
    """
    check_dask()
    
    results = {}
    
    # Element-wise operations (lazy)
    squared = arr ** 2
    
    # Reductions (lazy until computed)
    results['mean'] = float(arr.mean().compute())
    results['std'] = float(arr.std().compute())
    results['sum'] = float(arr.sum().compute())
    results['min'] = float(arr.min().compute())
    results['max'] = float(arr.max().compute())
    
    # Compute all at once (more efficient)
    results['squared_mean'] = float(squared.mean().compute())
    
    logger.info(f"Array statistics: mean={results['mean']:.4f}, std={results['std']:.4f}")
    
    return results


def read_large_csv_dask(
    path_pattern: str,
    blocksize: str = '64MB'
) -> dd.DataFrame:
    """
    Read large CSV file(s) as Dask DataFrame.
    
    Dask DataFrames partition data across chunks, enabling parallel
    processing of larger-than-memory datasets.
    
    Args:
        path_pattern: Path to CSV file(s), may include wildcards.
        blocksize: Size of each partition.
        
    Returns:
        Dask DataFrame.
    """
    check_dask()
    
    ddf = dd.read_csv(path_pattern, blocksize=blocksize)
    
    n_partitions = ddf.npartitions
    logger.info(f"Loaded DataFrame with {n_partitions} partitions")
    
    return ddf


def create_sample_dask_dataframe(n_rows: int = 1_000_000) -> dd.DataFrame:
    """
    Create a sample Dask DataFrame for demonstration.
    
    Args:
        n_rows: Number of rows.
        
    Returns:
        Dask DataFrame with sample data.
    """
    check_dask()
    
    # Create pandas DataFrame first
    n_partitions = 10
    rows_per_partition = n_rows // n_partitions
    
    dfs = []
    for i in range(n_partitions):
        np.random.seed(i)
        df = pd.DataFrame({
            'id': range(i * rows_per_partition, (i + 1) * rows_per_partition),
            'category': np.random.choice(['A', 'B', 'C', 'D'], rows_per_partition),
            'value': np.random.randn(rows_per_partition) * 100,
            'count': np.random.randint(1, 100, rows_per_partition)
        })
        dfs.append(df)
    
    # Convert to Dask DataFrame
    ddf = dd.from_pandas(pd.concat(dfs, ignore_index=True), npartitions=n_partitions)
    
    logger.info(f"Created sample DataFrame: {n_rows} rows, {n_partitions} partitions")
    
    return ddf


def dask_groupby_aggregate(
    ddf: dd.DataFrame,
    by: str | list[str],
    agg: dict[str, str | list[str]]
) -> pd.DataFrame:
    """
    Perform groupby aggregation on Dask DataFrame.
    
    Aggregations execute in parallel across partitions, then combine results.
    
    Args:
        ddf: Input Dask DataFrame.
        by: Column(s) to group by.
        agg: Aggregation specification.
        
    Returns:
        Aggregated pandas DataFrame (computed).
        
    Example:
        >>> dask_groupby_aggregate(ddf, 'category', {'value': ['mean', 'sum']})
    """
    check_dask()
    
    result = ddf.groupby(by).agg(agg).compute()
    
    logger.info(f"Grouped by {by}, computed {len(result)} groups")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OUT-OF-CORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def process_larger_than_memory(
    arr: da.Array,
    chunk_func: Callable[[np.ndarray], np.ndarray]
) -> da.Array:
    """
    Apply function to array larger than memory.
    
    Dask processes chunks individually, never loading the full array.
    
    Args:
        arr: Input Dask array.
        chunk_func: Function to apply to each chunk.
        
    Returns:
        Transformed Dask array.
    """
    check_dask()
    
    # map_blocks applies function to each chunk
    result = arr.map_blocks(chunk_func)
    
    return result


def incremental_statistics(arr: da.Array) -> dict[str, float]:
    """
    Compute statistics incrementally without loading full array.
    
    Demonstrates out-of-core aggregation algorithms.
    
    Args:
        arr: Input Dask array.
        
    Returns:
        Dictionary of statistics.
    """
    check_dask()
    
    # These operations work chunk-by-chunk
    results = {
        'mean': float(arr.mean().compute()),
        'var': float(arr.var().compute()),
        'sum': float(arr.sum().compute()),
        'size': int(np.prod(arr.shape))
    }
    
    return results


def persist_intermediate_results(
    ddf: dd.DataFrame,
    output_path: str | Path,
    format: str = 'parquet'
) -> Path:
    """
    Persist Dask DataFrame to disk.
    
    Useful for checkpointing long computations or sharing results.
    
    Args:
        ddf: Dask DataFrame to persist.
        output_path: Output directory path.
        format: Output format ('parquet' or 'csv').
        
    Returns:
        Path to output directory.
    """
    check_dask()
    
    output_path = Path(output_path)
    
    if format == 'parquet':
        ddf.to_parquet(output_path)
    elif format == 'csv':
        ddf.to_csv(output_path / '*.csv')
    else:
        raise ValueError(f"Unknown format: {format}")
    
    logger.info(f"Persisted DataFrame to {output_path}")
    
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PROFILING AND OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProfileResult:
    """Container for profiling results."""
    
    function_name: str
    total_time: float
    calls: int
    time_per_call: float
    cumulative_time: float
    top_functions: list[tuple[str, float]]
    
    def __str__(self) -> str:
        lines = [
            f"Profile: {self.function_name}",
            f"  Total time: {self.total_time:.4f}s",
            f"  Calls: {self.calls}",
            f"  Time/call: {self.time_per_call:.6f}s",
            f"  Top functions:"
        ]
        for name, time_spent in self.top_functions[:5]:
            lines.append(f"    {name}: {time_spent:.4f}s")
        return '\n'.join(lines)


def profile_function(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> tuple[T, ProfileResult]:
    """
    Profile function execution using cProfile.
    
    Args:
        func: Function to profile.
        *args: Positional arguments for function.
        **kwargs: Keyword arguments for function.
        
    Returns:
        Tuple of (function result, profile results).
        
    Example:
        >>> result, profile = profile_function(expensive_func, arg1, arg2)
        >>> print(profile)
    """
    profiler = cProfile.Profile()
    
    # Profile the function execution
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    
    # Parse statistics
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    
    # Extract top functions
    top_functions: list[tuple[str, float]] = []
    stats_list = stats.get_stats_profile().func_profiles
    
    for func_key, func_stats in sorted(
        stats_list.items(),
        key=lambda x: x[1].cumtime,
        reverse=True
    )[:10]:
        if isinstance(func_key, tuple):
            name = f"{func_key[2]} ({func_key[0]}:{func_key[1]})"
        else:
            name = str(func_key)
        top_functions.append((name, func_stats.cumtime))
    
    # Calculate totals
    total_time = sum(s.tottime for s in stats_list.values())
    total_calls = sum(s.ncalls for s in stats_list.values())
    
    profile_result = ProfileResult(
        function_name=func.__name__,
        total_time=total_time,
        calls=total_calls,
        time_per_call=total_time / max(total_calls, 1),
        cumulative_time=sum(s.cumtime for s in stats_list.values()),
        top_functions=top_functions
    )
    
    return result, profile_result


def memory_profile_function(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any
) -> tuple[T, dict[str, Any]]:
    """
    Profile memory usage of function execution.
    
    Uses tracemalloc for memory tracking.
    
    Args:
        func: Function to profile.
        *args: Positional arguments.
        **kwargs: Keyword arguments.
        
    Returns:
        Tuple of (result, memory statistics).
    """
    import tracemalloc
    
    tracemalloc.start()
    
    result = func(*args, **kwargs)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_stats = {
        'current_mb': current / (1024 * 1024),
        'peak_mb': peak / (1024 * 1024),
    }
    
    logger.info(f"Memory: current={memory_stats['current_mb']:.2f}MB, "
               f"peak={memory_stats['peak_mb']:.2f}MB")
    
    return result, memory_stats


def identify_bottlenecks(profile_stats: ProfileResult) -> list[str]:
    """
    Identify performance bottlenecks from profile results.
    
    Args:
        profile_stats: Profile results from profile_function.
        
    Returns:
        List of recommendations.
    """
    recommendations: list[str] = []
    
    # Check for obvious bottlenecks
    if profile_stats.top_functions:
        top_name, top_time = profile_stats.top_functions[0]
        
        if top_time > profile_stats.total_time * 0.8:
            recommendations.append(
                f"Function '{top_name}' consumes {top_time/profile_stats.total_time:.0%} "
                f"of time. Focus optimisation efforts here."
            )
    
    # Check call count
    if profile_stats.calls > 1_000_000:
        recommendations.append(
            f"High call count ({profile_stats.calls:,}). "
            f"Consider vectorisation or batching."
        )
    
    # Check time per call
    if profile_stats.time_per_call > 0.001:
        recommendations.append(
            f"Slow individual calls ({profile_stats.time_per_call*1000:.2f}ms each). "
            f"Look for algorithmic improvements."
        )
    
    if not recommendations:
        recommendations.append("No obvious bottlenecks detected. Profile at finer granularity.")
    
    return recommendations


def optimisation_report(
    before_stats: ProfileResult,
    after_stats: ProfileResult
) -> str:
    """
    Generate optimisation comparison report.
    
    Args:
        before_stats: Profile before optimisation.
        after_stats: Profile after optimisation.
        
    Returns:
        Formatted report string.
    """
    speedup = before_stats.total_time / max(after_stats.total_time, 1e-9)
    
    lines = [
        "=" * 50,
        "OPTIMISATION REPORT",
        "=" * 50,
        f"Before: {before_stats.total_time:.4f}s",
        f"After:  {after_stats.total_time:.4f}s",
        f"Speedup: {speedup:.2f}x",
        "",
        "Call count:",
        f"  Before: {before_stats.calls:,}",
        f"  After:  {after_stats.calls:,}",
        "=" * 50
    ]
    
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo() -> None:
    """Run demonstration of all lab components."""
    logger.info("=" * 70)
    logger.info("14UNIT Lab 02: Dask and Profiling - Demonstration")
    logger.info("=" * 70)
    
    check_dask()
    
    # Section 1: Dask Delayed
    logger.info("\n--- Section 1: Dask Delayed ---")
    
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    logger.info("Sequential pipeline (baseline)...")
    start = time.perf_counter()
    seq_result = sequential_pipeline(data)
    seq_time = time.perf_counter() - start
    logger.info(f"Sequential result: {seq_result}, time: {seq_time:.3f}s")
    
    logger.info("\nDelayed pipeline...")
    computation = delayed_pipeline(data)
    
    logger.info("Comparing schedulers...")
    scheduler_times = compare_schedulers(
        computation,
        schedulers=['synchronous', 'threads']
    )
    
    # Section 2: Dask Arrays
    logger.info("\n--- Section 2: Dask Arrays and DataFrames ---")
    
    arr = create_dask_array((1000, 1000), chunks=(250, 250))
    stats = dask_array_operations(arr)
    logger.info(f"Array stats: {stats}")
    
    logger.info("\nCreating sample DataFrame...")
    ddf = create_sample_dask_dataframe(n_rows=100_000)
    
    logger.info("Groupby aggregation...")
    agg_result = dask_groupby_aggregate(
        ddf,
        by='category',
        agg={'value': ['mean', 'sum'], 'count': 'sum'}
    )
    logger.info(f"Aggregation result:\n{agg_result}")
    
    # Section 3: Out-of-Core
    logger.info("\n--- Section 3: Out-of-Core Computation ---")
    
    large_arr = create_dask_array((5000, 5000), chunks=(500, 500))
    stats = incremental_statistics(large_arr)
    logger.info(f"Incremental statistics: {stats}")
    
    # Section 4: Profiling
    logger.info("\n--- Section 4: Profiling ---")
    
    def sample_workload(n: int) -> float:
        total = 0.0
        for i in range(n):
            total += i ** 0.5
        return total
    
    result, profile = profile_function(sample_workload, 100_000)
    logger.info(f"\n{profile}")
    
    bottlenecks = identify_bottlenecks(profile)
    logger.info("\nBottleneck analysis:")
    for rec in bottlenecks:
        logger.info(f"  • {rec}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Demonstration Complete")
    logger.info("=" * 70)


def main() -> None:
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='14UNIT Lab 02: Dask and Profiling'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demonstration mode'
    )
    parser.add_argument(
        '--section',
        type=int,
        choices=[1, 2, 3, 4],
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
        check_dask()
        logger.info(f"Running Section {args.section}...")
        if args.section == 1:
            comp = delayed_pipeline([1.0, 2.0, 3.0])
            result = comp.compute()
            logger.info(f"Result: {result}")
        elif args.section == 2:
            arr = create_dask_array((100, 100))
            dask_array_operations(arr)
        elif args.section == 3:
            arr = create_dask_array((1000, 1000))
            incremental_statistics(arr)
        elif args.section == 4:
            def sample(n: int) -> float:
                return sum(i ** 0.5 for i in range(n))
            _, profile = profile_function(sample, 10000)
            logger.info(str(profile))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
