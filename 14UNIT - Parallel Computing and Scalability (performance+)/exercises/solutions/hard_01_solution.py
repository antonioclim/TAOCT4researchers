#!/usr/bin/env python3
"""
14UNIT Exercise Solution: Dask Pipeline (Hard)
© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import dask
    import dask.dataframe as dd
    from dask import delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def check_dask() -> None:
    if not HAS_DASK:
        raise ImportError("Dask required: pip install dask[complete]")


@delayed
def _generate_partition(partition_id: int, rows: int) -> pd.DataFrame:
    """Generate a single partition of data."""
    np.random.seed(partition_id)
    return pd.DataFrame({
        'transaction_id': range(partition_id * rows, (partition_id + 1) * rows),
        'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], rows),
        'amount': np.random.normal(100, 30, rows),
        'quantity': np.random.randint(1, 101, rows)
    })


def create_large_dataset(n_rows: int = 1_000_000, n_partitions: int = 10) -> dd.DataFrame:
    """Create a sample Dask DataFrame."""
    check_dask()
    rows_per_partition = n_rows // n_partitions
    
    delayed_dfs = [
        _generate_partition(i, rows_per_partition)
        for i in range(n_partitions)
    ]
    
    meta = pd.DataFrame({
        'transaction_id': pd.Series(dtype='int64'),
        'category': pd.Series(dtype='object'),
        'region': pd.Series(dtype='object'),
        'amount': pd.Series(dtype='float64'),
        'quantity': pd.Series(dtype='int64')
    })
    
    return dd.from_delayed(delayed_dfs, meta=meta)


def _transform_partition_impl(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transformations to a partition."""
    df = df.copy()
    df['revenue'] = df['amount'] * df['quantity']
    df['size_category'] = pd.cut(
        df['quantity'],
        bins=[0, 25, 75, 101],
        labels=['small', 'medium', 'large']
    )
    df['category'] = df['category'].str.upper()
    return df


def transform_pipeline(ddf: dd.DataFrame) -> dd.DataFrame:
    """Apply transformations to Dask DataFrame."""
    check_dask()
    return ddf.map_partitions(_transform_partition_impl)


def aggregate_by_groups(
    ddf: dd.DataFrame,
    group_cols: list[str],
    agg_dict: dict[str, str | list[str]]
) -> pd.DataFrame:
    """Perform groupby aggregation."""
    check_dask()
    return ddf.groupby(group_cols).agg(agg_dict).compute()


def full_pipeline(n_rows: int = 100_000, output_path: Path | None = None) -> dict[str, Any]:
    """Execute complete ETL pipeline."""
    check_dask()
    start = time.perf_counter()
    
    # Generate
    ddf = create_large_dataset(n_rows, n_partitions=10)
    
    # Transform
    ddf = transform_pipeline(ddf)
    
    # Aggregate
    summary = aggregate_by_groups(
        ddf, ['category', 'region'],
        {'revenue': ['sum', 'mean'], 'quantity': 'sum'}
    )
    
    # Total revenue
    total_revenue = float(ddf['revenue'].sum().compute())
    
    elapsed = time.perf_counter() - start
    
    # Save if requested
    if output_path:
        summary.to_csv(output_path / 'summary.csv')
    
    return {
        'summary': summary,
        'total_revenue': total_revenue,
        'rows_processed': n_rows,
        'execution_time': elapsed
    }


if __name__ == '__main__':
    results = full_pipeline(50_000)
    print(f"Processed {results['rows_processed']} rows in {results['execution_time']:.2f}s")
