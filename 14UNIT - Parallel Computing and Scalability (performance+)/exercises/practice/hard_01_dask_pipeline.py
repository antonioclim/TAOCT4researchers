#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Dask Pipeline (Hard)
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ★★★★★ (Hard)
ESTIMATED TIME: 40 minutes
PREREQUISITES: Dask basics, DataFrames, delayed computation

LEARNING OBJECTIVES
───────────────────
- LO5: Build complete Dask data processing pipelines
- LO5: Select appropriate chunking and scheduler strategies

PROBLEM DESCRIPTION
───────────────────
Build an end-to-end data processing pipeline using Dask that can handle
datasets larger than memory. The pipeline should read, transform, aggregate
and persist results.

TASKS
─────
1. Implement `create_large_dataset` - generate sample data as Dask DataFrame
2. Implement `transform_pipeline` - apply transformations with delayed
3. Implement `aggregate_by_groups` - perform groupby aggregation
4. Implement `full_pipeline` - orchestrate complete ETL workflow

HINTS
─────
- Hint 1: Use dask.dataframe.from_delayed for custom data generation
- Hint 2: Apply @delayed to transformation functions
- Hint 3: Choose scheduler based on workload (threads vs processes)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Dask imports with fallback
try:
    import dask
    import dask.dataframe as dd
    from dask import delayed
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    delayed = lambda x: x


def check_dask() -> None:
    """Verify Dask is available."""
    if not HAS_DASK:
        raise ImportError("Dask required: pip install dask[complete]")


def create_large_dataset(
    n_rows: int = 1_000_000,
    n_partitions: int = 10
) -> dd.DataFrame:
    """
    Create a sample Dask DataFrame for pipeline testing.
    
    Generate synthetic data representing sales transactions with:
    - transaction_id: unique identifier
    - category: product category (A, B, C, D, E)
    - region: geographic region (North, South, East, West)
    - amount: transaction amount (normal distribution)
    - quantity: items purchased (1-100)
    
    Args:
        n_rows: Total number of rows.
        n_partitions: Number of Dask partitions.
        
    Returns:
        Dask DataFrame with the specified schema.
        
    Example:
        >>> ddf = create_large_dataset(100_000, n_partitions=4)
        >>> ddf.npartitions
        4
    """
    check_dask()
    # TODO: Implement this function
    # 1. Calculate rows per partition
    # 2. Create delayed functions that generate pandas DataFrames
    # 3. Use dd.from_delayed to combine into Dask DataFrame
    raise NotImplementedError("Implement create_large_dataset")


@delayed
def _transform_partition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to a single partition.
    
    Transformations:
    - Add 'revenue' column: amount * quantity
    - Add 'size_category': 'small' if quantity < 25, 'medium' if < 75, else 'large'
    - Convert category to uppercase
    """
    # TODO: Implement this function (delayed)
    raise NotImplementedError("Implement _transform_partition")


def transform_pipeline(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Apply transformations to Dask DataFrame using delayed.
    
    Args:
        ddf: Input Dask DataFrame.
        
    Returns:
        Transformed Dask DataFrame with additional columns.
        
    Example:
        >>> ddf = create_large_dataset(10000)
        >>> transformed = transform_pipeline(ddf)
        >>> 'revenue' in transformed.columns
        True
    """
    check_dask()
    # TODO: Implement this function
    # Apply _transform_partition to each partition
    raise NotImplementedError("Implement transform_pipeline")


def aggregate_by_groups(
    ddf: dd.DataFrame,
    group_cols: list[str],
    agg_dict: dict[str, str | list[str]]
) -> pd.DataFrame:
    """
    Perform groupby aggregation on Dask DataFrame.
    
    Args:
        ddf: Input Dask DataFrame.
        group_cols: Columns to group by.
        agg_dict: Aggregation specification.
        
    Returns:
        Aggregated pandas DataFrame (computed).
        
    Example:
        >>> ddf = create_large_dataset(10000)
        >>> result = aggregate_by_groups(
        ...     ddf, ['category'], {'amount': ['sum', 'mean']}
        ... )
        >>> len(result) == 5  # 5 categories
        True
    """
    check_dask()
    # TODO: Implement this function
    raise NotImplementedError("Implement aggregate_by_groups")


def full_pipeline(
    n_rows: int = 100_000,
    output_path: Path | None = None
) -> dict[str, Any]:
    """
    Execute complete ETL pipeline.
    
    Pipeline stages:
    1. Generate synthetic data
    2. Apply transformations
    3. Aggregate by category and region
    4. Optionally persist to disk
    
    Args:
        n_rows: Number of rows to generate.
        output_path: Optional path to save results.
        
    Returns:
        Dictionary with:
        - 'summary': Aggregated DataFrame
        - 'total_revenue': Sum of all revenue
        - 'rows_processed': Number of rows
        - 'execution_time': Time in seconds
        
    Example:
        >>> results = full_pipeline(n_rows=50000)
        >>> results['rows_processed']
        50000
    """
    check_dask()
    # TODO: Implement this function
    # 1. Create dataset
    # 2. Transform
    # 3. Aggregate
    # 4. Compute total revenue
    # 5. Optionally save
    # 6. Return results dict
    raise NotImplementedError("Implement full_pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_create_large_dataset() -> None:
    """Test dataset creation."""
    ddf = create_large_dataset(10_000, n_partitions=4)
    assert ddf.npartitions == 4
    assert len(ddf) == 10_000
    assert set(ddf.columns) == {'transaction_id', 'category', 'region', 'amount', 'quantity'}
    print("✓ create_large_dataset tests passed")


def test_transform_pipeline() -> None:
    """Test transformation pipeline."""
    ddf = create_large_dataset(1000, n_partitions=2)
    transformed = transform_pipeline(ddf)
    
    # Check new columns exist
    cols = transformed.columns.tolist()
    assert 'revenue' in cols
    assert 'size_category' in cols
    
    # Compute and verify
    df = transformed.compute()
    assert (df['revenue'] == df['amount'] * df['quantity']).all()
    print("✓ transform_pipeline tests passed")


def test_aggregate_by_groups() -> None:
    """Test aggregation."""
    ddf = create_large_dataset(5000, n_partitions=2)
    result = aggregate_by_groups(
        ddf,
        ['category'],
        {'amount': ['sum', 'mean'], 'quantity': 'sum'}
    )
    assert len(result) == 5  # 5 categories
    print("✓ aggregate_by_groups tests passed")


def test_full_pipeline() -> None:
    """Test complete pipeline."""
    results = full_pipeline(n_rows=10_000)
    assert results['rows_processed'] == 10_000
    assert results['total_revenue'] > 0
    assert 'summary' in results
    print(f"✓ full_pipeline: processed {results['rows_processed']} rows "
          f"in {results['execution_time']:.2f}s")


def main() -> None:
    """Run all tests."""
    print("Running hard_01_dask_pipeline tests...")
    print("-" * 50)
    
    try:
        check_dask()
        test_create_large_dataset()
        test_transform_pipeline()
        test_aggregate_by_groups()
        test_full_pipeline()
        print("-" * 50)
        print("All tests passed! ✓")
    except ImportError as e:
        print(f"Missing dependency: {e}")
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}")
    except AssertionError as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    main()
