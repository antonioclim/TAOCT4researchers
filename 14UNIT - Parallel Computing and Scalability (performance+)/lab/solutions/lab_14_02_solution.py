#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Lab 02: Dask and Profiling - SOLUTION
═══════════════════════════════════════════════════════════════════════════════

This solution file provides complete, documented implementations of all
laboratory exercises. Use for self-assessment after attempting exercises.

Author: Antonio Clim
Version: 4.0.0
Licence: Restrictive - See README.md
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lab.lab_14_02_dask_profiling import (
    # Section 1: Dask Delayed
    expensive_operation,
    sequential_pipeline,
    delayed_pipeline,
    visualise_task_graph,
    compare_schedulers,
    # Section 2: Arrays and DataFrames
    create_dask_array,
    dask_array_operations,
    create_sample_dask_dataframe,
    dask_groupby_aggregate,
    # Section 3: Out-of-Core
    process_larger_than_memory,
    incremental_statistics,
    persist_intermediate_results,
    # Section 4: Profiling
    ProfileResult,
    profile_function,
    memory_profile_function,
    identify_bottlenecks,
    optimisation_report,
    run_demo,
)

__all__ = [
    'expensive_operation', 'sequential_pipeline', 'delayed_pipeline',
    'visualise_task_graph', 'compare_schedulers', 'create_dask_array',
    'dask_array_operations', 'create_sample_dask_dataframe',
    'dask_groupby_aggregate', 'process_larger_than_memory',
    'incremental_statistics', 'persist_intermediate_results',
    'ProfileResult', 'profile_function', 'memory_profile_function',
    'identify_bottlenecks', 'optimisation_report',
]

if __name__ == '__main__':
    run_demo()
