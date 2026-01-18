#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Lab 01: Multiprocessing and Threading - SOLUTION
═══════════════════════════════════════════════════════════════════════════════

This solution file provides complete, documented implementations of all
laboratory exercises. Use for self-assessment after attempting exercises.

IMPORTANT: Consult this only after making a genuine attempt at the exercises.

Author: Antonio Clim
Version: 4.0.0
Licence: Restrictive - See README.md
═══════════════════════════════════════════════════════════════════════════════
"""

# The solution re-exports all functions from the main lab
# The lab file already contains complete implementations

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lab.lab_14_01_multiprocessing import (
    # Section 1
    cpu_intensive_task,
    spawn_process,
    process_with_return_value,
    demonstrate_gil,
    # Section 2
    parallel_map,
    parallel_starmap,
    parallel_async_with_callback,
    chunk_and_process,
    # Section 3
    estimate_pi_sequential,
    estimate_pi_parallel,
    benchmark_scaling,
    parallel_bootstrap,
    # Section 4
    shared_counter_race,
    shared_counter_lock,
    shared_array_processing,
    producer_consumer_queue,
    # Section 5
    download_urls_sequential,
    download_urls_threaded,
    demonstrate_io_speedup,
    run_demo,
)

__all__ = [
    'cpu_intensive_task', 'spawn_process', 'process_with_return_value',
    'demonstrate_gil', 'parallel_map', 'parallel_starmap',
    'parallel_async_with_callback', 'chunk_and_process',
    'estimate_pi_sequential', 'estimate_pi_parallel', 'benchmark_scaling',
    'parallel_bootstrap', 'shared_counter_race', 'shared_counter_lock',
    'shared_array_processing', 'producer_consumer_queue',
    'download_urls_sequential', 'download_urls_threaded', 'demonstrate_io_speedup',
]

if __name__ == '__main__':
    run_demo()
