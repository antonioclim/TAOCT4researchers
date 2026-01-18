"""
14UNIT Laboratory Package: Parallel Computing and Scalability

This package provides laboratory exercises for developing practical skills
in parallel and scalable computing with Python.

Modules:
    lab_14_01_multiprocessing: Process and thread-based parallelism
    lab_14_02_dask_profiling: Dask framework and performance profiling

Author: Antonio Clim
Version: 4.1.0
Date: January 2025
Licence: Restrictive - See README.md

Example:
    >>> from lab import lab_14_01_multiprocessing as mp_lab
    >>> mp_lab.estimate_pi_parallel(1_000_000, n_workers=4)
"""

from __future__ import annotations

__version__ = '4.1.0'
__author__ = 'Antonio Clim'
__all__ = ['lab_14_01_multiprocessing', 'lab_14_02_dask_profiling']
