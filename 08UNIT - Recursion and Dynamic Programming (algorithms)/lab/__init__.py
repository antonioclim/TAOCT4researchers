"""
Lab modules for Unit 08: Recursion and Dynamic Programming.

This package contains laboratory exercises exploring recursive algorithm design,
memoisation techniques and dynamic programming optimisation strategies.

Modules:
    lab_08_01_recursive_patterns: Recursive algorithms and memoisation
    lab_08_02_dynamic_programming: Bottom-up DP solutions

Example:
    >>> from lab import lab_08_01_recursive_patterns as lab01
    >>> lab01.fibonacci_memoised(50)
    12586269025

Author: Dr. Antonio Clim
Institution: Academy of Economic Studies, Bucharest (ASE-CSIE)
Version: 4.0.0
"""

from . import lab_08_01_recursive_patterns
from . import lab_08_02_dynamic_programming

__all__ = [
    "lab_08_01_recursive_patterns",
    "lab_08_02_dynamic_programming",
]

__version__ = "4.0.0"
__author__ = "Dr. Antonio Clim"
