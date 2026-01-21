"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Algorithmic Complexity — Laboratory Package
═══════════════════════════════════════════════════════════════════════════════

This package provides tools for benchmarking algorithms and analysing their
computational complexity both empirically and theoretically.

MODULES
───────
- lab_3_01_benchmark_suite: Thorough benchmarking framework with
  statistical analysis, warmup handling and result visualisation.

- lab_3_02_complexity_analyser: Automatic complexity estimation using
  log-log regression and curve fitting techniques.

USAGE
─────
    from lab import BenchmarkSuite, ComplexityAnalyser

    # Benchmark sorting algorithms
    suite = BenchmarkSuite()
    suite.add_algorithm("bubble", bubble_sort)
    suite.add_algorithm("merge", merge_sort)
    results = suite.run([100, 1000, 10000])

    # Estimate complexity
    analyser = ComplexityAnalyser(results)
    complexity = analyser.estimate_all()

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

__version__ = "1.0.0"
__author__ = "Antonio Clim"
__all__ = [
    # Benchmark Suite
    "BenchmarkResult",
    "BenchmarkSuite",
    "timer",
    "benchmark",
    # Complexity Analyser
    "ComplexityAnalyser",
    "ComplexityModel",
    "estimate_complexity",
    # Sorting Algorithms
    "bubble_sort",
    "insertion_sort",
    "merge_sort",
    "quicksort",
    # Utilities
    "generate_test_data",
    "compute_speedup",
]

# Lazy imports for better startup time
def __getattr__(name: str):
    """Lazy import mechanism for package components."""
    if name in {
        "BenchmarkResult",
        "BenchmarkSuite", 
        "timer",
        "benchmark",
        "bubble_sort",
        "insertion_sort",
        "merge_sort",
        "quicksort",
        "generate_test_data",
        "compute_speedup",
    }:
        from . import lab_3_01_benchmark_suite as bs
        return getattr(bs, name)
    
    if name in {
        "ComplexityAnalyser",
        "ComplexityModel",
        "estimate_complexity",
    }:
        from . import lab_3_02_complexity_analyser as ca
        return getattr(ca, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
