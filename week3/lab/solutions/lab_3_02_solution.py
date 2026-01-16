#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Lab 2: Complexity Analyser — Solution File
═══════════════════════════════════════════════════════════════════════════════

Extended solutions demonstrating advanced complexity analysis techniques.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from lab_3_02_complexity_analyser import (
    ComplexityAnalyser,
    estimate_exponent,
    classify_exponent,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED SOLUTION: Automatic Algorithm Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlgorithmProfile:
    """Complete profile of an algorithm's complexity."""
    name: str
    theoretical_complexity: str
    empirical_complexity: str
    exponent: float
    r_squared: float
    matches_theory: bool
    analysis_notes: str


def analyse_algorithm(
    algorithm: Callable[[list[float]], list[float]],
    name: str,
    theoretical_complexity: str,
    sizes: list[int] | None = None,
    runs: int = 5
) -> AlgorithmProfile:
    """
    Complete analysis pipeline for an algorithm.
    
    Args:
        algorithm: Function to analyse.
        name: Algorithm name.
        theoretical_complexity: Expected complexity (e.g., "O(n²)").
        sizes: Input sizes to test.
        runs: Number of runs per size.
    
    Returns:
        Complete AlgorithmProfile.
    """
    import time
    import gc
    import random
    
    if sizes is None:
        sizes = [100, 500, 1000, 2000, 5000]
    
    times: list[float] = []
    
    for n in sizes:
        data = [random.random() for _ in range(n)]
        
        # Warmup
        algorithm(data.copy())
        
        gc.collect()
        gc.disable()
        
        run_times: list[float] = []
        for _ in range(runs):
            data_copy = data.copy()
            start = time.perf_counter()
            algorithm(data_copy)
            elapsed = time.perf_counter() - start
            run_times.append(elapsed * 1000)
        
        gc.enable()
        
        # Use median for robustness
        times.append(sorted(run_times)[len(run_times) // 2])
    
    # Analyse
    exponent, r_squared = estimate_exponent(sizes, times)
    empirical = classify_exponent(exponent)
    
    # Check if empirical matches theoretical
    # Extract expected exponent from theoretical complexity
    theory_exp = _parse_complexity_exponent(theoretical_complexity)
    matches = abs(exponent - theory_exp) < 0.3
    
    # Generate notes
    notes = []
    if not matches:
        notes.append(f"Empirical exponent ({exponent:.2f}) differs from theoretical ({theory_exp:.2f})")
    if r_squared < 0.95:
        notes.append(f"Low R² ({r_squared:.3f}) suggests noisy measurements")
    if r_squared > 0.99:
        notes.append("Excellent fit to power-law model")
    
    return AlgorithmProfile(
        name=name,
        theoretical_complexity=theoretical_complexity,
        empirical_complexity=empirical,
        exponent=exponent,
        r_squared=r_squared,
        matches_theory=matches,
        analysis_notes="; ".join(notes) if notes else "Analysis consistent with theory"
    )


def _parse_complexity_exponent(complexity: str) -> float:
    """Extract exponent from complexity string."""
    complexity = complexity.lower().replace(" ", "")
    
    if "o(1)" in complexity:
        return 0.0
    elif "o(logn)" in complexity:
        return 0.0  # Not a power law
    elif "o(n)" in complexity and "log" not in complexity:
        return 1.0
    elif "o(nlogn)" in complexity:
        return 1.0  # Approximately
    elif "o(n²)" in complexity or "o(n^2)" in complexity:
        return 2.0
    elif "o(n³)" in complexity or "o(n^3)" in complexity:
        return 3.0
    else:
        # Try to extract exponent
        import re
        match = re.search(r'n\^?([\d.]+)', complexity)
        if match:
            return float(match.group(1))
        return 1.0


def demo_algorithm_analysis() -> None:
    """Demonstrate the complete analysis pipeline."""
    logger.info("═" * 60)
    logger.info("  SOLUTION: Automatic Algorithm Analysis Pipeline")
    logger.info("═" * 60)
    
    # Define algorithms to analyse
    def bubble_sort(arr: list[float]) -> list[float]:
        arr = arr.copy()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    def insertion_sort(arr: list[float]) -> list[float]:
        arr = arr.copy()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return arr
    
    algorithms = [
        (bubble_sort, "Bubble Sort", "O(n²)"),
        (insertion_sort, "Insertion Sort", "O(n²)"),
        (sorted, "Python sorted()", "O(n log n)"),
    ]
    
    logger.info("Analysing algorithms...\n")
    
    for algo, name, theoretical in algorithms:
        profile = analyse_algorithm(algo, name, theoretical)
        
        logger.info(f"▸ {profile.name}")
        logger.info(f"  Theoretical: {profile.theoretical_complexity}")
        logger.info(f"  Empirical:   {profile.empirical_complexity} (exp={profile.exponent:.2f})")
        logger.info(f"  R²: {profile.r_squared:.4f}")
        logger.info(f"  Match: {'✓' if profile.matches_theory else '✗'}")
        logger.info(f"  Notes: {profile.analysis_notes}")
        logger.info("")


if __name__ == "__main__":
    demo_algorithm_analysis()
