#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Easy 01 — Monte Carlo Area Estimation
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes
TOPIC: Monte Carlo basics

OBJECTIVE
─────────
Estimate the area of a quarter circle using the hit-or-miss Monte Carlo method.

BACKGROUND
──────────
The area of a quarter circle with radius 1 is π/4. By sampling random points
in the unit square [0,1]×[0,1] and counting how many fall inside the quarter
circle (x² + y² ≤ 1), we can estimate this area.

TASKS
─────
1. Complete the `estimate_quarter_circle_area` function
2. Run with different sample sizes and observe convergence
3. Calculate the estimated value of π from your result

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np


def estimate_quarter_circle_area(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate area of quarter circle using Monte Carlo.
    
    Args:
        n_samples: Number of random points to sample
        rng: Random number generator for reproducibility
        
    Returns:
        Tuple of (estimated_area, estimated_pi)
        
    Example:
        >>> rng = np.random.default_rng(42)
        >>> area, pi_est = estimate_quarter_circle_area(100_000, rng)
        >>> abs(area - np.pi/4) < 0.01
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # TODO: Generate n_samples random (x, y) points in [0, 1] × [0, 1]
    # x = ...
    # y = ...
    
    # TODO: Count points inside the quarter circle (x² + y² ≤ 1)
    # inside = ...
    
    # TODO: Estimate area as fraction inside × area of square (which is 1)
    # area = ...
    
    # TODO: Estimate π from the area (area of quarter circle = π/4)
    # pi_estimate = ...
    
    raise NotImplementedError("Complete this function")


def convergence_study(
    sample_sizes: list[int],
    rng: np.random.Generator | None = None,
) -> dict[str, list[float]]:
    """Study convergence of Monte Carlo estimate.
    
    Args:
        sample_sizes: List of sample sizes to test
        rng: Random number generator
        
    Returns:
        Dictionary with 'n', 'area', 'pi_estimate', 'error' lists
    """
    if rng is None:
        rng = np.random.default_rng()
    
    results = {"n": [], "area": [], "pi_estimate": [], "error": []}
    
    for n in sample_sizes:
        area, pi_est = estimate_quarter_circle_area(n, rng)
        results["n"].append(n)
        results["area"].append(area)
        results["pi_estimate"].append(pi_est)
        results["error"].append(abs(pi_est - np.pi))
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    # Test with increasing sample sizes
    sizes = [100, 1_000, 10_000, 100_000, 1_000_000]
    
    print("Monte Carlo π Estimation")
    print("=" * 50)
    print(f"{'n':>10} {'π estimate':>12} {'Error':>12}")
    print("-" * 50)
    
    for n in sizes:
        try:
            _, pi_est = estimate_quarter_circle_area(n, rng)
            error = abs(pi_est - np.pi)
            print(f"{n:>10,} {pi_est:>12.6f} {error:>12.6f}")
        except NotImplementedError:
            print("Complete the estimate_quarter_circle_area function first!")
            break
    
    print("-" * 50)
    print(f"{'True π':>10} {np.pi:>12.6f}")
