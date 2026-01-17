#!/usr/bin/env python3
"""Solution for Easy 01: Monte Carlo Area Estimation."""

from __future__ import annotations

import numpy as np


def estimate_quarter_circle_area(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate area of quarter circle using Monte Carlo."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate random points in [0,1] × [0,1]
    x = rng.uniform(0, 1, n_samples)
    y = rng.uniform(0, 1, n_samples)
    
    # Count points inside quarter circle
    inside = (x**2 + y**2) <= 1
    
    # Estimate area
    area = np.mean(inside)
    
    # Estimate π (area = π/4, so π = 4 × area)
    pi_estimate = 4 * area
    
    return area, pi_estimate


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
        _, pi_est = estimate_quarter_circle_area(n, rng)
        print(f"n={n:>10,}: π ≈ {pi_est:.6f}, error = {abs(pi_est - np.pi):.6f}")
