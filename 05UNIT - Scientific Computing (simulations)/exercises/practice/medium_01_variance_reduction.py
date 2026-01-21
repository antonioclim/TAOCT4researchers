#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Medium 01 — Variance Reduction Techniques
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 35 minutes
TOPIC: Monte Carlo variance reduction

OBJECTIVE
─────────
Implement antithetic variates to reduce variance in Monte Carlo integration.

BACKGROUND
──────────
For monotonic functions, pairing each sample X with its "mirror" (1-X for
uniform [0,1]) creates negative correlation that reduces variance. If f is
increasing, when f(X) is above average, f(1-X) tends to be below average.

The antithetic estimator is: I = (1/n) Σ [f(Xᵢ) + f(1-Xᵢ)] / 2

TASKS
─────
1. Complete the `standard_mc` function
2. Complete the `antithetic_mc` function
3. Compare variances empirically

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MCResult:
    """Monte Carlo integration result."""
    
    estimate: float
    standard_error: float
    n_samples: int
    method: str


def standard_mc(
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> MCResult:
    """Standard Monte Carlo integration of f over [0, 1].
    
    Args:
        f: Function to integrate (vectorised)
        n_samples: Number of samples
        rng: Random number generator
        
    Returns:
        MCResult with estimate and standard error
        
    Example:
        >>> rng = np.random.default_rng(42)
        >>> result = standard_mc(lambda x: x**2, 10000, rng)
        >>> abs(result.estimate - 1/3) < 0.01
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # TODO: Generate n_samples uniform random numbers in [0, 1]
    # x = rng.uniform(0, 1, n_samples)
    
    # TODO: Evaluate f at all sample points
    # y = f(x)
    
    # TODO: Compute estimate (sample mean)
    # estimate = np.mean(y)
    
    # TODO: Compute standard error (std / sqrt(n))
    # se = np.std(y, ddof=1) / np.sqrt(n_samples)
    
    raise NotImplementedError("Complete this function")


def antithetic_mc(
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> MCResult:
    """Antithetic variates Monte Carlo integration over [0, 1].
    
    Args:
        f: Function to integrate (vectorised)
        n_samples: Total effective samples (n_samples/2 pairs)
        rng: Random number generator
        
    Returns:
        MCResult with estimate and standard error
        
    Example:
        >>> rng = np.random.default_rng(42)
        >>> result = antithetic_mc(lambda x: np.exp(x), 10000, rng)
        >>> abs(result.estimate - (np.e - 1)) < 0.01
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    _n_pairs = n_samples // 2
    
    # TODO: Generate n_pairs uniform random numbers
    # x = rng.uniform(0, 1, n_pairs)
    
    # TODO: Create antithetic samples
    # x_anti = 1 - x
    
    # TODO: Evaluate f at both x and antithetic x
    # y = f(x)
    # y_anti = f(x_anti)
    
    # TODO: Compute paired averages
    # paired_avg = (y + y_anti) / 2
    
    # TODO: Compute estimate from paired averages
    # estimate = np.mean(paired_avg)
    
    # TODO: Compute standard error of paired averages
    # se = np.std(paired_avg, ddof=1) / np.sqrt(n_pairs)
    
    raise NotImplementedError("Complete this function")


def compare_methods(
    f: Callable[[np.ndarray], np.ndarray],
    true_value: float,
    n_samples: int,
    n_trials: int,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Compare standard MC vs antithetic variates.
    
    Args:
        f: Function to integrate
        true_value: Exact integral value
        n_samples: Samples per trial
        n_trials: Number of independent trials
        rng: Random number generator
        
    Returns:
        Dictionary with statistics for each method
    """
    if rng is None:
        rng = np.random.default_rng()
    
    std_estimates = []
    anti_estimates = []
    
    for _ in range(n_trials):
        std_result = standard_mc(f, n_samples, rng)
        anti_result = antithetic_mc(f, n_samples, rng)
        
        std_estimates.append(std_result.estimate)
        anti_estimates.append(anti_result.estimate)
    
    std_arr = np.array(std_estimates)
    anti_arr = np.array(anti_estimates)
    
    return {
        "standard": {
            "mean": float(np.mean(std_arr)),
            "variance": float(np.var(std_arr, ddof=1)),
            "mse": float(np.mean((std_arr - true_value) ** 2)),
        },
        "antithetic": {
            "mean": float(np.mean(anti_arr)),
            "variance": float(np.var(anti_arr, ddof=1)),
            "mse": float(np.mean((anti_arr - true_value) ** 2)),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def exponential(x: np.ndarray) -> np.ndarray:
    """f(x) = e^x, integral over [0,1] = e - 1 ≈ 1.718."""
    return np.exp(x)


def quadratic(x: np.ndarray) -> np.ndarray:
    """f(x) = x², integral over [0,1] = 1/3."""
    return x ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    print("Variance Reduction: Antithetic Variates")
    print("=" * 60)
    
    test_cases = [
        ("f(x) = e^x", exponential, np.e - 1),
        ("f(x) = x²", quadratic, 1/3),
    ]
    
    for name, f, true_val in test_cases:
        print(f"\nFunction: {name}")
        print(f"True integral: {true_val:.6f}")
        print("-" * 60)
        
        try:
            results = compare_methods(f, true_val, n_samples=1000, n_trials=100, rng=rng)
            
            print(f"{'Method':<15} {'Mean':>12} {'Variance':>12} {'MSE':>12}")
            print("-" * 60)
            
            for method, stats in results.items():
                print(f"{method:<15} {stats['mean']:>12.6f} {stats['variance']:>12.8f} {stats['mse']:>12.8f}")
            
            variance_reduction = 1 - results["antithetic"]["variance"] / results["standard"]["variance"]
            print(f"\nVariance reduction: {variance_reduction*100:.1f}%")
            
        except NotImplementedError:
            print("Complete the standard_mc and antithetic_mc functions first!")
