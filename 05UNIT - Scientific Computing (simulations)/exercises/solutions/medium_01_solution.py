#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: Variance Reduction Techniques — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for medium_01_variance_reduction.py

Demonstrates antithetic variates and stratified sampling for Monte Carlo.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class MCResult:
    """Monte Carlo result container."""
    
    estimate: float
    standard_error: float
    n_samples: int
    method: str


def standard_mc(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int,
    rng: np.random.Generator,
) -> MCResult:
    """Standard Monte Carlo integration."""
    X = rng.uniform(a, b, n_samples)
    f_values = np.array([f(x) for x in X])
    
    domain = b - a
    estimate = domain * np.mean(f_values)
    se = domain * np.std(f_values, ddof=1) / np.sqrt(n_samples)
    
    return MCResult(estimate, se, n_samples, "standard")


def antithetic_mc(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_pairs: int,
    rng: np.random.Generator,
) -> MCResult:
    """
    Antithetic variates Monte Carlo.
    
    For monotonic functions, f(X) and f(a+b-X) are negatively correlated,
    reducing variance of the paired average.
    """
    X = rng.uniform(a, b, n_pairs)
    X_anti = a + b - X  # Mirror around midpoint
    
    f_X = np.array([f(x) for x in X])
    f_anti = np.array([f(x) for x in X_anti])
    
    # Paired averages
    paired = (f_X + f_anti) / 2
    
    domain = b - a
    estimate = domain * np.mean(paired)
    se = domain * np.std(paired, ddof=1) / np.sqrt(n_pairs)
    
    return MCResult(estimate, se, 2 * n_pairs, "antithetic")


def stratified_mc(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_strata: int,
    samples_per_stratum: int,
    rng: np.random.Generator,
) -> MCResult:
    """
    Stratified sampling Monte Carlo.
    
    Divides domain into equal strata and samples uniformly within each.
    Ensures coverage and reduces variance.
    """
    stratum_width = (b - a) / n_strata
    stratum_means = []
    stratum_vars = []
    
    for k in range(n_strata):
        low = a + k * stratum_width
        high = a + (k + 1) * stratum_width
        
        X_k = rng.uniform(low, high, samples_per_stratum)
        f_values = np.array([f(x) for x in X_k])
        
        stratum_means.append(np.mean(f_values))
        stratum_vars.append(np.var(f_values, ddof=1))
    
    domain = b - a
    estimate = domain * np.mean(stratum_means)
    
    # Variance of stratified estimator
    var_est = np.sum(stratum_vars) / (n_strata**2 * samples_per_stratum)
    se = domain * np.sqrt(var_est)
    
    n_total = n_strata * samples_per_stratum
    return MCResult(estimate, se, n_total, "stratified")


def compare_methods(
    f: Callable[[float], float],
    a: float,
    b: float,
    true_value: float,
    n_samples: int = 10000,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, dict]:
    """
    Compare variance reduction methods.
    
    Args:
        f: Function to integrate
        a, b: Integration bounds
        true_value: Known integral value
        n_samples: Samples per trial
        n_trials: Number of trials for statistics
        seed: Random seed
        
    Returns:
        Dictionary with statistics for each method
    """
    rng = np.random.default_rng(seed)
    
    methods = {
        "standard": lambda: standard_mc(f, a, b, n_samples, rng),
        "antithetic": lambda: antithetic_mc(f, a, b, n_samples // 2, rng),
        "stratified": lambda: stratified_mc(f, a, b, 20, n_samples // 20, rng),
    }
    
    results = {}
    
    for name, method in methods.items():
        estimates = []
        for _ in range(n_trials):
            result = method()
            estimates.append(result.estimate)
        
        estimates = np.array(estimates)
        results[name] = {
            "mean": np.mean(estimates),
            "std": np.std(estimates),
            "bias": np.mean(estimates) - true_value,
            "mse": np.mean((estimates - true_value) ** 2),
        }
    
    return results


def variance_reduction_ratio(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int = 10000,
    n_trials: int = 100,
    seed: int = 42,
) -> dict[str, float]:
    """
    Compute variance reduction ratio for each method.
    
    Ratio = Var(standard) / Var(method)
    Values > 1 indicate variance reduction.
    """
    rng = np.random.default_rng(seed)
    
    # Collect estimates
    standard_est = []
    antithetic_est = []
    stratified_est = []
    
    for _ in range(n_trials):
        standard_est.append(standard_mc(f, a, b, n_samples, rng).estimate)
        antithetic_est.append(antithetic_mc(f, a, b, n_samples // 2, rng).estimate)
        stratified_est.append(stratified_mc(f, a, b, 20, n_samples // 20, rng).estimate)
    
    var_standard = np.var(standard_est)
    
    return {
        "standard": 1.0,
        "antithetic": var_standard / np.var(antithetic_est),
        "stratified": var_standard / np.var(stratified_est),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test function: ∫₀¹ e^x dx = e - 1 ≈ 1.7183
    def f(x: float) -> float:
        return np.exp(x)
    
    true_value = np.e - 1
    
    logger.info("=" * 60)
    logger.info("Variance Reduction Comparison")
    logger.info("=" * 60)
    logger.info(f"Function: e^x on [0, 1]")
    logger.info(f"True value: {true_value:.6f}")
    logger.info("")
    
    # Compare methods
    results = compare_methods(f, 0, 1, true_value, n_samples=10000)
    
    for method, stats in results.items():
        logger.info(f"{method.upper()}")
        logger.info(f"  Mean estimate: {stats['mean']:.6f}")
        logger.info(f"  Std deviation: {stats['std']:.6f}")
        logger.info(f"  MSE: {stats['mse']:.2e}")
    
    logger.info("")
    
    # Variance reduction ratios
    ratios = variance_reduction_ratio(f, 0, 1)
    logger.info("Variance Reduction Ratios:")
    for method, ratio in ratios.items():
        logger.info(f"  {method}: {ratio:.2f}x")
    
    logger.info("")
    logger.info("Antithetic works well for monotonic functions (e^x).")
    logger.info("Stratified ensures coverage across domain.")
