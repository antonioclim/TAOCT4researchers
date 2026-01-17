#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Hard 01 — Importance Sampling
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 50 minutes
TOPIC: Advanced Monte Carlo

OBJECTIVE
─────────
Implement importance sampling for estimating rare event probabilities.

BACKGROUND
──────────
Standard Monte Carlo fails when the integrand is concentrated in a small
region. Importance sampling draws from a proposal distribution g(x) that
concentrates probability where f(x) is large, then reweights:

  E[f(X)] = ∫ f(x)p(x)dx = ∫ f(x) [p(x)/g(x)] g(x)dx = E_g[f(X) w(X)]

where w(x) = p(x)/g(x) is the importance weight.

TASKS
─────
1. Implement importance sampling for a tail probability
2. Estimate P(X > 4) where X ~ N(0, 1) using shifted proposal
3. Compare efficiency with standard Monte Carlo

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ISResult:
    """Importance sampling result."""
    
    estimate: float
    standard_error: float
    effective_sample_size: float
    n_samples: int


def standard_mc_tail(
    threshold: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate P(X > threshold) for X ~ N(0,1) using standard MC.
    
    Args:
        threshold: Value to exceed
        n_samples: Number of samples
        rng: Random generator
        
    Returns:
        (estimate, standard_error)
        
    Example:
        >>> rng = np.random.default_rng(42)
        >>> est, se = standard_mc_tail(2.0, 100000, rng)
        >>> abs(est - 0.0228) < 0.005  # P(X > 2) ≈ 0.0228
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample from N(0, 1)
    x = rng.standard_normal(n_samples)
    
    # Indicator: 1 if x > threshold, 0 otherwise
    indicator = (x > threshold).astype(float)
    
    estimate = np.mean(indicator)
    se = np.std(indicator, ddof=1) / np.sqrt(n_samples)
    
    return estimate, se


def importance_sampling_tail(
    threshold: float,
    n_samples: int,
    shift: float,
    rng: np.random.Generator | None = None,
) -> ISResult:
    """Estimate P(X > threshold) for X ~ N(0,1) using importance sampling.
    
    Uses a shifted normal N(shift, 1) as proposal distribution.
    
    Args:
        threshold: Value to exceed
        n_samples: Number of samples
        shift: Mean of proposal distribution
        rng: Random generator
        
    Returns:
        ISResult with estimate, SE and effective sample size
        
    Example:
        >>> rng = np.random.default_rng(42)
        >>> result = importance_sampling_tail(4.0, 10000, shift=4.0, rng=rng)
        >>> abs(result.estimate - 3.17e-5) < 1e-5
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # TODO: Sample from proposal distribution N(shift, 1)
    # x = rng.normal(shift, 1.0, n_samples)
    
    # TODO: Compute indicator function
    # indicator = (x > threshold).astype(float)
    
    # TODO: Compute importance weights w(x) = p(x) / g(x)
    # p(x) = N(0,1) pdf at x
    # g(x) = N(shift, 1) pdf at x
    # 
    # log_p = -0.5 * x**2  # up to constant
    # log_g = -0.5 * (x - shift)**2  # up to constant
    # log_w = log_p - log_g
    # w = np.exp(log_w)
    
    # TODO: Compute weighted estimate
    # weighted_samples = indicator * w
    # estimate = np.mean(weighted_samples)
    
    # TODO: Compute standard error
    # se = np.std(weighted_samples, ddof=1) / np.sqrt(n_samples)
    
    # TODO: Compute effective sample size
    # ESS = (sum(w))^2 / sum(w^2)
    # ess = np.sum(w)**2 / np.sum(w**2)
    
    raise NotImplementedError("Complete this function")


def optimal_shift(threshold: float) -> float:
    """Compute optimal shift for exponential tilting.
    
    For estimating P(X > a) with X ~ N(0,1), the optimal
    proposal is N(a, 1) (shift to threshold).
    """
    return threshold


def compare_methods(
    threshold: float,
    n_samples: int,
    n_trials: int,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, float]]:
    """Compare standard MC vs importance sampling.
    
    Args:
        threshold: Tail threshold
        n_samples: Samples per trial
        n_trials: Number of trials
        rng: Random generator
        
    Returns:
        Dictionary with statistics for each method
    """
    if rng is None:
        rng = np.random.default_rng()
    
    true_prob = 1 - stats.norm.cdf(threshold)
    shift = optimal_shift(threshold)
    
    std_estimates = []
    is_estimates = []
    
    for _ in range(n_trials):
        std_est, _ = standard_mc_tail(threshold, n_samples, rng)
        std_estimates.append(std_est)
        
        is_result = importance_sampling_tail(threshold, n_samples, shift, rng)
        is_estimates.append(is_result.estimate)
    
    std_arr = np.array(std_estimates)
    is_arr = np.array(is_estimates)
    
    return {
        "true_probability": true_prob,
        "standard_mc": {
            "mean": float(np.mean(std_arr)),
            "variance": float(np.var(std_arr, ddof=1)),
            "rmse": float(np.sqrt(np.mean((std_arr - true_prob) ** 2))),
            "zeros": int(np.sum(std_arr == 0)),
        },
        "importance_sampling": {
            "mean": float(np.mean(is_arr)),
            "variance": float(np.var(is_arr, ddof=1)),
            "rmse": float(np.sqrt(np.mean((is_arr - true_prob) ** 2))),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    print("Importance Sampling for Tail Probabilities")
    print("=" * 70)
    
    # Test different thresholds
    thresholds = [2.0, 3.0, 4.0, 5.0]
    
    for threshold in thresholds:
        true_prob = 1 - stats.norm.cdf(threshold)
        
        print(f"\nThreshold = {threshold}")
        print(f"True P(X > {threshold}) = {true_prob:.2e}")
        print("-" * 70)
        
        try:
            # Standard MC
            std_est, std_se = standard_mc_tail(threshold, 100_000, rng)
            print(f"Standard MC:        {std_est:.2e} ± {std_se:.2e}")
            
            # Importance sampling
            is_result = importance_sampling_tail(threshold, 10_000, threshold, rng)
            print(f"Importance Sampling: {is_result.estimate:.2e} ± {is_result.standard_error:.2e}")
            print(f"  Effective sample size: {is_result.effective_sample_size:.0f}")
            
            # Variance reduction
            if std_se > 0 and is_result.standard_error > 0:
                var_ratio = (std_se / is_result.standard_error) ** 2
                print(f"  Variance reduction factor: {var_ratio:.1f}x")
        
        except NotImplementedError:
            print("Complete importance_sampling_tail function first!")
            break
    
    # Detailed comparison for threshold = 4
    print("\n" + "=" * 70)
    print("Detailed Comparison (threshold = 4, 100 trials)")
    print("=" * 70)
    
    try:
        results = compare_methods(4.0, 10_000, 100, rng)
        
        print(f"\nTrue probability: {results['true_probability']:.2e}")
        print(f"\n{'Method':<20} {'Mean':>12} {'Variance':>12} {'RMSE':>12}")
        print("-" * 60)
        
        std = results["standard_mc"]
        print(f"{'Standard MC':<20} {std['mean']:>12.2e} {std['variance']:>12.2e} {std['rmse']:>12.2e}")
        print(f"  ({std['zeros']} trials returned exactly 0)")
        
        iss = results["importance_sampling"]
        print(f"{'Importance Sampling':<20} {iss['mean']:>12.2e} {iss['variance']:>12.2e} {iss['rmse']:>12.2e}")
        
    except NotImplementedError:
        print("Complete importance_sampling_tail function first!")
