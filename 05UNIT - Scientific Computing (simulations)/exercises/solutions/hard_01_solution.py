#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: Importance Sampling — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for hard_01_importance_sampling.py

Advanced variance reduction using importance sampling.

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
from scipy import stats


@dataclass
class ISResult:
    """Importance sampling result."""
    
    estimate: float
    standard_error: float
    effective_sample_size: float
    n_samples: int


def standard_mc(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Standard uniform MC for comparison."""
    X = rng.uniform(a, b, n_samples)
    f_values = np.array([f(x) for x in X])
    
    domain = b - a
    estimate = domain * np.mean(f_values)
    se = domain * np.std(f_values, ddof=1) / np.sqrt(n_samples)
    
    return estimate, se


def importance_sampling(
    f: Callable[[float], float],
    g_sample: Callable[[int, np.random.Generator], np.ndarray],
    g_pdf: Callable[[np.ndarray], np.ndarray],
    p_pdf: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> ISResult:
    """
    Importance sampling Monte Carlo.
    
    Estimates E_p[f(X)] using samples from proposal g(x):
    
    Î = (1/n) Σ f(Xᵢ) × p(Xᵢ)/g(Xᵢ)
    
    where Xᵢ ~ g(x) and p(x) is the target distribution.
    
    Args:
        f: Function to compute expectation of
        g_sample: Function to sample from proposal: g_sample(n, rng) → samples
        g_pdf: Proposal PDF
        p_pdf: Target PDF
        n_samples: Number of samples
        rng: Random generator
        
    Returns:
        ISResult with estimate and diagnostics
    """
    # Sample from proposal
    X = g_sample(n_samples, rng)
    
    # Compute importance weights: w(x) = p(x) / g(x)
    weights = p_pdf(X) / g_pdf(X)
    
    # Evaluate function
    f_values = np.array([f(x) for x in X])
    
    # Weighted estimate
    weighted = f_values * weights
    estimate = np.mean(weighted)
    
    # Standard error
    se = np.std(weighted, ddof=1) / np.sqrt(n_samples)
    
    # Effective sample size: n_eff = (Σw)² / Σw²
    normalised_weights = weights / np.sum(weights)
    ess = 1.0 / np.sum(normalised_weights ** 2)
    
    return ISResult(
        estimate=estimate,
        standard_error=se,
        effective_sample_size=ess,
        n_samples=n_samples,
    )


def self_normalised_is(
    f: Callable[[float], float],
    g_sample: Callable[[int, np.random.Generator], np.ndarray],
    g_pdf: Callable[[np.ndarray], np.ndarray],
    p_pdf: Callable[[np.ndarray], np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> ISResult:
    """
    Self-normalised importance sampling.
    
    More stable when p(x) is only known up to normalisation:
    
    Î = Σ f(Xᵢ)w(Xᵢ) / Σ w(Xᵢ)
    
    This is biased but consistent.
    """
    X = g_sample(n_samples, rng)
    weights = p_pdf(X) / g_pdf(X)
    f_values = np.array([f(x) for x in X])
    
    # Self-normalised estimate
    estimate = np.sum(f_values * weights) / np.sum(weights)
    
    # Approximate standard error using delta method
    normalised_weights = weights / np.sum(weights)
    var_weighted = np.sum(normalised_weights * (f_values - estimate)**2)
    se = np.sqrt(var_weighted / n_samples)
    
    ess = 1.0 / np.sum(normalised_weights ** 2)
    
    return ISResult(
        estimate=estimate,
        standard_error=se,
        effective_sample_size=ess,
        n_samples=n_samples,
    )


def optimal_proposal_gaussian(
    f: Callable[[float], float],
    mu: float,
    sigma: float,
    n_samples: int,
    rng: np.random.Generator,
) -> ISResult:
    """
    IS with Gaussian proposal for integrals over ℝ.
    
    Target: E[f(X)] where X ~ N(0, 1)
    Proposal: N(μ, σ²)
    
    Best μ is near where |f(x)| × φ(x) is large.
    """
    # Sample from proposal N(μ, σ)
    X = rng.normal(mu, sigma, n_samples)
    
    # Standard normal target
    p_pdf = stats.norm(0, 1).pdf(X)
    g_pdf = stats.norm(mu, sigma).pdf(X)
    
    weights = p_pdf / g_pdf
    f_values = np.array([f(x) for x in X])
    
    weighted = f_values * weights
    estimate = np.mean(weighted)
    se = np.std(weighted, ddof=1) / np.sqrt(n_samples)
    
    normalised_weights = weights / np.sum(weights)
    ess = 1.0 / np.sum(normalised_weights ** 2)
    
    return ISResult(estimate, se, ess, n_samples)


def rare_event_estimation(
    threshold: float,
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Estimate P(X > threshold) for X ~ N(0,1) using IS.
    
    Standard MC fails for rare events (large threshold).
    IS with shifted proposal is much more efficient.
    
    Args:
        threshold: Probability of exceeding this value
        n_samples: Number of samples
        seed: Random seed
        
    Returns:
        Comparison of standard MC vs importance sampling
    """
    rng = np.random.default_rng(seed)
    true_prob = 1 - stats.norm.cdf(threshold)
    
    # Standard MC
    X_standard = rng.normal(0, 1, n_samples)
    hits_standard = X_standard > threshold
    mc_estimate = np.mean(hits_standard)
    mc_se = np.std(hits_standard, ddof=1) / np.sqrt(n_samples)
    
    # Importance sampling: shift proposal to threshold
    mu_proposal = threshold  # Optimal is around threshold
    X_is = rng.normal(mu_proposal, 1, n_samples)
    
    # Indicator function
    hits_is = X_is > threshold
    
    # Weights
    weights = stats.norm(0, 1).pdf(X_is) / stats.norm(mu_proposal, 1).pdf(X_is)
    weighted_hits = hits_is * weights
    
    is_estimate = np.mean(weighted_hits)
    is_se = np.std(weighted_hits, ddof=1) / np.sqrt(n_samples)
    
    return {
        "true_probability": true_prob,
        "standard_mc": {
            "estimate": mc_estimate,
            "se": mc_se,
            "relative_error": mc_se / true_prob if true_prob > 0 else float('inf'),
        },
        "importance_sampling": {
            "estimate": is_estimate,
            "se": is_se,
            "relative_error": is_se / true_prob if true_prob > 0 else float('inf'),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Importance Sampling for Rare Event Estimation")
    logger.info("=" * 60)
    
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        results = rare_event_estimation(threshold, n_samples=10000)
        
        logger.info(f"\nP(X > {threshold}) where X ~ N(0,1)")
        logger.info(f"  True probability: {results['true_probability']:.2e}")
        logger.info(f"  Standard MC:")
        mc = results['standard_mc']
        logger.info(f"    Estimate: {mc['estimate']:.2e} ± {mc['se']:.2e}")
        logger.info(f"    Relative error: {mc['relative_error']:.1%}")
        logger.info(f"  Importance Sampling:")
        is_r = results['importance_sampling']
        logger.info(f"    Estimate: {is_r['estimate']:.2e} ± {is_r['se']:.2e}")
        logger.info(f"    Relative error: {is_r['relative_error']:.1%}")
        
        if mc['se'] > 0 and is_r['se'] > 0:
            efficiency = (mc['se'] / is_r['se']) ** 2
            logger.info(f"  Efficiency gain: {efficiency:.1f}x")
    
    logger.info("\n" + "=" * 60)
    logger.info("Key insight: IS dramatically improves rare event estimation")
    logger.info("by sampling from where events actually occur.")
