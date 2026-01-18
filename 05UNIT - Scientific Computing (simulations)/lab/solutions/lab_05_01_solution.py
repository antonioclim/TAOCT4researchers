#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab 1: Monte Carlo Methods — SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

Complete reference implementation for lab_5_01_monte_carlo.py

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

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]


@dataclass
class MonteCarloResult:
    """Result container for Monte Carlo integration."""
    
    estimate: float
    standard_error: float
    n_samples: int
    confidence_interval: tuple[float, float]
    method: str = "standard"
    
    @property
    def relative_error(self) -> float:
        """Relative standard error as fraction of estimate."""
        if abs(self.estimate) < 1e-10:
            return float('inf')
        return self.standard_error / abs(self.estimate)


def monte_carlo_integrate(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """
    Estimate ∫ₐᵇ f(x) dx using Monte Carlo integration.
    
    The estimator is: Î = (b-a) × mean(f(Xᵢ))
    where Xᵢ ~ Uniform(a, b).
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n_samples: Number of random samples
        rng: Random number generator
        
    Returns:
        MonteCarloResult with estimate and statistics
        
    Raises:
        ValueError: If n_samples < 1 or a >= b
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate uniform samples
    X = rng.uniform(a, b, n_samples)
    
    # Evaluate function (vectorised if possible)
    try:
        f_values = f(X)
    except (TypeError, ValueError):
        f_values = np.array([f(x) for x in X])
    
    # Compute estimate: Î = (b-a) × mean(f(X))
    domain_length = b - a
    estimate = domain_length * np.mean(f_values)
    
    # Compute standard error
    if n_samples > 1:
        variance = np.var(f_values, ddof=1)
        standard_error = domain_length * np.sqrt(variance / n_samples)
    else:
        standard_error = float('nan')
    
    # 95% confidence interval
    z = 1.96
    ci_low = estimate - z * standard_error
    ci_high = estimate + z * standard_error
    
    logger.debug(
        f"MC integration: n={n_samples}, estimate={estimate:.6f}, "
        f"SE={standard_error:.6f}"
    )
    
    return MonteCarloResult(
        estimate=estimate,
        standard_error=standard_error,
        n_samples=n_samples,
        confidence_interval=(ci_low, ci_high),
        method="standard",
    )


def estimate_pi(
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """
    Estimate π using the hit-or-miss method.
    
    Samples points uniformly in [0,1]² and counts the fraction
    inside the unit quarter-circle. π ≈ 4 × (points inside / total).
    
    Args:
        n_samples: Number of random points
        rng: Random number generator
        
    Returns:
        MonteCarloResult with π estimate
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate points in unit square
    X = rng.uniform(0, 1, n_samples)
    Y = rng.uniform(0, 1, n_samples)
    
    # Check if inside quarter circle
    inside = (X**2 + Y**2) <= 1.0
    
    # π ≈ 4 × (area of quarter circle) / (area of square)
    p = np.mean(inside)
    estimate = 4.0 * p
    
    # Standard error for proportion
    if n_samples > 1:
        se_p = np.sqrt(p * (1 - p) / n_samples)
        standard_error = 4.0 * se_p
    else:
        standard_error = float('nan')
    
    z = 1.96
    ci_low = estimate - z * standard_error
    ci_high = estimate + z * standard_error
    
    return MonteCarloResult(
        estimate=estimate,
        standard_error=standard_error,
        n_samples=n_samples,
        confidence_interval=(ci_low, ci_high),
        method="hit-or-miss",
    )


def antithetic_variates(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_pairs: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """
    Monte Carlo with antithetic variates for variance reduction.
    
    For each sample X, also uses the "mirror" sample (a + b - X).
    The negative correlation between f(X) and f(a+b-X) for monotonic
    functions reduces variance.
    
    Args:
        f: Function to integrate (should be monotonic for best results)
        a: Lower bound
        b: Upper bound
        n_pairs: Number of sample pairs (total samples = 2×n_pairs)
        rng: Random number generator
        
    Returns:
        MonteCarloResult with reduced-variance estimate
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate primary samples
    X = rng.uniform(a, b, n_pairs)
    
    # Antithetic samples: reflected around midpoint
    X_anti = a + b - X
    
    # Evaluate at both
    try:
        f_X = f(X)
        f_anti = f(X_anti)
    except (TypeError, ValueError):
        f_X = np.array([f(x) for x in X])
        f_anti = np.array([f(x) for x in X_anti])
    
    # Paired averages
    paired_means = (f_X + f_anti) / 2
    
    # Estimate
    domain_length = b - a
    estimate = domain_length * np.mean(paired_means)
    
    # Standard error of paired means
    if n_pairs > 1:
        variance = np.var(paired_means, ddof=1)
        standard_error = domain_length * np.sqrt(variance / n_pairs)
    else:
        standard_error = float('nan')
    
    z = 1.96
    ci_low = estimate - z * standard_error
    ci_high = estimate + z * standard_error
    
    return MonteCarloResult(
        estimate=estimate,
        standard_error=standard_error,
        n_samples=2 * n_pairs,
        confidence_interval=(ci_low, ci_high),
        method="antithetic",
    )


def stratified_sampling(
    f: Callable[[float], float],
    a: float,
    b: float,
    n_strata: int,
    samples_per_stratum: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """
    Stratified Monte Carlo integration.
    
    Divides [a, b] into n_strata equal intervals and samples
    uniformly within each stratum. Reduces variance by ensuring
    coverage across the domain.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        n_strata: Number of strata (equal intervals)
        samples_per_stratum: Samples per stratum
        rng: Random number generator
        
    Returns:
        MonteCarloResult with stratified estimate
    """
    if rng is None:
        rng = np.random.default_rng()
    
    stratum_width = (b - a) / n_strata
    stratum_means = np.zeros(n_strata)
    stratum_vars = np.zeros(n_strata)
    
    for k in range(n_strata):
        # Stratum bounds
        low = a + k * stratum_width
        high = a + (k + 1) * stratum_width
        
        # Sample within stratum
        X_k = rng.uniform(low, high, samples_per_stratum)
        
        try:
            f_values = f(X_k)
        except (TypeError, ValueError):
            f_values = np.array([f(x) for x in X_k])
        
        stratum_means[k] = np.mean(f_values)
        if samples_per_stratum > 1:
            stratum_vars[k] = np.var(f_values, ddof=1)
    
    # Overall estimate: average of stratum means × domain length
    domain_length = b - a
    estimate = domain_length * np.mean(stratum_means)
    
    # Variance of stratified estimator
    n_total = n_strata * samples_per_stratum
    variance_est = np.sum(stratum_vars) / (n_strata**2 * samples_per_stratum)
    standard_error = domain_length * np.sqrt(variance_est)
    
    z = 1.96
    ci_low = estimate - z * standard_error
    ci_high = estimate + z * standard_error
    
    return MonteCarloResult(
        estimate=estimate,
        standard_error=standard_error,
        n_samples=n_total,
        confidence_interval=(ci_low, ci_high),
        method="stratified",
    )


@dataclass
class ConvergenceStudy:
    """Results from convergence analysis."""
    
    n_values: list[int]
    mean_errors: list[float]
    empirical_order: float
    r_squared: float


def convergence_study(
    f: Callable[[float], float],
    a: float,
    b: float,
    true_value: float,
    n_values: list[int],
    n_trials: int = 10,
    rng: np.random.Generator | None = None,
) -> ConvergenceStudy:
    """
    Empirically verify O(1/√n) convergence.
    
    Fits log(error) vs log(n) to estimate convergence order.
    Theoretical order for MC is 0.5.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        true_value: Known integral value
        n_values: Sample sizes to test
        n_trials: Trials per sample size (for averaging)
        rng: Random number generator
        
    Returns:
        ConvergenceStudy with empirical order
    """
    if rng is None:
        rng = np.random.default_rng()
    
    mean_errors = []
    
    for n in n_values:
        errors = []
        for _ in range(n_trials):
            result = monte_carlo_integrate(f, a, b, n, rng)
            errors.append(abs(result.estimate - true_value))
        mean_errors.append(np.mean(errors))
    
    # Linear regression on log-log scale
    log_n = np.log(n_values)
    log_err = np.log(mean_errors)
    
    # slope gives -order (since error ∝ n^(-order))
    coeffs = np.polyfit(log_n, log_err, 1)
    empirical_order = -coeffs[0]
    
    # R² for goodness of fit
    predicted = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_err - predicted) ** 2)
    ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    return ConvergenceStudy(
        n_values=list(n_values),
        mean_errors=mean_errors,
        empirical_order=empirical_order,
        r_squared=r_squared,
    )


class MonteCarloIntegrator:
    """
    High-level interface for Monte Carlo integration.
    
    Supports multiple methods: standard, antithetic, stratified.
    
    Example:
        >>> mc = MonteCarloIntegrator(method='antithetic', seed=42)
        >>> result = mc.integrate(lambda x: x**2, 0, 1)
        >>> print(f"Estimate: {result.estimate:.6f}")
    """
    
    def __init__(
        self,
        method: str = "standard",
        n_samples: int = 10_000,
        seed: int | None = None,
    ) -> None:
        """
        Initialise integrator.
        
        Args:
            method: 'standard', 'antithetic', or 'stratified'
            n_samples: Default number of samples
            seed: Random seed for reproducibility
        """
        if method not in ("standard", "antithetic", "stratified"):
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
    
    def integrate(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n_samples: int | None = None,
    ) -> MonteCarloResult:
        """
        Integrate f over [a, b].
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            n_samples: Override default sample count
            
        Returns:
            MonteCarloResult with estimate and statistics
        """
        n = n_samples if n_samples is not None else self.n_samples
        
        if self.method == "standard":
            return monte_carlo_integrate(f, a, b, n, self.rng)
        elif self.method == "antithetic":
            return antithetic_variates(f, a, b, n // 2, self.rng)
        elif self.method == "stratified":
            n_strata = min(20, int(np.sqrt(n)))
            samples_per = n // n_strata
            return stratified_sampling(f, a, b, n_strata, samples_per, self.rng)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def monte_carlo_integrate_nd(
    f: Callable[[FloatArray], float],
    bounds: list[tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """
    Multidimensional Monte Carlo integration.
    
    Args:
        f: Function taking d-dimensional array, returning scalar
        bounds: List of (low, high) for each dimension
        n_samples: Number of samples
        rng: Random number generator
        
    Returns:
        MonteCarloResult with estimate
    """
    if rng is None:
        rng = np.random.default_rng()
    
    d = len(bounds)
    volume = np.prod([b - a for a, b in bounds])
    
    # Generate samples
    samples = np.zeros((n_samples, d))
    for i, (a, b) in enumerate(bounds):
        samples[:, i] = rng.uniform(a, b, n_samples)
    
    # Evaluate function
    f_values = np.array([f(samples[i]) for i in range(n_samples)])
    
    estimate = volume * np.mean(f_values)
    
    if n_samples > 1:
        variance = np.var(f_values, ddof=1)
        standard_error = volume * np.sqrt(variance / n_samples)
    else:
        standard_error = float('nan')
    
    z = 1.96
    ci_low = estimate - z * standard_error
    ci_high = estimate + z * standard_error
    
    return MonteCarloResult(
        estimate=estimate,
        standard_error=standard_error,
        n_samples=n_samples,
        confidence_interval=(ci_low, ci_high),
        method="nd-standard",
    )
