#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
05UNIT, Lab 1: Monte Carlo Methods
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Monte Carlo methods employ random sampling to solve deterministic problems,
particularly numerical integration. Unlike deterministic quadrature rules that
suffer exponentially in high dimensions, Monte Carlo convergence at O(1/√n)
is dimension-independent—making these methods indispensable for complex
integrals in physics, finance and machine learning.

PREREQUISITES
─────────────
- Week 4: Probabilistic data structures (randomness foundations)
- Python: NumPy array operations, basic statistics
- Mathematics: Integration, probability distributions

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement Monte Carlo integration with error estimation
2. Apply variance reduction techniques (antithetic, stratified)
3. Analyse convergence empirically and theoretically

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 60 minutes
- Total: 90 minutes

DEPENDENCIES
────────────
numpy>=1.24, scipy>=1.11, matplotlib>=3.7

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type aliases
FloatArray = NDArray[np.floating]
IntegrandFunc = Callable[[FloatArray], FloatArray]
T = TypeVar("T", bound=np.floating)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BASIC MONTE CARLO INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo integration.
    
    Attributes:
        estimate: The estimated integral value
        standard_error: Standard error of the estimate
        n_samples: Number of samples used
        confidence_interval: 95% confidence interval (low, high)
        samples: Raw function evaluations (optional)
    """
    estimate: float
    standard_error: float
    n_samples: int
    confidence_interval: tuple[float, float] = field(init=False)
    samples: FloatArray | None = None
    
    def __post_init__(self) -> None:
        """Compute 95% confidence interval."""
        margin = 1.96 * self.standard_error
        self.confidence_interval = (
            self.estimate - margin,
            self.estimate + margin,
        )


def monte_carlo_integrate(
    f: IntegrandFunc,
    a: float,
    b: float,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
    return_samples: bool = False,
) -> MonteCarloResult:
    """Estimate integral of f over [a, b] using Monte Carlo sampling.
    
    Uses the estimator: Î = (b-a) * mean(f(X_i)) where X_i ~ Uniform(a, b).
    
    Args:
        f: Function to integrate, must accept array input
        a: Lower bound of integration
        b: Upper bound of integration
        n_samples: Number of random samples
        rng: NumPy random generator for reproducibility
        return_samples: Whether to include raw samples in result
        
    Returns:
        MonteCarloResult with estimate, standard error and confidence interval
        
    Examples:
        >>> def quadratic(x): return x**2
        >>> result = monte_carlo_integrate(quadratic, 0, 1, n_samples=100_000)
        >>> abs(result.estimate - 1/3) < 0.01  # True with high probability
        True
        
    Notes:
        The standard error decreases as O(1/√n), so quadrupling samples
        approximately halves the error.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if a >= b:
        raise ValueError("Lower bound must be less than upper bound")
    
    # Generate uniform samples
    x = rng.uniform(a, b, size=n_samples)
    
    # Evaluate integrand
    fx = f(x)
    
    # Compute estimate and statistics
    width = b - a
    mean_fx = np.mean(fx)
    var_fx = np.var(fx, ddof=1)  # Sample variance
    
    estimate = width * mean_fx
    standard_error = width * np.sqrt(var_fx / n_samples)
    
    logger.debug(
        f"MC integration: n={n_samples}, estimate={estimate:.6f}, "
        f"SE={standard_error:.6f}"
    )
    
    return MonteCarloResult(
        estimate=float(estimate),
        standard_error=float(standard_error),
        n_samples=n_samples,
        samples=fx if return_samples else None,
    )


def monte_carlo_integrate_nd(
    f: Callable[[FloatArray], float],
    bounds: list[tuple[float, float]],
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Monte Carlo integration in multiple dimensions.
    
    Args:
        f: Function accepting (n_samples, n_dims) array, returning (n_samples,)
        bounds: List of (low, high) tuples for each dimension
        n_samples: Number of random samples
        rng: Random generator for reproducibility
        
    Returns:
        MonteCarloResult with estimate and error bounds
        
    Examples:
        >>> def sphere(x): return np.sum(x**2, axis=1) <= 1  # Indicator
        >>> bounds = [(-1, 1), (-1, 1), (-1, 1)]  # Unit cube
        >>> result = monte_carlo_integrate_nd(sphere, bounds, n_samples=100_000)
        >>> abs(result.estimate - 4*np.pi/3) < 0.1  # Volume of unit sphere
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_dims = len(bounds)
    
    # Compute volume of integration region
    volume = 1.0
    for low, high in bounds:
        volume *= (high - low)
    
    # Generate samples in unit hypercube, then scale
    samples = np.empty((n_samples, n_dims))
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = rng.uniform(low, high, size=n_samples)
    
    # Evaluate function
    fx = np.array([f(samples[j]) for j in range(n_samples)])
    
    # Compute statistics
    mean_fx = np.mean(fx)
    var_fx = np.var(fx, ddof=1)
    
    estimate = volume * mean_fx
    standard_error = volume * np.sqrt(var_fx / n_samples)
    
    return MonteCarloResult(
        estimate=float(estimate),
        standard_error=float(standard_error),
        n_samples=n_samples,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PI ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_pi(
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Estimate π using the hit-or-miss Monte Carlo method.
    
    Samples points uniformly in the unit square [0,1]² and counts
    the fraction landing inside the quarter unit circle. Since
    Area(quarter circle) / Area(square) = π/4, we have π ≈ 4 * (hits/total).
    
    Args:
        n_samples: Number of random points to generate
        rng: Random generator for reproducibility
        
    Returns:
        MonteCarloResult with π estimate and error bounds
        
    Examples:
        >>> result = estimate_pi(n_samples=1_000_000, rng=np.random.default_rng(42))
        >>> abs(result.estimate - np.pi) < 0.01
        True
        
    Notes:
        Convergence is O(1/√n). For 4 decimal places, need ~10⁸ samples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample in unit square
    x = rng.uniform(0, 1, size=n_samples)
    y = rng.uniform(0, 1, size=n_samples)
    
    # Check if inside quarter circle
    inside = (x**2 + y**2) <= 1.0
    hits = np.sum(inside)
    
    # Estimate π
    p_hat = hits / n_samples
    pi_estimate = 4.0 * p_hat
    
    # Binomial standard error: sqrt(p(1-p)/n)
    se_p = np.sqrt(p_hat * (1 - p_hat) / n_samples)
    standard_error = 4.0 * se_p
    
    logger.debug(
        f"π estimation: n={n_samples}, hits={hits}, "
        f"estimate={pi_estimate:.6f}, SE={standard_error:.6f}"
    )
    
    return MonteCarloResult(
        estimate=pi_estimate,
        standard_error=standard_error,
        n_samples=n_samples,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: VARIANCE REDUCTION TECHNIQUES
# ═══════════════════════════════════════════════════════════════════════════════


def antithetic_variates(
    f: IntegrandFunc,
    a: float,
    b: float,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Monte Carlo integration using antithetic variates.
    
    For each sample X, also evaluates f at the "mirror" point X' = a + b - X.
    When f is monotonic, f(X) and f(X') are negatively correlated, reducing
    variance of their average.
    
    Args:
        f: Integrand function (should be monotonic for best results)
        a: Lower integration bound
        b: Upper integration bound
        n_samples: Number of primary samples (total evaluations = 2*n_samples)
        rng: Random generator for reproducibility
        
    Returns:
        MonteCarloResult with reduced variance estimate
        
    Examples:
        >>> def monotonic(x): return np.exp(x)
        >>> result_std = monte_carlo_integrate(monotonic, 0, 1, n_samples=10_000)
        >>> result_anti = antithetic_variates(monotonic, 0, 1, n_samples=5_000)
        >>> # Both use same total evaluations, but antithetic has lower variance
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    # Generate primary samples
    x = rng.uniform(a, b, size=n_samples)
    
    # Compute antithetic samples
    x_anti = a + b - x
    
    # Evaluate at both
    fx = f(x)
    fx_anti = f(x_anti)
    
    # Average paired values
    paired_avg = (fx + fx_anti) / 2.0
    
    # Statistics
    width = b - a
    mean_paired = np.mean(paired_avg)
    var_paired = np.var(paired_avg, ddof=1)
    
    estimate = width * mean_paired
    standard_error = width * np.sqrt(var_paired / n_samples)
    
    logger.debug(
        f"Antithetic variates: n={n_samples}, estimate={estimate:.6f}, "
        f"SE={standard_error:.6f}"
    )
    
    return MonteCarloResult(
        estimate=float(estimate),
        standard_error=float(standard_error),
        n_samples=n_samples * 2,  # Total function evaluations
    )


def stratified_sampling(
    f: IntegrandFunc,
    a: float,
    b: float,
    n_strata: int = 10,
    samples_per_stratum: int = 1000,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Monte Carlo integration using stratified sampling.
    
    Divides [a, b] into equal strata and samples uniformly within each.
    Guarantees coverage across the domain, reducing variance when f varies.
    
    Args:
        f: Integrand function
        a: Lower integration bound
        b: Upper integration bound
        n_strata: Number of equal-width strata
        samples_per_stratum: Samples drawn from each stratum
        rng: Random generator for reproducibility
        
    Returns:
        MonteCarloResult with stratified estimate
        
    Examples:
        >>> def variable(x): return np.sin(10*x) + 1  # Highly variable
        >>> result = stratified_sampling(variable, 0, np.pi, n_strata=20)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    width = b - a
    stratum_width = width / n_strata
    
    stratum_means = []
    stratum_vars = []
    
    for k in range(n_strata):
        # Stratum bounds
        low = a + k * stratum_width
        high = low + stratum_width
        
        # Sample within stratum
        x = rng.uniform(low, high, size=samples_per_stratum)
        fx = f(x)
        
        stratum_means.append(np.mean(fx))
        stratum_vars.append(np.var(fx, ddof=1))
    
    # Combine strata (equal weights for equal strata)
    estimate = width * np.mean(stratum_means)
    
    # Variance of stratified estimator
    # Var(Î) = (width/K)² * Σ Var(f in stratum k) / n_k
    n_k = samples_per_stratum
    var_estimate = (stratum_width ** 2) * np.sum(stratum_vars) / n_k
    standard_error = np.sqrt(var_estimate)
    
    total_samples = n_strata * samples_per_stratum
    
    logger.debug(
        f"Stratified sampling: K={n_strata}, n/K={samples_per_stratum}, "
        f"estimate={estimate:.6f}, SE={standard_error:.6f}"
    )
    
    return MonteCarloResult(
        estimate=float(estimate),
        standard_error=float(standard_error),
        n_samples=total_samples,
    )


def importance_sampling(
    f: IntegrandFunc,
    g: Callable[[FloatArray], FloatArray],
    g_sample: Callable[[int, np.random.Generator], FloatArray],
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> MonteCarloResult:
    """Monte Carlo integration using importance sampling.
    
    Samples from proposal distribution g and reweights by f(x)/g(x).
    Most effective when g(x) ∝ |f(x)|.
    
    Args:
        f: Integrand function (target)
        g: Proposal density function (must be > 0 where f ≠ 0)
        g_sample: Function to draw n samples from g
        n_samples: Number of samples
        rng: Random generator
        
    Returns:
        MonteCarloResult with importance-sampled estimate
        
    Examples:
        >>> # Estimate E[X^2] where X ~ N(0,1) by sampling from t-distribution
        >>> def f(x): return x**2 * np.exp(-x**2/2) / np.sqrt(2*np.pi)
        >>> def g(x): return stats.t.pdf(x, df=3)  # Heavier tails
        >>> def g_sample(n, rng): return stats.t.rvs(df=3, size=n)
        >>> result = importance_sampling(f, g, g_sample, n_samples=10_000)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Sample from proposal
    x = g_sample(n_samples, rng)
    
    # Compute importance weights
    gx = g(x)
    
    # Avoid division by zero
    valid = gx > 1e-15
    if not np.all(valid):
        logger.warning(
            f"Dropping {np.sum(~valid)} samples with g(x) ≈ 0"
        )
        x = x[valid]
        gx = gx[valid]
    
    fx = f(x)
    weights = fx / gx
    
    # Estimate and variance
    estimate = np.mean(weights)
    var_weights = np.var(weights, ddof=1)
    standard_error = np.sqrt(var_weights / len(weights))
    
    return MonteCarloResult(
        estimate=float(estimate),
        standard_error=float(standard_error),
        n_samples=len(weights),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONVERGENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConvergenceStudy:
    """Results from Monte Carlo convergence analysis.
    
    Attributes:
        sample_sizes: Array of n values tested
        estimates: Estimate at each n
        errors: Absolute error at each n (if true value known)
        standard_errors: Standard error at each n
        true_value: Analytical value (if known)
        empirical_order: Fitted convergence order
    """
    sample_sizes: FloatArray
    estimates: FloatArray
    errors: FloatArray | None
    standard_errors: FloatArray
    true_value: float | None
    empirical_order: float | None = None


def convergence_study(
    f: IntegrandFunc,
    a: float,
    b: float,
    true_value: float | None = None,
    n_values: list[int] | None = None,
    n_trials: int = 10,
    rng: np.random.Generator | None = None,
) -> ConvergenceStudy:
    """Study Monte Carlo convergence rate empirically.
    
    Runs integration at multiple sample sizes and fits convergence order.
    
    Args:
        f: Integrand function
        a: Lower bound
        b: Upper bound
        true_value: Analytical integral value (if known)
        n_values: Sample sizes to test
        n_trials: Trials per sample size (for averaging)
        rng: Random generator
        
    Returns:
        ConvergenceStudy with estimates, errors and fitted order
        
    Examples:
        >>> def quadratic(x): return x**2
        >>> study = convergence_study(quadratic, 0, 1, true_value=1/3)
        >>> abs(study.empirical_order - 0.5) < 0.1  # Should be ~0.5
        True
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if n_values is None:
        n_values = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    estimates = []
    errors = [] if true_value is not None else None
    standard_errors = []
    
    for n in n_values:
        trial_estimates = []
        trial_ses = []
        
        for _ in range(n_trials):
            result = monte_carlo_integrate(f, a, b, n_samples=n, rng=rng)
            trial_estimates.append(result.estimate)
            trial_ses.append(result.standard_error)
        
        mean_estimate = np.mean(trial_estimates)
        mean_se = np.mean(trial_ses)
        
        estimates.append(mean_estimate)
        standard_errors.append(mean_se)
        
        if true_value is not None:
            errors.append(abs(mean_estimate - true_value))
    
    estimates = np.array(estimates)
    standard_errors = np.array(standard_errors)
    n_array = np.array(n_values, dtype=float)
    
    if errors is not None:
        errors = np.array(errors)
    
    # Fit convergence order: SE ~ n^(-α)
    # log(SE) = -α * log(n) + c
    log_n = np.log(n_array)
    log_se = np.log(standard_errors)
    
    # Linear regression
    slope, _ = np.polyfit(log_n, log_se, 1)
    empirical_order = -slope
    
    logger.info(f"Empirical convergence order: {empirical_order:.3f} (theory: 0.5)")
    
    return ConvergenceStudy(
        sample_sizes=n_array,
        estimates=estimates,
        errors=errors,
        standard_errors=standard_errors,
        true_value=true_value,
        empirical_order=empirical_order,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: HIGHER-LEVEL INTEGRATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════


class VarianceReductionMethod(Protocol):
    """Protocol for variance reduction methods."""
    
    def __call__(
        self,
        f: IntegrandFunc,
        a: float,
        b: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> MonteCarloResult: ...


class MonteCarloIntegrator:
    """High-level Monte Carlo integrator with method selection.
    
    Attributes:
        method: Variance reduction method ('standard', 'antithetic', 'stratified')
        n_samples: Default number of samples
        rng: Random generator
        
    Examples:
        >>> mc = MonteCarloIntegrator(method='antithetic', n_samples=50_000)
        >>> result = mc.integrate(lambda x: np.exp(-x**2), -5, 5)
    """
    
    def __init__(
        self,
        method: str = "standard",
        n_samples: int = 10_000,
        seed: int | None = None,
    ) -> None:
        """Initialise integrator.
        
        Args:
            method: 'standard', 'antithetic', or 'stratified'
            n_samples: Default sample count
            seed: Random seed for reproducibility
        """
        self.method = method
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        
        self._methods: dict[str, VarianceReductionMethod] = {
            "standard": self._standard,
            "antithetic": self._antithetic,
            "stratified": self._stratified,
        }
        
        if method not in self._methods:
            raise ValueError(f"Unknown method: {method}")
    
    def _standard(
        self,
        f: IntegrandFunc,
        a: float,
        b: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> MonteCarloResult:
        return monte_carlo_integrate(f, a, b, n_samples, rng)
    
    def _antithetic(
        self,
        f: IntegrandFunc,
        a: float,
        b: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> MonteCarloResult:
        return antithetic_variates(f, a, b, n_samples // 2, rng)
    
    def _stratified(
        self,
        f: IntegrandFunc,
        a: float,
        b: float,
        n_samples: int,
        rng: np.random.Generator,
    ) -> MonteCarloResult:
        n_strata = max(10, int(np.sqrt(n_samples)))
        samples_per = n_samples // n_strata
        return stratified_sampling(f, a, b, n_strata, samples_per, rng)
    
    def integrate(
        self,
        f: IntegrandFunc,
        a: float,
        b: float,
        n_samples: int | None = None,
    ) -> MonteCarloResult:
        """Integrate f over [a, b].
        
        Args:
            f: Integrand function
            a: Lower bound
            b: Upper bound
            n_samples: Override default sample count
            
        Returns:
            MonteCarloResult with estimate and error
        """
        n = n_samples or self.n_samples
        method_func = self._methods[self.method]
        return method_func(f, a, b, n, self.rng)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_basic_integration() -> None:
    """Demonstrate basic Monte Carlo integration."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Monte Carlo Integration")
    logger.info("=" * 60)
    
    # Integrate x² from 0 to 1 (true value = 1/3)
    def f(x: FloatArray) -> FloatArray:
        return x ** 2
    
    rng = np.random.default_rng(42)
    
    for n in [1000, 10_000, 100_000]:
        result = monte_carlo_integrate(f, 0, 1, n_samples=n, rng=rng)
        error = abs(result.estimate - 1/3)
        logger.info(
            f"n={n:>7}: estimate={result.estimate:.6f}, "
            f"SE={result.standard_error:.6f}, error={error:.6f}"
        )


def demo_pi_estimation() -> None:
    """Demonstrate π estimation convergence."""
    logger.info("=" * 60)
    logger.info("DEMO: π Estimation")
    logger.info("=" * 60)
    
    rng = np.random.default_rng(42)
    
    for n in [1000, 10_000, 100_000, 1_000_000]:
        result = estimate_pi(n_samples=n, rng=rng)
        error = abs(result.estimate - np.pi)
        logger.info(
            f"n={n:>8}: π̂={result.estimate:.6f}, "
            f"error={error:.6f}, 95% CI={result.confidence_interval}"
        )


def demo_variance_reduction() -> None:
    """Compare variance reduction techniques."""
    logger.info("=" * 60)
    logger.info("DEMO: Variance Reduction Comparison")
    logger.info("=" * 60)
    
    # Monotonic function (good for antithetic)
    def f(x: FloatArray) -> FloatArray:
        return np.exp(x)
    
    true_value = np.e - 1  # ∫₀¹ eˣ dx = e - 1
    n = 10_000
    rng = np.random.default_rng(42)
    
    # Run multiple trials
    n_trials = 20
    
    methods = {
        "Standard": lambda: monte_carlo_integrate(f, 0, 1, n, rng),
        "Antithetic": lambda: antithetic_variates(f, 0, 1, n // 2, rng),
        "Stratified": lambda: stratified_sampling(f, 0, 1, 10, n // 10, rng),
    }
    
    for name, method in methods.items():
        errors = []
        ses = []
        for _ in range(n_trials):
            result = method()
            errors.append(abs(result.estimate - true_value))
            ses.append(result.standard_error)
        
        logger.info(
            f"{name:>12}: mean error={np.mean(errors):.6f}, "
            f"mean SE={np.mean(ses):.6f}"
        )


def demo_convergence() -> None:
    """Demonstrate convergence analysis."""
    logger.info("=" * 60)
    logger.info("DEMO: Convergence Analysis")
    logger.info("=" * 60)
    
    def f(x: FloatArray) -> FloatArray:
        return np.sin(x)
    
    # True value: ∫₀^π sin(x) dx = 2
    study = convergence_study(
        f, 0, np.pi,
        true_value=2.0,
        n_values=[100, 500, 1000, 5000, 10000],
        n_trials=5,
        rng=np.random.default_rng(42),
    )
    
    logger.info(f"Empirical convergence order: {study.empirical_order:.3f}")
    logger.info("Sample sizes vs errors:")
    for n, err, se in zip(
        study.sample_sizes, study.errors, study.standard_errors
    ):
        logger.info(f"  n={int(n):>6}: error={err:.6f}, SE={se:.6f}")


def demo_multidimensional() -> None:
    """Demonstrate multidimensional integration."""
    logger.info("=" * 60)
    logger.info("DEMO: Multidimensional Integration")
    logger.info("=" * 60)
    
    # Volume of unit sphere in 3D (true value = 4π/3)
    def sphere_indicator(x: FloatArray) -> float:
        return float(np.sum(x**2) <= 1.0)
    
    bounds = [(-1, 1), (-1, 1), (-1, 1)]
    rng = np.random.default_rng(42)
    
    for n in [10_000, 100_000, 1_000_000]:
        result = monte_carlo_integrate_nd(
            sphere_indicator, bounds, n_samples=n, rng=rng
        )
        true_vol = 4 * np.pi / 3
        error = abs(result.estimate - true_vol)
        logger.info(
            f"n={n:>8}: volume={result.estimate:.4f}, "
            f"true={true_vol:.4f}, error={error:.4f}"
        )


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_basic_integration()
    print()
    demo_pi_estimation()
    print()
    demo_variance_reduction()
    print()
    demo_convergence()
    print()
    demo_multidimensional()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Integration Laboratory"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration examples",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--estimate-pi",
        type=int,
        metavar="N",
        help="Estimate π with N samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.estimate_pi:
        rng = np.random.default_rng(args.seed)
        result = estimate_pi(n_samples=args.estimate_pi, rng=rng)
        print(f"π estimate: {result.estimate:.10f}")
        print(f"Standard error: {result.standard_error:.10f}")
        print(f"95% CI: {result.confidence_interval}")
        return
    
    if args.demo:
        run_all_demos()
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
