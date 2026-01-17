#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5: Scientific Computing — Monte Carlo Tests
═══════════════════════════════════════════════════════════════════════════════

Test suite for lab_5_01_monte_carlo.py

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
import pytest

from lab.lab_5_01_monte_carlo import (
    MonteCarloIntegrator,
    MonteCarloResult,
    antithetic_variates,
    convergence_study,
    estimate_pi,
    monte_carlo_integrate,
    monte_carlo_integrate_nd,
    stratified_sampling,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BASIC MONTE CARLO INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBasicMonteCarloIntegration:
    """Tests for basic Monte Carlo integration."""
    
    def test_constant_function_exact(
        self,
        constant_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Constant function should give exact result (zero variance)."""
        f, a, b, exact = constant_function
        result = monte_carlo_integrate(f, a, b, n_samples=1000, rng=rng)
        
        assert abs(result.estimate - exact) < 1e-10
        assert result.standard_error < 1e-10
    
    def test_linear_function_convergence(
        self,
        linear_function: tuple,
        rng: np.random.Generator,
        loose_tolerance: float,
    ) -> None:
        """Linear function should converge to 0.5."""
        f, a, b, exact = linear_function
        result = monte_carlo_integrate(f, a, b, n_samples=100_000, rng=rng)
        
        assert abs(result.estimate - exact) < loose_tolerance
    
    def test_quadratic_function_convergence(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
        loose_tolerance: float,
    ) -> None:
        """Quadratic function should converge to 1/3."""
        f, a, b, exact = quadratic_function
        result = monte_carlo_integrate(f, a, b, n_samples=100_000, rng=rng)
        
        assert abs(result.estimate - exact) < loose_tolerance
    
    def test_exponential_function(
        self,
        exponential_function: tuple,
        rng: np.random.Generator,
        loose_tolerance: float,
    ) -> None:
        """Exponential function should converge to e-1."""
        f, a, b, exact = exponential_function
        result = monte_carlo_integrate(f, a, b, n_samples=100_000, rng=rng)
        
        assert abs(result.estimate - exact) < loose_tolerance
    
    def test_result_has_correct_attributes(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Result should have all required attributes."""
        f, a, b, _ = quadratic_function
        result = monte_carlo_integrate(f, a, b, n_samples=1000, rng=rng)
        
        assert hasattr(result, "estimate")
        assert hasattr(result, "standard_error")
        assert hasattr(result, "n_samples")
        assert hasattr(result, "confidence_interval")
        
        assert result.n_samples == 1000
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] < result.estimate
        assert result.confidence_interval[1] > result.estimate


class TestMonteCarloConvergenceRate:
    """Tests for O(1/√n) convergence."""
    
    def test_standard_error_decreases_with_n(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Standard error should decrease as n increases."""
        f, a, b, _ = quadratic_function
        
        se_1000 = monte_carlo_integrate(f, a, b, 1000, rng).standard_error
        se_10000 = monte_carlo_integrate(f, a, b, 10000, rng).standard_error
        
        assert se_10000 < se_1000
    
    def test_quadrupling_samples_halves_error(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Quadrupling n should roughly halve SE (O(1/√n))."""
        f, a, b, _ = quadratic_function
        
        # Average over trials for stability
        se_n = []
        se_4n = []
        
        for _ in range(10):
            se_n.append(monte_carlo_integrate(f, a, b, 1000, rng).standard_error)
            se_4n.append(monte_carlo_integrate(f, a, b, 4000, rng).standard_error)
        
        ratio = np.mean(se_n) / np.mean(se_4n)
        # Should be close to 2 (√4 = 2)
        assert 1.5 < ratio < 2.5
    
    def test_empirical_convergence_order(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Empirically verify O(1/√n) convergence."""
        f, a, b, exact = quadratic_function
        
        study = convergence_study(
            f, a, b,
            true_value=exact,
            n_values=[100, 500, 1000, 5000],
            n_trials=5,
            rng=rng,
        )
        
        # Convergence order should be approximately 0.5
        assert 0.3 < study.empirical_order < 0.7


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PI ESTIMATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPiEstimation:
    """Tests for π estimation."""
    
    def test_pi_estimate_reasonable(self, rng: np.random.Generator) -> None:
        """π estimate should be in reasonable range."""
        result = estimate_pi(n_samples=10_000, rng=rng)
        
        assert 2.5 < result.estimate < 4.0
    
    def test_pi_estimate_converges(self, rng: np.random.Generator) -> None:
        """With many samples, should be close to π."""
        result = estimate_pi(n_samples=1_000_000, rng=rng)
        
        assert abs(result.estimate - np.pi) < 0.01
    
    def test_pi_reproducible_with_seed(self) -> None:
        """Same seed should give same result."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        result1 = estimate_pi(n_samples=10_000, rng=rng1)
        result2 = estimate_pi(n_samples=10_000, rng=rng2)
        
        assert result1.estimate == result2.estimate
    
    def test_confidence_interval_contains_pi(
        self,
        rng: np.random.Generator,
    ) -> None:
        """95% CI should usually contain true π."""
        # Run multiple trials
        contains_pi = 0
        n_trials = 20
        
        for _ in range(n_trials):
            result = estimate_pi(n_samples=50_000, rng=rng)
            if result.confidence_interval[0] <= np.pi <= result.confidence_interval[1]:
                contains_pi += 1
        
        # Should contain π in ~95% of trials (allow some margin)
        assert contains_pi >= n_trials * 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: VARIANCE REDUCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAntitheticVariates:
    """Tests for antithetic variates variance reduction."""
    
    def test_same_expected_value(
        self,
        exponential_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Antithetic should give same expected value."""
        f, a, b, exact = exponential_function
        
        # Run many trials to estimate expected value
        estimates_std = []
        estimates_anti = []
        
        for _ in range(50):
            estimates_std.append(
                monte_carlo_integrate(f, a, b, 5000, rng).estimate
            )
            estimates_anti.append(
                antithetic_variates(f, a, b, 2500, rng).estimate
            )
        
        # Means should be similar
        assert abs(np.mean(estimates_std) - np.mean(estimates_anti)) < 0.05
    
    def test_variance_reduction_for_monotonic(
        self,
        exponential_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Antithetic should reduce variance for monotonic functions."""
        f, a, b, _ = exponential_function
        
        # Collect standard errors over trials
        se_std = []
        se_anti = []
        
        for _ in range(20):
            se_std.append(
                monte_carlo_integrate(f, a, b, 10000, rng).standard_error
            )
            se_anti.append(
                antithetic_variates(f, a, b, 5000, rng).standard_error
            )
        
        # Antithetic should have lower average SE
        assert np.mean(se_anti) < np.mean(se_std)


class TestStratifiedSampling:
    """Tests for stratified sampling variance reduction."""
    
    def test_same_expected_value(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Stratified should give same expected value."""
        f, a, b, exact = quadratic_function
        
        estimates = []
        for _ in range(30):
            result = stratified_sampling(f, a, b, n_strata=10, samples_per_stratum=500, rng=rng)
            estimates.append(result.estimate)
        
        assert abs(np.mean(estimates) - exact) < 0.02
    
    def test_variance_reduction(
        self,
        sine_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Stratified should reduce variance for variable functions."""
        f, a, b, _ = sine_function
        
        # Collect estimates
        estimates_std = []
        estimates_strat = []
        
        for _ in range(30):
            estimates_std.append(
                monte_carlo_integrate(f, a, b, 5000, rng).estimate
            )
            estimates_strat.append(
                stratified_sampling(f, a, b, 10, 500, rng).estimate
            )
        
        # Stratified should have lower variance
        var_std = np.var(estimates_std)
        var_strat = np.var(estimates_strat)
        
        assert var_strat < var_std * 1.5  # Allow some margin


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_samples_raises(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Zero samples should raise ValueError."""
        f, a, b, _ = quadratic_function
        
        with pytest.raises(ValueError):
            monte_carlo_integrate(f, a, b, n_samples=0, rng=rng)
    
    def test_negative_samples_raises(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Negative samples should raise ValueError."""
        f, a, b, _ = quadratic_function
        
        with pytest.raises(ValueError):
            monte_carlo_integrate(f, a, b, n_samples=-100, rng=rng)
    
    def test_inverted_bounds_raises(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """a >= b should raise ValueError."""
        f, _, _, _ = quadratic_function
        
        with pytest.raises(ValueError):
            monte_carlo_integrate(f, 1.0, 0.0, n_samples=100, rng=rng)
    
    def test_single_sample(
        self,
        linear_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Single sample should work (no variance estimate)."""
        f, a, b, _ = linear_function
        
        # Should not raise
        result = monte_carlo_integrate(f, a, b, n_samples=1, rng=rng)
        assert result.n_samples == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MULTIDIMENSIONAL INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMultidimensionalIntegration:
    """Tests for multidimensional Monte Carlo."""
    
    def test_unit_square_area(self, rng: np.random.Generator) -> None:
        """Integral of 1 over unit square should be 1."""
        def f(x):
            return 1.0
        
        bounds = [(0, 1), (0, 1)]
        result = monte_carlo_integrate_nd(f, bounds, n_samples=10_000, rng=rng)
        
        assert abs(result.estimate - 1.0) < 0.05
    
    def test_circle_area(self, rng: np.random.Generator) -> None:
        """Area of unit circle should be π."""
        def indicator(x):
            return float(x[0]**2 + x[1]**2 <= 1)
        
        bounds = [(-1, 1), (-1, 1)]
        result = monte_carlo_integrate_nd(indicator, bounds, n_samples=100_000, rng=rng)
        
        assert abs(result.estimate - np.pi) < 0.1
    
    def test_sphere_volume(self, rng: np.random.Generator) -> None:
        """Volume of unit sphere should be 4π/3."""
        def indicator(x):
            return float(np.sum(x**2) <= 1)
        
        bounds = [(-1, 1), (-1, 1), (-1, 1)]
        result = monte_carlo_integrate_nd(indicator, bounds, n_samples=500_000, rng=rng)
        
        true_vol = 4 * np.pi / 3
        assert abs(result.estimate - true_vol) < 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: INTEGRATOR CLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMonteCarloIntegrator:
    """Tests for the MonteCarloIntegrator class."""
    
    def test_standard_method(
        self,
        quadratic_function: tuple,
    ) -> None:
        """Standard method should work."""
        f, a, b, exact = quadratic_function
        
        mc = MonteCarloIntegrator(method="standard", n_samples=50_000, seed=42)
        result = mc.integrate(f, a, b)
        
        assert abs(result.estimate - exact) < 0.05
    
    def test_antithetic_method(
        self,
        exponential_function: tuple,
    ) -> None:
        """Antithetic method should work."""
        f, a, b, exact = exponential_function
        
        mc = MonteCarloIntegrator(method="antithetic", n_samples=50_000, seed=42)
        result = mc.integrate(f, a, b)
        
        assert abs(result.estimate - exact) < 0.05
    
    def test_stratified_method(
        self,
        sine_function: tuple,
    ) -> None:
        """Stratified method should work."""
        f, a, b, exact = sine_function
        
        mc = MonteCarloIntegrator(method="stratified", n_samples=50_000, seed=42)
        result = mc.integrate(f, a, b)
        
        assert abs(result.estimate - exact) < 0.1
    
    def test_unknown_method_raises(self) -> None:
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError):
            MonteCarloIntegrator(method="invalid")
    
    def test_override_n_samples(
        self,
        quadratic_function: tuple,
    ) -> None:
        """Should be able to override n_samples in integrate()."""
        f, a, b, _ = quadratic_function
        
        mc = MonteCarloIntegrator(n_samples=1000, seed=42)
        result = mc.integrate(f, a, b, n_samples=5000)
        
        # Result should reflect overridden n_samples
        assert result.n_samples == 5000


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: STATISTICAL PROPERTIES TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestStatisticalProperties:
    """Tests for statistical properties of MC estimator."""
    
    def test_unbiasedness(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """MC estimator should be unbiased (E[Î] = I)."""
        f, a, b, exact = quadratic_function
        
        estimates = []
        for _ in range(100):
            result = monte_carlo_integrate(f, a, b, 1000, rng)
            estimates.append(result.estimate)
        
        # Mean of estimates should be close to true value
        assert abs(np.mean(estimates) - exact) < 0.02
    
    def test_consistency(
        self,
        quadratic_function: tuple,
        rng: np.random.Generator,
    ) -> None:
        """Estimates should converge to true value as n → ∞."""
        f, a, b, exact = quadratic_function
        
        errors = []
        for n in [100, 1000, 10000, 100000]:
            result = monte_carlo_integrate(f, a, b, n, rng)
            errors.append(abs(result.estimate - exact))
        
        # Errors should generally decrease
        assert errors[-1] < errors[0]


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRISED TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_varying_sample_sizes(
    n_samples: int,
    quadratic_function: tuple,
    rng: np.random.Generator,
) -> None:
    """Test integration with varying sample sizes."""
    f, a, b, exact = quadratic_function
    
    result = monte_carlo_integrate(f, a, b, n_samples, rng)
    
    # Should produce some estimate
    assert result.estimate is not None
    assert result.n_samples == n_samples


@pytest.mark.parametrize("method", ["standard", "antithetic", "stratified"])
def test_all_methods_converge(
    method: str,
    quadratic_function: tuple,
) -> None:
    """All methods should converge to correct value."""
    f, a, b, exact = quadratic_function
    
    mc = MonteCarloIntegrator(method=method, n_samples=50_000, seed=42)
    result = mc.integrate(f, a, b)
    
    assert abs(result.estimate - exact) < 0.1
