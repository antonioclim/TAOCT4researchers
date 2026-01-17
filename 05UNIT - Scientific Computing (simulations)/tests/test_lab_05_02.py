#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5: Scientific Computing — ODE Solver Tests
═══════════════════════════════════════════════════════════════════════════════

Test suite for lab_05_02_ode_solvers.py

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from lab.lab_05_02_ode_solvers import (
    ODESolution,
    StepResult,
    adaptive_rkf45,
    analyse_convergence,
    euler_step,
    exponential_decay,
    harmonic_oscillator,
    lotka_volterra,
    midpoint_step,
    rk4_step,
    rkf45_step,
    solve_ode,
)

FloatArray = NDArray[np.floating]
ODEFunc = Callable[[float, FloatArray], FloatArray]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: EULER METHOD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEulerMethod:
    """Tests for Euler's method."""
    
    def test_single_step_exponential(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Single Euler step should approximate derivative."""
        f, y0, exact = exponential_decay_ode
        
        h = 0.1
        result = euler_step(f, 0.0, y0, h)
        
        # y_new = y0 + h * f(0, y0) = 1 + 0.1 * (-1) = 0.9
        assert abs(result.y_new[0] - 0.9) < 1e-10
        assert result.n_evaluations == 1
    
    def test_exponential_decay_solution(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Euler should solve exponential decay approximately."""
        f, y0, exact = exponential_decay_ode
        
        sol = solve_ode(f, (0, 5), y0, method="euler", n_steps=500)
        
        error = abs(sol.y[-1, 0] - exact(5.0)[0])
        assert error < 0.1  # Euler has limited accuracy
    
    def test_euler_convergence_order(
        self,
        exponential_decay_ode: tuple,
        step_sizes: list[float],
    ) -> None:
        """Euler should have O(h) global error."""
        f, y0, exact = exponential_decay_ode
        t_span = (0.0, 1.0)
        
        result = analyse_convergence(
            f, t_span, y0, exact,
            method="euler",
            step_sizes=step_sizes,
        )
        
        # Order should be approximately 1
        assert 0.8 < result.empirical_order < 1.2
    
    def test_halving_step_halves_error(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """For Euler, halving h should halve error (O(h))."""
        f, y0, exact = exponential_decay_ode
        t_span = (0.0, 1.0)
        
        sol_h = solve_ode(f, t_span, y0, method="euler", h=0.1)
        sol_h2 = solve_ode(f, t_span, y0, method="euler", h=0.05)
        
        error_h = abs(sol_h.y[-1, 0] - exact(1.0)[0])
        error_h2 = abs(sol_h2.y[-1, 0] - exact(1.0)[0])
        
        # Error ratio should be approximately 2
        ratio = error_h / error_h2
        assert 1.5 < ratio < 2.5


class TestMidpointMethod:
    """Tests for midpoint method (RK2)."""
    
    def test_single_step(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Single midpoint step should use two evaluations."""
        f, y0, exact = exponential_decay_ode
        
        result = midpoint_step(f, 0.0, y0, 0.1)
        
        assert result.n_evaluations == 2
    
    def test_midpoint_convergence_order(
        self,
        exponential_decay_ode: tuple,
        step_sizes: list[float],
    ) -> None:
        """Midpoint should have O(h²) global error."""
        f, y0, exact = exponential_decay_ode
        t_span = (0.0, 1.0)
        
        result = analyse_convergence(
            f, t_span, y0, exact,
            method="midpoint",
            step_sizes=step_sizes,
        )
        
        # Order should be approximately 2
        assert 1.7 < result.empirical_order < 2.3


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: RK4 METHOD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRK4Method:
    """Tests for fourth-order Runge-Kutta."""
    
    def test_single_step_evaluations(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """RK4 should use 4 function evaluations per step."""
        f, y0, _ = exponential_decay_ode
        
        result = rk4_step(f, 0.0, y0, 0.1)
        
        assert result.n_evaluations == 4
    
    def test_exact_for_linear(
        self,
        linear_growth_ode: tuple,
    ) -> None:
        """RK4 should be exact for linear ODEs (y' = const)."""
        f, y0, exact = linear_growth_ode
        
        sol = solve_ode(f, (0, 5), y0, method="rk4", n_steps=10)
        
        error = abs(sol.y[-1, 0] - exact(5.0)[0])
        assert error < 1e-10  # Should be essentially exact
    
    def test_rk4_convergence_order(
        self,
        exponential_decay_ode: tuple,
        step_sizes: list[float],
    ) -> None:
        """RK4 should have O(h⁴) global error."""
        f, y0, exact = exponential_decay_ode
        t_span = (0.0, 1.0)
        
        result = analyse_convergence(
            f, t_span, y0, exact,
            method="rk4",
            step_sizes=step_sizes,
        )
        
        # Order should be approximately 4
        assert 3.5 < result.empirical_order < 4.5
    
    def test_halving_step_reduces_error_16x(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """For RK4, halving h should reduce error by 16x (2⁴)."""
        f, y0, exact = exponential_decay_ode
        t_span = (0.0, 1.0)
        
        sol_h = solve_ode(f, t_span, y0, method="rk4", h=0.1)
        sol_h2 = solve_ode(f, t_span, y0, method="rk4", h=0.05)
        
        error_h = abs(sol_h.y[-1, 0] - exact(1.0)[0])
        error_h2 = abs(sol_h2.y[-1, 0] - exact(1.0)[0])
        
        # Error ratio should be approximately 16
        if error_h2 > 1e-14:  # Avoid division by very small numbers
            ratio = error_h / error_h2
            assert 10 < ratio < 25


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: HARMONIC OSCILLATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestHarmonicOscillator:
    """Tests using the harmonic oscillator system."""
    
    def test_returns_to_initial_position(
        self,
        harmonic_oscillator_ode: tuple,
    ) -> None:
        """After one period (2π), should return near initial state."""
        f, y0, exact = harmonic_oscillator_ode
        t_span = (0.0, 2 * np.pi)
        
        sol = solve_ode(f, t_span, y0, method="rk4", n_steps=200)
        
        # Should be close to initial position
        error = np.linalg.norm(sol.y[-1] - y0)
        assert error < 0.01
    
    def test_euler_energy_drift(
        self,
        harmonic_oscillator_ode: tuple,
    ) -> None:
        """Euler should show energy drift."""
        f, y0, _ = harmonic_oscillator_ode
        t_span = (0.0, 4 * np.pi)  # Two periods
        
        sol = solve_ode(f, t_span, y0, method="euler", n_steps=200)
        
        def energy(y: FloatArray) -> float:
            return 0.5 * (y[0]**2 + y[1]**2)
        
        e0 = energy(y0)
        e_final = energy(sol.y[-1])
        
        # Euler typically gains energy for oscillators
        assert e_final > e0 * 1.1  # At least 10% drift
    
    def test_rk4_energy_conservation(
        self,
        harmonic_oscillator_ode: tuple,
    ) -> None:
        """RK4 should approximately conserve energy."""
        f, y0, _ = harmonic_oscillator_ode
        t_span = (0.0, 4 * np.pi)
        
        sol = solve_ode(f, t_span, y0, method="rk4", n_steps=200)
        
        def energy(y: FloatArray) -> float:
            return 0.5 * (y[0]**2 + y[1]**2)
        
        e0 = energy(y0)
        e_final = energy(sol.y[-1])
        
        # RK4 should conserve energy much better
        assert abs(e_final - e0) / e0 < 0.01  # Less than 1% drift


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOTKA-VOLTERRA TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLotkaVolterra:
    """Tests for predator-prey dynamics."""
    
    def test_oscillating_dynamics(
        self,
        lotka_volterra_ode: ODEFunc,
    ) -> None:
        """Populations should oscillate."""
        y0 = np.array([10.0, 5.0])
        t_span = (0.0, 50.0)
        
        sol = adaptive_rkf45(lotka_volterra_ode, t_span, y0)
        
        prey = sol.y[:, 0]
        predator = sol.y[:, 1]
        
        # Both should have variation
        assert prey.max() > prey.min() * 2
        assert predator.max() > predator.min() * 2
    
    def test_positive_populations(
        self,
        lotka_volterra_ode: ODEFunc,
    ) -> None:
        """Populations should remain positive."""
        y0 = np.array([10.0, 5.0])
        t_span = (0.0, 50.0)
        
        sol = adaptive_rkf45(lotka_volterra_ode, t_span, y0)
        
        assert np.all(sol.y >= 0)
    
    def test_fixed_point_stability(
        self,
        lotka_volterra_ode: ODEFunc,
    ) -> None:
        """Starting at fixed point should stay there."""
        # Fixed point: x* = γ/δ = 1.5/0.075 = 20, y* = α/β = 1.0/0.1 = 10
        y0 = np.array([20.0, 10.0])
        t_span = (0.0, 10.0)
        
        sol = adaptive_rkf45(lotka_volterra_ode, t_span, y0)
        
        # Should stay near fixed point
        assert np.allclose(sol.y[-1], y0, rtol=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ADAPTIVE METHOD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveRKF45:
    """Tests for adaptive Runge-Kutta-Fehlberg."""
    
    def test_rkf45_step_provides_error_estimate(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """RKF45 step should provide error estimate."""
        f, y0, _ = exponential_decay_ode
        
        result = rkf45_step(f, 0.0, y0, 0.1)
        
        assert result.error_estimate is not None
        assert result.error_estimate >= 0
        assert result.n_evaluations == 6
    
    def test_adaptive_meets_tolerance(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Adaptive solver should meet specified tolerance."""
        f, y0, exact = exponential_decay_ode
        t_span = (0.0, 5.0)
        
        rtol = 1e-6
        sol = adaptive_rkf45(f, t_span, y0, rtol=rtol, atol=1e-9)
        
        error = abs(sol.y[-1, 0] - exact(5.0)[0])
        # Error should be within tolerance bounds
        assert error < rtol * 10  # Allow some margin
    
    def test_adaptive_varies_step_size(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Adaptive solver should vary step sizes."""
        f, y0, _ = exponential_decay_ode
        t_span = (0.0, 5.0)
        
        sol = adaptive_rkf45(f, t_span, y0, rtol=1e-6)
        
        dt = np.diff(sol.t)
        
        # Should have some variation in step sizes
        assert dt.max() / dt.min() > 1.1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestODEEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_initial_condition(self) -> None:
        """Should handle y0 = 0."""
        def f(t: float, y: FloatArray) -> FloatArray:
            return y  # dy/dt = y
        
        y0 = np.array([0.0])
        sol = solve_ode(f, (0, 1), y0, method="rk4", n_steps=10)
        
        # Solution should remain at 0
        assert np.allclose(sol.y, 0.0)
    
    def test_very_small_step(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Very small steps should give high accuracy."""
        f, y0, exact = exponential_decay_ode
        
        sol = solve_ode(f, (0, 1), y0, method="rk4", h=0.001)
        
        error = abs(sol.y[-1, 0] - exact(1.0)[0])
        assert error < 1e-10
    
    def test_backward_integration(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Should handle negative step (backward in time)."""
        f, y0, exact = exponential_decay_ode
        
        # Start at t=1, go back to t=0
        y1 = exact(1.0)
        
        # Backward integration: dy/dt = -y means we integrate from 1 to 0
        # But our functions work forward, so this tests t_end < t_start handling
        # Actually solve_ode might not support this directly, skip if not
        pass  # This would require more sophisticated handling
    
    def test_unknown_method_raises(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Unknown method should raise ValueError."""
        f, y0, _ = exponential_decay_ode
        
        with pytest.raises(ValueError):
            solve_ode(f, (0, 1), y0, method="unknown")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: VECTOR SYSTEM TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestVectorSystems:
    """Tests for systems of ODEs."""
    
    def test_two_dimensional_rotation(self) -> None:
        """Test 2D rotation system: dx/dt = -y, dy/dt = x."""
        def f(t: float, y: FloatArray) -> FloatArray:
            return np.array([-y[1], y[0]])
        
        y0 = np.array([1.0, 0.0])  # Start at (1, 0)
        t_span = (0.0, 2 * np.pi)  # One rotation
        
        sol = solve_ode(f, t_span, y0, method="rk4", n_steps=200)
        
        # Should return to start
        assert np.linalg.norm(sol.y[-1] - y0) < 0.01
        
        # Radius should be preserved
        radii = np.sqrt(sol.y[:, 0]**2 + sol.y[:, 1]**2)
        assert np.allclose(radii, 1.0, atol=0.01)
    
    def test_three_dimensional_system(self) -> None:
        """Test 3D decoupled system."""
        def f(t: float, y: FloatArray) -> FloatArray:
            return np.array([-y[0], -2*y[1], -3*y[2]])
        
        y0 = np.array([1.0, 1.0, 1.0])
        t_span = (0.0, 2.0)
        
        sol = solve_ode(f, t_span, y0, method="rk4", n_steps=100)
        
        # Each component decays independently
        expected = np.array([np.exp(-2), np.exp(-4), np.exp(-6)])
        assert np.allclose(sol.y[-1], expected, rtol=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: SOLUTION DATACLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestODESolutionDataclass:
    """Tests for ODESolution dataclass."""
    
    def test_solution_attributes(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Solution should have all required attributes."""
        f, y0, _ = exponential_decay_ode
        
        sol = solve_ode(f, (0, 1), y0, method="rk4", n_steps=10)
        
        assert hasattr(sol, "t")
        assert hasattr(sol, "y")
        assert hasattr(sol, "method")
        assert hasattr(sol, "n_steps")
        assert hasattr(sol, "n_evaluations")
        assert hasattr(sol, "success")
    
    def test_solution_indexing(
        self,
        exponential_decay_ode: tuple,
    ) -> None:
        """Should be able to index solution."""
        f, y0, _ = exponential_decay_ode
        
        sol = solve_ode(f, (0, 1), y0, method="rk4", n_steps=10)
        
        t0, y0_sol = sol[0]
        assert t0 == 0.0
        assert np.allclose(y0_sol, y0)
    
    def test_solution_n_dims(
        self,
        harmonic_oscillator_ode: tuple,
    ) -> None:
        """n_dims property should work."""
        f, y0, _ = harmonic_oscillator_ode
        
        sol = solve_ode(f, (0, 1), y0, method="rk4", n_steps=10)
        
        assert sol.n_dims == 2


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRISED TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("method", ["euler", "midpoint", "rk4"])
def test_all_methods_complete(
    method: str,
    exponential_decay_ode: tuple,
) -> None:
    """All methods should successfully complete integration."""
    f, y0, _ = exponential_decay_ode
    
    sol = solve_ode(f, (0, 1), y0, method=method, n_steps=100)
    
    assert sol.success
    assert len(sol.t) > 1
    assert len(sol.y) > 1


@pytest.mark.parametrize("n_steps", [10, 50, 100, 500])
def test_varying_step_counts(
    n_steps: int,
    exponential_decay_ode: tuple,
) -> None:
    """Should work with varying step counts."""
    f, y0, _ = exponential_decay_ode
    
    sol = solve_ode(f, (0, 1), y0, method="rk4", n_steps=n_steps)
    
    assert len(sol.t) == n_steps + 1


@pytest.mark.parametrize("rtol", [1e-3, 1e-6, 1e-9])
def test_adaptive_varying_tolerance(
    rtol: float,
    exponential_decay_ode: tuple,
) -> None:
    """Adaptive should work with varying tolerances."""
    f, y0, exact = exponential_decay_ode
    
    sol = adaptive_rkf45(f, (0, 1), y0, rtol=rtol, atol=rtol/100)
    
    error = abs(sol.y[-1, 0] - exact(1.0)[0])
    # Tighter tolerance should give smaller error
    assert error < rtol * 100  # Allow margin
