#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5: Scientific Computing — Test Configuration
═══════════════════════════════════════════════════════════════════════════════

Shared pytest fixtures for Week 5 laboratory tests.

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

# Type aliases
FloatArray = NDArray[np.floating]
IntegrandFunc = Callable[[FloatArray], FloatArray]
ODEFunc = Callable[[float, FloatArray], FloatArray]


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def rng() -> np.random.Generator:
    """Provide seeded random generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def rng_varied() -> np.random.Generator:
    """Alternative seed for testing variance."""
    return np.random.default_rng(12345)


# ═══════════════════════════════════════════════════════════════════════════════
# TOLERANCE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def loose_tolerance() -> float:
    """Loose tolerance for Monte Carlo tests."""
    return 0.1


@pytest.fixture
def medium_tolerance() -> float:
    """Medium tolerance for ODE tests."""
    return 0.01


@pytest.fixture
def tight_tolerance() -> float:
    """Tight tolerance for exact comparisons."""
    return 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO TEST FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def constant_function() -> tuple[IntegrandFunc, float, float, float]:
    """Constant function f(x) = 5 with known integral.
    
    Returns:
        (function, a, b, exact_integral)
    """
    def f(x: FloatArray) -> FloatArray:
        return np.full_like(x, 5.0)
    return f, 0.0, 1.0, 5.0


@pytest.fixture
def linear_function() -> tuple[IntegrandFunc, float, float, float]:
    """Linear function f(x) = x with known integral.
    
    ∫₀¹ x dx = 0.5
    
    Returns:
        (function, a, b, exact_integral)
    """
    def f(x: FloatArray) -> FloatArray:
        return x
    return f, 0.0, 1.0, 0.5


@pytest.fixture
def quadratic_function() -> tuple[IntegrandFunc, float, float, float]:
    """Quadratic function f(x) = x² with known integral.
    
    ∫₀¹ x² dx = 1/3
    
    Returns:
        (function, a, b, exact_integral)
    """
    def f(x: FloatArray) -> FloatArray:
        return x ** 2
    return f, 0.0, 1.0, 1.0 / 3.0


@pytest.fixture
def exponential_function() -> tuple[IntegrandFunc, float, float, float]:
    """Exponential function f(x) = eˣ with known integral.
    
    ∫₀¹ eˣ dx = e - 1
    
    Returns:
        (function, a, b, exact_integral)
    """
    def f(x: FloatArray) -> FloatArray:
        return np.exp(x)
    return f, 0.0, 1.0, np.e - 1


@pytest.fixture
def sine_function() -> tuple[IntegrandFunc, float, float, float]:
    """Sine function with known integral.
    
    ∫₀^π sin(x) dx = 2
    
    Returns:
        (function, a, b, exact_integral)
    """
    def f(x: FloatArray) -> FloatArray:
        return np.sin(x)
    return f, 0.0, np.pi, 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# ODE TEST SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def exponential_decay_ode() -> tuple[ODEFunc, FloatArray, Callable[[float], FloatArray]]:
    """Exponential decay: dy/dt = -y, y(0) = 1.
    
    Exact solution: y(t) = e^(-t)
    
    Returns:
        (ode_function, y0, exact_solution)
    """
    def f(t: float, y: FloatArray) -> FloatArray:
        return -y
    
    y0 = np.array([1.0])
    
    def exact(t: float) -> FloatArray:
        return np.array([np.exp(-t)])
    
    return f, y0, exact


@pytest.fixture
def linear_growth_ode() -> tuple[ODEFunc, FloatArray, Callable[[float], FloatArray]]:
    """Linear growth: dy/dt = 2, y(0) = 1.
    
    Exact solution: y(t) = 2t + 1
    
    Returns:
        (ode_function, y0, exact_solution)
    """
    def f(t: float, y: FloatArray) -> FloatArray:
        return np.array([2.0])
    
    y0 = np.array([1.0])
    
    def exact(t: float) -> FloatArray:
        return np.array([2.0 * t + 1.0])
    
    return f, y0, exact


@pytest.fixture
def harmonic_oscillator_ode() -> tuple[ODEFunc, FloatArray, Callable[[float], FloatArray]]:
    """Harmonic oscillator: y'' + y = 0, y(0) = 1, y'(0) = 0.
    
    Exact solution: y(t) = cos(t), y'(t) = -sin(t)
    
    Returns:
        (ode_function, y0, exact_solution)
    """
    def f(t: float, y: FloatArray) -> FloatArray:
        return np.array([y[1], -y[0]])
    
    y0 = np.array([1.0, 0.0])
    
    def exact(t: float) -> FloatArray:
        return np.array([np.cos(t), -np.sin(t)])
    
    return f, y0, exact


@pytest.fixture
def lotka_volterra_ode() -> ODEFunc:
    """Lotka-Volterra predator-prey system.
    
    dx/dt = αx - βxy
    dy/dt = δxy - γy
    
    Returns:
        ODE function with standard parameters
    """
    alpha, beta, gamma, delta = 1.0, 0.1, 1.5, 0.075
    
    def f(t: float, y: FloatArray) -> FloatArray:
        x, y_pred = y
        return np.array([
            alpha * x - beta * x * y_pred,
            delta * x * y_pred - gamma * y_pred,
        ])
    
    return f


# ═══════════════════════════════════════════════════════════════════════════════
# ABM TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_schelling_model():
    """Create small Schelling model for testing."""
    from lab.lab_05_03_agent_based_modelling import SchellingModel
    return SchellingModel(
        grid_size=10,
        empty_ratio=0.2,
        threshold=0.3,
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def small_boids_simulation():
    """Create small boids simulation for testing."""
    from lab.lab_05_03_agent_based_modelling import BoidsSimulation
    return BoidsSimulation(
        width=400,
        height=300,
        n_boids=20,
        rng=np.random.default_rng(42),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_sizes() -> list[int]:
    """Standard sample sizes for convergence tests."""
    return [100, 500, 1000, 5000, 10000]


@pytest.fixture
def step_sizes() -> list[float]:
    """Standard step sizes for ODE convergence tests."""
    return [0.1, 0.05, 0.025, 0.0125, 0.00625]
