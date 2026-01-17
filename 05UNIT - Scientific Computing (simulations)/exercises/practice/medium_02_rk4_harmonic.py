#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Medium 02 — Fourth-Order Runge-Kutta
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 40 minutes
TOPIC: ODE solvers (RK4)

OBJECTIVE
─────────
Implement the classical fourth-order Runge-Kutta method for solving ODEs.

BACKGROUND
──────────
RK4 achieves O(h⁴) accuracy by evaluating the derivative at four points:
  k₁ = f(tₙ, yₙ)
  k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
  k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
  k₄ = f(tₙ + h, yₙ + h·k₃)
  yₙ₊₁ = yₙ + h·(k₁ + 2k₂ + 2k₃ + k₄)/6

TASKS
─────
1. Complete the `rk4_step` function
2. Solve the harmonic oscillator y'' + y = 0
3. Verify fourth-order convergence

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Type alias for ODE function
ODEFunc = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]


def rk4_step(
    f: ODEFunc,
    t: float,
    y: NDArray[np.floating],
    h: float,
) -> NDArray[np.floating]:
    """Perform single RK4 step.
    
    Args:
        f: Derivative function f(t, y)
        t: Current time
        y: Current state vector
        h: Step size
        
    Returns:
        Next state vector y_{n+1}
        
    Example:
        >>> def decay(t, y): return -y
        >>> y0 = np.array([1.0])
        >>> y1 = rk4_step(decay, 0.0, y0, 0.1)
        >>> abs(y1[0] - np.exp(-0.1)) < 1e-6
        True
    """
    # TODO: Compute k1
    # k1 = f(t, y)
    
    # TODO: Compute k2
    # k2 = f(t + h/2, y + h*k1/2)
    
    # TODO: Compute k3
    # k3 = f(t + h/2, y + h*k2/2)
    
    # TODO: Compute k4
    # k4 = f(t + h, y + h*k3)
    
    # TODO: Combine to get y_{n+1}
    # y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    raise NotImplementedError("Complete this function")


def solve_ode(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: NDArray[np.floating],
    h: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve ODE using RK4.
    
    Args:
        f: Derivative function f(t, y)
        t_span: (t_start, t_end)
        y0: Initial condition
        h: Step size
        
    Returns:
        Tuple of (t_values, y_values) where y_values has shape (n_steps+1, dim)
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / h)
    
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0
    
    for i in range(n_steps):
        y[i + 1] = rk4_step(f, t[i], y[i], h)
    
    return t, y


# ═══════════════════════════════════════════════════════════════════════════════
# HARMONIC OSCILLATOR
# ═══════════════════════════════════════════════════════════════════════════════

def harmonic_oscillator(
    t: float,
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Harmonic oscillator: y'' + y = 0.
    
    Written as system: y' = v, v' = -y
    State vector: [y, v]
    """
    position, velocity = y
    return np.array([velocity, -position])


def exact_harmonic(
    t: NDArray[np.floating],
    y0: float,
    v0: float,
) -> NDArray[np.floating]:
    """Exact solution: y(t) = y0·cos(t) + v0·sin(t)."""
    return y0 * np.cos(t) + v0 * np.sin(t)


def compute_energy(y: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute total energy E = (y² + v²)/2."""
    return 0.5 * (y[:, 0] ** 2 + y[:, 1] ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def convergence_study(
    step_sizes: list[float],
) -> dict[str, list[float]]:
    """Study RK4 convergence order.
    
    Args:
        step_sizes: List of h values to test
        
    Returns:
        Dictionary with 'h', 'error', 'order' lists
    """
    y0 = np.array([1.0, 0.0])  # Initial: y=1, v=0
    t_end = 2 * np.pi  # One full period
    
    results = {"h": [], "error": [], "order": []}
    prev_error = None
    
    for h in step_sizes:
        t, y = solve_ode(harmonic_oscillator, (0, t_end), y0, h)
        y_exact = exact_harmonic(t, y0[0], y0[1])
        
        error = float(np.max(np.abs(y[:, 0] - y_exact)))
        results["h"].append(h)
        results["error"].append(error)
        
        if prev_error is not None:
            order = np.log(prev_error / error) / np.log(2)
            results["order"].append(order)
        else:
            results["order"].append(float("nan"))
        
        prev_error = error
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("RK4 Implementation: Harmonic Oscillator")
    print("=" * 60)
    
    # Test single step
    print("\n1. Single Step Test (exponential decay):")
    try:
        def decay(t, y):
            return -y
        
        y0 = np.array([1.0])
        y1 = rk4_step(decay, 0.0, y0, 0.1)
        exact = np.exp(-0.1)
        error = abs(y1[0] - exact)
        print(f"   y(0.1) computed: {y1[0]:.10f}")
        print(f"   y(0.1) exact:    {exact:.10f}")
        print(f"   Error:           {error:.2e}")
    except NotImplementedError:
        print("   Complete the rk4_step function first!")
    
    # Convergence study
    print("\n2. Convergence Study (harmonic oscillator):")
    print(f"{'h':>10} {'Max Error':>15} {'Order':>10}")
    print("-" * 40)
    
    try:
        results = convergence_study([0.2, 0.1, 0.05, 0.025, 0.0125])
        
        for h, err, order in zip(results["h"], results["error"], results["order"]):
            print(f"{h:>10.4f} {err:>15.2e} {order:>10.2f}")
        
        print("-" * 40)
        print("Note: Order ≈ 4 confirms O(h⁴) convergence")
        
        # Energy conservation
        print("\n3. Energy Conservation:")
        t, y = solve_ode(harmonic_oscillator, (0, 4*np.pi), np.array([1.0, 0.0]), 0.05)
        E = compute_energy(y)
        E_drift = (E[-1] - E[0]) / E[0] * 100
        print(f"   Initial energy: {E[0]:.6f}")
        print(f"   Final energy:   {E[-1]:.6f}")
        print(f"   Energy drift:   {E_drift:.4f}%")
        
    except NotImplementedError:
        print("Complete the rk4_step function first!")
