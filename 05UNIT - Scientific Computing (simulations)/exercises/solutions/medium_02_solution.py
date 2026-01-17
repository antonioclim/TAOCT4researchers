#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: RK4 Harmonic Oscillator — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for medium_02_rk4_harmonic.py

Demonstrates RK4 on harmonic oscillator and energy conservation analysis.

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
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
ODEFunc = Callable[[float, FloatArray], FloatArray]


def harmonic_oscillator(t: float, y: FloatArray, omega: float = 1.0) -> FloatArray:
    """
    Simple harmonic oscillator: y'' + ω²y = 0
    
    As first-order system: [y, y'] → [y', -ω²y]
    
    Args:
        t: Time (unused, system is autonomous)
        y: State [position, velocity]
        omega: Angular frequency
        
    Returns:
        Derivatives [velocity, acceleration]
    """
    return np.array([y[1], -omega**2 * y[0]])


def rk4_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> FloatArray:
    """
    Single RK4 step.
    
    Args:
        f: Derivative function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        New state after step
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h * k1 / 2)
    k3 = f(t + h/2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


def euler_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> FloatArray:
    """Single Euler step for comparison."""
    return y + h * f(t, y)


@dataclass
class ODESolution:
    """Container for ODE solution."""
    
    t: FloatArray
    y: FloatArray
    method: str


def solve_harmonic(
    y0: FloatArray,
    t_span: tuple[float, float],
    n_steps: int,
    method: str = "rk4",
    omega: float = 1.0,
) -> ODESolution:
    """
    Solve harmonic oscillator.
    
    Args:
        y0: Initial [position, velocity]
        t_span: (t_start, t_end)
        n_steps: Number of steps
        method: 'euler' or 'rk4'
        omega: Angular frequency
        
    Returns:
        ODESolution with trajectory
    """
    t_start, t_end = t_span
    t = np.linspace(t_start, t_end, n_steps + 1)
    h = (t_end - t_start) / n_steps
    
    y = np.zeros((n_steps + 1, 2))
    y[0] = y0
    
    f = lambda t_, y_: harmonic_oscillator(t_, y_, omega)
    step_func = rk4_step if method == "rk4" else euler_step
    
    for i in range(n_steps):
        y[i + 1] = step_func(f, t[i], y[i], h)
    
    return ODESolution(t=t, y=y, method=method)


def compute_energy(y: FloatArray, omega: float = 1.0) -> FloatArray:
    """
    Compute total energy E = ½(y'² + ω²y²).
    
    For harmonic oscillator, energy should be conserved.
    
    Args:
        y: State array of shape (n, 2)
        omega: Angular frequency
        
    Returns:
        Energy at each time step
    """
    position = y[:, 0]
    velocity = y[:, 1]
    
    kinetic = 0.5 * velocity**2
    potential = 0.5 * omega**2 * position**2
    
    return kinetic + potential


def exact_solution(
    t: FloatArray,
    y0: FloatArray,
    omega: float = 1.0,
) -> FloatArray:
    """
    Exact solution for harmonic oscillator.
    
    y(t) = A cos(ωt) + B sin(ωt)
    where A = y(0), B = y'(0)/ω
    """
    A = y0[0]
    B = y0[1] / omega
    
    position = A * np.cos(omega * t) + B * np.sin(omega * t)
    velocity = omega * (-A * np.sin(omega * t) + B * np.cos(omega * t))
    
    return np.column_stack([position, velocity])


def energy_drift_analysis(
    y0: FloatArray,
    t_end: float = 100.0,
    n_steps: int = 1000,
    omega: float = 1.0,
) -> dict:
    """
    Compare energy conservation between Euler and RK4.
    
    Returns:
        Dictionary with energy statistics for each method
    """
    results = {}
    
    for method in ["euler", "rk4"]:
        sol = solve_harmonic(y0, (0, t_end), n_steps, method, omega)
        energy = compute_energy(sol.y, omega)
        
        E0 = energy[0]
        drift = (energy - E0) / E0  # Relative drift
        
        results[method] = {
            "initial_energy": E0,
            "final_energy": energy[-1],
            "max_drift": np.max(np.abs(drift)),
            "final_drift": drift[-1],
            "drift_per_step": drift[-1] / n_steps,
        }
    
    return results


def period_analysis(
    y0: FloatArray,
    omega: float = 1.0,
    n_periods: int = 10,
    steps_per_period: int = 100,
) -> dict:
    """
    Analyse period accuracy.
    
    True period T = 2π/ω
    """
    true_period = 2 * np.pi / omega
    t_end = n_periods * true_period
    n_steps = n_periods * steps_per_period
    
    results = {}
    
    for method in ["euler", "rk4"]:
        sol = solve_harmonic(y0, (0, t_end), n_steps, method, omega)
        
        # Find zero crossings (position going from - to +)
        position = sol.y[:, 0]
        crossings = []
        for i in range(len(position) - 1):
            if position[i] <= 0 < position[i + 1]:
                # Linear interpolation for crossing time
                t_cross = sol.t[i] - position[i] * (sol.t[i+1] - sol.t[i]) / (position[i+1] - position[i])
                crossings.append(t_cross)
        
        if len(crossings) >= 2:
            measured_periods = np.diff(crossings)
            avg_period = np.mean(measured_periods)
            period_error = (avg_period - true_period) / true_period
        else:
            avg_period = float('nan')
            period_error = float('nan')
        
        results[method] = {
            "true_period": true_period,
            "measured_period": avg_period,
            "relative_error": period_error,
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initial conditions: position=1, velocity=0
    y0 = np.array([1.0, 0.0])
    omega = 1.0
    
    logger.info("=" * 60)
    logger.info("Harmonic Oscillator: Euler vs RK4")
    logger.info("=" * 60)
    logger.info(f"Initial: position={y0[0]}, velocity={y0[1]}")
    logger.info(f"Angular frequency: ω = {omega}")
    logger.info("")
    
    # Energy drift analysis
    energy = energy_drift_analysis(y0, t_end=100, n_steps=1000)
    
    logger.info("Energy Conservation (100 time units, 1000 steps):")
    for method, stats in energy.items():
        logger.info(f"  {method.upper()}:")
        logger.info(f"    Initial energy: {stats['initial_energy']:.6f}")
        logger.info(f"    Final energy:   {stats['final_energy']:.6f}")
        logger.info(f"    Max drift:      {stats['max_drift']:.2e}")
    
    logger.info("")
    
    # Period analysis
    periods = period_analysis(y0, omega, n_periods=10)
    
    logger.info("Period Accuracy (10 periods):")
    for method, stats in periods.items():
        logger.info(f"  {method.upper()}:")
        logger.info(f"    True period:     {stats['true_period']:.6f}")
        logger.info(f"    Measured period: {stats['measured_period']:.6f}")
        logger.info(f"    Relative error:  {stats['relative_error']:.2e}")
    
    logger.info("")
    logger.info("Conclusion: RK4 conserves energy much better than Euler.")
    logger.info("Euler exhibits systematic energy drift (typically growth).")
