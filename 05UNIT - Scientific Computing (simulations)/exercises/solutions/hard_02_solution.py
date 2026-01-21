#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: Adaptive RKF45 Solver — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for hard_02_adaptive_rkf45.py

Full implementation of Runge-Kutta-Fehlberg 4(5) with adaptive stepping.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
ODEFunc = Callable[[float, FloatArray], FloatArray]


@dataclass
class AdaptiveSolution:
    """Solution from adaptive solver."""
    
    t: list[float] = field(default_factory=list)
    y: list[FloatArray] = field(default_factory=list)
    h_history: list[float] = field(default_factory=list)
    n_accepted: int = 0
    n_rejected: int = 0
    n_evaluations: int = 0
    
    def to_arrays(self) -> tuple[FloatArray, FloatArray]:
        """Convert to numpy arrays."""
        return np.array(self.t), np.array(self.y)


def rkf45_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> tuple[FloatArray, FloatArray, float]:
    """
    Single RKF45 step.
    
    Computes 4th and 5th order estimates using 6 function evaluations.
    
    Returns:
        (y4, y5, error) - 4th order result, 5th order result, error estimate
    """
    # RKF45 coefficients (Butcher tableau)
    k1 = f(t, y)
    k2 = f(t + h/4, y + h * k1/4)
    k3 = f(t + 3*h/8, y + h * (3*k1 + 9*k2)/32)
    k4 = f(t + 12*h/13, y + h * (1932*k1 - 7200*k2 + 7296*k3)/2197)
    k5 = f(t + h, y + h * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
    k6 = f(t + h/2, y + h * (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
    
    # 4th order estimate
    y4 = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
    
    # 5th order estimate
    y5 = y + h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    
    # Error estimate (difference between orders)
    error = np.linalg.norm(y5 - y4)
    
    return y4, y5, error


def compute_optimal_step(
    h: float,
    error: float,
    tol: float,
    order: int = 4,
    safety: float = 0.9,
    h_min_factor: float = 0.2,
    h_max_factor: float = 5.0,
) -> tuple[float, bool]:
    """
    Compute optimal step size based on error.
    
    New step: h_new = safety × h × (tol / error)^(1/(order+1))
    
    Args:
        h: Current step size
        error: Estimated error
        tol: Tolerance
        order: Method order (4 for RKF45)
        safety: Safety factor (< 1)
        h_min_factor: Minimum step reduction
        h_max_factor: Maximum step increase
        
    Returns:
        (h_new, accept) - new step size and whether to accept step
    """
    if error < 1e-15:
        # Error negligibly small, increase step
        return h * h_max_factor, True
    
    # Optimal step factor
    factor = safety * (tol / error) ** (1 / (order + 1))
    
    # Limit growth/reduction
    factor = max(h_min_factor, min(h_max_factor, factor))
    
    h_new = h * factor
    accept = error <= tol
    
    return h_new, accept


def adaptive_rkf45(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: FloatArray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_init: float | None = None,
    h_min: float = 1e-12,
    h_max: float | None = None,
    max_steps: int = 100000,
) -> AdaptiveSolution:
    """
    Adaptive RKF45 ODE solver.
    
    Automatically adjusts step size to maintain error within tolerance.
    
    Args:
        f: Derivative function dy/dt = f(t, y)
        t_span: (t_start, t_end)
        y0: Initial condition
        rtol: Relative tolerance
        atol: Absolute tolerance
        h_init: Initial step size (auto if None)
        h_min: Minimum allowed step size
        h_max: Maximum allowed step size
        max_steps: Maximum number of steps
        
    Returns:
        AdaptiveSolution with trajectory and statistics
    """
    t_start, t_end = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    # Set defaults
    if h_max is None:
        h_max = (t_end - t_start) / 10
    if h_init is None:
        h_init = (t_end - t_start) / 1000
    
    # Initialise solution
    sol = AdaptiveSolution()
    sol.t.append(t_start)
    sol.y.append(y0.copy())
    
    t = t_start
    y = y0.copy()
    h = min(h_init, h_max)
    
    while t < t_end and sol.n_accepted < max_steps:
        # Don't overshoot
        if t + h > t_end:
            h = t_end - t
        
        # Attempt step
        y4, y5, error = rkf45_step(f, t, y, h)
        sol.n_evaluations += 6
        
        # Compute tolerance scale
        scale = atol + rtol * max(np.linalg.norm(y), np.linalg.norm(y4))
        
        # Decide accept/reject
        h_new, accept = compute_optimal_step(h, error, scale)
        
        if accept:
            # Accept step
            t = t + h
            y = y4  # Use 4th order result (local extrapolation uses y5)
            
            sol.t.append(t)
            sol.y.append(y.copy())
            sol.h_history.append(h)
            sol.n_accepted += 1
            
            # Update step size for next iteration
            h = min(h_new, h_max)
        else:
            # Reject step
            sol.n_rejected += 1
            h = max(h_new, h_min)
            
            if h <= h_min:
                # Step size too small, accept anyway
                t = t + h
                y = y4
                sol.t.append(t)
                sol.y.append(y.copy())
                sol.h_history.append(h)
                sol.n_accepted += 1
    
    return sol


# ═══════════════════════════════════════════════════════════════════════════════
# TEST SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════


def exponential_decay(t: float, y: FloatArray) -> FloatArray:
    """dy/dt = -y, exact: y = y0 × e^(-t)"""
    return -y


def stiff_system(t: float, y: FloatArray) -> FloatArray:
    """
    Stiff system with two timescales.
    
    dy1/dt = -0.04y1 + 10⁴ y2 y3
    dy2/dt = 0.04y1 - 10⁴ y2 y3 - 3×10⁷ y2²
    dy3/dt = 3×10⁷ y2²
    """
    y1, y2, y3 = y
    return np.array([
        -0.04 * y1 + 1e4 * y2 * y3,
        0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
        3e7 * y2**2,
    ])


def van_der_pol(t: float, y: FloatArray, mu: float = 10.0) -> FloatArray:
    """Van der Pol oscillator (stiff for large μ)."""
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])


def lorenz(t: float, y: FloatArray, sigma: float = 10, rho: float = 28, beta: float = 8/3) -> FloatArray:
    """Lorenz system (chaotic)."""
    x, z, w = y
    return np.array([
        sigma * (z - x),
        x * (rho - w) - z,
        x * z - beta * w,
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Adaptive RKF45 Solver")
    logger.info("=" * 60)
    
    # Test on exponential decay
    logger.info("\n1. Exponential Decay: dy/dt = -y")
    y0 = np.array([1.0])
    sol = adaptive_rkf45(exponential_decay, (0, 10), y0, rtol=1e-8)
    t, y = sol.to_arrays()
    
    exact = np.exp(-t[-1])
    error = abs(y[-1, 0] - exact)
    
    logger.info(f"   Final t: {t[-1]:.1f}")
    logger.info(f"   Final y: {y[-1, 0]:.10f}")
    logger.info(f"   Exact:   {exact:.10f}")
    logger.info(f"   Error:   {error:.2e}")
    logger.info(f"   Steps: {sol.n_accepted} accepted, {sol.n_rejected} rejected")
    logger.info(f"   Evaluations: {sol.n_evaluations}")
    
    # Test on Lorenz system
    logger.info("\n2. Lorenz System (chaotic)")
    y0 = np.array([1.0, 1.0, 1.0])
    sol = adaptive_rkf45(lorenz, (0, 50), y0, rtol=1e-8)
    t, y = sol.to_arrays()
    
    logger.info(f"   Integrated to t = {t[-1]:.1f}")
    logger.info(f"   Final state: [{y[-1, 0]:.4f}, {y[-1, 1]:.4f}, {y[-1, 2]:.4f}]")
    logger.info(f"   Steps: {sol.n_accepted} accepted, {sol.n_rejected} rejected")
    
    # Analyse step size adaptation
    h_array = np.array(sol.h_history)
    logger.info(f"   Step sizes: min={h_array.min():.2e}, max={h_array.max():.2e}")
    logger.info(f"   Step size ratio: {h_array.max() / h_array.min():.1f}x")
    
    # Test on Van der Pol
    logger.info("\n3. Van der Pol (μ=10, mildly stiff)")
    y0 = np.array([2.0, 0.0])
    def f_vdp(t: float, y: np.ndarray) -> np.ndarray:
        return van_der_pol(t, y, mu=10)
    sol = adaptive_rkf45(f_vdp, (0, 30), y0, rtol=1e-6)
    t, y = sol.to_arrays()
    
    logger.info(f"   Integrated to t = {t[-1]:.1f}")
    logger.info(f"   Steps: {sol.n_accepted} accepted, {sol.n_rejected} rejected")
    logger.info(f"   Evaluations: {sol.n_evaluations}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Adaptive stepping automatically uses small steps where needed")
    logger.info("and large steps where the solution is smooth.")
