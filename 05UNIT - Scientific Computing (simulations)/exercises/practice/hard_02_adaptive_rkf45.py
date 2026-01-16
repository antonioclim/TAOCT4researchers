#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Hard 02 — Adaptive Runge-Kutta-Fehlberg
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 60 minutes
TOPIC: Adaptive ODE solvers

OBJECTIVE
─────────
Implement the RKF45 embedded pair method with adaptive step-size control.

BACKGROUND
──────────
RKF45 computes both 4th and 5th order estimates using 6 function evaluations.
The difference provides an error estimate for step-size adjustment:

  error ≈ |y5 - y4|
  h_new = h × (tolerance / error)^(1/5)

Accept step if error < tolerance; otherwise reject and retry with smaller h.

TASKS
─────
1. Implement `rkf45_step` with embedded error estimation
2. Implement `adaptive_solve` with step-size control
3. Solve the Lorenz system and verify tolerance is met

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

ODEFunc = Callable[[float, NDArray[np.floating]], NDArray[np.floating]]


# RKF45 Butcher tableau coefficients
A = np.array([
    [0, 0, 0, 0, 0, 0],
    [1/4, 0, 0, 0, 0, 0],
    [3/32, 9/32, 0, 0, 0, 0],
    [1932/2197, -7200/2197, 7296/2197, 0, 0, 0],
    [439/216, -8, 3680/513, -845/4104, 0, 0],
    [-8/27, 2, -3544/2565, 1859/4104, -11/40, 0],
])

B4 = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])  # 4th order
B5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])  # 5th order

C = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])


@dataclass
class RKF45StepResult:
    """Result of single RKF45 step."""
    
    y4: NDArray[np.floating]  # 4th order estimate
    y5: NDArray[np.floating]  # 5th order estimate
    error: float               # Error estimate
    n_evaluations: int = 6


@dataclass
class AdaptiveSolution:
    """Solution from adaptive ODE solver."""
    
    t: NDArray[np.floating]
    y: NDArray[np.floating]
    n_steps: int
    n_evaluations: int
    n_rejected: int


def rkf45_step(
    f: ODEFunc,
    t: float,
    y: NDArray[np.floating],
    h: float,
) -> RKF45StepResult:
    """Perform single RKF45 step with error estimation.
    
    Args:
        f: Derivative function f(t, y)
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        RKF45StepResult with y4, y5 and error estimate
        
    Example:
        >>> def decay(t, y): return -y
        >>> result = rkf45_step(decay, 0.0, np.array([1.0]), 0.1)
        >>> result.error < 1e-6
        True
    """
    # TODO: Compute the 6 stages k1, k2, ..., k6
    # k = np.zeros((6, len(y)))
    #
    # for i in range(6):
    #     ti = t + C[i] * h
    #     yi = y + h * sum(A[i, j] * k[j] for j in range(i))
    #     k[i] = f(ti, yi)
    
    # TODO: Compute 4th and 5th order estimates
    # y4 = y + h * sum(B4[i] * k[i] for i in range(6))
    # y5 = y + h * sum(B5[i] * k[i] for i in range(6))
    
    # TODO: Compute error estimate (norm of difference)
    # error = np.linalg.norm(y5 - y4)
    
    raise NotImplementedError("Complete this function")


def adaptive_solve(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: NDArray[np.floating],
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_init: float = 0.01,
    h_min: float = 1e-10,
    h_max: float = 1.0,
) -> AdaptiveSolution:
    """Solve ODE with adaptive step-size control.
    
    Args:
        f: Derivative function
        t_span: (t_start, t_end)
        y0: Initial condition
        rtol: Relative tolerance
        atol: Absolute tolerance
        h_init: Initial step size
        h_min: Minimum step size
        h_max: Maximum step size
        
    Returns:
        AdaptiveSolution with trajectory and statistics
    """
    t_start, t_end = t_span
    
    t_list = [t_start]
    y_list = [y0.copy()]
    
    t = t_start
    y = y0.copy()
    h = h_init
    
    n_evaluations = 0
    n_rejected = 0
    
    while t < t_end:
        # Don't overshoot
        if t + h > t_end:
            h = t_end - t
        
        # TODO: Perform RKF45 step
        # result = rkf45_step(f, t, y, h)
        # n_evaluations += result.n_evaluations
        
        # TODO: Compute tolerance
        # scale = atol + rtol * max(np.abs(y).max(), np.abs(result.y5).max())
        
        # TODO: Accept or reject step
        # if result.error <= scale:
        #     # Accept step
        #     t = t + h
        #     y = result.y5  # Use 5th order estimate
        #     t_list.append(t)
        #     y_list.append(y.copy())
        # else:
        #     # Reject step
        #     n_rejected += 1
        
        # TODO: Adjust step size
        # safety = 0.9
        # if result.error > 0:
        #     h_new = safety * h * (scale / result.error) ** 0.2
        # else:
        #     h_new = h * 2  # Error is zero, increase step
        #
        # h = max(h_min, min(h_max, h_new))
        
        raise NotImplementedError("Complete the adaptive loop")
    
    return AdaptiveSolution(
        t=np.array(t_list),
        y=np.array(y_list),
        n_steps=len(t_list) - 1,
        n_evaluations=n_evaluations,
        n_rejected=n_rejected,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LORENZ SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def lorenz(
    t: float,
    y: NDArray[np.floating],
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8/3,
) -> NDArray[np.floating]:
    """Lorenz system derivatives."""
    x, y_coord, z = y
    return np.array([
        sigma * (y_coord - x),
        x * (rho - z) - y_coord,
        x * y_coord - beta * z,
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Adaptive RKF45 ODE Solver")
    print("=" * 70)
    
    # Test with simple exponential decay
    print("\n1. Exponential Decay Test:")
    
    def decay(t: float, y: NDArray[np.floating]) -> NDArray[np.floating]:
        return -y
    
    try:
        sol = adaptive_solve(decay, (0, 5), np.array([1.0]), rtol=1e-6)
        exact = np.exp(-sol.t)
        max_error = np.max(np.abs(sol.y[:, 0] - exact))
        
        print(f"   Steps: {sol.n_steps}")
        print(f"   Function evaluations: {sol.n_evaluations}")
        print(f"   Rejected steps: {sol.n_rejected}")
        print(f"   Maximum error: {max_error:.2e}")
    except NotImplementedError:
        print("   Complete rkf45_step and adaptive_solve first!")
    
    # Test with Lorenz system
    print("\n2. Lorenz System (chaotic):")
    
    try:
        y0 = np.array([1.0, 1.0, 1.0])
        sol = adaptive_solve(lorenz, (0, 20), y0, rtol=1e-6)
        
        print(f"   Steps: {sol.n_steps}")
        print(f"   Function evaluations: {sol.n_evaluations}")
        print(f"   Rejected steps: {sol.n_rejected}")
        print(f"   Average step size: {20 / sol.n_steps:.4f}")
        
        # Step size variation
        dt = np.diff(sol.t)
        print(f"   Step size range: [{dt.min():.2e}, {dt.max():.2e}]")
        
    except NotImplementedError:
        print("   Complete the adaptive solver first!")
    
    # Compare tolerances
    print("\n3. Tolerance Comparison (decay to t=5):")
    print(f"{'Tolerance':>12} {'Steps':>8} {'Evals':>8} {'Error':>12}")
    print("-" * 45)
    
    try:
        for rtol in [1e-3, 1e-6, 1e-9]:
            sol = adaptive_solve(decay, (0, 5), np.array([1.0]), rtol=rtol)
            exact = np.exp(-sol.t)
            max_error = np.max(np.abs(sol.y[:, 0] - exact))
            print(f"{rtol:>12.0e} {sol.n_steps:>8} {sol.n_evaluations:>8} {max_error:>12.2e}")
    except NotImplementedError:
        print("Complete the adaptive solver first!")
