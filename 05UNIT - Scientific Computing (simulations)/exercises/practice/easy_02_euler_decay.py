#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Easy 02 — Euler's Method for Exponential Decay
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes
TOPIC: ODE solvers (Euler)

OBJECTIVE
─────────
Implement Euler's method to solve the exponential decay equation dy/dt = -y.

BACKGROUND
──────────
Exponential decay is described by dy/dt = -y with solution y(t) = y₀·e^(-t).
Euler's method approximates: y_{n+1} = y_n + h·f(t_n, y_n)

TASKS
─────
1. Complete the `euler_step` function
2. Complete the `solve_decay` function
3. Compare numerical solution with exact solution

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def euler_step(
    y: float,
    h: float,
) -> float:
    """Perform single Euler step for dy/dt = -y.
    
    Args:
        y: Current value
        h: Step size
        
    Returns:
        Next value y_{n+1}
        
    Example:
        >>> euler_step(1.0, 0.1)
        0.9
    """
    # TODO: Implement Euler's method: y_new = y + h * f(y)
    # For dy/dt = -y, we have f(y) = -y
    
    raise NotImplementedError("Complete this function")


def solve_decay(
    y0: float,
    t_end: float,
    h: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve exponential decay from t=0 to t=t_end.
    
    Args:
        y0: Initial condition y(0)
        t_end: Final time
        h: Step size
        
    Returns:
        Tuple of (t_values, y_values) arrays
        
    Example:
        >>> t, y = solve_decay(1.0, 1.0, 0.1)
        >>> len(t) == len(y) == 11
        True
    """
    # TODO: Create array of time values from 0 to t_end with step h
    # t = ...
    
    # TODO: Initialise y array with same length
    # y = ...
    # y[0] = y0
    
    # TODO: Loop and apply Euler steps
    # for i in range(len(t) - 1):
    #     y[i+1] = euler_step(y[i], h)
    
    raise NotImplementedError("Complete this function")


def exact_solution(t: NDArray[np.floating], y0: float) -> NDArray[np.floating]:
    """Exact solution y(t) = y0 * exp(-t)."""
    return y0 * np.exp(-t)


def compute_error(
    y_numerical: NDArray[np.floating],
    y_exact: NDArray[np.floating],
) -> float:
    """Compute maximum absolute error."""
    return float(np.max(np.abs(y_numerical - y_exact)))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    y0 = 1.0
    t_end = 5.0
    
    print("Euler's Method for Exponential Decay")
    print("=" * 50)
    print(f"{'Step size h':>12} {'Max Error':>15} {'Error Ratio':>12}")
    print("-" * 50)
    
    prev_error = None
    
    for h in [0.5, 0.25, 0.125, 0.0625]:
        try:
            t, y_num = solve_decay(y0, t_end, h)
            y_exact = exact_solution(t, y0)
            error = compute_error(y_num, y_exact)
            
            ratio = prev_error / error if prev_error else float("nan")
            print(f"{h:>12.4f} {error:>15.6f} {ratio:>12.2f}")
            
            prev_error = error
        except NotImplementedError:
            print("Complete the euler_step and solve_decay functions first!")
            break
    
    print("-" * 50)
    print("Note: Error ratio ≈ 2 confirms O(h) convergence")
