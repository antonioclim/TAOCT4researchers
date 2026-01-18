#!/usr/bin/env python3
"""Solution for Easy 02: Euler's Method."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def euler_step(y: float, h: float) -> float:
    """Perform single Euler step for dy/dt = -y."""
    return y + h * (-y)


def solve_decay(
    y0: float,
    t_end: float,
    h: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve exponential decay from t=0 to t=t_end."""
    n_steps = int(t_end / h)
    t = np.linspace(0, t_end, n_steps + 1)
    y = np.zeros(n_steps + 1)
    y[0] = y0
    
    for i in range(n_steps):
        y[i + 1] = euler_step(y[i], h)
    
    return t, y


if __name__ == "__main__":
    y0 = 1.0
    t_end = 5.0
    
    prev_error = None
    for h in [0.5, 0.25, 0.125, 0.0625]:
        t, y_num = solve_decay(y0, t_end, h)
        y_exact = y0 * np.exp(-t)
        error = np.max(np.abs(y_num - y_exact))
        
        ratio = prev_error / error if prev_error else float("nan")
        print(f"h={h:.4f}: error={error:.6f}, ratio={ratio:.2f}")
        prev_error = error
