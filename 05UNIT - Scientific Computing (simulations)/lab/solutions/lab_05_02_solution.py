#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab 2: ODE Solvers — SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

Complete reference implementation for lab_5_02_ode_solvers.py

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]
ODEFunc = Callable[[float, FloatArray], FloatArray]


@dataclass
class StepResult:
    """Result from a single ODE step."""
    
    y_new: FloatArray
    error_estimate: float | None = None
    n_evaluations: int = 1


@dataclass
class ODESolution:
    """Container for ODE solution."""
    
    t: FloatArray
    y: FloatArray
    method: str
    n_steps: int
    n_evaluations: int
    success: bool = True
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions in state vector."""
        return self.y.shape[1] if self.y.ndim > 1 else 1
    
    def __getitem__(self, idx: int) -> tuple[float, FloatArray]:
        """Get (t, y) at index."""
        return self.t[idx], self.y[idx]
    
    def __iter__(self) -> Iterator[tuple[float, FloatArray]]:
        """Iterate over (t, y) pairs."""
        for i in range(len(self.t)):
            yield self.t[i], self.y[i]


def euler_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> StepResult:
    """
    Single Euler step: y_{n+1} = y_n + h·f(t_n, y_n)
    
    Args:
        f: Derivative function dy/dt = f(t, y)
        t: Current time
        y: Current state vector
        h: Step size
        
    Returns:
        StepResult with new state
    """
    k1 = f(t, y)
    y_new = y + h * k1
    return StepResult(y_new=y_new, n_evaluations=1)


def midpoint_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> StepResult:
    """
    Midpoint method (RK2): evaluate at midpoint of interval.
    
    k1 = f(t, y)
    k2 = f(t + h/2, y + h·k1/2)
    y_{n+1} = y_n + h·k2
    
    Args:
        f: Derivative function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        StepResult with new state
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h * k1 / 2)
    y_new = y + h * k2
    return StepResult(y_new=y_new, n_evaluations=2)


def rk4_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> StepResult:
    """
    Classical fourth-order Runge-Kutta step.
    
    k1 = f(t, y)
    k2 = f(t + h/2, y + h·k1/2)
    k3 = f(t + h/2, y + h·k2/2)
    k4 = f(t + h, y + h·k3)
    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
    
    Args:
        f: Derivative function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        StepResult with new state (4 evaluations)
    """
    k1 = f(t, y)
    k2 = f(t + h/2, y + h * k1 / 2)
    k3 = f(t + h/2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    
    y_new = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return StepResult(y_new=y_new, n_evaluations=4)


def rkf45_step(
    f: ODEFunc,
    t: float,
    y: FloatArray,
    h: float,
) -> StepResult:
    """
    Runge-Kutta-Fehlberg 4(5) step with embedded error estimate.
    
    Computes both 4th and 5th order estimates using 6 evaluations.
    The difference provides local error estimate for adaptive stepping.
    
    Args:
        f: Derivative function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        StepResult with 4th order estimate and error
    """
    # RKF45 coefficients
    k1 = f(t, y)
    k2 = f(t + h/4, y + h * k1 / 4)
    k3 = f(t + 3*h/8, y + h * (3*k1 + 9*k2) / 32)
    k4 = f(t + 12*h/13, y + h * (1932*k1 - 7200*k2 + 7296*k3) / 2197)
    k5 = f(t + h, y + h * (439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
    k6 = f(t + h/2, y + h * (-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
    
    # 4th order estimate
    y4 = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
    
    # 5th order estimate
    y5 = y + h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    
    # Error estimate
    error = np.linalg.norm(y5 - y4)
    
    return StepResult(y_new=y4, error_estimate=error, n_evaluations=6)


def solve_ode(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: FloatArray,
    method: str = "rk4",
    h: float | None = None,
    n_steps: int | None = None,
) -> ODESolution:
    """
    Solve ODE using fixed-step method.
    
    Args:
        f: Derivative function dy/dt = f(t, y)
        t_span: (t_start, t_end)
        y0: Initial condition
        method: 'euler', 'midpoint', or 'rk4'
        h: Fixed step size (specify either h or n_steps)
        n_steps: Number of steps (specify either h or n_steps)
        
    Returns:
        ODESolution with trajectory
        
    Raises:
        ValueError: If method is unknown or step size invalid
    """
    t_start, t_end = t_span
    
    # Determine step size
    if h is not None and n_steps is not None:
        raise ValueError("Specify either h or n_steps, not both")
    elif h is not None:
        n_steps = int(np.ceil((t_end - t_start) / h))
    elif n_steps is not None:
        h = (t_end - t_start) / n_steps
    else:
        n_steps = 100
        h = (t_end - t_start) / n_steps
    
    # Select step function
    step_funcs = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }
    
    if method not in step_funcs:
        raise ValueError(f"Unknown method: {method}")
    
    step_func = step_funcs[method]
    
    # Ensure y0 is array
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    # Allocate arrays
    t_values = np.linspace(t_start, t_end, n_steps + 1)
    y_values = np.zeros((n_steps + 1, len(y0)))
    y_values[0] = y0
    
    total_evals = 0
    y = y0.copy()
    
    for i in range(n_steps):
        result = step_func(f, t_values[i], y, h)
        y = result.y_new
        y_values[i + 1] = y
        total_evals += result.n_evaluations
    
    logger.debug(
        f"ODE solved: method={method}, steps={n_steps}, evals={total_evals}"
    )
    
    return ODESolution(
        t=t_values,
        y=y_values,
        method=method,
        n_steps=n_steps,
        n_evaluations=total_evals,
    )


def adaptive_rkf45(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: FloatArray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_init: float | None = None,
    h_min: float = 1e-10,
    h_max: float | None = None,
    max_steps: int = 100_000,
) -> ODESolution:
    """
    Adaptive RKF45 solver with automatic step size control.
    
    Args:
        f: Derivative function
        t_span: (t_start, t_end)
        y0: Initial condition
        rtol: Relative tolerance
        atol: Absolute tolerance
        h_init: Initial step size (auto if None)
        h_min: Minimum step size
        h_max: Maximum step size
        max_steps: Maximum number of steps
        
    Returns:
        ODESolution with adaptive trajectory
    """
    t_start, t_end = t_span
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    
    if h_max is None:
        h_max = (t_end - t_start) / 10
    if h_init is None:
        h_init = h_max / 100
    
    t_list = [t_start]
    y_list = [y0.copy()]
    
    t = t_start
    y = y0.copy()
    h = h_init
    total_evals = 0
    n_steps = 0
    
    while t < t_end and n_steps < max_steps:
        # Don't overshoot
        if t + h > t_end:
            h = t_end - t
        
        # Take RKF45 step
        result = rkf45_step(f, t, y, h)
        total_evals += result.n_evaluations
        
        # Error control
        scale = atol + rtol * max(np.linalg.norm(y), np.linalg.norm(result.y_new))
        error_ratio = result.error_estimate / scale if scale > 0 else 0
        
        if error_ratio <= 1.0:
            # Accept step
            t = t + h
            y = result.y_new
            t_list.append(t)
            y_list.append(y.copy())
            n_steps += 1
            
            # Increase step size
            if error_ratio > 0:
                h_new = 0.9 * h * (1.0 / error_ratio) ** 0.2
            else:
                h_new = h * 2
            h = min(h_new, h_max)
        else:
            # Reject step, decrease h
            h_new = 0.9 * h * (1.0 / error_ratio) ** 0.25
            h = max(h_new, h_min)
    
    return ODESolution(
        t=np.array(t_list),
        y=np.array(y_list),
        method="rkf45",
        n_steps=n_steps,
        n_evaluations=total_evals,
        success=(t >= t_end - 1e-10),
    )


@dataclass
class ConvergenceResult:
    """Result of convergence analysis."""
    
    step_sizes: list[float]
    errors: list[float]
    empirical_order: float
    r_squared: float


def analyse_convergence(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: FloatArray,
    exact: Callable[[float], FloatArray],
    method: str = "rk4",
    step_sizes: list[float] | None = None,
) -> ConvergenceResult:
    """
    Empirically verify convergence order of ODE method.
    
    Args:
        f: Derivative function
        t_span: Integration interval
        y0: Initial condition
        exact: Exact solution function
        method: Method to analyse
        step_sizes: Step sizes to test
        
    Returns:
        ConvergenceResult with empirical order
    """
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    t_end = t_span[1]
    y_exact = np.atleast_1d(exact(t_end))
    
    errors = []
    for h in step_sizes:
        sol = solve_ode(f, t_span, y0, method=method, h=h)
        error = np.linalg.norm(sol.y[-1] - y_exact)
        errors.append(error)
    
    # Linear regression on log-log scale
    log_h = np.log(step_sizes)
    log_err = np.log(errors)
    
    coeffs = np.polyfit(log_h, log_err, 1)
    empirical_order = coeffs[0]
    
    predicted = np.polyval(coeffs, log_h)
    ss_res = np.sum((log_err - predicted) ** 2)
    ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    
    return ConvergenceResult(
        step_sizes=step_sizes,
        errors=errors,
        empirical_order=empirical_order,
        r_squared=r_squared,
    )


# Common test systems

def exponential_decay(t: float, y: FloatArray) -> FloatArray:
    """dy/dt = -y, exact: y(t) = y0·e^(-t)"""
    return -y


def harmonic_oscillator(t: float, y: FloatArray) -> FloatArray:
    """y'' + y = 0, as system: [y, y'] → [y', -y]"""
    return np.array([y[1], -y[0]])


def damped_oscillator(t: float, y: FloatArray, gamma: float = 0.1) -> FloatArray:
    """y'' + γy' + y = 0"""
    return np.array([y[1], -gamma * y[1] - y[0]])


def lotka_volterra(
    t: float,
    y: FloatArray,
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 1.5,
    delta: float = 0.075,
) -> FloatArray:
    """
    Predator-prey model.
    
    dx/dt = αx - βxy  (prey)
    dy/dt = δxy - γy  (predator)
    """
    x, p = y
    return np.array([
        alpha * x - beta * x * p,
        delta * x * p - gamma * p,
    ])


def van_der_pol(t: float, y: FloatArray, mu: float = 1.0) -> FloatArray:
    """Van der Pol oscillator: y'' - μ(1-y²)y' + y = 0"""
    return np.array([y[1], mu * (1 - y[0]**2) * y[1] - y[0]])
