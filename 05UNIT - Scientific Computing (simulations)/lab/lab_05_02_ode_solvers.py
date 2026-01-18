#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab 2: Ordinary Differential Equation Solvers
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Differential equations model systems that evolve in time—from planetary orbits
to population dynamics to neural activity. When analytical solutions are
unavailable (the common case), numerical methods approximate solutions by
stepping through time. This lab implements the fundamental algorithms from
simple Euler to adaptive Runge-Kutta-Fehlberg.

PREREQUISITES
─────────────
- Week 4: Algorithm efficiency and complexity analysis
- Python: NumPy arrays, function composition
- Mathematics: Derivatives, Taylor series (basic)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement Euler, RK4 and adaptive RKF45 methods
2. Verify convergence order empirically
3. Solve systems of coupled ODEs

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 60 minutes
- Total: 90 minutes

DEPENDENCIES
────────────
numpy>=1.24, scipy>=1.11, matplotlib>=3.7

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type aliases
FloatArray = NDArray[np.floating]
StateVector = FloatArray  # Shape (n,) for n-dimensional state
ODEFunc = Callable[[float, StateVector], StateVector]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ODESolution:
    """Solution to an initial value problem.
    
    Attributes:
        t: Time points where solution was computed
        y: Solution values at each time point (shape: n_times × n_dims)
        method: Name of the method used
        n_steps: Number of steps taken
        n_evaluations: Number of function evaluations
        success: Whether solver completed successfully
        message: Status message
    """
    t: FloatArray
    y: FloatArray
    method: str
    n_steps: int
    n_evaluations: int
    success: bool = True
    message: str = "Success"
    
    @property
    def n_dims(self) -> int:
        """Dimension of state vector."""
        return self.y.shape[1] if self.y.ndim > 1 else 1
    
    def __getitem__(self, idx: int) -> tuple[float, StateVector]:
        """Get (t, y) at index."""
        return float(self.t[idx]), self.y[idx]


@dataclass
class StepResult:
    """Result of a single integration step.
    
    Attributes:
        y_new: New state vector
        error_estimate: Local error estimate (for adaptive methods)
        n_evaluations: Function evaluations in this step
    """
    y_new: StateVector
    error_estimate: float | None = None
    n_evaluations: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BASIC METHODS
# ═══════════════════════════════════════════════════════════════════════════════


def euler_step(
    f: ODEFunc,
    t: float,
    y: StateVector,
    h: float,
) -> StepResult:
    """Single step of Euler's method.
    
    Euler's method uses the derivative at the current point to extrapolate:
        y_{n+1} = y_n + h * f(t_n, y_n)
    
    This is first-order accurate: local error O(h²), global error O(h).
    
    Args:
        f: Right-hand side function dy/dt = f(t, y)
        t: Current time
        y: Current state vector
        h: Step size
        
    Returns:
        StepResult with new state
        
    Examples:
        >>> def exponential(t, y): return y  # dy/dt = y
        >>> result = euler_step(exponential, 0.0, np.array([1.0]), 0.1)
        >>> result.y_new  # Approximates e^0.1 ≈ 1.105
        array([1.1])
    """
    y = np.asarray(y)
    k1 = f(t, y)
    y_new = y + h * k1
    return StepResult(y_new=y_new, n_evaluations=1)


def midpoint_step(
    f: ODEFunc,
    t: float,
    y: StateVector,
    h: float,
) -> StepResult:
    """Single step of the midpoint method (RK2).
    
    Uses the derivative at the midpoint for better accuracy:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        y_{n+1} = y_n + h * k2
    
    Second-order accurate: local error O(h³), global error O(h²).
    
    Args:
        f: Right-hand side function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        StepResult with new state
    """
    y = np.asarray(y)
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    y_new = y + h * k2
    return StepResult(y_new=y_new, n_evaluations=2)


def rk4_step(
    f: ODEFunc,
    t: float,
    y: StateVector,
    h: float,
) -> StepResult:
    """Single step of the classical 4th-order Runge-Kutta method.
    
    The gold-standard explicit method for non-stiff problems:
        k1 = f(t_n, y_n)
        k2 = f(t_n + h/2, y_n + h*k1/2)
        k3 = f(t_n + h/2, y_n + h*k2/2)
        k4 = f(t_n + h, y_n + h*k3)
        y_{n+1} = y_n + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    Fourth-order accurate: local error O(h⁵), global error O(h⁴).
    
    Args:
        f: Right-hand side function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        StepResult with new state
        
    Examples:
        >>> def harmonic(t, y): return np.array([y[1], -y[0]])  # y'' = -y
        >>> y0 = np.array([1.0, 0.0])  # cos(0), -sin(0)
        >>> result = rk4_step(harmonic, 0.0, y0, 0.1)
        >>> np.allclose(result.y_new[0], np.cos(0.1), atol=1e-5)
        True
    """
    y = np.asarray(y)
    
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    
    y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return StepResult(y_new=y_new, n_evaluations=4)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ADAPTIVE METHODS
# ═══════════════════════════════════════════════════════════════════════════════


def rkf45_step(
    f: ODEFunc,
    t: float,
    y: StateVector,
    h: float,
) -> StepResult:
    """Single step of Runge-Kutta-Fehlberg 4(5) method with error estimate.
    
    Computes both 4th and 5th order estimates using 6 function evaluations.
    The difference provides a local error estimate for adaptive stepping.
    
    Args:
        f: Right-hand side function
        t: Current time
        y: Current state
        h: Step size
        
    Returns:
        StepResult with y_new (4th order) and error estimate
    """
    y = np.asarray(y)
    
    # Fehlberg coefficients
    k1 = f(t, y)
    k2 = f(t + h/4, y + h*k1/4)
    k3 = f(t + 3*h/8, y + h*(3*k1 + 9*k2)/32)
    k4 = f(t + 12*h/13, y + h*(1932*k1 - 7200*k2 + 7296*k3)/2197)
    k5 = f(t + h, y + h*(439*k1/216 - 8*k2 + 3680*k3/513 - 845*k4/4104))
    k6 = f(t + h/2, y + h*(-8*k1/27 + 2*k2 - 3544*k3/2565 + 1859*k4/4104 - 11*k5/40))
    
    # 4th order solution (used for stepping)
    y4 = y + h * (25*k1/216 + 1408*k3/2565 + 2197*k4/4104 - k5/5)
    
    # 5th order solution (for error estimation only)
    y5 = y + h * (16*k1/135 + 6656*k3/12825 + 28561*k4/56430 - 9*k5/50 + 2*k6/55)
    
    # Error estimate
    error = np.max(np.abs(y5 - y4))
    
    return StepResult(y_new=y4, error_estimate=error, n_evaluations=6)


def adaptive_rkf45(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: StateVector,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    h_init: float | None = None,
    h_min: float = 1e-12,
    h_max: float | None = None,
    max_steps: int = 100_000,
) -> ODESolution:
    """Solve ODE with adaptive Runge-Kutta-Fehlberg 4(5).
    
    Automatically adjusts step size to maintain error within tolerance.
    
    Args:
        f: Right-hand side dy/dt = f(t, y)
        t_span: (t_start, t_end) integration interval
        y0: Initial state vector
        rtol: Relative tolerance
        atol: Absolute tolerance
        h_init: Initial step size (auto-selected if None)
        h_min: Minimum allowed step size
        h_max: Maximum allowed step size
        max_steps: Maximum number of steps
        
    Returns:
        ODESolution with trajectory
        
    Examples:
        >>> def decay(t, y): return -0.5 * y
        >>> sol = adaptive_rkf45(decay, (0, 10), np.array([1.0]))
        >>> np.allclose(sol.y[-1], np.exp(-5), rtol=1e-5)
        True
    """
    y0 = np.asarray(y0)
    t_start, t_end = t_span
    
    if h_max is None:
        h_max = (t_end - t_start) / 10
    
    if h_init is None:
        # Estimate initial step from derivative magnitude
        f0 = f(t_start, y0)
        scale = atol + rtol * np.abs(y0)
        h_init = min(h_max, 0.01 * (t_end - t_start))
        if np.any(f0 != 0):
            h_init = min(h_init, np.min(scale / np.abs(f0)))
    
    # Storage
    t_list = [t_start]
    y_list = [y0.copy()]
    
    t = t_start
    y = y0.copy()
    h = h_init
    n_steps = 0
    n_evals = 1  # Initial f evaluation
    
    while t < t_end:
        if n_steps >= max_steps:
            logger.warning(f"Max steps ({max_steps}) reached at t={t:.4f}")
            break
        
        # Don't step past end
        if t + h > t_end:
            h = t_end - t
        
        # Take RKF45 step
        result = rkf45_step(f, t, y, h)
        n_evals += result.n_evaluations
        
        # Error control
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(result.y_new))
        error_ratio = result.error_estimate / np.max(scale) if result.error_estimate else 0
        
        if error_ratio <= 1.0:
            # Accept step
            t = t + h
            y = result.y_new
            t_list.append(t)
            y_list.append(y.copy())
            n_steps += 1
            
            # Increase step size
            if error_ratio > 0:
                h_new = h * min(5.0, 0.9 * error_ratio ** (-0.2))
            else:
                h_new = h * 2
            h = min(h_max, max(h_min, h_new))
        else:
            # Reject step, decrease h
            h_new = h * max(0.1, 0.9 * error_ratio ** (-0.25))
            h = max(h_min, h_new)
            
            if h <= h_min:
                logger.warning(f"Step size below minimum at t={t:.6f}")
                # Accept anyway to avoid infinite loop
                t = t + h
                y = result.y_new
                t_list.append(t)
                y_list.append(y.copy())
                n_steps += 1
    
    return ODESolution(
        t=np.array(t_list),
        y=np.array(y_list),
        method="RKF45 (adaptive)",
        n_steps=n_steps,
        n_evaluations=n_evals,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FIXED-STEP SOLVER
# ═══════════════════════════════════════════════════════════════════════════════


class StepMethod(Protocol):
    """Protocol for step methods."""
    def __call__(
        self, f: ODEFunc, t: float, y: StateVector, h: float
    ) -> StepResult: ...


def solve_ode(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: StateVector,
    method: str = "rk4",
    n_steps: int | None = None,
    h: float | None = None,
    t_eval: FloatArray | None = None,
) -> ODESolution:
    """Solve initial value problem with fixed step size.
    
    Args:
        f: Right-hand side dy/dt = f(t, y)
        t_span: (t_start, t_end)
        y0: Initial state
        method: 'euler', 'midpoint', or 'rk4'
        n_steps: Number of steps (alternative to h)
        h: Step size (alternative to n_steps)
        t_eval: Specific times to evaluate (interpolated)
        
    Returns:
        ODESolution with trajectory
        
    Examples:
        >>> def decay(t, y): return -y
        >>> sol = solve_ode(decay, (0, 5), np.array([1.0]), method='rk4', n_steps=100)
        >>> np.allclose(sol.y[-1], np.exp(-5), rtol=1e-4)
        True
    """
    methods: dict[str, StepMethod] = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods)}")
    
    step_func = methods[method]
    y0 = np.asarray(y0)
    t_start, t_end = t_span
    
    # Determine step size
    if n_steps is not None:
        h = (t_end - t_start) / n_steps
    elif h is None:
        n_steps = 100
        h = (t_end - t_start) / n_steps
    else:
        n_steps = int(np.ceil((t_end - t_start) / h))
    
    # Integration
    t_list = [t_start]
    y_list = [y0.copy()]
    
    t = t_start
    y = y0.copy()
    total_evals = 0
    
    while t < t_end - 1e-12:
        # Don't overshoot
        step_h = min(h, t_end - t)
        
        result = step_func(f, t, y, step_h)
        total_evals += result.n_evaluations
        
        t = t + step_h
        y = result.y_new
        
        t_list.append(t)
        y_list.append(y.copy())
    
    t_arr = np.array(t_list)
    y_arr = np.array(y_list)
    
    # Interpolate to t_eval if requested
    if t_eval is not None:
        from scipy.interpolate import interp1d
        
        interp = interp1d(t_arr, y_arr, axis=0, kind="cubic")
        t_arr = np.asarray(t_eval)
        y_arr = interp(t_arr)
    
    return ODESolution(
        t=t_arr,
        y=y_arr,
        method=method,
        n_steps=len(t_list) - 1,
        n_evaluations=total_evals,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CONVERGENCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ConvergenceResult:
    """Result of convergence order analysis.
    
    Attributes:
        step_sizes: Array of h values tested
        errors: Error at each h
        empirical_order: Fitted convergence order
        theoretical_order: Expected order for method
    """
    step_sizes: FloatArray
    errors: FloatArray
    empirical_order: float
    theoretical_order: int


def analyse_convergence(
    f: ODEFunc,
    t_span: tuple[float, float],
    y0: StateVector,
    y_exact: Callable[[float], StateVector],
    method: str = "rk4",
    step_sizes: list[float] | None = None,
) -> ConvergenceResult:
    """Analyse convergence order of an ODE method.
    
    Compares numerical solutions at the final time against exact solution
    for decreasing step sizes. Fits log(error) vs log(h) to estimate order.
    
    Args:
        f: ODE right-hand side
        t_span: Integration interval
        y0: Initial condition
        y_exact: Exact solution function
        method: Method name ('euler', 'midpoint', 'rk4')
        step_sizes: Step sizes to test
        
    Returns:
        ConvergenceResult with errors and fitted order
        
    Examples:
        >>> def f(t, y): return y
        >>> y_exact = lambda t: np.array([np.exp(t)])
        >>> result = analyse_convergence(f, (0, 1), np.array([1.0]), y_exact)
        >>> abs(result.empirical_order - 4) < 0.5  # RK4 is O(h⁴)
        True
    """
    theoretical_orders = {"euler": 1, "midpoint": 2, "rk4": 4}
    
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.025, 0.0125, 0.00625]
    
    t_end = t_span[1]
    y_true = y_exact(t_end)
    
    errors = []
    for h in step_sizes:
        sol = solve_ode(f, t_span, y0, method=method, h=h)
        error = np.max(np.abs(sol.y[-1] - y_true))
        errors.append(error)
        logger.debug(f"h={h:.6f}: error={error:.2e}")
    
    # Fit order: log(error) = p * log(h) + c
    log_h = np.log(step_sizes)
    log_err = np.log(errors)
    
    slope, _ = np.polyfit(log_h, log_err, 1)
    
    logger.info(
        f"Method {method}: empirical order = {slope:.2f} "
        f"(theoretical: {theoretical_orders.get(method, '?')})"
    )
    
    return ConvergenceResult(
        step_sizes=np.array(step_sizes),
        errors=np.array(errors),
        empirical_order=slope,
        theoretical_order=theoretical_orders.get(method, 0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COMMON ODE SYSTEMS
# ═══════════════════════════════════════════════════════════════════════════════


def harmonic_oscillator(t: float, y: StateVector) -> StateVector:
    """Simple harmonic oscillator: y'' + y = 0.
    
    State: y = [position, velocity]
    Returns: [velocity, -position]
    
    Exact solution: y(t) = [cos(t + φ), -sin(t + φ)]
    """
    return np.array([y[1], -y[0]])


def damped_oscillator(
    gamma: float = 0.1,
    omega: float = 1.0,
) -> ODEFunc:
    """Damped harmonic oscillator: y'' + 2γy' + ω²y = 0.
    
    Args:
        gamma: Damping coefficient
        omega: Natural frequency
        
    Returns:
        ODE function
    """
    def f(t: float, y: StateVector) -> StateVector:
        return np.array([y[1], -2*gamma*y[1] - omega**2*y[0]])
    return f


def lotka_volterra(
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 1.5,
    delta: float = 0.075,
) -> ODEFunc:
    """Lotka-Volterra predator-prey model.
    
    dx/dt = αx - βxy  (prey growth - predation)
    dy/dt = δxy - γy  (predator growth - death)
    
    Args:
        alpha: Prey birth rate
        beta: Predation rate
        gamma: Predator death rate
        delta: Predator birth rate per prey
        
    Returns:
        ODE function for state [prey, predator]
    """
    def f(t: float, y: StateVector) -> StateVector:
        x, y_pred = y
        dxdt = alpha * x - beta * x * y_pred
        dydt = delta * x * y_pred - gamma * y_pred
        return np.array([dxdt, dydt])
    return f


def exponential_decay(lam: float = 1.0) -> ODEFunc:
    """Exponential decay: dy/dt = -λy.
    
    Args:
        lam: Decay rate (positive)
        
    Returns:
        ODE function
    """
    def f(t: float, y: StateVector) -> StateVector:
        return -lam * y
    return f


def van_der_pol(mu: float = 1.0) -> ODEFunc:
    """Van der Pol oscillator: y'' - μ(1-y²)y' + y = 0.
    
    A nonlinear oscillator with limit cycle behaviour.
    
    Args:
        mu: Nonlinearity parameter
        
    Returns:
        ODE function for state [y, y']
    """
    def f(t: float, state: StateVector) -> StateVector:
        y, ydot = state
        return np.array([ydot, mu * (1 - y**2) * ydot - y])
    return f


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_euler_vs_rk4() -> None:
    """Compare Euler and RK4 on exponential decay."""
    logger.info("=" * 60)
    logger.info("DEMO: Euler vs RK4 Comparison")
    logger.info("=" * 60)
    
    f = exponential_decay(1.0)
    y0 = np.array([1.0])
    t_span = (0.0, 5.0)
    y_exact = lambda t: np.array([np.exp(-t)])
    
    for method in ["euler", "rk4"]:
        for n_steps in [10, 50, 100]:
            sol = solve_ode(f, t_span, y0, method=method, n_steps=n_steps)
            error = abs(sol.y[-1, 0] - y_exact(5.0)[0])
            logger.info(
                f"{method:>8}, n={n_steps:>3}: y(5)={sol.y[-1,0]:.6f}, "
                f"exact={y_exact(5.0)[0]:.6f}, error={error:.2e}"
            )


def demo_convergence() -> None:
    """Demonstrate convergence order analysis."""
    logger.info("=" * 60)
    logger.info("DEMO: Convergence Order Analysis")
    logger.info("=" * 60)
    
    f = exponential_decay(1.0)
    y0 = np.array([1.0])
    t_span = (0.0, 1.0)
    y_exact = lambda t: np.array([np.exp(-t)])
    
    for method in ["euler", "midpoint", "rk4"]:
        result = analyse_convergence(
            f, t_span, y0, y_exact, method=method,
            step_sizes=[0.1, 0.05, 0.025, 0.0125],
        )
        logger.info(
            f"{method:>8}: empirical order = {result.empirical_order:.2f}, "
            f"theoretical = {result.theoretical_order}"
        )


def demo_harmonic_oscillator() -> None:
    """Demonstrate harmonic oscillator solution."""
    logger.info("=" * 60)
    logger.info("DEMO: Harmonic Oscillator")
    logger.info("=" * 60)
    
    y0 = np.array([1.0, 0.0])  # Start at max displacement
    t_span = (0.0, 2 * np.pi)  # One period
    
    # Exact solution
    y_exact = lambda t: np.array([np.cos(t), -np.sin(t)])
    
    sol_euler = solve_ode(harmonic_oscillator, t_span, y0, "euler", n_steps=100)
    sol_rk4 = solve_ode(harmonic_oscillator, t_span, y0, "rk4", n_steps=100)
    
    # Check energy conservation (E = y² + v²)/2
    def energy(y: FloatArray) -> float:
        return 0.5 * (y[0]**2 + y[1]**2)
    
    e0 = energy(y0)
    e_euler = energy(sol_euler.y[-1])
    e_rk4 = energy(sol_rk4.y[-1])
    
    logger.info(f"Initial energy: {e0:.6f}")
    logger.info(f"Euler final energy: {e_euler:.6f} (drift: {e_euler - e0:.6f})")
    logger.info(f"RK4 final energy: {e_rk4:.6f} (drift: {e_rk4 - e0:.6f})")


def demo_lotka_volterra() -> None:
    """Demonstrate predator-prey dynamics."""
    logger.info("=" * 60)
    logger.info("DEMO: Lotka-Volterra Predator-Prey")
    logger.info("=" * 60)
    
    f = lotka_volterra(alpha=1.0, beta=0.1, gamma=1.5, delta=0.075)
    y0 = np.array([10.0, 5.0])  # 10 prey, 5 predators
    t_span = (0.0, 50.0)
    
    sol = adaptive_rkf45(f, t_span, y0)
    
    logger.info(f"Solved in {sol.n_steps} steps, {sol.n_evaluations} evaluations")
    logger.info(f"Final populations: prey={sol.y[-1,0]:.2f}, predator={sol.y[-1,1]:.2f}")
    
    # Find extrema
    prey = sol.y[:, 0]
    predator = sol.y[:, 1]
    logger.info(f"Prey range: [{prey.min():.2f}, {prey.max():.2f}]")
    logger.info(f"Predator range: [{predator.min():.2f}, {predator.max():.2f}]")


def demo_adaptive() -> None:
    """Demonstrate adaptive step-size control."""
    logger.info("=" * 60)
    logger.info("DEMO: Adaptive Step-Size Control")
    logger.info("=" * 60)
    
    # Stiff-ish problem: Van der Pol with large μ
    f = van_der_pol(mu=3.0)
    y0 = np.array([2.0, 0.0])
    t_span = (0.0, 20.0)
    
    sol = adaptive_rkf45(f, t_span, y0, rtol=1e-6, atol=1e-9)
    
    logger.info(f"Steps: {sol.n_steps}")
    logger.info(f"Function evaluations: {sol.n_evaluations}")
    logger.info(f"Evals per step: {sol.n_evaluations / sol.n_steps:.2f}")
    
    # Compare step sizes
    dt = np.diff(sol.t)
    logger.info(f"Step size range: [{dt.min():.6f}, {dt.max():.6f}]")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_euler_vs_rk4()
    print()
    demo_convergence()
    print()
    demo_harmonic_oscillator()
    print()
    demo_lotka_volterra()
    print()
    demo_adaptive()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="ODE Solvers Laboratory"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration examples",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--convergence",
        choices=["euler", "midpoint", "rk4"],
        help="Run convergence analysis for specified method",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.convergence:
        f = exponential_decay(1.0)
        y0 = np.array([1.0])
        t_span = (0.0, 1.0)
        y_exact = lambda t: np.array([np.exp(-t)])
        
        result = analyse_convergence(
            f, t_span, y0, y_exact, method=args.convergence,
        )
        print(f"Empirical convergence order: {result.empirical_order:.3f}")
        return
    
    if args.demo:
        run_all_demos()
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
