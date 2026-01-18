#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5: Scientific Computing — Laboratory Package
═══════════════════════════════════════════════════════════════════════════════

This package provides implementations of numerical methods and simulation
techniques fundamental to computational research.

MODULES
───────
- lab_05_01_monte_carlo: Monte Carlo integration and estimation
- lab_05_02_ode_solvers: Ordinary differential equation solvers
- lab_05_03_agent_based_modelling: Agent-based simulation framework

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Antonio Clim"
__all__ = [
    # Monte Carlo
    "monte_carlo_integrate",
    "estimate_pi",
    "antithetic_variates",
    "stratified_sampling",
    
    # ODE Solvers
    "euler_step",
    "rk4_step",
    "solve_ode",
    "adaptive_rkf45",
    
    # Agent-Based Modelling
    "Agent",
    "SchellingModel",
    "Boid",
    "BoidsSimulation",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy import attributes from submodules."""
    if name in ("monte_carlo_integrate", "estimate_pi", 
                "antithetic_variates", "stratified_sampling"):
        from . import lab_05_01_monte_carlo as mc
        return getattr(mc, name)
    elif name in ("euler_step", "rk4_step", "solve_ode", "adaptive_rkf45"):
        from . import lab_05_02_ode_solvers as ode
        return getattr(ode, name)
    elif name in ("Agent", "SchellingModel", "Boid", "BoidsSimulation"):
        from . import lab_05_03_agent_based_modelling as abm
        return getattr(abm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
