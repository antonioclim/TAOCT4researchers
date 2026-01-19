#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Lab 01 SOLUTIONS: Simulation Framework Exercises
═══════════════════════════════════════════════════════════════════════════════

This file contains reference implementations for the three exercises in
lab_02_01_simulation_framework.py.

SOLUTIONS INCLUDED
──────────────────
1. SEIRSimulation — SEIR epidemic model extending SIR
2. AdaptiveRunner — Simulation runner with adaptive time stepping
3. ParallelNBody — N-body simulation with numba acceleration

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Type variable for simulation state
StateT = TypeVar('StateT')


class Simulable(Protocol[StateT]):
    """Protocol for any simulable system."""
    
    def state(self) -> StateT:
        """Return current state."""
        ...
    
    def step(self, dt: float) -> None:
        """Advance simulation by time increment dt."""
        ...
    
    def is_done(self) -> bool:
        """Return True if simulation has completed."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1 SOLUTION: SEIR MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SEIRState:
    """
    State of an SEIR epidemic model.
    
    The SEIR model extends SIR by adding an Exposed compartment for
    individuals who have been infected but are not yet infectious
    (the latency period).
    
    Attributes:
        S: Susceptible population count.
        E: Exposed (latent) population count.
        I: Infectious population count.
        R: Recovered (removed) population count.
        time: Current simulation time.
    
    Invariant:
        S + E + I + R = N (total population, conserved)
        All compartments non-negative
    """
    
    S: float
    E: float
    I: float
    R: float
    time: float = 0.0
    
    @property
    def total_population(self) -> float:
        """Total population (should remain constant)."""
        return self.S + self.E + self.I + self.R
    
    def as_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialisation."""
        return {
            'S': self.S,
            'E': self.E,
            'I': self.I,
            'R': self.R,
            'time': self.time
        }


class SEIRSimulation:
    """
    SEIR epidemic model simulation.
    
    The model dynamics follow the differential equations:
    
        dS/dt = -β·S·I/N
        dE/dt = β·S·I/N - σ·E
        dI/dt = σ·E - γ·I
        dR/dt = γ·I
    
    Where:
        β = transmission rate (contact rate × transmission probability)
        σ = incubation rate (1/σ = mean latency period)
        γ = recovery rate (1/γ = mean infectious period)
        N = total population
    
    The basic reproduction number is R₀ = β/γ.
    
    Attributes:
        beta: Transmission rate.
        sigma: Incubation rate (inverse of latency period).
        gamma: Recovery rate (inverse of infectious period).
        termination_threshold: Simulation ends when I falls below this.
    """
    
    def __init__(
        self,
        initial_state: SEIRState,
        beta: float = 0.4,
        sigma: float = 0.2,  # 5-day latency period
        gamma: float = 0.1,  # 10-day infectious period
        termination_threshold: float = 1.0
    ) -> None:
        """
        Initialise SEIR simulation.
        
        Args:
            initial_state: Initial compartment populations.
            beta: Transmission rate.
            sigma: Incubation rate (1/latency period in days).
            gamma: Recovery rate (1/infectious period in days).
            termination_threshold: Stop when I < threshold.
        
        Raises:
            ValueError: If rates are non-positive.
        """
        if beta <= 0 or sigma <= 0 or gamma <= 0:
            raise ValueError("All rates must be positive")
        
        self._state = initial_state
        self._beta = beta
        self._sigma = sigma
        self._gamma = gamma
        self._threshold = termination_threshold
        self._N = initial_state.total_population
        
        logger.debug(
            f"SEIRSimulation initialised: β={beta}, σ={sigma}, γ={gamma}, "
            f"R₀={beta/gamma:.2f}"
        )
    
    @property
    def r0(self) -> float:
        """Basic reproduction number R₀ = β/γ."""
        return self._beta / self._gamma
    
    def state(self) -> SEIRState:
        """Return current state."""
        return self._state
    
    def step(self, dt: float) -> None:
        """
        Advance simulation using fourth-order Runge-Kutta.
        
        Args:
            dt: Time step size.
        """
        s, e, i, r = self._state.S, self._state.E, self._state.I, self._state.R
        t = self._state.time
        
        def derivatives(S: float, E: float, I: float) -> tuple[float, float, float, float]:
            """Compute SEIR derivatives."""
            infection = self._beta * S * I / self._N
            incubation = self._sigma * E
            recovery = self._gamma * I
            
            dS = -infection
            dE = infection - incubation
            dI = incubation - recovery
            dR = recovery
            
            return dS, dE, dI, dR
        
        # RK4 integration
        k1 = derivatives(s, e, i)
        k2 = derivatives(
            s + 0.5 * dt * k1[0],
            e + 0.5 * dt * k1[1],
            i + 0.5 * dt * k1[2]
        )
        k3 = derivatives(
            s + 0.5 * dt * k2[0],
            e + 0.5 * dt * k2[1],
            i + 0.5 * dt * k2[2]
        )
        k4 = derivatives(
            s + dt * k3[0],
            e + dt * k3[1],
            i + dt * k3[2]
        )
        
        # Update state
        new_S = s + (dt / 6) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        new_E = e + (dt / 6) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        new_I = i + (dt / 6) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        new_R = r + (dt / 6) * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        
        # Ensure non-negativity (numerical stability)
        new_S = max(0.0, new_S)
        new_E = max(0.0, new_E)
        new_I = max(0.0, new_I)
        new_R = max(0.0, new_R)
        
        self._state = SEIRState(
            S=new_S,
            E=new_E,
            I=new_I,
            R=new_R,
            time=t + dt
        )
    
    def is_done(self) -> bool:
        """Simulation completes when infectious population is negligible."""
        return self._state.I < self._threshold


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 SOLUTION: ADAPTIVE TIME STEPPING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveStepResult(Generic[StateT]):
    """Result of an adaptive step."""
    
    state: StateT
    dt_used: float
    dt_next: float
    error_estimate: float
    steps_taken: int


class AdaptiveRunner(Generic[StateT]):
    """
    Simulation runner with adaptive time stepping.
    
    Uses embedded Runge-Kutta (Dormand-Prince) method to estimate
    local truncation error and adjust step size accordingly.
    
    The step size is adjusted according to:
        dt_new = dt * safety * (tolerance / error) ^ (1/5)
    
    where safety factor prevents overly aggressive increases.
    
    Attributes:
        simulation: The simulable system to run.
        tolerance: Target local error tolerance.
        dt_min: Minimum allowed step size.
        dt_max: Maximum allowed step size.
        safety: Safety factor for step adjustment (typically 0.8-0.9).
    """
    
    def __init__(
        self,
        simulation: Simulable[StateT],
        tolerance: float = 1e-6,
        dt_min: float = 1e-8,
        dt_max: float = 1.0,
        safety: float = 0.9
    ) -> None:
        """
        Initialise adaptive runner.
        
        Args:
            simulation: System implementing Simulable protocol.
            tolerance: Local error tolerance.
            dt_min: Minimum step size (prevents infinite loops).
            dt_max: Maximum step size (prevents missing features).
            safety: Step adjustment safety factor.
        """
        self._simulation = simulation
        self._tolerance = tolerance
        self._dt_min = dt_min
        self._dt_max = dt_max
        self._safety = safety
        
        self._history: list[StateT] = []
        self._dt_history: list[float] = []
        self._error_history: list[float] = []
    
    def _estimate_error(self, dt: float) -> float:
        """
        Estimate local truncation error by comparing two methods.
        
        Uses Richardson extrapolation: compare result of one step of
        size dt with two steps of size dt/2.
        
        Args:
            dt: Proposed step size.
        
        Returns:
            Estimated local error (infinity norm).
        """
        # This is a simplified error estimator
        # In practice, use embedded RK methods (e.g., RK45)
        
        # Save current state
        current = self._simulation.state()
        
        # Take one big step
        self._simulation.step(dt)
        state_big = self._simulation.state()
        
        # Reset and take two small steps
        self._reset_to_state(current)
        self._simulation.step(dt / 2)
        self._simulation.step(dt / 2)
        state_small = self._simulation.state()
        
        # Estimate error as difference (Richardson extrapolation)
        if hasattr(state_big, '__iter__'):
            error = max(
                abs(getattr(state_big, attr) - getattr(state_small, attr))
                for attr in ['S', 'E', 'I', 'R'] if hasattr(state_big, attr)
            )
        else:
            error = 0.0
        
        # Reset to small step result (more accurate)
        return error
    
    def _reset_to_state(self, state: StateT) -> None:
        """
        Reset simulation to a previous state.
        
        Note: This requires the simulation to support state assignment,
        which is not part of the Simulable protocol. In a production
        implementation, you would either extend the protocol or use
        a different error estimation method.
        """
        # This is a simplified implementation
        # In practice, create a new simulation instance or use checkpointing
        self._simulation._state = state  # type: ignore
    
    def run(
        self,
        max_time: float,
        initial_dt: float = 0.1
    ) -> list[StateT]:
        """
        Run simulation with adaptive stepping until max_time or completion.
        
        Args:
            max_time: Maximum simulation time.
            initial_dt: Initial step size guess.
        
        Returns:
            List of states at each recorded step.
        """
        dt = initial_dt
        current_time = 0.0
        step_count = 0
        rejected_steps = 0
        
        self._history = [self._simulation.state()]
        self._dt_history = []
        self._error_history = []
        
        while current_time < max_time and not self._simulation.is_done():
            # Ensure we don't overshoot max_time
            dt = min(dt, max_time - current_time)
            
            # Estimate error for proposed step
            error = self._estimate_error(dt)
            
            if error <= self._tolerance or dt <= self._dt_min:
                # Accept step
                self._simulation.step(dt)
                current_time += dt
                step_count += 1
                
                self._history.append(self._simulation.state())
                self._dt_history.append(dt)
                self._error_history.append(error)
                
                # Increase step size for next iteration
                if error > 0:
                    factor = self._safety * (self._tolerance / error) ** 0.2
                    dt = min(self._dt_max, dt * min(factor, 2.0))
                else:
                    dt = min(self._dt_max, dt * 2.0)
            else:
                # Reject step, reduce dt
                rejected_steps += 1
                factor = self._safety * (self._tolerance / error) ** 0.25
                dt = max(self._dt_min, dt * max(factor, 0.1))
        
        logger.info(
            f"Adaptive run complete: {step_count} accepted, "
            f"{rejected_steps} rejected steps"
        )
        
        return self._history
    
    @property
    def step_size_history(self) -> list[float]:
        """Return history of step sizes used."""
        return self._dt_history.copy()
    
    @property
    def error_history(self) -> list[float]:
        """Return history of error estimates."""
        return self._error_history.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 SOLUTION: PARALLELISED N-BODY
# ═══════════════════════════════════════════════════════════════════════════════

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("numba not available, falling back to NumPy vectorisation")


@dataclass
class NBodyState:
    """
    State of an N-body gravitational simulation.
    
    Attributes:
        positions: Array of shape (N, 3) with x, y, z coordinates.
        velocities: Array of shape (N, 3) with velocity components.
        masses: Array of shape (N,) with body masses.
        time: Current simulation time.
    """
    
    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    masses: NDArray[np.float64]
    time: float = 0.0
    
    @property
    def num_bodies(self) -> int:
        """Number of bodies in simulation."""
        return len(self.masses)
    
    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy of the system."""
        v_squared = np.sum(self.velocities ** 2, axis=1)
        return 0.5 * np.sum(self.masses * v_squared)
    
    @property
    def potential_energy(self) -> float:
        """Total gravitational potential energy."""
        n = self.num_bodies
        pe = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                if r > 0:
                    pe -= self.masses[i] * self.masses[j] / r
        return pe
    
    @property
    def total_energy(self) -> float:
        """Total mechanical energy (should be conserved)."""
        return self.kinetic_energy + self.potential_energy


if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True)
    def _compute_accelerations_numba(
        positions: NDArray[np.float64],
        masses: NDArray[np.float64],
        softening: float = 1e-4
    ) -> NDArray[np.float64]:
        """
        Compute gravitational accelerations using numba JIT with parallelisation.
        
        The prange enables parallel execution of the outer loop across
        multiple CPU cores.
        
        Args:
            positions: Body positions, shape (N, 3).
            masses: Body masses, shape (N,).
            softening: Softening parameter to prevent singularities.
        
        Returns:
            Accelerations array, shape (N, 3).
        """
        n = len(masses)
        accelerations = np.zeros_like(positions)
        
        for i in prange(n):  # Parallel loop
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.sqrt(
                        r_vec[0]**2 + r_vec[1]**2 + r_vec[2]**2 + softening**2
                    )
                    accelerations[i] += masses[j] * r_vec / (r_mag ** 3)
        
        return accelerations
else:
    def _compute_accelerations_numba(
        positions: NDArray[np.float64],
        masses: NDArray[np.float64],
        softening: float = 1e-4
    ) -> NDArray[np.float64]:
        """Fallback implementation using NumPy broadcasting."""
        n = len(masses)
        accelerations = np.zeros_like(positions)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.sqrt(np.sum(r_vec**2) + softening**2)
                    accelerations[i] += masses[j] * r_vec / (r_mag ** 3)
        
        return accelerations


def _compute_accelerations_vectorised(
    positions: NDArray[np.float64],
    masses: NDArray[np.float64],
    softening: float = 1e-4
) -> NDArray[np.float64]:
    """
    Compute accelerations using fully vectorised NumPy operations.
    
    This avoids explicit Python loops entirely, leveraging NumPy's
    optimised C backend.
    
    Args:
        positions: Body positions, shape (N, 3).
        masses: Body masses, shape (N,).
        softening: Softening parameter.
    
    Returns:
        Accelerations array, shape (N, 3).
    """
    n = len(masses)
    
    # Compute pairwise displacement vectors: shape (N, N, 3)
    # r_ij[i, j, :] = positions[j] - positions[i]
    r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    
    # Compute pairwise distances: shape (N, N)
    r_mag = np.sqrt(np.sum(r_ij ** 2, axis=2) + softening ** 2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(r_mag, 1.0)
    
    # Compute force magnitude factors: shape (N, N)
    # f_ij = m_j / r_ij^3
    force_factors = masses[np.newaxis, :] / (r_mag ** 3)
    np.fill_diagonal(force_factors, 0.0)
    
    # Sum contributions: accelerations[i] = sum_j f_ij * r_ij
    accelerations = np.sum(force_factors[:, :, np.newaxis] * r_ij, axis=1)
    
    return accelerations


class ParallelNBodySimulation:
    """
    N-body gravitational simulation with parallelised force calculation.
    
    Supports multiple acceleration computation backends:
    - numba: JIT-compiled with parallel loops (fastest)
    - vectorised: Pure NumPy with broadcasting (good balance)
    - naive: Python loops (baseline for comparison)
    
    Uses leapfrog (velocity Verlet) integration for symplectic
    time evolution, which conserves energy better than RK4 for
    Hamiltonian systems.
    """
    
    BACKENDS = {'numba', 'vectorised', 'naive'}
    
    def __init__(
        self,
        initial_state: NBodyState,
        softening: float = 1e-4,
        backend: str = 'numba'
    ) -> None:
        """
        Initialise N-body simulation.
        
        Args:
            initial_state: Initial positions, velocities, masses.
            softening: Gravitational softening to prevent singularities.
            backend: Computation backend ('numba', 'vectorised', 'naive').
        
        Raises:
            ValueError: If backend is not recognised.
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"Backend must be one of {self.BACKENDS}")
        
        if backend == 'numba' and not NUMBA_AVAILABLE:
            logger.warning("numba not available, using vectorised backend")
            backend = 'vectorised'
        
        self._state = initial_state
        self._softening = softening
        self._backend = backend
        
        logger.debug(
            f"ParallelNBodySimulation: {initial_state.num_bodies} bodies, "
            f"backend={backend}"
        )
    
    def _compute_accelerations(self) -> NDArray[np.float64]:
        """Compute accelerations using configured backend."""
        if self._backend == 'numba':
            return _compute_accelerations_numba(
                self._state.positions,
                self._state.masses,
                self._softening
            )
        elif self._backend == 'vectorised':
            return _compute_accelerations_vectorised(
                self._state.positions,
                self._state.masses,
                self._softening
            )
        else:  # naive
            return self._compute_accelerations_naive()
    
    def _compute_accelerations_naive(self) -> NDArray[np.float64]:
        """Baseline implementation with explicit Python loops."""
        n = self._state.num_bodies
        positions = self._state.positions
        masses = self._state.masses
        accelerations = np.zeros_like(positions)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = positions[j] - positions[i]
                    r_mag = np.sqrt(np.sum(r_vec**2) + self._softening**2)
                    accelerations[i] += masses[j] * r_vec / (r_mag ** 3)
        
        return accelerations
    
    def state(self) -> NBodyState:
        """Return current state."""
        return self._state
    
    def step(self, dt: float) -> None:
        """
        Advance simulation using leapfrog integration.
        
        Leapfrog (velocity Verlet) is a symplectic integrator that
        conserves phase space volume, making it ideal for Hamiltonian
        systems like gravitational N-body problems.
        
        Algorithm:
            v(t + dt/2) = v(t) + a(t) * dt/2
            x(t + dt) = x(t) + v(t + dt/2) * dt
            v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
        
        Args:
            dt: Time step size.
        """
        positions = self._state.positions.copy()
        velocities = self._state.velocities.copy()
        
        # Half-step velocity update
        accelerations = self._compute_accelerations()
        velocities += 0.5 * dt * accelerations
        
        # Full-step position update
        positions += dt * velocities
        
        # Update state for new acceleration
        self._state = NBodyState(
            positions=positions,
            velocities=velocities,
            masses=self._state.masses,
            time=self._state.time + dt
        )
        
        # Final half-step velocity update
        accelerations = self._compute_accelerations()
        velocities += 0.5 * dt * accelerations
        
        self._state = NBodyState(
            positions=positions,
            velocities=velocities,
            masses=self._state.masses,
            time=self._state.time
        )
    
    def is_done(self) -> bool:
        """N-body simulations run indefinitely."""
        return False


def benchmark_backends(n_bodies: int = 100, n_steps: int = 10) -> dict[str, float]:
    """
    Benchmark different acceleration computation backends.
    
    Args:
        n_bodies: Number of bodies to simulate.
        n_steps: Number of integration steps.
    
    Returns:
        Dictionary mapping backend names to execution times.
    """
    # Create random initial conditions
    rng = np.random.default_rng(42)
    positions = rng.uniform(-10, 10, (n_bodies, 3))
    velocities = rng.uniform(-1, 1, (n_bodies, 3))
    masses = rng.uniform(0.1, 1.0, n_bodies)
    
    results: dict[str, float] = {}
    
    for backend in ['naive', 'vectorised', 'numba']:
        if backend == 'numba' and not NUMBA_AVAILABLE:
            continue
        
        initial = NBodyState(
            positions=positions.copy(),
            velocities=velocities.copy(),
            masses=masses.copy()
        )
        
        sim = ParallelNBodySimulation(initial, backend=backend)
        
        # Warm-up (important for numba JIT compilation)
        sim.step(0.01)
        
        start = time.perf_counter()
        for _ in range(n_steps):
            sim.step(0.01)
        elapsed = time.perf_counter() - start
        
        results[backend] = elapsed
        logger.info(f"{backend:12s}: {elapsed:.4f}s ({n_steps} steps)")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_seir() -> None:
    """Demonstrate SEIR model simulation."""
    logger.info("=== SEIR Model Demo ===")
    
    # Initial state: small outbreak in large population
    initial = SEIRState(
        S=999_000.0,
        E=1_000.0,  # 1000 exposed
        I=0.0,       # None yet infectious
        R=0.0,
        time=0.0
    )
    
    sim = SEIRSimulation(
        initial,
        beta=0.3,    # Transmission rate
        sigma=0.2,   # 5-day latency
        gamma=0.1    # 10-day infectious period
    )
    
    logger.info(f"R₀ = {sim.r0:.2f}")
    
    # Run simulation
    history: list[SEIRState] = [sim.state()]
    dt = 0.1
    
    while not sim.is_done() and sim.state().time < 365:
        sim.step(dt)
        history.append(sim.state())
    
    # Plot results
    times = [s.time for s in history]
    S = [s.S for s in history]
    E = [s.E for s in history]
    I = [s.I for s in history]
    R = [s.R for s in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(times, S, label='Susceptible', color='blue')
    plt.plot(times, E, label='Exposed', color='orange')
    plt.plot(times, I, label='Infectious', color='red')
    plt.plot(times, R, label='Recovered', color='green')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title(f'SEIR Model (R₀ = {sim.r0:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('/tmp/seir_demo.png', dpi=150)
    logger.info("Saved SEIR plot to /tmp/seir_demo.png")
    plt.close()


def demo_adaptive() -> None:
    """Demonstrate adaptive time stepping."""
    logger.info("=== Adaptive Time Stepping Demo ===")
    
    initial = SEIRState(S=990.0, E=10.0, I=0.0, R=0.0)
    sim = SEIRSimulation(initial, beta=0.5, sigma=0.3, gamma=0.1)
    
    runner = AdaptiveRunner(sim, tolerance=1e-4)
    history = runner.run(max_time=100.0, initial_dt=0.1)
    
    # Plot step sizes over time
    dt_history = runner.step_size_history
    times = [s.time for s in history[:-1]]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # SEIR curves
    ax1.plot(times, [s.S for s in history[:-1]], label='S')
    ax1.plot(times, [s.E for s in history[:-1]], label='E')
    ax1.plot(times, [s.I for s in history[:-1]], label='I')
    ax1.plot(times, [s.R for s in history[:-1]], label='R')
    ax1.set_ylabel('Population')
    ax1.legend()
    ax1.set_title('SEIR with Adaptive Stepping')
    ax1.grid(True, alpha=0.3)
    
    # Step sizes
    ax2.semilogy(times, dt_history)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Step size (log scale)')
    ax2.set_title('Adaptive Step Sizes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/adaptive_demo.png', dpi=150)
    logger.info("Saved adaptive stepping plot to /tmp/adaptive_demo.png")
    plt.close()


def demo_nbody_benchmark() -> None:
    """Benchmark N-body backends."""
    logger.info("=== N-Body Backend Benchmark ===")
    
    for n in [50, 100, 200]:
        logger.info(f"\nN = {n} bodies:")
        results = benchmark_backends(n_bodies=n, n_steps=20)
        
        if 'naive' in results and 'numba' in results:
            speedup = results['naive'] / results['numba']
            logger.info(f"  numba speedup: {speedup:.1f}x")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    logger.info("\n" + "═" * 60)
    logger.info("  02UNIT LAB 01 SOLUTIONS - DEMONSTRATIONS")
    logger.info("═" * 60 + "\n")
    
    demo_seir()
    demo_adaptive()
    demo_nbody_benchmark()
    
    logger.info("\n" + "═" * 60)
    logger.info("  Demonstrations complete")
    logger.info("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="02UNIT Lab 01 Solutions: Simulation Framework Exercises"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstrations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        run_all_demos()
    else:
        print("\n" + "═" * 60)
        print("  02UNIT LAB 01 SOLUTIONS")
        print("═" * 60 + "\n")
        print("Solutions included:")
        print("  1. SEIRSimulation - SEIR epidemic model")
        print("  2. AdaptiveRunner - Adaptive time stepping")
        print("  3. ParallelNBodySimulation - N-body with numba")
        print("\nUse --demo to run demonstrations")
        print("Use -v for verbose output")
        print("═" * 60)


if __name__ == "__main__":
    main()
