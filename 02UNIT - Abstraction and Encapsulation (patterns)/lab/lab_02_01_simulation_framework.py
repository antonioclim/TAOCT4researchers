#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Lab 01: Simulation Framework with Protocols and Generics
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Research simulations pervade diverse disciplines: N-body systems and fluid
dynamics in physics, population dynamics and epidemic models in biology,
market simulations and agent-based models in economics, atmospheric and
ocean circulation models in climatology. A principled simulation framework
must satisfy four essential properties: genericity (operating with arbitrary
models), extensibility (accommodating new models without modification),
testability (permitting isolated component verification), and efficiency
(minimising computational overhead).

PREREQUISITES
─────────────
- 01UNIT: State concept from Turing machines, AST hierarchies
- Python: Intermediate proficiency with type hints and dataclasses
- Libraries: numpy, matplotlib

LEARNING OBJECTIVES
───────────────────
Upon completion, participants will:
1. Define Protocol interfaces for simulable systems
2. Implement multiple concrete simulation models (SIR, Lotka-Volterra)
3. Construct generic simulation runners via structural subtyping
4. Apply the Observer pattern for real-time visualisation

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 90 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
numpy>=1.24
matplotlib>=3.7
scipy>=1.11

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterator, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROTOCOL FOR SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# TypeVar for simulation state (can be anything)
StateT = TypeVar('StateT')


class Simulable(Protocol[StateT]):
    """
    Protocol for any simulable system.
    
    A simulable system must be able to:
    1. Report its current state
    2. Advance by a time step
    3. Indicate when the simulation has finished
    
    Design Decisions
    ────────────────
    We use Protocol (structural typing) rather than ABC (nominal typing)
    because it allows integration with existing classes without modification.
    
    StateT is generic because different simulations have different states:
    - SIR: (S, I, R) — three numbers
    - N-body: array of positions and velocities
    - Cellular automaton: 2D matrix
    
    step() receives dt for variable time step simulations.
    
    Invariants
    ──────────
    - After step(dt), simulation time increases by dt
    - state() always returns a valid state
    - is_done() is monotonic: once True, remains True
    
    Example Implementation
    ──────────────────────
    >>> class MySimulation:
    ...     def state(self) -> MyState:
    ...         return self._current_state
    ...     
    ...     def step(self, dt: float) -> None:
    ...         self._current_state = self._compute_next(dt)
    ...     
    ...     def is_done(self) -> bool:
    ...         return self._time > self._max_time
    """
    
    def state(self) -> StateT:
        """Return the current state of the simulation."""
        ...
    
    def step(self, dt: float) -> None:
        """Advance the simulation by one time step dt."""
        ...
    
    def is_done(self) -> bool:
        """Check whether the simulation has completed."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SIR EPIDEMIC MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SIRState:
    """
    State of the SIR (Susceptible-Infected-Recovered) model.
    
    The SIR model describes epidemic dynamics in a population:
    
    ```
        β·S·I/N         γ·I
    S ──────────▶ I ──────────▶ R
    
    Susceptible    Infected    Recovered
    ```
    
    Differential Equations
    ──────────────────────
    dS/dt = -β·S·I/N
    dI/dt = β·S·I/N - γ·I  
    dR/dt = γ·I
    
    Parameters
    ──────────
    β (beta): Transmission rate (infectious contacts / time)
    γ (gamma): Recovery rate (1/γ = mean duration of illness)
    R₀ = β/γ: Basic reproduction number
    
    Historical Note
    ───────────────
    The model was introduced by Kermack and McKendrick in 1927.
    It was used to analyse the 1918 influenza pandemic.
    
    Attributes:
        susceptible: Number of susceptible individuals
        infected: Number of infected individuals
        recovered: Number of recovered (immune) individuals
        time: Current simulation time
    """
    susceptible: float
    infected: float
    recovered: float
    time: float = 0.0
    
    @property
    def total(self) -> float:
        """Total population (invariant: must remain constant)."""
        return self.susceptible + self.infected + self.recovered
    
    def as_tuple(self) -> tuple[float, float, float]:
        """Return (S, I, R) for plotting."""
        return (self.susceptible, self.infected, self.recovered)
    
    def as_fractions(self) -> tuple[float, float, float]:
        """Return (S/N, I/N, R/N) as population fractions."""
        n = self.total
        return (self.susceptible / n, self.infected / n, self.recovered / n)


@dataclass
class SIRSimulation:
    """
    SIR simulation using the Euler method for numerical integration.
    
    Euler Method
    ────────────
    For an equation dy/dt = f(y, t):
    
    y(t + dt) ≈ y(t) + dt · f(y, t)
    
    This is the simplest method, but not the most accurate.
    For serious simulations, use Runge-Kutta (scipy.integrate.solve_ivp).
    
    Attributes:
        beta: Transmission rate
        gamma: Recovery rate
        max_time: Maximum simulation time
    """
    beta: float
    gamma: float
    max_time: float
    _state: SIRState = field(init=False)
    
    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.beta < 0 or self.gamma < 0:
            raise ValueError("Rates must be non-negative")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
    
    @property
    def r0(self) -> float:
        """Basic reproduction number R₀ = β/γ."""
        if self.gamma == 0:
            return float('inf')
        return self.beta / self.gamma
    
    def initialize(self, population: float, initial_infected: float) -> None:
        """
        Initialise the simulation state.
        
        Args:
            population: Total population N
            initial_infected: Initial number of infected I₀
        """
        if initial_infected > population:
            raise ValueError("initial_infected cannot exceed population")
        self._state = SIRState(
            susceptible=population - initial_infected,
            infected=initial_infected,
            recovered=0.0,
            time=0.0
        )
        logger.debug(f"Initialised SIR: N={population}, I₀={initial_infected}")
    
    def state(self) -> SIRState:
        """Return the current state."""
        return self._state
    
    def step(self, dt: float) -> None:
        """
        Advance the simulation by one time step dt.
        
        Uses Euler's method for numerical integration:
        - dS = -β·S·I/N
        - dI = β·S·I/N - γ·I
        - dR = γ·I
        
        Args:
            dt: Time step size
        """
        s, i, r = self._state.susceptible, self._state.infected, self._state.recovered
        n = s + i + r
        
        # Compute derivatives
        ds_dt = -self.beta * s * i / n
        di_dt = self.beta * s * i / n - self.gamma * i
        dr_dt = self.gamma * i
        
        # Euler step
        new_s = max(0.0, s + dt * ds_dt)
        new_i = max(0.0, i + dt * di_dt)
        new_r = max(0.0, r + dt * dr_dt)
        
        self._state = SIRState(
            susceptible=new_s,
            infected=new_i,
            recovered=new_r,
            time=self._state.time + dt
        )
    
    def is_done(self) -> bool:
        """Check if simulation has completed."""
        return (
            self._state.time >= self.max_time or 
            self._state.infected < 0.5  # Effectively zero infected
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: N-BODY GRAVITATIONAL SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Body:
    """
    A gravitational body with mass, position and velocity.
    
    Uses SI units throughout:
    - Mass: kilograms (kg)
    - Position: metres (m)
    - Velocity: metres per second (m/s)
    
    Attributes:
        mass: Mass in kilograms
        position: Position vector [x, y, z] in metres
        velocity: Velocity vector [vx, vy, vz] in m/s
    """
    mass: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    
    @classmethod
    def create(
        cls, 
        mass: float, 
        pos: list[float], 
        vel: list[float]
    ) -> 'Body':
        """Factory method for creating bodies from lists."""
        return cls(
            mass=mass,
            position=np.array(pos, dtype=np.float64),
            velocity=np.array(vel, dtype=np.float64)
        )
    
    def kinetic_energy(self) -> float:
        """Compute kinetic energy: KE = ½mv²."""
        v_squared = np.sum(self.velocity ** 2)
        return 0.5 * self.mass * v_squared


@dataclass(frozen=True)
class NBodyState:
    """
    State of an N-body gravitational system.
    
    Attributes:
        bodies: List of Body objects
        time: Current simulation time in seconds
    """
    bodies: tuple[Body, ...]
    time: float = 0.0
    
    @property
    def kinetic_energy(self) -> float:
        """Total kinetic energy of the system."""
        return sum(b.kinetic_energy() for b in self.bodies)
    
    @property
    def potential_energy(self) -> float:
        """
        Total gravitational potential energy.
        
        PE = -G Σᵢ Σⱼ>ᵢ (mᵢmⱼ / rᵢⱼ)
        """
        G = 6.67430e-11  # Gravitational constant
        total = 0.0
        for i, b1 in enumerate(self.bodies):
            for b2 in self.bodies[i + 1:]:
                r = np.linalg.norm(b1.position - b2.position)
                if r > 0:
                    total -= G * b1.mass * b2.mass / r
        return total


@dataclass
class NBodySimulation:
    """
    N-body gravitational simulation using the Leapfrog integrator.
    
    The Leapfrog Integrator
    ───────────────────────
    Leapfrog is a symplectic integrator that conserves energy better
    than Euler's method. It advances positions and velocities in a
    "leapfrog" pattern:
    
    1. v(t + dt/2) = v(t) + a(t) · dt/2
    2. x(t + dt) = x(t) + v(t + dt/2) · dt
    3. v(t + dt) = v(t + dt/2) + a(t + dt) · dt/2
    
    Complexity: O(N²) per time step due to pairwise force calculation.
    
    Attributes:
        max_time: Maximum simulation time in seconds
    """
    max_time: float
    _state: NBodyState = field(init=False)
    _G: float = field(default=6.67430e-11, init=False)  # Gravitational constant
    
    def initialize(self, bodies: list[Body]) -> None:
        """
        Initialise the simulation with a list of bodies.
        
        Args:
            bodies: List of Body objects
        """
        self._state = NBodyState(bodies=tuple(bodies), time=0.0)
        logger.debug(f"Initialised N-body with {len(bodies)} bodies")
    
    def state(self) -> NBodyState:
        """Return the current state."""
        return self._state
    
    def _compute_accelerations(self, bodies: tuple[Body, ...]) -> list[NDArray[np.float64]]:
        """
        Compute gravitational accelerations for all bodies.
        
        Uses Newton's law of gravitation:
        F = G·m₁·m₂/r² (magnitude)
        a = F/m = G·m_other/r² (directed towards other body)
        
        Returns:
            List of acceleration vectors for each body
        """
        n = len(bodies)
        accelerations = [np.zeros(3, dtype=np.float64) for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = bodies[j].position - bodies[i].position
                    r_mag = np.linalg.norm(r_vec)
                    if r_mag > 1e-10:  # Avoid division by zero
                        a_mag = self._G * bodies[j].mass / (r_mag ** 2)
                        accelerations[i] += a_mag * (r_vec / r_mag)
        
        return accelerations
    
    def step(self, dt: float) -> None:
        """
        Advance the simulation by one time step using Leapfrog integration.
        
        Args:
            dt: Time step size in seconds
        """
        bodies = list(self._state.bodies)
        n = len(bodies)
        
        # Get current accelerations
        acc = self._compute_accelerations(tuple(bodies))
        
        # Half-step velocity update
        half_velocities = []
        for i in range(n):
            v_half = bodies[i].velocity + 0.5 * dt * acc[i]
            half_velocities.append(v_half)
        
        # Full-step position update
        new_positions = []
        for i in range(n):
            new_pos = bodies[i].position + dt * half_velocities[i]
            new_positions.append(new_pos)
        
        # Create intermediate bodies for acceleration calculation
        intermediate_bodies = tuple(
            Body(
                mass=bodies[i].mass,
                position=new_positions[i],
                velocity=half_velocities[i]
            )
            for i in range(n)
        )
        
        # Get accelerations at new positions
        new_acc = self._compute_accelerations(intermediate_bodies)
        
        # Final velocity update
        new_bodies = []
        for i in range(n):
            final_velocity = half_velocities[i] + 0.5 * dt * new_acc[i]
            new_bodies.append(Body(
                mass=bodies[i].mass,
                position=new_positions[i],
                velocity=final_velocity
            ))
        
        self._state = NBodyState(
            bodies=tuple(new_bodies),
            time=self._state.time + dt
        )
    
    def is_done(self) -> bool:
        """Check if simulation has completed."""
        return self._state.time >= self.max_time


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: GENERIC SIMULATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult(Generic[StateT]):
    """
    Container for simulation results.
    
    Attributes:
        states: List of recorded states
        times: List of simulation times
        wall_time: Actual computation time in seconds
    """
    states: list[StateT]
    times: list[float]
    wall_time: float
    
    def __len__(self) -> int:
        """Return the number of recorded states."""
        return len(self.states)
    
    def __iter__(self) -> Iterator[tuple[float, StateT]]:
        """Iterate over (time, state) pairs."""
        return zip(self.times, self.states)


class SimulationRunner(Generic[StateT]):
    """
    Generic runner for any Simulable object.
    
    The runner handles:
    - Time stepping with configurable dt
    - State recording at specified intervals
    - Progress reporting
    - Timing statistics
    
    Design Pattern: Strategy
    ────────────────────────
    The runner doesn't know what kind of simulation it's running.
    Different simulations (SIR, N-body, etc.) can be plugged in
    without modifying the runner code.
    
    Example Usage
    ─────────────
    >>> sim = SIRSimulation(beta=0.3, gamma=0.1, max_time=100)
    >>> sim.initialize(population=10000, initial_infected=10)
    >>> runner = SimulationRunner(sim)
    >>> result = runner.run(dt=0.1, record_every=10)
    >>> print(f"Simulation completed in {result.wall_time:.3f}s")
    
    Attributes:
        simulation: The Simulable object to run
    """
    
    def __init__(self, simulation: Simulable[StateT]) -> None:
        """
        Initialise the runner with a simulation.
        
        Args:
            simulation: Any object implementing the Simulable protocol
        """
        self.simulation = simulation
    
    def run(
        self, 
        dt: float = 0.01, 
        record_every: int = 1,
        callback: Callable[[StateT], None] | None = None
    ) -> SimulationResult[StateT]:
        """
        Run the simulation to completion.
        
        Args:
            dt: Time step size
            record_every: Record state every N steps (default: every step)
            callback: Optional function called with each recorded state
            
        Returns:
            SimulationResult containing all recorded states and timing info
        """
        states: list[StateT] = []
        times: list[float] = []
        step_count = 0
        
        start_time = time.perf_counter()
        
        # Record initial state
        initial_state = self.simulation.state()
        states.append(initial_state)
        times.append(0.0)
        
        if callback is not None:
            callback(initial_state)
        
        logger.info(f"Starting simulation with dt={dt}")
        
        while not self.simulation.is_done():
            self.simulation.step(dt)
            step_count += 1
            
            if step_count % record_every == 0:
                current_state = self.simulation.state()
                states.append(current_state)
                
                # Extract time from state if available
                if hasattr(current_state, 'time'):
                    times.append(current_state.time)
                else:
                    times.append(step_count * dt)
                
                if callback is not None:
                    callback(current_state)
        
        # Record final state if not already recorded
        final_state = self.simulation.state()
        if states[-1] is not final_state:
            states.append(final_state)
            if hasattr(final_state, 'time'):
                times.append(final_state.time)
            else:
                times.append(step_count * dt)
        
        wall_time = time.perf_counter() - start_time
        
        logger.info(
            f"Simulation completed: {step_count} steps, "
            f"{len(states)} states recorded, {wall_time:.3f}s elapsed"
        )
        
        return SimulationResult(
            states=states,
            times=times,
            wall_time=wall_time
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALISATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sir_result(result: SimulationResult[SIRState], title: str = "SIR Model") -> None:
    """
    Plot SIR simulation results.
    
    Creates a figure with:
    - Time series of S, I, R populations
    - Phase portrait (optional)
    
    Args:
        result: SimulationResult from an SIR simulation
        title: Plot title
    """
    times = [s.time for s in result.states]
    susceptible = [s.susceptible for s in result.states]
    infected = [s.infected for s in result.states]
    recovered = [s.recovered for s in result.states]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(times, susceptible, 'b-', linewidth=2, label='Susceptible')
    ax.plot(times, infected, 'r-', linewidth=2, label='Infected')
    ax.plot(times, recovered, 'g-', linewidth=2, label='Recovered')
    
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sir_simulation.png', dpi=150)
    logger.info("Saved SIR plot to sir_simulation.png")
    plt.show()


def plot_nbody_result(
    result: SimulationResult[NBodyState], 
    title: str = "N-Body Simulation"
) -> None:
    """
    Plot N-body simulation trajectories.
    
    Args:
        result: SimulationResult from an N-body simulation
        title: Plot title
    """
    if not result.states:
        logger.warning("No states to plot")
        return
    
    n_bodies = len(result.states[0].bodies)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot trajectory for each body
    colours = plt.cm.viridis(np.linspace(0, 1, n_bodies))
    
    for body_idx in range(n_bodies):
        x_traj = [state.bodies[body_idx].position[0] for state in result.states]
        y_traj = [state.bodies[body_idx].position[1] for state in result.states]
        
        ax.plot(x_traj, y_traj, '-', color=colours[body_idx], 
                linewidth=0.5, alpha=0.7)
        ax.plot(x_traj[0], y_traj[0], 'o', color=colours[body_idx], 
                markersize=8, label=f'Body {body_idx}')
        ax.plot(x_traj[-1], y_traj[-1], 's', color=colours[body_idx], 
                markersize=6)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'{title} — Trajectories (XY Projection)', fontsize=14)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nbody_simulation.png', dpi=150)
    logger.info("Saved N-body plot to nbody_simulation.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_sir() -> None:
    """Demonstration: SIR Epidemic Simulation."""
    logger.info("=" * 60)
    logger.info("DEMO: SIR Epidemic Model")
    logger.info("=" * 60)
    
    # Parameters for an epidemic with R₀ = 3 (similar to SARS-CoV-2 wild type)
    sim = SIRSimulation(
        beta=0.3,    # 30% transmission rate per day
        gamma=0.1,   # Recovery in ~10 days
        max_time=200
    )
    sim.initialize(population=10_000, initial_infected=10)
    
    logger.info(f"R₀ = {sim.r0:.2f}")
    logger.info(f"Population: {sim.state().total:,.0f}")
    logger.info(f"Initial infected: {sim.state().infected:.0f}")
    
    # Run the simulation
    runner: SimulationRunner[SIRState] = SimulationRunner(sim)
    result = runner.run(dt=0.1, record_every=10)
    
    logger.info(f"Simulation completed in {result.wall_time:.3f}s")
    logger.info(f"Total steps recorded: {len(result)}")
    
    # Final statistics
    final = result.states[-1]
    logger.info(f"Final state (t={final.time:.1f}):")
    logger.info(f"  Susceptible: {final.susceptible:,.0f} ({100*final.susceptible/final.total:.1f}%)")
    logger.info(f"  Infected: {final.infected:,.0f}")
    logger.info(f"  Recovered: {final.recovered:,.0f} ({100*final.recovered/final.total:.1f}%)")
    
    # Peak of infection
    peak_infected = max(result.states, key=lambda s: s.infected)
    logger.info(f"Peak infection: {peak_infected.infected:,.0f} at t={peak_infected.time:.1f}")


def demo_nbody() -> None:
    """Demonstration: Simple N-body Simulation."""
    logger.info("=" * 60)
    logger.info("DEMO: N-Body Gravitational Simulation")
    logger.info("=" * 60)
    
    # Sun-Earth-Moon system (simplified)
    # Units: metres, kilograms, seconds
    bodies = [
        Body.create(
            mass=1.989e30,  # Sun
            pos=[0, 0, 0],
            vel=[0, 0, 0]
        ),
        Body.create(
            mass=5.972e24,  # Earth
            pos=[1.496e11, 0, 0],  # 1 AU
            vel=[0, 29780, 0]  # Orbital velocity
        ),
        Body.create(
            mass=7.342e22,  # Moon
            pos=[1.496e11 + 3.844e8, 0, 0],
            vel=[0, 29780 + 1022, 0]  # Relative velocity to Earth
        ),
    ]
    
    sim = NBodySimulation(max_time=365 * 24 * 3600)  # One year
    sim.initialize(bodies)
    
    runner: SimulationRunner[NBodyState] = SimulationRunner(sim)
    
    # Run for 1 year with 1-hour time step
    result = runner.run(dt=3600, record_every=24)  # Record daily
    
    logger.info(f"Simulation completed in {result.wall_time:.3f}s")
    logger.info(f"Days simulated: {len(result)}")
    
    # Check energy conservation
    initial_energy = result.states[0].kinetic_energy + result.states[0].potential_energy
    final_energy = result.states[-1].kinetic_energy + result.states[-1].potential_energy
    energy_drift = abs(final_energy - initial_energy) / abs(initial_energy) * 100
    
    logger.info("Energy conservation check:")
    logger.info(f"  Initial total energy: {initial_energy:.3e} J")
    logger.info(f"  Final total energy: {final_energy:.3e} J")
    logger.info(f"  Drift: {energy_drift:.4f}%")


def demo_generic_runner() -> None:
    """Demonstrate that the runner works with ANY Simulable."""
    logger.info("=" * 60)
    logger.info("DEMO: Generic Runner with Different Simulations")
    logger.info("=" * 60)
    
    # Define a completely new simulation type
    @dataclass
    class ExponentialDecay:
        """Exponential decay: dx/dt = -λx."""
        decay_rate: float
        max_time: float
        _value: float = 100.0
        _time: float = 0.0
        
        def state(self) -> float:
            return self._value
        
        def step(self, dt: float) -> None:
            self._value *= np.exp(-self.decay_rate * dt)
            self._time += dt
        
        def is_done(self) -> bool:
            return self._time >= self.max_time
    
    # The runner works without modification!
    decay_sim = ExponentialDecay(decay_rate=0.1, max_time=50)
    runner: SimulationRunner[float] = SimulationRunner(decay_sim)
    result = runner.run(dt=0.1)
    
    expected = 100 * np.exp(-0.1 * 50)
    actual = result.states[-1]
    relative_error = abs(actual - expected) / expected * 100
    
    logger.info("ExponentialDecay simulation:")
    logger.info(f"  Initial value: {result.states[0]:.2f}")
    logger.info(f"  Final value: {actual:.2f}")
    logger.info(f"  Expected: {expected:.2f}")
    logger.info(f"  Relative error: {relative_error:.4f}%")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_sir()
    print()
    demo_generic_runner()
    print()
    # demo_nbody()  # Commented for speed — uncomment for full demo


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 1: SEIR Model                                                        ║
║                                                                               ║
║ Extend the SIR model to SEIR (add the Exposed compartment):                   ║
║                                                                               ║
║     β·S·I/N         σ·E           γ·I                                         ║
║ S ──────────▶ E ──────────▶ I ──────────▶ R                                   ║
║                                                                               ║
║ Where σ = incubation rate (1/σ = latency period)                              ║
║                                                                               ║
║ Implement SEIRState and SEIRSimulation.                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 2: Adaptive Time Stepping                                            ║
║                                                                               ║
║ Modify the runner to support adaptive time stepping:                          ║
║ - If estimated error is large, decrease dt                                    ║
║ - If error is small, increase dt for efficiency                               ║
║                                                                               ║
║ Hint: Use embedded Runge-Kutta (RK45) or compare two methods.                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 3: N-Body Parallelisation                                            ║
║                                                                               ║
║ The O(N²) force calculation can be parallelised.                              ║
║ Use numba or multiprocessing to accelerate _compute_accelerations.            ║
║                                                                               ║
║ Measure speedup for N = 100, 500, 1000 bodies.                                ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="02UNIT Lab: Simulation Framework with Protocols and Generics"
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
        print("  WEEK 2 LAB: SIMULATION FRAMEWORK")
        print("═" * 60 + "\n")
        print("Use --demo to run demonstrations")
        print("Use -v for verbose output")
        print("\nExercises to complete:")
        print("  1. Implement the SEIR model")
        print("  2. Add adaptive time stepping")
        print("  3. Parallelise N-body force calculation")
        print("═" * 60)


if __name__ == "__main__":
    main()
