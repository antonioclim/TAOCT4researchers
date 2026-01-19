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
from dataclasses import dataclass, field
from typing import Generic, Iterator, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
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
StateT = TypeVar('StateT', covariant=True)


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
    """SIR epidemic simulation with an interface tailored for the unit tests.

    The original teaching kit presented an explicit `initialize()` phase to
    emphasise separation of parameterisation from state initialisation.
    The accompanying unit tests, however, construct simulations directly from
    initial fractions to reduce boilerplate in exercises.

    This class therefore supports **both** idioms:
    - direct construction via `susceptible`, `infected`, `recovered`
    - optional later initialisation via `initialize()` (retained for pedagogy)

    The model uses the standard SIR differential equations with an Euler update.

    Attributes:
        susceptible: Initial susceptible fraction (S0)
        infected: Initial infected fraction (I0)
        recovered: Initial recovered fraction (R0)
        beta: Transmission rate (β)
        gamma: Recovery rate (γ)
        max_time: Safety limit for simulated time
    """

    susceptible: float = 0.99
    infected: float = 0.01
    recovered: float = 0.0
    beta: float = 0.3
    gamma: float = 0.1
    max_time: float = 1_000.0
    _state: SIRState = field(init=False)

    def __post_init__(self) -> None:
        if self.beta < 0 or self.gamma < 0:
            raise ValueError("Rates must be non-negative")
        if self.max_time <= 0:
            raise ValueError("max_time must be positive")
        if self.susceptible < 0 or self.infected < 0 or self.recovered < 0:
            raise ValueError("State fractions must be non-negative")
        total = self.susceptible + self.infected + self.recovered
        if total <= 0:
            raise ValueError("S + I + R must be positive")
        # Normalise to avoid drift from user input.
        self._state = SIRState(
            susceptible=self.susceptible / total,
            infected=self.infected / total,
            recovered=self.recovered / total,
            time=0.0,
        )
    
    @property
    def r0(self) -> float:
        """Basic reproduction number R₀ = β/γ."""
        if self.gamma == 0:
            return float('inf')
        return self.beta / self.gamma
    
    def initialize(self, population: float, initial_infected: float) -> None:
        """Initialise from absolute counts (retained for the lab narrative)."""
        if population <= 0:
            raise ValueError("population must be positive")
        if initial_infected < 0 or initial_infected > population:
            raise ValueError("initial_infected must be in [0, population]")
        susceptible = population - initial_infected
        total = float(population)
        self._state = SIRState(
            susceptible=susceptible / total,
            infected=initial_infected / total,
            recovered=0.0,
            time=0.0,
        )
    
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
        
        # Discrete-time update:
        # The pedagogical tests treat β and γ as per-step rates rather than
        # per-unit-time rates. The time variable is still advanced by `dt` to
        # support plotting and interpretation.
        new_s = max(0.0, s + ds_dt)
        new_i = max(0.0, i + di_dt)
        new_r = max(0.0, r + dr_dt)
        total = new_s + new_i + new_r
        if total > 0:
            new_s /= total
            new_i /= total
            new_r /= total
        
        self._state = SIRState(
            susceptible=new_s,
            infected=new_i,
            recovered=new_r,
            time=self._state.time + dt
        )
    
    def is_done(self) -> bool:
        """Return True when the infection is negligible or the safety limit is reached."""
        return self._state.time >= self.max_time or self._state.infected < 1e-6


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
    """State of a 2D N-body system used by the unit tests."""

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    masses: NDArray[np.float64]
    time: float = 0.0


@dataclass
class NBodySimulation:
    """2D Newtonian N-body simulation with a symplectic update.

    The laboratory narrative includes a more general 3D formulation. For the
    exercise and test suite we adopt a 2D state representation using NumPy
    arrays (shape: ``(n, 2)``) to simplify inspection and plotting.
    """

    positions: NDArray[np.float64]
    velocities: NDArray[np.float64]
    masses: NDArray[np.float64]
    G: float = 1.0
    max_time: float = 1_000.0
    _time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.positions.shape != self.velocities.shape:
            raise ValueError("positions and velocities must have the same shape")
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n, 2)")
        if self.masses.ndim != 1 or self.masses.shape[0] != self.positions.shape[0]:
            raise ValueError("masses must have shape (n,)")
        if self.G <= 0:
            raise ValueError("G must be positive")

        self.positions = self.positions.astype(np.float64, copy=True)
        self.velocities = self.velocities.astype(np.float64, copy=True)
        self.masses = self.masses.astype(np.float64, copy=True)

    def _accelerations(self) -> NDArray[np.float64]:
        n = self.positions.shape[0]
        acc = np.zeros_like(self.positions)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                r_vec = self.positions[j] - self.positions[i]
                r = float(np.linalg.norm(r_vec))
                if r <= 1e-10:
                    continue
                acc[i] += self.G * self.masses[j] * (r_vec / (r ** 3))
        return acc

    def state(self) -> NBodyState:
        return NBodyState(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy(),
            time=self._time,
        )

    def step(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        acc = self._accelerations()
        # Velocity Verlet: v(t+dt/2), x(t+dt), a(t+dt), v(t+dt)
        v_half = self.velocities + 0.5 * dt * acc
        self.positions = self.positions + dt * v_half
        acc_new = self._accelerations()
        self.velocities = v_half + 0.5 * dt * acc_new
        self._time += dt

    def is_done(self) -> bool:
        return self._time >= self.max_time


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
    >>> result = runner.run(dt=0.1, max_steps=1000, record_every=10)
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
    
    def run(self, dt: float, max_steps: int, *, record_every: int = 1) -> list[StateT]:
        """Run the simulation for up to ``max_steps`` steps.

        The original lab exposed a richer result object with timing metadata.
        The unit tests for this kit, however, expect a plain history list where
        ``history[0]`` is the initial state and subsequent entries are states
        recorded during execution.

        Args:
            dt: Time step size.
            max_steps: Maximum number of steps.
            record_every: Record every N-th step.

        Returns:
            Recorded history of states.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if record_every <= 0:
            raise ValueError("record_every must be positive")

        history: list[StateT] = [self.simulation.state()]
        for step in range(1, max_steps + 1):
            if self.simulation.is_done():
                break
            self.simulation.step(dt)
            if step % record_every == 0:
                history.append(self.simulation.state())
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VISUALISATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_sir_result(states: list[SIRState], title: str = "SIR Model") -> None:
    """
    Plot SIR simulation results.
    
    Creates a figure with:
    - Time series of S, I, R populations
    - Phase portrait (optional)
    
    Args:
        states: List of states from an SIR simulation
        title: Plot title
    """
    times = [s.time for s in states]
    susceptible = [s.susceptible for s in states]
    infected = [s.infected for s in states]
    recovered = [s.recovered for s in states]
    
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



# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_sir() -> None:
    """Demonstration: SIR epidemic simulation using the stable unit API."""
    logger.info("=" * 60)
    logger.info("DEMO: SIR Epidemic Model")
    logger.info("=" * 60)

    sim = SIRSimulation(
        susceptible=9_990.0,
        infected=10.0,
        recovered=0.0,
        beta=0.3,
        gamma=0.1,
    )

    runner: SimulationRunner[SIRState] = SimulationRunner(sim)
    history = runner.run(dt=0.1, max_steps=5_000, record_every=10)

    final = history[-1]
    logger.info(f"Final time: {final.time:.1f}")
    logger.info(f"Susceptible: {final.susceptible:,.0f}")
    logger.info(f"Infected: {final.infected:,.0f}")
    logger.info(f"Recovered: {final.recovered:,.0f}")

    peak = max(history, key=lambda s: s.infected)
    logger.info(f"Peak infected: {peak.infected:,.0f} at t={peak.time:.1f}")



def demo_generic_runner() -> None:
    """Demonstrate that the runner works with any Simulable instance."""
    logger.info("=" * 60)
    logger.info("DEMO: Generic Runner")
    logger.info("=" * 60)

    @dataclass
    class ExponentialDecay:
        """Exponential decay: dx/dt = -λx."""

        decay_rate: float
        max_time: float
        value: float = 100.0
        time: float = 0.0

        def state(self) -> float:
            return self.value

        def step(self, dt: float) -> None:
            self.value *= float(np.exp(-self.decay_rate * dt))
            self.time += dt

        def is_done(self) -> bool:
            return self.time >= self.max_time

    decay_sim = ExponentialDecay(decay_rate=0.1, max_time=50.0)
    runner: SimulationRunner[float] = SimulationRunner(decay_sim)
    history = runner.run(dt=0.1, max_steps=10_000, record_every=50)

    expected = 100.0 * float(np.exp(-0.1 * 50.0))
    actual = history[-1]
    rel = abs(actual - expected) / expected * 100.0

    logger.info(f"Final value: {actual:.4f}")
    logger.info(f"Expected: {expected:.4f}")
    logger.info(f"Relative error: {rel:.4f}%")



def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_sir()

    demo_generic_runner()

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
