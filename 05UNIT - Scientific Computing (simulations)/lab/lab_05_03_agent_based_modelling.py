#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab 3: Agent-Based Modelling
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Agent-based models (ABMs) simulate systems composed of autonomous individuals
whose interactions produce emergent collective behaviour. Unlike differential
equations that describe aggregate dynamics, ABMs capture heterogeneity, spatial
structure and adaptive behaviour—enabling insights into phenomena from urban
segregation to flocking birds.

PREREQUISITES
─────────────
- Week 4: Graph structures for agent interactions
- Python: Object-oriented programming, NumPy arrays
- Concepts: Randomness, spatial reasoning

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Design agent classes with state and behaviour
2. Implement Schelling segregation with emergence metrics
3. Create Boids flocking simulation with spatial rules

ESTIMATED TIME
──────────────
- Reading: 40 minutes
- Coding: 80 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
numpy>=1.24, matplotlib>=3.7

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Iterator

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
IntArray = NDArray[np.signedinteger]
Position = tuple[int, int] | FloatArray


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BASE AGENT FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════


class Agent(ABC):
    """Abstract base class for agents.
    
    Agents are autonomous entities with state and behaviour rules.
    Subclasses must implement the update() method.
    
    Attributes:
        agent_id: Unique identifier
        position: Current position (grid cell or continuous)
    """
    
    _id_counter: int = 0
    
    def __init__(self, position: Position) -> None:
        """Initialise agent.
        
        Args:
            position: Initial position
        """
        Agent._id_counter += 1
        self.agent_id = Agent._id_counter
        self.position = position
    
    @abstractmethod
    def update(self, environment: "Environment") -> None:
        """Update agent state based on environment.
        
        Args:
            environment: The environment containing all agents
        """
        ...
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, pos={self.position})"


class Environment(ABC):
    """Abstract base class for environments.
    
    Environments contain agents and manage their interactions.
    """
    
    @abstractmethod
    def get_neighbours(self, agent: Agent) -> list[Agent]:
        """Get neighbouring agents.
        
        Args:
            agent: The focal agent
            
        Returns:
            List of neighbouring agents
        """
        ...
    
    @abstractmethod
    def step(self) -> dict:
        """Advance simulation by one time step.
        
        Returns:
            Dictionary of metrics for this step
        """
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SCHELLING SEGREGATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SchellingAgent:
    """Agent in Schelling's segregation model.
    
    Attributes:
        agent_type: Type identifier (0 for empty, 1 or 2 for agent types)
        row: Grid row position
        col: Grid column position
        threshold: Minimum fraction of same-type neighbours to be happy
    """
    agent_type: int
    row: int
    col: int
    threshold: float = 0.3
    
    def is_happy(self, same_type_fraction: float) -> bool:
        """Check if agent is satisfied with neighbourhood.
        
        Args:
            same_type_fraction: Fraction of neighbours that are same type
            
        Returns:
            True if satisfied (won't move)
        """
        return same_type_fraction >= self.threshold


class SchellingModel:
    """Schelling's spatial segregation model.
    
    Agents of two types occupy cells on a grid. Each agent has a threshold
    for the minimum fraction of same-type neighbours to be "happy". Unhappy
    agents move to random empty cells. Despite mild individual preferences,
    strong collective segregation emerges.
    
    Attributes:
        grid_size: Width/height of square grid
        empty_ratio: Fraction of cells left empty
        threshold: Agent happiness threshold
        grid: 2D array (0=empty, 1=type A, 2=type B)
        
    Examples:
        >>> model = SchellingModel(grid_size=20, empty_ratio=0.2, threshold=0.3)
        >>> initial_seg = model.segregation_index()
        >>> for _ in range(50):
        ...     model.step()
        >>> final_seg = model.segregation_index()
        >>> final_seg > initial_seg  # Segregation increases
        True
    """
    
    def __init__(
        self,
        grid_size: int = 50,
        empty_ratio: float = 0.2,
        threshold: float = 0.3,
        type_ratio: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise Schelling model.
        
        Args:
            grid_size: Width and height of grid
            empty_ratio: Fraction of cells to leave empty
            threshold: Happiness threshold for all agents
            type_ratio: Fraction of non-empty cells that are type 1
            rng: Random generator for reproducibility
        """
        self.grid_size = grid_size
        self.empty_ratio = empty_ratio
        self.threshold = threshold
        self.type_ratio = type_ratio
        self.rng = rng or np.random.default_rng()
        
        # Initialise grid
        self.grid = self._create_random_grid()
        self.step_count = 0
        
        logger.debug(
            f"Created Schelling model: {grid_size}x{grid_size}, "
            f"threshold={threshold}, empty_ratio={empty_ratio}"
        )
    
    def _create_random_grid(self) -> IntArray:
        """Create randomly populated grid."""
        n_cells = self.grid_size ** 2
        n_empty = int(n_cells * self.empty_ratio)
        n_occupied = n_cells - n_empty
        n_type1 = int(n_occupied * self.type_ratio)
        n_type2 = n_occupied - n_type1
        
        # Create flat array and shuffle
        cells = np.array(
            [0] * n_empty + [1] * n_type1 + [2] * n_type2,
            dtype=np.int32,
        )
        self.rng.shuffle(cells)
        
        return cells.reshape((self.grid_size, self.grid_size))
    
    def get_neighbours(self, row: int, col: int) -> list[int]:
        """Get types of neighbours (Moore neighbourhood).
        
        Args:
            row: Cell row
            col: Cell column
            
        Returns:
            List of neighbour types (excluding empties and self)
        """
        neighbours = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                nc = col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    cell_type = self.grid[nr, nc]
                    if cell_type != 0:  # Not empty
                        neighbours.append(cell_type)
        return neighbours
    
    def is_happy(self, row: int, col: int) -> bool:
        """Check if agent at (row, col) is happy.
        
        Args:
            row: Cell row
            col: Cell column
            
        Returns:
            True if agent is satisfied or cell is empty
        """
        agent_type = self.grid[row, col]
        if agent_type == 0:
            return True  # Empty cells don't move
        
        neighbours = self.get_neighbours(row, col)
        if len(neighbours) == 0:
            return True  # No neighbours = happy
        
        same_type = sum(1 for n in neighbours if n == agent_type)
        fraction = same_type / len(neighbours)
        
        return fraction >= self.threshold
    
    def find_empty_cells(self) -> list[tuple[int, int]]:
        """Find all empty cells on grid."""
        empties = np.argwhere(self.grid == 0)
        return [(int(r), int(c)) for r, c in empties]
    
    def find_unhappy_agents(self) -> list[tuple[int, int]]:
        """Find all unhappy agents."""
        unhappy = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] != 0 and not self.is_happy(row, col):
                    unhappy.append((row, col))
        return unhappy
    
    def step(self) -> dict:
        """Execute one step: move all unhappy agents.
        
        Returns:
            Dictionary with step metrics
        """
        unhappy = self.find_unhappy_agents()
        empty_cells = self.find_empty_cells()
        
        if len(unhappy) == 0 or len(empty_cells) == 0:
            self.step_count += 1
            return {
                "step": self.step_count,
                "moved": 0,
                "happy_fraction": self.happy_fraction(),
                "segregation": self.segregation_index(),
            }
        
        # Shuffle unhappy agents for random order
        self.rng.shuffle(unhappy)
        
        moved = 0
        for row, col in unhappy:
            if len(empty_cells) == 0:
                break
            
            # Move to random empty cell
            new_idx = self.rng.integers(len(empty_cells))
            new_row, new_col = empty_cells.pop(new_idx)
            
            # Execute move
            self.grid[new_row, new_col] = self.grid[row, col]
            self.grid[row, col] = 0
            empty_cells.append((row, col))
            
            moved += 1
        
        self.step_count += 1
        
        return {
            "step": self.step_count,
            "moved": moved,
            "happy_fraction": self.happy_fraction(),
            "segregation": self.segregation_index(),
        }
    
    def happy_fraction(self) -> float:
        """Compute fraction of agents that are happy."""
        total = 0
        happy = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] != 0:
                    total += 1
                    if self.is_happy(row, col):
                        happy += 1
        return happy / total if total > 0 else 1.0
    
    def segregation_index(self) -> float:
        """Compute average same-type neighbour fraction.
        
        Higher values indicate more segregation.
        
        Returns:
            Mean fraction of same-type neighbours across all agents
        """
        total_fraction = 0.0
        n_agents = 0
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                agent_type = self.grid[row, col]
                if agent_type == 0:
                    continue
                
                neighbours = self.get_neighbours(row, col)
                if len(neighbours) > 0:
                    same = sum(1 for n in neighbours if n == agent_type)
                    total_fraction += same / len(neighbours)
                else:
                    total_fraction += 1.0  # No neighbours = "segregated"
                
                n_agents += 1
        
        return total_fraction / n_agents if n_agents > 0 else 0.0
    
    def run(self, max_steps: int = 100) -> list[dict]:
        """Run simulation until equilibrium or max steps.
        
        Args:
            max_steps: Maximum steps to run
            
        Returns:
            List of metrics dictionaries for each step
        """
        history = []
        
        for _ in range(max_steps):
            metrics = self.step()
            history.append(metrics)
            
            if metrics["moved"] == 0:
                logger.info(f"Equilibrium reached at step {metrics['step']}")
                break
        
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: BOIDS FLOCKING MODEL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Boid:
    """A single boid (bird-like object) in the flocking simulation.
    
    Attributes:
        position: 2D position [x, y]
        velocity: 2D velocity [vx, vy]
        boid_id: Unique identifier
    """
    position: FloatArray
    velocity: FloatArray
    boid_id: int = field(default_factory=lambda: Boid._get_next_id())
    
    _id_counter: int = 0
    
    @classmethod
    def _get_next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter
    
    @property
    def speed(self) -> float:
        """Current speed (velocity magnitude)."""
        return float(np.linalg.norm(self.velocity))
    
    def distance_to(self, other: "Boid") -> float:
        """Euclidean distance to another boid."""
        return float(np.linalg.norm(self.position - other.position))


class BoidsSimulation:
    """Boids flocking simulation.
    
    Implements Reynolds' (1987) three rules for flocking:
    1. Separation: Steer away from nearby boids
    2. Alignment: Match velocity with nearby boids
    3. Cohesion: Steer toward centre of nearby boids
    
    Attributes:
        width: Environment width
        height: Environment height
        boids: List of Boid objects
        visual_range: Distance within which boids see each other
        min_distance: Minimum desired separation
        
    Examples:
        >>> sim = BoidsSimulation(width=800, height=600, n_boids=50)
        >>> initial_pol = sim.polarisation()
        >>> for _ in range(100):
        ...     sim.step()
        >>> final_pol = sim.polarisation()
        >>> final_pol > initial_pol  # Alignment increases
        True
    """
    
    def __init__(
        self,
        width: float = 800,
        height: float = 600,
        n_boids: int = 100,
        visual_range: float = 75.0,
        min_distance: float = 20.0,
        max_speed: float = 5.0,
        separation_weight: float = 1.0,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        edge_margin: float = 50.0,
        turn_factor: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise boids simulation.
        
        Args:
            width: Environment width
            height: Environment height
            n_boids: Number of boids
            visual_range: Neighbour detection radius
            min_distance: Separation distance
            max_speed: Maximum boid speed
            separation_weight: Weight for separation rule
            alignment_weight: Weight for alignment rule
            cohesion_weight: Weight for cohesion rule
            edge_margin: Distance from edge to start turning
            turn_factor: Strength of edge avoidance
            rng: Random generator
        """
        self.width = width
        self.height = height
        self.visual_range = visual_range
        self.min_distance = min_distance
        self.max_speed = max_speed
        self.min_speed = max_speed * 0.5
        
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        
        self.edge_margin = edge_margin
        self.turn_factor = turn_factor
        
        self.rng = rng or np.random.default_rng()
        self.step_count = 0
        
        # Create boids with random positions and velocities
        self.boids: list[Boid] = []
        for _ in range(n_boids):
            pos = np.array([
                self.rng.uniform(0, width),
                self.rng.uniform(0, height),
            ])
            vel = np.array([
                self.rng.uniform(-max_speed, max_speed),
                self.rng.uniform(-max_speed, max_speed),
            ])
            self.boids.append(Boid(position=pos, velocity=vel))
        
        logger.debug(f"Created boids simulation with {n_boids} boids")
    
    def get_neighbours(self, boid: Boid) -> list[Boid]:
        """Get boids within visual range.
        
        Args:
            boid: The focal boid
            
        Returns:
            List of neighbouring boids
        """
        neighbours = []
        for other in self.boids:
            if other.boid_id == boid.boid_id:
                continue
            if boid.distance_to(other) < self.visual_range:
                neighbours.append(other)
        return neighbours
    
    def separation(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Compute separation steering (avoid crowding).
        
        Args:
            boid: Focal boid
            neighbours: Nearby boids
            
        Returns:
            Steering vector away from too-close neighbours
        """
        steer = np.zeros(2)
        for other in neighbours:
            dist = boid.distance_to(other)
            if dist < self.min_distance and dist > 0:
                # Vector pointing away, weighted by closeness
                diff = boid.position - other.position
                steer += diff / dist
        return steer
    
    def alignment(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Compute alignment steering (match velocity).
        
        Args:
            boid: Focal boid
            neighbours: Nearby boids
            
        Returns:
            Steering vector toward average neighbour velocity
        """
        if len(neighbours) == 0:
            return np.zeros(2)
        
        avg_vel = np.mean([n.velocity for n in neighbours], axis=0)
        return avg_vel - boid.velocity
    
    def cohesion(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Compute cohesion steering (steer toward centre).
        
        Args:
            boid: Focal boid
            neighbours: Nearby boids
            
        Returns:
            Steering vector toward centre of mass of neighbours
        """
        if len(neighbours) == 0:
            return np.zeros(2)
        
        centre = np.mean([n.position for n in neighbours], axis=0)
        return (centre - boid.position) / 100  # Scale down
    
    def edge_avoidance(self, boid: Boid) -> FloatArray:
        """Compute edge avoidance steering.
        
        Args:
            boid: Focal boid
            
        Returns:
            Steering vector away from edges
        """
        steer = np.zeros(2)
        
        if boid.position[0] < self.edge_margin:
            steer[0] += self.turn_factor
        elif boid.position[0] > self.width - self.edge_margin:
            steer[0] -= self.turn_factor
        
        if boid.position[1] < self.edge_margin:
            steer[1] += self.turn_factor
        elif boid.position[1] > self.height - self.edge_margin:
            steer[1] -= self.turn_factor
        
        return steer
    
    def update_boid(self, boid: Boid) -> None:
        """Update single boid's velocity and position.
        
        Args:
            boid: Boid to update
        """
        neighbours = self.get_neighbours(boid)
        
        # Compute steering forces
        sep = self.separation(boid, neighbours) * self.separation_weight
        ali = self.alignment(boid, neighbours) * self.alignment_weight
        coh = self.cohesion(boid, neighbours) * self.cohesion_weight
        edge = self.edge_avoidance(boid)
        
        # Update velocity
        boid.velocity += sep + ali + coh + edge
        
        # Limit speed
        speed = boid.speed
        if speed > self.max_speed:
            boid.velocity = boid.velocity / speed * self.max_speed
        elif speed < self.min_speed and speed > 0:
            boid.velocity = boid.velocity / speed * self.min_speed
        
        # Update position
        boid.position += boid.velocity
        
        # Wrap around edges (or bounce)
        boid.position[0] = boid.position[0] % self.width
        boid.position[1] = boid.position[1] % self.height
    
    def step(self) -> dict:
        """Advance simulation by one time step.
        
        Returns:
            Dictionary with step metrics
        """
        # Update all boids
        for boid in self.boids:
            self.update_boid(boid)
        
        self.step_count += 1
        
        return {
            "step": self.step_count,
            "polarisation": self.polarisation(),
            "avg_neighbours": self.average_neighbours(),
            "clustering": self.clustering_coefficient(),
        }
    
    def polarisation(self) -> float:
        """Compute alignment/polarisation metric.
        
        Polarisation is the magnitude of the average normalised velocity.
        1.0 = all boids moving same direction
        0.0 = random directions
        
        Returns:
            Polarisation value in [0, 1]
        """
        velocities = np.array([b.velocity for b in self.boids])
        speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
        
        # Avoid division by zero
        speeds = np.maximum(speeds, 1e-10)
        normalised = velocities / speeds
        
        avg_direction = np.mean(normalised, axis=0)
        return float(np.linalg.norm(avg_direction))
    
    def average_neighbours(self) -> float:
        """Compute average number of neighbours per boid."""
        total = 0
        for boid in self.boids:
            total += len(self.get_neighbours(boid))
        return total / len(self.boids) if self.boids else 0.0
    
    def clustering_coefficient(self) -> float:
        """Compute spatial clustering coefficient.
        
        Based on variance of local density.
        """
        densities = []
        for boid in self.boids:
            n_neighbours = len(self.get_neighbours(boid))
            densities.append(n_neighbours)
        
        if len(densities) == 0:
            return 0.0
        
        mean_d = np.mean(densities)
        if mean_d == 0:
            return 0.0
        
        # Coefficient of variation as clustering proxy
        return float(np.std(densities) / mean_d)
    
    def run(self, n_steps: int = 100) -> list[dict]:
        """Run simulation for specified steps.
        
        Args:
            n_steps: Number of steps to run
            
        Returns:
            List of metrics dictionaries
        """
        history = []
        for _ in range(n_steps):
            metrics = self.step()
            history.append(metrics)
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_schelling() -> None:
    """Demonstrate Schelling segregation model."""
    logger.info("=" * 60)
    logger.info("DEMO: Schelling Segregation Model")
    logger.info("=" * 60)
    
    model = SchellingModel(
        grid_size=30,
        empty_ratio=0.2,
        threshold=0.3,
        rng=np.random.default_rng(42),
    )
    
    initial_seg = model.segregation_index()
    initial_happy = model.happy_fraction()
    
    logger.info(f"Initial state:")
    logger.info(f"  Segregation index: {initial_seg:.3f}")
    logger.info(f"  Happy fraction: {initial_happy:.3f}")
    
    history = model.run(max_steps=100)
    
    final_seg = model.segregation_index()
    final_happy = model.happy_fraction()
    
    logger.info(f"\nFinal state (after {model.step_count} steps):")
    logger.info(f"  Segregation index: {final_seg:.3f}")
    logger.info(f"  Happy fraction: {final_happy:.3f}")
    logger.info(f"  Segregation increase: {(final_seg - initial_seg):.3f}")
    
    # Show progression
    logger.info("\nProgression (every 10 steps):")
    for i in range(0, len(history), 10):
        h = history[i]
        logger.info(
            f"  Step {h['step']:>3}: seg={h['segregation']:.3f}, "
            f"happy={h['happy_fraction']:.3f}, moved={h['moved']}"
        )


def demo_schelling_threshold() -> None:
    """Demonstrate effect of threshold on segregation."""
    logger.info("=" * 60)
    logger.info("DEMO: Threshold Effect on Segregation")
    logger.info("=" * 60)
    
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    for threshold in thresholds:
        model = SchellingModel(
            grid_size=30,
            threshold=threshold,
            rng=np.random.default_rng(42),
        )
        
        initial_seg = model.segregation_index()
        model.run(max_steps=100)
        final_seg = model.segregation_index()
        
        logger.info(
            f"Threshold {threshold:.1f}: "
            f"initial={initial_seg:.3f} → final={final_seg:.3f} "
            f"(Δ={final_seg - initial_seg:+.3f})"
        )


def demo_boids() -> None:
    """Demonstrate Boids flocking simulation."""
    logger.info("=" * 60)
    logger.info("DEMO: Boids Flocking Simulation")
    logger.info("=" * 60)
    
    sim = BoidsSimulation(
        width=800,
        height=600,
        n_boids=100,
        visual_range=75,
        rng=np.random.default_rng(42),
    )
    
    initial_pol = sim.polarisation()
    initial_neighbours = sim.average_neighbours()
    
    logger.info(f"Initial state:")
    logger.info(f"  Polarisation: {initial_pol:.3f}")
    logger.info(f"  Avg neighbours: {initial_neighbours:.1f}")
    
    history = sim.run(n_steps=200)
    
    final_pol = sim.polarisation()
    final_neighbours = sim.average_neighbours()
    
    logger.info(f"\nFinal state (after {sim.step_count} steps):")
    logger.info(f"  Polarisation: {final_pol:.3f}")
    logger.info(f"  Avg neighbours: {final_neighbours:.1f}")
    
    # Show progression
    logger.info("\nProgression (every 25 steps):")
    for i in range(0, len(history), 25):
        h = history[i]
        logger.info(
            f"  Step {h['step']:>3}: pol={h['polarisation']:.3f}, "
            f"neighbours={h['avg_neighbours']:.1f}"
        )


def demo_boids_parameters() -> None:
    """Demonstrate effect of parameters on flocking."""
    logger.info("=" * 60)
    logger.info("DEMO: Boids Parameter Effects")
    logger.info("=" * 60)
    
    configs = [
        {"name": "Balanced", "sep": 1.0, "ali": 1.0, "coh": 1.0},
        {"name": "High separation", "sep": 3.0, "ali": 1.0, "coh": 1.0},
        {"name": "High alignment", "sep": 1.0, "ali": 3.0, "coh": 1.0},
        {"name": "High cohesion", "sep": 1.0, "ali": 1.0, "coh": 3.0},
    ]
    
    for config in configs:
        sim = BoidsSimulation(
            n_boids=50,
            separation_weight=config["sep"],
            alignment_weight=config["ali"],
            cohesion_weight=config["coh"],
            rng=np.random.default_rng(42),
        )
        
        sim.run(n_steps=100)
        
        logger.info(
            f"{config['name']:>20}: "
            f"polarisation={sim.polarisation():.3f}, "
            f"clustering={sim.clustering_coefficient():.3f}"
        )


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_schelling()
    print()
    demo_schelling_threshold()
    print()
    demo_boids()
    print()
    demo_boids_parameters()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Agent-Based Modelling Laboratory"
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
        "--schelling",
        action="store_true",
        help="Run Schelling model demo only",
    )
    parser.add_argument(
        "--boids",
        action="store_true",
        help="Run Boids model demo only",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=30,
        help="Grid size for Schelling model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Happiness threshold for Schelling model",
    )
    parser.add_argument(
        "--n-boids",
        type=int,
        default=100,
        help="Number of boids",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    rng = np.random.default_rng(args.seed)
    
    if args.schelling:
        model = SchellingModel(
            grid_size=args.grid_size,
            threshold=args.threshold,
            rng=rng,
        )
        history = model.run(max_steps=100)
        print(f"Final segregation: {history[-1]['segregation']:.3f}")
        return
    
    if args.boids:
        sim = BoidsSimulation(n_boids=args.n_boids, rng=rng)
        history = sim.run(n_steps=200)
        print(f"Final polarisation: {history[-1]['polarisation']:.3f}")
        return
    
    if args.demo:
        run_all_demos()
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
