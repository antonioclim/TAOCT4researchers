#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Lab 3: Agent-Based Modelling — SOLUTIONS
═══════════════════════════════════════════════════════════════════════════════

Complete reference implementation for lab_5_03_agent_based_modelling.py

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.floating]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BASE CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


class Agent(ABC):
    """Abstract base class for agents."""
    
    _id_counter: int = 0
    
    def __init__(self) -> None:
        Agent._id_counter += 1
        self.agent_id = Agent._id_counter
    
    @abstractmethod
    def update(self, environment: "Environment") -> None:
        """Update agent state based on environment."""
        pass


class Environment(ABC):
    """Abstract base class for environments."""
    
    @abstractmethod
    def get_neighbours(self, agent: Agent) -> list[Agent]:
        """Get neighbouring agents."""
        pass
    
    @abstractmethod
    def step(self) -> dict:
        """Advance simulation by one step."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SCHELLING SEGREGATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SchellingAgent:
    """Agent in Schelling model."""
    
    agent_type: int  # 1 or 2
    row: int
    col: int
    threshold: float = 0.3
    
    def is_happy(self, neighbours: list[int]) -> bool:
        """Check if agent is happy with neighbourhood."""
        if not neighbours:
            return True
        same_type = sum(1 for n in neighbours if n == self.agent_type)
        return (same_type / len(neighbours)) >= self.threshold


class SchellingModel:
    """
    Schelling segregation model implementation.
    
    Demonstrates emergence: mild individual preferences (30% same-type)
    produce strong collective segregation (85%+).
    
    Example:
        >>> model = SchellingModel(grid_size=30, threshold=0.3)
        >>> history = model.run(max_steps=100)
        >>> print(f"Final segregation: {model.segregation_index():.2%}")
    """
    
    def __init__(
        self,
        grid_size: int = 30,
        empty_ratio: float = 0.2,
        threshold: float = 0.3,
        type_ratio: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        """
        Initialise Schelling model.
        
        Args:
            grid_size: Width and height of square grid
            empty_ratio: Fraction of cells that are empty
            threshold: Minimum same-type neighbour fraction for happiness
            type_ratio: Fraction of agents that are type 1
            rng: Random number generator
        """
        self.grid_size = grid_size
        self.threshold = threshold
        self.rng = rng if rng is not None else np.random.default_rng()
        self.step_count = 0
        
        # Create grid: 0=empty, 1=type A, 2=type B
        n_cells = grid_size * grid_size
        n_empty = int(n_cells * empty_ratio)
        n_agents = n_cells - n_empty
        n_type1 = int(n_agents * type_ratio)
        n_type2 = n_agents - n_type1
        
        cells = [0] * n_empty + [1] * n_type1 + [2] * n_type2
        self.rng.shuffle(cells)
        self.grid = np.array(cells).reshape(grid_size, grid_size)
        
        logger.debug(
            f"Created {grid_size}x{grid_size} grid: "
            f"{n_empty} empty, {n_type1} type1, {n_type2} type2"
        )
    
    def get_neighbours(self, row: int, col: int) -> list[int]:
        """Get non-empty Moore neighbours (8-connectivity)."""
        neighbours = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                    if self.grid[r, c] != 0:
                        neighbours.append(self.grid[r, c])
        return neighbours
    
    def is_happy(self, row: int, col: int) -> bool:
        """Check if agent at position is happy."""
        agent_type = self.grid[row, col]
        if agent_type == 0:
            return True
        
        neighbours = self.get_neighbours(row, col)
        if not neighbours:
            return True
        
        same_type = sum(1 for n in neighbours if n == agent_type)
        return (same_type / len(neighbours)) >= self.threshold
    
    def find_empty_cells(self) -> list[tuple[int, int]]:
        """Find all empty cells."""
        return [(r, c) for r in range(self.grid_size) 
                for c in range(self.grid_size) if self.grid[r, c] == 0]
    
    def step(self) -> dict:
        """Execute one simulation step."""
        self.step_count += 1
        moved = 0
        
        # Get all unhappy agents
        unhappy = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != 0 and not self.is_happy(r, c):
                    unhappy.append((r, c))
        
        # Shuffle for random order
        self.rng.shuffle(unhappy)
        
        # Move unhappy agents
        empty_cells = self.find_empty_cells()
        for row, col in unhappy:
            if not empty_cells:
                break
            
            # Pick random empty cell
            idx = self.rng.integers(len(empty_cells))
            new_row, new_col = empty_cells[idx]
            
            # Move agent
            self.grid[new_row, new_col] = self.grid[row, col]
            self.grid[row, col] = 0
            
            # Update empty cells
            empty_cells[idx] = (row, col)
            moved += 1
        
        return {
            "step": self.step_count,
            "moved": moved,
            "happy_fraction": self.happy_fraction(),
            "segregation": self.segregation_index(),
        }
    
    def happy_fraction(self) -> float:
        """Fraction of agents that are happy."""
        happy = 0
        total = 0
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != 0:
                    total += 1
                    if self.is_happy(r, c):
                        happy += 1
        
        return happy / total if total > 0 else 1.0
    
    def segregation_index(self) -> float:
        """
        Compute segregation index.
        
        Average fraction of same-type neighbours across all agents.
        """
        total_ratio = 0.0
        count = 0
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                agent_type = self.grid[r, c]
                if agent_type != 0:
                    neighbours = self.get_neighbours(r, c)
                    if neighbours:
                        same = sum(1 for n in neighbours if n == agent_type)
                        total_ratio += same / len(neighbours)
                        count += 1
        
        return total_ratio / count if count > 0 else 0.0
    
    def run(self, max_steps: int = 100, equilibrium_threshold: int = 5) -> list[dict]:
        """
        Run simulation until equilibrium or max_steps.
        
        Args:
            max_steps: Maximum steps to run
            equilibrium_threshold: Stop if no movement for this many steps
            
        Returns:
            List of metrics dictionaries
        """
        history = []
        no_movement_count = 0
        
        for _ in range(max_steps):
            metrics = self.step()
            history.append(metrics)
            
            if metrics["moved"] == 0:
                no_movement_count += 1
                if no_movement_count >= equilibrium_threshold:
                    logger.info(f"Equilibrium reached at step {self.step_count}")
                    break
            else:
                no_movement_count = 0
        
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: BOIDS FLOCKING MODEL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Boid:
    """A boid agent with position and velocity."""
    
    position: FloatArray
    velocity: FloatArray
    boid_id: int = field(default_factory=lambda: Boid._next_id())
    
    _counter: int = 0
    
    @staticmethod
    def _next_id() -> int:
        Boid._counter += 1
        return Boid._counter
    
    @property
    def speed(self) -> float:
        """Current speed (velocity magnitude)."""
        return float(np.linalg.norm(self.velocity))
    
    def distance_to(self, other: "Boid") -> float:
        """Euclidean distance to another boid."""
        return float(np.linalg.norm(self.position - other.position))


class BoidsSimulation:
    """
    Boids flocking simulation (Reynolds, 1987).
    
    Three rules create emergent flocking:
    1. Separation: Steer away from nearby boids
    2. Alignment: Match velocity with neighbours
    3. Cohesion: Steer toward centre of mass of neighbours
    
    Example:
        >>> sim = BoidsSimulation(n_boids=100, width=800, height=600)
        >>> history = sim.run(n_steps=500)
        >>> print(f"Polarisation: {sim.polarisation():.2f}")
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        n_boids: int = 100,
        visual_range: float = 75.0,
        min_distance: float = 20.0,
        max_speed: float = 5.0,
        min_speed: float = 2.0,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        turn_factor: float = 0.5,
        margin: float = 50.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise boids simulation."""
        self.width = width
        self.height = height
        self.visual_range = visual_range
        self.min_distance = min_distance
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.separation_weight = separation_weight
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.turn_factor = turn_factor
        self.margin = margin
        self.rng = rng if rng is not None else np.random.default_rng()
        self.step_count = 0
        
        # Create boids
        self.boids: list[Boid] = []
        for _ in range(n_boids):
            pos = np.array([
                self.rng.uniform(margin, width - margin),
                self.rng.uniform(margin, height - margin),
            ])
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(min_speed, max_speed)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            self.boids.append(Boid(position=pos, velocity=vel))
    
    def get_neighbours(self, boid: Boid) -> list[Boid]:
        """Get boids within visual range."""
        return [
            other for other in self.boids
            if other is not boid and boid.distance_to(other) < self.visual_range
        ]
    
    def separation(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Steer away from nearby boids to avoid crowding."""
        steer = np.zeros(2)
        for other in neighbours:
            dist = boid.distance_to(other)
            if dist < self.min_distance and dist > 0:
                diff = boid.position - other.position
                steer += diff / dist
        return steer
    
    def alignment(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Steer toward average velocity of neighbours."""
        if not neighbours:
            return np.zeros(2)
        avg_vel = np.mean([n.velocity for n in neighbours], axis=0)
        return (avg_vel - boid.velocity) * 0.05
    
    def cohesion(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Steer toward centre of mass of neighbours."""
        if not neighbours:
            return np.zeros(2)
        centre = np.mean([n.position for n in neighbours], axis=0)
        return (centre - boid.position) * 0.005
    
    def edge_avoidance(self, boid: Boid) -> FloatArray:
        """Steer away from edges."""
        steer = np.zeros(2)
        if boid.position[0] < self.margin:
            steer[0] += self.turn_factor
        if boid.position[0] > self.width - self.margin:
            steer[0] -= self.turn_factor
        if boid.position[1] < self.margin:
            steer[1] += self.turn_factor
        if boid.position[1] > self.height - self.margin:
            steer[1] -= self.turn_factor
        return steer
    
    def limit_speed(self, boid: Boid) -> None:
        """Limit boid speed to [min_speed, max_speed]."""
        speed = boid.speed
        if speed > self.max_speed:
            boid.velocity = (boid.velocity / speed) * self.max_speed
        elif speed < self.min_speed and speed > 0:
            boid.velocity = (boid.velocity / speed) * self.min_speed
    
    def step(self) -> dict:
        """Execute one simulation step."""
        self.step_count += 1
        
        # Calculate all updates first
        updates = []
        for boid in self.boids:
            neighbours = self.get_neighbours(boid)
            
            sep = self.separation(boid, neighbours) * self.separation_weight
            ali = self.alignment(boid, neighbours) * self.alignment_weight
            coh = self.cohesion(boid, neighbours) * self.cohesion_weight
            edge = self.edge_avoidance(boid)
            
            new_vel = boid.velocity + sep + ali + coh + edge
            updates.append((boid, new_vel, len(neighbours)))
        
        # Apply updates
        total_neighbours = 0
        for boid, new_vel, n_neighbours in updates:
            boid.velocity = new_vel
            self.limit_speed(boid)
            boid.position = boid.position + boid.velocity
            total_neighbours += n_neighbours
        
        return {
            "step": self.step_count,
            "polarisation": self.polarisation(),
            "avg_neighbours": total_neighbours / len(self.boids),
            "avg_speed": np.mean([b.speed for b in self.boids]),
        }
    
    def polarisation(self) -> float:
        """
        Measure of alignment (order parameter).
        
        Returns 1 if all boids move in same direction, 0 for random.
        """
        if not self.boids:
            return 0.0
        
        # Average of normalised velocities
        sum_v = np.zeros(2)
        for boid in self.boids:
            speed = boid.speed
            if speed > 0:
                sum_v += boid.velocity / speed
        
        return float(np.linalg.norm(sum_v) / len(self.boids))
    
    def average_neighbours(self) -> float:
        """Average number of neighbours per boid."""
        total = sum(len(self.get_neighbours(b)) for b in self.boids)
        return total / len(self.boids) if self.boids else 0.0
    
    def clustering_coefficient(self) -> float:
        """Measure of local clustering."""
        if not self.boids:
            return 0.0
        
        total = 0.0
        for boid in self.boids:
            neighbours = self.get_neighbours(boid)
            if len(neighbours) < 2:
                continue
            
            # Count edges between neighbours
            edges = 0
            for i, n1 in enumerate(neighbours):
                for n2 in neighbours[i+1:]:
                    if n1.distance_to(n2) < self.visual_range:
                        edges += 1
            
            max_edges = len(neighbours) * (len(neighbours) - 1) / 2
            if max_edges > 0:
                total += edges / max_edges
        
        return total / len(self.boids)
    
    def run(self, n_steps: int = 100) -> list[dict]:
        """Run simulation for n_steps."""
        history = []
        for _ in range(n_steps):
            metrics = self.step()
            history.append(metrics)
        return history
