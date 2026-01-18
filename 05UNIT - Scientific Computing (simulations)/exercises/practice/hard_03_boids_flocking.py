#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Hard 03 — Boids Flocking Simulation
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 60 minutes
TOPIC: Agent-based modelling (Boids)

OBJECTIVE
─────────
Implement Craig Reynolds' Boids algorithm for simulating flocking behaviour.

BACKGROUND
──────────
Boids uses three simple steering rules that produce emergent flocking:

1. SEPARATION: Steer away from nearby boids to avoid crowding
2. ALIGNMENT: Steer toward average velocity of neighbours
3. COHESION: Steer toward centre of mass of neighbours

Each rule produces a steering vector; the weighted sum updates velocity.

TASKS
─────
1. Implement the three steering rules
2. Implement velocity update with speed limiting
3. Compute polarisation metric to measure flock alignment
4. Run simulation and observe emergence

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class Boid:
    """Single boid agent."""
    
    position: NDArray[np.floating]
    velocity: NDArray[np.floating]
    boid_id: int = field(default_factory=lambda: id(object()))
    
    @property
    def speed(self) -> float:
        """Current speed (magnitude of velocity)."""
        return float(np.linalg.norm(self.velocity))
    
    def distance_to(self, other: Boid) -> float:
        """Euclidean distance to another boid."""
        return float(np.linalg.norm(self.position - other.position))


@dataclass
class BoidsConfig:
    """Configuration parameters for Boids simulation."""
    
    # Environment
    width: float = 800.0
    height: float = 600.0
    
    # Perception
    visual_range: float = 75.0
    min_distance: float = 20.0
    
    # Steering weights
    separation_weight: float = 1.5
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    
    # Speed limits
    min_speed: float = 2.0
    max_speed: float = 5.0
    
    # Edge behaviour
    margin: float = 50.0
    turn_factor: float = 0.5


class BoidsSimulation:
    """Boids flocking simulation."""
    
    def __init__(
        self,
        n_boids: int = 50,
        config: BoidsConfig | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise simulation.
        
        Args:
            n_boids: Number of boids
            config: Configuration parameters
            rng: Random generator
        """
        self.config = config or BoidsConfig()
        self.rng = rng or np.random.default_rng()
        
        # Initialise boids with random positions and velocities
        self.boids: list[Boid] = []
        for _ in range(n_boids):
            pos = np.array([
                self.rng.uniform(0, self.config.width),
                self.rng.uniform(0, self.config.height),
            ])
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(self.config.min_speed, self.config.max_speed)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            
            self.boids.append(Boid(position=pos, velocity=vel))
        
        self.step_count = 0
    
    def get_neighbours(self, boid: Boid) -> list[Boid]:
        """Find boids within visual range."""
        neighbours = []
        for other in self.boids:
            if other is not boid:
                if boid.distance_to(other) < self.config.visual_range:
                    neighbours.append(other)
        return neighbours
    
    def separation(
        self,
        boid: Boid,
        neighbours: list[Boid],
    ) -> NDArray[np.floating]:
        """Compute separation steering vector.
        
        Steer away from boids that are too close (within min_distance).
        
        Args:
            boid: The boid being updated
            neighbours: List of nearby boids
            
        Returns:
            Steering vector pointing away from close neighbours
        """
        steering = np.zeros(2)
        
        # TODO: For each neighbour within min_distance:
        #   - Compute vector pointing away from neighbour
        #   - Add to steering (optionally weight by inverse distance)
        #
        # for other in neighbours:
        #     dist = boid.distance_to(other)
        #     if dist < self.config.min_distance and dist > 0:
        #         diff = boid.position - other.position
        #         steering += diff / dist  # Weight by inverse distance
        
        raise NotImplementedError("Complete this method")
    
    def alignment(
        self,
        boid: Boid,
        neighbours: list[Boid],
    ) -> NDArray[np.floating]:
        """Compute alignment steering vector.
        
        Steer toward the average velocity of neighbours.
        
        Args:
            boid: The boid being updated
            neighbours: List of nearby boids
            
        Returns:
            Steering vector toward average velocity
        """
        if len(neighbours) == 0:
            return np.zeros(2)
        
        # TODO: Compute average velocity of neighbours
        # avg_velocity = np.mean([n.velocity for n in neighbours], axis=0)
        
        # TODO: Return steering toward average (difference from current)
        # return avg_velocity - boid.velocity
        
        raise NotImplementedError("Complete this method")
    
    def cohesion(
        self,
        boid: Boid,
        neighbours: list[Boid],
    ) -> NDArray[np.floating]:
        """Compute cohesion steering vector.
        
        Steer toward the centre of mass of neighbours.
        
        Args:
            boid: The boid being updated
            neighbours: List of nearby boids
            
        Returns:
            Steering vector toward centre of mass
        """
        if len(neighbours) == 0:
            return np.zeros(2)
        
        # TODO: Compute centre of mass of neighbours
        # centre = np.mean([n.position for n in neighbours], axis=0)
        
        # TODO: Return steering toward centre
        # return centre - boid.position
        
        raise NotImplementedError("Complete this method")
    
    def edge_avoidance(self, boid: Boid) -> NDArray[np.floating]:
        """Compute edge avoidance steering.
        
        Apply turning force when near boundaries.
        """
        steering = np.zeros(2)
        cfg = self.config
        
        if boid.position[0] < cfg.margin:
            steering[0] += cfg.turn_factor
        elif boid.position[0] > cfg.width - cfg.margin:
            steering[0] -= cfg.turn_factor
        
        if boid.position[1] < cfg.margin:
            steering[1] += cfg.turn_factor
        elif boid.position[1] > cfg.height - cfg.margin:
            steering[1] -= cfg.turn_factor
        
        return steering
    
    def update_boid(self, boid: Boid) -> None:
        """Update single boid's velocity and position."""
        neighbours = self.get_neighbours(boid)
        cfg = self.config
        
        # Compute steering forces
        sep = self.separation(boid, neighbours)
        ali = self.alignment(boid, neighbours)
        coh = self.cohesion(boid, neighbours)
        edge = self.edge_avoidance(boid)
        
        # Combine with weights
        steering = (
            cfg.separation_weight * sep
            + cfg.alignment_weight * ali
            + cfg.cohesion_weight * coh
            + edge
        )
        
        # Update velocity
        boid.velocity = boid.velocity + steering
        
        # Limit speed
        speed = boid.speed
        if speed > cfg.max_speed:
            boid.velocity = boid.velocity / speed * cfg.max_speed
        elif speed < cfg.min_speed and speed > 0:
            boid.velocity = boid.velocity / speed * cfg.min_speed
        
        # Update position
        boid.position = boid.position + boid.velocity
    
    def step(self) -> dict[str, float]:
        """Execute one simulation step."""
        for boid in self.boids:
            self.update_boid(boid)
        
        self.step_count += 1
        
        return {
            "step": self.step_count,
            "polarisation": self.polarisation(),
            "avg_neighbours": self.average_neighbours(),
        }
    
    def run(self, n_steps: int) -> list[dict[str, float]]:
        """Run simulation for n_steps."""
        history = []
        for _ in range(n_steps):
            metrics = self.step()
            history.append(metrics)
        return history
    
    def polarisation(self) -> float:
        """Compute flock polarisation (alignment metric).
        
        Polarisation = |average of normalised velocities|
        Range: 0 (random) to 1 (perfectly aligned)
        """
        if len(self.boids) == 0:
            return 0.0
        
        normalised = []
        for boid in self.boids:
            speed = boid.speed
            if speed > 0:
                normalised.append(boid.velocity / speed)
        
        if len(normalised) == 0:
            return 0.0
        
        avg = np.mean(normalised, axis=0)
        return float(np.linalg.norm(avg))
    
    def average_neighbours(self) -> float:
        """Average number of neighbours per boid."""
        total = sum(len(self.get_neighbours(b)) for b in self.boids)
        return total / len(self.boids) if self.boids else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    print("Boids Flocking Simulation")
    print("=" * 70)
    
    try:
        # Create simulation
        config = BoidsConfig(
            width=400,
            height=300,
            visual_range=75,
            separation_weight=1.5,
            alignment_weight=1.0,
            cohesion_weight=1.0,
        )
        
        sim = BoidsSimulation(n_boids=50, config=config, rng=rng)
        
        print(f"\nConfiguration:")
        print(f"  Boids: {len(sim.boids)}")
        print(f"  Visual range: {config.visual_range}")
        print(f"  Weights: sep={config.separation_weight}, "
              f"ali={config.alignment_weight}, coh={config.cohesion_weight}")
        
        print(f"\nInitial state:")
        print(f"  Polarisation: {sim.polarisation():.3f}")
        print(f"  Avg neighbours: {sim.average_neighbours():.1f}")
        
        # Run simulation
        print("\nRunning 200 steps...")
        history = sim.run(200)
        
        print(f"\nFinal state:")
        print(f"  Polarisation: {history[-1]['polarisation']:.3f}")
        print(f"  Avg neighbours: {history[-1]['avg_neighbours']:.1f}")
        
        # Show evolution
        print("\nEvolution:")
        print(f"{'Step':>6} {'Polarisation':>14} {'Avg Neighbours':>16}")
        print("-" * 40)
        
        for i in [0, 20, 50, 100, 150, 199]:
            h = history[i]
            print(f"{h['step']:>6} {h['polarisation']:>14.3f} {h['avg_neighbours']:>16.1f}")
        
        # Polarisation should increase
        initial_pol = history[0]["polarisation"]
        final_pol = history[-1]["polarisation"]
        print(f"\nPolarisation change: {initial_pol:.3f} → {final_pol:.3f}")
        
    except NotImplementedError as e:
        print(f"\nImplementation incomplete: {e}")
        print("Complete separation, alignment and cohesion methods first!")
