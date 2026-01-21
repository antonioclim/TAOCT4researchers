#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: Optimised Boids Flocking — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for hard_03_boids_flocking.py

Optimised Boids implementation with spatial hashing for O(n) performance.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass
class Boid:
    """A boid with position and velocity."""
    
    position: FloatArray
    velocity: FloatArray
    boid_id: int = 0
    
    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))
    
    def distance_to(self, other: "Boid") -> float:
        return float(np.linalg.norm(self.position - other.position))


class SpatialHash:
    """
    Spatial hash grid for efficient neighbour queries.
    
    Divides space into cells of size `cell_size`. Neighbour queries
    only check nearby cells, giving O(1) average case instead of O(n).
    """
    
    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self.grid: dict[tuple[int, int], list[Boid]] = defaultdict(list)
    
    def _cell_key(self, position: FloatArray) -> tuple[int, int]:
        """Compute grid cell for position."""
        return (
            int(position[0] // self.cell_size),
            int(position[1] // self.cell_size),
        )
    
    def clear(self) -> None:
        """Clear the grid."""
        self.grid.clear()
    
    def insert(self, boid: Boid) -> None:
        """Insert boid into grid."""
        key = self._cell_key(boid.position)
        self.grid[key].append(boid)
    
    def get_nearby(self, position: FloatArray, radius: float) -> Iterator[Boid]:
        """Get all boids within radius of position."""
        cx, cy = self._cell_key(position)
        
        # Check neighbouring cells (radius / cell_size cells in each direction)
        cells_to_check = int(np.ceil(radius / self.cell_size))
        
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dy in range(-cells_to_check, cells_to_check + 1):
                key = (cx + dx, cy + dy)
                for boid in self.grid.get(key, []):
                    if np.linalg.norm(boid.position - position) <= radius:
                        yield boid


class OptimisedBoids:
    """
    Optimised Boids simulation using spatial hashing.
    
    Achieves O(n) performance for neighbour queries instead of O(n²).
    """
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        n_boids: int = 500,
        visual_range: float = 75.0,
        min_distance: float = 20.0,
        max_speed: float = 5.0,
        min_speed: float = 2.0,
        separation_weight: float = 1.5,
        alignment_weight: float = 1.0,
        cohesion_weight: float = 1.0,
        margin: float = 50.0,
        turn_factor: float = 0.5,
        seed: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.visual_range = visual_range
        self.min_distance = min_distance
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.sep_weight = separation_weight
        self.ali_weight = alignment_weight
        self.coh_weight = cohesion_weight
        self.margin = margin
        self.turn_factor = turn_factor
        
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        
        # Spatial hash with cell size = visual range
        self.spatial_hash = SpatialHash(visual_range)
        
        # Create boids
        self.boids: list[Boid] = []
        for i in range(n_boids):
            pos = np.array([
                self.rng.uniform(margin, width - margin),
                self.rng.uniform(margin, height - margin),
            ])
            angle = self.rng.uniform(0, 2 * np.pi)
            speed = self.rng.uniform(min_speed, max_speed)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            self.boids.append(Boid(pos, vel, boid_id=i))
    
    def _rebuild_spatial_hash(self) -> None:
        """Rebuild spatial hash with current positions."""
        self.spatial_hash.clear()
        for boid in self.boids:
            self.spatial_hash.insert(boid)
    
    def get_neighbours(self, boid: Boid) -> list[Boid]:
        """Get neighbours using spatial hash."""
        neighbours = []
        for other in self.spatial_hash.get_nearby(boid.position, self.visual_range):
            if other is not boid:
                neighbours.append(other)
        return neighbours
    
    def separation(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Separation steering."""
        steer = np.zeros(2)
        for other in neighbours:
            dist = boid.distance_to(other)
            if dist < self.min_distance and dist > 0:
                diff = boid.position - other.position
                steer += diff / dist
        return steer
    
    def alignment(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Alignment steering."""
        if not neighbours:
            return np.zeros(2)
        avg_vel = np.mean([n.velocity for n in neighbours], axis=0)
        return (avg_vel - boid.velocity) * 0.05
    
    def cohesion(self, boid: Boid, neighbours: list[Boid]) -> FloatArray:
        """Cohesion steering."""
        if not neighbours:
            return np.zeros(2)
        centre = np.mean([n.position for n in neighbours], axis=0)
        return (centre - boid.position) * 0.005
    
    def edge_avoidance(self, boid: Boid) -> FloatArray:
        """Edge avoidance steering."""
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
        """Clamp speed to [min_speed, max_speed]."""
        speed = boid.speed
        if speed > self.max_speed:
            boid.velocity = (boid.velocity / speed) * self.max_speed
        elif speed < self.min_speed and speed > 0:
            boid.velocity = (boid.velocity / speed) * self.min_speed
    
    def step(self) -> dict:
        """Execute one simulation step."""
        self.step_count += 1
        
        # Rebuild spatial hash
        self._rebuild_spatial_hash()
        
        # Compute all updates
        updates = []
        total_neighbours = 0
        
        for boid in self.boids:
            neighbours = self.get_neighbours(boid)
            total_neighbours += len(neighbours)
            
            sep = self.separation(boid, neighbours) * self.sep_weight
            ali = self.alignment(boid, neighbours) * self.ali_weight
            coh = self.cohesion(boid, neighbours) * self.coh_weight
            edge = self.edge_avoidance(boid)
            
            new_vel = boid.velocity + sep + ali + coh + edge
            updates.append((boid, new_vel))
        
        # Apply updates
        for boid, new_vel in updates:
            boid.velocity = new_vel
            self.limit_speed(boid)
            boid.position = boid.position + boid.velocity
        
        return {
            "step": self.step_count,
            "polarisation": self.polarisation(),
            "avg_neighbours": total_neighbours / len(self.boids),
        }
    
    def polarisation(self) -> float:
        """Order parameter: 1 = aligned, 0 = random."""
        if not self.boids:
            return 0.0
        
        sum_v = np.zeros(2)
        for boid in self.boids:
            if boid.speed > 0:
                sum_v += boid.velocity / boid.speed
        
        return float(np.linalg.norm(sum_v) / len(self.boids))
    
    def run(self, n_steps: int = 100) -> list[dict]:
        """Run simulation."""
        history = []
        for _ in range(n_steps):
            metrics = self.step()
            history.append(metrics)
        return history


def benchmark_comparison(
    n_boids_values: list[int],
    n_steps: int = 10,
    seed: int = 42,
) -> dict[int, dict]:
    """
    Benchmark optimised vs naive implementation.
    
    Returns timing data for different population sizes.
    """
    import time
    
    results = {}
    
    for n_boids in n_boids_values:
        # Optimised version
        sim = OptimisedBoids(n_boids=n_boids, seed=seed)
        
        start = time.perf_counter()
        sim.run(n_steps)
        optimised_time = time.perf_counter() - start
        
        results[n_boids] = {
            "optimised_time": optimised_time,
            "time_per_step": optimised_time / n_steps,
            "time_per_boid_step": optimised_time / (n_steps * n_boids),
        }
    
    return results


def analyse_emergence(
    n_boids: int = 200,
    n_steps: int = 500,
    seed: int = 42,
) -> dict:
    """
    Analyse emergence of flocking behaviour.
    
    Tracks polarisation over time to see coordination emerge.
    """
    sim = OptimisedBoids(n_boids=n_boids, seed=seed)
    history = sim.run(n_steps)
    
    polarisation = [h["polarisation"] for h in history]
    
    # Find time to reach high polarisation
    threshold = 0.8
    time_to_flock = None
    for i, p in enumerate(polarisation):
        if p >= threshold:
            time_to_flock = i
            break
    
    return {
        "initial_polarisation": polarisation[0],
        "final_polarisation": polarisation[-1],
        "max_polarisation": max(polarisation),
        "time_to_flock": time_to_flock,
        "polarisation_history": polarisation,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Optimised Boids with Spatial Hashing")
    logger.info("=" * 60)
    
    # Emergence analysis
    logger.info("\n1. Emergence Analysis")
    results = analyse_emergence(n_boids=200, n_steps=300)
    
    logger.info(f"   Initial polarisation: {results['initial_polarisation']:.3f}")
    logger.info(f"   Final polarisation:   {results['final_polarisation']:.3f}")
    logger.info(f"   Time to flock (≥0.8): {results['time_to_flock']} steps")
    
    # Performance benchmark
    logger.info("\n2. Performance Benchmark")
    n_values = [100, 200, 500, 1000]
    bench = benchmark_comparison(n_values, n_steps=20)
    
    logger.info(f"   {'N Boids':<10} {'Time/Step (ms)':<15} {'Time/Boid (μs)':<15}")
    logger.info("   " + "-" * 40)
    for n, data in bench.items():
        logger.info(
            f"   {n:<10} {data['time_per_step']*1000:<15.2f} "
            f"{data['time_per_boid_step']*1e6:<15.2f}"
        )
    
    # Verify O(n) scaling
    times = [bench[n]["time_per_step"] for n in n_values]
    if len(times) >= 2:
        # Ratio of times should be ~= ratio of n for O(n)
        ratio_n = n_values[-1] / n_values[0]
        ratio_time = times[-1] / times[0]
        logger.info(f"\n   N ratio: {ratio_n:.1f}x")
        logger.info(f"   Time ratio: {ratio_time:.1f}x")
        logger.info(f"   (O(n) would give ~{ratio_n:.1f}x, O(n²) would give ~{ratio_n**2:.1f}x)")
    
    logger.info("\n" + "=" * 60)
    logger.info("Spatial hashing reduces complexity from O(n²) to O(n)")
