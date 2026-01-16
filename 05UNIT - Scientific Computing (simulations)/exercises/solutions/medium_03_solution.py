#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: Schelling Segregation Model — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for medium_03_schelling.py

Implements core Schelling model and analyses threshold sensitivity.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class SimulationResult:
    """Result from Schelling simulation."""
    
    final_segregation: float
    final_happy_fraction: float
    steps_to_equilibrium: int
    history: list[dict]


class SchellingModel:
    """
    Schelling segregation model.
    
    Grid with two types of agents. Each agent is happy if at least
    `threshold` fraction of neighbours are same type. Unhappy agents
    move to random empty cell.
    """
    
    def __init__(
        self,
        grid_size: int = 30,
        empty_ratio: float = 0.2,
        threshold: float = 0.3,
        seed: int | None = None,
    ) -> None:
        """
        Initialise model.
        
        Args:
            grid_size: Width and height of grid
            empty_ratio: Fraction of empty cells
            threshold: Minimum same-type neighbour fraction for happiness
            seed: Random seed
        """
        self.grid_size = grid_size
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        
        # Create grid: 0=empty, 1=type A, 2=type B
        n_cells = grid_size * grid_size
        n_empty = int(n_cells * empty_ratio)
        n_agents = n_cells - n_empty
        n_each = n_agents // 2
        
        cells = [0] * n_empty + [1] * n_each + [2] * (n_agents - n_each)
        self.rng.shuffle(cells)
        self.grid = np.array(cells).reshape(grid_size, grid_size)
    
    def get_neighbours(self, row: int, col: int) -> list[int]:
        """Get non-empty Moore neighbours."""
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
        
        same = sum(1 for n in neighbours if n == agent_type)
        return (same / len(neighbours)) >= self.threshold
    
    def segregation_index(self) -> float:
        """Average fraction of same-type neighbours."""
        total = 0.0
        count = 0
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r, c] != 0:
                    neighbours = self.get_neighbours(r, c)
                    if neighbours:
                        same = sum(1 for n in neighbours if n == self.grid[r, c])
                        total += same / len(neighbours)
                        count += 1
        
        return total / count if count > 0 else 0.0
    
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
    
    def step(self) -> dict:
        """Execute one step."""
        self.step_count += 1
        
        # Find unhappy agents
        unhappy = [
            (r, c) for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] != 0 and not self.is_happy(r, c)
        ]
        self.rng.shuffle(unhappy)
        
        # Find empty cells
        empty = [
            (r, c) for r in range(self.grid_size)
            for c in range(self.grid_size)
            if self.grid[r, c] == 0
        ]
        
        moved = 0
        for row, col in unhappy:
            if not empty:
                break
            
            idx = self.rng.integers(len(empty))
            new_row, new_col = empty[idx]
            
            self.grid[new_row, new_col] = self.grid[row, col]
            self.grid[row, col] = 0
            empty[idx] = (row, col)
            moved += 1
        
        return {
            "step": self.step_count,
            "moved": moved,
            "happy": self.happy_fraction(),
            "segregation": self.segregation_index(),
        }
    
    def run(self, max_steps: int = 100) -> SimulationResult:
        """Run until equilibrium or max_steps."""
        history = []
        no_move_count = 0
        
        for _ in range(max_steps):
            metrics = self.step()
            history.append(metrics)
            
            if metrics["moved"] == 0:
                no_move_count += 1
                if no_move_count >= 3:
                    break
            else:
                no_move_count = 0
        
        return SimulationResult(
            final_segregation=self.segregation_index(),
            final_happy_fraction=self.happy_fraction(),
            steps_to_equilibrium=self.step_count,
            history=history,
        )


def threshold_sensitivity(
    thresholds: list[float],
    grid_size: int = 30,
    n_trials: int = 10,
    seed: int = 42,
) -> dict[float, dict]:
    """
    Analyse how threshold affects final segregation.
    
    Args:
        thresholds: Threshold values to test
        grid_size: Grid size
        n_trials: Trials per threshold
        seed: Base seed
        
    Returns:
        Dictionary mapping threshold to statistics
    """
    results = {}
    
    for threshold in thresholds:
        segregations = []
        steps = []
        
        for trial in range(n_trials):
            model = SchellingModel(
                grid_size=grid_size,
                threshold=threshold,
                seed=seed + trial * 1000 + int(threshold * 100),
            )
            result = model.run(max_steps=200)
            segregations.append(result.final_segregation)
            steps.append(result.steps_to_equilibrium)
        
        results[threshold] = {
            "mean_segregation": np.mean(segregations),
            "std_segregation": np.std(segregations),
            "mean_steps": np.mean(steps),
            "min_segregation": np.min(segregations),
            "max_segregation": np.max(segregations),
        }
    
    return results


def emergence_demonstration(
    threshold: float = 0.3,
    grid_size: int = 30,
    seed: int = 42,
) -> None:
    """Demonstrate emergence: mild preference → strong segregation."""
    import logging
    logger = logging.getLogger(__name__)
    
    model = SchellingModel(grid_size=grid_size, threshold=threshold, seed=seed)
    
    logger.info(f"Initial state:")
    logger.info(f"  Threshold (individual preference): {threshold:.0%}")
    logger.info(f"  Segregation index: {model.segregation_index():.1%}")
    logger.info(f"  Happy agents: {model.happy_fraction():.1%}")
    
    result = model.run(max_steps=100)
    
    logger.info(f"\nFinal state (after {result.steps_to_equilibrium} steps):")
    logger.info(f"  Segregation index: {result.final_segregation:.1%}")
    logger.info(f"  Happy agents: {result.final_happy_fraction:.1%}")
    
    logger.info(f"\nEmergence ratio: {result.final_segregation / threshold:.1f}x")
    logger.info("(Collective segregation >> individual threshold)")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Schelling Segregation Model")
    logger.info("=" * 60)
    
    # Emergence demonstration
    emergence_demonstration()
    
    logger.info("\n" + "=" * 60)
    logger.info("Threshold Sensitivity Analysis")
    logger.info("=" * 60)
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results = threshold_sensitivity(thresholds, n_trials=5)
    
    logger.info("\nThreshold → Final Segregation:")
    logger.info("-" * 40)
    for threshold, stats in results.items():
        logger.info(
            f"  {threshold:.0%} threshold → "
            f"{stats['mean_segregation']:.1%} ± {stats['std_segregation']:.1%} "
            f"(~{stats['mean_steps']:.0f} steps)"
        )
    
    logger.info("\nKey insight: Even mild preferences (30%) produce")
    logger.info("strong segregation (85%+). This is emergence.")
