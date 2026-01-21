#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Medium 03 — Schelling Segregation Model
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 45 minutes
TOPIC: Agent-based modelling (Schelling)

OBJECTIVE
─────────
Implement the core Schelling segregation model with neighbour analysis.

BACKGROUND
──────────
Thomas Schelling's 1971 model shows how mild individual preferences for
same-type neighbours can produce strong collective segregation. Agents on
a grid move if fewer than a threshold fraction of their neighbours are
the same type.

TASKS
─────
1. Complete the `get_neighbours` function (Moore neighbourhood)
2. Complete the `is_happy` function
3. Complete the `step` function
4. Run simulation and observe emergence

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SchellingGrid:
    """Schelling segregation model on a 2D grid.
    
    Grid values:
      0 = empty cell
      1 = type A agent
      2 = type B agent
    """
    
    size: int = 20
    empty_ratio: float = 0.2
    threshold: float = 0.3
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng()
    )
    grid: np.ndarray = field(init=False)
    step_count: int = field(default=0, init=False)
    
    def __post_init__(self) -> None:
        """Initialise random grid."""
        n_cells = self.size ** 2
        n_empty = int(n_cells * self.empty_ratio)
        n_agents = n_cells - n_empty
        n_type_a = n_agents // 2
        n_type_b = n_agents - n_type_a
        
        # Create flat array and shuffle
        cells = np.array(
            [0] * n_empty + [1] * n_type_a + [2] * n_type_b
        )
        self.rng.shuffle(cells)
        self.grid = cells.reshape((self.size, self.size))
    
    def get_neighbours(self, row: int, col: int) -> list[int]:
        """Get values of occupied neighbouring cells (Moore neighbourhood).
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            List of neighbour values (1 or 2), excluding empty cells (0)
            
        Example:
            For a 3x3 grid with all cells occupied by type 1:
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]
            get_neighbours(1, 1) returns [1, 1, 1, 1, 1, 1, 1, 1] (8 neighbours)
        """
        _neighbours = []
        
        # TODO: Iterate over the 8 neighbouring cells (Moore neighbourhood)
        # for dr in [-1, 0, 1]:
        #     for dc in [-1, 0, 1]:
        #         if dr == 0 and dc == 0:
        #             continue  # Skip self
        #         
        #         nr, nc = row + dr, col + dc
        #         
        #         # TODO: Check bounds
        #         # if 0 <= nr < self.size and 0 <= nc < self.size:
        #         #     value = self.grid[nr, nc]
        #         #     # TODO: Only add non-empty cells
        #         #     if value != 0:
        #         #         neighbours.append(value)
        
        raise NotImplementedError("Complete this method")
    
    def is_happy(self, row: int, col: int) -> bool:
        """Check if agent at (row, col) is happy.
        
        An agent is happy if:
        - The cell is empty (trivially happy)
        - At least `threshold` fraction of neighbours are same type
        - No neighbours means happy (edge case)
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            True if agent is happy or cell is empty
        """
        agent_type = self.grid[row, col]
        
        # Empty cells are trivially "happy"
        if agent_type == 0:
            return True
        
        neighbours = self.get_neighbours(row, col)
        
        # No neighbours means happy
        if len(neighbours) == 0:
            return True
        
        # TODO: Count same-type neighbours
        # same_type = sum(1 for n in neighbours if n == agent_type)
        
        # TODO: Check if fraction meets threshold
        # fraction_same = same_type / len(neighbours)
        # return fraction_same >= self.threshold
        
        raise NotImplementedError("Complete this method")
    
    def step(self) -> dict[str, float]:
        """Execute one simulation step.
        
        All unhappy agents move to random empty cells.
        
        Returns:
            Dictionary with metrics: moved, happy_fraction, segregation
        """
        # Find unhappy agents and empty cells
        unhappy = []
        empty_cells = []
        
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 0:
                    empty_cells.append((row, col))
                elif not self.is_happy(row, col):
                    unhappy.append((row, col))
        
        # Shuffle both lists
        self.rng.shuffle(unhappy)
        self.rng.shuffle(empty_cells)
        
        # Move unhappy agents to empty cells
        moved = 0
        for (ur, uc), (er, ec) in zip(unhappy, empty_cells):
            # TODO: Move agent from (ur, uc) to (er, ec)
            # self.grid[er, ec] = self.grid[ur, uc]
            # self.grid[ur, uc] = 0
            # moved += 1
            raise NotImplementedError("Complete the movement logic")
        
        self.step_count += 1
        
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
        
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] != 0:
                    total += 1
                    if self.is_happy(row, col):
                        happy += 1
        
        return happy / total if total > 0 else 1.0
    
    def segregation_index(self) -> float:
        """Mean same-type neighbour fraction across all agents."""
        total_fraction = 0.0
        n_agents = 0
        
        for row in range(self.size):
            for col in range(self.size):
                agent_type = self.grid[row, col]
                if agent_type == 0:
                    continue
                
                neighbours = self.get_neighbours(row, col)
                if len(neighbours) == 0:
                    continue
                
                same = sum(1 for n in neighbours if n == agent_type)
                total_fraction += same / len(neighbours)
                n_agents += 1
        
        return total_fraction / n_agents if n_agents > 0 else 0.0
    
    def run(self, max_steps: int = 100) -> list[dict[str, float]]:
        """Run simulation until equilibrium or max_steps."""
        history = []
        
        for _ in range(max_steps):
            metrics = self.step()
            history.append(metrics)
            
            if metrics["moved"] == 0:
                break
        
        return history


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Schelling Segregation Model")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    
    try:
        model = SchellingGrid(size=20, threshold=0.3, rng=rng)
        
        print("\nInitial state:")
        print(f"  Grid size: {model.size}×{model.size}")
        print(f"  Threshold: {model.threshold}")
        print(f"  Happy fraction: {model.happy_fraction():.2%}")
        print(f"  Segregation index: {model.segregation_index():.3f}")
        
        print("\nRunning simulation...")
        history = model.run(max_steps=100)
        
        print(f"\nFinal state (after {len(history)} steps):")
        print(f"  Happy fraction: {history[-1]['happy_fraction']:.2%}")
        print(f"  Segregation index: {history[-1]['segregation']:.3f}")
        
        print("\nEvolution:")
        print(f"{'Step':>6} {'Moved':>8} {'Happy':>10} {'Segregation':>12}")
        print("-" * 40)
        for h in history[::max(1, len(history)//10)]:  # Show ~10 rows
            print(f"{h['step']:>6} {h['moved']:>8} {h['happy_fraction']:>10.2%} {h['segregation']:>12.3f}")
        
    except NotImplementedError as e:
        print(f"\nImplementation incomplete: {e}")
        print("Complete get_neighbours, is_happy and step methods first!")
