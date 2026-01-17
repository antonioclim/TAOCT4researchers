#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 5 Practice: Easy 03 — Simple Random Walk Agent
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 25 minutes
TOPIC: Agent-based modelling basics

OBJECTIVE
─────────
Implement a simple random walk agent on a 2D grid.

BACKGROUND
──────────
A random walk is the simplest form of agent-based model. At each time step,
the agent moves in a random direction (up, down, left, right) with equal
probability. This models diffusion, foraging behaviour and Brownian motion.

TASKS
─────
1. Complete the `RandomWalkAgent` class
2. Simulate multiple steps and track the path
3. Compute mean squared displacement

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RandomWalkAgent:
    """Agent that performs a 2D random walk.
    
    Attributes:
        x: Current x position
        y: Current y position
        history: List of (x, y) positions visited
    """
    
    x: int = 0
    y: int = 0
    history: list[tuple[int, int]] = field(default_factory=list)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng()
    )
    
    def __post_init__(self) -> None:
        """Record initial position."""
        self.history.append((self.x, self.y))
    
    def step(self) -> None:
        """Take one random step in a cardinal direction.
        
        Moves the agent one unit in a random direction:
        - 0: up (y + 1)
        - 1: down (y - 1)
        - 2: left (x - 1)
        - 3: right (x + 1)
        """
        # TODO: Generate random direction (0, 1, 2, or 3)
        # direction = self.rng.integers(0, 4)
        
        # TODO: Update position based on direction
        # if direction == 0:
        #     self.y += 1
        # elif direction == 1:
        #     ...
        
        # TODO: Record new position in history
        # self.history.append((self.x, self.y))
        
        raise NotImplementedError("Complete this method")
    
    def walk(self, n_steps: int) -> None:
        """Perform n_steps of random walk."""
        for _ in range(n_steps):
            self.step()
    
    def displacement(self) -> float:
        """Euclidean distance from origin."""
        return float(np.sqrt(self.x**2 + self.y**2))
    
    def squared_displacement(self) -> float:
        """Squared distance from origin (x² + y²)."""
        return float(self.x**2 + self.y**2)


def mean_squared_displacement(
    n_agents: int,
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> float:
    """Compute mean squared displacement over many agents.
    
    Args:
        n_agents: Number of independent random walks
        n_steps: Number of steps per walk
        rng: Random number generator
        
    Returns:
        Mean squared displacement ⟨r²⟩
        
    Note:
        For a 2D random walk, theory predicts ⟨r²⟩ = n_steps
    """
    if rng is None:
        rng = np.random.default_rng()
    
    total_sd = 0.0
    
    for _ in range(n_agents):
        agent = RandomWalkAgent(rng=rng)
        agent.walk(n_steps)
        total_sd += agent.squared_displacement()
    
    return total_sd / n_agents


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    
    print("Random Walk Agent Simulation")
    print("=" * 50)
    
    # Single agent demonstration
    print("\nSingle Agent Walk (100 steps):")
    try:
        agent = RandomWalkAgent(rng=np.random.default_rng(42))
        agent.walk(100)
        print(f"  Final position: ({agent.x}, {agent.y})")
        print(f"  Displacement: {agent.displacement():.2f}")
        print(f"  Squared displacement: {agent.squared_displacement()}")
    except NotImplementedError:
        print("  Complete the step() method first!")
    
    # Mean squared displacement study
    print("\nMean Squared Displacement Study:")
    print(f"{'Steps':>8} {'⟨r²⟩':>12} {'Theory':>12} {'Ratio':>8}")
    print("-" * 44)
    
    try:
        for n_steps in [10, 100, 1000]:
            msd = mean_squared_displacement(1000, n_steps, rng)
            theory = n_steps  # ⟨r²⟩ = n for 2D random walk
            ratio = msd / theory
            print(f"{n_steps:>8} {msd:>12.1f} {theory:>12} {ratio:>8.2f}")
    except NotImplementedError:
        print("Complete the RandomWalkAgent.step() method first!")
    
    print("-" * 44)
    print("Note: Ratio ≈ 1 confirms diffusive behaviour")
