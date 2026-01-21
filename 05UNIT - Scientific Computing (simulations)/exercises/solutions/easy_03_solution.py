#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5, Exercise: Random Walk — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

Solution for easy_03_random_walk.py

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def random_walk_1d(
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Simulate 1D random walk.
    
    At each step, move +1 or -1 with equal probability.
    
    Args:
        n_steps: Number of steps
        rng: Random number generator
        
    Returns:
        Array of positions (length n_steps + 1, starting at 0)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.zeros(n_steps + 1)
    positions[1:] = np.cumsum(steps)
    
    return positions


def random_walk_2d(
    n_steps: int,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Simulate 2D random walk on lattice.
    
    At each step, move in one of four directions: up, down, left, right.
    
    Args:
        n_steps: Number of steps
        rng: Random number generator
        
    Returns:
        Array of shape (n_steps + 1, 2) with (x, y) positions
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Directions: right, up, left, down
    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    
    choices = rng.integers(0, 4, size=n_steps)
    steps = directions[choices]
    
    positions = np.zeros((n_steps + 1, 2))
    positions[1:] = np.cumsum(steps, axis=0)
    
    return positions


def mean_squared_displacement(
    n_steps: int,
    n_walks: int = 1000,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """
    Compute mean squared displacement over many walks.
    
    For a random walk, MSD grows linearly with time: MSD(t) = D·t
    where D is the diffusion coefficient.
    
    Args:
        n_steps: Steps per walk
        n_walks: Number of walks to average
        rng: Random number generator
        
    Returns:
        Array of MSD values for each time step
    """
    if rng is None:
        rng = np.random.default_rng()
    
    squared_displacements = np.zeros((n_walks, n_steps + 1))
    
    for i in range(n_walks):
        positions = random_walk_1d(n_steps, rng)
        squared_displacements[i] = positions ** 2
    
    return np.mean(squared_displacements, axis=0)


def first_return_time(
    n_walks: int = 10000,
    max_steps: int = 10000,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Compute first return times to origin for 1D walks.
    
    Args:
        n_walks: Number of walks to simulate
        max_steps: Maximum steps before giving up
        rng: Random number generator
        
    Returns:
        List of return times (excluding walks that didn't return)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    return_times = []
    
    for _ in range(n_walks):
        position = 0
        for step in range(1, max_steps + 1):
            position += rng.choice([-1, 1])
            if position == 0:
                return_times.append(step)
                break
    
    return return_times


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    rng = np.random.default_rng(42)
    
    # 1D walk
    walk_1d = random_walk_1d(100, rng)
    logger.info(f"1D walk final position: {walk_1d[-1]}")
    
    # 2D walk
    walk_2d = random_walk_2d(100, rng)
    distance = np.linalg.norm(walk_2d[-1])
    logger.info(f"2D walk final distance from origin: {distance:.2f}")
    
    # MSD analysis
    msd = mean_squared_displacement(1000, n_walks=500, rng=rng)
    
    # Fit diffusion coefficient
    times = np.arange(len(msd))
    coeffs = np.polyfit(times[1:], msd[1:], 1)
    logger.info(f"Diffusion coefficient D ≈ {coeffs[0]:.3f}")
    logger.info("(Theory: D = 1 for simple random walk)")
    
    # First return times
    returns = first_return_time(n_walks=1000, rng=rng)
    logger.info(f"Walks returning to origin: {len(returns)}/1000")
    if returns:
        logger.info(f"Mean return time: {np.mean(returns):.1f} steps")
