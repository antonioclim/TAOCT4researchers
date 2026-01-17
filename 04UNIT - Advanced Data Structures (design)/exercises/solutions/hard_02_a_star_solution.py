#!/usr/bin/env python3
"""
Solution: A* Search Algorithm
============================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements the A* (A-star) pathfinding algorithm, an extension
of Dijkstra's algorithm that uses heuristics to guide the search towards
the goal more efficiently.

Complexity Analysis:
    - Time: O(b^d) where b is branching factor, d is depth (worst case)
    - Time: O(E) with perfect heuristic
    - Space: O(V) for open and closed sets

A* evaluates nodes using f(n) = g(n) + h(n) where:
    - g(n) = actual cost from start to n
    - h(n) = heuristic estimate from n to goal
    - f(n) = estimated total cost through n

The algorithm is optimal if the heuristic is admissible (never overestimates).

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import heapq
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')
INF = float('inf')


class Heuristic(ABC, Generic[T]):
    """
    Abstract base class for A* heuristics.
    
    A heuristic provides an estimate of the cost from a node to the goal.
    For A* to find optimal paths, the heuristic must be admissible:
    it must never overestimate the actual cost.
    """
    
    @abstractmethod
    def estimate(self, node: T, goal: T) -> float:
        """
        Estimate the cost from node to goal.
        
        Args:
            node: Current position.
            goal: Target position.
        
        Returns:
            Estimated cost (must not overestimate for optimality).
        """
        pass


@dataclass
class GridPosition:
    """
    A position on a 2D grid.
    
    Attributes:
        x: X coordinate (column).
        y: Y coordinate (row).
    """
    x: int
    y: int
    
    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridPosition):
            return NotImplemented
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other: GridPosition) -> bool:
        """For heap comparisons."""
        return (self.x, self.y) < (other.x, other.y)
    
    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


class ManhattanHeuristic(Heuristic[GridPosition]):
    """
    Manhattan distance heuristic for grid-based pathfinding.
    
    Calculates the sum of absolute differences in x and y coordinates.
    Admissible when only 4-directional movement is allowed.
    
    h(n) = |x₁ - x₂| + |y₁ - y₂|
    """
    
    def estimate(self, node: GridPosition, goal: GridPosition) -> float:
        return abs(node.x - goal.x) + abs(node.y - goal.y)


class EuclideanHeuristic(Heuristic[GridPosition]):
    """
    Euclidean distance heuristic.
    
    Calculates straight-line distance. Admissible for any movement,
    but may underestimate more than Manhattan for grid movement.
    
    h(n) = √((x₁ - x₂)² + (y₁ - y₂)²)
    """
    
    def estimate(self, node: GridPosition, goal: GridPosition) -> float:
        dx = node.x - goal.x
        dy = node.y - goal.y
        return math.sqrt(dx * dx + dy * dy)


class ChebyshevHeuristic(Heuristic[GridPosition]):
    """
    Chebyshev distance heuristic.
    
    Maximum of absolute differences. Admissible when diagonal
    movement costs the same as orthogonal movement.
    
    h(n) = max(|x₁ - x₂|, |y₁ - y₂|)
    """
    
    def estimate(self, node: GridPosition, goal: GridPosition) -> float:
        return max(abs(node.x - goal.x), abs(node.y - goal.y))


class OctileHeuristic(Heuristic[GridPosition]):
    """
    Octile distance heuristic.
    
    Optimal for grids where diagonal movement costs √2.
    Combines diagonal and straight movement optimally.
    """
    
    def estimate(self, node: GridPosition, goal: GridPosition) -> float:
        dx = abs(node.x - goal.x)
        dy = abs(node.y - goal.y)
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


class ZeroHeuristic(Heuristic[GridPosition]):
    """
    Zero heuristic (always returns 0).
    
    Using h(n) = 0 makes A* equivalent to Dijkstra's algorithm.
    Always admissible but provides no search guidance.
    """
    
    def estimate(self, node: GridPosition, goal: GridPosition) -> float:
        return 0


@dataclass
class Grid:
    """
    A 2D grid for pathfinding.
    
    Attributes:
        width: Grid width.
        height: Grid height.
        obstacles: Set of blocked positions.
        diagonal_movement: Whether diagonal moves are allowed.
        diagonal_cost: Cost of diagonal movement (default √2).
    """
    width: int
    height: int
    obstacles: set[GridPosition] = field(default_factory=set)
    diagonal_movement: bool = False
    diagonal_cost: float = math.sqrt(2)
    
    def is_valid(self, pos: GridPosition) -> bool:
        """Check if position is within bounds and not blocked."""
        return (
            0 <= pos.x < self.width and
            0 <= pos.y < self.height and
            pos not in self.obstacles
        )
    
    def neighbours(self, pos: GridPosition) -> list[tuple[GridPosition, float]]:
        """
        Get valid neighbours with movement costs.
        
        Returns list of (position, cost) tuples.
        """
        # Orthogonal directions (4-directional)
        directions_4 = [
            (0, 1), (0, -1), (1, 0), (-1, 0)
        ]
        
        # Diagonal directions
        directions_8 = directions_4 + [
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        directions = directions_8 if self.diagonal_movement else directions_4
        result = []
        
        for dx, dy in directions:
            new_pos = GridPosition(pos.x + dx, pos.y + dy)
            
            if self.is_valid(new_pos):
                # Diagonal moves cost more
                cost = self.diagonal_cost if (dx != 0 and dy != 0) else 1.0
                result.append((new_pos, cost))
        
        return result
    
    def add_obstacle(self, x: int, y: int) -> None:
        """Add an obstacle at the given position."""
        self.obstacles.add(GridPosition(x, y))
    
    def remove_obstacle(self, x: int, y: int) -> None:
        """Remove an obstacle from the given position."""
        self.obstacles.discard(GridPosition(x, y))
    
    def visualise(
        self,
        path: list[GridPosition] | None = None,
        start: GridPosition | None = None,
        goal: GridPosition | None = None
    ) -> str:
        """
        Create a text visualisation of the grid.
        
        Symbols:
            . = empty
            # = obstacle
            S = start
            G = goal
            * = path
        """
        lines = []
        path_set = set(path) if path else set()
        
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = GridPosition(x, y)
                
                if pos == start:
                    row.append('S')
                elif pos == goal:
                    row.append('G')
                elif pos in path_set:
                    row.append('*')
                elif pos in self.obstacles:
                    row.append('#')
                else:
                    row.append('.')
            
            lines.append(' '.join(row))
        
        return '\n'.join(lines)


@dataclass
class AStarResult:
    """
    Result of A* search.
    
    Attributes:
        success: Whether a path was found.
        path: The path as a list of positions.
        cost: Total path cost.
        nodes_explored: Number of nodes expanded.
        nodes_generated: Number of nodes added to open set.
    """
    success: bool = False
    path: list[GridPosition] = field(default_factory=list)
    cost: float = INF
    nodes_explored: int = 0
    nodes_generated: int = 0
    
    def __str__(self) -> str:
        if self.success:
            return (
                f"Path found! Cost: {self.cost:.2f}, "
                f"Length: {len(self.path)}, "
                f"Explored: {self.nodes_explored}, "
                f"Generated: {self.nodes_generated}"
            )
        return f"No path found. Explored {self.nodes_explored} nodes."


def a_star(
    grid: Grid,
    start: GridPosition,
    goal: GridPosition,
    heuristic: Heuristic[GridPosition] | None = None
) -> AStarResult:
    """
    A* pathfinding algorithm.
    
    Finds the shortest path from start to goal using A* search.
    Uses the provided heuristic to guide the search.
    
    Args:
        grid: The grid to search.
        start: Starting position.
        goal: Target position.
        heuristic: Heuristic function (defaults to Manhattan).
    
    Returns:
        AStarResult with path and statistics.
    
    Time Complexity: O(E log V) average with good heuristic.
    Space Complexity: O(V) for open and closed sets.
    
    Examples:
        >>> grid = Grid(10, 10)
        >>> start = GridPosition(0, 0)
        >>> goal = GridPosition(9, 9)
        >>> result = a_star(grid, start, goal)
        >>> result.success
        True
    """
    if heuristic is None:
        heuristic = ManhattanHeuristic()
    
    result = AStarResult()
    
    # Check validity
    if not grid.is_valid(start) or not grid.is_valid(goal):
        logger.warning("Start or goal position is invalid")
        return result
    
    # g_score: cost from start to node
    g_score: dict[GridPosition, float] = {start: 0}
    
    # f_score: estimated total cost through node
    f_score: dict[GridPosition, float] = {start: heuristic.estimate(start, goal)}
    
    # For path reconstruction
    came_from: dict[GridPosition, GridPosition] = {}
    
    # Priority queue: (f_score, counter, position)
    open_set: list[tuple[float, int, GridPosition]] = []
    counter = 0
    heapq.heappush(open_set, (f_score[start], counter, start))
    
    # Track what's in open set for O(1) lookup
    open_set_hash: set[GridPosition] = {start}
    
    # Closed set (already explored)
    closed_set: set[GridPosition] = set()
    
    logger.debug(f"Starting A* from {start} to {goal}")
    
    while open_set:
        # Get node with lowest f_score
        _, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        
        result.nodes_explored += 1
        
        # Skip if already processed (duplicate in heap)
        if current in closed_set:
            continue
        
        logger.debug(
            f"Exploring {current}, f={f_score.get(current, INF):.2f}, "
            f"g={g_score.get(current, INF):.2f}"
        )
        
        # Goal reached!
        if current == goal:
            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            
            result.success = True
            result.path = path
            result.cost = g_score[goal]
            
            logger.info(f"Path found: {result}")
            return result
        
        closed_set.add(current)
        
        # Explore neighbours
        for neighbour, move_cost in grid.neighbours(current):
            if neighbour in closed_set:
                continue
            
            tentative_g = g_score[current] + move_cost
            
            if tentative_g < g_score.get(neighbour, INF):
                # This is a better path
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f_score[neighbour] = tentative_g + heuristic.estimate(neighbour, goal)
                
                if neighbour not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbour], counter, neighbour))
                    open_set_hash.add(neighbour)
                    result.nodes_generated += 1
    
    logger.info(f"No path found after exploring {result.nodes_explored} nodes")
    return result


def compare_heuristics(
    grid: Grid,
    start: GridPosition,
    goal: GridPosition
) -> dict[str, AStarResult]:
    """
    Compare different heuristics on the same problem.
    
    Returns results for each heuristic to analyse their effectiveness.
    """
    heuristics = {
        'Manhattan': ManhattanHeuristic(),
        'Euclidean': EuclideanHeuristic(),
        'Chebyshev': ChebyshevHeuristic(),
        'Zero (Dijkstra)': ZeroHeuristic(),
    }
    
    if grid.diagonal_movement:
        heuristics['Octile'] = OctileHeuristic()
    
    results = {}
    for name, heuristic in heuristics.items():
        results[name] = a_star(grid, start, goal, heuristic)
    
    return results


def weighted_a_star(
    grid: Grid,
    start: GridPosition,
    goal: GridPosition,
    weight: float = 1.5,
    heuristic: Heuristic[GridPosition] | None = None
) -> AStarResult:
    """
    Weighted A* (WA*) algorithm.
    
    Uses f(n) = g(n) + w * h(n) where w > 1.
    Trades optimality for speed - finds paths faster but they
    may be up to w times longer than optimal.
    
    Args:
        grid: The grid to search.
        start: Starting position.
        goal: Target position.
        weight: Heuristic weight (w >= 1, larger = faster but less optimal).
        heuristic: Heuristic function.
    
    Returns:
        AStarResult with path and statistics.
    """
    if heuristic is None:
        heuristic = ManhattanHeuristic()
    
    result = AStarResult()
    
    if not grid.is_valid(start) or not grid.is_valid(goal):
        return result
    
    g_score: dict[GridPosition, float] = {start: 0}
    f_score: dict[GridPosition, float] = {
        start: weight * heuristic.estimate(start, goal)
    }
    came_from: dict[GridPosition, GridPosition] = {}
    
    open_set: list[tuple[float, int, GridPosition]] = []
    counter = 0
    heapq.heappush(open_set, (f_score[start], counter, start))
    open_set_hash: set[GridPosition] = {start}
    closed_set: set[GridPosition] = set()
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        open_set_hash.discard(current)
        
        result.nodes_explored += 1
        
        if current in closed_set:
            continue
        
        if current == goal:
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            
            result.success = True
            result.path = path
            result.cost = g_score[goal]
            return result
        
        closed_set.add(current)
        
        for neighbour, move_cost in grid.neighbours(current):
            if neighbour in closed_set:
                continue
            
            tentative_g = g_score[current] + move_cost
            
            if tentative_g < g_score.get(neighbour, INF):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f_score[neighbour] = tentative_g + weight * heuristic.estimate(
                    neighbour, goal
                )
                
                if neighbour not in open_set_hash:
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbour], counter, neighbour))
                    open_set_hash.add(neighbour)
                    result.nodes_generated += 1
    
    return result


def demonstrate_a_star() -> None:
    """Demonstrate A* pathfinding algorithm."""
    print("=" * 60)
    print("A* Search Algorithm Demonstration")
    print("=" * 60)
    
    # Example 1: Simple path
    print("\n1. Simple Grid Navigation")
    print("-" * 40)
    
    grid = Grid(10, 8)
    start = GridPosition(1, 1)
    goal = GridPosition(8, 6)
    
    result = a_star(grid, start, goal)
    print(f"   {result}")
    print(f"\n{grid.visualise(result.path, start, goal)}")
    
    # Example 2: With obstacles
    print("\n2. Navigating Around Obstacles")
    print("-" * 40)
    
    grid_obstacles = Grid(12, 10)
    
    # Add a wall
    for y in range(2, 8):
        grid_obstacles.add_obstacle(5, y)
    
    # Add some scattered obstacles
    for pos in [(7, 3), (7, 4), (8, 4), (3, 6), (3, 7)]:
        grid_obstacles.add_obstacle(*pos)
    
    start = GridPosition(2, 5)
    goal = GridPosition(10, 5)
    
    result = a_star(grid_obstacles, start, goal)
    print(f"   {result}")
    print(f"\n{grid_obstacles.visualise(result.path, start, goal)}")
    
    # Example 3: Compare heuristics
    print("\n3. Heuristic Comparison")
    print("-" * 40)
    
    grid_large = Grid(20, 20)
    # Add diagonal wall
    for i in range(15):
        grid_large.add_obstacle(5 + i // 2, 2 + i)
    
    start = GridPosition(2, 10)
    goal = GridPosition(18, 10)
    
    comparisons = compare_heuristics(grid_large, start, goal)
    
    print("   Heuristic        | Cost  | Explored | Generated")
    print("   " + "-" * 50)
    for name, res in sorted(comparisons.items(), key=lambda x: x[1].nodes_explored):
        print(
            f"   {name:17} | {res.cost:5.1f} | {res.nodes_explored:8} | "
            f"{res.nodes_generated:9}"
        )
    
    # Example 4: Diagonal movement
    print("\n4. With Diagonal Movement")
    print("-" * 40)
    
    grid_diagonal = Grid(10, 10, diagonal_movement=True)
    
    start = GridPosition(0, 0)
    goal = GridPosition(9, 9)
    
    # Using octile heuristic for diagonal grids
    result_octile = a_star(grid_diagonal, start, goal, OctileHeuristic())
    print(f"   With Octile heuristic: {result_octile}")
    
    result_manhattan = a_star(grid_diagonal, start, goal, ManhattanHeuristic())
    print(f"   With Manhattan heuristic: {result_manhattan}")
    
    print(f"   Note: Both find same cost but Manhattan explores more nodes")
    
    # Example 5: Weighted A*
    print("\n5. Weighted A* (Speed vs Optimality)")
    print("-" * 40)
    
    grid_weighted = Grid(30, 30)
    # Create a maze-like obstacle pattern
    for i in range(0, 25, 5):
        for j in range(i % 10, 28):
            grid_weighted.add_obstacle(i + 3, j)
    
    start = GridPosition(1, 15)
    goal = GridPosition(28, 15)
    
    print("   Weight | Cost   | Explored | Optimality Ratio")
    print("   " + "-" * 45)
    
    optimal_result = a_star(grid_weighted, start, goal)
    optimal_cost = optimal_result.cost
    
    for weight in [1.0, 1.5, 2.0, 3.0, 5.0]:
        result = weighted_a_star(grid_weighted, start, goal, weight)
        ratio = result.cost / optimal_cost if optimal_cost > 0 else 0
        print(
            f"   {weight:6.1f} | {result.cost:6.1f} | {result.nodes_explored:8} | "
            f"{ratio:.3f}"
        )
    
    # Example 6: No path exists
    print("\n6. Unreachable Goal")
    print("-" * 40)
    
    grid_blocked = Grid(10, 10)
    # Create a complete wall
    for y in range(10):
        grid_blocked.add_obstacle(5, y)
    
    start = GridPosition(2, 5)
    goal = GridPosition(8, 5)
    
    result = a_star(grid_blocked, start, goal)
    print(f"   {result}")
    print(f"\n{grid_blocked.visualise(None, start, goal)}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_a_star()
