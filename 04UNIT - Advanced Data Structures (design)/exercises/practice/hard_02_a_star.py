#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
04UNIT, Practice Exercise: A* Search Algorithm
Difficulty: HARD
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
A* is the gold standard for heuristic pathfinding. Unlike Dijkstra (which
explores uniformly in all directions), A* uses a heuristic to guide the
search towards the goal, dramatically reducing the number of nodes explored.

A* combines:
- g(n): actual cost from start to node n
- h(n): heuristic estimate from n to goal
- f(n) = g(n) + h(n): total estimated cost

The heuristic must be ADMISSIBLE (never overestimates) for A* to guarantee
optimality. Common admissible heuristics include Euclidean distance and
Manhattan distance for grid-based problems.

TASK
────
Implement the A* search algorithm with support for:
1. Custom heuristic functions
2. Path reconstruction
3. Node visit counting for performance comparison with Dijkstra
4. Grid-based navigation with obstacles

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Implement A* with proper f-score ordering
2. Design admissible heuristics for different problem domains
3. Compare A* performance against Dijkstra empirically
4. Apply A* to practical routing and navigation problems

ESTIMATED TIME
──────────────
90-120 minutes

DEPENDENCIES
────────────
- Python 3.12+
- heapq (standard library)
- typing (standard library)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Callable, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Type variable for node type
N = TypeVar("N")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GridPosition:
    """
    Represents a position on a 2D grid.
    
    Attributes:
        row: Row index (0-indexed)
        col: Column index (0-indexed)
    """
    row: int
    col: int
    
    def __hash__(self) -> int:
        return hash((self.row, self.col))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridPosition):
            return NotImplemented
        return self.row == other.row and self.col == other.col
    
    def __repr__(self) -> str:
        return f"({self.row}, {self.col})"


@dataclass
class AStarResult(Generic[N]):
    """
    Result of A* search.
    
    Attributes:
        path: List of nodes from start to goal (empty if no path found)
        cost: Total path cost (infinity if no path)
        nodes_visited: Number of nodes expanded during search
        nodes_explored: Total nodes added to open set
    """
    path: list[N]
    cost: float
    nodes_visited: int
    nodes_explored: int
    
    @property
    def found(self) -> bool:
        """Returns True if a path was found."""
        return len(self.path) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HEURISTIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def euclidean_distance(a: GridPosition, b: GridPosition) -> float:
    """
    Euclidean (straight-line) distance heuristic.
    
    Admissible for any movement cost >= 1 per cell.
    
    Args:
        a: Start position
        b: Goal position
        
    Returns:
        Straight-line distance between positions
        
    Example:
        >>> euclidean_distance(GridPosition(0, 0), GridPosition(3, 4))
        5.0
    """
    # TODO: Implement Euclidean distance
    # h = sqrt((a.row - b.row)^2 + (a.col - b.col)^2)
    raise NotImplementedError("Implement euclidean_distance")


def manhattan_distance(a: GridPosition, b: GridPosition) -> float:
    """
    Manhattan (taxicab) distance heuristic.
    
    Optimal for 4-directional movement with uniform cost.
    Admissible when diagonal movement is not allowed.
    
    Args:
        a: Start position
        b: Goal position
        
    Returns:
        Sum of absolute differences in row and column
        
    Example:
        >>> manhattan_distance(GridPosition(0, 0), GridPosition(3, 4))
        7
    """
    # TODO: Implement Manhattan distance
    # h = |a.row - b.row| + |a.col - b.col|
    raise NotImplementedError("Implement manhattan_distance")


def chebyshev_distance(a: GridPosition, b: GridPosition) -> float:
    """
    Chebyshev (chessboard) distance heuristic.
    
    Optimal for 8-directional movement with uniform cost.
    Admissible when diagonal movement has cost 1.
    
    Args:
        a: Start position
        b: Goal position
        
    Returns:
        Maximum of absolute differences in row and column
        
    Example:
        >>> chebyshev_distance(GridPosition(0, 0), GridPosition(3, 4))
        4
    """
    # TODO: Implement Chebyshev distance
    # h = max(|a.row - b.row|, |a.col - b.col|)
    raise NotImplementedError("Implement chebyshev_distance")


def zero_heuristic(a: GridPosition, b: GridPosition) -> float:
    """
    Zero heuristic (degrades A* to Dijkstra).
    
    Always admissible but provides no guidance.
    Useful for comparing A* performance with Dijkstra.
    
    Args:
        a: Start position (unused)
        b: Goal position (unused)
        
    Returns:
        Always 0
    """
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: GRID GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class GridGraph:
    """
    A 2D grid graph with obstacles.
    
    Supports 4-directional or 8-directional movement.
    
    Attributes:
        rows: Number of rows
        cols: Number of columns
        obstacles: Set of blocked positions
        diagonal: Whether diagonal movement is allowed
        
    Example:
        >>> grid = GridGraph(10, 10, diagonal=True)
        >>> grid.add_obstacle(GridPosition(5, 5))
        >>> list(grid.neighbours(GridPosition(4, 4)))
        [(GridPosition(3, 3), 1.414...), (GridPosition(3, 4), 1.0), ...]
    """
    
    def __init__(
        self,
        rows: int,
        cols: int,
        diagonal: bool = False
    ) -> None:
        """
        Initialise grid graph.
        
        Args:
            rows: Number of rows (height)
            cols: Number of columns (width)
            diagonal: Allow diagonal movement
        """
        self.rows = rows
        self.cols = cols
        self.obstacles: set[GridPosition] = set()
        self.diagonal = diagonal
        
        # Movement directions
        self._directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self._directions_8 = self._directions_4 + [
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
    
    def add_obstacle(self, pos: GridPosition) -> None:
        """Mark a position as blocked."""
        self.obstacles.add(pos)
    
    def remove_obstacle(self, pos: GridPosition) -> None:
        """Remove obstacle from position."""
        self.obstacles.discard(pos)
    
    def is_valid(self, pos: GridPosition) -> bool:
        """Check if position is within bounds and not an obstacle."""
        return (
            0 <= pos.row < self.rows and
            0 <= pos.col < self.cols and
            pos not in self.obstacles
        )
    
    def neighbours(
        self, pos: GridPosition
    ) -> Iterator[tuple[GridPosition, float]]:
        """
        Generate valid neighbours with movement costs.
        
        Args:
            pos: Current position
            
        Yields:
            Tuples of (neighbour_position, movement_cost)
            Diagonal moves cost sqrt(2), cardinal moves cost 1.0
        """
        directions = self._directions_8 if self.diagonal else self._directions_4
        
        for dr, dc in directions:
            new_pos = GridPosition(pos.row + dr, pos.col + dc)
            
            if self.is_valid(new_pos):
                # Diagonal moves cost sqrt(2), others cost 1
                cost = math.sqrt(2) if (dr != 0 and dc != 0) else 1.0
                yield new_pos, cost
    
    def __str__(self) -> str:
        """String representation of the grid."""
        lines = []
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                if GridPosition(r, c) in self.obstacles:
                    row_str += "█"
                else:
                    row_str += "·"
            lines.append(row_str)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: A* ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════

def a_star_search(
    grid: GridGraph,
    start: GridPosition,
    goal: GridPosition,
    heuristic: Callable[[GridPosition, GridPosition], float] = euclidean_distance
) -> AStarResult[GridPosition]:
    """
    A* search algorithm for grid pathfinding.
    
    Finds the shortest path from start to goal using the provided heuristic.
    
    Algorithm:
    1. Initialise open set with start node (priority = h(start, goal))
    2. While open set not empty:
       a. Pop node with lowest f = g + h
       b. If node is goal, reconstruct and return path
       c. For each neighbour:
          - Calculate tentative g score
          - If better than known g score, update and add to open set
    3. Return failure if goal not reached
    
    Args:
        grid: The grid graph to search
        start: Starting position
        goal: Goal position
        heuristic: Heuristic function h(current, goal) -> float
        
    Returns:
        AStarResult containing path, cost and statistics
        
    Raises:
        ValueError: If start or goal is invalid
        
    Example:
        >>> grid = GridGraph(10, 10, diagonal=True)
        >>> result = a_star_search(
        ...     grid,
        ...     GridPosition(0, 0),
        ...     GridPosition(9, 9),
        ...     heuristic=euclidean_distance
        ... )
        >>> result.found
        True
        >>> result.cost  # Approximately 12.73 for diagonal path
        12.727...
    """
    # Validate inputs
    if not grid.is_valid(start):
        raise ValueError(f"Invalid start position: {start}")
    if not grid.is_valid(goal):
        raise ValueError(f"Invalid goal position: {goal}")
    
    # Early exit if start equals goal
    if start == goal:
        return AStarResult(
            path=[start],
            cost=0.0,
            nodes_visited=1,
            nodes_explored=1
        )
    
    # TODO: Implement A* search
    #
    # Data structures needed:
    # - open_set: priority queue of (f_score, counter, node) tuples
    #   (counter breaks ties for deterministic ordering)
    # - came_from: dict mapping node -> predecessor node
    # - g_score: dict mapping node -> cost from start
    # - closed_set: set of visited nodes (optional but improves efficiency)
    #
    # Algorithm steps:
    # 1. Initialise g_score[start] = 0
    # 2. Initialise f_score[start] = h(start, goal)
    # 3. Add start to open_set
    # 4. While open_set not empty:
    #    - Pop node with lowest f_score
    #    - If node == goal: reconstruct path and return
    #    - Add node to closed_set
    #    - For each neighbour of node:
    #      - If neighbour in closed_set: skip
    #      - Calculate tentative_g = g_score[node] + edge_cost
    #      - If tentative_g < g_score[neighbour]:
    #        - Update came_from[neighbour] = node
    #        - Update g_score[neighbour] = tentative_g
    #        - Calculate f = tentative_g + h(neighbour, goal)
    #        - Add neighbour to open_set (if not already)
    # 5. Return failure (empty path)
    #
    # Path reconstruction:
    # - Start from goal
    # - Follow came_from links back to start
    # - Reverse the resulting list
    
    raise NotImplementedError("Implement a_star_search")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: COMPARISON WITH DIJKSTRA
# ═══════════════════════════════════════════════════════════════════════════════

def dijkstra_search(
    grid: GridGraph,
    start: GridPosition,
    goal: GridPosition
) -> AStarResult[GridPosition]:
    """
    Dijkstra's algorithm (A* with zero heuristic).
    
    Provided for comparison with A*.
    
    Args:
        grid: The grid graph to search
        start: Starting position
        goal: Goal position
        
    Returns:
        AStarResult containing path, cost and statistics
    """
    return a_star_search(grid, start, goal, heuristic=zero_heuristic)


def compare_algorithms(
    grid: GridGraph,
    start: GridPosition,
    goal: GridPosition
) -> dict[str, AStarResult[GridPosition]]:
    """
    Compare different search algorithms and heuristics.
    
    Args:
        grid: The grid graph to search
        start: Starting position
        goal: Goal position
        
    Returns:
        Dictionary mapping algorithm name to results
    """
    # TODO: Implement algorithm comparison
    #
    # Test the following:
    # - Dijkstra (zero heuristic)
    # - A* with Euclidean heuristic
    # - A* with Manhattan heuristic (if 4-directional)
    # - A* with Chebyshev heuristic (if 8-directional)
    #
    # Return a dictionary with results for each
    
    raise NotImplementedError("Implement compare_algorithms")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MAZE GENERATION (BONUS)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_random_maze(
    rows: int,
    cols: int,
    obstacle_density: float = 0.3,
    seed: int | None = None
) -> GridGraph:
    """
    Generate a random maze with obstacles.
    
    Ensures start (0,0) and goal (rows-1, cols-1) are not blocked.
    Does NOT guarantee a path exists.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        obstacle_density: Proportion of cells to block (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        GridGraph with random obstacles
    """
    # TODO: Implement random maze generation
    #
    # 1. Create GridGraph
    # 2. For each cell (except start and goal):
    #    - With probability = obstacle_density, add obstacle
    # 3. Optionally verify path exists using A*
    
    raise NotImplementedError("Implement generate_random_maze")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_heuristics() -> None:
    """Test heuristic functions."""
    a = GridPosition(0, 0)
    b = GridPosition(3, 4)
    
    # Euclidean: sqrt(3^2 + 4^2) = 5.0
    assert abs(euclidean_distance(a, b) - 5.0) < 1e-10
    
    # Manhattan: |3| + |4| = 7
    assert manhattan_distance(a, b) == 7
    
    # Chebyshev: max(3, 4) = 4
    assert chebyshev_distance(a, b) == 4
    
    # Zero: always 0
    assert zero_heuristic(a, b) == 0.0
    
    logger.info("✓ Heuristic tests passed")


def test_grid_graph() -> None:
    """Test grid graph operations."""
    grid = GridGraph(5, 5, diagonal=False)
    
    # Test validity
    assert grid.is_valid(GridPosition(0, 0))
    assert grid.is_valid(GridPosition(4, 4))
    assert not grid.is_valid(GridPosition(-1, 0))
    assert not grid.is_valid(GridPosition(5, 5))
    
    # Test obstacles
    grid.add_obstacle(GridPosition(2, 2))
    assert not grid.is_valid(GridPosition(2, 2))
    
    # Test neighbours (4-directional)
    centre = GridPosition(2, 3)
    neighbours = list(grid.neighbours(centre))
    # Should have 3 neighbours (one blocked by obstacle)
    assert len(neighbours) == 3
    
    logger.info("✓ Grid graph tests passed")


def test_a_star_simple() -> None:
    """Test A* on simple grid without obstacles."""
    grid = GridGraph(10, 10, diagonal=True)
    start = GridPosition(0, 0)
    goal = GridPosition(9, 9)
    
    result = a_star_search(grid, start, goal, euclidean_distance)
    
    assert result.found
    assert result.path[0] == start
    assert result.path[-1] == goal
    # Diagonal path should have approximately this cost
    expected_cost = 9 * math.sqrt(2)
    assert abs(result.cost - expected_cost) < 1e-10
    
    logger.info("✓ Simple A* test passed")


def test_a_star_with_obstacles() -> None:
    """Test A* navigation around obstacles."""
    grid = GridGraph(5, 5, diagonal=False)
    
    # Create a wall that must be navigated around
    for r in range(4):
        grid.add_obstacle(GridPosition(r, 2))
    
    start = GridPosition(2, 0)
    goal = GridPosition(2, 4)
    
    result = a_star_search(grid, start, goal, manhattan_distance)
    
    assert result.found
    # Path must go around the wall
    assert result.cost > manhattan_distance(start, goal)
    
    logger.info("✓ Obstacle navigation test passed")


def test_a_star_no_path() -> None:
    """Test A* when no path exists."""
    grid = GridGraph(5, 5, diagonal=False)
    
    # Completely surround the goal
    goal = GridPosition(2, 2)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr != 0 or dc != 0:
                grid.add_obstacle(GridPosition(2 + dr, 2 + dc))
    
    start = GridPosition(0, 0)
    result = a_star_search(grid, start, goal, manhattan_distance)
    
    assert not result.found
    assert result.cost == float("inf")
    
    logger.info("✓ No path test passed")


def test_a_star_vs_dijkstra() -> None:
    """Verify A* visits fewer nodes than Dijkstra."""
    grid = GridGraph(20, 20, diagonal=True)
    start = GridPosition(0, 0)
    goal = GridPosition(19, 19)
    
    dijkstra_result = dijkstra_search(grid, start, goal)
    astar_result = a_star_search(grid, start, goal, euclidean_distance)
    
    # Both should find optimal path
    assert abs(dijkstra_result.cost - astar_result.cost) < 1e-10
    
    # A* should visit fewer nodes
    assert astar_result.nodes_visited <= dijkstra_result.nodes_visited
    
    logger.info(
        f"✓ A* visited {astar_result.nodes_visited} nodes vs "
        f"Dijkstra's {dijkstra_result.nodes_visited}"
    )


def run_all_tests() -> None:
    """Run all test cases."""
    logger.info("Running A* tests...")
    
    test_heuristics()
    test_grid_graph()
    test_a_star_simple()
    test_a_star_with_obstacles()
    test_a_star_no_path()
    test_a_star_vs_dijkstra()
    
    logger.info("═" * 50)
    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with demonstration."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="A* Search Algorithm Exercise"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        run_all_tests()
    elif args.demo:
        logger.info("A* Search Demonstration")
        logger.info("=" * 50)
        logger.info("Implement the a_star_search function to see results!")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
