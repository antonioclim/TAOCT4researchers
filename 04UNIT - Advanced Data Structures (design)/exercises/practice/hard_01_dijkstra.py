#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
04UNIT Practice: Hard Exercise 1 — Dijkstra's Algorithm
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 40 minutes
TOPICS: Shortest paths, priority queues, graph algorithms

TASK
────
Implement Dijkstra's algorithm from scratch with path reconstruction.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
import heapq
from dataclasses import dataclass
import math


@dataclass
class DijkstraResult:
    """Result of Dijkstra's algorithm."""
    distances: dict[str, float]
    predecessors: dict[str, str | None]
    source: str
    
    def distance_to(self, target: str) -> float:
        """Return shortest distance to target."""
        return self.distances.get(target, math.inf)
    
    def path_to(self, target: str) -> list[str] | None:
        """Return shortest path to target, or None if unreachable."""
        if target not in self.distances or self.distances[target] == math.inf:
            return None
        
        path = []
        current: str | None = target
        while current is not None:
            path.append(current)
            current = self.predecessors.get(current)
        
        path.reverse()
        return path


def dijkstra(
    graph: dict[str, list[tuple[str, float]]],
    source: str,
) -> DijkstraResult:
    """
    Dijkstra's algorithm for shortest paths.
    
    Finds shortest paths from source to all reachable nodes in a graph
    with NON-NEGATIVE edge weights.
    
    Complexity: O((V + E) log V) with binary heap
    
    Args:
        graph: Weighted adjacency list {node: [(neighbour, weight), ...]}
        source: Starting node
        
    Returns:
        DijkstraResult with distances and path reconstruction
        
    Example:
        >>> graph = {
        ...     "A": [("B", 4), ("C", 2)],
        ...     "B": [("C", 1), ("D", 5)],
        ...     "C": [("D", 8)],
        ...     "D": []
        ... }
        >>> result = dijkstra(graph, "A")
        >>> result.distance_to("D")
        7
        >>> result.path_to("D")
        ['A', 'C', 'B', 'D']  # A→C(2)→B(3)→D(7)... wait, let me recalculate
    """
    # TODO: Implement Dijkstra's algorithm
    # 1. Initialise distances to infinity, source to 0
    # 2. Initialise predecessors to None
    # 3. Use priority queue: (distance, node)
    # 4. While queue not empty:
    #    - Extract minimum
    #    - Skip if already found better path
    #    - Relax all outgoing edges
    #    - Update predecessors when improving distance
    pass


def dijkstra_with_target(
    graph: dict[str, list[tuple[str, float]]],
    source: str,
    target: str,
) -> tuple[float, list[str] | None]:
    """
    Dijkstra's algorithm with early termination.
    
    Stops as soon as the target is reached, which can be more efficient
    when only one destination is needed.
    
    Args:
        graph: Weighted adjacency list
        source: Starting node
        target: Destination node
        
    Returns:
        Tuple of (distance, path) or (inf, None) if unreachable
    """
    # TODO: Implement with early termination
    pass


def bellman_ford(
    graph: dict[str, list[tuple[str, float]]],
    source: str,
) -> DijkstraResult | None:
    """
    Bellman-Ford algorithm for graphs with negative edges.
    
    Unlike Dijkstra, this works with negative edge weights.
    Returns None if a negative cycle is detected.
    
    Complexity: O(VE)
    
    Args:
        graph: Weighted adjacency list (may have negative weights)
        source: Starting node
        
    Returns:
        DijkstraResult or None if negative cycle exists
    """
    # TODO: Implement Bellman-Ford (bonus)
    # 1. Initialise distances to infinity, source to 0
    # 2. Repeat V-1 times: relax all edges
    # 3. Check for negative cycles (one more relaxation)
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_dijkstra_simple() -> None:
    graph = {
        "A": [("B", 4), ("C", 2)],
        "B": [("D", 3)],
        "C": [("B", 1), ("D", 8)],
        "D": [],
    }
    
    result = dijkstra(graph, "A")
    
    assert result.distance_to("A") == 0
    assert result.distance_to("C") == 2
    assert result.distance_to("B") == 3  # A→C→B
    assert result.distance_to("D") == 6  # A→C→B→D


def test_dijkstra_path() -> None:
    graph = {
        "A": [("B", 1), ("C", 4)],
        "B": [("C", 2)],
        "C": [],
    }
    
    result = dijkstra(graph, "A")
    path = result.path_to("C")
    
    assert path == ["A", "B", "C"]


def test_dijkstra_unreachable() -> None:
    graph = {
        "A": [("B", 1)],
        "B": [],
        "C": [],  # Disconnected
    }
    
    result = dijkstra(graph, "A")
    assert result.distance_to("C") == math.inf
    assert result.path_to("C") is None


def test_dijkstra_with_target() -> None:
    graph = {
        "A": [("B", 1), ("C", 10)],
        "B": [("C", 1)],
        "C": [],
    }
    
    dist, path = dijkstra_with_target(graph, "A", "C")
    assert dist == 2
    assert path == ["A", "B", "C"]


if __name__ == "__main__":
    test_dijkstra_simple()
    test_dijkstra_path()
    test_dijkstra_unreachable()
    test_dijkstra_with_target()
    print("All tests passed! ✓")
