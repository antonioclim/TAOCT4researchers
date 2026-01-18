#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
04UNIT Practice: Easy Exercise 3 — Basic BFS
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes
TOPICS: Graph traversal, BFS

TASK
────
Implement breadth-first search to find all reachable nodes.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
from collections import deque


def bfs_reachable(graph: dict[str, list[str]], start: str) -> set[str]:
    """
    Find all nodes reachable from start using BFS.
    
    Args:
        graph: Adjacency list {node: [neighbours]}
        start: Starting node
        
    Returns:
        Set of all reachable nodes (including start)
        
    Example:
        >>> graph = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
        >>> bfs_reachable(graph, "A")
        {'A', 'B', 'C', 'D'}
    """
    # TODO: Implement BFS
    # 1. Create a set for visited nodes
    # 2. Create a queue starting with 'start'
    # 3. While queue not empty:
    #    - Pop front node
    #    - If not visited, mark visited and add neighbours to queue
    # 4. Return visited set
    pass


def bfs_distance(graph: dict[str, list[str]], start: str, target: str) -> int:
    """
    Find shortest distance (number of edges) from start to target.
    
    Args:
        graph: Adjacency list
        start: Starting node
        target: Target node
        
    Returns:
        Shortest distance, or -1 if target unreachable
        
    Example:
        >>> graph = {"A": ["B"], "B": ["C"], "C": []}
        >>> bfs_distance(graph, "A", "C")
        2
    """
    # TODO: Implement BFS with distance tracking
    pass


def bfs_levels(graph: dict[str, list[str]], start: str) -> dict[str, int]:
    """
    Find the level (distance from start) for each reachable node.
    
    Args:
        graph: Adjacency list
        start: Starting node
        
    Returns:
        Dictionary mapping each node to its level
        
    Example:
        >>> graph = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
        >>> bfs_levels(graph, "A")
        {'A': 0, 'B': 1, 'C': 1, 'D': 2}
    """
    # TODO: Implement this
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_bfs_reachable() -> None:
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
        "E": ["F"],  # Disconnected
        "F": [],
    }
    
    result = bfs_reachable(graph, "A")
    assert result == {"A", "B", "C", "D"}


def test_bfs_distance() -> None:
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": ["E"],
        "E": [],
    }
    
    assert bfs_distance(graph, "A", "D") == 2
    assert bfs_distance(graph, "A", "E") == 3
    assert bfs_distance(graph, "A", "A") == 0
    assert bfs_distance(graph, "A", "X") == -1


def test_bfs_levels() -> None:
    graph = {
        "A": ["B", "C"],
        "B": ["D"],
        "C": ["D"],
        "D": [],
    }
    
    levels = bfs_levels(graph, "A")
    assert levels["A"] == 0
    assert levels["B"] == 1
    assert levels["C"] == 1
    assert levels["D"] == 2


if __name__ == "__main__":
    test_bfs_reachable()
    test_bfs_distance()
    test_bfs_levels()
    print("All tests passed! ✓")
