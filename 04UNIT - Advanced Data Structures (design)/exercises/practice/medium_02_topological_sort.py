#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 4 Practice: Medium Exercise 2 — Topological Sort
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 25 minutes
TOPICS: DAG, topological ordering, dependency resolution

TASK
────
Implement topological sort using both Kahn's algorithm and DFS.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
from collections import deque


def topological_sort_kahn(graph: dict[str, list[str]]) -> list[str] | None:
    """
    Topological sort using Kahn's algorithm (BFS-based).
    
    Repeatedly removes nodes with zero in-degree.
    
    Args:
        graph: Directed graph as adjacency list
        
    Returns:
        List of nodes in topological order, or None if cycle exists
        
    Example:
        >>> graph = {"compile": ["link"], "test": ["link"], "link": []}
        >>> topological_sort_kahn(graph)
        ['compile', 'test', 'link']  # or ['test', 'compile', 'link']
    """
    # TODO: Implement Kahn's algorithm
    # 1. Calculate in-degree for each node
    # 2. Add all nodes with in-degree 0 to queue
    # 3. While queue not empty:
    #    - Remove node, add to result
    #    - Decrease in-degree of neighbours
    #    - If neighbour's in-degree becomes 0, add to queue
    # 4. If result length != node count, there's a cycle
    pass


def topological_sort_dfs(graph: dict[str, list[str]]) -> list[str] | None:
    """
    Topological sort using DFS.
    
    Nodes are added to result in reverse post-order.
    
    Args:
        graph: Directed graph as adjacency list
        
    Returns:
        List of nodes in topological order, or None if cycle exists
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {node: WHITE for node in graph}
    result: list[str] = []
    has_cycle = False
    
    def dfs(node: str) -> None:
        nonlocal has_cycle
        # TODO: Implement DFS-based topological sort
        # 1. Mark node GREY
        # 2. Visit all WHITE neighbours (detect cycle if GREY)
        # 3. Mark node BLACK and prepend to result
        pass
    
    # TODO: Call dfs on all WHITE nodes
    pass


def is_valid_topological_order(graph: dict[str, list[str]], order: list[str]) -> bool:
    """
    Verify if a given order is a valid topological sort.
    
    Args:
        graph: Directed graph
        order: Proposed topological order
        
    Returns:
        True if order is valid
    """
    # TODO: Implement validation
    # For each edge (u, v), u must appear before v in the order
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_kahn_simple() -> None:
    # Linear dependency: A → B → C
    graph = {"A": ["B"], "B": ["C"], "C": []}
    result = topological_sort_kahn(graph)
    assert result == ["A", "B", "C"]


def test_kahn_multiple_valid() -> None:
    # A → C, B → C (both A and B can come first)
    graph = {"A": ["C"], "B": ["C"], "C": []}
    result = topological_sort_kahn(graph)
    assert result is not None
    assert is_valid_topological_order(graph, result)


def test_kahn_cycle() -> None:
    # Cycle: A → B → C → A
    graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
    result = topological_sort_kahn(graph)
    assert result is None


def test_dfs_simple() -> None:
    graph = {"A": ["B"], "B": ["C"], "C": []}
    result = topological_sort_dfs(graph)
    assert result == ["A", "B", "C"]


def test_validation() -> None:
    graph = {"A": ["B"], "B": ["C"], "C": []}
    assert is_valid_topological_order(graph, ["A", "B", "C"]) is True
    assert is_valid_topological_order(graph, ["C", "B", "A"]) is False


if __name__ == "__main__":
    test_kahn_simple()
    test_kahn_multiple_valid()
    test_kahn_cycle()
    test_dfs_simple()
    test_validation()
    print("All tests passed! ✓")
