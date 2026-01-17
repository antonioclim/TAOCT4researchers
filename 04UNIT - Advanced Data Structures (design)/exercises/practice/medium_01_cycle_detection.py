#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 4 Practice: Medium Exercise 1 — Cycle Detection
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 25 minutes
TOPICS: DFS, cycle detection, graph colouring

TASK
────
Implement cycle detection for both directed and undirected graphs.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations


def has_cycle_directed(graph: dict[str, list[str]]) -> bool:
    """
    Detect if a directed graph contains a cycle.
    
    Uses DFS with three colours:
    - WHITE (0): unvisited
    - GREY (1): being processed (on current DFS path)
    - BLACK (2): completely processed
    
    A back edge (edge to GREY node) indicates a cycle.
    
    Args:
        graph: Directed graph as adjacency list
        
    Returns:
        True if the graph contains a cycle
        
    Example:
        >>> graph = {"A": ["B"], "B": ["C"], "C": ["A"]}  # A→B→C→A cycle
        >>> has_cycle_directed(graph)
        True
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {node: WHITE for node in graph}
    
    def dfs(node: str) -> bool:
        # TODO: Implement DFS with cycle detection
        # 1. Mark node as GREY
        # 2. For each neighbour:
        #    - If GREY, cycle found
        #    - If WHITE, recurse
        # 3. Mark node as BLACK
        # 4. Return False if no cycle found
        pass
    
    # TODO: Call dfs on all WHITE nodes
    pass


def has_cycle_undirected(graph: dict[str, list[str]]) -> bool:
    """
    Detect if an undirected graph contains a cycle.
    
    Uses DFS tracking the parent to avoid counting the edge we came from.
    
    Args:
        graph: Undirected graph as adjacency list
        
    Returns:
        True if the graph contains a cycle
        
    Example:
        >>> graph = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}  # Triangle
        >>> has_cycle_undirected(graph)
        True
    """
    visited: set[str] = set()
    
    def dfs(node: str, parent: str | None) -> bool:
        # TODO: Implement DFS with parent tracking
        # A cycle exists if we visit a node that's already visited
        # (and it's not the parent we came from)
        pass
    
    # TODO: Call dfs on all unvisited nodes
    pass


def find_cycle(graph: dict[str, list[str]]) -> list[str] | None:
    """
    Find and return a cycle in a directed graph.
    
    Args:
        graph: Directed graph as adjacency list
        
    Returns:
        List of nodes forming a cycle, or None if no cycle exists
        
    Example:
        >>> graph = {"A": ["B"], "B": ["C"], "C": ["A"]}
        >>> find_cycle(graph)
        ['A', 'B', 'C', 'A']  # or similar cycle
    """
    # TODO: Implement cycle finding (bonus)
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_directed_cycle() -> None:
    # Cycle: A → B → C → A
    graph_with_cycle = {"A": ["B"], "B": ["C"], "C": ["A"]}
    assert has_cycle_directed(graph_with_cycle) is True
    
    # No cycle: A → B → C
    graph_no_cycle = {"A": ["B"], "B": ["C"], "C": []}
    assert has_cycle_directed(graph_no_cycle) is False


def test_undirected_cycle() -> None:
    # Triangle: A-B-C-A
    triangle = {"A": ["B", "C"], "B": ["A", "C"], "C": ["A", "B"]}
    assert has_cycle_undirected(triangle) is True
    
    # Tree: no cycle
    tree = {"A": ["B", "C"], "B": ["A"], "C": ["A"]}
    assert has_cycle_undirected(tree) is False


def test_self_loop() -> None:
    # Self loop is a cycle
    self_loop = {"A": ["A"]}
    assert has_cycle_directed(self_loop) is True


if __name__ == "__main__":
    test_directed_cycle()
    test_undirected_cycle()
    test_self_loop()
    print("All tests passed! ✓")
