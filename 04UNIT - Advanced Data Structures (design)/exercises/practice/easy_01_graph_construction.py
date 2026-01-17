#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 4 Practice: Easy Exercise 1 — Graph Construction
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 15 minutes
TOPICS: Graph basics, adjacency list

TASK
────
Implement a function to construct a graph from a list of edges.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations


def build_graph(edges: list[tuple[str, str, int]], directed: bool = False) -> dict[str, dict[str, int]]:
    """
    Build an adjacency list graph from a list of edges.
    
    Args:
        edges: List of (source, target, weight) tuples
        directed: If True, create directed graph; otherwise undirected
        
    Returns:
        Adjacency list representation: {node: {neighbour: weight}}
        
    Example:
        >>> edges = [("A", "B", 1), ("B", "C", 2)]
        >>> g = build_graph(edges, directed=False)
        >>> g["A"]["B"]
        1
        >>> g["B"]["A"]
        1
    """
    # TODO: Implement this function
    # 1. Create an empty dictionary for the adjacency list
    # 2. For each edge, add both nodes if not present
    # 3. Add the edge (and reverse if undirected)
    pass


def count_nodes(graph: dict[str, dict[str, int]]) -> int:
    """
    Count the number of nodes in a graph.
    
    Args:
        graph: Adjacency list representation
        
    Returns:
        Number of nodes
    """
    # TODO: Implement this function
    pass


def count_edges(graph: dict[str, dict[str, int]], directed: bool = False) -> int:
    """
    Count the number of edges in a graph.
    
    Args:
        graph: Adjacency list representation
        directed: Whether the graph is directed
        
    Returns:
        Number of edges
    """
    # TODO: Implement this function
    # Hint: For undirected graphs, each edge is stored twice
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_build_graph_undirected() -> None:
    edges = [("A", "B", 1), ("B", "C", 2), ("A", "C", 3)]
    g = build_graph(edges, directed=False)
    
    assert g["A"]["B"] == 1
    assert g["B"]["A"] == 1  # Reverse edge
    assert g["B"]["C"] == 2
    assert g["A"]["C"] == 3


def test_build_graph_directed() -> None:
    edges = [("A", "B", 1), ("B", "C", 2)]
    g = build_graph(edges, directed=True)
    
    assert g["A"]["B"] == 1
    assert "A" not in g.get("B", {})  # No reverse edge


def test_count_nodes() -> None:
    edges = [("A", "B", 1), ("B", "C", 2), ("C", "D", 3)]
    g = build_graph(edges)
    assert count_nodes(g) == 4


def test_count_edges() -> None:
    edges = [("A", "B", 1), ("B", "C", 2), ("A", "C", 3)]
    g = build_graph(edges, directed=False)
    assert count_edges(g, directed=False) == 3


if __name__ == "__main__":
    test_build_graph_undirected()
    test_build_graph_directed()
    test_count_nodes()
    test_count_edges()
    print("All tests passed! ✓")
