"""
═══════════════════════════════════════════════════════════════════════════════
04UNIT: Advanced Data Structures - Laboratory Package
═══════════════════════════════════════════════════════════════════════════════

This package contains the laboratory exercises for 04UNIT.

Modules:
    lab_4_01_graph_library: Graph data structures and algorithms
    lab_4_02_probabilistic_ds: Bloom filters and Count-Min sketch

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
"""

from .lab_4_01_graph_library import (
    AdjacencyListGraph,
    Edge,
    ShortestPathResult,
    astar,
    bfs,
    dfs,
    dijkstra,
    find_connected_components,
    has_cycle,
    is_bipartite,
    topological_sort,
)
from .lab_4_02_probabilistic_ds import BloomFilter, CountMinSketch

__all__ = [
    # Graph structures
    "AdjacencyListGraph",
    "Edge",
    "ShortestPathResult",
    # Traversal algorithms
    "bfs",
    "dfs",
    # Shortest path algorithms
    "dijkstra",
    "astar",
    # Graph analysis
    "find_connected_components",
    "topological_sort",
    "has_cycle",
    "is_bipartite",
    # Probabilistic structures
    "BloomFilter",
    "CountMinSketch",
]
