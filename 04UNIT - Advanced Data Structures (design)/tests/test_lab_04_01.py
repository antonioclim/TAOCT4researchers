#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4: Test Suite for Lab 4.1 - Graph Library
═══════════════════════════════════════════════════════════════════════════════

Comprehensive tests for the graph library implementation.

Coverage targets:
- Edge data class
- AdjacencyListGraph operations
- BFS and DFS traversals
- Dijkstra's algorithm
- A* algorithm
- Connected components
- Topological sort
- Cycle detection
- Bipartite checking

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add lab directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_4_01_graph_library import (
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


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: EDGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdge:
    """Tests for the Edge data class."""

    def test_edge_creation(self) -> None:
        """Test basic edge creation."""
        edge = Edge("A", "B", 5.0)
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.weight == 5.0

    def test_edge_default_weight(self) -> None:
        """Test edge with default weight."""
        edge = Edge("A", "B")
        assert edge.weight == 1

    def test_edge_reversed(self) -> None:
        """Test edge reversal."""
        edge = Edge("A", "B", 3.0)
        reversed_edge = edge.reversed()
        assert reversed_edge.source == "B"
        assert reversed_edge.target == "A"
        assert reversed_edge.weight == 3.0

    def test_edge_equality(self) -> None:
        """Test edge equality comparison."""
        edge1 = Edge("A", "B", 5.0)
        edge2 = Edge("A", "B", 5.0)
        edge3 = Edge("A", "B", 10.0)
        assert edge1 == edge2
        assert edge1 != edge3

    def test_edge_hashable(self) -> None:
        """Test that edges are hashable (frozen dataclass)."""
        edge = Edge("A", "B", 5.0)
        edge_set = {edge}
        assert edge in edge_set


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ADJACENCY LIST GRAPH TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdjacencyListGraph:
    """Tests for the AdjacencyListGraph class."""

    def test_empty_graph(self) -> None:
        """Test empty graph creation."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        assert graph.num_nodes() == 0
        assert graph.num_edges() == 0

    def test_add_node(self) -> None:
        """Test adding nodes."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        assert graph.has_node("A")
        assert graph.num_nodes() == 1

    def test_add_duplicate_node(self) -> None:
        """Test adding duplicate node is idempotent."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        graph.add_node("A")
        assert graph.num_nodes() == 1

    def test_add_edge_directed(self) -> None:
        """Test adding edge to directed graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B", 5.0)
        assert graph.has_edge("A", "B")
        assert not graph.has_edge("B", "A")
        assert graph.num_edges() == 1

    def test_add_edge_undirected(self) -> None:
        """Test adding edge to undirected graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        graph.add_edge("A", "B", 5.0)
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "A")
        assert graph.num_edges() == 1  # Counted once

    def test_remove_edge(self) -> None:
        """Test removing an edge."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B", 5.0)
        assert graph.remove_edge("A", "B")
        assert not graph.has_edge("A", "B")

    def test_remove_nonexistent_edge(self) -> None:
        """Test removing non-existent edge returns False."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        graph.add_node("B")
        assert not graph.remove_edge("A", "B")

    def test_get_weight(self) -> None:
        """Test retrieving edge weight."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B", 7.5)
        assert graph.get_weight("A", "B") == 7.5

    def test_get_weight_nonexistent(self) -> None:
        """Test getting weight of non-existent edge."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        graph.add_node("B")
        assert graph.get_weight("A", "B") is None

    def test_neighbours(self) -> None:
        """Test getting neighbours of a node."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        neighbours = set(graph.neighbours("A"))
        assert neighbours == {"B", "C"}

    def test_edges_from(self) -> None:
        """Test getting edges from a node."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("A", "C", 2.0)
        edges = list(graph.edges_from("A"))
        assert len(edges) == 2
        targets = {e.target for e in edges}
        assert targets == {"B", "C"}

    def test_nodes_iterator(self) -> None:
        """Test iterating over nodes."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        nodes = set(graph.nodes())
        assert nodes == {"A", "B", "C"}

    def test_edges_iterator(self) -> None:
        """Test iterating over edges."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        edges = list(graph.edges())
        assert len(edges) == 2

    def test_degree_undirected(self) -> None:
        """Test degree in undirected graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("A", "D")
        assert graph.degree("A") == 3

    def test_in_out_degree_directed(self) -> None:
        """Test in-degree and out-degree in directed graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "A")
        assert graph.out_degree("A") == 2
        assert graph.in_degree("A") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAVERSAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBFS:
    """Tests for breadth-first search."""

    def test_bfs_simple(self, simple_directed_graph: dict) -> None:
        """Test BFS on simple directed graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in simple_directed_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        result = list(bfs(graph, "A"))
        assert result[0] == "A"  # Start node first
        assert set(result) >= {"A", "B", "C"}  # At least these reachable

    def test_bfs_visits_all_reachable(self) -> None:
        """Test BFS visits all reachable nodes."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        
        result = list(bfs(graph, "A"))
        assert set(result) == {"A", "B", "C", "D"}

    def test_bfs_level_order(self) -> None:
        """Test BFS visits nodes level by level."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        # A -> B, C (level 1)
        # B -> D, C -> D (level 2)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")
        
        result = list(bfs(graph, "A"))
        assert result[0] == "A"
        assert result.index("B") < result.index("D")
        assert result.index("C") < result.index("D")

    def test_bfs_disconnected(self) -> None:
        """Test BFS handles disconnected nodes."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B")
        graph.add_node("C")  # Isolated node
        
        result = list(bfs(graph, "A"))
        assert "C" not in result


class TestDFS:
    """Tests for depth-first search."""

    def test_dfs_simple(self) -> None:
        """Test DFS on simple graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        
        result = list(dfs(graph, "A"))
        assert result[0] == "A"
        assert set(result) == {"A", "B", "C", "D"}

    def test_dfs_visits_all_reachable(self) -> None:
        """Test DFS visits all reachable nodes."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "D")
        
        result = list(dfs(graph, "A"))
        assert set(result) == {"A", "B", "C", "D"}

    def test_dfs_handles_cycles(self) -> None:
        """Test DFS handles cyclic graphs."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")  # Cycle
        
        result = list(dfs(graph, "A"))
        assert len(result) == 3  # Each node visited once


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SHORTEST PATH TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDijkstra:
    """Tests for Dijkstra's algorithm."""

    def test_dijkstra_simple(self, weighted_graph: dict) -> None:
        """Test Dijkstra on weighted graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in weighted_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        result = dijkstra(graph, "A")
        assert result.distance_to("D") == 3.0  # A->C->D, not A->B->D

    def test_dijkstra_path_reconstruction(self, weighted_graph: dict) -> None:
        """Test Dijkstra path reconstruction."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in weighted_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        result = dijkstra(graph, "A")
        path = result.path_to("D")
        assert path == ["A", "C", "D"]

    def test_dijkstra_unreachable(self) -> None:
        """Test Dijkstra with unreachable node."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B")
        graph.add_node("C")  # Isolated
        
        result = dijkstra(graph, "A")
        assert result.distance_to("C") == float("inf")
        assert result.path_to("C") is None

    def test_dijkstra_single_node(self) -> None:
        """Test Dijkstra on single node graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        
        result = dijkstra(graph, "A")
        assert result.distance_to("A") == 0


class TestAStar:
    """Tests for A* algorithm."""

    def test_astar_finds_path(self) -> None:
        """Test A* finds shortest path."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B", 1.0)
        graph.add_edge("B", "C", 1.0)
        graph.add_edge("A", "C", 3.0)  # Direct but longer
        
        def heuristic(node: str, goal: str) -> float:
            return 0.0  # Admissible heuristic
        
        result = astar(graph, "A", "C", heuristic)
        assert result.distance_to("C") == 2.0

    def test_astar_with_heuristic(self) -> None:
        """Test A* with non-trivial heuristic."""
        # Grid-like graph
        graph: AdjacencyListGraph[tuple[int, int], float] = AdjacencyListGraph()
        graph.add_edge((0, 0), (1, 0), 1.0)
        graph.add_edge((1, 0), (2, 0), 1.0)
        graph.add_edge((0, 0), (0, 1), 1.0)
        graph.add_edge((0, 1), (1, 1), 1.0)
        graph.add_edge((1, 1), (2, 0), 1.0)
        
        def manhattan(node: tuple[int, int], goal: tuple[int, int]) -> float:
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
        
        result = astar(graph, (0, 0), (2, 0), manhattan)
        assert result.distance_to((2, 0)) == 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GRAPH ANALYSIS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectedComponents:
    """Tests for connected components detection."""

    def test_single_component(self) -> None:
        """Test graph with single component."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        components = find_connected_components(graph)
        assert len(components) == 1
        assert set(components[0]) == {"A", "B", "C"}

    def test_multiple_components(self, disconnected_graph: dict) -> None:
        """Test graph with multiple components."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        for node, edges in disconnected_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        components = find_connected_components(graph)
        assert len(components) == 2

    def test_isolated_nodes(self) -> None:
        """Test isolated nodes form their own components."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")
        
        components = find_connected_components(graph)
        assert len(components) == 3


class TestTopologicalSort:
    """Tests for topological sorting."""

    def test_topological_sort_valid(self, dag_graph: dict) -> None:
        """Test topological sort on valid DAG."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in dag_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        result = topological_sort(graph)
        assert result is not None
        
        # Verify ordering: for each edge (u,v), u comes before v
        positions = {node: i for i, node in enumerate(result)}
        for node, edges in dag_graph.items():
            for target, _ in edges:
                assert positions[node] < positions[target]

    def test_topological_sort_cyclic(self, cyclic_directed_graph: dict) -> None:
        """Test topological sort returns None for cyclic graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in cyclic_directed_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        result = topological_sort(graph)
        assert result is None

    def test_topological_sort_single_node(self) -> None:
        """Test topological sort on single node."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_node("A")
        
        result = topological_sort(graph)
        assert result == ["A"]


class TestCycleDetection:
    """Tests for cycle detection."""

    def test_has_cycle_directed(self, cyclic_directed_graph: dict) -> None:
        """Test cycle detection in directed graph with cycle."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in cyclic_directed_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        assert has_cycle(graph) is True

    def test_no_cycle_dag(self, dag_graph: dict) -> None:
        """Test cycle detection in DAG (no cycle)."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        for node, edges in dag_graph.items():
            for target, weight in edges:
                graph.add_edge(node, target, weight)
        
        assert has_cycle(graph) is False

    def test_has_cycle_undirected(self, cyclic_undirected_graph: dict) -> None:
        """Test cycle detection in undirected graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        # Need to add only one direction as undirected adds both
        added = set()
        for node, edges in cyclic_undirected_graph.items():
            for target, weight in edges:
                if (target, node) not in added:
                    graph.add_edge(node, target, weight)
                    added.add((node, target))
        
        assert has_cycle(graph) is True

    def test_no_cycle_tree(self) -> None:
        """Test tree has no cycle."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        
        assert has_cycle(graph) is False


class TestBipartite:
    """Tests for bipartite checking."""

    def test_is_bipartite(self, bipartite_graph: dict) -> None:
        """Test bipartite graph detection."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        added = set()
        for node, edges in bipartite_graph.items():
            for target, weight in edges:
                if (target, node) not in added:
                    graph.add_edge(node, target, weight)
                    added.add((node, target))
        
        assert is_bipartite(graph) is True

    def test_not_bipartite(self, non_bipartite_graph: dict) -> None:
        """Test non-bipartite graph detection (odd cycle)."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        added = set()
        for node, edges in non_bipartite_graph.items():
            for target, weight in edges:
                if (target, node) not in added:
                    graph.add_edge(node, target, weight)
                    added.add((node, target))
        
        assert is_bipartite(graph) is False

    def test_bipartite_empty(self) -> None:
        """Test empty graph is bipartite."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        assert is_bipartite(graph) is True

    def test_bipartite_single_edge(self) -> None:
        """Test single edge is bipartite."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        graph.add_edge("A", "B")
        assert is_bipartite(graph) is True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EDGE CASES AND PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_self_loop(self) -> None:
        """Test handling of self-loops."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "A")  # Self-loop
        
        assert graph.has_edge("A", "A")
        assert has_cycle(graph) is True

    def test_parallel_edges(self) -> None:
        """Test multiple edges between same nodes."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        graph.add_edge("A", "B", 5.0)
        graph.add_edge("A", "B", 3.0)  # Parallel edge
        
        # Implementation may vary - test basic functionality
        assert graph.has_edge("A", "B")

    def test_numeric_node_labels(self) -> None:
        """Test graph with numeric node labels."""
        graph: AdjacencyListGraph[int, float] = AdjacencyListGraph()
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 3, 1.0)
        
        result = list(bfs(graph, 1))
        assert set(result) == {1, 2, 3}

    def test_large_graph_traversal(self, large_graph: dict) -> None:
        """Test traversal on larger graph."""
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph()
        added = set()
        for node, edges in large_graph.items():
            for target, weight in edges:
                if (target, node) not in added:
                    graph.add_edge(node, target, weight)
                    added.add((node, target))
        
        result = list(bfs(graph, "N0_0"))
        assert len(result) == 100  # All 10x10 nodes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_build_and_analyse_graph(self) -> None:
        """Test building and analysing a complete graph."""
        # Build a small social network
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=False)
        
        edges = [
            ("Alice", "Bob"),
            ("Bob", "Charlie"),
            ("Charlie", "David"),
            ("David", "Eve"),
            ("Eve", "Alice"),  # Creates cycle
        ]
        
        for source, target in edges:
            graph.add_edge(source, target)
        
        # Verify structure
        assert graph.num_nodes() == 5
        assert graph.num_edges() == 5
        
        # Check connectivity
        components = find_connected_components(graph)
        assert len(components) == 1
        
        # Check for cycle
        assert has_cycle(graph) is True
        
        # BFS from Alice
        bfs_order = list(bfs(graph, "Alice"))
        assert bfs_order[0] == "Alice"
        assert set(bfs_order) == {"Alice", "Bob", "Charlie", "David", "Eve"}

    def test_shortest_path_workflow(self) -> None:
        """Test complete shortest path workflow."""
        # Build weighted graph
        graph: AdjacencyListGraph[str, float] = AdjacencyListGraph(directed=True)
        
        graph.add_edge("Home", "Work", 10.0)
        graph.add_edge("Home", "Coffee", 2.0)
        graph.add_edge("Coffee", "Work", 7.0)
        graph.add_edge("Home", "Park", 5.0)
        graph.add_edge("Park", "Work", 3.0)
        
        # Find shortest path
        result = dijkstra(graph, "Home")
        
        # Direct: 10, via Coffee: 9, via Park: 8
        assert result.distance_to("Work") == 8.0
        assert result.path_to("Work") == ["Home", "Park", "Work"]
