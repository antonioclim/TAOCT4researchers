#!/usr/bin/env python3
"""
Solution: Cycle Detection in Graphs
====================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements cycle detection algorithms for both directed
and undirected graphs using DFS-based approaches. Includes detection
of back edges, cycle extraction, and topological ordering validation.

Complexity Analysis:
    - Directed Graph Cycle Detection: O(V + E) time, O(V) space
    - Undirected Graph Cycle Detection: O(V + E) time, O(V) space
    - Cycle Extraction: O(V) additional time when cycle is found

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Generic, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


class VertexState(Enum):
    """
    Vertex states during DFS traversal.
    
    The three-colour marking scheme is essential for cycle detection
    in directed graphs:
    
    - UNVISITED (white): Vertex has not been discovered
    - VISITING (grey): Vertex is in the current DFS path (on the stack)
    - VISITED (black): Vertex and all descendants have been processed
    
    A back edge (cycle) exists when we encounter a VISITING vertex.
    """
    UNVISITED = auto()
    VISITING = auto()
    VISITED = auto()


@dataclass
class Graph(Generic[T]):
    """
    Graph class supporting both directed and undirected representations.
    
    Attributes:
        directed: Whether edges are directed.
        _adjacency: Adjacency list mapping vertices to neighbours.
    """
    directed: bool = True
    _adjacency: dict[T, list[T]] = field(
        default_factory=lambda: defaultdict(list)
    )
    
    def add_edge(self, source: T, target: T) -> None:
        """Add an edge to the graph."""
        # Ensure both vertices exist
        if source not in self._adjacency:
            self._adjacency[source] = []
        if target not in self._adjacency:
            self._adjacency[target] = []
        
        self._adjacency[source].append(target)
        
        if not self.directed:
            self._adjacency[target].append(source)
    
    def add_vertex(self, vertex: T) -> None:
        """Add an isolated vertex."""
        if vertex not in self._adjacency:
            self._adjacency[vertex] = []
    
    def neighbours(self, vertex: T) -> list[T]:
        """Return neighbours of a vertex."""
        return self._adjacency.get(vertex, [])
    
    @property
    def vertices(self) -> set[T]:
        """Return all vertices."""
        return set(self._adjacency.keys())
    
    @property
    def edge_count(self) -> int:
        """Return the number of edges."""
        total = sum(len(adj) for adj in self._adjacency.values())
        return total if self.directed else total // 2


@dataclass
class CycleDetectionResult(Generic[T]):
    """
    Result of cycle detection.
    
    Attributes:
        has_cycle: Whether a cycle was detected.
        cycle: The cycle as a list of vertices (if found).
        back_edge: The back edge that closes the cycle.
    """
    has_cycle: bool = False
    cycle: list[T] = field(default_factory=list)
    back_edge: tuple[T, T] | None = None
    
    def __str__(self) -> str:
        """Format the result for display."""
        if not self.has_cycle:
            return "No cycle detected"
        
        cycle_str = " -> ".join(str(v) for v in self.cycle)
        return f"Cycle found: {cycle_str}"


def detect_cycle_directed(graph: Graph[T]) -> CycleDetectionResult[T]:
    """
    Detect cycles in a directed graph using DFS.
    
    Uses the three-colour (white-grey-black) marking scheme. A cycle
    exists if and only if we encounter a grey vertex (one currently
    in our DFS path) while exploring.
    
    The algorithm tracks the recursion stack to extract the actual
    cycle when one is found.
    
    Args:
        graph: A directed graph.
    
    Returns:
        CycleDetectionResult with cycle information.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    
    Examples:
        >>> g = Graph[str](directed=True)
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('B', 'C')
        >>> g.add_edge('C', 'A')  # Creates cycle
        >>> result = detect_cycle_directed(g)
        >>> result.has_cycle
        True
    """
    state: dict[T, VertexState] = {
        v: VertexState.UNVISITED for v in graph.vertices
    }
    parent: dict[T, T | None] = {}
    result = CycleDetectionResult[T]()
    
    def dfs(vertex: T, path: list[T]) -> bool:
        """
        DFS traversal with cycle detection.
        
        Args:
            vertex: Current vertex being explored.
            path: Current DFS path (recursion stack).
        
        Returns:
            True if a cycle is found, False otherwise.
        """
        state[vertex] = VertexState.VISITING
        path.append(vertex)
        
        logger.debug(f"Visiting {vertex}, path: {path}")
        
        for neighbour in graph.neighbours(vertex):
            if state[neighbour] == VertexState.UNVISITED:
                parent[neighbour] = vertex
                if dfs(neighbour, path):
                    return True
            
            elif state[neighbour] == VertexState.VISITING:
                # Back edge found - cycle detected!
                logger.debug(f"Back edge found: {vertex} -> {neighbour}")
                result.has_cycle = True
                result.back_edge = (vertex, neighbour)
                
                # Extract the cycle from the path
                cycle_start = path.index(neighbour)
                result.cycle = path[cycle_start:] + [neighbour]
                return True
        
        state[vertex] = VertexState.VISITED
        path.pop()
        return False
    
    # Run DFS from each unvisited vertex
    for vertex in graph.vertices:
        if state[vertex] == VertexState.UNVISITED:
            if dfs(vertex, []):
                logger.info(f"Cycle detected: {result.cycle}")
                return result
    
    logger.info("No cycles detected in directed graph")
    return result


def detect_cycle_undirected(graph: Graph[T]) -> CycleDetectionResult[T]:
    """
    Detect cycles in an undirected graph using DFS.
    
    For undirected graphs, we need to track the parent vertex to
    avoid false positives from the edge we just traversed.
    A cycle exists if we visit a vertex that is already visited
    and is not the parent of the current vertex.
    
    Args:
        graph: An undirected graph.
    
    Returns:
        CycleDetectionResult with cycle information.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    
    Examples:
        >>> g = Graph[int](directed=False)
        >>> g.add_edge(1, 2)
        >>> g.add_edge(2, 3)
        >>> g.add_edge(3, 1)  # Creates cycle
        >>> result = detect_cycle_undirected(g)
        >>> result.has_cycle
        True
    """
    visited: set[T] = set()
    parent: dict[T, T | None] = {}
    result = CycleDetectionResult[T]()
    
    def dfs(vertex: T, prev: T | None, path: list[T]) -> bool:
        """
        DFS with parent tracking for undirected graphs.
        
        Args:
            vertex: Current vertex.
            prev: Parent vertex (where we came from).
            path: Current path for cycle extraction.
        
        Returns:
            True if cycle found.
        """
        visited.add(vertex)
        parent[vertex] = prev
        path.append(vertex)
        
        for neighbour in graph.neighbours(vertex):
            if neighbour not in visited:
                if dfs(neighbour, vertex, path):
                    return True
            
            elif neighbour != prev:
                # Visited vertex that isn't our parent = cycle
                logger.debug(f"Cycle edge found: {vertex} -> {neighbour}")
                result.has_cycle = True
                result.back_edge = (vertex, neighbour)
                
                # Extract cycle
                cycle_start = path.index(neighbour)
                result.cycle = path[cycle_start:] + [neighbour]
                return True
        
        path.pop()
        return False
    
    # Check all components
    for vertex in graph.vertices:
        if vertex not in visited:
            if dfs(vertex, None, []):
                logger.info(f"Cycle detected: {result.cycle}")
                return result
    
    logger.info("No cycles detected in undirected graph")
    return result


def detect_cycle(graph: Graph[T]) -> CycleDetectionResult[T]:
    """
    Detect cycles in a graph (directed or undirected).
    
    Automatically selects the appropriate algorithm based on
    the graph's directed attribute.
    
    Args:
        graph: The graph to check.
    
    Returns:
        CycleDetectionResult with cycle information.
    """
    if graph.directed:
        return detect_cycle_directed(graph)
    else:
        return detect_cycle_undirected(graph)


def find_all_cycles_directed(graph: Graph[T]) -> list[list[T]]:
    """
    Find all simple cycles in a directed graph.
    
    Uses Johnson's algorithm concept to find all elementary cycles.
    Note: This can be exponential in the number of cycles.
    
    Args:
        graph: A directed graph.
    
    Returns:
        List of cycles, each as a list of vertices.
    
    Time Complexity: O((V + E) * C) where C is the number of cycles.
    """
    cycles: list[list[T]] = []
    blocked: set[T] = set()
    blocked_map: dict[T, set[T]] = defaultdict(set)
    stack: list[T] = []
    
    def unblock(vertex: T) -> None:
        """Remove vertex from blocked set and propagate."""
        blocked.discard(vertex)
        while blocked_map[vertex]:
            w = blocked_map[vertex].pop()
            if w in blocked:
                unblock(w)
    
    def circuit(vertex: T, start: T) -> bool:
        """Find cycles starting from start that pass through vertex."""
        found_cycle = False
        stack.append(vertex)
        blocked.add(vertex)
        
        for neighbour in graph.neighbours(vertex):
            if neighbour == start:
                # Found a cycle back to start
                cycles.append(stack + [start])
                found_cycle = True
            elif neighbour not in blocked:
                if circuit(neighbour, start):
                    found_cycle = True
        
        if found_cycle:
            unblock(vertex)
        else:
            for neighbour in graph.neighbours(vertex):
                blocked_map[neighbour].add(vertex)
        
        stack.pop()
        return found_cycle
    
    # Find cycles starting from each vertex
    vertices = sorted(graph.vertices, key=str)  # Consistent ordering
    
    for i, start in enumerate(vertices):
        # Only consider vertices >= start to avoid duplicate cycles
        circuit(start, start)
        blocked.clear()
        blocked_map.clear()
    
    logger.info(f"Found {len(cycles)} cycles")
    return cycles


def is_dag(graph: Graph[T]) -> bool:
    """
    Check if a directed graph is a Directed Acyclic Graph (DAG).
    
    A DAG is a directed graph with no cycles. DAGs are important
    for topological sorting, dependency resolution, and scheduling.
    
    Args:
        graph: A directed graph.
    
    Returns:
        True if the graph is a DAG, False if it contains cycles.
    
    Time Complexity: O(V + E)
    """
    result = detect_cycle_directed(graph)
    return not result.has_cycle


def find_back_edges(graph: Graph[T]) -> list[tuple[T, T]]:
    """
    Find all back edges in a directed graph.
    
    A back edge is an edge from a vertex to one of its ancestors
    in the DFS tree. Back edges indicate cycles.
    
    Args:
        graph: A directed graph.
    
    Returns:
        List of back edges as (source, target) tuples.
    """
    back_edges: list[tuple[T, T]] = []
    state: dict[T, VertexState] = {
        v: VertexState.UNVISITED for v in graph.vertices
    }
    
    def dfs(vertex: T) -> None:
        """DFS to find back edges."""
        state[vertex] = VertexState.VISITING
        
        for neighbour in graph.neighbours(vertex):
            if state[neighbour] == VertexState.UNVISITED:
                dfs(neighbour)
            elif state[neighbour] == VertexState.VISITING:
                back_edges.append((vertex, neighbour))
        
        state[vertex] = VertexState.VISITED
    
    for vertex in graph.vertices:
        if state[vertex] == VertexState.UNVISITED:
            dfs(vertex)
    
    return back_edges


def demonstrate_cycle_detection() -> None:
    """Demonstrate cycle detection algorithms."""
    print("=" * 60)
    print("Cycle Detection Demonstration")
    print("=" * 60)
    
    # Example 1: Directed graph with cycle
    print("\n1. Directed Graph with Cycle")
    print("-" * 40)
    
    directed_cycle: Graph[str] = Graph(directed=True)
    edges_dc = [
        ('A', 'B'), ('B', 'C'), ('C', 'D'),
        ('D', 'B'),  # Creates cycle B -> C -> D -> B
        ('A', 'E'), ('E', 'F'),
    ]
    for u, v in edges_dc:
        directed_cycle.add_edge(u, v)
    
    result = detect_cycle_directed(directed_cycle)
    print(f"   {result}")
    if result.back_edge:
        print(f"   Back edge: {result.back_edge[0]} -> {result.back_edge[1]}")
    
    # Example 2: DAG (no cycles)
    print("\n2. Directed Acyclic Graph (DAG)")
    print("-" * 40)
    
    dag: Graph[str] = Graph(directed=True)
    edges_dag = [
        ('A', 'B'), ('A', 'C'),
        ('B', 'D'), ('C', 'D'),
        ('D', 'E'),
    ]
    for u, v in edges_dag:
        dag.add_edge(u, v)
    
    result = detect_cycle_directed(dag)
    print(f"   {result}")
    print(f"   Is DAG: {is_dag(dag)}")
    
    # Example 3: Undirected graph with cycle
    print("\n3. Undirected Graph with Cycle")
    print("-" * 40)
    
    undirected_cycle: Graph[int] = Graph(directed=False)
    edges_uc = [
        (1, 2), (2, 3), (3, 4),
        (4, 2),  # Creates cycle
        (1, 5),
    ]
    for u, v in edges_uc:
        undirected_cycle.add_edge(u, v)
    
    result = detect_cycle_undirected(undirected_cycle)
    print(f"   {result}")
    
    # Example 4: Tree (no cycles)
    print("\n4. Tree (No Cycles)")
    print("-" * 40)
    
    tree: Graph[str] = Graph(directed=False)
    edges_tree = [
        ('root', 'left'), ('root', 'right'),
        ('left', 'L1'), ('left', 'L2'),
        ('right', 'R1'),
    ]
    for u, v in edges_tree:
        tree.add_edge(u, v)
    
    result = detect_cycle_undirected(tree)
    print(f"   {result}")
    
    # Example 5: Multiple cycles
    print("\n5. Graph with Multiple Cycles")
    print("-" * 40)
    
    multi_cycle: Graph[int] = Graph(directed=True)
    edges_mc = [
        (1, 2), (2, 3), (3, 1),  # Cycle 1
        (3, 4), (4, 5), (5, 3),  # Cycle 2
        (1, 6), (6, 7),
    ]
    for u, v in edges_mc:
        multi_cycle.add_edge(u, v)
    
    back_edges = find_back_edges(multi_cycle)
    print(f"   Number of back edges: {len(back_edges)}")
    print(f"   Back edges: {back_edges}")
    
    # Example 6: Self-loop
    print("\n6. Self-Loop Detection")
    print("-" * 40)
    
    self_loop: Graph[str] = Graph(directed=True)
    self_loop.add_edge('A', 'B')
    self_loop.add_edge('B', 'B')  # Self-loop
    self_loop.add_edge('B', 'C')
    
    result = detect_cycle_directed(self_loop)
    print(f"   {result}")
    
    # Example 7: Dependency graph
    print("\n7. Real-World Example: Task Dependencies")
    print("-" * 40)
    
    tasks: Graph[str] = Graph(directed=True)
    dependencies = [
        ('compile', 'link'),
        ('link', 'test'),
        ('test', 'deploy'),
        ('deploy', 'compile'),  # Circular dependency!
    ]
    for dep, task in dependencies:
        tasks.add_edge(dep, task)
    
    result = detect_cycle_directed(tasks)
    if result.has_cycle:
        print("   ERROR: Circular dependency detected!")
        print(f"   {result}")
    else:
        print("   Dependencies are valid (no cycles)")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_cycle_detection()
