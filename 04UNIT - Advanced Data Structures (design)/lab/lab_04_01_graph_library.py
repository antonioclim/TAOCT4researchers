#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
04UNIT, Lab 1: Graph Library
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Graphs are ubiquitous in research: social networks model relationships between
individuals, protein interaction networks reveal biological pathways, road
networks enable navigation, and dependency graphs structure build systems.
This library provides an educational implementation covering multiple
representations, classic algorithms and practical applications.

PREREQUISITES
─────────────
- 03UNIT: Complexity analysis, Big-O notation, benchmarking
- Python: Classes, generics, type hints, protocols
- Mathematics: Basic graph theory terminology

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement graph data structures with adjacency list representation
2. Apply BFS, DFS, Dijkstra and A* algorithms correctly
3. Analyse trade-offs between different graph representations
4. Detect cycles and compute topological orderings

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 90 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
- Python 3.12+
- networkx >= 3.0 (optional, for validation)
- matplotlib >= 3.7 (optional, for visualisation)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Callable,
    Generic,
    Hashable,
    Iterator,
    Protocol,
    TypeVar,
    runtime_checkable,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type variables for generic node and weight types
N = TypeVar("N", bound=Hashable)  # Node type
W = TypeVar("W", int, float)  # Weight type


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Edge(Generic[N, W]):
    """
    An edge in a graph.

    Attributes:
        source: The source node of the edge
        target: The target node of the edge
        weight: The weight/cost of the edge (default: 1)

    Example:
        >>> edge = Edge("London", "Paris", 344)
        >>> edge.reversed()
        Edge(source='Paris', target='London', weight=344)
    """

    source: N
    target: N
    weight: W = 1  # type: ignore

    def reversed(self) -> Edge[N, W]:
        """Return the edge with source and target swapped."""
        return Edge(self.target, self.source, self.weight)


@runtime_checkable
class Graph(Protocol[N, W]):
    """
    Protocol defining the interface for graph implementations.

    This protocol enables polymorphic code that works with any graph
    representation (adjacency list, matrix, etc.) as long as it
    provides the required operations.
    """

    def add_node(self, node: N) -> None:
        """Add a node to the graph."""
        ...

    def add_edge(self, source: N, target: N, weight: W = ...) -> None:
        """Add an edge to the graph."""
        ...

    def has_node(self, node: N) -> bool:
        """Check if a node exists in the graph."""
        ...

    def has_edge(self, source: N, target: N) -> bool:
        """Check if an edge exists in the graph."""
        ...

    def neighbours(self, node: N) -> Iterator[N]:
        """Return an iterator over the neighbours of a node."""
        ...

    def edges_from(self, node: N) -> Iterator[Edge[N, W]]:
        """Return an iterator over edges originating from a node."""
        ...

    def nodes(self) -> Iterator[N]:
        """Return an iterator over all nodes in the graph."""
        ...

    def edges(self) -> Iterator[Edge[N, W]]:
        """Return an iterator over all edges in the graph."""
        ...

    def num_nodes(self) -> int:
        """The number of nodes in the graph."""
        ...

    def num_edges(self) -> int:
        """The number of edges in the graph."""
        ...

    @property
    def is_directed(self) -> bool:
        """True if the graph is directed, False otherwise."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: ADJACENCY LIST IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class AdjacencyListGraph(Generic[N, W]):
    """
    Graph implemented using adjacency lists.

    Trade-offs:
    ───────────
    Space: O(V + E) — optimal for sparse graphs
    Add node: O(1)
    Add edge: O(1)
    Check edge: O(degree) — slower than matrix
    Get neighbours: O(degree) — optimal
    Iterate all edges: O(E) — optimal

    Recommended for:
    ────────────────
    - Sparse graphs (E << V²)
    - Frequent traversals (BFS, DFS)
    - Large graphs where memory matters

    Example:
        >>> g = AdjacencyListGraph[str, int](directed=False)
        >>> g.add_edge("A", "B", 5)
        >>> g.add_edge("B", "C", 3)
        >>> list(g.neighbours("B"))
        ['A', 'C']
    """

    def __init__(self, directed: bool = False) -> None:
        """
        Initialise an empty graph.

        Args:
            directed: If True, create a directed graph; otherwise undirected
        """
        self._directed = directed
        self._adj: dict[N, dict[N, W]] = {}
        self._num_edges = 0

    def add_node(self, node: N) -> None:
        """Add a node to the graph if it does not already exist."""
        if node not in self._adj:
            self._adj[node] = {}
            logger.debug("Added node: %s", node)

    def add_edge(self, source: N, target: N, weight: W = 1) -> None:  # type: ignore
        """
        Add an edge to the graph.

        Nodes are created automatically if they do not exist.
        For undirected graphs, the reverse edge is also added.

        Args:
            source: The source node
            target: The target node
            weight: The edge weight (default: 1)
        """
        self.add_node(source)
        self.add_node(target)

        if target not in self._adj[source]:
            self._num_edges += 1
        self._adj[source][target] = weight

        if not self._directed:
            self._adj[target][source] = weight

        logger.debug("Added edge: %s -> %s (weight=%s)", source, target, weight)

    def remove_edge(self, source: N, target: N) -> bool:
        """
        Remove an edge from the graph.

        Args:
            source: The source node
            target: The target node

        Returns:
            True if the edge was removed, False if it did not exist
        """
        if not self.has_edge(source, target):
            return False

        del self._adj[source][target]
        if not self._directed:
            del self._adj[target][source]
        self._num_edges -= 1
        return True

    def has_node(self, node: N) -> bool:
        """Check if a node exists in the graph."""
        return node in self._adj

    def has_edge(self, source: N, target: N) -> bool:
        """Check if an edge exists in the graph."""
        return source in self._adj and target in self._adj[source]

    def get_weight(self, source: N, target: N) -> W | None:
        """
        Get the weight of an edge.

        Returns:
            The edge weight, or None if the edge does not exist
        """
        if self.has_edge(source, target):
            return self._adj[source][target]
        return None

    def neighbours(self, node: N) -> Iterator[N]:
        """Yield the neighbours of a node."""
        if node in self._adj:
            yield from self._adj[node].keys()

    def edges_from(self, node: N) -> Iterator[Edge[N, W]]:
        """Yield edges originating from a node."""
        if node in self._adj:
            for target, weight in self._adj[node].items():
                yield Edge(node, target, weight)

    def nodes(self) -> Iterator[N]:
        """Yield all nodes in the graph."""
        yield from self._adj.keys()

    def edges(self) -> Iterator[Edge[N, W]]:
        """
        Yield all edges in the graph.

        For undirected graphs, each edge is yielded only once.
        """
        seen: set[tuple[N, N]] = set()
        for source in self._adj:
            for target, weight in self._adj[source].items():
                if self._directed:
                    edge_key = (source, target)
                else:
                    edge_key = tuple(sorted([str(source), str(target)]))  # type: ignore
                if edge_key not in seen:
                    seen.add(edge_key)  # type: ignore
                    yield Edge(source, target, weight)

    def num_nodes(self) -> int:
        """The number of nodes in the graph."""
        return len(self._adj)

    def num_edges(self) -> int:
        """The number of edges in the graph."""
        return self._num_edges

    @property
    def is_directed(self) -> bool:
        """True if the graph is directed."""
        return self._directed

    def degree(self, node: N) -> int:
        """Return the degree of a node (number of incident edges)."""
        return len(self._adj.get(node, {}))

    def in_degree(self, node: N) -> int:
        """
        Return the in-degree of a node (for directed graphs).

        For undirected graphs, this equals the degree.
        """
        if not self._directed:
            return self.degree(node)
        count = 0
        for source in self._adj:
            if node in self._adj[source]:
                count += 1
        return count

    def out_degree(self, node: N) -> int:
        """
        Return the out-degree of a node (for directed graphs).

        For undirected graphs, this equals the degree.
        """
        return self.degree(node)

    def __repr__(self) -> str:
        kind = "Directed" if self._directed else "Undirected"
        return f"{kind}Graph(nodes={self.num_nodes()}, edges={self.num_edges()})"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAVERSAL ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def bfs(graph: AdjacencyListGraph[N, W], start: N) -> Iterator[N]:
    """
    Breadth-First Search — level-by-level traversal.

    BFS explores the graph level by level: first all direct neighbours,
    then neighbours of neighbours, and so on. This guarantees finding
    the shortest path (fewest edges) in unweighted graphs.

    Complexity: O(V + E)
    Space: O(V) for the queue and visited set

    Properties:
    ───────────
    - Finds shortest paths in UNWEIGHTED graphs
    - Visit order: by distance from start
    - First discovered node at distance d is optimal

    Args:
        graph: The graph to traverse
        start: The starting node

    Yields:
        Nodes in BFS order

    Example:
        >>> g = AdjacencyListGraph[int, int]()
        >>> for u, v in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        ...     g.add_edge(u, v)
        >>> list(bfs(g, 1))
        [1, 2, 3, 4]
    """
    if not graph.has_node(start):
        return

    visited: set[N] = {start}
    queue: deque[N] = deque([start])

    while queue:
        current = queue.popleft()
        yield current

        for neighbour in graph.neighbours(current):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)


def dfs(graph: AdjacencyListGraph[N, W], start: N) -> Iterator[N]:
    """
    Depth-First Search — explore as deep as possible before backtracking.

    DFS explores one branch completely before exploring siblings.
    This is useful for detecting cycles, topological sorting and
    finding connected components.

    Complexity: O(V + E)
    Space: O(V) for the stack and visited set

    Properties:
    ───────────
    - Does NOT find shortest paths
    - Useful for cycle detection
    - Forms the basis of topological sort

    Args:
        graph: The graph to traverse
        start: The starting node

    Yields:
        Nodes in DFS order (pre-order)

    Example:
        >>> g = AdjacencyListGraph[int, int]()
        >>> for u, v in [(1, 2), (1, 3), (2, 4)]:
        ...     g.add_edge(u, v)
        >>> list(dfs(g, 1))  # Order depends on neighbour ordering
        [1, 2, 4, 3]
    """
    if not graph.has_node(start):
        return

    visited: set[N] = set()
    stack: list[N] = [start]

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        yield current

        # Add neighbours in reverse order for consistent ordering
        neighbours = list(graph.neighbours(current))
        for neighbour in reversed(neighbours):
            if neighbour not in visited:
                stack.append(neighbour)


def dfs_recursive(
    graph: AdjacencyListGraph[N, W],
    start: N,
    visited: set[N] | None = None,
) -> Iterator[N]:
    """
    Recursive Depth-First Search implementation.

    This variant is often clearer for understanding but may cause
    stack overflow for very deep graphs. Use the iterative version
    for production code.

    Args:
        graph: The graph to traverse
        start: The starting node
        visited: Set of already visited nodes (for internal use)

    Yields:
        Nodes in DFS order (pre-order)
    """
    if visited is None:
        visited = set()

    if start in visited or not graph.has_node(start):
        return

    visited.add(start)
    yield start

    for neighbour in graph.neighbours(start):
        yield from dfs_recursive(graph, neighbour, visited)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SHORTEST PATH ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ShortestPathResult(Generic[N]):
    """
    Result of a shortest path computation.

    Attributes:
        distances: Mapping from node to shortest distance from source
        predecessors: Mapping from node to predecessor on shortest path
        source: The source node
    """

    distances: dict[N, float]
    predecessors: dict[N, N | None]
    source: N

    def distance_to(self, target: N) -> float:
        """Return the shortest distance to the target node."""
        return self.distances.get(target, math.inf)

    def path_to(self, target: N) -> list[N] | None:
        """
        Return the shortest path to the target node.

        Returns:
            List of nodes from source to target, or None if unreachable
        """
        if target not in self.distances or self.distances[target] == math.inf:
            return None

        path: list[N] = []
        current: N | None = target

        while current is not None:
            path.append(current)
            current = self.predecessors.get(current)

        path.reverse()
        return path


def dijkstra(
    graph: AdjacencyListGraph[N, W],
    source: N,
) -> ShortestPathResult[N]:
    """
    Dijkstra's algorithm for shortest paths from a single source.

    Finds the shortest path from source to all reachable nodes in a
    graph with NON-NEGATIVE edge weights. Uses a priority queue to
    always process the nearest unvisited node.

    Complexity: O((V + E) log V) with binary heap

    IMPORTANT: Does not work with negative edge weights. For graphs
    with negative weights, use Bellman-Ford instead.

    Args:
        graph: The graph (must have non-negative weights)
        source: The source node

    Returns:
        ShortestPathResult containing distances and paths

    Example:
        >>> g = AdjacencyListGraph[str, int]()
        >>> g.add_edge("A", "B", 4)
        >>> g.add_edge("A", "C", 2)
        >>> g.add_edge("C", "B", 1)
        >>> result = dijkstra(g, "A")
        >>> result.distance_to("B")
        3
        >>> result.path_to("B")
        ['A', 'C', 'B']
    """
    distances: dict[N, float] = {node: math.inf for node in graph.nodes()}
    predecessors: dict[N, N | None] = {node: None for node in graph.nodes()}
    distances[source] = 0

    # Priority queue: (distance, node)
    pq: list[tuple[float, N]] = [(0, source)]

    while pq:
        dist, current = heapq.heappop(pq)

        # Skip if we have already found a better path
        if dist > distances[current]:
            continue

        for edge in graph.edges_from(current):
            new_dist = distances[current] + edge.weight

            if new_dist < distances[edge.target]:
                distances[edge.target] = new_dist
                predecessors[edge.target] = current
                heapq.heappush(pq, (new_dist, edge.target))

    return ShortestPathResult(distances, predecessors, source)


def astar(
    graph: AdjacencyListGraph[N, W],
    source: N,
    goal: N,
    heuristic: Callable[[N, N], float],
) -> ShortestPathResult[N]:
    """
    A* algorithm for shortest path with heuristic guidance.

    A* extends Dijkstra with a heuristic function h(n) that estimates
    the distance from node n to the goal. The priority becomes
    f(n) = g(n) + h(n), where g(n) is the known distance from source.

    Requirements for optimality:
    ─────────────────────────────
    - Admissible: h(n) ≤ actual distance to goal (never overestimates)
    - Consistent: h(n) ≤ cost(n, m) + h(m) for all edges (n, m)

    Common heuristics:
    ─────────────────
    - Euclidean distance (for geometric graphs)
    - Manhattan distance (for grid-based graphs)
    - Zero (reduces to Dijkstra)

    Args:
        graph: The graph
        source: The source node
        goal: The goal node
        heuristic: Function h(n) estimating distance from n to goal

    Returns:
        ShortestPathResult (may terminate early when goal is reached)

    Example:
        >>> g = AdjacencyListGraph[tuple[int, int], float]()
        >>> # Grid graph with Euclidean heuristic
        >>> def euclidean(p):
        ...     return math.sqrt((p[0] - goal[0])**2 + (p[1] - goal[1])**2)
        >>> goal = (5, 5)
        >>> result = astar(g, (0, 0), goal, euclidean)
    """
    distances: dict[N, float] = {node: math.inf for node in graph.nodes()}
    predecessors: dict[N, N | None] = {node: None for node in graph.nodes()}
    distances[source] = 0

    # Priority queue: (f_score, g_score, node)
    # f_score = g_score + heuristic
    pq: list[tuple[float, float, N]] = [(heuristic(source, goal), 0, source)]
    visited: set[N] = set()

    while pq:
        _, g_score, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        # Early termination when goal is reached
        if current == goal:
            break

        for edge in graph.edges_from(current):
            if edge.target in visited:
                continue

            new_g = g_score + edge.weight

            if new_g < distances[edge.target]:
                distances[edge.target] = new_g
                predecessors[edge.target] = current
                f_score = new_g + heuristic(edge.target, goal)
                heapq.heappush(pq, (f_score, new_g, edge.target))

    return ShortestPathResult(distances, predecessors, source)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: GRAPH ANALYSIS ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════


def find_connected_components(
    graph: AdjacencyListGraph[N, W],
) -> list[set[N]]:
    """
    Find all connected components in an undirected graph.

    A connected component is a maximal set of vertices such that
    there is a path between every pair of vertices.

    Complexity: O(V + E)

    Args:
        graph: An undirected graph

    Returns:
        List of sets, each set containing nodes of one component

    Raises:
        ValueError: If the graph is directed

    Example:
        >>> g = AdjacencyListGraph[int, int]()
        >>> g.add_edge(1, 2)
        >>> g.add_edge(3, 4)
        >>> find_connected_components(g)
        [{1, 2}, {3, 4}]
    """
    if graph.is_directed:
        raise ValueError("Connected components only defined for undirected graphs")

    visited: set[N] = set()
    components: list[set[N]] = []

    for node in graph.nodes():
        if node not in visited:
            component: set[N] = set()
            queue: deque[N] = deque([node])

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbour in graph.neighbours(current):
                    if neighbour not in visited:
                        queue.append(neighbour)

            components.append(component)

    return components


def topological_sort(graph: AdjacencyListGraph[N, W]) -> list[N] | None:
    """
    Topological sort for directed acyclic graphs (DAG).

    Returns an ordering of nodes such that for every edge (u, v),
    u appears before v in the ordering. Returns None if the graph
    contains cycles.

    Complexity: O(V + E)

    Applications:
    ─────────────
    - Build systems (Makefile dependencies)
    - Task scheduling
    - Course prerequisites

    Args:
        graph: A directed graph

    Returns:
        List of nodes in topological order, or None if cycles exist

    Raises:
        ValueError: If the graph is undirected

    Example:
        >>> g = AdjacencyListGraph[str, int](directed=True)
        >>> g.add_edge("compile", "link")
        >>> g.add_edge("test", "link")
        >>> topological_sort(g)
        ['compile', 'test', 'link']  # or ['test', 'compile', 'link']
    """
    if not graph.is_directed:
        raise ValueError("Topological sort only defined for directed graphs")

    # Calculate in-degree for each node
    in_degree: dict[N, int] = {node: 0 for node in graph.nodes()}
    for edge in graph.edges():
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

    # Queue with nodes of in-degree 0
    queue: deque[N] = deque(node for node, deg in in_degree.items() if deg == 0)
    result: list[N] = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbour in graph.neighbours(node):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    # If not all nodes processed, there is a cycle
    if len(result) != graph.num_nodes():
        return None

    return result


def has_cycle(graph: AdjacencyListGraph[N, W]) -> bool:
    """
    Detect whether a graph contains a cycle.

    For directed graphs, this uses a DFS recursion-stack test (white/grey/black).
    For undirected graphs, it uses DFS with an explicit parent to avoid treating
    the immediate back-edge as a cycle.
    """
    visited: set[N] = set()

    if graph.is_directed:
        in_stack: set[N] = set()

        def dfs_dir(u: N) -> bool:
            visited.add(u)
            in_stack.add(u)
            for edge in graph.edges_from(u):
                v = edge.target
                if v not in visited:
                    if dfs_dir(v):
                        return True
                elif v in in_stack:
                    return True
            in_stack.remove(u)
            return False

        for node in graph.nodes():
            if node not in visited and dfs_dir(node):
                return True
        return False

    def dfs_undir(u: N, parent: N | None) -> bool:
        visited.add(u)
        for edge in graph.edges_from(u):
            v = edge.target
            if v == parent:
                continue
            if v in visited:
                return True
            if dfs_undir(v, u):
                return True
        return False

    for node in graph.nodes():
        if node not in visited and dfs_undir(node, None):
            return True
    return False

def is_bipartite(graph: AdjacencyListGraph[N, W]) -> bool:
    """
    Check if an undirected graph is bipartite.

    A graph is bipartite if its vertices can be divided into two
    disjoint sets such that every edge connects vertices in
    different sets. Equivalently, a graph is bipartite if and
    only if it contains no odd-length cycles.

    Complexity: O(V + E)

    Args:
        graph: An undirected graph

    Returns:
        True if the graph is bipartite

    Example:
        >>> g = AdjacencyListGraph[int, int]()
        >>> g.add_edge(1, 2)
        >>> g.add_edge(2, 3)
        >>> g.add_edge(3, 1)  # Odd cycle
        >>> is_bipartite(g)
        False
    """
    colour: dict[N, int] = {}

    for start in graph.nodes():
        if start in colour:
            continue

        queue: deque[N] = deque([start])
        colour[start] = 0

        while queue:
            current = queue.popleft()

            for neighbour in graph.neighbours(current):
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[current]
                    queue.append(neighbour)
                elif colour[neighbour] == colour[current]:
                    return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_basic_operations() -> None:
    """Demonstrate basic graph operations."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic Graph Operations")
    logger.info("=" * 60)

    # Create a simple undirected graph
    g: AdjacencyListGraph[str, int] = AdjacencyListGraph(directed=False)

    # Add edges (nodes are created automatically)
    edges = [
        ("A", "B", 4),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 5),
        ("C", "D", 8),
        ("C", "E", 10),
        ("D", "E", 2),
        ("D", "F", 6),
        ("E", "F", 3),
    ]

    for src, dst, weight in edges:
        g.add_edge(src, dst, weight)

    logger.info("Graph created: %s", g)
    logger.info("Nodes: %s", list(g.nodes()))
    logger.info("Edges: %s", [(e.source, e.target, e.weight) for e in g.edges()])

    # Queries
    logger.info("Edge A-B exists? %s", g.has_edge("A", "B"))
    logger.info("Edge A-F exists? %s", g.has_edge("A", "F"))
    logger.info("Neighbours of C: %s", list(g.neighbours("C")))
    logger.info("Degree of C: %d", g.degree("C"))


def demo_traversals() -> None:
    """Demonstrate BFS and DFS traversals."""
    logger.info("=" * 60)
    logger.info("DEMO: Graph Traversals")
    logger.info("=" * 60)

    g: AdjacencyListGraph[int, int] = AdjacencyListGraph()

    # Create a tree-like structure
    #       1
    #      /|\
    #     2 3 4
    #    /|   |
    #   5 6   7
    #         |
    #         8

    edges = [(1, 2), (1, 3), (1, 4), (2, 5), (2, 6), (4, 7), (7, 8)]
    for src, dst in edges:
        g.add_edge(src, dst)

    logger.info("BFS from node 1: %s", list(bfs(g, 1)))
    logger.info("DFS from node 1: %s", list(dfs(g, 1)))

    # Connected components
    components = find_connected_components(g)
    logger.info("Connected components: %s", components)


def demo_shortest_path() -> None:
    """Demonstrate Dijkstra's algorithm."""
    logger.info("=" * 60)
    logger.info("DEMO: Shortest Path Algorithms")
    logger.info("=" * 60)

    # Weighted graph representing distances between cities
    g: AdjacencyListGraph[str, int] = AdjacencyListGraph()

    distances = [
        ("London", "Birmingham", 160),
        ("London", "Cambridge", 80),
        ("Cambridge", "Norwich", 100),
        ("Birmingham", "Manchester", 140),
        ("Norwich", "Manchester", 200),
        ("Manchester", "Edinburgh", 350),
        ("Norwich", "Edinburgh", 400),
    ]

    for src, dst, dist in distances:
        g.add_edge(src, dst, dist)

    logger.info("Dijkstra from London:")
    result = dijkstra(g, "London")

    for city in ["Birmingham", "Cambridge", "Norwich", "Manchester", "Edinburgh"]:
        dist = result.distance_to(city)
        path = result.path_to(city)
        logger.info(
            "  %s: %d km via %s", city, dist, " → ".join(path) if path else "N/A"
        )


def demo_topological_sort() -> None:
    """Demonstrate topological sorting."""
    logger.info("=" * 60)
    logger.info("DEMO: Topological Sort")
    logger.info("=" * 60)

    # Build system dependencies
    g: AdjacencyListGraph[str, int] = AdjacencyListGraph(directed=True)

    # dependency → dependent
    dependencies = [
        ("parse", "typecheck"),
        ("parse", "optimise"),
        ("typecheck", "codegen"),
        ("optimise", "codegen"),
        ("codegen", "link"),
        ("stdlib", "link"),
    ]

    for prereq, step in dependencies:
        g.add_edge(prereq, step)

    order = topological_sort(g)

    if order:
        logger.info("Valid build order:")
        for i, step in enumerate(order, 1):
            logger.info("  %d. %s", i, step)
    else:
        logger.info("  Circular dependencies detected!")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_basic_operations()
    demo_traversals()
    demo_shortest_path()
    demo_topological_sort()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="04UNIT Lab: Graph Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_4_01_graph_library.py --demo
  python lab_4_01_graph_library.py -v --demo
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration examples",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_all_demos()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()