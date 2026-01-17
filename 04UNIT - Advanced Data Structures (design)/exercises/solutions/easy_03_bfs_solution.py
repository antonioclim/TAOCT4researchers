#!/usr/bin/env python3
"""
Solution: Breadth-First Search Implementation
=============================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements Breadth-First Search (BFS) for graph traversal,
demonstrating level-order exploration, shortest path finding in unweighted
graphs, and connected component discovery.

Complexity Analysis:
    - Time: O(V + E) where V is vertices, E is edges
    - Space: O(V) for the queue and visited set

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Graph(Generic[T]):
    """
    Simple graph class for BFS demonstration.
    
    Attributes:
        directed: Whether the graph is directed.
        _adjacency: Adjacency list representation.
    """
    directed: bool = False
    _adjacency: dict[T, list[T]] = field(default_factory=dict)
    
    def add_edge(self, source: T, target: T) -> None:
        """Add an edge to the graph."""
        if source not in self._adjacency:
            self._adjacency[source] = []
        if target not in self._adjacency:
            self._adjacency[target] = []
        
        self._adjacency[source].append(target)
        
        if not self.directed:
            self._adjacency[target].append(source)
    
    def neighbours(self, vertex: T) -> list[T]:
        """Return the neighbours of a vertex."""
        return self._adjacency.get(vertex, [])
    
    @property
    def vertices(self) -> set[T]:
        """Return all vertices in the graph."""
        return set(self._adjacency.keys())


@dataclass
class BFSResult(Generic[T]):
    """
    Result container for BFS traversal.
    
    Attributes:
        visited_order: Order in which vertices were visited.
        distances: Shortest distance from source to each vertex.
        predecessors: Predecessor of each vertex in the BFS tree.
        levels: Vertices grouped by their distance from source.
    """
    visited_order: list[T] = field(default_factory=list)
    distances: dict[T, int] = field(default_factory=dict)
    predecessors: dict[T, T | None] = field(default_factory=dict)
    levels: dict[int, list[T]] = field(default_factory=dict)
    
    def get_path(self, target: T) -> list[T] | None:
        """
        Reconstruct the shortest path to a target vertex.
        
        Args:
            target: The target vertex.
        
        Returns:
            The path as a list of vertices, or None if unreachable.
        """
        if target not in self.predecessors:
            return None
        
        path = []
        current: T | None = target
        
        while current is not None:
            path.append(current)
            current = self.predecessors.get(current)
        
        return list(reversed(path))


def bfs(
    graph: Graph[T],
    source: T,
    visitor: Callable[[T, int], None] | None = None
) -> BFSResult[T]:
    """
    Perform Breadth-First Search from a source vertex.
    
    BFS explores the graph level by level, visiting all vertices at
    distance k from the source before visiting vertices at distance k+1.
    This property makes BFS ideal for finding shortest paths in
    unweighted graphs.
    
    Args:
        graph: The graph to traverse.
        source: The starting vertex.
        visitor: Optional callback function called for each visited vertex.
                 Receives (vertex, distance) as arguments.
    
    Returns:
        A BFSResult containing traversal information.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    
    Examples:
        >>> g = Graph[str]()
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('A', 'C')
        >>> g.add_edge('B', 'D')
        >>> result = bfs(g, 'A')
        >>> result.distances['D']
        2
    """
    result = BFSResult[T]()
    visited: set[T] = set()
    queue: deque[tuple[T, int]] = deque()
    
    # Initialise with source vertex
    queue.append((source, 0))
    visited.add(source)
    result.distances[source] = 0
    result.predecessors[source] = None
    
    logger.debug(f"Starting BFS from {source}")
    
    while queue:
        current, distance = queue.popleft()
        
        # Record visit
        result.visited_order.append(current)
        
        # Group by level
        if distance not in result.levels:
            result.levels[distance] = []
        result.levels[distance].append(current)
        
        # Call visitor if provided
        if visitor:
            visitor(current, distance)
        
        logger.debug(f"Visiting {current} at distance {distance}")
        
        # Explore neighbours
        for neighbour in graph.neighbours(current):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append((neighbour, distance + 1))
                result.distances[neighbour] = distance + 1
                result.predecessors[neighbour] = current
    
    logger.info(
        f"BFS complete: visited {len(result.visited_order)} vertices, "
        f"max distance {max(result.levels.keys()) if result.levels else 0}"
    )
    
    return result


def bfs_iterative_deepening(
    graph: Graph[T],
    source: T,
    max_depth: int
) -> Iterator[tuple[T, int]]:
    """
    BFS with depth limiting for large graphs.
    
    Yields vertices up to a maximum depth from the source.
    Useful when you only need nearby vertices.
    
    Args:
        graph: The graph to traverse.
        source: The starting vertex.
        max_depth: Maximum distance from source to explore.
    
    Yields:
        Tuples of (vertex, distance) for each visited vertex.
    """
    visited: set[T] = set()
    queue: deque[tuple[T, int]] = deque()
    
    queue.append((source, 0))
    visited.add(source)
    
    while queue:
        current, depth = queue.popleft()
        yield current, depth
        
        if depth < max_depth:
            for neighbour in graph.neighbours(current):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append((neighbour, depth + 1))


def shortest_path(
    graph: Graph[T],
    source: T,
    target: T
) -> tuple[list[T] | None, int]:
    """
    Find the shortest path between two vertices.
    
    In an unweighted graph, BFS guarantees finding the shortest path
    (minimum number of edges) between any two vertices.
    
    Args:
        graph: The graph to search.
        source: The starting vertex.
        target: The destination vertex.
    
    Returns:
        A tuple of (path, distance) where path is a list of vertices
        or None if no path exists, and distance is the path length.
    
    Time Complexity: O(V + E)
    
    Examples:
        >>> g = Graph[str]()
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('B', 'C')
        >>> g.add_edge('A', 'C')
        >>> path, dist = shortest_path(g, 'A', 'C')
        >>> dist
        1
    """
    result = bfs(graph, source)
    path = result.get_path(target)
    distance = result.distances.get(target, -1)
    
    return path, distance


def find_connected_components(graph: Graph[T]) -> list[set[T]]:
    """
    Find all connected components in an undirected graph.
    
    A connected component is a maximal set of vertices such that
    there is a path between every pair of vertices.
    
    Args:
        graph: An undirected graph.
    
    Returns:
        A list of sets, each containing the vertices of one component.
    
    Time Complexity: O(V + E)
    
    Examples:
        >>> g = Graph[int]()
        >>> g.add_edge(1, 2)
        >>> g.add_edge(3, 4)
        >>> components = find_connected_components(g)
        >>> len(components)
        2
    """
    visited: set[T] = set()
    components: list[set[T]] = []
    
    for vertex in graph.vertices:
        if vertex not in visited:
            # Start BFS from this vertex
            component: set[T] = set()
            result = bfs(graph, vertex)
            
            for v in result.visited_order:
                component.add(v)
                visited.add(v)
            
            components.append(component)
            logger.debug(f"Found component with {len(component)} vertices")
    
    logger.info(f"Found {len(components)} connected components")
    return components


def is_bipartite(graph: Graph[T]) -> tuple[bool, dict[T, int] | None]:
    """
    Check if a graph is bipartite using BFS colouring.
    
    A graph is bipartite if its vertices can be coloured with two colours
    such that no two adjacent vertices have the same colour. This is
    equivalent to the graph having no odd-length cycles.
    
    Args:
        graph: The graph to check.
    
    Returns:
        A tuple (is_bipartite, colouring) where colouring maps vertices
        to colours (0 or 1), or None if not bipartite.
    
    Time Complexity: O(V + E)
    """
    if not graph.vertices:
        return True, {}
    
    colours: dict[T, int] = {}
    
    for start in graph.vertices:
        if start in colours:
            continue
        
        # BFS with colouring
        queue: deque[T] = deque([start])
        colours[start] = 0
        
        while queue:
            current = queue.popleft()
            current_colour = colours[current]
            
            for neighbour in graph.neighbours(current):
                if neighbour not in colours:
                    colours[neighbour] = 1 - current_colour
                    queue.append(neighbour)
                elif colours[neighbour] == current_colour:
                    logger.debug(
                        f"Graph is not bipartite: {current} and {neighbour} "
                        f"are adjacent with same colour"
                    )
                    return False, None
    
    logger.info("Graph is bipartite")
    return True, colours


def bfs_all_paths(
    graph: Graph[T],
    source: T,
    target: T,
    max_paths: int = 10
) -> list[list[T]]:
    """
    Find multiple shortest paths between two vertices.
    
    When multiple shortest paths exist, this function finds all of them
    (up to a maximum limit).
    
    Args:
        graph: The graph to search.
        source: The starting vertex.
        target: The destination vertex.
        max_paths: Maximum number of paths to return.
    
    Returns:
        A list of paths (each path is a list of vertices).
    """
    if source == target:
        return [[source]]
    
    # First, find the shortest distance
    result = bfs(graph, source)
    
    if target not in result.distances:
        return []
    
    target_distance = result.distances[target]
    
    # Then find all paths of that length using modified BFS
    paths: list[list[T]] = []
    queue: deque[tuple[T, list[T]]] = deque()
    queue.append((source, [source]))
    
    while queue and len(paths) < max_paths:
        current, path = queue.popleft()
        current_distance = len(path) - 1
        
        if current_distance >= target_distance:
            continue
        
        for neighbour in graph.neighbours(current):
            if neighbour == target and current_distance + 1 == target_distance:
                paths.append(path + [neighbour])
            elif (
                neighbour not in path and 
                result.distances.get(neighbour, float('inf')) == current_distance + 1
            ):
                queue.append((neighbour, path + [neighbour]))
    
    return paths


def demonstrate_bfs() -> None:
    """Demonstrate BFS algorithms."""
    print("=" * 60)
    print("Breadth-First Search Demonstration")
    print("=" * 60)
    
    # Example 1: Basic BFS traversal
    print("\n1. Basic BFS Traversal")
    print("-" * 40)
    
    graph: Graph[str] = Graph()
    edges = [
        ('A', 'B'), ('A', 'C'), ('A', 'D'),
        ('B', 'E'), ('B', 'F'),
        ('C', 'G'),
        ('D', 'H'), ('D', 'I'),
        ('E', 'J'),
    ]
    
    for u, v in edges:
        graph.add_edge(u, v)
    
    result = bfs(graph, 'A')
    print(f"   Starting from 'A'")
    print(f"   Visit order: {result.visited_order}")
    print(f"   Distances: {dict(sorted(result.distances.items()))}")
    
    print("\n   Levels:")
    for level, vertices in sorted(result.levels.items()):
        print(f"      Level {level}: {vertices}")
    
    # Example 2: Shortest path
    print("\n2. Shortest Path Finding")
    print("-" * 40)
    
    path, distance = shortest_path(graph, 'A', 'J')
    print(f"   Path from A to J: {path}")
    print(f"   Distance: {distance} edges")
    
    path, distance = shortest_path(graph, 'C', 'I')
    print(f"   Path from C to I: {path}")
    print(f"   Distance: {distance} edges")
    
    # Example 3: Connected components
    print("\n3. Connected Components")
    print("-" * 40)
    
    disconnected: Graph[int] = Graph()
    disconnected.add_edge(1, 2)
    disconnected.add_edge(2, 3)
    disconnected.add_edge(4, 5)
    disconnected.add_edge(6, 7)
    disconnected.add_edge(7, 8)
    disconnected.add_edge(8, 6)
    
    components = find_connected_components(disconnected)
    print(f"   Number of components: {len(components)}")
    for i, component in enumerate(components, 1):
        print(f"   Component {i}: {sorted(component)}")
    
    # Example 4: Bipartite check
    print("\n4. Bipartite Graph Check")
    print("-" * 40)
    
    # Bipartite graph (tree is always bipartite)
    bipartite_graph: Graph[str] = Graph()
    for u, v in [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')]:
        bipartite_graph.add_edge(u, v)
    
    is_bip, colouring = is_bipartite(bipartite_graph)
    print(f"   Tree graph is bipartite: {is_bip}")
    if colouring:
        group_0 = [v for v, c in colouring.items() if c == 0]
        group_1 = [v for v, c in colouring.items() if c == 1]
        print(f"   Group 0: {group_0}")
        print(f"   Group 1: {group_1}")
    
    # Non-bipartite graph (triangle)
    triangle: Graph[str] = Graph()
    for u, v in [('X', 'Y'), ('Y', 'Z'), ('Z', 'X')]:
        triangle.add_edge(u, v)
    
    is_bip, _ = is_bipartite(triangle)
    print(f"   Triangle graph is bipartite: {is_bip}")
    
    # Example 5: Multiple shortest paths
    print("\n5. Multiple Shortest Paths")
    print("-" * 40)
    
    grid: Graph[str] = Graph()
    # Create a small grid with multiple equal-length paths
    grid_edges = [
        ('S', 'A'), ('S', 'B'),
        ('A', 'C'), ('A', 'D'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'T'), ('D', 'T'), ('E', 'T'),
    ]
    for u, v in grid_edges:
        grid.add_edge(u, v)
    
    paths = bfs_all_paths(grid, 'S', 'T')
    print(f"   All shortest paths from S to T:")
    for i, path in enumerate(paths, 1):
        print(f"      Path {i}: {' -> '.join(path)}")
    
    # Example 6: Depth-limited BFS
    print("\n6. Depth-Limited BFS")
    print("-" * 40)
    
    print(f"   Vertices within distance 2 from 'A':")
    for vertex, depth in bfs_iterative_deepening(graph, 'A', max_depth=2):
        print(f"      {vertex} at distance {depth}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_bfs()
