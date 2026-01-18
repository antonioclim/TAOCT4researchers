#!/usr/bin/env python3
"""
Solution: Dijkstra's Shortest Path Algorithm
============================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements Dijkstra's algorithm for finding shortest paths
in weighted graphs with non-negative edge weights. Includes both basic
and optimised implementations with detailed path reconstruction.

Complexity Analysis:
    - Basic (linear search): O(V² + E)
    - With binary heap: O((V + E) log V)
    - With Fibonacci heap: O(V log V + E) - theoretical

The algorithm uses a greedy approach: at each step, it selects the
unvisited vertex with the smallest tentative distance and relaxes
all edges from that vertex.

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')
INF = float('inf')


@dataclass
class WeightedGraph(Generic[T]):
    """
    Weighted graph representation using adjacency lists.
    
    Supports both directed and undirected graphs with positive weights.
    
    Attributes:
        directed: Whether the graph is directed.
        _adjacency: Maps vertices to list of (neighbour, weight) pairs.
    """
    directed: bool = False
    _adjacency: dict[T, list[tuple[T, float]]] = field(default_factory=dict)
    
    def add_edge(self, source: T, target: T, weight: float) -> None:
        """
        Add a weighted edge to the graph.
        
        Args:
            source: The source vertex.
            target: The target vertex.
            weight: The edge weight (must be non-negative).
        
        Raises:
            ValueError: If weight is negative.
        """
        if weight < 0:
            raise ValueError(f"Dijkstra requires non-negative weights, got {weight}")
        
        if source not in self._adjacency:
            self._adjacency[source] = []
        if target not in self._adjacency:
            self._adjacency[target] = []
        
        self._adjacency[source].append((target, weight))
        
        if not self.directed:
            self._adjacency[target].append((source, weight))
    
    def add_vertex(self, vertex: T) -> None:
        """Add an isolated vertex."""
        if vertex not in self._adjacency:
            self._adjacency[vertex] = []
    
    def neighbours(self, vertex: T) -> Iterator[tuple[T, float]]:
        """Yield (neighbour, weight) pairs for a vertex."""
        yield from self._adjacency.get(vertex, [])
    
    @property
    def vertices(self) -> set[T]:
        """Return all vertices."""
        return set(self._adjacency.keys())
    
    @property
    def vertex_count(self) -> int:
        """Return number of vertices."""
        return len(self._adjacency)


@dataclass
class DijkstraResult(Generic[T]):
    """
    Result of Dijkstra's algorithm.
    
    Attributes:
        source: The source vertex.
        distances: Shortest distance from source to each vertex.
        predecessors: Previous vertex on shortest path.
        visited_order: Order in which vertices were finalised.
    """
    source: T
    distances: dict[T, float] = field(default_factory=dict)
    predecessors: dict[T, T | None] = field(default_factory=dict)
    visited_order: list[T] = field(default_factory=list)
    
    def get_distance(self, target: T) -> float:
        """
        Get the shortest distance to a target.
        
        Returns INF if target is unreachable.
        """
        return self.distances.get(target, INF)
    
    def get_path(self, target: T) -> list[T] | None:
        """
        Reconstruct the shortest path to a target.
        
        Args:
            target: The destination vertex.
        
        Returns:
            The path as a list of vertices, or None if unreachable.
        """
        if target not in self.distances or self.distances[target] == INF:
            return None
        
        path = []
        current: T | None = target
        
        while current is not None:
            path.append(current)
            current = self.predecessors.get(current)
        
        return list(reversed(path))
    
    def get_path_with_weights(
        self,
        target: T,
        graph: WeightedGraph[T]
    ) -> list[tuple[T, float]] | None:
        """
        Reconstruct path with edge weights.
        
        Returns list of (vertex, cumulative_distance) pairs.
        """
        path = self.get_path(target)
        if path is None:
            return None
        
        result = [(path[0], 0.0)]
        cumulative = 0.0
        
        for i in range(1, len(path)):
            # Find edge weight
            for neighbour, weight in graph.neighbours(path[i - 1]):
                if neighbour == path[i]:
                    cumulative += weight
                    result.append((path[i], cumulative))
                    break
        
        return result


def dijkstra_basic(
    graph: WeightedGraph[T],
    source: T
) -> DijkstraResult[T]:
    """
    Basic Dijkstra's algorithm with linear vertex selection.
    
    This implementation uses a simple loop to find the unvisited
    vertex with minimum distance. While less efficient than the
    heap-based version, it's easier to understand.
    
    Args:
        graph: The weighted graph.
        source: The starting vertex.
    
    Returns:
        DijkstraResult with distances and paths.
    
    Time Complexity: O(V² + E)
    Space Complexity: O(V)
    """
    result = DijkstraResult(source=source)
    
    # Initialise distances
    for vertex in graph.vertices:
        result.distances[vertex] = INF
        result.predecessors[vertex] = None
    
    result.distances[source] = 0
    unvisited = set(graph.vertices)
    
    logger.debug(f"Starting basic Dijkstra from {source}")
    
    while unvisited:
        # Find unvisited vertex with minimum distance (linear search)
        current = min(unvisited, key=lambda v: result.distances[v])
        
        # If minimum is infinity, remaining vertices are unreachable
        if result.distances[current] == INF:
            logger.debug(f"Remaining vertices unreachable")
            break
        
        unvisited.remove(current)
        result.visited_order.append(current)
        
        logger.debug(
            f"Processing {current} with distance {result.distances[current]}"
        )
        
        # Relax edges from current vertex
        for neighbour, weight in graph.neighbours(current):
            if neighbour in unvisited:
                new_distance = result.distances[current] + weight
                
                if new_distance < result.distances[neighbour]:
                    result.distances[neighbour] = new_distance
                    result.predecessors[neighbour] = current
                    logger.debug(
                        f"  Updated {neighbour}: distance = {new_distance}"
                    )
    
    logger.info(f"Basic Dijkstra complete, visited {len(result.visited_order)} vertices")
    return result


def dijkstra(
    graph: WeightedGraph[T],
    source: T
) -> DijkstraResult[T]:
    """
    Dijkstra's algorithm with binary heap optimisation.
    
    Uses a priority queue (min-heap) to efficiently select the
    next vertex to process. This is the standard efficient implementation.
    
    The key insight is that we only need to process each vertex once,
    when we first pop it from the heap with its final shortest distance.
    
    Args:
        graph: The weighted graph.
        source: The starting vertex.
    
    Returns:
        DijkstraResult with distances and paths.
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    
    Examples:
        >>> g = WeightedGraph[str]()
        >>> g.add_edge('A', 'B', 4)
        >>> g.add_edge('A', 'C', 2)
        >>> g.add_edge('C', 'B', 1)
        >>> result = dijkstra(g, 'A')
        >>> result.get_distance('B')
        3
    """
    result = DijkstraResult(source=source)
    
    # Initialise distances
    for vertex in graph.vertices:
        result.distances[vertex] = INF
        result.predecessors[vertex] = None
    
    result.distances[source] = 0
    
    # Priority queue: (distance, vertex)
    # Using counter to break ties and ensure unique entries
    heap: list[tuple[float, int, T]] = []
    counter = 0
    heapq.heappush(heap, (0, counter, source))
    
    visited: set[T] = set()
    
    logger.debug(f"Starting heap-based Dijkstra from {source}")
    
    while heap:
        current_dist, _, current = heapq.heappop(heap)
        
        # Skip if already processed
        if current in visited:
            continue
        
        visited.add(current)
        result.visited_order.append(current)
        
        logger.debug(f"Processing {current} with distance {current_dist}")
        
        # Relax edges from current vertex
        for neighbour, weight in graph.neighbours(current):
            if neighbour not in visited:
                new_distance = current_dist + weight
                
                if new_distance < result.distances[neighbour]:
                    result.distances[neighbour] = new_distance
                    result.predecessors[neighbour] = current
                    
                    counter += 1
                    heapq.heappush(heap, (new_distance, counter, neighbour))
                    
                    logger.debug(
                        f"  Updated {neighbour}: distance = {new_distance}"
                    )
    
    logger.info(f"Heap-based Dijkstra complete, visited {len(result.visited_order)} vertices")
    return result


def dijkstra_single_target(
    graph: WeightedGraph[T],
    source: T,
    target: T
) -> tuple[float, list[T] | None]:
    """
    Dijkstra's algorithm optimised for single-target queries.
    
    Stops as soon as the target is reached, which can be faster
    when we only need one shortest path.
    
    Args:
        graph: The weighted graph.
        source: The starting vertex.
        target: The destination vertex.
    
    Returns:
        Tuple of (distance, path) where path is None if unreachable.
    """
    distances: dict[T, float] = {v: INF for v in graph.vertices}
    predecessors: dict[T, T | None] = {}
    distances[source] = 0
    
    heap: list[tuple[float, int, T]] = [(0, 0, source)]
    counter = 0
    visited: set[T] = set()
    
    while heap:
        current_dist, _, current = heapq.heappop(heap)
        
        if current in visited:
            continue
        
        # Early termination when target is reached
        if current == target:
            # Reconstruct path
            path = []
            node: T | None = target
            while node is not None:
                path.append(node)
                node = predecessors.get(node)
            return current_dist, list(reversed(path))
        
        visited.add(current)
        
        for neighbour, weight in graph.neighbours(current):
            if neighbour not in visited:
                new_dist = current_dist + weight
                if new_dist < distances[neighbour]:
                    distances[neighbour] = new_dist
                    predecessors[neighbour] = current
                    counter += 1
                    heapq.heappush(heap, (new_dist, counter, neighbour))
    
    return INF, None


def dijkstra_all_pairs(graph: WeightedGraph[T]) -> dict[T, DijkstraResult[T]]:
    """
    Compute shortest paths between all pairs of vertices.
    
    Runs Dijkstra from each vertex.
    
    Args:
        graph: The weighted graph.
    
    Returns:
        Dictionary mapping each vertex to its DijkstraResult.
    
    Time Complexity: O(V * (V + E) log V)
    """
    return {vertex: dijkstra(graph, vertex) for vertex in graph.vertices}


def reconstruct_all_shortest_paths(
    graph: WeightedGraph[T],
    source: T,
    target: T
) -> list[list[T]]:
    """
    Find all shortest paths between source and target.
    
    When multiple shortest paths exist (same total weight),
    this function returns all of them.
    
    Args:
        graph: The weighted graph.
        source: Starting vertex.
        target: Destination vertex.
    
    Returns:
        List of all shortest paths.
    """
    # First run Dijkstra to get distances
    result = dijkstra(graph, source)
    target_dist = result.get_distance(target)
    
    if target_dist == INF:
        return []
    
    # BFS backwards from target using distance constraints
    paths: list[list[T]] = []
    stack: list[tuple[T, list[T]]] = [(target, [target])]
    
    while stack:
        current, path = stack.pop()
        
        if current == source:
            paths.append(list(reversed(path)))
            continue
        
        # Find all valid predecessors
        current_dist = result.distances[current]
        
        for vertex in graph.vertices:
            for neighbour, weight in graph.neighbours(vertex):
                if neighbour == current:
                    # Check if this edge is on a shortest path
                    if abs(result.distances[vertex] + weight - current_dist) < 1e-9:
                        if vertex not in path:  # Avoid cycles
                            stack.append((vertex, path + [vertex]))
    
    return paths


def demonstrate_dijkstra() -> None:
    """Demonstrate Dijkstra's algorithm."""
    print("=" * 60)
    print("Dijkstra's Shortest Path Algorithm")
    print("=" * 60)
    
    # Example 1: Simple graph
    print("\n1. Simple Weighted Graph")
    print("-" * 40)
    
    simple: WeightedGraph[str] = WeightedGraph()
    edges_simple = [
        ('A', 'B', 4),
        ('A', 'C', 2),
        ('B', 'C', 1),
        ('B', 'D', 5),
        ('C', 'D', 8),
        ('C', 'E', 10),
        ('D', 'E', 2),
    ]
    for u, v, w in edges_simple:
        simple.add_edge(u, v, w)
    
    result = dijkstra(simple, 'A')
    
    print("   Shortest distances from A:")
    for vertex in sorted(result.distances.keys()):
        dist = result.distances[vertex]
        path = result.get_path(vertex)
        print(f"      {vertex}: distance = {dist}, path = {' -> '.join(path or [])}")
    
    # Example 2: Compare basic vs heap-based
    print("\n2. Basic vs Heap-Based Implementation")
    print("-" * 40)
    
    result_basic = dijkstra_basic(simple, 'A')
    result_heap = dijkstra(simple, 'A')
    
    print("   Both implementations produce same results:")
    for v in sorted(simple.vertices):
        basic_dist = result_basic.distances[v]
        heap_dist = result_heap.distances[v]
        match = "✓" if basic_dist == heap_dist else "✗"
        print(f"      {v}: basic={basic_dist}, heap={heap_dist} {match}")
    
    # Example 3: Single target optimisation
    print("\n3. Single-Target Shortest Path")
    print("-" * 40)
    
    distance, path = dijkstra_single_target(simple, 'A', 'E')
    print(f"   Shortest path A -> E:")
    print(f"      Distance: {distance}")
    print(f"      Path: {' -> '.join(path or [])}")
    
    # Example 4: Road network
    print("\n4. Road Network Example")
    print("-" * 40)
    
    roads: WeightedGraph[str] = WeightedGraph()
    road_edges = [
        ('London', 'Birmingham', 126),
        ('London', 'Bristol', 118),
        ('Birmingham', 'Manchester', 88),
        ('Birmingham', 'Leeds', 118),
        ('Bristol', 'Cardiff', 45),
        ('Manchester', 'Leeds', 44),
        ('Manchester', 'Liverpool', 35),
        ('Leeds', 'Newcastle', 95),
        ('Liverpool', 'Edinburgh', 221),
        ('Newcastle', 'Edinburgh', 105),
    ]
    for city1, city2, distance in road_edges:
        roads.add_edge(city1, city2, distance)
    
    result = dijkstra(roads, 'London')
    
    print("   Shortest routes from London:")
    for city in sorted(result.distances.keys()):
        if city != 'London':
            path = result.get_path(city)
            path_str = ' -> '.join(path or [])
            print(f"      {city}: {result.distances[city]} miles via {path_str}")
    
    # Example 5: Path with weights
    print("\n5. Detailed Path Information")
    print("-" * 40)
    
    path_details = result.get_path_with_weights('Edinburgh', roads)
    if path_details:
        print("   London to Edinburgh journey:")
        for city, cumulative in path_details:
            print(f"      {city}: {cumulative} miles total")
    
    # Example 6: Disconnected graph
    print("\n6. Handling Disconnected Graphs")
    print("-" * 40)
    
    disconnected: WeightedGraph[int] = WeightedGraph()
    disconnected.add_edge(1, 2, 5)
    disconnected.add_edge(2, 3, 3)
    disconnected.add_vertex(4)  # Isolated vertex
    disconnected.add_edge(5, 6, 2)
    
    result = dijkstra(disconnected, 1)
    print("   From vertex 1:")
    for v in sorted(disconnected.vertices):
        dist = result.distances[v]
        status = "unreachable" if dist == INF else f"distance = {dist}"
        print(f"      Vertex {v}: {status}")
    
    # Example 7: Multiple shortest paths
    print("\n7. Multiple Shortest Paths")
    print("-" * 40)
    
    multi_path: WeightedGraph[str] = WeightedGraph()
    multi_path.add_edge('S', 'A', 1)
    multi_path.add_edge('S', 'B', 1)
    multi_path.add_edge('A', 'C', 1)
    multi_path.add_edge('B', 'C', 1)
    multi_path.add_edge('C', 'T', 1)
    
    paths = reconstruct_all_shortest_paths(multi_path, 'S', 'T')
    print("   All shortest paths S -> T:")
    for i, path in enumerate(paths, 1):
        print(f"      {i}. {' -> '.join(path)}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_dijkstra()
