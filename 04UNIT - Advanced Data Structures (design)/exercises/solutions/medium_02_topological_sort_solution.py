#!/usr/bin/env python3
"""
Solution: Topological Sort Algorithms
=====================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements two classic topological sorting algorithms:
Kahn's algorithm (BFS-based) and DFS-based topological sort. Includes
validation, all possible orderings, and practical applications.

Complexity Analysis:
    - Kahn's Algorithm: O(V + E) time, O(V) space
    - DFS-based Sort: O(V + E) time, O(V) space
    - All Topological Orders: O(V! * V) worst case

Topological sort produces a linear ordering of vertices such that for
every directed edge (u, v), vertex u comes before vertex v.

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class DirectedGraph(Generic[T]):
    """
    Directed graph for topological sorting.
    
    Attributes:
        _adjacency: Forward adjacency list.
        _in_degree: In-degree count for each vertex.
    """
    _adjacency: dict[T, list[T]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _in_degree: dict[T, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    
    def add_edge(self, source: T, target: T) -> None:
        """
        Add a directed edge from source to target.
        
        Args:
            source: The source vertex.
            target: The target vertex.
        """
        # Ensure both vertices exist
        if source not in self._adjacency:
            self._adjacency[source] = []
        if target not in self._adjacency:
            self._adjacency[target] = []
        
        # Initialise in-degrees
        if source not in self._in_degree:
            self._in_degree[source] = 0
        if target not in self._in_degree:
            self._in_degree[target] = 0
        
        self._adjacency[source].append(target)
        self._in_degree[target] += 1
    
    def add_vertex(self, vertex: T) -> None:
        """Add an isolated vertex."""
        if vertex not in self._adjacency:
            self._adjacency[vertex] = []
        if vertex not in self._in_degree:
            self._in_degree[vertex] = 0
    
    def neighbours(self, vertex: T) -> list[T]:
        """Return outgoing neighbours of a vertex."""
        return self._adjacency.get(vertex, [])
    
    def in_degree(self, vertex: T) -> int:
        """Return the in-degree of a vertex."""
        return self._in_degree.get(vertex, 0)
    
    @property
    def vertices(self) -> set[T]:
        """Return all vertices."""
        return set(self._adjacency.keys())
    
    @property
    def vertex_count(self) -> int:
        """Return number of vertices."""
        return len(self._adjacency)
    
    @property
    def edge_count(self) -> int:
        """Return number of edges."""
        return sum(len(adj) for adj in self._adjacency.values())


@dataclass
class TopologicalSortResult(Generic[T]):
    """
    Result of topological sort.
    
    Attributes:
        success: Whether a valid ordering was found.
        ordering: The topological ordering (if successful).
        cycle_vertices: Vertices involved in a cycle (if unsuccessful).
    """
    success: bool = False
    ordering: list[T] = field(default_factory=list)
    cycle_vertices: list[T] = field(default_factory=list)
    
    def __str__(self) -> str:
        """Format result for display."""
        if self.success:
            return f"Topological order: {' -> '.join(str(v) for v in self.ordering)}"
        return f"No valid ordering (cycle detected involving: {self.cycle_vertices})"


def topological_sort_kahn(graph: DirectedGraph[T]) -> TopologicalSortResult[T]:
    """
    Perform topological sort using Kahn's algorithm.
    
    Kahn's algorithm works by repeatedly removing vertices with
    in-degree zero. It's based on the observation that a DAG must
    have at least one vertex with no incoming edges.
    
    Algorithm:
        1. Compute in-degree of all vertices
        2. Add all vertices with in-degree 0 to a queue
        3. While queue is not empty:
           a. Remove vertex v from queue
           b. Add v to result
           c. For each neighbour u of v:
              - Decrement in-degree of u
              - If in-degree of u becomes 0, add to queue
        4. If result contains all vertices, return it
           Otherwise, graph has a cycle
    
    Args:
        graph: A directed graph.
    
    Returns:
        TopologicalSortResult with ordering or cycle information.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    
    Examples:
        >>> g = DirectedGraph[str]()
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('A', 'C')
        >>> g.add_edge('B', 'D')
        >>> result = topological_sort_kahn(g)
        >>> result.success
        True
    """
    result = TopologicalSortResult[T]()
    
    # Copy in-degrees (we'll modify them)
    in_degree: dict[T, int] = dict(graph._in_degree)
    
    # Queue of vertices with in-degree 0
    queue: deque[T] = deque()
    
    for vertex in graph.vertices:
        if in_degree[vertex] == 0:
            queue.append(vertex)
            logger.debug(f"Initial zero in-degree vertex: {vertex}")
    
    while queue:
        # Remove vertex with no dependencies
        vertex = queue.popleft()
        result.ordering.append(vertex)
        logger.debug(f"Processing: {vertex}")
        
        # Remove edges from this vertex
        for neighbour in graph.neighbours(vertex):
            in_degree[neighbour] -= 1
            
            if in_degree[neighbour] == 0:
                queue.append(neighbour)
                logger.debug(f"  {neighbour} now has in-degree 0")
    
    # Check if all vertices were processed
    if len(result.ordering) == graph.vertex_count:
        result.success = True
        logger.info(f"Topological sort successful: {result.ordering}")
    else:
        # Remaining vertices are in cycles
        result.cycle_vertices = [
            v for v in graph.vertices if v not in result.ordering
        ]
        logger.warning(f"Cycle detected, vertices in cycle: {result.cycle_vertices}")
    
    return result


def topological_sort_dfs(graph: DirectedGraph[T]) -> TopologicalSortResult[T]:
    """
    Perform topological sort using DFS.
    
    DFS-based topological sort uses the property that in a DFS tree
    of a DAG, a vertex is finished (all descendants processed) before
    any of its ancestors. By prepending vertices to the result as they
    finish, we get a valid topological ordering.
    
    Uses three-colour marking to detect cycles:
    - White: unvisited
    - Grey: visiting (in current path)
    - Black: finished
    
    Args:
        graph: A directed graph.
    
    Returns:
        TopologicalSortResult with ordering or cycle information.
    
    Time Complexity: O(V + E)
    Space Complexity: O(V) for recursion stack
    """
    result = TopologicalSortResult[T]()
    
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[T, int] = {v: WHITE for v in graph.vertices}
    
    # Use list instead of deque for prepending (will reverse at end)
    ordering: list[T] = []
    has_cycle = False
    
    def dfs(vertex: T) -> bool:
        """
        DFS visit with cycle detection.
        
        Returns True if cycle detected.
        """
        nonlocal has_cycle
        
        colour[vertex] = GREY
        logger.debug(f"Visiting: {vertex}")
        
        for neighbour in graph.neighbours(vertex):
            if colour[neighbour] == GREY:
                # Back edge = cycle
                logger.debug(f"Back edge detected: {vertex} -> {neighbour}")
                has_cycle = True
                return True
            
            if colour[neighbour] == WHITE:
                if dfs(neighbour):
                    return True
        
        colour[vertex] = BLACK
        ordering.append(vertex)  # Finished processing
        logger.debug(f"Finished: {vertex}")
        return False
    
    # Visit all vertices
    for vertex in graph.vertices:
        if colour[vertex] == WHITE:
            if dfs(vertex):
                # Cycle detected
                result.cycle_vertices = [v for v, c in colour.items() if c == GREY]
                return result
    
    # Reverse to get correct order (we appended in reverse finish order)
    result.ordering = list(reversed(ordering))
    result.success = True
    
    logger.info(f"DFS topological sort successful: {result.ordering}")
    return result


def all_topological_orderings(
    graph: DirectedGraph[T],
    max_results: int = 100
) -> Iterator[list[T]]:
    """
    Generate all possible topological orderings.
    
    Different orderings arise when multiple vertices have in-degree 0
    simultaneously. This function uses backtracking to enumerate all
    valid orderings.
    
    Warning: The number of orderings can be factorial in the worst case.
    
    Args:
        graph: A directed graph.
        max_results: Maximum number of orderings to generate.
    
    Yields:
        Each valid topological ordering.
    
    Examples:
        >>> g = DirectedGraph[str]()
        >>> g.add_edge('A', 'C')
        >>> g.add_edge('B', 'C')
        >>> list(all_topological_orderings(g))
        [['A', 'B', 'C'], ['B', 'A', 'C']]
    """
    in_degree: dict[T, int] = dict(graph._in_degree)
    visited: set[T] = set()
    current_path: list[T] = []
    count = 0
    
    def backtrack() -> Iterator[list[T]]:
        """Generate orderings via backtracking."""
        nonlocal count
        
        if count >= max_results:
            return
        
        # Find all vertices with current in-degree 0
        available = [
            v for v in graph.vertices
            if v not in visited and in_degree[v] == 0
        ]
        
        if not available:
            if len(visited) == graph.vertex_count:
                # Found a valid ordering
                count += 1
                yield list(current_path)
            return
        
        # Try each available vertex
        for vertex in available:
            visited.add(vertex)
            current_path.append(vertex)
            
            # Decrease in-degree of neighbours
            for neighbour in graph.neighbours(vertex):
                in_degree[neighbour] -= 1
            
            yield from backtrack()
            
            # Backtrack
            visited.remove(vertex)
            current_path.pop()
            
            for neighbour in graph.neighbours(vertex):
                in_degree[neighbour] += 1
    
    yield from backtrack()


def is_valid_topological_order(graph: DirectedGraph[T], ordering: list[T]) -> bool:
    """
    Verify if an ordering is a valid topological sort.
    
    Args:
        graph: The directed graph.
        ordering: A proposed ordering.
    
    Returns:
        True if the ordering is valid.
    """
    if set(ordering) != graph.vertices:
        return False
    
    position = {v: i for i, v in enumerate(ordering)}
    
    for vertex in graph.vertices:
        for neighbour in graph.neighbours(vertex):
            if position[vertex] >= position[neighbour]:
                logger.debug(
                    f"Invalid: {vertex} (pos {position[vertex]}) "
                    f"should come before {neighbour} (pos {position[neighbour]})"
                )
                return False
    
    return True


def lexicographically_smallest_topological_sort(
    graph: DirectedGraph[T]
) -> TopologicalSortResult[T]:
    """
    Find the lexicographically smallest topological ordering.
    
    Uses a min-heap instead of a regular queue in Kahn's algorithm
    to always select the smallest available vertex.
    
    Args:
        graph: A directed graph.
    
    Returns:
        TopologicalSortResult with the lexicographically smallest ordering.
    """
    import heapq
    
    result = TopologicalSortResult[T]()
    in_degree: dict[T, int] = dict(graph._in_degree)
    
    # Use min-heap for lexicographic ordering
    heap: list[T] = []
    
    for vertex in graph.vertices:
        if in_degree[vertex] == 0:
            heapq.heappush(heap, vertex)
    
    while heap:
        vertex = heapq.heappop(heap)
        result.ordering.append(vertex)
        
        for neighbour in graph.neighbours(vertex):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                heapq.heappush(heap, neighbour)
    
    result.success = len(result.ordering) == graph.vertex_count
    return result


def parallel_scheduling(
    graph: DirectedGraph[T]
) -> dict[int, list[T]]:
    """
    Determine parallel scheduling levels for tasks.
    
    Groups vertices into levels where all vertices at the same level
    can be executed in parallel (all their dependencies are satisfied).
    
    Args:
        graph: A DAG representing task dependencies.
    
    Returns:
        Dictionary mapping level number to list of vertices at that level.
    """
    in_degree: dict[T, int] = dict(graph._in_degree)
    levels: dict[int, list[T]] = defaultdict(list)
    
    current_level = 0
    current_batch = [v for v in graph.vertices if in_degree[v] == 0]
    
    while current_batch:
        levels[current_level] = current_batch
        next_batch: list[T] = []
        
        for vertex in current_batch:
            for neighbour in graph.neighbours(vertex):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    next_batch.append(neighbour)
        
        current_batch = next_batch
        current_level += 1
    
    return dict(levels)


def demonstrate_topological_sort() -> None:
    """Demonstrate topological sorting algorithms."""
    print("=" * 60)
    print("Topological Sort Demonstration")
    print("=" * 60)
    
    # Example 1: Simple DAG
    print("\n1. Simple DAG - Kahn's Algorithm")
    print("-" * 40)
    
    simple_dag: DirectedGraph[str] = DirectedGraph()
    edges_simple = [
        ('A', 'C'), ('A', 'D'),
        ('B', 'D'), ('B', 'E'),
        ('C', 'F'),
        ('D', 'F'),
        ('E', 'F'),
    ]
    for u, v in edges_simple:
        simple_dag.add_edge(u, v)
    
    result = topological_sort_kahn(simple_dag)
    print(f"   {result}")
    
    # Example 2: DFS-based sort
    print("\n2. Same DAG - DFS-Based Algorithm")
    print("-" * 40)
    
    result_dfs = topological_sort_dfs(simple_dag)
    print(f"   {result_dfs}")
    
    # Verify both are valid
    print(f"   Kahn result valid: {is_valid_topological_order(simple_dag, result.ordering)}")
    print(f"   DFS result valid: {is_valid_topological_order(simple_dag, result_dfs.ordering)}")
    
    # Example 3: Graph with cycle
    print("\n3. Graph with Cycle")
    print("-" * 40)
    
    cyclic: DirectedGraph[int] = DirectedGraph()
    edges_cyclic = [
        (1, 2), (2, 3), (3, 4), (4, 2),  # Cycle: 2 -> 3 -> 4 -> 2
        (1, 5), (5, 6),
    ]
    for u, v in edges_cyclic:
        cyclic.add_edge(u, v)
    
    result = topological_sort_kahn(cyclic)
    print(f"   {result}")
    
    # Example 4: All topological orderings
    print("\n4. All Topological Orderings")
    print("-" * 40)
    
    small_dag: DirectedGraph[str] = DirectedGraph()
    edges_small = [
        ('A', 'C'),
        ('B', 'C'),
        ('C', 'D'),
    ]
    for u, v in edges_small:
        small_dag.add_edge(u, v)
    
    print("   Graph: A -> C, B -> C, C -> D")
    print("   All valid orderings:")
    for i, ordering in enumerate(all_topological_orderings(small_dag), 1):
        print(f"      {i}. {' -> '.join(ordering)}")
    
    # Example 5: Lexicographically smallest
    print("\n5. Lexicographically Smallest Ordering")
    print("-" * 40)
    
    lex_dag: DirectedGraph[str] = DirectedGraph()
    edges_lex = [
        ('C', 'E'),
        ('A', 'D'),
        ('B', 'D'),
        ('D', 'E'),
    ]
    for u, v in edges_lex:
        lex_dag.add_edge(u, v)
    
    result = lexicographically_smallest_topological_sort(lex_dag)
    print(f"   Lexicographically smallest: {' -> '.join(result.ordering)}")
    
    # Example 6: Parallel scheduling
    print("\n6. Parallel Task Scheduling")
    print("-" * 40)
    
    tasks: DirectedGraph[str] = DirectedGraph()
    task_deps = [
        ('fetch_data', 'process_A'),
        ('fetch_data', 'process_B'),
        ('process_A', 'merge'),
        ('process_B', 'merge'),
        ('merge', 'validate'),
        ('validate', 'deploy'),
        ('config', 'deploy'),
    ]
    for dep, task in task_deps:
        tasks.add_edge(dep, task)
    
    levels = parallel_scheduling(tasks)
    print("   Parallel execution levels:")
    for level, task_list in sorted(levels.items()):
        print(f"      Level {level}: {task_list}")
    
    # Example 7: Course prerequisites
    print("\n7. Real-World Example: Course Prerequisites")
    print("-" * 40)
    
    courses: DirectedGraph[str] = DirectedGraph()
    prerequisites = [
        ('Maths 101', 'Maths 201'),
        ('Maths 201', 'Maths 301'),
        ('CS 101', 'CS 201'),
        ('CS 101', 'Data Structures'),
        ('Maths 101', 'Data Structures'),
        ('Data Structures', 'Algorithms'),
        ('CS 201', 'Algorithms'),
        ('Algorithms', 'AI'),
        ('Maths 301', 'AI'),
    ]
    for prereq, course in prerequisites:
        courses.add_edge(prereq, course)
    
    result = topological_sort_kahn(courses)
    print("   Valid course order:")
    for i, course in enumerate(result.ordering, 1):
        print(f"      Semester {i}: {course}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_topological_sort()
