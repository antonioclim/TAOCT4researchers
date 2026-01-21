#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4, Lab 1: Graph Library - SOLUTION KEY
═══════════════════════════════════════════════════════════════════════════════

INSTRUCTOR NOTES
────────────────
This solution file provides reference implementations and additional context
for grading student submissions. Key assessment criteria are highlighted.

ASSESSMENT FOCUS
────────────────
1. Correctness of algorithm implementations (40%)
2. Code quality: type hints, docstrings, logging (20%)
3. Edge case handling (20%)
4. Performance considerations (20%)

COMMON STUDENT MISTAKES
───────────────────────
1. Forgetting to track visited nodes in BFS/DFS (infinite loops)
2. Using list.pop(0) instead of deque.popleft() for BFS
3. Not handling disconnected graphs
4. Incorrect handling of directed vs undirected edges
5. Missing base cases in recursive implementations

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, Iterator, TypeVar

N = TypeVar("N", bound=Hashable)
W = TypeVar("W", int, float)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 1: ADJACENCY LIST GRAPH
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Edge(Generic[N, W]):
    """
    Represents a weighted edge in a graph.
    
    GRADING NOTES:
    - Students should use frozen=True for hashability
    - Generic types demonstrate advanced Python skills
    """
    source: N
    target: N
    weight: W = field(default=1)  # type: ignore
    
    def reversed(self) -> Edge[N, W]:
        """Return edge with swapped source and target."""
        return Edge(self.target, self.source, self.weight)


class AdjacencyListGraph(Generic[N, W]):
    """
    Graph implementation using adjacency lists.
    
    GRADING CRITERIA:
    - O(1) node/edge lookup
    - Proper handling of directed/undirected
    - Clean separation of concerns
    
    INSTRUCTOR NOTES:
    - This is the preferred implementation for sparse graphs
    - Students often confuse edge count in undirected graphs
    """
    
    def __init__(self, directed: bool = False) -> None:
        self._adj: dict[N, list[tuple[N, W]]] = defaultdict(list)
        self._directed = directed
        self._node_set: set[N] = set()
    
    def add_node(self, node: N) -> None:
        """Add isolated node. Idempotent operation."""
        self._node_set.add(node)
        if node not in self._adj:
            self._adj[node] = []
    
    def add_edge(self, source: N, target: N, weight: W = 1) -> None:  # type: ignore
        """
        Add edge between nodes.
        
        COMMON MISTAKE: Students forget to add reverse edge for undirected.
        """
        self._node_set.add(source)
        self._node_set.add(target)
        
        self._adj[source].append((target, weight))
        
        if not self._directed:
            self._adj[target].append((source, weight))
    
    def has_edge(self, source: N, target: N) -> bool:
        """Check if edge exists."""
        return any(t == target for t, _ in self._adj.get(source, []))
    
    def neighbours(self, node: N) -> Iterator[N]:
        """Yield all neighbours of node."""
        for target, _ in self._adj.get(node, []):
            yield target
    
    def nodes(self) -> Iterator[N]:
        """Yield all nodes."""
        yield from self._node_set
    
    def num_nodes(self) -> int:
        """Return number of nodes."""
        return len(self._node_set)
    
    def num_edges(self) -> int:
        """
        Return number of edges.
        
        INSTRUCTOR NOTE: For undirected graphs, each edge is stored twice
        but counted once. Students often get this wrong.
        """
        total = sum(len(adj) for adj in self._adj.values())
        return total if self._directed else total // 2
    
    @property
    def is_directed(self) -> bool:
        """Return whether graph is directed."""
        return self._directed


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 2: BREADTH-FIRST SEARCH
# ═══════════════════════════════════════════════════════════════════════════════


def bfs(graph: AdjacencyListGraph[N, W], start: N) -> Iterator[N]:
    """
    Breadth-first search traversal.
    
    ALGORITHM:
    1. Initialise queue with start node
    2. Mark start as visited
    3. While queue not empty:
       a. Dequeue front node
       b. Yield node
       c. Enqueue unvisited neighbours
    
    COMPLEXITY: O(V + E) time, O(V) space
    
    GRADING NOTES:
    - MUST use collections.deque, not list (O(1) vs O(n) popleft)
    - MUST track visited before enqueueing (not after dequeueing)
    - The latter is a subtle but important distinction
    """
    visited: set[N] = {start}
    queue: deque[N] = deque([start])
    
    while queue:
        node = queue.popleft()  # O(1) - critical!
        yield node
        
        for neighbour in graph.neighbours(node):
            if neighbour not in visited:
                visited.add(neighbour)  # Mark BEFORE enqueueing
                queue.append(neighbour)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 3: DEPTH-FIRST SEARCH
# ═══════════════════════════════════════════════════════════════════════════════


def dfs(graph: AdjacencyListGraph[N, W], start: N) -> Iterator[N]:
    """
    Depth-first search traversal (iterative).
    
    INSTRUCTOR NOTES:
    - Iterative version avoids stack overflow on deep graphs
    - Students may submit recursive version which is acceptable
    - Key difference from BFS: stack vs queue, marking on pop
    
    COMPLEXITY: O(V + E) time, O(V) space
    """
    visited: set[N] = set()
    stack: list[N] = [start]
    
    while stack:
        node = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        yield node
        
        # Reverse to maintain left-to-right order
        for neighbour in reversed(list(graph.neighbours(node))):
            if neighbour not in visited:
                stack.append(neighbour)


def dfs_recursive(
    graph: AdjacencyListGraph[N, W],
    start: N,
    visited: set[N] | None = None
) -> Iterator[N]:
    """
    Recursive DFS (alternative implementation).
    
    GRADING: Accept either iterative or recursive.
    Recursive is cleaner but has stack depth limits.
    """
    if visited is None:
        visited = set()
    
    visited.add(start)
    yield start
    
    for neighbour in graph.neighbours(start):
        if neighbour not in visited:
            yield from dfs_recursive(graph, neighbour, visited)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 4: DIJKSTRA'S ALGORITHM
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ShortestPathResult(Generic[N]):
    """
    Container for shortest path results.
    
    GRADING: Students should return both distances and paths.
    """
    distances: dict[N, float]
    predecessors: dict[N, N | None]
    
    def distance_to(self, target: N) -> float:
        """Get distance to target node."""
        return self.distances.get(target, float('inf'))
    
    def path_to(self, target: N) -> list[N] | None:
        """
        Reconstruct path to target.
        
        COMMON MISTAKE: Students forget to reverse the path.
        """
        if target not in self.distances or self.distances[target] == float('inf'):
            return None
        
        path: list[N] = []
        current: N | None = target
        
        while current is not None:
            path.append(current)
            current = self.predecessors.get(current)
        
        path.reverse()  # Don't forget!
        return path


def dijkstra(
    graph: AdjacencyListGraph[N, W],
    start: N
) -> ShortestPathResult[N]:
    """
    Dijkstra's shortest path algorithm.
    
    ALGORITHM:
    1. Initialise distances (start=0, others=∞)
    2. Add start to priority queue
    3. While PQ not empty:
       a. Extract minimum distance node
       b. For each neighbour, relax edge if shorter
       c. Update PQ with new distances
    
    COMPLEXITY: O((V + E) log V) with binary heap
    
    CRITICAL GRADING POINTS:
    1. MUST use priority queue (heapq), not linear search
    2. MUST handle visited/stale entries correctly
    3. MUST store predecessors for path reconstruction
    4. MUST NOT work with negative weights (should document this)
    """
    distances: dict[N, float] = {start: 0}
    predecessors: dict[N, N | None] = {start: None}
    
    # Priority queue: (distance, node)
    pq: list[tuple[float, N]] = [(0, start)]
    
    while pq:
        dist, node = heapq.heappop(pq)
        
        # Skip stale entries
        # INSTRUCTOR NOTE: This is the "lazy deletion" approach
        if dist > distances.get(node, float('inf')):
            continue
        
        # Relax edges
        for neighbour in graph.neighbours(node):
            # Get edge weight (simplified - real implementation needs weight lookup)
            weight = 1.0  # Placeholder
            for n, w in graph._adj[node]:
                if n == neighbour:
                    weight = float(w)
                    break
            
            new_dist = dist + weight
            
            if new_dist < distances.get(neighbour, float('inf')):
                distances[neighbour] = new_dist
                predecessors[neighbour] = node
                heapq.heappush(pq, (new_dist, neighbour))
    
    return ShortestPathResult(distances, predecessors)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 5: TOPOLOGICAL SORT
# ═══════════════════════════════════════════════════════════════════════════════


def topological_sort(graph: AdjacencyListGraph[N, W]) -> list[N] | None:
    """
    Topological sort using Kahn's algorithm.
    
    ALGORITHM (Kahn's):
    1. Compute in-degrees for all nodes
    2. Add all nodes with in-degree 0 to queue
    3. While queue not empty:
       a. Remove node, add to result
       b. Decrease in-degree of neighbours
       c. Add neighbours with in-degree 0 to queue
    4. If result contains all nodes, return it; else cycle exists
    
    ALTERNATIVE: DFS-based (students may use either)
    
    GRADING:
    - Must return None for cyclic graphs
    - Order must be valid (all dependencies before dependents)
    """
    # Compute in-degrees
    in_degree: dict[N, int] = defaultdict(int)
    for node in graph.nodes():
        in_degree[node]  # Ensure all nodes present
        for neighbour in graph.neighbours(node):
            in_degree[neighbour] += 1
    
    # Nodes with no dependencies
    queue: deque[N] = deque([n for n, d in in_degree.items() if d == 0])
    result: list[N] = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbour in graph.neighbours(node):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)
    
    # Check for cycle
    if len(result) != graph.num_nodes():
        return None  # Cycle detected
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 6: CYCLE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def has_cycle(graph: AdjacencyListGraph[N, W]) -> bool:
    """
    Detect cycle in graph using DFS colouring.
    
    ALGORITHM:
    - WHITE (0): Not visited
    - GREY (1): Currently in recursion stack
    - BLACK (2): Completely processed
    
    Cycle exists iff we encounter a GREY node.
    
    GRADING NOTES:
    - Must handle both directed and undirected graphs
    - For undirected, must not count parent edge as back edge
    """
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[N, int] = {n: WHITE for n in graph.nodes()}
    
    def dfs_visit(node: N, parent: N | None = None) -> bool:
        colour[node] = GREY
        
        for neighbour in graph.neighbours(node):
            # For undirected graphs, skip parent
            if not graph.is_directed and neighbour == parent:
                continue
            
            if colour[neighbour] == GREY:
                return True  # Back edge found
            
            if colour[neighbour] == WHITE:
                if dfs_visit(neighbour, node):
                    return True
        
        colour[node] = BLACK
        return False
    
    for node in graph.nodes():
        if colour[node] == WHITE:
            if dfs_visit(node):
                return True
    
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 7: BIPARTITE CHECK
# ═══════════════════════════════════════════════════════════════════════════════


def is_bipartite(graph: AdjacencyListGraph[N, W]) -> bool:
    """
    Check if graph is bipartite using 2-colouring.
    
    ALGORITHM:
    - Attempt to colour graph with 2 colours
    - BFS/DFS, alternate colours for neighbours
    - If conflict found, not bipartite
    
    THEOREM: Graph is bipartite iff it contains no odd-length cycles.
    
    GRADING: Accept BFS or DFS based solutions.
    """
    colour: dict[N, int] = {}
    
    for start in graph.nodes():
        if start in colour:
            continue
        
        queue: deque[N] = deque([start])
        colour[start] = 0
        
        while queue:
            node = queue.popleft()
            
            for neighbour in graph.neighbours(node):
                if neighbour not in colour:
                    colour[neighbour] = 1 - colour[node]
                    queue.append(neighbour)
                elif colour[neighbour] == colour[node]:
                    return False  # Same colour conflict
    
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# GRADING RUBRIC SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
"""
GRAPH OPERATIONS (25 points)
- Correct adjacency list implementation: 10 pts
- Proper directed/undirected handling: 5 pts
- Type hints and docstrings: 5 pts
- Edge cases (empty, self-loops): 5 pts

TRAVERSALS (25 points)
- Correct BFS implementation: 10 pts
- Correct DFS implementation: 10 pts
- Uses appropriate data structures: 5 pts

SHORTEST PATH (25 points)
- Correct Dijkstra implementation: 15 pts
- Path reconstruction: 5 pts
- Handles unreachable nodes: 5 pts

GRAPH ANALYSIS (25 points)
- Topological sort (with cycle detection): 10 pts
- Cycle detection: 8 pts
- Bipartite check: 7 pts

BONUS (10 points)
- A* implementation: 5 pts
- Connected components: 3 pts
- Performance optimisations: 2 pts
"""
