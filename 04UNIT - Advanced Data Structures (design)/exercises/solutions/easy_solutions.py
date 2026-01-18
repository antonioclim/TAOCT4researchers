#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4: Easy Exercise Solutions
═══════════════════════════════════════════════════════════════════════════════

Solutions for easy-level practice exercises.

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterator, TypeVar

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# EASY 01: GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_graph_from_edges(
    edges: list[tuple[str, str]],
    directed: bool = False
) -> dict[str, list[str]]:
    """
    Build adjacency list from edge list.
    
    SOLUTION APPROACH:
    - Use defaultdict for automatic list initialisation
    - For undirected graphs, add edge in both directions
    
    TIME: O(E)
    SPACE: O(V + E)
    """
    graph: dict[str, list[str]] = defaultdict(list)
    
    for source, target in edges:
        graph[source].append(target)
        if not directed:
            graph[target].append(source)
    
    return dict(graph)


def build_weighted_graph(
    edges: list[tuple[str, str, float]]
) -> dict[str, list[tuple[str, float]]]:
    """
    Build weighted adjacency list from weighted edge list.
    
    SOLUTION APPROACH:
    Store (neighbour, weight) tuples in adjacency list.
    """
    graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
    
    for source, target, weight in edges:
        graph[source].append((target, weight))
        graph[target].append((source, weight))  # Undirected
    
    return dict(graph)


# Test
def test_easy_01() -> None:
    """Test graph construction solutions."""
    edges = [("A", "B"), ("B", "C"), ("A", "C")]
    
    # Undirected
    graph = build_graph_from_edges(edges, directed=False)
    assert "B" in graph["A"]
    assert "A" in graph["B"]
    
    # Directed
    graph = build_graph_from_edges(edges, directed=True)
    assert "B" in graph["A"]
    assert "A" not in graph.get("B", [])
    
    print("✓ Easy 01 tests passed")


# ═══════════════════════════════════════════════════════════════════════════════
# EASY 02: HASH SET
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleHashSet:
    """
    Simple hash set implementation with chaining.
    
    SOLUTION APPROACH:
    - Fixed-size bucket array
    - Chaining for collision resolution
    - Basic hash function using built-in hash()
    """
    
    def __init__(self, capacity: int = 16) -> None:
        """Initialise with given bucket capacity."""
        self._capacity = capacity
        self._buckets: list[list[T]] = [[] for _ in range(capacity)]
        self._size = 0
    
    def _get_bucket_index(self, item: T) -> int:
        """Get bucket index for item."""
        return hash(item) % self._capacity
    
    def add(self, item: T) -> bool:
        """
        Add item to set.
        
        Returns True if item was added (not already present).
        """
        bucket_idx = self._get_bucket_index(item)
        bucket = self._buckets[bucket_idx]
        
        if item in bucket:
            return False
        
        bucket.append(item)
        self._size += 1
        return True
    
    def contains(self, item: T) -> bool:
        """Check if item is in set."""
        bucket_idx = self._get_bucket_index(item)
        return item in self._buckets[bucket_idx]
    
    def remove(self, item: T) -> bool:
        """
        Remove item from set.
        
        Returns True if item was removed (was present).
        """
        bucket_idx = self._get_bucket_index(item)
        bucket = self._buckets[bucket_idx]
        
        if item in bucket:
            bucket.remove(item)
            self._size -= 1
            return True
        
        return False
    
    def __len__(self) -> int:
        """Return number of elements."""
        return self._size
    
    def __contains__(self, item: T) -> bool:
        """Support 'in' operator."""
        return self.contains(item)


# Test
def test_easy_02() -> None:
    """Test hash set solutions."""
    hs = SimpleHashSet()
    
    assert hs.add("apple")
    assert hs.add("banana")
    assert not hs.add("apple")  # Duplicate
    
    assert hs.contains("apple")
    assert not hs.contains("cherry")
    
    assert len(hs) == 2
    
    assert hs.remove("apple")
    assert not hs.contains("apple")
    
    print("✓ Easy 02 tests passed")


# ═══════════════════════════════════════════════════════════════════════════════
# EASY 03: BFS
# ═══════════════════════════════════════════════════════════════════════════════


def bfs_traversal(
    graph: dict[str, list[str]],
    start: str
) -> list[str]:
    """
    Breadth-first search traversal.
    
    SOLUTION APPROACH:
    1. Use deque for O(1) popleft
    2. Mark visited BEFORE enqueueing
    3. Return list of nodes in visit order
    """
    if start not in graph:
        return [start] if start else []
    
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    result: list[str] = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    
    return result


def bfs_shortest_path(
    graph: dict[str, list[str]],
    start: str,
    end: str
) -> list[str] | None:
    """
    Find shortest path using BFS (unweighted graph).
    
    SOLUTION APPROACH:
    Track predecessors during BFS, then reconstruct path.
    """
    if start == end:
        return [start]
    
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    predecessor: dict[str, str] = {}
    
    while queue:
        node = queue.popleft()
        
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                predecessor[neighbour] = node
                queue.append(neighbour)
                
                if neighbour == end:
                    # Reconstruct path
                    path: list[str] = []
                    current: str | None = end
                    while current is not None:
                        path.append(current)
                        current = predecessor.get(current)
                    path.reverse()
                    return path
    
    return None  # No path found


def bfs_level_order(
    graph: dict[str, list[str]],
    start: str
) -> list[list[str]]:
    """
    Return nodes grouped by BFS level.
    
    SOLUTION APPROACH:
    Process queue in batches, one level at a time.
    """
    if start not in graph and start:
        return [[start]]
    
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    levels: list[list[str]] = []
    
    while queue:
        level_size = len(queue)
        current_level: list[str] = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        
        levels.append(current_level)
    
    return levels


# Test
def test_easy_03() -> None:
    """Test BFS solutions."""
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D"],
        "D": ["B", "C"],
    }
    
    # Basic traversal
    result = bfs_traversal(graph, "A")
    assert result[0] == "A"
    assert set(result) == {"A", "B", "C", "D"}
    
    # Shortest path
    path = bfs_shortest_path(graph, "A", "D")
    assert path is not None
    assert path[0] == "A"
    assert path[-1] == "D"
    assert len(path) == 3  # A -> B/C -> D
    
    # Level order
    levels = bfs_level_order(graph, "A")
    assert levels[0] == ["A"]
    assert set(levels[1]) == {"B", "C"}
    
    print("✓ Easy 03 tests passed")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    test_easy_01()
    test_easy_02()
    test_easy_03()
    print("\n✓ All easy exercise solutions verified")
