#!/usr/bin/env python3
"""
Solution: Graph Construction from Edge Lists
============================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution demonstrates the construction of graph data structures from
various edge list representations, implementing both adjacency list and
adjacency matrix representations with full type annotations.

Complexity Analysis:
    - Adjacency List Construction: O(E) time, O(V + E) space
    - Adjacency Matrix Construction: O(V² + E) time, O(V²) space
    - Edge List to Graph: O(E) time for parsing

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class Edge(Generic[T]):
    """
    Represents a weighted edge in a graph.
    
    Attributes:
        source: The source vertex of the edge.
        target: The target vertex of the edge.
        weight: The weight of the edge (default 1.0 for unweighted graphs).
    
    Examples:
        >>> edge = Edge('A', 'B', 2.5)
        >>> edge.source
        'A'
        >>> edge.weight
        2.5
    """
    source: T
    target: T
    weight: float = 1.0
    
    def reversed(self) -> Edge[T]:
        """Return a new edge with source and target swapped."""
        return Edge(self.target, self.source, self.weight)


@dataclass
class AdjacencyListGraph(Generic[T]):
    """
    Graph implementation using adjacency lists.
    
    The adjacency list representation stores, for each vertex, a list of
    its neighbouring vertices. This representation is memory-efficient for
    sparse graphs where E << V².
    
    Attributes:
        directed: Whether the graph is directed.
        weighted: Whether edges carry weights.
        _adjacency: Internal adjacency list structure.
        _vertices: Set of all vertices in the graph.
    
    Space Complexity: O(V + E)
    
    Examples:
        >>> g = AdjacencyListGraph[str](directed=False)
        >>> g.add_edge('A', 'B')
        >>> g.add_edge('B', 'C')
        >>> list(g.neighbours('B'))
        ['A', 'C']
    """
    directed: bool = False
    weighted: bool = False
    _adjacency: dict[T, list[tuple[T, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _vertices: set[T] = field(default_factory=set)
    
    def add_vertex(self, vertex: T) -> None:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add.
        
        Time Complexity: O(1) amortised.
        """
        self._vertices.add(vertex)
        if vertex not in self._adjacency:
            self._adjacency[vertex] = []
        logger.debug(f"Added vertex: {vertex}")
    
    def add_edge(self, source: T, target: T, weight: float = 1.0) -> None:
        """
        Add an edge to the graph.
        
        For undirected graphs, adds edges in both directions.
        
        Args:
            source: The source vertex.
            target: The target vertex.
            weight: The edge weight (default 1.0).
        
        Time Complexity: O(1) amortised.
        """
        self.add_vertex(source)
        self.add_vertex(target)
        
        self._adjacency[source].append((target, weight))
        
        if not self.directed:
            self._adjacency[target].append((source, weight))
        
        logger.debug(f"Added edge: {source} -> {target} (weight={weight})")
    
    def neighbours(self, vertex: T) -> Iterator[T]:
        """
        Yield all neighbours of a vertex.
        
        Args:
            vertex: The vertex whose neighbours to retrieve.
        
        Yields:
            Each neighbouring vertex.
        
        Time Complexity: O(degree(vertex)).
        """
        for neighbour, _ in self._adjacency.get(vertex, []):
            yield neighbour
    
    def neighbours_with_weights(self, vertex: T) -> Iterator[tuple[T, float]]:
        """
        Yield all neighbours of a vertex with their edge weights.
        
        Args:
            vertex: The vertex whose neighbours to retrieve.
        
        Yields:
            Tuples of (neighbour, weight).
        """
        yield from self._adjacency.get(vertex, [])
    
    @property
    def vertices(self) -> set[T]:
        """Return the set of all vertices."""
        return self._vertices.copy()
    
    @property
    def vertex_count(self) -> int:
        """Return the number of vertices."""
        return len(self._vertices)
    
    @property
    def edge_count(self) -> int:
        """
        Return the number of edges.
        
        For undirected graphs, each edge is counted once.
        """
        total = sum(len(adj) for adj in self._adjacency.values())
        return total if self.directed else total // 2
    
    def has_edge(self, source: T, target: T) -> bool:
        """
        Check whether an edge exists between two vertices.
        
        Args:
            source: The source vertex.
            target: The target vertex.
        
        Returns:
            True if the edge exists, False otherwise.
        
        Time Complexity: O(degree(source)).
        """
        return any(n == target for n, _ in self._adjacency.get(source, []))
    
    def degree(self, vertex: T) -> int:
        """
        Return the degree of a vertex.
        
        For directed graphs, returns the out-degree.
        
        Args:
            vertex: The vertex to query.
        
        Returns:
            The degree of the vertex.
        """
        return len(self._adjacency.get(vertex, []))
    
    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[T, T] | tuple[T, T, float]],
        directed: bool = False
    ) -> AdjacencyListGraph[T]:
        """
        Construct a graph from an edge list.
        
        Args:
            edges: List of edges as (source, target) or (source, target, weight).
            directed: Whether the graph is directed.
        
        Returns:
            A new AdjacencyListGraph instance.
        
        Time Complexity: O(E).
        
        Examples:
            >>> edges = [('A', 'B'), ('B', 'C'), ('C', 'A')]
            >>> g = AdjacencyListGraph.from_edge_list(edges)
            >>> g.vertex_count
            3
        """
        weighted = any(len(e) == 3 for e in edges)
        graph = cls(directed=directed, weighted=weighted)
        
        for edge in edges:
            if len(edge) == 2:
                source, target = edge
                weight = 1.0
            else:
                source, target, weight = edge
            graph.add_edge(source, target, weight)
        
        logger.info(
            f"Constructed graph with {graph.vertex_count} vertices "
            f"and {graph.edge_count} edges"
        )
        return graph
    
    def __repr__(self) -> str:
        """Return a string representation of the graph."""
        graph_type = "Directed" if self.directed else "Undirected"
        return (
            f"{graph_type}Graph(vertices={self.vertex_count}, "
            f"edges={self.edge_count})"
        )


@dataclass
class AdjacencyMatrixGraph(Generic[T]):
    """
    Graph implementation using an adjacency matrix.
    
    The adjacency matrix representation uses a V×V matrix where entry (i,j)
    indicates the presence (and weight) of an edge from vertex i to vertex j.
    This representation enables O(1) edge queries but requires O(V²) space.
    
    Attributes:
        directed: Whether the graph is directed.
        _vertex_to_index: Mapping from vertex to matrix index.
        _index_to_vertex: Mapping from matrix index to vertex.
        _matrix: The adjacency matrix (None indicates no edge).
    
    Space Complexity: O(V²)
    """
    directed: bool = False
    _vertex_to_index: dict[T, int] = field(default_factory=dict)
    _index_to_vertex: dict[int, T] = field(default_factory=dict)
    _matrix: list[list[float | None]] = field(default_factory=list)
    
    def add_vertex(self, vertex: T) -> int:
        """
        Add a vertex to the graph.
        
        Args:
            vertex: The vertex to add.
        
        Returns:
            The index assigned to the vertex.
        
        Time Complexity: O(V) for matrix expansion.
        """
        if vertex in self._vertex_to_index:
            return self._vertex_to_index[vertex]
        
        index = len(self._vertex_to_index)
        self._vertex_to_index[vertex] = index
        self._index_to_vertex[index] = vertex
        
        # Expand existing rows
        for row in self._matrix:
            row.append(None)
        
        # Add new row
        self._matrix.append([None] * (index + 1))
        
        logger.debug(f"Added vertex: {vertex} at index {index}")
        return index
    
    def add_edge(self, source: T, target: T, weight: float = 1.0) -> None:
        """
        Add an edge to the graph.
        
        Args:
            source: The source vertex.
            target: The target vertex.
            weight: The edge weight.
        
        Time Complexity: O(1) if vertices exist, O(V) otherwise.
        """
        i = self.add_vertex(source)
        j = self.add_vertex(target)
        
        self._matrix[i][j] = weight
        
        if not self.directed:
            self._matrix[j][i] = weight
        
        logger.debug(f"Added edge: {source} -> {target} (weight={weight})")
    
    def has_edge(self, source: T, target: T) -> bool:
        """
        Check whether an edge exists.
        
        Time Complexity: O(1).
        """
        i = self._vertex_to_index.get(source)
        j = self._vertex_to_index.get(target)
        
        if i is None or j is None:
            return False
        
        return self._matrix[i][j] is not None
    
    def get_weight(self, source: T, target: T) -> float | None:
        """
        Get the weight of an edge.
        
        Returns None if the edge does not exist.
        
        Time Complexity: O(1).
        """
        i = self._vertex_to_index.get(source)
        j = self._vertex_to_index.get(target)
        
        if i is None or j is None:
            return None
        
        return self._matrix[i][j]
    
    def neighbours(self, vertex: T) -> Iterator[T]:
        """
        Yield all neighbours of a vertex.
        
        Time Complexity: O(V).
        """
        i = self._vertex_to_index.get(vertex)
        if i is None:
            return
        
        for j, weight in enumerate(self._matrix[i]):
            if weight is not None:
                yield self._index_to_vertex[j]
    
    @property
    def vertex_count(self) -> int:
        """Return the number of vertices."""
        return len(self._vertex_to_index)
    
    @property
    def edge_count(self) -> int:
        """Return the number of edges."""
        count = sum(
            1 for row in self._matrix for weight in row if weight is not None
        )
        return count if self.directed else count // 2
    
    @classmethod
    def from_edge_list(
        cls,
        edges: list[tuple[T, T] | tuple[T, T, float]],
        directed: bool = False
    ) -> AdjacencyMatrixGraph[T]:
        """
        Construct a graph from an edge list.
        
        Args:
            edges: List of edges.
            directed: Whether the graph is directed.
        
        Returns:
            A new AdjacencyMatrixGraph instance.
        """
        graph = cls(directed=directed)
        
        for edge in edges:
            if len(edge) == 2:
                source, target = edge
                weight = 1.0
            else:
                source, target, weight = edge
            graph.add_edge(source, target, weight)
        
        logger.info(
            f"Constructed matrix graph with {graph.vertex_count} vertices "
            f"and {graph.edge_count} edges"
        )
        return graph


def parse_edge_list_string(
    text: str,
    delimiter: str = ',',
    weighted: bool = False
) -> list[tuple[str, str] | tuple[str, str, float]]:
    """
    Parse an edge list from a string representation.
    
    Args:
        text: The input text, one edge per line.
        delimiter: The delimiter between vertex names.
        weighted: Whether to parse weights.
    
    Returns:
        A list of edge tuples.
    
    Examples:
        >>> text = "A,B\\nB,C\\nC,A"
        >>> parse_edge_list_string(text)
        [('A', 'B'), ('B', 'C'), ('C', 'A')]
        
        >>> text = "A,B,1.5\\nB,C,2.0"
        >>> parse_edge_list_string(text, weighted=True)
        [('A', 'B', 1.5), ('B', 'C', 2.0)]
    """
    edges = []
    
    for line_num, line in enumerate(text.strip().split('\n'), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = [p.strip() for p in line.split(delimiter)]
        
        if len(parts) < 2:
            logger.warning(f"Line {line_num}: Invalid edge format: {line}")
            continue
        
        source, target = parts[0], parts[1]
        
        if weighted and len(parts) >= 3:
            try:
                weight = float(parts[2])
                edges.append((source, target, weight))
            except ValueError:
                logger.warning(f"Line {line_num}: Invalid weight: {parts[2]}")
                edges.append((source, target))
        else:
            edges.append((source, target))
    
    return edges


def demonstrate_graph_construction() -> None:
    """Demonstrate graph construction techniques."""
    print("=" * 60)
    print("Graph Construction Demonstration")
    print("=" * 60)
    
    # Example 1: Simple undirected graph
    print("\n1. Undirected Graph from Edge List")
    print("-" * 40)
    
    edges_simple = [
        ('A', 'B'),
        ('B', 'C'),
        ('C', 'D'),
        ('D', 'A'),
        ('A', 'C'),
    ]
    
    graph_list = AdjacencyListGraph.from_edge_list(edges_simple)
    print(f"   Graph: {graph_list}")
    print(f"   Vertices: {sorted(graph_list.vertices)}")
    print(f"   Neighbours of 'A': {list(graph_list.neighbours('A'))}")
    print(f"   Has edge A-B: {graph_list.has_edge('A', 'B')}")
    print(f"   Has edge A-D: {graph_list.has_edge('A', 'D')}")
    
    # Example 2: Directed weighted graph
    print("\n2. Directed Weighted Graph")
    print("-" * 40)
    
    edges_weighted = [
        ('London', 'Paris', 344),
        ('Paris', 'Berlin', 878),
        ('Berlin', 'Vienna', 524),
        ('Vienna', 'Rome', 764),
        ('Rome', 'Paris', 1105),
    ]
    
    directed_graph = AdjacencyListGraph.from_edge_list(
        edges_weighted, directed=True
    )
    print(f"   Graph: {directed_graph}")
    
    for vertex in sorted(directed_graph.vertices):
        neighbours = list(directed_graph.neighbours_with_weights(vertex))
        print(f"   From {vertex}: {neighbours}")
    
    # Example 3: Matrix representation
    print("\n3. Adjacency Matrix Representation")
    print("-" * 40)
    
    matrix_graph = AdjacencyMatrixGraph.from_edge_list(edges_simple)
    print(f"   Graph: {matrix_graph}")
    print(f"   Edge A-B weight: {matrix_graph.get_weight('A', 'B')}")
    print(f"   Edge B-D exists: {matrix_graph.has_edge('B', 'D')}")
    
    # Example 4: Parse from string
    print("\n4. Parsing Edge List from String")
    print("-" * 40)
    
    edge_string = """
    # Social network connections
    Alice,Bob
    Bob,Charlie
    Charlie,Diana
    Diana,Alice
    Bob,Diana
    """
    
    parsed_edges = parse_edge_list_string(edge_string)
    social_graph = AdjacencyListGraph.from_edge_list(parsed_edges)
    print(f"   Parsed {len(parsed_edges)} edges")
    print(f"   Graph: {social_graph}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_graph_construction()
