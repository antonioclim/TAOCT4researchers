#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4: Advanced Data Structures - Test Fixtures
═══════════════════════════════════════════════════════════════════════════════

Shared pytest fixtures for Week 4 test suite.

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def empty_adjacency_list() -> dict[str, list[tuple[str, float]]]:
    """Provide an empty adjacency list."""
    return {}


@pytest.fixture
def simple_undirected_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a simple undirected graph.
    
    Structure:
        A --- B
        |     |
        C --- D
    
    All edges have weight 1.0.
    """
    return {
        "A": [("B", 1.0), ("C", 1.0)],
        "B": [("A", 1.0), ("D", 1.0)],
        "C": [("A", 1.0), ("D", 1.0)],
        "D": [("B", 1.0), ("C", 1.0)],
    }


@pytest.fixture
def simple_directed_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a simple directed graph.
    
    Structure:
        A → B → D
        ↓   ↓
        C → E
    
    All edges have weight 1.0.
    """
    return {
        "A": [("B", 1.0), ("C", 1.0)],
        "B": [("D", 1.0), ("E", 1.0)],
        "C": [("E", 1.0)],
        "D": [],
        "E": [],
    }


@pytest.fixture
def weighted_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a weighted directed graph for shortest path testing.
    
    Structure:
        A --5-- B
        |       |
        2       3
        |       |
        C --1-- D
    
    Shortest A→D: A→C→D = 3 (not A→B→D = 8)
    """
    return {
        "A": [("B", 5.0), ("C", 2.0)],
        "B": [("D", 3.0)],
        "C": [("D", 1.0)],
        "D": [],
    }


@pytest.fixture
def dag_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a directed acyclic graph for topological sort testing.
    
    Structure (dependencies flow downward):
        A → B → D
        ↓   ↓   ↓
        C → E → F
    
    Valid topological orders include: [A, B, C, D, E, F], [A, C, B, E, D, F]
    """
    return {
        "A": [("B", 1.0), ("C", 1.0)],
        "B": [("D", 1.0), ("E", 1.0)],
        "C": [("E", 1.0)],
        "D": [("F", 1.0)],
        "E": [("F", 1.0)],
        "F": [],
    }


@pytest.fixture
def cyclic_directed_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a directed graph with a cycle.
    
    Structure:
        A → B → C
        ↑       ↓
        └───────┘
    
    Cycle: A → B → C → A
    """
    return {
        "A": [("B", 1.0)],
        "B": [("C", 1.0)],
        "C": [("A", 1.0)],
    }


@pytest.fixture
def cyclic_undirected_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide an undirected graph with a cycle.
    
    Structure:
        A --- B
        |     |
        └──C──┘
    
    Cycle: A - B - C - A
    """
    return {
        "A": [("B", 1.0), ("C", 1.0)],
        "B": [("A", 1.0), ("C", 1.0)],
        "C": [("A", 1.0), ("B", 1.0)],
    }


@pytest.fixture
def disconnected_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a disconnected graph with two components.
    
    Component 1: A --- B
    Component 2: C --- D
    """
    return {
        "A": [("B", 1.0)],
        "B": [("A", 1.0)],
        "C": [("D", 1.0)],
        "D": [("C", 1.0)],
    }


@pytest.fixture
def bipartite_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a bipartite graph.
    
    Set 1: {A, B, C}
    Set 2: {X, Y}
    
    All edges go between sets.
    """
    return {
        "A": [("X", 1.0), ("Y", 1.0)],
        "B": [("X", 1.0)],
        "C": [("Y", 1.0)],
        "X": [("A", 1.0), ("B", 1.0)],
        "Y": [("A", 1.0), ("C", 1.0)],
    }


@pytest.fixture
def non_bipartite_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a non-bipartite graph (contains odd cycle).
    
    Structure (triangle):
        A --- B
         \   /
          \ /
           C
    """
    return {
        "A": [("B", 1.0), ("C", 1.0)],
        "B": [("A", 1.0), ("C", 1.0)],
        "C": [("A", 1.0), ("B", 1.0)],
    }


@pytest.fixture
def large_graph() -> dict[str, list[tuple[str, float]]]:
    """
    Provide a larger graph for performance testing.
    
    Creates a grid graph with 100 nodes (10x10).
    """
    graph: dict[str, list[tuple[str, float]]] = {}
    size = 10
    
    for i in range(size):
        for j in range(size):
            node = f"N{i}_{j}"
            graph[node] = []
            
            # Connect to right neighbour
            if j < size - 1:
                neighbour = f"N{i}_{j + 1}"
                graph[node].append((neighbour, 1.0))
            
            # Connect to bottom neighbour
            if i < size - 1:
                neighbour = f"N{i + 1}_{j}"
                graph[node].append((neighbour, 1.0))
            
            # Connect from left neighbour (for undirected)
            if j > 0:
                neighbour = f"N{i}_{j - 1}"
                graph[node].append((neighbour, 1.0))
            
            # Connect from top neighbour (for undirected)
            if i > 0:
                neighbour = f"N{i - 1}_{j}"
                graph[node].append((neighbour, 1.0))
    
    return graph


# ═══════════════════════════════════════════════════════════════════════════════
# BLOOM FILTER FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def small_word_set() -> set[str]:
    """Provide a small set of words for Bloom filter testing."""
    return {"apple", "banana", "cherry", "date", "elderberry"}


@pytest.fixture
def medium_word_set() -> set[str]:
    """Provide a medium set of words for Bloom filter testing."""
    return {f"word_{i}" for i in range(100)}


@pytest.fixture
def large_word_set() -> set[str]:
    """Provide a large set of words for Bloom filter testing."""
    return {f"item_{i}" for i in range(10000)}


@pytest.fixture
def test_negative_set() -> set[str]:
    """Provide a set of words guaranteed not in small_word_set."""
    return {"fig", "grape", "honeydew", "kiwi", "lemon"}


# ═══════════════════════════════════════════════════════════════════════════════
# COUNT-MIN SKETCH FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def frequency_data() -> list[str]:
    """
    Provide frequency data for Count-Min sketch testing.
    
    Returns a list with known frequencies:
    - "a": 100 occurrences
    - "b": 50 occurrences
    - "c": 25 occurrences
    - "d": 10 occurrences
    - "e": 5 occurrences
    """
    return (
        ["a"] * 100 +
        ["b"] * 50 +
        ["c"] * 25 +
        ["d"] * 10 +
        ["e"] * 5
    )


@pytest.fixture
def zipf_data() -> list[str]:
    """
    Provide Zipf-distributed data for realistic testing.
    
    Simulates a power-law distribution common in real-world data.
    """
    import random
    random.seed(42)
    
    data: list[str] = []
    for i in range(1, 101):
        # Zipf: frequency ∝ 1/rank
        count = int(1000 / i)
        data.extend([f"term_{i}"] * count)
    
    random.shuffle(data)
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def random_seed() -> Generator[int, None, None]:
    """
    Provide a fixed random seed for reproducible tests.
    
    Yields the seed and resets random state after test.
    """
    import random
    
    seed = 42
    original_state = random.getstate()
    random.seed(seed)
    
    yield seed
    
    random.setstate(original_state)


@pytest.fixture
def temp_file(tmp_path: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Provide a temporary file path for testing file operations."""
    file_path = tmp_path / "test_output.txt"  # type: ignore[operator]
    yield str(file_path)


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRISED DATA
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate parametrised test cases."""
    # Parametrise graph traversal start nodes
    if "start_node" in metafunc.fixturenames:
        metafunc.parametrize("start_node", ["A", "B", "C", "D"])
    
    # Parametrise Bloom filter sizes
    if "bloom_size" in metafunc.fixturenames:
        metafunc.parametrize("bloom_size", [100, 1000, 10000])
    
    # Parametrise false positive rates
    if "target_fpr" in metafunc.fixturenames:
        metafunc.parametrize("target_fpr", [0.1, 0.01, 0.001])
