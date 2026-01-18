"""
Pytest configuration and fixtures for Unit 8 tests.
"""

import pytest
import sys
from pathlib import Path

# Add lab directory to path
lab_dir = Path(__file__).parent.parent / "lab"
sys.path.insert(0, str(lab_dir))


@pytest.fixture
def small_array():
    """Small test array for sorting algorithms."""
    return [64, 34, 25, 12, 22, 11, 90]


@pytest.fixture
def sorted_array():
    """Sorted array for search algorithms."""
    return [11, 12, 22, 25, 34, 64, 90]


@pytest.fixture
def fibonacci_values():
    """First 20 Fibonacci numbers for validation."""
    return [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 
            610, 987, 1597, 2584, 4181]


@pytest.fixture
def knapsack_instance():
    """Standard knapsack test instance."""
    return {
        "weights": [2, 3, 4, 5],
        "values": [3, 4, 5, 6],
        "capacity": 8,
        "expected_max": 10
    }


@pytest.fixture
def lcs_instance():
    """Standard LCS test instance."""
    return {
        "s1": "ABCDGH",
        "s2": "AEDFHR",
        "expected_length": 3,
        "expected_lcs": "ADH"
    }


@pytest.fixture
def edit_distance_instance():
    """Standard edit distance test instance."""
    return {
        "s1": "kitten",
        "s2": "sitting",
        "expected_distance": 3
    }


@pytest.fixture
def matrix_chain_instance():
    """Standard matrix chain test instance."""
    return {
        "dimensions": [10, 30, 5, 60],
        "expected_cost": 4500
    }


@pytest.fixture
def binary_tree():
    """Sample binary tree for traversal tests."""
    from lab_08_01_recursive_patterns import TreeNode
    
    #       1
    #      / \
    #     2   3
    #    / \
    #   4   5
    
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)
    
    return root


@pytest.fixture
def n_queens_solutions():
    """Expected N-Queens solution counts."""
    return {
        1: 1,
        2: 0,
        3: 0,
        4: 2,
        5: 10,
        6: 4,
        7: 40,
        8: 92
    }
