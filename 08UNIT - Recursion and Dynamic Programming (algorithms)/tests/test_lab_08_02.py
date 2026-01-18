"""
Tests for Lab 08.02: Dynamic Programming

Run with: pytest tests/test_lab_08_02.py -v
"""

import pytest
import sys
from pathlib import Path

# Add lab directory to path
lab_dir = Path(__file__).parent.parent / "lab"
sys.path.insert(0, str(lab_dir))

from lab_08_02_dynamic_programming import (
    fibonacci_tabulated,
    fibonacci_space_optimised,
    knapsack_recursive,
    knapsack_memoised,
    knapsack_tabulated,
    knapsack_space_optimised,
    lcs_tabulated,
    lcs_space_optimised,
    edit_distance,
    matrix_chain_order,
)


class TestFibonacciDP:
    """Tests for DP Fibonacci implementations."""
    
    def test_tabulated_base_cases(self):
        assert fibonacci_tabulated(0) == 0
        assert fibonacci_tabulated(1) == 1
    
    def test_tabulated_correctness(self, fibonacci_values):
        for n in range(20):
            assert fibonacci_tabulated(n) == fibonacci_values[n]
    
    def test_space_optimised_correctness(self, fibonacci_values):
        for n in range(20):
            assert fibonacci_space_optimised(n) == fibonacci_values[n]
    
    def test_large_fibonacci(self):
        expected = 12586269025
        assert fibonacci_tabulated(50) == expected
        assert fibonacci_space_optimised(50) == expected
    
    def test_negative_raises(self):
        with pytest.raises(ValueError):
            fibonacci_tabulated(-1)
        with pytest.raises(ValueError):
            fibonacci_space_optimised(-1)


class TestKnapsack:
    """Tests for 0-1 Knapsack implementations."""
    
    def test_recursive_correctness(self, knapsack_instance):
        result = knapsack_recursive(
            knapsack_instance["weights"],
            knapsack_instance["values"],
            knapsack_instance["capacity"]
        )
        assert result == knapsack_instance["expected_max"]
    
    def test_memoised_correctness(self, knapsack_instance):
        result = knapsack_memoised(
            knapsack_instance["weights"],
            knapsack_instance["values"],
            knapsack_instance["capacity"]
        )
        assert result == knapsack_instance["expected_max"]
    
    def test_tabulated_correctness(self, knapsack_instance):
        result = knapsack_tabulated(
            knapsack_instance["weights"],
            knapsack_instance["values"],
            knapsack_instance["capacity"]
        )
        assert result.max_value == knapsack_instance["expected_max"]
    
    def test_space_optimised_correctness(self, knapsack_instance):
        result = knapsack_space_optimised(
            knapsack_instance["weights"],
            knapsack_instance["values"],
            knapsack_instance["capacity"]
        )
        assert result == knapsack_instance["expected_max"]
    
    def test_item_selection(self, knapsack_instance):
        result = knapsack_tabulated(
            knapsack_instance["weights"],
            knapsack_instance["values"],
            knapsack_instance["capacity"]
        )
        # Verify selected items are valid
        total_weight = sum(knapsack_instance["weights"][i] for i in result.selected_items)
        total_value = sum(knapsack_instance["values"][i] for i in result.selected_items)
        assert total_weight <= knapsack_instance["capacity"]
        assert total_value == result.max_value
    
    def test_empty_knapsack(self):
        assert knapsack_recursive([], [], 10) == 0
        assert knapsack_tabulated([], [], 10).max_value == 0
    
    def test_zero_capacity(self):
        assert knapsack_recursive([1, 2, 3], [10, 20, 30], 0) == 0
    
    def test_single_item_fits(self):
        result = knapsack_tabulated([5], [10], 10)
        assert result.max_value == 10
        assert 0 in result.selected_items
    
    def test_single_item_too_heavy(self):
        result = knapsack_tabulated([15], [10], 10)
        assert result.max_value == 0
        assert len(result.selected_items) == 0


class TestLCS:
    """Tests for Longest Common Subsequence."""
    
    def test_basic_lcs(self, lcs_instance):
        result = lcs_tabulated(lcs_instance["s1"], lcs_instance["s2"])
        assert result.length == lcs_instance["expected_length"]
        assert result.subsequence == lcs_instance["expected_lcs"]
    
    def test_space_optimised(self, lcs_instance):
        result = lcs_space_optimised(lcs_instance["s1"], lcs_instance["s2"])
        assert result == lcs_instance["expected_length"]
    
    def test_identical_strings(self):
        result = lcs_tabulated("ABCD", "ABCD")
        assert result.length == 4
        assert result.subsequence == "ABCD"
    
    def test_no_common(self):
        result = lcs_tabulated("ABC", "XYZ")
        assert result.length == 0
        assert result.subsequence == ""
    
    def test_empty_string(self):
        assert lcs_tabulated("", "ABC").length == 0
        assert lcs_tabulated("ABC", "").length == 0
        assert lcs_tabulated("", "").length == 0
    
    def test_single_char_match(self):
        result = lcs_tabulated("A", "A")
        assert result.length == 1
        assert result.subsequence == "A"
    
    def test_single_char_no_match(self):
        result = lcs_tabulated("A", "B")
        assert result.length == 0


class TestEditDistance:
    """Tests for Edit Distance (Levenshtein)."""
    
    def test_basic_edit_distance(self, edit_distance_instance):
        result = edit_distance(
            edit_distance_instance["s1"],
            edit_distance_instance["s2"]
        )
        assert result.distance == edit_distance_instance["expected_distance"]
    
    def test_identical_strings(self):
        result = edit_distance("hello", "hello")
        assert result.distance == 0
    
    def test_empty_to_string(self):
        result = edit_distance("", "abc")
        assert result.distance == 3  # 3 insertions
    
    def test_string_to_empty(self):
        result = edit_distance("abc", "")
        assert result.distance == 3  # 3 deletions
    
    def test_single_substitution(self):
        result = edit_distance("cat", "bat")
        assert result.distance == 1
    
    def test_single_insertion(self):
        result = edit_distance("cat", "cart")
        assert result.distance == 1
    
    def test_single_deletion(self):
        result = edit_distance("cart", "cat")
        assert result.distance == 1
    
    def test_operations_list(self, edit_distance_instance):
        result = edit_distance(
            edit_distance_instance["s1"],
            edit_distance_instance["s2"]
        )
        # Verify operations are provided
        assert len(result.operations) > 0


class TestMatrixChain:
    """Tests for Matrix Chain Multiplication."""
    
    def test_basic_matrix_chain(self, matrix_chain_instance):
        result = matrix_chain_order(matrix_chain_instance["dimensions"])
        assert result.min_operations == matrix_chain_instance["expected_cost"]
    
    def test_two_matrices(self):
        # A1: 10x20, A2: 20x30
        result = matrix_chain_order([10, 20, 30])
        assert result.min_operations == 10 * 20 * 30
    
    def test_parenthesisation_format(self, matrix_chain_instance):
        result = matrix_chain_order(matrix_chain_instance["dimensions"])
        # Should contain matrix names and parentheses
        assert "A" in result.optimal_parenthesisation
        assert "×" in result.optimal_parenthesisation
    
    def test_single_matrix(self):
        # Single matrix = no multiplication needed
        result = matrix_chain_order([10, 20])
        assert result.min_operations == 0
    
    def test_three_matrices(self):
        # A1: 10x30, A2: 30x5, A3: 5x60
        result = matrix_chain_order([10, 30, 5, 60])
        # Optimal: (A1 × A2) × A3 = 10*30*5 + 10*5*60 = 1500 + 3000 = 4500
        assert result.min_operations == 4500


class TestDPTableGeneration:
    """Tests verifying DP tables are generated correctly."""
    
    def test_knapsack_table_dimensions(self, knapsack_instance):
        result = knapsack_tabulated(
            knapsack_instance["weights"],
            knapsack_instance["values"],
            knapsack_instance["capacity"]
        )
        n = len(knapsack_instance["weights"])
        w = knapsack_instance["capacity"]
        assert len(result.dp_table) == n + 1
        assert len(result.dp_table[0]) == w + 1
    
    def test_lcs_table_dimensions(self, lcs_instance):
        result = lcs_tabulated(lcs_instance["s1"], lcs_instance["s2"])
        m = len(lcs_instance["s1"])
        n = len(lcs_instance["s2"])
        assert len(result.dp_table) == m + 1
        assert len(result.dp_table[0]) == n + 1
    
    def test_edit_distance_table_dimensions(self, edit_distance_instance):
        result = edit_distance(
            edit_distance_instance["s1"],
            edit_distance_instance["s2"]
        )
        m = len(edit_distance_instance["s1"])
        n = len(edit_distance_instance["s2"])
        assert len(result.dp_table) == m + 1
        assert len(result.dp_table[0]) == n + 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_knapsack_all_items_too_heavy(self):
        result = knapsack_tabulated([10, 20, 30], [1, 2, 3], 5)
        assert result.max_value == 0
    
    def test_knapsack_exact_fit(self):
        result = knapsack_tabulated([5, 5], [10, 10], 10)
        assert result.max_value == 20
        assert len(result.selected_items) == 2
    
    def test_lcs_repeated_chars(self):
        result = lcs_tabulated("AAA", "AA")
        assert result.length == 2
    
    def test_edit_distance_repeated_chars(self):
        result = edit_distance("AAA", "AAAA")
        assert result.distance == 1
