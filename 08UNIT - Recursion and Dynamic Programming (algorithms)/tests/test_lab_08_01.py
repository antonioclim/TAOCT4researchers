"""
Tests for Lab 08.01: Recursive Patterns and Memoisation

Run with: pytest tests/test_lab_08_01.py -v
"""

import pytest
import sys
from pathlib import Path

# Add lab directory to path
lab_dir = Path(__file__).parent.parent / "lab"
sys.path.insert(0, str(lab_dir))

from lab_08_01_recursive_patterns import (
    factorial_naive,
    factorial_tail,
    fibonacci_naive,
    fibonacci_memoised,
    fibonacci_lru,
    fibonacci_iterative,
    binary_search_recursive,
    merge_sort,
    quick_sort,
    TreeNode,
    preorder_traversal,
    inorder_traversal,
    postorder_traversal,
    tree_height,
    tree_size,
    generate_permutations,
    generate_subsets,
    solve_n_queens,
    subset_sum,
    MemoCache,
)


class TestFactorial:
    """Tests for factorial implementations."""
    
    def test_factorial_base_cases(self):
        assert factorial_naive(0) == 1
        assert factorial_naive(1) == 1
        assert factorial_tail(0) == 1
        assert factorial_tail(1) == 1
    
    def test_factorial_small_values(self):
        expected = [1, 1, 2, 6, 24, 120, 720, 5040]
        for n, exp in enumerate(expected):
            assert factorial_naive(n) == exp
            assert factorial_tail(n) == exp
    
    def test_factorial_negative_raises(self):
        with pytest.raises(ValueError):
            factorial_naive(-1)
        with pytest.raises(ValueError):
            factorial_tail(-1)


class TestFibonacci:
    """Tests for Fibonacci implementations."""
    
    def test_fibonacci_base_cases(self, fibonacci_values):
        for impl in [fibonacci_naive, fibonacci_memoised, fibonacci_lru, fibonacci_iterative]:
            assert impl(0) == 0
            assert impl(1) == 1
    
    def test_fibonacci_small_values(self, fibonacci_values):
        # Test first 15 values (naive is too slow for larger)
        for n in range(15):
            assert fibonacci_naive(n) == fibonacci_values[n]
    
    def test_fibonacci_memoised_correctness(self, fibonacci_values):
        cache = MemoCache()
        for n in range(20):
            assert fibonacci_memoised(n, cache) == fibonacci_values[n]
    
    def test_fibonacci_lru_correctness(self, fibonacci_values):
        fibonacci_lru.cache_clear()
        for n in range(20):
            assert fibonacci_lru(n) == fibonacci_values[n]
    
    def test_fibonacci_iterative_correctness(self, fibonacci_values):
        for n in range(20):
            assert fibonacci_iterative(n) == fibonacci_values[n]
    
    def test_fibonacci_large_value(self):
        # Test memoised and iterative for large n
        expected = 12586269025
        cache = MemoCache()
        assert fibonacci_memoised(50, cache) == expected
        assert fibonacci_iterative(50) == expected
    
    def test_fibonacci_negative_raises(self):
        with pytest.raises(ValueError):
            fibonacci_naive(-1)
        with pytest.raises(ValueError):
            fibonacci_memoised(-1)
        with pytest.raises(ValueError):
            fibonacci_iterative(-1)


class TestBinarySearch:
    """Tests for binary search."""
    
    def test_binary_search_found(self, sorted_array):
        for i, val in enumerate(sorted_array):
            assert binary_search_recursive(sorted_array, val) == i
    
    def test_binary_search_not_found(self, sorted_array):
        assert binary_search_recursive(sorted_array, 100) == -1
        assert binary_search_recursive(sorted_array, 0) == -1
        assert binary_search_recursive(sorted_array, 15) == -1
    
    def test_binary_search_empty(self):
        assert binary_search_recursive([], 5) == -1
    
    def test_binary_search_single_element(self):
        assert binary_search_recursive([5], 5) == 0
        assert binary_search_recursive([5], 3) == -1


class TestSorting:
    """Tests for sorting algorithms."""
    
    def test_merge_sort_correctness(self, small_array):
        result = merge_sort(small_array)
        assert result == sorted(small_array)
    
    def test_merge_sort_stability(self):
        # Original should not be modified
        arr = [3, 1, 4, 1, 5]
        original = arr.copy()
        merge_sort(arr)
        assert arr == original
    
    def test_merge_sort_empty(self):
        assert merge_sort([]) == []
    
    def test_merge_sort_single(self):
        assert merge_sort([42]) == [42]
    
    def test_quick_sort_correctness(self, small_array):
        result = quick_sort(small_array)
        assert result == sorted(small_array)
    
    def test_quick_sort_duplicates(self):
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        assert quick_sort(arr) == sorted(arr)


class TestTreeTraversal:
    """Tests for tree traversal algorithms."""
    
    def test_preorder(self, binary_tree):
        assert preorder_traversal(binary_tree) == [1, 2, 4, 5, 3]
    
    def test_inorder(self, binary_tree):
        assert inorder_traversal(binary_tree) == [4, 2, 5, 1, 3]
    
    def test_postorder(self, binary_tree):
        assert postorder_traversal(binary_tree) == [4, 5, 2, 3, 1]
    
    def test_traversal_empty(self):
        assert preorder_traversal(None) == []
        assert inorder_traversal(None) == []
        assert postorder_traversal(None) == []
    
    def test_tree_height(self, binary_tree):
        assert tree_height(binary_tree) == 2
        assert tree_height(None) == -1
        assert tree_height(TreeNode(1)) == 0
    
    def test_tree_size(self, binary_tree):
        assert tree_size(binary_tree) == 5
        assert tree_size(None) == 0
        assert tree_size(TreeNode(1)) == 1


class TestBacktracking:
    """Tests for backtracking algorithms."""
    
    def test_permutations_length(self):
        perms = generate_permutations([1, 2, 3])
        assert len(perms) == 6  # 3! = 6
    
    def test_permutations_content(self):
        perms = generate_permutations([1, 2])
        assert [1, 2] in perms
        assert [2, 1] in perms
    
    def test_subsets_length(self):
        subsets = generate_subsets([1, 2, 3])
        assert len(subsets) == 8  # 2^3 = 8
    
    def test_subsets_content(self):
        subsets = generate_subsets([1, 2])
        assert [] in subsets
        assert [1] in subsets
        assert [2] in subsets
        assert [1, 2] in subsets
    
    def test_n_queens_count(self, n_queens_solutions):
        for n in [1, 4, 5]:
            solutions = solve_n_queens(n)
            assert len(solutions) == n_queens_solutions[n]
    
    def test_n_queens_validity(self):
        solutions = solve_n_queens(4)
        for sol in solutions:
            # Check no two queens in same column
            assert len(sol) == len(set(sol))
            # Check no two queens on same diagonal
            for i in range(len(sol)):
                for j in range(i + 1, len(sol)):
                    assert abs(i - j) != abs(sol[i] - sol[j])
    
    def test_subset_sum_correctness(self):
        results = subset_sum([2, 3, 5, 7], 10)
        for subset in results:
            assert sum(subset) == 10


class TestMemoCache:
    """Tests for MemoCache utility class."""
    
    def test_cache_basic(self):
        cache = MemoCache()
        cache.set("key1", 42)
        found, value = cache.get("key1")
        assert found is True
        assert value == 42
    
    def test_cache_miss(self):
        cache = MemoCache()
        found, value = cache.get("nonexistent")
        assert found is False
    
    def test_cache_stats(self):
        cache = MemoCache()
        cache.set(1, 100)
        cache.get(1)  # Hit
        cache.get(2)  # Miss
        cache.get(1)  # Hit
        
        assert cache.stats.hits == 2
        assert cache.stats.misses == 1
    
    def test_cache_clear(self):
        cache = MemoCache()
        cache.set(1, 100)
        cache.clear()
        found, _ = cache.get(1)
        assert found is False
        assert cache.stats.hits == 0
