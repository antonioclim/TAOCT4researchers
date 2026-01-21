#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTIONS: Lab 08.01 - Recursive Patterns and Memoisation
═══════════════════════════════════════════════════════════════════════════════

Reference implementations for all laboratory exercises.
These solutions demonstrate recommended methods and optimal approaches.

WARNING: Students should attempt exercises before consulting solutions.

© 2026 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, TypeVar

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FUNDAMENTAL RECURSION
# ═══════════════════════════════════════════════════════════════════════════════

def factorial_solution(n: int) -> int:
    """
    Solution: Factorial with input validation.
    
    Time: O(n), Space: O(n)
    """
    if n < 0:
        raise ValueError(f"Factorial undefined for negative: {n}")
    if n <= 1:
        return 1
    return n * factorial_solution(n - 1)


def fibonacci_memoised_solution(n: int, memo: dict[int, int] | None = None) -> int:
    """
    Solution: Fibonacci with explicit memoisation.
    
    Time: O(n), Space: O(n)
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative: {n}")
    
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 1:
        return n
    
    memo[n] = fibonacci_memoised_solution(n - 1, memo) + \
              fibonacci_memoised_solution(n - 2, memo)
    return memo[n]


@lru_cache(maxsize=None)
def fibonacci_lru_solution(n: int) -> int:
    """
    Solution: Fibonacci with @lru_cache decorator.
    
    Time: O(n), Space: O(n)
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative: {n}")
    if n <= 1:
        return n
    return fibonacci_lru_solution(n - 1) + fibonacci_lru_solution(n - 2)


def fibonacci_iterative_solution(n: int) -> int:
    """
    Solution: Fibonacci with O(1) space.
    
    Time: O(n), Space: O(1)
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative: {n}")
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DIVIDE AND CONQUER
# ═══════════════════════════════════════════════════════════════════════════════

def binary_search_solution(
    arr: list[T],
    target: T,
    low: int = 0,
    high: int | None = None
) -> int:
    """
    Solution: Binary search recursive implementation.
    
    Time: O(log n), Space: O(log n)
    """
    if high is None:
        high = len(arr) - 1
    
    if low > high:
        return -1
    
    mid = (low + high) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search_solution(arr, target, low, mid - 1)
    else:
        return binary_search_solution(arr, target, mid + 1, high)


def merge_sort_solution(arr: list[T]) -> list[T]:
    """
    Solution: Merge sort with auxiliary merge function.
    
    Time: O(n log n), Space: O(n)
    """
    if len(arr) <= 1:
        return arr.copy()
    
    mid = len(arr) // 2
    left = merge_sort_solution(arr[:mid])
    right = merge_sort_solution(arr[mid:])
    
    return _merge_solution(left, right)


def _merge_solution(left: list[T], right: list[T]) -> list[T]:
    """Merge two sorted lists."""
    result: list[T] = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result


def quick_sort_solution(arr: list[T]) -> list[T]:
    """
    Solution: Quick sort with last element as pivot.
    
    Time: O(n log n) average, O(n²) worst
    Space: O(log n) average
    """
    if len(arr) <= 1:
        return arr.copy()
    
    pivot = arr[-1]
    less = [x for x in arr[:-1] if x <= pivot]
    greater = [x for x in arr[:-1] if x > pivot]
    
    return quick_sort_solution(less) + [pivot] + quick_sort_solution(greater)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TREE OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TreeNode:
    """Binary tree node."""
    def __init__(
        self,
        value: Any,
        left: TreeNode | None = None,
        right: TreeNode | None = None
    ):
        self.value = value
        self.left = left
        self.right = right


def preorder_solution(root: TreeNode | None) -> list[Any]:
    """Solution: Preorder traversal (root, left, right)."""
    if root is None:
        return []
    return [root.value] + preorder_solution(root.left) + preorder_solution(root.right)


def inorder_solution(root: TreeNode | None) -> list[Any]:
    """Solution: Inorder traversal (left, root, right)."""
    if root is None:
        return []
    return inorder_solution(root.left) + [root.value] + inorder_solution(root.right)


def postorder_solution(root: TreeNode | None) -> list[Any]:
    """Solution: Postorder traversal (left, right, root)."""
    if root is None:
        return []
    return postorder_solution(root.left) + postorder_solution(root.right) + [root.value]


def tree_height_solution(root: TreeNode | None) -> int:
    """Solution: Compute tree height."""
    if root is None:
        return -1
    return 1 + max(tree_height_solution(root.left), tree_height_solution(root.right))


def tree_size_solution(root: TreeNode | None) -> int:
    """Solution: Count nodes in tree."""
    if root is None:
        return 0
    return 1 + tree_size_solution(root.left) + tree_size_solution(root.right)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BACKTRACKING
# ═══════════════════════════════════════════════════════════════════════════════

def permutations_solution(elements: list[T]) -> list[list[T]]:
    """
    Solution: Generate all permutations.
    
    Time: O(n! × n), Space: O(n!)
    """
    results: list[list[T]] = []
    
    def backtrack(current: list[T], remaining: list[T]) -> None:
        if not remaining:
            results.append(current.copy())
            return
        
        for i, elem in enumerate(remaining):
            current.append(elem)
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()
    
    backtrack([], elements)
    return results


def subsets_solution(elements: list[T]) -> list[list[T]]:
    """
    Solution: Generate power set.
    
    Time: O(2ⁿ × n), Space: O(2ⁿ)
    """
    results: list[list[T]] = []
    
    def backtrack(index: int, current: list[T]) -> None:
        if index == len(elements):
            results.append(current.copy())
            return
        
        # Exclude
        backtrack(index + 1, current)
        
        # Include
        current.append(elements[index])
        backtrack(index + 1, current)
        current.pop()
    
    backtrack(0, [])
    return results


def n_queens_solution(n: int) -> list[list[int]]:
    """
    Solution: Solve N-Queens problem.
    
    Time: O(n!), Space: O(n)
    """
    solutions: list[list[int]] = []
    
    def is_safe(queens: list[int], row: int, col: int) -> bool:
        for prev_row, prev_col in enumerate(queens):
            if prev_col == col:
                return False
            if abs(prev_row - row) == abs(prev_col - col):
                return False
        return True
    
    def backtrack(queens: list[int]) -> None:
        row = len(queens)
        if row == n:
            solutions.append(queens.copy())
            return
        
        for col in range(n):
            if is_safe(queens, row, col):
                queens.append(col)
                backtrack(queens)
                queens.pop()
    
    backtrack([])
    return solutions


def subset_sum_solution(numbers: list[int], target: int) -> list[list[int]]:
    """
    Solution: Find subsets summing to target.
    
    Time: O(2ⁿ), Space: O(n)
    """
    results: list[list[int]] = []
    numbers_sorted = sorted(numbers, reverse=True)
    
    def backtrack(index: int, current: list[int], remaining: int) -> None:
        if remaining == 0:
            results.append(current.copy())
            return
        
        if remaining < 0 or index >= len(numbers_sorted):
            return
        
        # Pruning
        if numbers_sorted[index] > remaining:
            backtrack(index + 1, current, remaining)
            return
        
        # Include
        current.append(numbers_sorted[index])
        backtrack(index + 1, current, remaining - numbers_sorted[index])
        current.pop()
        
        # Exclude
        backtrack(index + 1, current, remaining)
    
    backtrack(0, [], target)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Verify solutions
    assert factorial_solution(5) == 120
    assert fibonacci_iterative_solution(10) == 55
    assert merge_sort_solution([3, 1, 4, 1, 5]) == [1, 1, 3, 4, 5]
    assert len(n_queens_solution(4)) == 2
    print("All solutions verified successfully.")
