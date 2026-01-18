#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Unit 8, Lab 1: Recursive Patterns and Memoisation
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Recursive thinking embodies a powerful form of reductionism—the discipline of defining
something in terms of simpler versions of itself. This laboratory develops
systematic proficiency in recursive algorithm design, from identifying base
cases through memoisation optimisation and complexity analysis.

The critical insight is that recursion's inefficiency is not inherent to the
paradigm itself but rather to specific implementations that fail to address
redundant computation. Techniques like memoisation can transform exponential
algorithms into linear ones without sacrificing the clarity of the recursive
approach (Cormen et al., 2009).

PREREQUISITES
─────────────
- 01UNIT: Computational foundations, recursive definitions
- 02UNIT: Abstraction patterns, function composition
- 03UNIT: Complexity analysis, Big-O notation
- Python: Intermediate (recursion, decorators, type hints)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement recursive solutions with appropriate base cases
2. Analyse recursive algorithms using recurrence relations
3. Apply memoisation to eliminate redundant computation
4. Transform naive recursion into optimised implementations

ESTIMATED TIME
──────────────
- Reading: 45 minutes
- Coding: 135 minutes
- Total: 180 minutes

DEPENDENCIES
────────────
Python 3.12+ (for type parameter syntax)

LICENCE
───────
© 2026 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from typing import Any, ParamSpec, TypeVar

# Configure logging (no print statements per specification)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# Type variables for generic implementations
T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: MEMOISATION INFRASTRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoStats:
    """Statistics for memoisation cache performance.
    
    Tracks cache hits, misses and computation time to evaluate
    the effectiveness of memoisation strategies.
    
    Attributes:
        hits: Number of cache hits (results retrieved from cache).
        misses: Number of cache misses (new computations required).
        total_time_ns: Total computation time in nanoseconds.
    """
    hits: int = 0
    misses: int = 0
    total_time_ns: int = 0
    
    @property
    def total_calls(self) -> int:
        """Return total number of function calls."""
        return self.hits + self.misses
    
    @property
    def hit_ratio(self) -> float:
        """Return cache hit ratio as a percentage.
        
        Returns:
            Hit ratio between 0.0 and 100.0, or 0.0 if no calls made.
        """
        if self.total_calls == 0:
            return 0.0
        return (self.hits / self.total_calls) * 100.0
    
    def __str__(self) -> str:
        """Return human-readable statistics summary."""
        return (
            f"Calls: {self.total_calls} | "
            f"Hits: {self.hits} | "
            f"Misses: {self.misses} | "
            f"Hit Ratio: {self.hit_ratio:.1f}%"
        )


def memoize_with_stats(
    func: Callable[P, R]
) -> tuple[Callable[P, R], MemoStats, dict[Hashable, R]]:
    """Decorator that adds memoisation with performance tracking.
    
    Unlike functools.lru_cache, this implementation exposes the cache
    and statistics for educational analysis.
    
    Args:
        func: The function to memoize.
    
    Returns:
        Tuple of (wrapped function, statistics object, cache dictionary).
    
    Example:
        >>> @memoize_with_stats
        ... def fib(n):
        ...     if n <= 1: return n
        ...     return fib(n-1) + fib(n-2)
        >>> fib_func, stats, cache = fib
        >>> result = fib_func(10)
        >>> logger.info(stats)
    """
    cache: dict[Hashable, R] = {}
    stats = MemoStats()
    
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Create hashable key from arguments
        key: Hashable = args
        if kwargs:
            key = (args, tuple(sorted(kwargs.items())))
        
        start_time = time.perf_counter_ns()
        
        if key in cache:
            stats.hits += 1
            result = cache[key]
        else:
            stats.misses += 1
            result = func(*args, **kwargs)
            cache[key] = result
        
        stats.total_time_ns += time.perf_counter_ns() - start_time
        return result
    
    return wrapper, stats, cache


class MemoCache:
    """Explicit memoisation cache with configurable eviction.
    
    Provides a reusable cache object that can be passed to recursive
    functions, allowing fine-grained control over caching behaviour.
    
    Attributes:
        max_size: Maximum number of entries (None for unlimited).
        _cache: Internal dictionary storing cached values.
        _stats: Statistics tracking cache performance.
    """
    
    def __init__(self, max_size: int | None = None) -> None:
        """Initialise cache with optional size limit.
        
        Args:
            max_size: Maximum entries before eviction (None = unlimited).
        """
        self.max_size = max_size
        self._cache: dict[Hashable, Any] = {}
        self._stats = MemoStats()
    
    def get(self, key: Hashable) -> tuple[bool, Any]:
        """Retrieve value from cache.
        
        Args:
            key: The cache key to look up.
        
        Returns:
            Tuple of (found: bool, value: Any). Value is None if not found.
        """
        if key in self._cache:
            self._stats.hits += 1
            return True, self._cache[key]
        self._stats.misses += 1
        return False, None
    
    def set(self, key: Hashable, value: Any) -> None:
        """Store value in cache.
        
        If max_size is set and exceeded, evicts oldest entry.
        
        Args:
            key: The cache key.
            value: The value to store.
        """
        if self.max_size and len(self._cache) >= self.max_size:
            # Simple FIFO eviction (first key added)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = value
    
    def contains(self, key: Hashable) -> bool:
        """Check if key exists in cache without affecting stats."""
        return key in self._cache
    
    @property
    def stats(self) -> MemoStats:
        """Return cache statistics."""
        return self._stats
    
    def clear(self) -> None:
        """Clear all cached values and reset statistics."""
        self._cache.clear()
        self._stats = MemoStats()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FUNDAMENTAL RECURSIVE PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

def factorial_naive(n: int) -> int:
    """Compute factorial using naive recursion.
    
    Demonstrates linear recursion where each call makes exactly one
    recursive call, forming a single chain of depth n.
    
    Mathematical definition:
        n! = n × (n-1)! for n > 0
        0! = 1
    
    Args:
        n: Non-negative integer.
    
    Returns:
        n factorial (n!).
    
    Raises:
        ValueError: If n is negative.
    
    Complexity:
        Time: O(n) — n recursive calls.
        Space: O(n) — recursion stack depth.
    
    Example:
        >>> factorial_naive(5)
        120
    """
    if n < 0:
        raise ValueError(f"Factorial undefined for negative numbers: {n}")
    
    # Base case: 0! = 1! = 1
    if n <= 1:
        return 1
    
    # Recursive case: n! = n × (n-1)!
    return n * factorial_naive(n - 1)


def factorial_tail(n: int, accumulator: int = 1) -> int:
    """Compute factorial using tail recursion.
    
    Tail recursion places the recursive call as the last operation,
    enabling potential optimisation by compilers/interpreters that
    support tail call elimination (note: CPython does not).
    
    Args:
        n: Non-negative integer.
        accumulator: Running product (internal use).
    
    Returns:
        n factorial (n!).
    
    Complexity:
        Time: O(n).
        Space: O(n) in Python (O(1) with tail call optimisation).
    
    Example:
        >>> factorial_tail(5)
        120
    """
    if n < 0:
        raise ValueError(f"Factorial undefined for negative numbers: {n}")
    
    if n <= 1:
        return accumulator
    
    return factorial_tail(n - 1, n * accumulator)


def fibonacci_naive(n: int) -> int:
    """Compute nth Fibonacci number using naive recursion.
    
    Demonstrates binary recursion with overlapping subproblems,
    resulting in exponential time complexity. This implementation
    is intentionally inefficient for educational comparison.
    
    The Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...
    
    Recurrence relation:
        F(n) = F(n-1) + F(n-2) for n > 1
        F(0) = 0, F(1) = 1
    
    Args:
        n: Non-negative integer index (0-indexed).
    
    Returns:
        The nth Fibonacci number.
    
    Raises:
        ValueError: If n is negative.
    
    Complexity:
        Time: O(φⁿ) where φ ≈ 1.618 (golden ratio).
        Space: O(n) for recursion stack.
    
    Warning:
        Impractical for n > 35 due to exponential growth.
    
    Example:
        >>> fibonacci_naive(10)
        55
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative index: {n}")
    
    # Base cases
    if n <= 1:
        return n
    
    # Recursive case (exponential due to overlapping subproblems)
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memoised(
    n: int,
    cache: MemoCache | None = None
) -> int:
    """Compute nth Fibonacci number with memoisation.
    
    Eliminates redundant computation by caching previously computed
    results. Transforms exponential O(φⁿ) time to linear O(n).
    
    Args:
        n: Non-negative integer index.
        cache: Optional MemoCache instance for result storage.
    
    Returns:
        The nth Fibonacci number.
    
    Complexity:
        Time: O(n) — each subproblem computed once.
        Space: O(n) for cache plus O(n) for recursion stack.
    
    Example:
        >>> cache = MemoCache()
        >>> fibonacci_memoised(50, cache)
        12586269025
        >>> logger.info(cache.stats)
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative index: {n}")
    
    if cache is None:
        cache = MemoCache()
    
    # Check cache first
    found, value = cache.get(n)
    if found:
        return value
    
    # Base cases
    if n <= 1:
        cache.set(n, n)
        return n
    
    # Recursive case with caching
    result = fibonacci_memoised(n - 1, cache) + fibonacci_memoised(n - 2, cache)
    cache.set(n, result)
    return result


@lru_cache(maxsize=None)
def fibonacci_lru(n: int) -> int:
    """Compute nth Fibonacci using functools.lru_cache.
    
    Demonstrates the standard library approach to memoisation.
    The @lru_cache decorator automatically handles caching with
    minimal code changes.
    
    Args:
        n: Non-negative integer index.
    
    Returns:
        The nth Fibonacci number.
    
    Complexity:
        Time: O(n).
        Space: O(n) for cache.
    
    Note:
        Call fibonacci_lru.cache_clear() to reset between tests.
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative index: {n}")
    
    if n <= 1:
        return n
    
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Compute nth Fibonacci iteratively with O(1) space.
    
    Demonstrates that any recursive solution can be converted to
    an iterative one. This version is optimal in both time and space.
    
    Args:
        n: Non-negative integer index.
    
    Returns:
        The nth Fibonacci number.
    
    Complexity:
        Time: O(n).
        Space: O(1).
    
    Example:
        >>> fibonacci_iterative(100)
        354224848179261915075
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative index: {n}")
    
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    
    return prev1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DIVIDE-AND-CONQUER ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def binary_search_recursive(
    arr: Sequence[T],
    target: T,
    low: int = 0,
    high: int | None = None
) -> int:
    """Find target in sorted sequence using recursive binary search.
    
    Demonstrates logarithmic recursion where each call halves the
    problem size, yielding O(log n) complexity.
    
    Args:
        arr: Sorted sequence to search.
        target: Value to find.
        low: Lower bound index (inclusive).
        high: Upper bound index (inclusive).
    
    Returns:
        Index of target if found, -1 otherwise.
    
    Complexity:
        Time: O(log n).
        Space: O(log n) for recursion stack.
    
    Example:
        >>> binary_search_recursive([1, 3, 5, 7, 9], 5)
        2
    """
    if high is None:
        high = len(arr) - 1
    
    # Base case: search space exhausted
    if low > high:
        return -1
    
    mid = (low + high) // 2
    
    # Base case: found
    if arr[mid] == target:
        return mid
    
    # Recursive cases: search appropriate half
    if arr[mid] > target:
        return binary_search_recursive(arr, target, low, mid - 1)
    else:
        return binary_search_recursive(arr, target, mid + 1, high)


def merge_sort(arr: list[T]) -> list[T]:
    """Sort list using merge sort algorithm.
    
    Demonstrates divide-and-conquer recursion where the problem is
    split into equal halves, solved recursively, then merged.
    
    Recurrence: T(n) = 2T(n/2) + O(n)
    By Master Theorem: T(n) = O(n log n)
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list (original unchanged).
    
    Complexity:
        Time: O(n log n) — guaranteed.
        Space: O(n) for temporary arrays.
    
    Example:
        >>> merge_sort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
    """
    # Base case: single element or empty
    if len(arr) <= 1:
        return arr.copy()
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer (merge)
    return _merge(left, right)


def _merge(left: list[T], right: list[T]) -> list[T]:
    """Merge two sorted lists into one sorted list.
    
    Args:
        left: First sorted list.
        right: Second sorted list.
    
    Returns:
        Merged sorted list.
    """
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


def quick_sort(arr: list[T]) -> list[T]:
    """Sort list using quick sort algorithm.
    
    Demonstrates divide-and-conquer with variable partition sizes.
    Uses the last element as pivot for simplicity.
    
    Args:
        arr: List to sort.
    
    Returns:
        New sorted list (original unchanged).
    
    Complexity:
        Time: O(n log n) average, O(n²) worst case.
        Space: O(log n) average for recursion stack.
    
    Example:
        >>> quick_sort([3, 1, 4, 1, 5, 9, 2, 6])
        [1, 1, 2, 3, 4, 5, 6, 9]
    """
    if len(arr) <= 1:
        return arr.copy()
    
    pivot = arr[-1]
    less = [x for x in arr[:-1] if x <= pivot]
    greater = [x for x in arr[:-1] if x > pivot]
    
    return quick_sort(less) + [pivot] + quick_sort(greater)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TREE TRAVERSAL ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TreeNode:
    """Binary tree node for traversal demonstrations.
    
    Attributes:
        value: The value stored in this node.
        left: Left child node (or None).
        right: Right child node (or None).
    """
    value: T
    left: TreeNode | None = None
    right: TreeNode | None = None


def preorder_traversal(root: TreeNode | None) -> list[Any]:
    """Visit nodes in preorder: root, left, right.
    
    Used for: Creating a copy of the tree, prefix expression evaluation.
    
    Args:
        root: Root node of the tree.
    
    Returns:
        List of node values in preorder.
    
    Complexity:
        Time: O(n) where n is number of nodes.
        Space: O(h) where h is tree height.
    
    Example:
        >>> tree = TreeNode(1, TreeNode(2), TreeNode(3))
        >>> preorder_traversal(tree)
        [1, 2, 3]
    """
    if root is None:
        return []
    
    result = [root.value]
    result.extend(preorder_traversal(root.left))
    result.extend(preorder_traversal(root.right))
    return result


def inorder_traversal(root: TreeNode | None) -> list[Any]:
    """Visit nodes in inorder: left, root, right.
    
    Used for: BST sorted output, infix expression evaluation.
    
    Args:
        root: Root node of the tree.
    
    Returns:
        List of node values in inorder.
    
    Complexity:
        Time: O(n).
        Space: O(h).
    """
    if root is None:
        return []
    
    result = inorder_traversal(root.left)
    result.append(root.value)
    result.extend(inorder_traversal(root.right))
    return result


def postorder_traversal(root: TreeNode | None) -> list[Any]:
    """Visit nodes in postorder: left, right, root.
    
    Used for: Deleting tree, postfix expression evaluation.
    
    Args:
        root: Root node of the tree.
    
    Returns:
        List of node values in postorder.
    
    Complexity:
        Time: O(n).
        Space: O(h).
    """
    if root is None:
        return []
    
    result = postorder_traversal(root.left)
    result.extend(postorder_traversal(root.right))
    result.append(root.value)
    return result


def tree_height(root: TreeNode | None) -> int:
    """Compute the height of a binary tree.
    
    Height is the number of edges on the longest path from root to leaf.
    
    Args:
        root: Root node of the tree.
    
    Returns:
        Tree height (-1 for empty tree, 0 for single node).
    
    Complexity:
        Time: O(n).
        Space: O(h).
    """
    if root is None:
        return -1
    
    left_height = tree_height(root.left)
    right_height = tree_height(root.right)
    
    return 1 + max(left_height, right_height)


def tree_size(root: TreeNode | None) -> int:
    """Count the number of nodes in a binary tree.
    
    Args:
        root: Root node of the tree.
    
    Returns:
        Number of nodes in the tree.
    
    Complexity:
        Time: O(n).
        Space: O(h).
    """
    if root is None:
        return 0
    
    return 1 + tree_size(root.left) + tree_size(root.right)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BACKTRACKING ALGORITHMS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_permutations(elements: list[T]) -> list[list[T]]:
    """Generate all permutations of a list.
    
    Uses backtracking to systematically explore all arrangements.
    
    Args:
        elements: List of elements to permute.
    
    Returns:
        List of all permutations.
    
    Complexity:
        Time: O(n! × n) — n! permutations, O(n) to copy each.
        Space: O(n! × n) for results, O(n) for recursion.
    
    Example:
        >>> generate_permutations([1, 2, 3])
        [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
    """
    results: list[list[T]] = []
    
    def backtrack(current: list[T], remaining: list[T]) -> None:
        if not remaining:
            results.append(current.copy())
            return
        
        for i, elem in enumerate(remaining):
            current.append(elem)
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()  # Backtrack
    
    backtrack([], elements)
    return results


def generate_subsets(elements: list[T]) -> list[list[T]]:
    """Generate all subsets (power set) of a list.
    
    Uses recursive backtracking with inclusion/exclusion choices.
    
    Args:
        elements: List of elements.
    
    Returns:
        List of all subsets (including empty set).
    
    Complexity:
        Time: O(2ⁿ × n) — 2ⁿ subsets, O(n) to copy each.
        Space: O(2ⁿ × n) for results.
    
    Example:
        >>> generate_subsets([1, 2])
        [[], [1], [1, 2], [2]]
    """
    results: list[list[T]] = []
    
    def backtrack(index: int, current: list[T]) -> None:
        if index == len(elements):
            results.append(current.copy())
            return
        
        # Exclude current element
        backtrack(index + 1, current)
        
        # Include current element
        current.append(elements[index])
        backtrack(index + 1, current)
        current.pop()  # Backtrack
    
    backtrack(0, [])
    return results


def solve_n_queens(n: int) -> list[list[int]]:
    """Solve the N-Queens problem using backtracking.
    
    Find all arrangements of n queens on an n×n chessboard such
    that no two queens threaten each other.
    
    Args:
        n: Board size and number of queens.
    
    Returns:
        List of solutions, each as a list of column positions by row.
    
    Complexity:
        Time: O(n!) in worst case.
        Space: O(n) for recursion and tracking sets.
    
    Example:
        >>> solve_n_queens(4)
        [[1, 3, 0, 2], [2, 0, 3, 1]]
    """
    solutions: list[list[int]] = []
    
    def is_safe(
        queens: list[int],
        row: int,
        col: int
    ) -> bool:
        """Check if placing queen at (row, col) is safe."""
        for prev_row, prev_col in enumerate(queens):
            # Same column
            if prev_col == col:
                return False
            # Same diagonal
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
                queens.pop()  # Backtrack
    
    backtrack([])
    return solutions


def subset_sum(
    numbers: list[int],
    target: int
) -> list[list[int]]:
    """Find all subsets that sum to target value.
    
    Uses backtracking with pruning for efficiency.
    
    Args:
        numbers: List of positive integers.
        target: Target sum to achieve.
    
    Returns:
        List of subsets that sum to target.
    
    Complexity:
        Time: O(2ⁿ) worst case.
        Space: O(n) for recursion.
    
    Example:
        >>> subset_sum([2, 3, 5, 7], 10)
        [[3, 7], [2, 3, 5]]
    """
    results: list[list[int]] = []
    numbers_sorted = sorted(numbers, reverse=True)  # Enables early termination
    
    def backtrack(
        index: int,
        current: list[int],
        remaining: int
    ) -> None:
        if remaining == 0:
            results.append(current.copy())
            return
        
        if remaining < 0 or index >= len(numbers_sorted):
            return
        
        # Pruning: if smallest remaining number exceeds target, stop
        if numbers_sorted[index] > remaining:
            # Try next (smaller) numbers
            backtrack(index + 1, current, remaining)
            return
        
        # Include current number
        current.append(numbers_sorted[index])
        backtrack(index + 1, current, remaining - numbers_sorted[index])
        current.pop()
        
        # Exclude current number
        backtrack(index + 1, current, remaining)
    
    backtrack(0, [], target)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COMPLEXITY ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CallCounter:
    """Tracks recursive call counts for complexity analysis.
    
    Attributes:
        calls: Total number of function invocations.
        max_depth: Maximum recursion depth reached.
        current_depth: Current recursion depth (internal).
    """
    calls: int = 0
    max_depth: int = 0
    current_depth: int = 0
    
    def enter(self) -> None:
        """Record entry into a recursive call."""
        self.calls += 1
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
    
    def exit(self) -> None:
        """Record exit from a recursive call."""
        self.current_depth -= 1
    
    def reset(self) -> None:
        """Reset all counters."""
        self.calls = 0
        self.max_depth = 0
        self.current_depth = 0


def fibonacci_counted(n: int, counter: CallCounter) -> int:
    """Fibonacci with call counting for analysis.
    
    Args:
        n: Fibonacci index.
        counter: CallCounter instance to track calls.
    
    Returns:
        nth Fibonacci number.
    """
    counter.enter()
    try:
        if n <= 1:
            return n
        return (
            fibonacci_counted(n - 1, counter) +
            fibonacci_counted(n - 2, counter)
        )
    finally:
        counter.exit()


def compare_fibonacci_implementations(n: int) -> dict[str, dict[str, Any]]:
    """Compare different Fibonacci implementations.
    
    Args:
        n: Fibonacci index to compute.
    
    Returns:
        Dictionary with timing and call statistics for each implementation.
    """
    results: dict[str, dict[str, Any]] = {}
    
    # Naive (only for small n)
    if n <= 30:
        counter = CallCounter()
        start = time.perf_counter_ns()
        value = fibonacci_counted(n, counter)
        elapsed = time.perf_counter_ns() - start
        results["naive"] = {
            "value": value,
            "calls": counter.calls,
            "max_depth": counter.max_depth,
            "time_ns": elapsed
        }
    
    # Memoised
    cache = MemoCache()
    start = time.perf_counter_ns()
    value = fibonacci_memoised(n, cache)
    elapsed = time.perf_counter_ns() - start
    results["memoised"] = {
        "value": value,
        "cache_stats": str(cache.stats),
        "time_ns": elapsed
    }
    
    # LRU cache
    fibonacci_lru.cache_clear()
    start = time.perf_counter_ns()
    value = fibonacci_lru(n)
    elapsed = time.perf_counter_ns() - start
    cache_info = fibonacci_lru.cache_info()
    results["lru_cache"] = {
        "value": value,
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "time_ns": elapsed
    }
    
    # Iterative
    start = time.perf_counter_ns()
    value = fibonacci_iterative(n)
    elapsed = time.perf_counter_ns() - start
    results["iterative"] = {
        "value": value,
        "time_ns": elapsed
    }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DEMONSTRATION AND CLI
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_memoisation() -> None:
    """Demonstrate the power of memoisation."""
    logger.info("=" * 70)
    logger.info("MEMOISATION DEMONSTRATION")
    logger.info("=" * 70)
    
    test_values = [10, 20, 30]
    
    for n in test_values:
        logger.info(f"\nFibonacci({n}) comparison:")
        results = compare_fibonacci_implementations(n)
        
        for impl_name, data in results.items():
            logger.info(f"  {impl_name}: {data}")


def demonstrate_divide_and_conquer() -> None:
    """Demonstrate divide-and-conquer algorithms."""
    logger.info("=" * 70)
    logger.info("DIVIDE-AND-CONQUER DEMONSTRATION")
    logger.info("=" * 70)
    
    test_array = [64, 34, 25, 12, 22, 11, 90]
    
    logger.info(f"\nOriginal array: {test_array}")
    logger.info(f"Merge sort result: {merge_sort(test_array)}")
    logger.info(f"Quick sort result: {quick_sort(test_array)}")
    
    sorted_array = [11, 12, 22, 25, 34, 64, 90]
    target = 25
    idx = binary_search_recursive(sorted_array, target)
    logger.info(f"Binary search for {target} in {sorted_array}: index {idx}")


def demonstrate_backtracking() -> None:
    """Demonstrate backtracking algorithms."""
    logger.info("=" * 70)
    logger.info("BACKTRACKING DEMONSTRATION")
    logger.info("=" * 70)
    
    # Permutations
    elements = [1, 2, 3]
    perms = generate_permutations(elements)
    logger.info(f"\nPermutations of {elements}: {len(perms)} total")
    for p in perms[:3]:
        logger.info(f"  {p}")
    logger.info("  ...")
    
    # N-Queens
    n = 4
    solutions = solve_n_queens(n)
    logger.info(f"\n{n}-Queens solutions: {len(solutions)} total")
    for sol in solutions:
        logger.info(f"  {sol}")
    
    # Subset sum
    numbers = [2, 3, 5, 7]
    target = 10
    subsets = subset_sum(numbers, target)
    logger.info(f"\nSubsets of {numbers} summing to {target}:")
    for s in subsets:
        logger.info(f"  {s}")


def run_demo() -> None:
    """Run all demonstrations."""
    demonstrate_memoisation()
    demonstrate_divide_and_conquer()
    demonstrate_backtracking()


def main() -> None:
    """Entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Lab 08.01: Recursive Patterns and Memoisation"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration of all algorithms"
    )
    parser.add_argument(
        "--fibonacci",
        type=int,
        metavar="N",
        help="Compute and compare Fibonacci implementations for N"
    )
    parser.add_argument(
        "--n-queens",
        type=int,
        metavar="N",
        help="Solve N-Queens problem"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        run_demo()
    elif args.fibonacci:
        results = compare_fibonacci_implementations(args.fibonacci)
        for impl, data in results.items():
            logger.info(f"{impl}: {data}")
    elif args.n_queens:
        solutions = solve_n_queens(args.n_queens)
        logger.info(f"Found {len(solutions)} solutions for {args.n_queens}-Queens")
        for sol in solutions:
            logger.info(f"  {sol}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
