#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Unit 8, Lab 2: Dynamic Programming
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Dynamic programming provides a systematic way to build solutions from the
bottom up, eliminating redundancy by solving each subproblem exactly once.
Richard Bellman coined the term in the 1950s whilst working at RAND Corporation,
deliberately choosing nomenclature that would appeal to military sponsors whilst
obscuring the theoretical nature of his research (Bellman, 1984).

The technique derives its power from two key properties: optimal substructure
(optimal solutions contain optimal subsolutions) and overlapping subproblems
(the same subproblems recur during computation). When both properties are
present, dynamic programming transforms exponential-time recursive solutions
into polynomial-time algorithms.

PREREQUISITES
─────────────
- Lab 08.01: Recursive Patterns and Memoisation
- Familiarity with 2D arrays and state transitions

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Identify optimal substructure in optimisation problems
2. Formulate state transitions for DP tables
3. Implement bottom-up tabulation with proper initialisation
4. Apply space optimisation techniques

ESTIMATED TIME
──────────────
- Reading: 45 minutes
- Coding: 135 minutes
- Total: 180 minutes

DEPENDENCIES
────────────
Python 3.12+

LICENCE
───────
© 2026 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FIBONACCI — FROM RECURSION TO DP
# ═══════════════════════════════════════════════════════════════════════════════

def fibonacci_tabulated(n: int) -> int:
    """Compute nth Fibonacci using bottom-up dynamic programming.
    
    Builds the solution iteratively from base cases, storing all
    intermediate results in a table.
    
    Args:
        n: Non-negative integer index.
    
    Returns:
        The nth Fibonacci number.
    
    Complexity:
        Time: O(n).
        Space: O(n) for the table.
    
    Example:
        >>> fibonacci_tabulated(10)
        55
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative index: {n}")
    
    if n <= 1:
        return n
    
    # Initialise DP table
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    # Fill table bottom-up
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def fibonacci_space_optimised(n: int) -> int:
    """Compute nth Fibonacci with O(1) space.
    
    Recognises that each Fibonacci number depends only on the previous
    two, so we need not store the entire table.
    
    Args:
        n: Non-negative integer index.
    
    Returns:
        The nth Fibonacci number.
    
    Complexity:
        Time: O(n).
        Space: O(1).
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
# SECTION 2: 0-1 KNAPSACK PROBLEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnapsackResult:
    """Result of knapsack optimisation.
    
    Attributes:
        max_value: Maximum achievable value.
        selected_items: Indices of items to include.
        dp_table: The complete DP table (for analysis).
    """
    max_value: int
    selected_items: list[int]
    dp_table: list[list[int]] | None = None


def knapsack_recursive(
    weights: list[int],
    values: list[int],
    capacity: int,
    n: int | None = None
) -> int:
    """Solve 0-1 Knapsack using naive recursion.
    
    For each item, we have two choices: include it or exclude it.
    This leads to exponential time complexity.
    
    Args:
        weights: List of item weights.
        values: List of item values.
        capacity: Maximum weight capacity.
        n: Number of items to consider (defaults to all).
    
    Returns:
        Maximum achievable value.
    
    Complexity:
        Time: O(2ⁿ).
        Space: O(n) for recursion stack.
    
    Warning:
        Impractical for n > 25 due to exponential growth.
    """
    if n is None:
        n = len(weights)
    
    # Base case: no items or no capacity
    if n == 0 or capacity == 0:
        return 0
    
    # If current item exceeds capacity, skip it
    if weights[n - 1] > capacity:
        return knapsack_recursive(weights, values, capacity, n - 1)
    
    # Return maximum of including or excluding current item
    include = values[n - 1] + knapsack_recursive(
        weights, values, capacity - weights[n - 1], n - 1
    )
    exclude = knapsack_recursive(weights, values, capacity, n - 1)
    
    return max(include, exclude)


def knapsack_memoised(
    weights: list[int],
    values: list[int],
    capacity: int,
    n: int | None = None,
    memo: dict[tuple[int, int], int] | None = None
) -> int:
    """Solve 0-1 Knapsack with memoisation.
    
    Top-down approach that caches results to avoid redundant computation.
    
    Args:
        weights: List of item weights.
        values: List of item values.
        capacity: Maximum weight capacity.
        n: Number of items to consider.
        memo: Memoisation dictionary.
    
    Returns:
        Maximum achievable value.
    
    Complexity:
        Time: O(n × W) where W is capacity.
        Space: O(n × W) for cache.
    """
    if n is None:
        n = len(weights)
    if memo is None:
        memo = {}
    
    # Check cache
    if (n, capacity) in memo:
        return memo[(n, capacity)]
    
    # Base case
    if n == 0 or capacity == 0:
        return 0
    
    # Compute result
    if weights[n - 1] > capacity:
        result = knapsack_memoised(weights, values, capacity, n - 1, memo)
    else:
        include = values[n - 1] + knapsack_memoised(
            weights, values, capacity - weights[n - 1], n - 1, memo
        )
        exclude = knapsack_memoised(weights, values, capacity, n - 1, memo)
        result = max(include, exclude)
    
    memo[(n, capacity)] = result
    return result


def knapsack_tabulated(
    weights: list[int],
    values: list[int],
    capacity: int
) -> KnapsackResult:
    """Solve 0-1 Knapsack using bottom-up dynamic programming.
    
    Builds a 2D table where dp[i][w] represents the maximum value
    achievable using items 0..i-1 with capacity w.
    
    State transition:
        dp[i][w] = max(dp[i-1][w], values[i-1] + dp[i-1][w - weights[i-1]])
                   if weights[i-1] <= w, else dp[i-1][w]
    
    Args:
        weights: List of item weights.
        values: List of item values.
        capacity: Maximum weight capacity.
    
    Returns:
        KnapsackResult with max value, selected items and DP table.
    
    Complexity:
        Time: O(n × W).
        Space: O(n × W) for the table.
    
    Example:
        >>> result = knapsack_tabulated([2, 3, 4, 5], [3, 4, 5, 6], 8)
        >>> result.max_value
        10
    """
    n = len(weights)
    
    # Initialise DP table with zeros
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill the table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                # Cannot include this item
                dp[i][w] = dp[i - 1][w]
            else:
                # Maximum of including or excluding
                include = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                exclude = dp[i - 1][w]
                dp[i][w] = max(include, exclude)
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)  # 0-indexed
            w -= weights[i - 1]
    
    return KnapsackResult(
        max_value=dp[n][capacity],
        selected_items=selected[::-1],  # Reverse to get original order
        dp_table=dp
    )


def knapsack_space_optimised(
    weights: list[int],
    values: list[int],
    capacity: int
) -> int:
    """Solve 0-1 Knapsack with O(W) space.
    
    Since each row depends only on the previous row, we can use a
    single array. We iterate from right to left to avoid overwriting
    values we still need.
    
    Args:
        weights: List of item weights.
        values: List of item values.
        capacity: Maximum weight capacity.
    
    Returns:
        Maximum achievable value.
    
    Complexity:
        Time: O(n × W).
        Space: O(W).
    
    Note:
        This optimisation precludes item reconstruction.
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Process from right to left
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
    
    return dp[capacity]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LONGEST COMMON SUBSEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LCSResult:
    """Result of LCS computation.
    
    Attributes:
        length: Length of longest common subsequence.
        subsequence: The actual LCS string.
        dp_table: The complete DP table (for analysis).
    """
    length: int
    subsequence: str
    dp_table: list[list[int]] | None = None


def lcs_tabulated(s1: str, s2: str) -> LCSResult:
    """Find longest common subsequence using dynamic programming.
    
    A subsequence is a sequence derived by deleting some or no elements
    without changing the order of remaining elements.
    
    State transition:
        dp[i][j] = dp[i-1][j-1] + 1           if s1[i-1] == s2[j-1]
                 = max(dp[i-1][j], dp[i][j-1]) otherwise
    
    Args:
        s1: First string.
        s2: Second string.
    
    Returns:
        LCSResult with length, subsequence and DP table.
    
    Complexity:
        Time: O(m × n) where m, n are string lengths.
        Space: O(m × n) for the table.
    
    Example:
        >>> result = lcs_tabulated("ABCDGH", "AEDFHR")
        >>> result.subsequence
        'ADH'
    """
    m, n = len(s1), len(s2)
    
    # Initialise DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to reconstruct the subsequence
    subsequence = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            subsequence.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    
    return LCSResult(
        length=dp[m][n],
        subsequence="".join(reversed(subsequence)),
        dp_table=dp
    )


def lcs_space_optimised(s1: str, s2: str) -> int:
    """Find LCS length with O(min(m,n)) space.
    
    Uses two rows since each cell depends only on the previous row.
    
    Args:
        s1: First string.
        s2: Second string.
    
    Returns:
        Length of longest common subsequence.
    
    Complexity:
        Time: O(m × n).
        Space: O(min(m, n)).
    """
    # Ensure s2 is shorter for space efficiency
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    m, n = len(s1), len(s2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    
    return prev[n]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EDIT DISTANCE (LEVENSHTEIN)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EditDistanceResult:
    """Result of edit distance computation.
    
    Attributes:
        distance: Minimum number of operations.
        operations: Sequence of operations to transform s1 to s2.
        dp_table: The complete DP table.
    """
    distance: int
    operations: list[str]
    dp_table: list[list[int]] | None = None


def edit_distance(s1: str, s2: str) -> EditDistanceResult:
    """Compute minimum edit distance (Levenshtein distance).
    
    The edit distance is the minimum number of single-character edits
    (insertions, deletions, substitutions) to transform s1 into s2.
    
    State transition:
        dp[i][j] = dp[i-1][j-1]                 if s1[i-1] == s2[j-1]
                 = 1 + min(dp[i-1][j],          (delete from s1)
                          dp[i][j-1],           (insert into s1)
                          dp[i-1][j-1])         (substitute)
    
    Args:
        s1: Source string.
        s2: Target string.
    
    Returns:
        EditDistanceResult with distance, operations and DP table.
    
    Complexity:
        Time: O(m × n).
        Space: O(m × n).
    
    Example:
        >>> result = edit_distance("kitten", "sitting")
        >>> result.distance
        3
    """
    m, n = len(s1), len(s2)
    
    # Initialise DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases: transforming to/from empty string
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters
    
    # Fill the table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Substitute
                )
    
    # Backtrack to reconstruct operations
    operations = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            operations.append(f"KEEP '{s1[i - 1]}'")
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            operations.append(f"SUBSTITUTE '{s1[i - 1]}' -> '{s2[j - 1]}'")
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            operations.append(f"INSERT '{s2[j - 1]}'")
            j -= 1
        else:
            operations.append(f"DELETE '{s1[i - 1]}'")
            i -= 1
    
    return EditDistanceResult(
        distance=dp[m][n],
        operations=operations[::-1],
        dp_table=dp
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: MATRIX CHAIN MULTIPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatrixChainResult:
    """Result of matrix chain multiplication optimisation.
    
    Attributes:
        min_operations: Minimum scalar multiplications required.
        optimal_parenthesisation: String showing optimal grouping.
        dp_table: The complete DP table.
    """
    min_operations: int
    optimal_parenthesisation: str
    dp_table: list[list[int]] | None = None


def matrix_chain_order(dimensions: list[int]) -> MatrixChainResult:
    """Find optimal parenthesisation for matrix chain multiplication.
    
    Given matrices A₁, A₂, ..., Aₙ where Aᵢ has dimensions
    dimensions[i-1] × dimensions[i], find the order of multiplication
    that minimises total scalar multiplications.
    
    State transition:
        dp[i][j] = min(dp[i][k] + dp[k+1][j] + d[i-1]×d[k]×d[j])
                   for k in range(i, j)
    
    Args:
        dimensions: List where dimensions[i-1] × dimensions[i]
                   gives the dimensions of matrix i.
    
    Returns:
        MatrixChainResult with min operations, parenthesisation and table.
    
    Complexity:
        Time: O(n³).
        Space: O(n²).
    
    Example:
        >>> result = matrix_chain_order([10, 30, 5, 60])
        >>> result.min_operations
        4500
    """
    n = len(dimensions) - 1  # Number of matrices
    
    # dp[i][j] = minimum cost to multiply matrices i through j
    dp = [[0] * n for _ in range(n)]
    
    # split[i][j] = optimal split point for matrices i through j
    split = [[0] * n for _ in range(n)]
    
    # Fill table for increasing chain lengths
    for length in range(2, n + 1):  # length of chain
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float("inf")
            
            for k in range(i, j):
                cost = (
                    dp[i][k] +
                    dp[k + 1][j] +
                    dimensions[i] * dimensions[k + 1] * dimensions[j + 1]
                )
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k
    
    # Reconstruct optimal parenthesisation
    def build_parens(i: int, j: int) -> str:
        if i == j:
            return f"A{i + 1}"
        k = split[i][j]
        left = build_parens(i, k)
        right = build_parens(k + 1, j)
        return f"({left} × {right})"
    
    return MatrixChainResult(
        min_operations=dp[0][n - 1],
        optimal_parenthesisation=build_parens(0, n - 1),
        dp_table=dp
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATION AND CLI
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_knapsack() -> None:
    """Demonstrate the knapsack problem."""
    logger.info("=" * 70)
    logger.info("KNAPSACK PROBLEM DEMONSTRATION")
    logger.info("=" * 70)
    
    weights = [2, 3, 4, 5]
    values = [3, 4, 5, 6]
    capacity = 8
    
    logger.info(f"\nItems: weights={weights}, values={values}")
    logger.info(f"Capacity: {capacity}")
    
    result = knapsack_tabulated(weights, values, capacity)
    logger.info(f"Maximum value: {result.max_value}")
    logger.info(f"Selected items (indices): {result.selected_items}")
    
    # Space-optimised
    max_val = knapsack_space_optimised(weights, values, capacity)
    logger.info(f"Space-optimised result: {max_val}")


def demonstrate_lcs() -> None:
    """Demonstrate longest common subsequence."""
    logger.info("=" * 70)
    logger.info("LONGEST COMMON SUBSEQUENCE DEMONSTRATION")
    logger.info("=" * 70)
    
    s1, s2 = "ABCDGH", "AEDFHR"
    
    logger.info(f"\nStrings: '{s1}' and '{s2}'")
    
    result = lcs_tabulated(s1, s2)
    logger.info(f"LCS length: {result.length}")
    logger.info(f"LCS: '{result.subsequence}'")


def demonstrate_edit_distance() -> None:
    """Demonstrate edit distance computation."""
    logger.info("=" * 70)
    logger.info("EDIT DISTANCE DEMONSTRATION")
    logger.info("=" * 70)
    
    s1, s2 = "kitten", "sitting"
    
    logger.info(f"\nTransform '{s1}' to '{s2}'")
    
    result = edit_distance(s1, s2)
    logger.info(f"Edit distance: {result.distance}")
    logger.info("Operations:")
    for op in result.operations:
        logger.info(f"  {op}")


def demonstrate_matrix_chain() -> None:
    """Demonstrate matrix chain multiplication."""
    logger.info("=" * 70)
    logger.info("MATRIX CHAIN MULTIPLICATION DEMONSTRATION")
    logger.info("=" * 70)
    
    # Matrices: A₁(10×30), A₂(30×5), A₃(5×60)
    dimensions = [10, 30, 5, 60]
    
    logger.info(f"\nMatrix dimensions: {dimensions}")
    logger.info("A₁: 10×30, A₂: 30×5, A₃: 5×60")
    
    result = matrix_chain_order(dimensions)
    logger.info(f"Minimum scalar multiplications: {result.min_operations}")
    logger.info(f"Optimal parenthesisation: {result.optimal_parenthesisation}")


def run_demo() -> None:
    """Run all demonstrations."""
    demonstrate_knapsack()
    demonstrate_lcs()
    demonstrate_edit_distance()
    demonstrate_matrix_chain()


def main() -> None:
    """Entry point for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Lab 08.02: Dynamic Programming"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration of all algorithms"
    )
    parser.add_argument(
        "--knapsack",
        action="store_true",
        help="Run knapsack demonstration"
    )
    parser.add_argument(
        "--lcs",
        nargs=2,
        metavar=("S1", "S2"),
        help="Compute LCS of two strings"
    )
    parser.add_argument(
        "--edit-distance",
        nargs=2,
        metavar=("S1", "S2"),
        help="Compute edit distance between two strings"
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
    elif args.knapsack:
        demonstrate_knapsack()
    elif args.lcs:
        result = lcs_tabulated(args.lcs[0], args.lcs[1])
        logger.info(f"LCS: '{result.subsequence}' (length: {result.length})")
    elif args.edit_distance:
        result = edit_distance(args.edit_distance[0], args.edit_distance[1])
        logger.info(f"Edit distance: {result.distance}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
