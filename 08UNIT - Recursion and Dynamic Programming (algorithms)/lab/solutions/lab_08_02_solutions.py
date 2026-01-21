#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTIONS: Lab 08.02 - Dynamic Programming
═══════════════════════════════════════════════════════════════════════════════

Reference implementations for all laboratory exercises.

WARNING: Students should attempt exercises before consulting solutions.

© 2026 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FIBONACCI DP VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════

def fibonacci_tabulated_solution(n: int) -> int:
    """
    Solution: Bottom-up Fibonacci.
    
    Time: O(n), Space: O(n)
    """
    if n < 0:
        raise ValueError(f"Fibonacci undefined for negative: {n}")
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def fibonacci_space_optimised_solution(n: int) -> int:
    """
    Solution: O(1) space Fibonacci.
    
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
# SECTION 2: KNAPSACK PROBLEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnapsackResult:
    """Knapsack solution result."""
    max_value: int
    selected_items: list[int]
    dp_table: list[list[int]] | None = None


def knapsack_tabulated_solution(
    weights: list[int],
    values: list[int],
    capacity: int
) -> KnapsackResult:
    """
    Solution: 0-1 Knapsack with item reconstruction.
    
    Time: O(nW), Space: O(nW)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(
                    dp[i - 1][w],
                    values[i - 1] + dp[i - 1][w - weights[i - 1]]
                )
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    
    return KnapsackResult(
        max_value=dp[n][capacity],
        selected_items=selected[::-1],
        dp_table=dp
    )


def knapsack_space_optimised_solution(
    weights: list[int],
    values: list[int],
    capacity: int
) -> int:
    """
    Solution: Space-optimised knapsack (no item reconstruction).
    
    Time: O(nW), Space: O(W)
    """
    n = len(weights)
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
    
    return dp[capacity]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LONGEST COMMON SUBSEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LCSResult:
    """LCS solution result."""
    length: int
    subsequence: str
    dp_table: list[list[int]] | None = None


def lcs_tabulated_solution(s1: str, s2: str) -> LCSResult:
    """
    Solution: LCS with subsequence reconstruction.
    
    Time: O(mn), Space: O(mn)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    # Backtrack to reconstruct LCS
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


def lcs_space_optimised_solution(s1: str, s2: str) -> int:
    """
    Solution: Space-optimised LCS (length only).
    
    Time: O(mn), Space: O(min(m,n))
    """
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
# SECTION 4: EDIT DISTANCE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EditDistanceResult:
    """Edit distance solution result."""
    distance: int
    operations: list[str]
    dp_table: list[list[int]] | None = None


def edit_distance_solution(s1: str, s2: str) -> EditDistanceResult:
    """
    Solution: Levenshtein distance with operation reconstruction.
    
    Time: O(mn), Space: O(mn)
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
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
    """Matrix chain multiplication result."""
    min_operations: int
    optimal_parenthesisation: str
    dp_table: list[list[int]] | None = None


def matrix_chain_solution(dimensions: list[int]) -> MatrixChainResult:
    """
    Solution: Matrix chain multiplication with optimal parenthesisation.
    
    Time: O(n³), Space: O(n²)
    """
    n = len(dimensions) - 1
    
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    
    # Fill for increasing chain lengths
    for length in range(2, n + 1):
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
    
    # Build parenthesisation string
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
# SECTION 6: COIN CHANGE PROBLEM
# ═══════════════════════════════════════════════════════════════════════════════

def coin_change_min_coins_solution(coins: list[int], amount: int) -> int:
    """
    Solution: Minimum coins needed to make amount.
    
    Time: O(amount × len(coins)), Space: O(amount)
    
    Returns -1 if impossible.
    """
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    return dp[amount] if dp[amount] != float("inf") else -1


def coin_change_count_ways_solution(coins: list[int], amount: int) -> int:
    """
    Solution: Count number of ways to make amount.
    
    Time: O(amount × len(coins)), Space: O(amount)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    
    return dp[amount]


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Verify solutions
    assert fibonacci_space_optimised_solution(50) == 12586269025
    
    ks = knapsack_tabulated_solution([2, 3, 4, 5], [3, 4, 5, 6], 8)
    assert ks.max_value == 10
    
    lcs = lcs_tabulated_solution("ABCDGH", "AEDFHR")
    assert lcs.subsequence == "ADH"
    
    ed = edit_distance_solution("kitten", "sitting")
    assert ed.distance == 3
    
    mc = matrix_chain_solution([10, 30, 5, 60])
    assert mc.min_operations == 4500
    
    assert coin_change_min_coins_solution([1, 2, 5], 11) == 3
    
    print("All DP solutions verified successfully.")
