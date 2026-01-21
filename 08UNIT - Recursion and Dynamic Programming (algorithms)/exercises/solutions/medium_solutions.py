#!/usr/bin/env python3
"""
Solutions for Medium Practice Exercises
=======================================

Unit 8: Recursion and Dynamic Programming
"""

from __future__ import annotations

from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM 01: Unique Paths
# ═══════════════════════════════════════════════════════════════════════════════

def unique_paths_naive(m: int, n: int) -> int:
    """
    Count unique paths using naive recursion.
    
    Time: O(2^(m+n))
    Space: O(m+n)
    """
    # Base case: reached destination or edge
    if m == 1 or n == 1:
        return 1
    
    # Recursive case: paths from above + paths from left
    return unique_paths_naive(m - 1, n) + unique_paths_naive(m, n - 1)


def unique_paths_memoised(m: int, n: int) -> int:
    """
    Count unique paths with memoisation.
    
    Time: O(m × n)
    Space: O(m × n)
    """
    memo: dict[tuple[int, int], int] = {}
    
    def helper(i: int, j: int) -> int:
        if i == 1 or j == 1:
            return 1
        
        if (i, j) in memo:
            return memo[(i, j)]
        
        memo[(i, j)] = helper(i - 1, j) + helper(i, j - 1)
        return memo[(i, j)]
    
    return helper(m, n)


def unique_paths_tabulated(m: int, n: int) -> int:
    """
    Count unique paths using tabulation.
    
    Time: O(m × n)
    Space: O(m × n)
    """
    dp = [[1] * n for _ in range(m)]
    
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    
    return dp[m - 1][n - 1]


def unique_paths_optimised(m: int, n: int) -> int:
    """
    Count unique paths with O(n) space.
    
    Time: O(m × n)
    Space: O(n)
    """
    dp = [1] * n
    
    for _ in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM 02: Coin Change
# ═══════════════════════════════════════════════════════════════════════════════

def coin_change_min(coins: list[int], amount: int) -> int:
    """
    Find minimum coins needed to make amount.
    
    Time: O(amount × len(coins))
    Space: O(amount)
    
    Returns -1 if impossible.
    """
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)
    
    return dp[amount] if dp[amount] != float("inf") else -1


def coin_change_ways(coins: list[int], amount: int) -> int:
    """
    Count number of ways to make amount.
    
    Time: O(amount × len(coins))
    Space: O(amount)
    
    Note: Coins in outer loop to count combinations, not permutations.
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    
    return dp[amount]


def coin_change_with_coins(coins: list[int], amount: int) -> list[int] | None:
    """
    Return actual coins used (one valid solution).
    
    Time: O(amount × len(coins))
    Space: O(amount)
    """
    dp = [float("inf")] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] + 1 < dp[x]:
                dp[x] = dp[x - coin] + 1
                parent[x] = coin
    
    if dp[amount] == float("inf"):
        return None
    
    # Reconstruct solution
    result = []
    current = amount
    while current > 0:
        result.append(parent[current])
        current -= parent[current]
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIUM 03: Longest Increasing Subsequence
# ═══════════════════════════════════════════════════════════════════════════════

def lis_dp(nums: list[int]) -> int:
    """
    Find length of longest increasing subsequence using O(n²) DP.
    
    Time: O(n²)
    Space: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = LIS ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def lis_binary_search(nums: list[int]) -> int:
    """
    Find length of LIS using binary search.
    
    Time: O(n log n)
    Space: O(n)
    
    tails[i] = smallest tail of all increasing subsequences of length i+1
    """
    if not nums:
        return 0
    
    tails: list[int] = []
    
    for num in nums:
        # Binary search for insertion position
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid
        
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num
    
    return len(tails)


def lis_with_sequence(nums: list[int]) -> list[int]:
    """
    Return one of the longest increasing subsequences.
    
    Time: O(n²)
    Space: O(n)
    """
    if not nums:
        return []
    
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find index of maximum
    max_idx = max(range(n), key=lambda i: dp[i])
    
    # Reconstruct sequence
    result = []
    idx = max_idx
    while idx != -1:
        result.append(nums[idx])
        idx = parent[idx]
    
    return result[::-1]


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test unique_paths
    assert unique_paths_naive(3, 2) == 3
    assert unique_paths_memoised(3, 7) == 28
    assert unique_paths_tabulated(3, 7) == 28
    assert unique_paths_optimised(3, 7) == 28
    print("✓ unique_paths tests passed")
    
    # Test coin_change
    assert coin_change_min([1, 2, 5], 11) == 3
    assert coin_change_min([2], 3) == -1
    assert coin_change_ways([1, 2, 5], 5) == 4
    print("✓ coin_change tests passed")
    
    # Test LIS
    assert lis_dp([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis_binary_search([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    seq = lis_with_sequence([10, 9, 2, 5, 3, 7, 101, 18])
    assert len(seq) == 4
    print("✓ LIS tests passed")
    
    print("\nAll medium solutions verified!")
