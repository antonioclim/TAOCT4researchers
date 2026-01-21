#!/usr/bin/env python3
"""
Solution: Unique Paths (Medium)
"""

from functools import lru_cache


def unique_paths_naive(m: int, n: int) -> int:
    """
    Naive recursive solution.
    
    Time: O(2^(m+n)) - exponential
    Space: O(m+n) for recursion stack
    """
    if m == 1 or n == 1:
        return 1
    return unique_paths_naive(m - 1, n) + unique_paths_naive(m, n - 1)


def unique_paths_memoised(m: int, n: int) -> int:
    """
    Memoised recursive solution.
    
    Time: O(m × n)
    Space: O(m × n)
    """
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> int:
        if i == 1 or j == 1:
            return 1
        return dp(i - 1, j) + dp(i, j - 1)
    
    return dp(m, n)


def unique_paths_tabulated(m: int, n: int) -> int:
    """
    Bottom-up DP solution.
    
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
    Space-optimised DP solution.
    
    Time: O(m × n)
    Space: O(n)
    """
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]


def unique_paths_combinatorial(m: int, n: int) -> int:
    """
    Mathematical solution using combinatorics.
    
    Total moves = (m-1) + (n-1) = m + n - 2
    Need to choose (m-1) down moves from total
    Answer = C(m+n-2, m-1)
    
    Time: O(min(m, n))
    Space: O(1)
    """
    from math import comb
    return comb(m + n - 2, m - 1)


if __name__ == "__main__":
    # Test cases
    test_cases = [
        (1, 1, 1),
        (1, 5, 1),
        (5, 1, 1),
        (3, 2, 3),
        (3, 7, 28),
        (7, 3, 28),
    ]
    
    for m, n, expected in test_cases:
        assert unique_paths_memoised(m, n) == expected
        assert unique_paths_tabulated(m, n) == expected
        assert unique_paths_optimised(m, n) == expected
        assert unique_paths_combinatorial(m, n) == expected
    
    print("All tests passed!")
