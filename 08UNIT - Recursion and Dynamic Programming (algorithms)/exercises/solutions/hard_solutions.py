#!/usr/bin/env python3
"""
Solutions for Hard Practice Exercises
=====================================

Unit 8: Recursion and Dynamic Programming
"""

from __future__ import annotations

from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════════════
# HARD 01: Regular Expression Matching
# ═══════════════════════════════════════════════════════════════════════════════

def is_match(s: str, p: str) -> bool:
    """
    Regular expression matching with '.' and '*'.
    
    Time: O(m × n)
    Space: O(m × n)
    
    State: dp[i][j] = True if s[0:i] matches p[0:j]
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    
    # Base case: empty string matches empty pattern
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, etc. that can match empty string
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                # Zero occurrences of preceding element
                dp[i][j] = dp[i][j - 2]
                
                # One or more occurrences
                if p[j - 2] == "." or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            
            elif p[j - 1] == "." or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]


def is_match_recursive(s: str, p: str) -> bool:
    """Recursive solution with memoisation."""
    
    @lru_cache(maxsize=None)
    def helper(i: int, j: int) -> bool:
        # Base case: pattern exhausted
        if j == len(p):
            return i == len(s)
        
        # Check if current characters match
        first_match = i < len(s) and (p[j] == "." or p[j] == s[i])
        
        # Handle '*' pattern
        if j + 1 < len(p) and p[j + 1] == "*":
            # Skip x* (zero occurrences) OR match one and continue
            return helper(i, j + 2) or (first_match and helper(i + 1, j))
        
        # Normal character match
        return first_match and helper(i + 1, j + 1)
    
    return helper(0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# HARD 02: Burst Balloons
# ═══════════════════════════════════════════════════════════════════════════════

def max_coins(nums: list[int]) -> int:
    """
    Maximum coins from bursting all balloons.
    
    Time: O(n³)
    Space: O(n²)
    
    Key insight: Think about which balloon to burst LAST in a range.
    """
    if not nums:
        return 0
    
    # Add virtual balloons with value 1 at both ends
    nums = [1] + nums + [1]
    n = len(nums)
    
    # dp[i][j] = max coins from bursting all balloons in range (i, j) exclusive
    dp = [[0] * n for _ in range(n)]
    
    # Fill for increasing range lengths
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            
            # Try each balloon k as the LAST to burst in range (i, j)
            for k in range(i + 1, j):
                # When k is last, only nums[i] and nums[j] are neighbours
                coins = nums[i] * nums[k] * nums[j]
                coins += dp[i][k] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)
    
    return dp[0][n - 1]


def max_coins_recursive(nums: list[int]) -> int:
    """Recursive solution with memoisation."""
    if not nums:
        return 0
    
    nums = [1] + nums + [1]
    n = len(nums)
    
    @lru_cache(maxsize=None)
    def helper(left: int, right: int) -> int:
        if left + 1 == right:
            return 0
        
        result = 0
        for k in range(left + 1, right):
            coins = nums[left] * nums[k] * nums[right]
            coins += helper(left, k) + helper(k, right)
            result = max(result, coins)
        
        return result
    
    return helper(0, n - 1)


# ═══════════════════════════════════════════════════════════════════════════════
# HARD 03: Palindrome Partitioning
# ═══════════════════════════════════════════════════════════════════════════════

def min_cut(s: str) -> int:
    """
    Minimum cuts needed for palindrome partitioning.
    
    Time: O(n²)
    Space: O(n²)
    """
    n = len(s)
    if n <= 1:
        return 0
    
    # Precompute palindrome table
    is_pal = [[False] * n for _ in range(n)]
    
    for i in range(n):
        is_pal[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = s[i] == s[j]
            else:
                is_pal[i][j] = s[i] == s[j] and is_pal[i + 1][j - 1]
    
    # dp[i] = minimum cuts for s[0:i+1]
    dp = list(range(n))  # Worst case: cut between every character
    
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0
        else:
            for j in range(1, i + 1):
                if is_pal[j][i]:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
    
    return dp[n - 1]


def partition(s: str) -> list[list[str]]:
    """
    Return all palindrome partitionings.
    
    Time: O(n × 2^n) worst case
    Space: O(n) for recursion
    """
    n = len(s)
    results: list[list[str]] = []
    
    # Precompute palindrome table
    is_pal = [[False] * n for _ in range(n)]
    
    for i in range(n):
        is_pal[i][i] = True
    
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if length == 2:
                is_pal[i][j] = s[i] == s[j]
            else:
                is_pal[i][j] = s[i] == s[j] and is_pal[i + 1][j - 1]
    
    def backtrack(start: int, path: list[str]) -> None:
        if start == n:
            results.append(path.copy())
            return
        
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end + 1])
                backtrack(end + 1, path)
                path.pop()
    
    backtrack(0, [])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test regex matching
    assert is_match("aa", "a") is False
    assert is_match("aa", "a*") is True
    assert is_match("ab", ".*") is True
    assert is_match("aab", "c*a*b") is True
    assert is_match_recursive("mississippi", "mis*is*p*.") is False
    print("✓ regex matching tests passed")
    
    # Test burst balloons
    assert max_coins([3, 1, 5, 8]) == 167
    assert max_coins([1, 5]) == 10
    assert max_coins_recursive([3, 1, 5, 8]) == 167
    print("✓ burst balloons tests passed")
    
    # Test palindrome partitioning
    assert min_cut("aab") == 1
    assert min_cut("aba") == 0
    parts = partition("aab")
    assert ["a", "a", "b"] in parts
    assert ["aa", "b"] in parts
    print("✓ palindrome partitioning tests passed")
    
    print("\nAll hard solutions verified!")
