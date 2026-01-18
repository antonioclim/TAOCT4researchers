#!/usr/bin/env python3
"""
Solution: Palindrome Partitioning (Hard)
"""

from functools import lru_cache


def min_cut(s: str) -> int:
    """
    Find minimum cuts needed for palindrome partitioning.
    
    Time: O(n²)
    Space: O(n²) for palindrome table + O(n) for DP
    
    Approach:
    1. Precompute palindrome table
    2. DP: dp[i] = minimum cuts for s[0:i+1]
    """
    if not s or len(s) <= 1:
        return 0
    
    n = len(s)
    
    # Precompute palindrome table
    is_pal = [[False] * n for _ in range(n)]
    
    # All single characters are palindromes
    for i in range(n):
        is_pal[i][i] = True
    
    # Check length 2
    for i in range(n - 1):
        is_pal[i][i + 1] = (s[i] == s[i + 1])
    
    # Check longer lengths
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]
    
    # DP: dp[i] = minimum cuts for s[0:i+1]
    dp = [0] * n
    
    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0  # Entire prefix is palindrome
        else:
            dp[i] = i  # Worst case: cut after each character
            for j in range(1, i + 1):
                if is_pal[j][i]:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
    
    return dp[n - 1]


def min_cut_optimised(s: str) -> int:
    """
    Space-optimised solution computing palindromes on-the-fly.
    
    Time: O(n²)
    Space: O(n)
    """
    if not s or len(s) <= 1:
        return 0
    
    n = len(s)
    dp = list(range(n))  # dp[i] = i initially (max cuts)
    
    # Expand around centres
    for centre in range(n):
        # Odd-length palindromes
        left, right = centre, centre
        while left >= 0 and right < n and s[left] == s[right]:
            if left == 0:
                dp[right] = 0
            else:
                dp[right] = min(dp[right], dp[left - 1] + 1)
            left -= 1
            right += 1
        
        # Even-length palindromes
        left, right = centre, centre + 1
        while left >= 0 and right < n and s[left] == s[right]:
            if left == 0:
                dp[right] = 0
            else:
                dp[right] = min(dp[right], dp[left - 1] + 1)
            left -= 1
            right += 1
    
    return dp[n - 1]


def partition(s: str) -> list[list[str]]:
    """
    Return all palindrome partitionings.
    
    Time: O(n × 2^n) worst case
    Space: O(n) for recursion + output
    """
    if not s:
        return [[]]
    
    n = len(s)
    
    # Precompute palindrome table
    is_pal = [[False] * n for _ in range(n)]
    for i in range(n):
        is_pal[i][i] = True
    for i in range(n - 1):
        is_pal[i][i + 1] = (s[i] == s[i + 1])
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_pal[i][j] = (s[i] == s[j]) and is_pal[i + 1][j - 1]
    
    result: list[list[str]] = []
    
    def backtrack(start: int, path: list[str]) -> None:
        if start == n:
            result.append(path.copy())
            return
        
        for end in range(start, n):
            if is_pal[start][end]:
                path.append(s[start:end + 1])
                backtrack(end + 1, path)
                path.pop()
    
    backtrack(0, [])
    return result


def partition_memoised(s: str) -> list[list[str]]:
    """
    Memoised version for all partitions.
    
    Returns list of all valid partitions.
    """
    n = len(s)
    
    @lru_cache(maxsize=None)
    def is_palindrome(i: int, j: int) -> bool:
        if i >= j:
            return True
        return s[i] == s[j] and is_palindrome(i + 1, j - 1)
    
    @lru_cache(maxsize=None)
    def dp(start: int) -> list[tuple[str, ...]]:
        if start == n:
            return [()]
        
        partitions = []
        for end in range(start, n):
            if is_palindrome(start, end):
                prefix = s[start:end + 1]
                for suffix in dp(end + 1):
                    partitions.append((prefix,) + suffix)
        
        return partitions
    
    return [list(p) for p in dp(0)]


if __name__ == "__main__":
    # Minimum cuts tests
    assert min_cut("a") == 0
    assert min_cut("ab") == 1
    assert min_cut("aab") == 1
    assert min_cut("aba") == 0
    assert min_cut("abcba") == 0
    
    # Optimised version
    assert min_cut_optimised("aab") == 1
    assert min_cut_optimised("aba") == 0
    
    # All partitions
    result = partition("aab")
    assert sorted([sorted(p) for p in result]) == sorted([["a", "a", "b"], ["aa", "b"]])
    
    assert partition("a") == [["a"]]
    
    # Memoised version
    result2 = partition_memoised("aab")
    assert len(result2) == 2
    
    print("All tests passed!")
