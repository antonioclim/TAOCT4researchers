#!/usr/bin/env python3
"""
Solution: Regular Expression Matching (Hard)
"""


def is_match(s: str, p: str) -> bool:
    """
    Regular expression matching with '.' and '*'.
    
    Time: O(m × n) where m = len(s), n = len(p)
    Space: O(m × n)
    
    State: dp[i][j] = True if s[0:i] matches p[0:j]
    
    Transitions:
    1. p[j-1] is letter: dp[i][j] = dp[i-1][j-1] and s[i-1] == p[j-1]
    2. p[j-1] is '.': dp[i][j] = dp[i-1][j-1]
    3. p[j-1] is '*':
       - Zero occurrences: dp[i][j-2]
       - One+ occurrences: dp[i-1][j] if s[i-1] matches p[j-2]
    """
    m, n = len(s), len(p)
    
    # dp[i][j] = s[0:i] matches p[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    
    # Handle patterns like a*, a*b*, a*b*c* that can match empty string
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            dp[0][j] = dp[0][j - 2]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                # Zero occurrences of preceding element
                dp[i][j] = dp[i][j - 2]
                
                # One or more occurrences
                if _matches(s[i - 1], p[j - 2]):
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            else:
                # Direct character or '.' match
                if _matches(s[i - 1], p[j - 1]):
                    dp[i][j] = dp[i - 1][j - 1]
    
    return dp[m][n]


def _matches(char: str, pattern_char: str) -> bool:
    """Check if character matches pattern character."""
    return pattern_char == "." or char == pattern_char


def is_match_recursive(s: str, p: str) -> bool:
    """
    Recursive solution with memoisation.
    
    Time: O(m × n)
    Space: O(m × n)
    """
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> bool:
        # Base case: pattern exhausted
        if j == len(p):
            return i == len(s)
        
        # Check if first character matches
        first_match = i < len(s) and (p[j] == s[i] or p[j] == ".")
        
        # Handle '*' pattern
        if j + 1 < len(p) and p[j + 1] == "*":
            # Zero occurrences OR one+ occurrences
            return dp(i, j + 2) or (first_match and dp(i + 1, j))
        
        # No '*', must match and continue
        return first_match and dp(i + 1, j + 1)
    
    return dp(0, 0)


def is_match_space_optimised(s: str, p: str) -> bool:
    """
    Space-optimised DP solution.
    
    Time: O(m × n)
    Space: O(n)
    """
    m, n = len(s), len(p)
    
    prev = [False] * (n + 1)
    prev[0] = True
    
    for j in range(2, n + 1):
        if p[j - 1] == "*":
            prev[j] = prev[j - 2]
    
    for i in range(1, m + 1):
        curr = [False] * (n + 1)
        
        for j in range(1, n + 1):
            if p[j - 1] == "*":
                curr[j] = curr[j - 2]
                if _matches(s[i - 1], p[j - 2]):
                    curr[j] = curr[j] or prev[j]
            else:
                if _matches(s[i - 1], p[j - 1]):
                    curr[j] = prev[j - 1]
        
        prev = curr
    
    return prev[n]


if __name__ == "__main__":
    # Test cases
    assert is_match("aa", "a") == False
    assert is_match("aa", "a*") == True
    assert is_match("ab", ".*") == True
    assert is_match("aab", "c*a*b") == True
    assert is_match("mississippi", "mis*is*p*.") == False
    assert is_match("", ".*") == True
    assert is_match("", "") == True
    assert is_match("a", "") == False
    assert is_match("ab", ".*c") == False
    assert is_match("aaa", "a*a") == True
    
    # Test recursive version
    assert is_match_recursive("aa", "a*") == True
    assert is_match_recursive("aab", "c*a*b") == True
    
    print("All tests passed!")
