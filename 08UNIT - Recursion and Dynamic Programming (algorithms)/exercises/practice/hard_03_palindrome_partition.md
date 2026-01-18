# Practice Exercise: Palindrome Partitioning (Hard)

**Difficulty**: ★★★★★  
**Estimated Time**: 45 minutes  
**Topic**: Dynamic Programming with Backtracking

---

## Problem Statement

Given a string s, partition it such that every substring of the partition is a palindrome.

Part A: Find the minimum number of cuts needed.
Part B: Return all possible palindrome partitionings.

## Function Signatures

```python
def min_cut(s: str) -> int:
    """
    Find minimum cuts needed for palindrome partitioning.
    
    Args:
        s: Input string
    
    Returns:
        Minimum number of cuts
    
    Examples:
        >>> min_cut("aab")
        1  # ["aa", "b"]
        >>> min_cut("ab")
        1  # ["a", "b"]
        >>> min_cut("aba")
        0  # ["aba"] - already palindrome
    """
    pass


def partition(s: str) -> list[list[str]]:
    """
    Return all palindrome partitionings of string.
    
    Args:
        s: Input string
    
    Returns:
        List of all valid partitions
    
    Examples:
        >>> partition("aab")
        [["a", "a", "b"], ["aa", "b"]]
        >>> partition("a")
        [["a"]]
    """
    pass
```

## Test Cases

```python
# Minimum cuts
assert min_cut("a") == 0
assert min_cut("ab") == 1
assert min_cut("aab") == 1
assert min_cut("aba") == 0
assert min_cut("abcba") == 0
assert min_cut("aaabaa") == 1  # "aaa" | "baa" or "aa" | "aba" | "a"

# All partitions
assert partition("a") == [["a"]]
assert sorted(partition("aab")) == sorted([["a", "a", "b"], ["aa", "b"]])
```

## Expected Complexity

**Minimum cuts:**
- Time: O(n²)
- Space: O(n²) for palindrome precomputation, O(n) for DP

**All partitions:**
- Time: O(n × 2^n) worst case
- Space: O(n) for recursion + output

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**Minimum cuts (Part A):**

1. Precompute palindrome table: `is_pal[i][j]` = True if s[i:j+1] is palindrome
2. DP: `dp[i]` = minimum cuts for s[0:i+1]
3. Transition: `dp[i] = min(dp[j-1] + 1)` for all j where s[j:i+1] is palindrome

**All partitions (Part B):**

Use backtracking:
1. Try each prefix that is a palindrome
2. Recursively partition the rest
3. Collect all valid complete partitions

```python
def backtrack(start, path, result):
    if start == len(s):
        result.append(path.copy())
        return
    for end in range(start + 1, len(s) + 1):
        if is_palindrome(s[start:end]):
            path.append(s[start:end])
            backtrack(end, path, result)
            path.pop()
```

</details>
