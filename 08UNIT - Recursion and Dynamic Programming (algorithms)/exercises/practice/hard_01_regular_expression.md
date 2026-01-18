# Practice Exercise: Regular Expression Matching (Hard)

**Difficulty**: ★★★★★  
**Estimated Time**: 45 minutes  
**Topic**: Dynamic Programming (2D String Matching)

---

## Problem Statement

Implement regular expression matching with support for `.` and `*`:

- `.` matches any single character
- `*` matches zero or more of the preceding element

The matching should cover the entire input string (not partial).

## Function Signature

```python
def is_match(s: str, p: str) -> bool:
    """
    Determine if string s matches pattern p.
    
    Args:
        s: Input string (lowercase letters only)
        p: Pattern with '.', '*', and lowercase letters
    
    Returns:
        True if s matches p entirely, False otherwise
    
    Examples:
        >>> is_match("aa", "a")
        False
        >>> is_match("aa", "a*")
        True
        >>> is_match("ab", ".*")
        True
        >>> is_match("aab", "c*a*b")
        True  # c* matches empty, a* matches "aa"
    """
    pass
```

## Test Cases

```python
assert is_match("aa", "a") == False
assert is_match("aa", "a*") == True
assert is_match("ab", ".*") == True
assert is_match("aab", "c*a*b") == True
assert is_match("mississippi", "mis*is*p*.") == False
assert is_match("", ".*") == True
assert is_match("", "") == True
assert is_match("a", "") == False
```

## Expected Complexity

- **Time**: O(m × n) where m = len(s), n = len(p)
- **Space**: O(m × n), can be optimised to O(n)

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**State**: `dp[i][j]` = True if s[0:i] matches p[0:j]

**Transition cases:**

1. If `p[j-1]` is a letter: match if `s[i-1] == p[j-1]` and `dp[i-1][j-1]`

2. If `p[j-1]` is `.`: match if `dp[i-1][j-1]` (. matches any char)

3. If `p[j-1]` is `*`:
   - Zero occurrences: `dp[i][j-2]` (skip pattern x*)
   - One+ occurrences: `dp[i-1][j]` if `s[i-1]` matches `p[j-2]`

**Base case**: `dp[0][0] = True`  
**Edge case**: `dp[0][j]` can be True if pattern is all `x*` form

</details>
