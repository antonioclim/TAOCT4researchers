# Practice Exercise: Unique Paths (Medium)

**Difficulty**: ★★★☆☆  
**Estimated Time**: 30 minutes  
**Topic**: Dynamic Programming (2D Grid)

---

## Problem Statement

A robot is located at the top-left corner of an m × n grid. The robot can only move either down or right at any point. How many unique paths are there to reach the bottom-right corner?

## Function Signature

```python
def unique_paths(m: int, n: int) -> int:
    """
    Count unique paths in m×n grid from top-left to bottom-right.
    
    Args:
        m: Number of rows
        n: Number of columns
    
    Returns:
        Number of unique paths
    
    Examples:
        >>> unique_paths(3, 7)
        28
        >>> unique_paths(3, 2)
        3
    """
    pass
```

## Requirements

1. Implement naive recursive solution
2. Implement memoised solution
3. Implement tabulated DP solution
4. Implement space-optimised solution using O(n) space

## Test Cases

```python
assert unique_paths(1, 1) == 1
assert unique_paths(1, 5) == 1
assert unique_paths(5, 1) == 1
assert unique_paths(3, 2) == 3
assert unique_paths(3, 7) == 28
assert unique_paths(7, 3) == 28
```

## Expected Complexity

- **Naive**: Time O(2^(m+n)), Space O(m+n)
- **Memoised/DP**: Time O(m×n), Space O(m×n)
- **Optimised**: Time O(m×n), Space O(n)

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**State**: `dp[i][j]` = number of paths to reach cell (i, j)

**Transition**: `dp[i][j] = dp[i-1][j] + dp[i][j-1]`

**Base case**: First row and first column all have value 1 (only one way to reach them)

**Space optimisation**: Each row only depends on the previous row

</details>
