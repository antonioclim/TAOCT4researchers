# Practice Exercise: Burst Balloons (Hard)

**Difficulty**: ★★★★★  
**Estimated Time**: 50 minutes  
**Topic**: Interval Dynamic Programming

---

## Problem Statement

You have n balloons, indexed 0 to n-1. Each balloon has a number painted on it represented by array `nums`. You must burst all balloons.

If you burst balloon i, you gain `nums[i-1] * nums[i] * nums[i+1]` coins. If i-1 or i+1 goes out of bounds, treat it as if there is a balloon with 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

## Function Signature

```python
def max_coins(nums: list[int]) -> int:
    """
    Find maximum coins from bursting all balloons.
    
    Args:
        nums: List of balloon values
    
    Returns:
        Maximum coins obtainable
    
    Examples:
        >>> max_coins([3, 1, 5, 8])
        167
        # Burst order: 1, 5, 3, 8
        # Coins: 3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167
    """
    pass
```

## Test Cases

```python
assert max_coins([3, 1, 5, 8]) == 167
assert max_coins([1, 5]) == 10
assert max_coins([1]) == 1
assert max_coins([]) == 0
assert max_coins([9, 76, 64, 21]) == 116718
```

## Expected Complexity

- **Time**: O(n³)
- **Space**: O(n²)

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**Key insight**: Think about which balloon to burst LAST in a range, not first.

**State**: `dp[i][j]` = max coins from bursting all balloons in range (i, j) exclusive

**Setup**: Add 1 to both ends: `nums = [1] + nums + [1]`

**Transition**: For each k in range (i+1, j):
```
dp[i][j] = max(dp[i][j], 
               dp[i][k] + nums[i]*nums[k]*nums[j] + dp[k][j])
```

k is the LAST balloon to burst in range. At that point, only nums[i] and nums[j] remain as neighbours.

**Iteration order**: Increasing range length (j - i)

</details>
