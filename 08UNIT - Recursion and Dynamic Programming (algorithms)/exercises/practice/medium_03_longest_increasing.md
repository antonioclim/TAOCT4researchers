# Practice Exercise: Longest Increasing Subsequence (Medium)

**Difficulty**: ★★★☆☆  
**Estimated Time**: 35 minutes  
**Topic**: Dynamic Programming (1D Sequence)

---

## Problem Statement

Given an integer array, find the length of the longest strictly increasing subsequence.

A subsequence is derived by deleting some or no elements without changing the order of the remaining elements.

## Function Signature

```python
def longest_increasing_subsequence(nums: list[int]) -> int:
    """
    Find length of longest increasing subsequence.
    
    Args:
        nums: List of integers
    
    Returns:
        Length of LIS
    
    Examples:
        >>> longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18])
        4  # [2, 3, 7, 101] or [2, 5, 7, 101]
        >>> longest_increasing_subsequence([0, 1, 0, 3, 2, 3])
        4  # [0, 1, 2, 3]
    """
    pass
```

## Variant (Harder)

Return the actual subsequence, not just its length:

```python
def lis_with_sequence(nums: list[int]) -> list[int]:
    """
    Return one of the longest increasing subsequences.
    """
    pass
```

## Test Cases

```python
assert longest_increasing_subsequence([]) == 0
assert longest_increasing_subsequence([1]) == 1
assert longest_increasing_subsequence([1, 2, 3, 4, 5]) == 5
assert longest_increasing_subsequence([5, 4, 3, 2, 1]) == 1
assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4
```

## Expected Complexity

- **O(n²) DP**: Time O(n²), Space O(n)
- **O(n log n) Binary Search**: Time O(n log n), Space O(n)

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**O(n²) DP:**
- State: `dp[i]` = length of LIS ending at index i
- Transition: `dp[i] = max(dp[j] + 1)` for all j < i where nums[j] < nums[i]
- Answer: `max(dp)`

**O(n log n) with Binary Search:**
- Maintain array `tails` where `tails[i]` = smallest tail of all increasing subsequences of length i+1
- For each number, binary search for its position in tails
- If larger than all, append; otherwise replace

</details>
