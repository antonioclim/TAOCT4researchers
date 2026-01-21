# Practice Exercise: Coin Change (Medium)

**Difficulty**: ★★★☆☆  
**Estimated Time**: 30 minutes  
**Topic**: Dynamic Programming (Unbounded Knapsack)

---

## Problem Statement

Given an array of coin denominations and a target amount, find the minimum number of coins needed to make that amount. You have an unlimited supply of each coin denomination.

## Function Signature

```python
def coin_change(coins: list[int], amount: int) -> int:
    """
    Find minimum coins needed to make amount.
    
    Args:
        coins: List of coin denominations
        amount: Target amount
    
    Returns:
        Minimum number of coins, or -1 if impossible
    
    Examples:
        >>> coin_change([1, 2, 5], 11)
        3  # 5 + 5 + 1
        >>> coin_change([2], 3)
        -1  # impossible
    """
    pass
```

## Variant

Also implement a function to count the total number of ways to make the amount:

```python
def coin_change_ways(coins: list[int], amount: int) -> int:
    """
    Count number of ways to make amount using given coins.
    
    Examples:
        >>> coin_change_ways([1, 2, 5], 5)
        4  # (5), (2+2+1), (2+1+1+1), (1+1+1+1+1)
    """
    pass
```

## Test Cases

```python
# Minimum coins
assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([1], 2) == 2
assert coin_change([1, 2, 5], 100) == 20

# Number of ways
assert coin_change_ways([1, 2, 5], 5) == 4
assert coin_change_ways([2], 3) == 0
assert coin_change_ways([1, 2, 3], 4) == 4
```

## Expected Complexity

- **Time**: O(amount × len(coins))
- **Space**: O(amount)

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**Minimum coins:**
- State: `dp[i]` = minimum coins to make amount i
- Transition: `dp[i] = min(dp[i], dp[i - coin] + 1)` for each coin
- Initialisation: `dp[0] = 0`, others = infinity

**Number of ways:**
- State: `dp[i]` = number of ways to make amount i
- Transition: `dp[i] += dp[i - coin]` for each coin
- Initialisation: `dp[0] = 1`, others = 0
- Note: Iterate coins in outer loop to avoid counting permutations

</details>
