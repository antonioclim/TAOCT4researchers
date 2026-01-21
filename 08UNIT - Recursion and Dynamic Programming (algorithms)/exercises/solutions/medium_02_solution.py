#!/usr/bin/env python3
"""
Solution: Coin Change (Medium)
"""


def coin_change(coins: list[int], amount: int) -> int:
    """
    Find minimum coins needed to make amount.
    
    Time: O(amount × len(coins))
    Space: O(amount)
    
    State: dp[i] = minimum coins to make amount i
    Transition: dp[i] = min(dp[i], dp[i - coin] + 1) for each coin
    """
    if amount == 0:
        return 0
    
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] != float("inf"):
                dp[x] = min(dp[x], dp[x - coin] + 1)
    
    return dp[amount] if dp[amount] != float("inf") else -1


def coin_change_ways(coins: list[int], amount: int) -> int:
    """
    Count number of ways to make amount.
    
    Time: O(amount × len(coins))
    Space: O(amount)
    
    Note: Coins in outer loop to count combinations (not permutations)
    """
    dp = [0] * (amount + 1)
    dp[0] = 1
    
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    
    return dp[amount]


def coin_change_with_coins(coins: list[int], amount: int) -> tuple[int, list[int]]:
    """
    Return minimum coins and the actual coins used.
    
    Time: O(amount × len(coins))
    Space: O(amount)
    """
    if amount == 0:
        return 0, []
    
    dp = [float("inf")] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0
    
    for coin in coins:
        for x in range(coin, amount + 1):
            if dp[x - coin] + 1 < dp[x]:
                dp[x] = dp[x - coin] + 1
                parent[x] = coin
    
    if dp[amount] == float("inf"):
        return -1, []
    
    # Reconstruct coins used
    result = []
    current = amount
    while current > 0:
        result.append(parent[current])
        current -= parent[current]
    
    return dp[amount], result


if __name__ == "__main__":
    # Minimum coins tests
    assert coin_change([1, 2, 5], 11) == 3
    assert coin_change([2], 3) == -1
    assert coin_change([1], 0) == 0
    assert coin_change([1], 2) == 2
    assert coin_change([1, 2, 5], 100) == 20
    
    # Number of ways tests
    assert coin_change_ways([1, 2, 5], 5) == 4
    assert coin_change_ways([2], 3) == 0
    assert coin_change_ways([1, 2, 3], 4) == 4
    
    # With reconstruction
    count, coins_used = coin_change_with_coins([1, 2, 5], 11)
    assert count == 3
    assert sum(coins_used) == 11
    
    print("All tests passed!")
