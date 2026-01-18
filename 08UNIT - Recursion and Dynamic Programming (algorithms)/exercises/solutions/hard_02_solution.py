#!/usr/bin/env python3
"""
Solution: Burst Balloons (Hard)
"""

from functools import lru_cache


def max_coins(nums: list[int]) -> int:
    """
    Find maximum coins from bursting all balloons.
    
    Time: O(n³)
    Space: O(n²)
    
    Key insight: Think about which balloon to burst LAST in a range.
    
    State: dp[i][j] = max coins from bursting balloons in range (i, j) exclusive
    
    After adding boundary 1s: nums = [1] + nums + [1]
    
    Transition: For k in (i+1, j):
        dp[i][j] = max(dp[i][j], dp[i][k] + nums[i]*nums[k]*nums[j] + dp[k][j])
    
    k is the LAST balloon to burst. At that point, only nums[i] and nums[j]
    remain as neighbours (everything in between has been burst).
    """
    if not nums:
        return 0
    
    # Add boundary balloons with value 1
    nums = [1] + nums + [1]
    n = len(nums)
    
    # dp[i][j] = max coins for range (i, j) exclusive
    dp = [[0] * n for _ in range(n)]
    
    # Iterate by increasing range length
    for length in range(2, n):  # length = j - i
        for i in range(n - length):
            j = i + length
            
            # Try each balloon k as the LAST one to burst in range (i, j)
            for k in range(i + 1, j):
                coins = dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)
    
    return dp[0][n - 1]


def max_coins_recursive(nums: list[int]) -> int:
    """
    Recursive solution with memoisation.
    
    Time: O(n³)
    Space: O(n²)
    """
    if not nums:
        return 0
    
    nums = [1] + nums + [1]
    n = len(nums)
    
    @lru_cache(maxsize=None)
    def dp(i: int, j: int) -> int:
        if j - i <= 1:
            return 0
        
        max_coins = 0
        for k in range(i + 1, j):
            coins = dp(i, k) + nums[i] * nums[k] * nums[j] + dp(k, j)
            max_coins = max(max_coins, coins)
        
        return max_coins
    
    return dp(0, n - 1)


def max_coins_with_order(nums: list[int]) -> tuple[int, list[int]]:
    """
    Return maximum coins and the optimal burst order.
    
    Time: O(n³)
    Space: O(n²)
    """
    if not nums:
        return 0, []
    
    original_len = len(nums)
    nums = [1] + nums + [1]
    n = len(nums)
    
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)]
    
    for length in range(2, n):
        for i in range(n - length):
            j = i + length
            
            for k in range(i + 1, j):
                coins = dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j]
                if coins > dp[i][j]:
                    dp[i][j] = coins
                    split[i][j] = k
    
    # Reconstruct burst order (k is LAST to burst in range)
    order = []
    
    def reconstruct(i: int, j: int) -> None:
        if j - i <= 1:
            return
        k = split[i][j]
        reconstruct(i, k)
        reconstruct(k, j)
        # k is last to burst, so add after recursion
        order.append(k - 1)  # Adjust for added boundary
    
    reconstruct(0, n - 1)
    
    # Filter to original indices
    order = [idx for idx in order if 0 <= idx < original_len]
    
    return dp[0][n - 1], order


if __name__ == "__main__":
    # Test cases
    assert max_coins([3, 1, 5, 8]) == 167
    assert max_coins([1, 5]) == 10
    assert max_coins([1]) == 1
    assert max_coins([]) == 0
    
    # Recursive version
    assert max_coins_recursive([3, 1, 5, 8]) == 167
    
    # With order
    coins, order = max_coins_with_order([3, 1, 5, 8])
    assert coins == 167
    
    print("All tests passed!")
