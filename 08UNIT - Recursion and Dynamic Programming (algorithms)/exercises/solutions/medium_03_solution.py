#!/usr/bin/env python3
"""
Solution: Longest Increasing Subsequence (Medium)
"""

from bisect import bisect_left


def longest_increasing_subsequence(nums: list[int]) -> int:
    """
    Find length of LIS using O(n²) DP.
    
    Time: O(n²)
    Space: O(n)
    
    State: dp[i] = length of LIS ending at index i
    Transition: dp[i] = max(dp[j] + 1) for all j < i where nums[j] < nums[i]
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)


def lis_binary_search(nums: list[int]) -> int:
    """
    Find length of LIS using O(n log n) binary search.
    
    Time: O(n log n)
    Space: O(n)
    
    Maintain array 'tails' where tails[i] = smallest tail element
    of all increasing subsequences of length i+1.
    """
    if not nums:
        return 0
    
    tails = []
    
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)


def lis_with_sequence(nums: list[int]) -> list[int]:
    """
    Return the actual LIS, not just its length.
    
    Time: O(n²)
    Space: O(n)
    """
    if not nums:
        return []
    
    n = len(nums)
    dp = [1] * n
    parent = [-1] * n
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    
    # Find ending position of LIS
    max_length = max(dp)
    end_idx = dp.index(max_length)
    
    # Reconstruct sequence
    result = []
    idx = end_idx
    while idx != -1:
        result.append(nums[idx])
        idx = parent[idx]
    
    return result[::-1]


def lis_binary_search_with_sequence(nums: list[int]) -> list[int]:
    """
    Return actual LIS using O(n log n) approach.
    
    Time: O(n log n)
    Space: O(n)
    """
    if not nums:
        return []
    
    n = len(nums)
    tails = []
    tail_indices = []
    parent = [-1] * n
    
    for i, num in enumerate(nums):
        pos = bisect_left(tails, num)
        
        if pos == len(tails):
            tails.append(num)
            tail_indices.append(i)
        else:
            tails[pos] = num
            tail_indices[pos] = i
        
        if pos > 0:
            parent[i] = tail_indices[pos - 1]
    
    # Reconstruct
    result = []
    idx = tail_indices[-1] if tail_indices else -1
    while idx != -1:
        result.append(nums[idx])
        idx = parent[idx]
    
    return result[::-1]


if __name__ == "__main__":
    # Length tests
    assert longest_increasing_subsequence([]) == 0
    assert longest_increasing_subsequence([1]) == 1
    assert longest_increasing_subsequence([1, 2, 3, 4, 5]) == 5
    assert longest_increasing_subsequence([5, 4, 3, 2, 1]) == 1
    assert longest_increasing_subsequence([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert longest_increasing_subsequence([0, 1, 0, 3, 2, 3]) == 4
    
    # Binary search version
    assert lis_binary_search([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    
    # With sequence
    seq = lis_with_sequence([10, 9, 2, 5, 3, 7, 101, 18])
    assert len(seq) == 4
    assert seq == sorted(seq)  # Must be increasing
    
    print("All tests passed!")
