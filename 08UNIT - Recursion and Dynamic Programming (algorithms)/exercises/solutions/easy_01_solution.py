#!/usr/bin/env python3
"""
Solution: Sum of Digits (Easy)
"""


def sum_digits(n: int) -> int:
    """
    Compute sum of digits recursively.
    
    Time: O(d) where d is number of digits
    Space: O(d) for recursion stack
    """
    if n < 0:
        raise ValueError("Input must be non-negative")
    
    # Base case: single digit
    if n < 10:
        return n
    
    # Recursive case: last digit + sum of remaining digits
    return (n % 10) + sum_digits(n // 10)


def sum_digits_iterative(n: int) -> int:
    """Iterative version for comparison."""
    if n < 0:
        raise ValueError("Input must be non-negative")
    
    total = 0
    while n > 0:
        total += n % 10
        n //= 10
    return total


if __name__ == "__main__":
    # Test cases
    assert sum_digits(0) == 0
    assert sum_digits(5) == 5
    assert sum_digits(123) == 6
    assert sum_digits(9999) == 36
    assert sum_digits(10000) == 1
    print("All tests passed!")
