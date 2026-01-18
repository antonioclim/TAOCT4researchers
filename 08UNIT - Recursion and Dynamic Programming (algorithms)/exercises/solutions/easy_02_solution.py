#!/usr/bin/env python3
"""
Solution: Power Function (Easy)
"""


def power_naive(x: float, n: int) -> float:
    """
    Compute x^n using naive recursion.
    
    Time: O(n)
    Space: O(n)
    """
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    if n == 0:
        return 1
    return x * power_naive(x, n - 1)


def power_optimised(x: float, n: int) -> float:
    """
    Compute x^n using fast exponentiation.
    
    Time: O(log n)
    Space: O(log n)
    
    Uses the identity:
    - x^n = (x^(n/2))^2 if n is even
    - x^n = x * (x^(n/2))^2 if n is odd
    """
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    if n == 0:
        return 1
    if n == 1:
        return x
    
    half = power_optimised(x, n // 2)
    
    if n % 2 == 0:
        return half * half
    else:
        return x * half * half


def power_iterative(x: float, n: int) -> float:
    """
    Iterative fast exponentiation.
    
    Time: O(log n)
    Space: O(1)
    """
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    
    result = 1
    while n > 0:
        if n % 2 == 1:
            result *= x
        x *= x
        n //= 2
    return result


if __name__ == "__main__":
    # Test cases
    assert power_naive(2, 0) == 1
    assert power_naive(2, 1) == 2
    assert power_naive(2, 10) == 1024
    
    assert power_optimised(2, 10) == 1024
    assert power_optimised(3, 4) == 81
    
    assert abs(power_optimised(2.5, 2) - 6.25) < 0.001
    
    print("All tests passed!")
