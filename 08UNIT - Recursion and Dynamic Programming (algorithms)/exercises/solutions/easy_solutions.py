#!/usr/bin/env python3
"""
Solutions for Easy Practice Exercises
=====================================

Unit 8: Recursion and Dynamic Programming
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# EASY 01: Sum of Digits
# ═══════════════════════════════════════════════════════════════════════════════

def sum_digits(n: int) -> int:
    """
    Compute sum of digits recursively.
    
    Time: O(d) where d is number of digits
    Space: O(d) for recursion stack
    
    Args:
        n: Non-negative integer
    
    Returns:
        Sum of all digits in n
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


# ═══════════════════════════════════════════════════════════════════════════════
# EASY 02: Power Function
# ═══════════════════════════════════════════════════════════════════════════════

def power_naive(x: float, n: int) -> float:
    """
    Compute x^n using naive recursion.
    
    Time: O(n)
    Space: O(n)
    """
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    
    # Base case
    if n == 0:
        return 1
    
    # Recursive case
    return x * power_naive(x, n - 1)


def power_optimised(x: float, n: int) -> float:
    """
    Compute x^n using fast exponentiation.
    
    Time: O(log n)
    Space: O(log n)
    
    Uses the property:
    - x^n = (x^(n/2))^2 if n is even
    - x^n = x × (x^(n/2))^2 if n is odd
    """
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    
    # Base case
    if n == 0:
        return 1
    
    # Recursive case
    half = power_optimised(x, n // 2)
    
    if n % 2 == 0:
        return half * half
    else:
        return x * half * half


def power_iterative(x: float, n: int) -> float:
    """Iterative fast exponentiation."""
    if n < 0:
        raise ValueError("Exponent must be non-negative")
    
    result = 1
    base = x
    
    while n > 0:
        if n % 2 == 1:
            result *= base
        base *= base
        n //= 2
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EASY 03: Reverse String
# ═══════════════════════════════════════════════════════════════════════════════

def reverse_string(s: str) -> str:
    """
    Reverse a string recursively.
    
    Time: O(n)
    Space: O(n)
    """
    # Base case: empty or single character
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])


def reverse_string_alt(s: str) -> str:
    """Alternative recursive approach."""
    if len(s) <= 1:
        return s
    
    # First char goes to end
    return reverse_string_alt(s[1:]) + s[0]


def reverse_string_iterative(s: str) -> str:
    """Iterative version using two pointers."""
    chars = list(s)
    left, right = 0, len(chars) - 1
    
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    
    return "".join(chars)


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test sum_digits
    assert sum_digits(0) == 0
    assert sum_digits(5) == 5
    assert sum_digits(123) == 6
    assert sum_digits(9999) == 36
    print("✓ sum_digits tests passed")
    
    # Test power
    assert power_naive(2, 10) == 1024
    assert power_optimised(2, 10) == 1024
    assert power_optimised(3, 0) == 1
    print("✓ power tests passed")
    
    # Test reverse_string
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
    assert reverse_string("hello") == "olleh"
    print("✓ reverse_string tests passed")
    
    print("\nAll easy solutions verified!")
