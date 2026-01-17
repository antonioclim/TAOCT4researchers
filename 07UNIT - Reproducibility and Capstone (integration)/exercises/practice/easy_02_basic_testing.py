#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Easy 02 - Basic Testing with pytest
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Testing is essential for ensuring code correctness and maintaining confidence
during refactoring. This exercise introduces the fundamentals of writing tests
using pytest, following the Arrange-Act-Assert (AAA) pattern.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Write basic pytest test functions
2. Use assertions effectively
3. Apply the Arrange-Act-Assert pattern

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 25 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# CODE TO TEST (Do not modify)
# ═══════════════════════════════════════════════════════════════════════════════

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ZeroDivisionError if b is zero."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Calculate factorial of n. Raises ValueError if n is negative."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome (case-insensitive)."""
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Write Basic Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_positive_numbers() -> None:
    """
    Test that add() correctly adds two positive numbers.

    Follow the AAA pattern:
    - Arrange: Set up test data
    - Act: Call the function
    - Assert: Check the result

    Example:
        # Arrange
        a, b = 3, 5
        # Act
        result = add(a, b)
        # Assert
        assert result == 8
    """
    # TODO: Implement this test
    pass


def test_add_negative_numbers() -> None:
    """Test that add() correctly handles negative numbers."""
    # TODO: Implement this test
    pass


def test_add_floats() -> None:
    """Test that add() correctly adds floating-point numbers."""
    # TODO: Implement this test
    # Hint: Use pytest.approx() for float comparisons
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Test Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_divide_normal() -> None:
    """Test normal division."""
    # TODO: Implement this test
    pass


def test_divide_by_zero_raises() -> None:
    """
    Test that divide() raises ZeroDivisionError when dividing by zero.

    Hint: Use pytest.raises() context manager:
        import pytest
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)
    """
    # TODO: Implement this test
    pass


def test_factorial_zero() -> None:
    """Test that factorial(0) returns 1."""
    # TODO: Implement this test
    pass


def test_factorial_positive() -> None:
    """Test factorial with positive numbers."""
    # TODO: Implement this test
    pass


def test_factorial_negative_raises() -> None:
    """Test that factorial raises ValueError for negative input."""
    # TODO: Implement this test
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Test String Functions
# ═══════════════════════════════════════════════════════════════════════════════

def test_is_palindrome_simple() -> None:
    """Test is_palindrome with simple palindromes."""
    # TODO: Test cases like "radar", "level", "noon"
    pass


def test_is_palindrome_with_spaces() -> None:
    """Test is_palindrome with spaces and punctuation."""
    # TODO: Test cases like "A man a plan a canal Panama"
    pass


def test_is_palindrome_not_palindrome() -> None:
    """Test is_palindrome returns False for non-palindromes."""
    # TODO: Test cases like "hello", "python"
    pass


def test_is_palindrome_empty_string() -> None:
    """Test is_palindrome with empty string."""
    # TODO: Empty string should be considered a palindrome
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests() -> None:
    """Run all tests manually (without pytest)."""
    import pytest

    tests = [
        test_add_positive_numbers,
        test_add_negative_numbers,
        test_add_floats,
        test_divide_normal,
        test_divide_by_zero_raises,
        test_factorial_zero,
        test_factorial_positive,
        test_factorial_negative_raises,
        test_is_palindrome_simple,
        test_is_palindrome_with_spaces,
        test_is_palindrome_not_palindrome,
        test_is_palindrome_empty_string,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"  ✓ {test.__name__}")
            passed += 1
        except (AssertionError, NotImplementedError) as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__}: Unexpected error: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed! 🎉")
    print("=" * 60)
    print("\nTo run with pytest, use: pytest easy_02_basic_testing.py -v")


if __name__ == "__main__":
    run_all_tests()
