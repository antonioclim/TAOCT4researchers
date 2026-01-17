#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Easy Exercise 2 — Basic Testing with pytest
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for fundamental pytest exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# CODE UNDER TEST (provided functions)
# ═══════════════════════════════════════════════════════════════════════════════


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def is_palindrome(text: str) -> bool:
    """Check if text is a palindrome (case-insensitive, ignores spaces)."""
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: BASIC TEST CASES — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdd:
    """Test cases for the add function following AAA pattern."""

    def test_add_positive_numbers(self) -> None:
        """Test addition of two positive numbers."""
        # Arrange
        a, b = 3, 5

        # Act
        result = add(a, b)

        # Assert
        assert result == 8

    def test_add_negative_numbers(self) -> None:
        """Test addition of two negative numbers."""
        # Arrange
        a, b = -3, -5

        # Act
        result = add(a, b)

        # Assert
        assert result == -8

    def test_add_mixed_signs(self) -> None:
        """Test addition of positive and negative numbers."""
        # Arrange
        a, b = 10, -3

        # Act
        result = add(a, b)

        # Assert
        assert result == 7

    def test_add_with_zero(self) -> None:
        """Test addition with zero."""
        # Arrange
        a, b = 5, 0

        # Act
        result = add(a, b)

        # Assert
        assert result == 5

    def test_add_floats(self) -> None:
        """Test addition of floating-point numbers."""
        # Arrange
        a, b = 1.5, 2.5

        # Act
        result = add(a, b)

        # Assert
        assert result == pytest.approx(4.0)


class TestMultiply:
    """Test cases for the multiply function."""

    def test_multiply_positive_numbers(self) -> None:
        """Test multiplication of two positive numbers."""
        assert multiply(4, 5) == 20

    def test_multiply_with_zero(self) -> None:
        """Test multiplication by zero."""
        assert multiply(100, 0) == 0

    def test_multiply_negative_numbers(self) -> None:
        """Test multiplication of two negatives (positive result)."""
        assert multiply(-3, -4) == 12

    def test_multiply_mixed_signs(self) -> None:
        """Test multiplication with mixed signs (negative result)."""
        assert multiply(5, -3) == -15


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: EXCEPTION TESTING — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestDivide:
    """Test cases for the divide function, including exceptions."""

    def test_divide_integers(self) -> None:
        """Test integer division."""
        assert divide(10, 2) == 5.0

    def test_divide_results_in_float(self) -> None:
        """Test division resulting in a non-integer."""
        assert divide(7, 2) == pytest.approx(3.5)

    def test_divide_by_zero_raises_error(self) -> None:
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(10, 0)

        assert "Cannot divide by zero" in str(exc_info.value)

    def test_divide_negative_numbers(self) -> None:
        """Test division with negative numbers."""
        assert divide(-10, 2) == -5.0
        assert divide(10, -2) == -5.0
        assert divide(-10, -2) == 5.0


class TestFactorial:
    """Test cases for the factorial function."""

    def test_factorial_zero(self) -> None:
        """Test factorial of zero (edge case: 0! = 1)."""
        assert factorial(0) == 1

    def test_factorial_one(self) -> None:
        """Test factorial of one."""
        assert factorial(1) == 1

    def test_factorial_small_numbers(self) -> None:
        """Test factorial of small positive numbers."""
        assert factorial(5) == 120
        assert factorial(3) == 6

    def test_factorial_larger_number(self) -> None:
        """Test factorial of a larger number."""
        assert factorial(10) == 3628800

    def test_factorial_negative_raises_error(self) -> None:
        """Test that negative input raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            factorial(-1)

        assert "negative" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: STRING FUNCTION TESTING — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


class TestIsPalindrome:
    """Comprehensive test cases for the is_palindrome function."""

    def test_simple_palindrome(self) -> None:
        """Test a simple palindrome word."""
        assert is_palindrome("radar") is True

    def test_non_palindrome(self) -> None:
        """Test a non-palindrome word."""
        assert is_palindrome("hello") is False

    def test_palindrome_with_spaces(self) -> None:
        """Test palindrome phrase with spaces."""
        assert is_palindrome("race car") is True

    def test_palindrome_mixed_case(self) -> None:
        """Test case-insensitive palindrome detection."""
        assert is_palindrome("Radar") is True
        assert is_palindrome("RaCeCaR") is True

    def test_single_character(self) -> None:
        """Test single character (always a palindrome)."""
        assert is_palindrome("a") is True

    def test_empty_string(self) -> None:
        """Test empty string (considered a palindrome)."""
        assert is_palindrome("") is True

    def test_two_character_palindrome(self) -> None:
        """Test two-character palindrome."""
        assert is_palindrome("aa") is True

    def test_two_character_non_palindrome(self) -> None:
        """Test two different characters."""
        assert is_palindrome("ab") is False

    def test_numeric_palindrome(self) -> None:
        """Test numeric string palindrome."""
        assert is_palindrome("12321") is True
        assert is_palindrome("12345") is False


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRISED TESTS (BONUS) — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
    (1.5, 2.5, 4.0),
])
def test_add_parametrised(a: float, b: float, expected: float) -> None:
    """Parametrised tests for add function."""
    assert add(a, b) == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("radar", True),
    ("hello", False),
    ("A man a plan a canal Panama".replace(" ", ""), True),
    ("", True),
    ("a", True),
    ("ab", False),
])
def test_palindrome_parametrised(text: str, expected: bool) -> None:
    """Parametrised tests for is_palindrome function."""
    assert is_palindrome(text) is expected


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_tests() -> None:
    """Run all tests manually without pytest."""
    print("=" * 70)
    print("SOLUTION VALIDATION: Easy Exercise 2 — Basic Testing")
    print("=" * 70)

    # Test add
    print("\n--- Testing add() ---")
    assert add(3, 5) == 8
    assert add(-3, -5) == -8
    assert add(10, -3) == 7
    print("✓ add() tests passed")

    # Test multiply
    print("\n--- Testing multiply() ---")
    assert multiply(4, 5) == 20
    assert multiply(100, 0) == 0
    assert multiply(-3, -4) == 12
    print("✓ multiply() tests passed")

    # Test divide
    print("\n--- Testing divide() ---")
    assert divide(10, 2) == 5.0
    try:
        divide(10, 0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero" in str(e).lower()
    print("✓ divide() tests passed")

    # Test factorial
    print("\n--- Testing factorial() ---")
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    try:
        factorial(-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "negative" in str(e).lower()
    print("✓ factorial() tests passed")

    # Test is_palindrome
    print("\n--- Testing is_palindrome() ---")
    assert is_palindrome("radar") is True
    assert is_palindrome("hello") is False
    assert is_palindrome("race car") is True
    assert is_palindrome("Radar") is True
    assert is_palindrome("") is True
    print("✓ is_palindrome() tests passed")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
