#!/usr/bin/env python3
"""Exercise 03: Input Validation (Easy).

This exercise introduces basic input validation patterns
with informative error messages.

Learning Objectives:
    - Validate function inputs before processing
    - Provide clear, actionable error messages
    - Use type checking for runtime validation

Estimated Time: 10 minutes
Difficulty: Easy (★☆☆)

Instructions:
    1. Complete each TODO section
    2. Run the tests: pytest tests/test_exercises.py::test_e03
    3. Focus on clear, helpful error messages
"""

from __future__ import annotations

from typing import Any


def validate_non_empty_string(value: Any, name: str = "value") -> str:
    """Validate that value is a non-empty string.
    
    Args:
        value: Value to validate.
        name: Name for error messages.
        
    Returns:
        The validated string.
        
    Raises:
        TypeError: If value is not a string.
        ValueError: If string is empty or only whitespace.
        
    Example:
        >>> validate_non_empty_string("hello")
        'hello'
        >>> validate_non_empty_string(123, "username")
        Traceback (most recent call last):
            ...
        TypeError: username must be a string, got int
        >>> validate_non_empty_string("   ", "username")
        Traceback (most recent call last):
            ...
        ValueError: username cannot be empty or whitespace
    """
    # TODO: Implement validation
    # 1. Check if value is a string (isinstance)
    # 2. Check if string is empty or only whitespace (.strip())
    # Your code here
    pass


def validate_in_range(
    value: int | float,
    min_val: int | float,
    max_val: int | float,
    name: str = "value",
) -> int | float:
    """Validate that value is within a range (inclusive).
    
    Args:
        value: Value to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Name for error messages.
        
    Returns:
        The validated value.
        
    Raises:
        TypeError: If value is not numeric.
        ValueError: If value is outside the range.
        
    Example:
        >>> validate_in_range(5, 0, 10)
        5
        >>> validate_in_range(15, 0, 10, "score")
        Traceback (most recent call last):
            ...
        ValueError: score must be between 0 and 10, got 15
    """
    # TODO: Implement validation
    # 1. Check if value is int or float
    # 2. Check if value is in range [min_val, max_val]
    # Your code here
    pass


def validate_list_of_type(
    items: Any,
    expected_type: type,
    name: str = "items",
) -> list[Any]:
    """Validate that input is a list containing only items of expected type.
    
    Args:
        items: Value to validate.
        expected_type: Required type for all items.
        name: Name for error messages.
        
    Returns:
        The validated list.
        
    Raises:
        TypeError: If items is not a list or contains wrong types.
        
    Example:
        >>> validate_list_of_type([1, 2, 3], int)
        [1, 2, 3]
        >>> validate_list_of_type("not a list", int, "numbers")
        Traceback (most recent call last):
            ...
        TypeError: numbers must be a list, got str
        >>> validate_list_of_type([1, "two", 3], int, "numbers")
        Traceback (most recent call last):
            ...
        TypeError: numbers[1] must be int, got str
    """
    # TODO: Implement validation
    # 1. Check if items is a list
    # 2. Check if each item has the expected type
    # Your code here
    pass


def validate_dict_keys(
    data: Any,
    required_keys: set[str],
    name: str = "data",
) -> dict[str, Any]:
    """Validate that input is a dict containing required keys.
    
    Args:
        data: Value to validate.
        required_keys: Set of keys that must be present.
        name: Name for error messages.
        
    Returns:
        The validated dictionary.
        
    Raises:
        TypeError: If data is not a dict.
        KeyError: If required keys are missing.
        
    Example:
        >>> validate_dict_keys({"a": 1, "b": 2}, {"a", "b"})
        {'a': 1, 'b': 2}
        >>> validate_dict_keys({"a": 1}, {"a", "b"}, "config")
        Traceback (most recent call last):
            ...
        KeyError: "config is missing required keys: {'b'}"
    """
    # TODO: Implement validation
    # 1. Check if data is a dict
    # 2. Check if all required_keys are present
    # Your code here
    pass


def process_user_data(
    name: str,
    age: int,
    email: str,
) -> dict[str, Any]:
    """Process user data with full validation.
    
    Validates all inputs before processing:
    - name: non-empty string
    - age: integer between 0 and 150
    - email: non-empty string containing '@'
    
    Args:
        name: User's name.
        age: User's age.
        email: User's email address.
        
    Returns:
        Dictionary with validated user data.
        
    Raises:
        TypeError: If any input has wrong type.
        ValueError: If any input is invalid.
        
    Example:
        >>> process_user_data("Alice", 30, "alice@example.com")
        {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}
    """
    # TODO: Implement validation using the functions above
    # 1. Validate name using validate_non_empty_string
    # 2. Validate age using validate_in_range (0-150)
    # 3. Validate email using validate_non_empty_string
    # 4. Check email contains '@' (raise ValueError if not)
    # 5. Return dict with validated data
    # Your code here
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
    print("Testing validate_non_empty_string...")
    assert validate_non_empty_string("hello") == "hello"
    try:
        validate_non_empty_string(123)
    except TypeError as e:
        assert "string" in str(e).lower()
    try:
        validate_non_empty_string("   ")
    except ValueError as e:
        assert "empty" in str(e).lower() or "whitespace" in str(e).lower()
    print("  ✓ validate_non_empty_string passed")
    
    print("Testing validate_in_range...")
    assert validate_in_range(5, 0, 10) == 5
    assert validate_in_range(0, 0, 10) == 0
    assert validate_in_range(10, 0, 10) == 10
    try:
        validate_in_range(15, 0, 10)
    except ValueError:
        pass
    try:
        validate_in_range("five", 0, 10)
    except TypeError:
        pass
    print("  ✓ validate_in_range passed")
    
    print("Testing validate_list_of_type...")
    assert validate_list_of_type([1, 2, 3], int) == [1, 2, 3]
    assert validate_list_of_type([], str) == []
    try:
        validate_list_of_type("not a list", int)
    except TypeError:
        pass
    try:
        validate_list_of_type([1, "two"], int)
    except TypeError as e:
        assert "[1]" in str(e)  # Should mention the index
    print("  ✓ validate_list_of_type passed")
    
    print("Testing validate_dict_keys...")
    assert validate_dict_keys({"a": 1, "b": 2}, {"a", "b"}) == {"a": 1, "b": 2}
    assert validate_dict_keys({"a": 1, "b": 2, "c": 3}, {"a"}) == {"a": 1, "b": 2, "c": 3}
    try:
        validate_dict_keys({"a": 1}, {"a", "b"})
    except KeyError:
        pass
    print("  ✓ validate_dict_keys passed")
    
    print("Testing process_user_data...")
    result = process_user_data("Alice", 30, "alice@example.com")
    assert result["name"] == "Alice"
    assert result["age"] == 30
    assert result["email"] == "alice@example.com"
    try:
        process_user_data("", 30, "alice@example.com")
    except ValueError:
        pass
    try:
        process_user_data("Alice", 200, "alice@example.com")
    except ValueError:
        pass
    try:
        process_user_data("Alice", 30, "invalid-email")
    except ValueError:
        pass
    print("  ✓ process_user_data passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
