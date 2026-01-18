#!/usr/bin/env python3
"""Solution for Exercise 03: Input Validation.

This module provides reference implementations for input
validation patterns with informative error messages.
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
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    
    if not value.strip():
        raise ValueError(f"{name} cannot be empty or whitespace")
    
    return value


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
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    
    return value


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
    """
    if not isinstance(items, list):
        raise TypeError(f"{name} must be a list, got {type(items).__name__}")
    
    for i, item in enumerate(items):
        if not isinstance(item, expected_type):
            raise TypeError(
                f"{name}[{i}] must be {expected_type.__name__}, "
                f"got {type(item).__name__}"
            )
    
    return items


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
    """
    if not isinstance(data, dict):
        raise TypeError(f"{name} must be a dict, got {type(data).__name__}")
    
    missing = required_keys - set(data.keys())
    if missing:
        raise KeyError(f"{name} is missing required keys: {missing}")
    
    return data


def process_user_data(
    name: str,
    age: int,
    email: str,
) -> dict[str, Any]:
    """Process user data with full validation.
    
    Args:
        name: User's name.
        age: User's age.
        email: User's email address.
        
    Returns:
        Dictionary with validated user data.
        
    Raises:
        TypeError: If any input has wrong type.
        ValueError: If any input is invalid.
    """
    validated_name = validate_non_empty_string(name, "name")
    validated_age = validate_in_range(age, 0, 150, "age")
    validated_email = validate_non_empty_string(email, "email")
    
    if "@" not in validated_email:
        raise ValueError("email must contain '@'")
    
    return {
        "name": validated_name,
        "age": validated_age,
        "email": validated_email,
    }


def main() -> None:
    """Verify solution implementations."""
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
    print("  ✓ validate_in_range passed")
    
    print("Testing validate_list_of_type...")
    assert validate_list_of_type([1, 2, 3], int) == [1, 2, 3]
    try:
        validate_list_of_type([1, "two"], int)
    except TypeError as e:
        assert "[1]" in str(e)
    print("  ✓ validate_list_of_type passed")
    
    print("Testing validate_dict_keys...")
    assert validate_dict_keys({"a": 1, "b": 2}, {"a", "b"}) == {"a": 1, "b": 2}
    try:
        validate_dict_keys({"a": 1}, {"a", "b"})
    except KeyError:
        pass
    print("  ✓ validate_dict_keys passed")
    
    print("Testing process_user_data...")
    result = process_user_data("Alice", 30, "alice@example.com")
    assert result["name"] == "Alice"
    assert result["age"] == 30
    print("  ✓ process_user_data passed")
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
