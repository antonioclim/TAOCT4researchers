#!/usr/bin/env python3
"""Solution for Exercise 01: Basic Exception Handling.

This module provides reference implementations for basic
exception handling patterns.
"""

from __future__ import annotations

from typing import Any


def safe_list_access(items: list[Any], index: int, default: Any = None) -> Any:
    """Safely access a list element by index.
    
    Returns the element at the given index, or the default value
    if the index is out of bounds.
    
    Args:
        items: List to access.
        index: Index to retrieve.
        default: Value to return if index is invalid.
        
    Returns:
        Element at index or default value.
    """
    try:
        return items[index]
    except IndexError:
        return default


def safe_string_to_int(value: str, default: int = 0) -> int:
    """Convert string to integer safely.
    
    Attempts to convert the string to an integer. Returns the
    default value if conversion fails.
    
    Args:
        value: String to convert.
        default: Value to return on conversion failure.
        
    Returns:
        Converted integer or default value.
    """
    try:
        return int(value)
    except ValueError:
        return default


def divide_with_cleanup(
    numerator: float,
    denominator: float,
    *,
    log_operation: bool = False,
) -> float:
    """Perform division with guaranteed cleanup logging.
    
    Args:
        numerator: Dividend.
        denominator: Divisor.
        log_operation: If True, print operation details.
        
    Returns:
        Result of division.
        
    Raises:
        ZeroDivisionError: If denominator is zero.
    """
    try:
        if log_operation:
            print(f"Starting division: {numerator} / {denominator}")
        return numerator / denominator
    finally:
        print("Division operation complete")


def validate_positive_number(value: float, name: str = "value") -> float:
    """Validate that a number is positive.
    
    Args:
        value: Number to validate.
        name: Name of the value for error messages.
        
    Returns:
        The value if valid.
        
    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def read_config_value(
    config: dict[str, Any],
    key: str,
    expected_type: type,
) -> Any:
    """Read and validate a configuration value.
    
    Args:
        config: Configuration dictionary.
        key: Key to retrieve.
        expected_type: Expected type of the value.
        
    Returns:
        The configuration value.
        
    Raises:
        KeyError: If key is not in config.
        TypeError: If value is not of expected type.
    """
    if key not in config:
        raise KeyError(f"Config key '{key}' not found")
    
    value = config[key]
    if not isinstance(value, expected_type):
        actual_type = type(value).__name__
        raise TypeError(
            f"Config '{key}' expected {expected_type.__name__}, got {actual_type}"
        )
    
    return value


def main() -> None:
    """Verify solution implementations."""
    print("Testing safe_list_access...")
    assert safe_list_access([1, 2, 3], 0) == 1
    assert safe_list_access([1, 2, 3], 10) is None
    assert safe_list_access([1, 2, 3], -10, default="x") == "x"
    print("  ✓ safe_list_access passed")
    
    print("Testing safe_string_to_int...")
    assert safe_string_to_int("42") == 42
    assert safe_string_to_int("-17") == -17
    assert safe_string_to_int("abc") == 0
    assert safe_string_to_int("", default=-1) == -1
    print("  ✓ safe_string_to_int passed")
    
    print("Testing validate_positive_number...")
    assert validate_positive_number(5.0) == 5.0
    try:
        validate_positive_number(-3.0)
        assert False
    except ValueError as e:
        assert "positive" in str(e).lower()
    print("  ✓ validate_positive_number passed")
    
    print("Testing read_config_value...")
    assert read_config_value({"port": 8080}, "port", int) == 8080
    try:
        read_config_value({}, "missing", int)
        assert False
    except KeyError:
        pass
    try:
        read_config_value({"port": "8080"}, "port", int)
        assert False
    except TypeError:
        pass
    print("  ✓ read_config_value passed")
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
