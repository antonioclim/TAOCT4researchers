#!/usr/bin/env python3
"""Exercise 01: Basic Exception Handling (Easy).

This exercise introduces fundamental exception handling patterns
using try/except/finally blocks.

Learning Objectives:
    - Use try/except blocks to catch specific exceptions
    - Implement finally for cleanup operations
    - Raise exceptions with informative messages

Estimated Time: 10 minutes
Difficulty: Easy (★☆☆)

Instructions:
    1. Complete each TODO section
    2. Run the tests: pytest tests/test_exercises.py::test_e01
    3. Ensure all functions have proper type hints
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
        
    Example:
        >>> safe_list_access([1, 2, 3], 1)
        2
        >>> safe_list_access([1, 2, 3], 10, default="missing")
        'missing'
    """
    # TODO: Implement using try/except for IndexError
    # Your code here
    pass


def safe_string_to_int(value: str, default: int = 0) -> int:
    """Convert string to integer safely.
    
    Attempts to convert the string to an integer. Returns the
    default value if conversion fails.
    
    Args:
        value: String to convert.
        default: Value to return on conversion failure.
        
    Returns:
        Converted integer or default value.
        
    Example:
        >>> safe_string_to_int("42")
        42
        >>> safe_string_to_int("not_a_number", default=-1)
        -1
    """
    # TODO: Implement using try/except for ValueError
    # Your code here
    pass


def divide_with_cleanup(
    numerator: float,
    denominator: float,
    *,
    log_operation: bool = False,
) -> float:
    """Perform division with guaranteed cleanup logging.
    
    Divides numerator by denominator, always printing a completion
    message regardless of success or failure.
    
    Args:
        numerator: Dividend.
        denominator: Divisor.
        log_operation: If True, print operation details.
        
    Returns:
        Result of division.
        
    Raises:
        ZeroDivisionError: If denominator is zero.
        
    Example:
        >>> divide_with_cleanup(10, 2)
        5.0
    """
    # TODO: Implement using try/finally
    # - If log_operation is True, print "Starting division: {numerator} / {denominator}"
    # - In finally block, print "Division operation complete"
    # Your code here
    pass


def validate_positive_number(value: float, name: str = "value") -> float:
    """Validate that a number is positive.
    
    Raises a ValueError with informative message if the value
    is not positive (greater than zero).
    
    Args:
        value: Number to validate.
        name: Name of the value for error messages.
        
    Returns:
        The value if valid.
        
    Raises:
        ValueError: If value is not positive.
        
    Example:
        >>> validate_positive_number(5.0)
        5.0
        >>> validate_positive_number(-3.0, "temperature")
        Traceback (most recent call last):
            ...
        ValueError: temperature must be positive, got -3.0
    """
    # TODO: Implement validation with informative error message
    # Error message format: "{name} must be positive, got {value}"
    # Your code here
    pass


def read_config_value(
    config: dict[str, Any],
    key: str,
    expected_type: type,
) -> Any:
    """Read and validate a configuration value.
    
    Retrieves a value from the config dictionary, validating that
    it exists and has the expected type.
    
    Args:
        config: Configuration dictionary.
        key: Key to retrieve.
        expected_type: Expected type of the value.
        
    Returns:
        The configuration value.
        
    Raises:
        KeyError: If key is not in config.
        TypeError: If value is not of expected type.
        
    Example:
        >>> read_config_value({"port": 8080}, "port", int)
        8080
        >>> read_config_value({"port": "8080"}, "port", int)
        Traceback (most recent call last):
            ...
        TypeError: Config 'port' expected int, got str
    """
    # TODO: Implement with appropriate exception handling
    # - Raise KeyError if key not found (with message "Config key '{key}' not found")
    # - Raise TypeError if wrong type (with message "Config '{key}' expected {type}, got {actual}")
    # Your code here
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
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
    
    print("Testing divide_with_cleanup...")
    assert divide_with_cleanup(10, 2) == 5.0
    try:
        divide_with_cleanup(10, 0)
        assert False, "Should have raised ZeroDivisionError"
    except ZeroDivisionError:
        pass
    print("  ✓ divide_with_cleanup passed")
    
    print("Testing validate_positive_number...")
    assert validate_positive_number(5.0) == 5.0
    try:
        validate_positive_number(-3.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "positive" in str(e).lower()
    print("  ✓ validate_positive_number passed")
    
    print("Testing read_config_value...")
    assert read_config_value({"port": 8080}, "port", int) == 8080
    try:
        read_config_value({}, "missing", int)
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    try:
        read_config_value({"port": "8080"}, "port", int)
        assert False, "Should have raised TypeError"
    except TypeError:
        pass
    print("  ✓ read_config_value passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
