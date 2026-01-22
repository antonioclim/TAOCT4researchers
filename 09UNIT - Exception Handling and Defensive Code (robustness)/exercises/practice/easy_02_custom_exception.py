#!/usr/bin/env python3
"""Exercise 02: Custom Exception Classes (Easy).

This exercise introduces creating custom exception classes
for domain-specific error handling.

Learning Objectives:
    - Create custom exception classes inheriting from Exception
    - Add meaningful attributes to exceptions
    - Implement __str__ for informative error messages

Estimated Time: 10 minutes
Difficulty: Easy (★☆☆)

Instructions:
    1. Complete each TODO section
    2. Run the tests: pytest tests/test_exercises.py::test_e02
    3. Ensure all classes have proper docstrings
"""

from __future__ import annotations


# =============================================================================
# TASK 1: Create a simple custom exception
# =============================================================================

class TemperatureError(Exception):
    """Exception raised for invalid temperature values.
    
    Attributes:
        temperature: The invalid temperature value.
        message: Explanation of the error.
        
    Example:
        >>> raise TemperatureError(-500, "below absolute zero")
        Traceback (most recent call last):
            ...
        TemperatureError: Invalid temperature -500: below absolute zero
    """
    
    def __init__(self, temperature: float, message: str) -> None:
        """Initialise temperature error.
        
        Args:
            temperature: The invalid temperature value.
            message: Explanation of the error.
        """
        # TODO: Store attributes and call super().__init__()
        # Your code here
        pass
    
    def __str__(self) -> str:
        """Return formatted error message."""
        # TODO: Return "Invalid temperature {temperature}: {message}"
        # Your code here
        pass


# =============================================================================
# TASK 2: Create an exception with multiple attributes
# =============================================================================

class MeasurementError(Exception):
    """Exception raised for invalid measurements.
    
    Attributes:
        value: The measured value.
        unit: Unit of measurement.
        min_allowed: Minimum allowed value.
        max_allowed: Maximum allowed value.
        
    Example:
        >>> raise MeasurementError(150, "kg", 0, 100)
        Traceback (most recent call last):
            ...
        MeasurementError: Measurement 150 kg out of range [0, 100]
    """
    
    def __init__(
        self,
        value: float,
        unit: str,
        min_allowed: float,
        max_allowed: float,
    ) -> None:
        """Initialise measurement error.
        
        Args:
            value: The measured value.
            unit: Unit of measurement.
            min_allowed: Minimum allowed value.
            max_allowed: Maximum allowed value.
        """
        # TODO: Store all attributes and call super().__init__()
        # Your code here
        pass
    
    def __str__(self) -> str:
        """Return formatted error message."""
        # TODO: Return "Measurement {value} {unit} out of range [{min}, {max}]"
        # Your code here
        pass
    
    @property
    def is_too_low(self) -> bool:
        """Check if value is below minimum."""
        # TODO: Return True if value < min_allowed
        # Your code here
        pass
    
    @property
    def is_too_high(self) -> bool:
        """Check if value is above maximum."""
        # TODO: Return True if value > max_allowed
        # Your code here
        pass


# =============================================================================
# TASK 3: Use custom exceptions in validation functions
# =============================================================================

def validate_temperature_celsius(temp: float) -> float:
    """Validate a temperature in Celsius.
    
    Args:
        temp: Temperature value in Celsius.
        
    Returns:
        The temperature if valid.
        
    Raises:
        TemperatureError: If temperature is below absolute zero (-273.15°C)
                         or above 1000°C (arbitrary upper limit).
                         
    Example:
        >>> validate_temperature_celsius(25.0)
        25.0
        >>> validate_temperature_celsius(-300)
        Traceback (most recent call last):
            ...
        TemperatureError: Invalid temperature -300: below absolute zero (-273.15°C)
    """
    # TODO: Implement validation
    # - Raise TemperatureError if temp < -273.15 with message "below absolute zero (-273.15°C)"
    # - Raise TemperatureError if temp > 1000 with message "exceeds maximum (1000°C)"
    # - Return temp if valid
    # Your code here
    pass


def validate_measurement(
    value: float,
    unit: str,
    *,
    min_value: float = 0,
    max_value: float = 100,
) -> float:
    """Validate a measurement value against bounds.
    
    Args:
        value: Measurement value.
        unit: Unit of measurement.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.
        
    Returns:
        The value if valid.
        
    Raises:
        MeasurementError: If value is outside allowed range.
        
    Example:
        >>> validate_measurement(50, "percent")
        50
        >>> validate_measurement(150, "kg", max_value=100)
        Traceback (most recent call last):
            ...
        MeasurementError: Measurement 150 kg out of range [0, 100]
    """
    # TODO: Implement validation
    # - Raise MeasurementError if value < min_value or value > max_value
    # - Return value if valid
    # Your code here
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
    print("Testing TemperatureError...")
    try:
        raise TemperatureError(-500, "below absolute zero")
    except TemperatureError as e:
        assert e.temperature == -500
        assert "below absolute zero" in e.message
        assert "-500" in str(e)
    print("  ✓ TemperatureError passed")
    
    print("Testing MeasurementError...")
    try:
        raise MeasurementError(150, "kg", 0, 100)
    except MeasurementError as e:
        assert e.value == 150
        assert e.unit == "kg"
        assert e.is_too_high
        assert not e.is_too_low
        assert "150" in str(e)
        assert "kg" in str(e)
    print("  ✓ MeasurementError passed")
    
    print("Testing validate_temperature_celsius...")
    assert validate_temperature_celsius(25.0) == 25.0
    assert validate_temperature_celsius(-273.15) == -273.15
    try:
        validate_temperature_celsius(-300)
        assert False, "Should have raised TemperatureError"
    except TemperatureError as e:
        assert e.temperature == -300
    print("  ✓ validate_temperature_celsius passed")
    
    print("Testing validate_measurement...")
    assert validate_measurement(50, "percent") == 50
    try:
        validate_measurement(150, "kg", max_value=100)
        assert False, "Should have raised MeasurementError"
    except MeasurementError as e:
        assert e.value == 150
        assert e.is_too_high
    print("  ✓ validate_measurement passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
