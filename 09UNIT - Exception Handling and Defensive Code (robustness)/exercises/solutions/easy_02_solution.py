#!/usr/bin/env python3
"""Solution for Exercise 02: Custom Exception Classes.

This module provides reference implementations for custom
exception classes and validation functions.
"""

from __future__ import annotations


class TemperatureError(Exception):
    """Exception raised for invalid temperature values."""
    
    def __init__(self, temperature: float, message: str) -> None:
        self.temperature = temperature
        self.message = message
        super().__init__(f"Invalid temperature {temperature}: {message}")
    
    def __str__(self) -> str:
        return f"Invalid temperature {self.temperature}: {self.message}"


class MeasurementError(Exception):
    """Exception raised for invalid measurements."""
    
    def __init__(
        self,
        value: float,
        unit: str,
        min_allowed: float,
        max_allowed: float,
    ) -> None:
        self.value = value
        self.unit = unit
        self.min_allowed = min_allowed
        self.max_allowed = max_allowed
        super().__init__(
            f"Measurement {value} {unit} out of range [{min_allowed}, {max_allowed}]"
        )
    
    def __str__(self) -> str:
        return (
            f"Measurement {self.value} {self.unit} "
            f"out of range [{self.min_allowed}, {self.max_allowed}]"
        )
    
    @property
    def is_too_low(self) -> bool:
        return self.value < self.min_allowed
    
    @property
    def is_too_high(self) -> bool:
        return self.value > self.max_allowed


def validate_temperature_celsius(temp: float) -> float:
    """Validate a temperature in Celsius.
    
    Args:
        temp: Temperature value in Celsius.
        
    Returns:
        The temperature if valid.
        
    Raises:
        TemperatureError: If temperature is outside valid range.
    """
    if temp < -273.15:
        raise TemperatureError(temp, "below absolute zero (-273.15°C)")
    if temp > 1000:
        raise TemperatureError(temp, "exceeds maximum (1000°C)")
    return temp


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
    """
    if value < min_value or value > max_value:
        raise MeasurementError(value, unit, min_value, max_value)
    return value


def main() -> None:
    """Verify solution implementations."""
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
    print("  ✓ MeasurementError passed")
    
    print("Testing validate_temperature_celsius...")
    assert validate_temperature_celsius(25.0) == 25.0
    assert validate_temperature_celsius(-273.15) == -273.15
    try:
        validate_temperature_celsius(-300)
        assert False
    except TemperatureError as e:
        assert e.temperature == -300
    print("  ✓ validate_temperature_celsius passed")
    
    print("Testing validate_measurement...")
    assert validate_measurement(50, "percent") == 50
    try:
        validate_measurement(150, "kg", max_value=100)
        assert False
    except MeasurementError as e:
        assert e.value == 150
        assert e.is_too_high
    print("  ✓ validate_measurement passed")
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
