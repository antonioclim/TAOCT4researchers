#!/usr/bin/env python3
"""Exercise 08: Validation Framework (Hard).

This exercise builds a composable validation framework for
scientific data processing.

Learning Objectives:
    - Design composable validators
    - Implement validation pipelines
    - Create clear error reporting

Estimated Time: 20 minutes
Difficulty: Hard (★★★)

Instructions:
    1. Implement base Validator protocol and concrete validators
    2. Build composite validators (And, Or, Not)
    3. Create validation pipeline
    4. Run tests to verify your implementation
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


# =============================================================================
# TASK 1: Validation Result
# =============================================================================

@dataclass
class ValidationError:
    """A single validation error.
    
    Attributes:
        path: Path to the invalid field (e.g., "user.address.postcode").
        message: Description of the error.
        code: Optional error code for programmatic handling.
        value: The invalid value.
    """
    
    path: str
    message: str
    code: str | None = None
    value: Any = None
    
    def __str__(self) -> str:
        """Return formatted error message."""
        if self.path:
            return f"{self.path}: {self.message}"
        return self.message


@dataclass
class ValidationResult:
    """Result of a validation operation.
    
    Attributes:
        errors: List of validation errors.
        
    Example:
        >>> result = ValidationResult()
        >>> result.is_valid
        True
        >>> result.add_error("age", "must be positive", value=-5)
        >>> result.is_valid
        False
    """
    
    errors: list[ValidationError] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Whether validation passed (no errors)."""
        # TODO: Return True if no errors
        pass
    
    def add_error(
        self,
        path: str,
        message: str,
        code: str | None = None,
        value: Any = None,
    ) -> None:
        """Add a validation error.
        
        Args:
            path: Field path.
            message: Error message.
            code: Optional error code.
            value: The invalid value.
        """
        # TODO: Create and append ValidationError
        pass
    
    def merge(self, other: ValidationResult) -> None:
        """Merge errors from another result.
        
        Args:
            other: Result to merge errors from.
        """
        # TODO: Extend errors with other.errors
        pass
    
    def with_prefix(self, prefix: str) -> ValidationResult:
        """Create new result with prefixed paths.
        
        Args:
            prefix: Prefix to add to all error paths.
            
        Returns:
            New ValidationResult with prefixed paths.
        """
        # TODO: Create new result with prefixed error paths
        # Path format: "{prefix}.{original_path}" or just "{prefix}" if no path
        pass
    
    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed.
        
        Raises:
            ValueError: With formatted error messages.
        """
        # TODO: Raise ValueError with all error messages
        pass


# =============================================================================
# TASK 2: Validator Protocol
# =============================================================================

class Validator(ABC, Generic[T]):
    """Abstract base class for validators.
    
    Validators check values and return ValidationResult.
    They can be composed using And, Or, Not operators.
    """
    
    @abstractmethod
    def validate(self, value: T, path: str = "") -> ValidationResult:
        """Validate a value.
        
        Args:
            value: Value to validate.
            path: Path for error messages.
            
        Returns:
            ValidationResult with any errors.
        """
        pass
    
    def __and__(self, other: Validator[T]) -> Validator[T]:
        """Combine with AND logic."""
        return AndValidator(self, other)
    
    def __or__(self, other: Validator[T]) -> Validator[T]:
        """Combine with OR logic."""
        return OrValidator(self, other)
    
    def __invert__(self) -> Validator[T]:
        """Negate validator."""
        return NotValidator(self)


# =============================================================================
# TASK 3: Concrete Validators
# =============================================================================

class RequiredValidator(Validator[Any]):
    """Validates that value is not None or empty."""
    
    def __init__(self, message: str = "This field is required") -> None:
        self.message = message
    
    def validate(self, value: Any, path: str = "") -> ValidationResult:
        """Check that value is present and not empty."""
        # TODO: Implement validation
        # - None is invalid
        # - Empty string (after strip) is invalid
        # - Empty list/dict is invalid
        pass


class TypeValidator(Validator[Any]):
    """Validates that value is of expected type."""
    
    def __init__(self, expected_type: type | tuple[type, ...], message: str | None = None) -> None:
        self.expected_type = expected_type
        self.message = message
    
    def validate(self, value: Any, path: str = "") -> ValidationResult:
        """Check that value is of expected type."""
        # TODO: Implement type checking using isinstance
        pass


class RangeValidator(Validator[int | float]):
    """Validates that numeric value is within range."""
    
    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        message: str | None = None,
    ) -> None:
        self.min_value = min_value
        self.max_value = max_value
        self.message = message
    
    def validate(self, value: int | float, path: str = "") -> ValidationResult:
        """Check that value is within range."""
        # TODO: Implement range validation
        pass


class PatternValidator(Validator[str]):
    """Validates that string matches pattern."""
    
    def __init__(self, pattern: str, message: str | None = None) -> None:
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.message = message or f"Must match pattern: {pattern}"
    
    def validate(self, value: str, path: str = "") -> ValidationResult:
        """Check that value matches pattern."""
        # TODO: Implement pattern matching
        pass


class LengthValidator(Validator[str | list[Any]]):
    """Validates length of strings or lists."""
    
    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        message: str | None = None,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.message = message
    
    def validate(self, value: str | list[Any], path: str = "") -> ValidationResult:
        """Check length constraints."""
        # TODO: Implement length validation
        pass


class CustomValidator(Validator[T]):
    """Validator using custom predicate function."""
    
    def __init__(
        self,
        predicate: Callable[[T], bool],
        message: str = "Validation failed",
        code: str | None = None,
    ) -> None:
        self.predicate = predicate
        self.message = message
        self.code = code
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        """Check custom predicate."""
        # TODO: Call predicate and return result
        pass


# =============================================================================
# TASK 4: Composite Validators
# =============================================================================

class AndValidator(Validator[T]):
    """Combines validators with AND logic (all must pass)."""
    
    def __init__(self, *validators: Validator[T]) -> None:
        self.validators = validators
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        """All validators must pass."""
        # TODO: Run all validators, collect all errors
        pass


class OrValidator(Validator[T]):
    """Combines validators with OR logic (at least one must pass)."""
    
    def __init__(self, *validators: Validator[T]) -> None:
        self.validators = validators
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        """At least one validator must pass."""
        # TODO: Return success if any validator passes
        # TODO: Return all errors if all fail
        pass


class NotValidator(Validator[T]):
    """Negates a validator."""
    
    def __init__(self, validator: Validator[T], message: str | None = None) -> None:
        self.validator = validator
        self.message = message or "Validation should have failed"
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        """Validator must fail for this to pass."""
        # TODO: Negate the result
        pass


# =============================================================================
# TASK 5: Schema Validator
# =============================================================================

class SchemaValidator(Validator[dict[str, Any]]):
    """Validates dictionary against a schema of field validators.
    
    Example:
        >>> schema = SchemaValidator({
        ...     "name": RequiredValidator() & LengthValidator(min_length=1),
        ...     "age": TypeValidator(int) & RangeValidator(min_value=0),
        ... })
        >>> result = schema.validate({"name": "Alice", "age": 30})
        >>> result.is_valid
        True
    """
    
    def __init__(
        self,
        schema: dict[str, Validator[Any]],
        allow_extra: bool = True,
    ) -> None:
        """Initialise schema validator.
        
        Args:
            schema: Mapping of field names to validators.
            allow_extra: Whether extra fields are allowed.
        """
        self.schema = schema
        self.allow_extra = allow_extra
    
    def validate(self, value: dict[str, Any], path: str = "") -> ValidationResult:
        """Validate dictionary against schema."""
        # TODO: Implement schema validation
        # 1. Check value is a dict
        # 2. Validate each field in schema
        # 3. Check for extra fields if not allowed
        # 4. Merge all results with proper paths
        pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
    print("Testing ValidationResult...")
    result = ValidationResult()
    assert result.is_valid
    result.add_error("field", "error message", value="bad")
    assert not result.is_valid
    assert len(result.errors) == 1
    print("  ✓ ValidationResult passed")
    
    print("Testing RequiredValidator...")
    v = RequiredValidator()
    assert v.validate("hello").is_valid
    assert not v.validate(None).is_valid
    assert not v.validate("").is_valid
    assert not v.validate([]).is_valid
    print("  ✓ RequiredValidator passed")
    
    print("Testing TypeValidator...")
    v = TypeValidator(int)
    assert v.validate(42).is_valid
    assert not v.validate("42").is_valid
    print("  ✓ TypeValidator passed")
    
    print("Testing RangeValidator...")
    v = RangeValidator(min_value=0, max_value=100)
    assert v.validate(50).is_valid
    assert not v.validate(-1).is_valid
    assert not v.validate(101).is_valid
    print("  ✓ RangeValidator passed")
    
    print("Testing PatternValidator...")
    v = PatternValidator(r"^[A-Z]{2}\d{4}$")
    assert v.validate("AB1234").is_valid
    assert not v.validate("invalid").is_valid
    print("  ✓ PatternValidator passed")
    
    print("Testing LengthValidator...")
    v = LengthValidator(min_length=2, max_length=5)
    assert v.validate("abc").is_valid
    assert not v.validate("a").is_valid
    assert not v.validate("toolong").is_valid
    print("  ✓ LengthValidator passed")
    
    print("Testing AndValidator...")
    v = RequiredValidator() & RangeValidator(min_value=0)
    assert v.validate(10).is_valid
    assert not v.validate(None).is_valid
    assert not v.validate(-1).is_valid
    print("  ✓ AndValidator passed")
    
    print("Testing OrValidator...")
    v = TypeValidator(int) | TypeValidator(str)
    assert v.validate(42).is_valid
    assert v.validate("hello").is_valid
    assert not v.validate([]).is_valid
    print("  ✓ OrValidator passed")
    
    print("Testing NotValidator...")
    v = ~PatternValidator(r"^test")
    assert v.validate("hello").is_valid
    assert not v.validate("test123").is_valid
    print("  ✓ NotValidator passed")
    
    print("Testing SchemaValidator...")
    schema = SchemaValidator({
        "name": RequiredValidator() & TypeValidator(str),
        "age": TypeValidator(int) & RangeValidator(min_value=0, max_value=150),
        "email": PatternValidator(r".+@.+\..+"),
    })
    
    valid_data = {"name": "Alice", "age": 30, "email": "alice@example.com"}
    assert schema.validate(valid_data).is_valid
    
    invalid_data = {"name": "", "age": -5, "email": "invalid"}
    result = schema.validate(invalid_data)
    assert not result.is_valid
    assert len(result.errors) >= 3
    print("  ✓ SchemaValidator passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
