#!/usr/bin/env python3
"""Solution for Exercise 08: Validation Framework.

This module provides reference implementations for a composable
validation framework.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass
class ValidationError:
    """A single validation error."""
    
    path: str
    message: str
    code: str | None = None
    value: Any = None
    
    def __str__(self) -> str:
        if self.path:
            return f"{self.path}: {self.message}"
        return self.message


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    
    errors: list[ValidationError] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0
    
    def add_error(
        self,
        path: str,
        message: str,
        code: str | None = None,
        value: Any = None,
    ) -> None:
        self.errors.append(ValidationError(path, message, code, value))
    
    def merge(self, other: ValidationResult) -> None:
        self.errors.extend(other.errors)
    
    def with_prefix(self, prefix: str) -> ValidationResult:
        result = ValidationResult()
        for error in self.errors:
            new_path = f"{prefix}.{error.path}" if error.path else prefix
            result.errors.append(
                ValidationError(new_path, error.message, error.code, error.value)
            )
        return result
    
    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            messages = [str(e) for e in self.errors]
            raise ValueError(f"Validation failed: {'; '.join(messages)}")


class Validator(ABC, Generic[T]):
    """Abstract base class for validators."""
    
    @abstractmethod
    def validate(self, value: T, path: str = "") -> ValidationResult:
        pass
    
    def __and__(self, other: Validator[T]) -> Validator[T]:
        return AndValidator(self, other)
    
    def __or__(self, other: Validator[T]) -> Validator[T]:
        return OrValidator(self, other)
    
    def __invert__(self) -> Validator[T]:
        return NotValidator(self)


class RequiredValidator(Validator[Any]):
    """Validates that value is not None or empty."""
    
    def __init__(self, message: str = "This field is required") -> None:
        self.message = message
    
    def validate(self, value: Any, path: str = "") -> ValidationResult:
        result = ValidationResult()
        
        if value is None:
            result.add_error(path, self.message, "required", value)
        elif isinstance(value, str) and not value.strip():
            result.add_error(path, self.message, "required", value)
        elif isinstance(value, (list, dict)) and len(value) == 0:
            result.add_error(path, self.message, "required", value)
        
        return result


class TypeValidator(Validator[Any]):
    """Validates that value is of expected type."""
    
    def __init__(
        self,
        expected_type: type | tuple[type, ...],
        message: str | None = None,
    ) -> None:
        self.expected_type = expected_type
        self.message = message
    
    def validate(self, value: Any, path: str = "") -> ValidationResult:
        result = ValidationResult()
        
        if not isinstance(value, self.expected_type):
            if isinstance(self.expected_type, tuple):
                type_names = ", ".join(t.__name__ for t in self.expected_type)
            else:
                type_names = self.expected_type.__name__
            
            msg = self.message or f"Expected {type_names}, got {type(value).__name__}"
            result.add_error(path, msg, "type", value)
        
        return result


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
        result = ValidationResult()
        
        if self.min_value is not None and value < self.min_value:
            msg = self.message or f"Value must be >= {self.min_value}"
            result.add_error(path, msg, "range_min", value)
        
        if self.max_value is not None and value > self.max_value:
            msg = self.message or f"Value must be <= {self.max_value}"
            result.add_error(path, msg, "range_max", value)
        
        return result


class PatternValidator(Validator[str]):
    """Validates that string matches pattern."""
    
    def __init__(self, pattern: str, message: str | None = None) -> None:
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.message = message or f"Must match pattern: {pattern}"
    
    def validate(self, value: str, path: str = "") -> ValidationResult:
        result = ValidationResult()
        
        if not self.compiled.match(value):
            result.add_error(path, self.message, "pattern", value)
        
        return result


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
        result = ValidationResult()
        length = len(value)
        
        if self.min_length is not None and length < self.min_length:
            msg = self.message or f"Length must be >= {self.min_length}"
            result.add_error(path, msg, "length_min", value)
        
        if self.max_length is not None and length > self.max_length:
            msg = self.message or f"Length must be <= {self.max_length}"
            result.add_error(path, msg, "length_max", value)
        
        return result


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
        result = ValidationResult()
        
        if not self.predicate(value):
            result.add_error(path, self.message, self.code, value)
        
        return result


class AndValidator(Validator[T]):
    """Combines validators with AND logic."""
    
    def __init__(self, *validators: Validator[T]) -> None:
        self.validators = validators
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        result = ValidationResult()
        
        for validator in self.validators:
            result.merge(validator.validate(value, path))
        
        return result


class OrValidator(Validator[T]):
    """Combines validators with OR logic."""
    
    def __init__(self, *validators: Validator[T]) -> None:
        self.validators = validators
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        all_errors: list[ValidationError] = []
        
        for validator in self.validators:
            result = validator.validate(value, path)
            if result.is_valid:
                return ValidationResult()  # Success
            all_errors.extend(result.errors)
        
        # All validators failed
        final_result = ValidationResult()
        final_result.errors = all_errors
        return final_result


class NotValidator(Validator[T]):
    """Negates a validator."""
    
    def __init__(
        self,
        validator: Validator[T],
        message: str | None = None,
    ) -> None:
        self.validator = validator
        self.message = message or "Validation should have failed"
    
    def validate(self, value: T, path: str = "") -> ValidationResult:
        result = self.validator.validate(value, path)
        
        if result.is_valid:
            # Original passed, so negation fails
            neg_result = ValidationResult()
            neg_result.add_error(path, self.message, "not", value)
            return neg_result
        
        return ValidationResult()  # Original failed, so negation passes


class SchemaValidator(Validator[dict[str, Any]]):
    """Validates dictionary against a schema."""
    
    def __init__(
        self,
        schema: dict[str, Validator[Any]],
        allow_extra: bool = True,
    ) -> None:
        self.schema = schema
        self.allow_extra = allow_extra
    
    def validate(self, value: dict[str, Any], path: str = "") -> ValidationResult:
        result = ValidationResult()
        
        if not isinstance(value, dict):
            result.add_error(path, "Expected a dictionary", "type", value)
            return result
        
        for field_name, validator in self.schema.items():
            field_path = f"{path}.{field_name}" if path else field_name
            
            if field_name in value:
                field_result = validator.validate(value[field_name], field_path)
                result.merge(field_result)
            else:
                # Field missing - check if required
                req_result = RequiredValidator().validate(None, field_path)
                if not req_result.is_valid:
                    # Only add error if the validator would have failed on None
                    pass  # Optional field
        
        if not self.allow_extra:
            extra_keys = set(value.keys()) - set(self.schema.keys())
            for key in extra_keys:
                key_path = f"{path}.{key}" if path else key
                result.add_error(key_path, "Extra field not allowed", "extra", value[key])
        
        return result


def main() -> None:
    """Verify solution implementations."""
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
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
