#!/usr/bin/env python3
"""Solutions for Lab 09-02: Defensive Programming.

Complete reference implementations for all laboratory exercises.
These solutions demonstrate established conventions for defensive programming
in Python scientific computing applications.

IMPORTANT: This file is for instructor reference only.
Students should attempt exercises independently before consulting.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generator, ParamSpec, Sequence, TypeVar

# Import exceptions from lab 01
from lab.lab_09_01_exception_handling import ContractViolationError

# Configure module logger
logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# §1: DESIGN BY CONTRACT SOLUTIONS
# =============================================================================


def precondition(
    condition: Callable[..., bool],
    message: str = "Precondition failed",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that enforces a precondition on function arguments.
    
    The condition function receives the same arguments as the decorated
    function and should return True if the precondition is satisfied.
    
    Args:
        condition: Function that checks arguments.
        message: Error message on violation.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not condition(*args, **kwargs):
                raise ContractViolationError(
                    f"Precondition failed for {func.__name__}: {message}",
                    "precondition",
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def postcondition(
    condition: Callable[[T], bool],
    message: str = "Postcondition failed",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that enforces a postcondition on function result.
    
    The condition function receives the return value and should
    return True if the postcondition is satisfied.
    
    Args:
        condition: Function that checks result.
        message: Error message on violation.
        
    Returns:
        Decorator function.
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)
            if not condition(result):
                raise ContractViolationError(
                    f"Postcondition failed for {func.__name__}: {message}",
                    "postcondition",
                )
            return result
        return wrapper
    return decorator


def invariant(
    condition: Callable[[Any], bool],
    message: str = "Invariant violated",
) -> Callable[[type[T]], type[T]]:
    """Class decorator that enforces an invariant after each method call.
    
    The condition function receives self and should return True
    if the invariant is maintained.
    
    Args:
        condition: Function that checks object state.
        message: Error message on violation.
        
    Returns:
        Class decorator.
    """
    def decorator(cls: type[T]) -> type[T]:
        original_init = cls.__init__
        
        def check_invariant(self: Any) -> None:
            if not condition(self):
                raise ContractViolationError(
                    f"Invariant violated for {cls.__name__}: {message}",
                    "invariant",
                )
        
        def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            check_invariant(self)
        
        cls.__init__ = wrapped_init  # type: ignore[method-assign]
        
        # Wrap all public methods
        for name in dir(cls):
            if not name.startswith("_"):
                attr = getattr(cls, name)
                if callable(attr) and not isinstance(attr, type):
                    def make_wrapper(method: Callable[..., Any]) -> Callable[..., Any]:
                        @wraps(method)
                        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                            result = method(self, *args, **kwargs)
                            check_invariant(self)
                            return result
                        return wrapper
                    setattr(cls, name, make_wrapper(attr))
        
        return cls
    return decorator


# =============================================================================
# §2: INPUT VALIDATION SOLUTIONS
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a validation operation.
    
    Attributes:
        is_valid: Whether validation passed.
        errors: List of error messages.
        warnings: List of warning messages.
    """
    
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        """Add an error and mark as invalid."""
        self.is_valid = False
        self.errors.append(message)
    
    def add_warning(self, message: str) -> None:
        """Add a warning without affecting validity."""
        self.warnings.append(message)
    
    def merge(self, other: ValidationResult) -> ValidationResult:
        """Merge another result into this one."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
        )


def validate_numeric_range(
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    field_name: str = "value",
) -> ValidationResult:
    """Validate that a numeric value is within a specified range.
    
    Args:
        value: Value to validate.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        field_name: Name for error messages.
        
    Returns:
        ValidationResult with any errors.
    """
    result = ValidationResult()
    
    # Check for NaN
    if math.isnan(value):
        result.add_error(f"{field_name} is NaN (not a number)")
        return result
    
    # Check for infinity
    if math.isinf(value):
        result.add_error(f"{field_name} is infinite")
        return result
    
    # Check minimum
    if min_value is not None and value < min_value:
        result.add_error(
            f"{field_name} ({value}) is below minimum ({min_value})"
        )
    
    # Check maximum
    if max_value is not None and value > max_value:
        result.add_error(
            f"{field_name} ({value}) is above maximum ({max_value})"
        )
    
    return result


def validate_string_pattern(
    value: str,
    *,
    pattern: str,
    field_name: str = "value",
) -> ValidationResult:
    """Validate that a string matches a regex pattern.
    
    Args:
        value: String to validate.
        pattern: Regex pattern to match.
        field_name: Name for error messages.
        
    Returns:
        ValidationResult with any errors.
    """
    import re
    
    result = ValidationResult()
    
    if not re.match(pattern, value):
        result.add_error(
            f"{field_name} '{value}' does not match pattern '{pattern}'"
        )
    
    return result


def validate_collection(
    items: Sequence[T],
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    element_validator: Callable[[T], bool] | None = None,
    field_name: str = "collection",
) -> ValidationResult:
    """Validate a collection's length and elements.
    
    Args:
        items: Collection to validate.
        min_length: Minimum required length.
        max_length: Maximum allowed length.
        element_validator: Function to validate each element.
        field_name: Name for error messages.
        
    Returns:
        ValidationResult with any errors.
    """
    result = ValidationResult()
    length = len(items)
    
    # Check length constraints
    if min_length is not None and length < min_length:
        result.add_error(
            f"{field_name} length ({length}) is below minimum ({min_length})"
        )
    
    if max_length is not None and length > max_length:
        result.add_error(
            f"{field_name} length ({length}) exceeds maximum ({max_length})"
        )
    
    # Validate elements
    if element_validator is not None:
        for i, item in enumerate(items):
            if not element_validator(item):
                result.add_error(
                    f"{field_name}[{i}] failed element validation"
                )
    
    return result


def validate_dataframe_schema(
    df: Any,
    schema: dict[str, type],
) -> ValidationResult:
    """Validate pandas DataFrame against expected schema.
    
    Args:
        df: DataFrame to validate.
        schema: Expected column names and types.
        
    Returns:
        ValidationResult with any errors.
    """
    result = ValidationResult()
    
    # Check for missing columns
    missing = set(schema.keys()) - set(df.columns)
    if missing:
        result.add_error(f"Missing columns: {missing}")
    
    # Check column types (for existing columns)
    for col, expected_type in schema.items():
        if col in df.columns:
            # Check if dtype is compatible
            dtype = df[col].dtype
            if expected_type == str and dtype != object:
                result.add_error(f"Column '{col}' expected string, got {dtype}")
            elif expected_type == int and not str(dtype).startswith("int"):
                result.add_error(f"Column '{col}' expected int, got {dtype}")
            elif expected_type == float and not str(dtype).startswith("float"):
                result.add_warning(f"Column '{col}' expected float, got {dtype}")
    
    return result


# =============================================================================
# §3: NUMERICAL resilience SOLUTIONS
# =============================================================================


def safe_float_comparison(
    a: float,
    b: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    """Compare floating-point numbers with tolerance.
    
    Uses both relative and absolute tolerance for resilient comparison
    across different magnitudes.
    
    Args:
        a: First value.
        b: Second value.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.
        
    Returns:
        True if values are approximately equal.
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def detect_numerical_instability(
    values: Sequence[float],
    *,
    overflow_threshold: float = 1e308,
    underflow_threshold: float = 1e-308,
) -> dict[str, Any]:
    """Detect potential numerical instability in a sequence of values.
    
    Args:
        values: Sequence of values to check.
        overflow_threshold: Value magnitude indicating overflow risk.
        underflow_threshold: Value magnitude indicating underflow risk.
        
    Returns:
        Dictionary with stability assessment.
    """
    result: dict[str, Any] = {
        "is_stable": True,
        "has_nan": False,
        "has_inf": False,
        "overflow_risk": False,
        "underflow_risk": False,
        "warnings": [],
    }
    
    for i, v in enumerate(values):
        if math.isnan(v):
            result["is_stable"] = False
            result["has_nan"] = True
            result["warnings"].append(f"NaN at index {i}")
        elif math.isinf(v):
            result["is_stable"] = False
            result["has_inf"] = True
            result["warnings"].append(f"Infinity at index {i}")
        elif abs(v) > overflow_threshold:
            result["overflow_risk"] = True
            result["warnings"].append(f"Overflow risk at index {i}: {v}")
        elif 0 < abs(v) < underflow_threshold:
            result["underflow_risk"] = True
            result["warnings"].append(f"Underflow risk at index {i}: {v}")
    
    return result


def kahan_summation(values: Sequence[float]) -> float:
    """Sum values using Kahan summation algorithm.
    
    Compensated summation that reduces numerical error when
    summing many floating-point numbers.
    
    Args:
        values: Values to sum.
        
    Returns:
        Sum with reduced numerical error.
    """
    total = 0.0
    compensation = 0.0
    
    for value in values:
        y = value - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    
    return total


def stable_mean(values: Sequence[float]) -> float:
    """Calculate mean using numerically stable algorithm.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        values: Values to average.
        
    Returns:
        Mean of values.
        
    Raises:
        ValueError: If values is empty.
    """
    if not values:
        raise ValueError("Cannot compute mean of empty sequence")
    
    mean = 0.0
    count = 0
    
    for value in values:
        count += 1
        delta = value - mean
        mean += delta / count
    
    return mean


def stable_variance(
    values: Sequence[float],
    *,
    ddof: int = 0,
) -> float:
    """Calculate variance using numerically stable algorithm.
    
    Uses Welford's online algorithm for numerical stability.
    
    Args:
        values: Values to compute variance for.
        ddof: Delta degrees of freedom (0 for population, 1 for sample).
        
    Returns:
        Variance of values.
        
    Raises:
        ValueError: If values is empty or has insufficient elements.
    """
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute variance of empty sequence")
    if n <= ddof:
        raise ValueError(f"Need at least {ddof + 1} values for ddof={ddof}")
    
    mean = 0.0
    m2 = 0.0
    count = 0
    
    for value in values:
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        m2 += delta * delta2
    
    return m2 / (count - ddof)


# =============================================================================
# §4: DEFENSIVE DATA PROCESSING SOLUTIONS
# =============================================================================


def safe_json_load(
    path: Path,
    *,
    default: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Safely load JSON file with error handling.
    
    Args:
        path: Path to JSON file.
        default: Default value on error.
        
    Returns:
        Parsed JSON or default value.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("JSON file not found: %s", path)
        return default
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", path, e)
        return default
    except PermissionError:
        logger.error("Permission denied reading: %s", path)
        return default


def resilient_csv_reader(
    path: Path,
    *,
    skip_errors: bool = True,
    encoding: str = "utf-8",
) -> Generator[dict[str, str], None, None]:
    """Read CSV file with reliable error handling.
    
    Args:
        path: Path to CSV file.
        skip_errors: Skip malformed rows instead of failing.
        encoding: File encoding.
        
    Yields:
        Dictionary for each row.
    """
    try:
        with open(path, encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            
            for line_num, row in enumerate(reader, start=2):
                try:
                    # Validate row has expected number of fields
                    if None in row.values():
                        if skip_errors:
                            logger.warning(
                                "Malformed row at line %d in %s",
                                line_num,
                                path,
                            )
                            continue
                        else:
                            raise ValueError(f"Malformed row at line {line_num}")
                    
                    yield row
                    
                except Exception as e:
                    if skip_errors:
                        logger.warning(
                            "Error processing line %d in %s: %s",
                            line_num,
                            path,
                            e,
                        )
                    else:
                        raise
                        
    except FileNotFoundError:
        logger.error("CSV file not found: %s", path)
    except PermissionError:
        logger.error("Permission denied reading: %s", path)


class CheckpointManager:
    """Manages checkpoint persistence for resumable computations.
    
    Provides atomic save/restore operations for computation state.
    
    Attributes:
        checkpoint_path: Path to checkpoint file.
        state: Current state dictionary.
    """
    
    def __init__(self, checkpoint_path: Path) -> None:
        """Initialise checkpoint manager.
        
        Args:
            checkpoint_path: Path for checkpoint file.
        """
        self.checkpoint_path = checkpoint_path
        self.state: dict[str, Any] = {}
    
    def save(self) -> None:
        """Save checkpoint atomically."""
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        
        try:
            with open(temp_path, "w") as f:
                json.dump(self.state, f, indent=2)
            
            temp_path.replace(self.checkpoint_path)
            logger.debug("Checkpoint saved: %s", self.checkpoint_path)
            
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def load(self) -> bool:
        """Load checkpoint from file.
        
        Returns:
            True if checkpoint was loaded, False if not found.
        """
        if not self.checkpoint_path.exists():
            logger.debug("No checkpoint found at %s", self.checkpoint_path)
            return False
        
        try:
            with open(self.checkpoint_path) as f:
                self.state = json.load(f)
            
            logger.info("Checkpoint loaded: %s", self.checkpoint_path)
            return True
            
        except json.JSONDecodeError as e:
            logger.error("Corrupted checkpoint: %s", e)
            return False


def checkpoint_processor(
    items: list[T],
    processor: Callable[[T], Any],
    *,
    checkpoint_path: Path,
    checkpoint_interval: int = 100,
) -> list[Any]:
    """Process items with checkpoint-based recovery.
    
    Args:
        items: Items to process.
        processor: Function to process each item.
        checkpoint_path: Path for checkpoint file.
        checkpoint_interval: Items between checkpoints.
        
    Returns:
        List of processed results.
    """
    manager = CheckpointManager(checkpoint_path)
    
    # Try to resume from checkpoint
    if manager.load():
        start_index = manager.state.get("last_completed", -1) + 1
        results = manager.state.get("results", [])
        logger.info("Resuming from index %d", start_index)
    else:
        start_index = 0
        results = []
    
    try:
        for i in range(start_index, len(items)):
            result = processor(items[i])
            results.append(result)
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                manager.state = {
                    "last_completed": i,
                    "results": results,
                }
                manager.save()
                logger.debug("Checkpoint at index %d", i)
        
        # Final checkpoint
        manager.state = {
            "last_completed": len(items) - 1,
            "results": results,
            "completed": True,
        }
        manager.save()
        
        return results
        
    except Exception as e:
        # Save progress on failure
        manager.state = {
            "last_completed": start_index + len(results) - 1,
            "results": results,
            "error": str(e),
        }
        manager.save()
        raise
