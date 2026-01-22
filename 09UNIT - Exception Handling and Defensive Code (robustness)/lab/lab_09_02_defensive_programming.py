#!/usr/bin/env python3
"""09UNIT Lab 02: Defensive Programming.

This laboratory module explores defensive programming techniques that
create consistent, failure-resistant code for scientific computing applications.

Scientific computing often involves complex data transformations where errors
can be subtle and difficult to detect. Defensive programming—anticipating and
handling potential errors and edge cases—helps create more consistent functions
that fail gracefully when presented with unexpected inputs. This approach is
particularly valuable in research contexts, where data may contain anomalies
or where models may be applied to novel situations (Downey, 2015, p. 126).

Research in software engineering consistently shows that modular designs with
high cohesion (functions focused on single tasks) and low coupling (minimal
dependencies between functions) lead to more maintainable, consistent code
(Yourdon & Constantine, 1979, p. 85).

Sections:
    §1. Design by Contract
    §2. Input Validation Patterns
    §3. Numerical resilience
    §4. Defensive Data Processing

Author: Course Team
Version: 1.0.0
"""

from __future__ import annotations

import csv
import json
import logging
import math
import pickle
import re
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, ParamSpec, Sequence, TypeVar

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables
T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# §1. DESIGN BY CONTRACT
# =============================================================================


class ContractViolationError(Exception):
    """Raised when a design-by-contract condition is violated.
    
    Attributes:
        message: Human-readable error description.
        contract_type: Type of contract (precondition, postcondition, invariant).
        condition: The violated condition description.
    """
    
    def __init__(
        self,
        message: str,
        contract_type: str | None = None,
        condition: str | None = None,
    ) -> None:
        """Initialise contract violation error.
        
        Args:
            message: Human-readable error description.
            contract_type: Type of violated contract.
            condition: Description of the violated condition.
        """
        super().__init__(message)
        self.message = message
        self.contract_type = contract_type
        self.condition = condition


def precondition(
    pred: Callable[..., bool],
    message: str = "Precondition violated",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator enforcing a precondition on function arguments.
    
    The predicate receives the same arguments as the decorated function.
    If the predicate returns False, ContractViolationError is raised
    before the function executes.
    
    Args:
        pred: Predicate function testing precondition.
        message: Error message on violation.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @precondition(lambda x: x >= 0, "x must be non-negative")
        ... def sqrt_safe(x: float) -> float:
        ...     return math.sqrt(x)
        >>> sqrt_safe(4)
        2.0
        >>> sqrt_safe(-1)
        Traceback (most recent call last):
            ...
        ContractViolationError: x must be non-negative
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not pred(*args, **kwargs):
                logger.error(
                    "Precondition violated for %s: %s",
                    func.__name__,
                    message,
                )
                raise ContractViolationError(
                    message,
                    contract_type="precondition",
                    condition=str(pred),
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def postcondition(
    pred: Callable[[T], bool],
    message: str = "Postcondition violated",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator enforcing a postcondition on function return value.
    
    The predicate receives the return value. If it returns False,
    ContractViolationError is raised after function execution.
    
    Args:
        pred: Predicate function testing postcondition.
        message: Error message on violation.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @postcondition(lambda r: r >= 0, "result must be non-negative")
        ... def abs_value(x: float) -> float:
        ...     return abs(x)
        >>> abs_value(-5)
        5
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)
            if not pred(result):
                logger.error(
                    "Postcondition violated for %s: %s (result=%s)",
                    func.__name__,
                    message,
                    result,
                )
                raise ContractViolationError(
                    message,
                    contract_type="postcondition",
                    condition=str(pred),
                )
            return result
        return wrapper
    return decorator


def invariant(
    pred: Callable[[Any], bool],
    message: str = "Invariant violated",
) -> Callable[[type[T]], type[T]]:
    """Class decorator enforcing an invariant after public method calls.
    
    Wraps all public methods (not starting with _) to check the
    invariant predicate on self after each call.
    
    Args:
        pred: Predicate function testing invariant on self.
        message: Error message on violation.
        
    Returns:
        Class decorator.
        
    Example:
        >>> @invariant(lambda self: self.balance >= 0, "balance must be non-negative")
        ... class BankAccount:
        ...     def __init__(self, initial: float) -> None:
        ...         self.balance = initial
        ...     def withdraw(self, amount: float) -> None:
        ...         self.balance -= amount
    """
    def class_decorator(cls: type[T]) -> type[T]:
        # Wrap each public method
        for name in dir(cls):
            if name.startswith("_"):
                continue
            
            attr = getattr(cls, name)
            if not callable(attr):
                continue
            
            # Create wrapper that checks invariant
            original_method = attr
            
            def make_wrapper(method: Callable[..., Any]) -> Callable[..., Any]:
                @wraps(method)
                def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                    result = method(self, *args, **kwargs)
                    if not pred(self):
                        logger.error(
                            "Invariant violated after %s.%s: %s",
                            cls.__name__,
                            method.__name__,
                            message,
                        )
                        raise ContractViolationError(
                            message,
                            contract_type="invariant",
                            condition=str(pred),
                        )
                    return result
                return wrapper
            
            setattr(cls, name, make_wrapper(original_method))
        
        return cls
    return class_decorator


# =============================================================================
# §2. INPUT VALIDATION PATTERNS
# =============================================================================


@dataclass
class ValidationResult:
    """Result of a validation operation.
    
    Provides structured feedback about validation outcomes including
    whether validation passed, error messages and warnings.
    
    Attributes:
        is_valid: Whether validation passed.
        errors: List of error messages for failures.
        warnings: List of warning messages (non-fatal issues).
        
    Example:
        >>> result = ValidationResult(is_valid=False, errors=["Value out of range"])
        >>> result.is_valid
        False
        >>> result.raise_if_invalid()
        Traceback (most recent call last):
            ...
        ValueError: Validation failed: Value out of range
    """
    
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed.
        
        Raises:
            ValueError: If is_valid is False.
        """
        if not self.is_valid:
            error_msg = "; ".join(self.errors) if self.errors else "Unknown error"
            raise ValueError(f"Validation failed: {error_msg}")
    
    def __bool__(self) -> bool:
        """Allow use in boolean context."""
        return self.is_valid
    
    @classmethod
    def success(cls, warnings: list[str] | None = None) -> ValidationResult:
        """Create a successful validation result.
        
        Args:
            warnings: Optional warning messages.
            
        Returns:
            Successful ValidationResult.
        """
        return cls(is_valid=True, warnings=warnings or [])
    
    @classmethod
    def failure(cls, *errors: str) -> ValidationResult:
        """Create a failed validation result.
        
        Args:
            *errors: Error messages.
            
        Returns:
            Failed ValidationResult.
        """
        return cls(is_valid=False, errors=list(errors))


def validate_numeric_range(
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
    label: str = "value",
) -> ValidationResult:
    """Validate a numeric value against range constraints.
    
    For scientific computing, error handling extends beyond technical
    correctness to domain-specific validity. Functions should check not
    just that inputs are well-formed but that they are meaningful in the
    relevant scientific context (Downey, 2015, p. 127).
    
    Args:
        value: Numeric value to validate.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).
        allow_nan: Whether NaN values are permitted.
        allow_inf: Whether infinite values are permitted.
        label: Label for error messages.
        
    Returns:
        ValidationResult indicating success or failure.
        
    Example:
        >>> result = validate_numeric_range(5.0, min_value=0, max_value=10)
        >>> result.is_valid
        True
        >>> result = validate_numeric_range(-5.0, min_value=0)
        >>> result.is_valid
        False
        >>> result.errors[0]
        'value (-5.0) is below minimum (0)'
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    # Check for special values
    if math.isnan(value):
        if not allow_nan:
            errors.append(f"{label} is NaN (not a number)")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    if math.isinf(value):
        if not allow_inf:
            errors.append(f"{label} is infinite")
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    # Range validation
    if min_value is not None and value < min_value:
        errors.append(f"{label} ({value}) is below minimum ({min_value})")
    
    if max_value is not None and value > max_value:
        errors.append(f"{label} ({value}) is above maximum ({max_value})")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_string_pattern(
    value: str,
    pattern: str,
    *,
    label: str = "value",
    pattern_description: str | None = None,
) -> ValidationResult:
    """Validate a string against a regular expression pattern.
    
    Args:
        value: String to validate.
        pattern: Regular expression pattern.
        label: Label for error messages.
        pattern_description: Human-readable pattern description.
        
    Returns:
        ValidationResult indicating success or failure.
        
    Example:
        >>> result = validate_string_pattern("ABC123", r"^[A-Z]+[0-9]+$")
        >>> result.is_valid
        True
        >>> result = validate_string_pattern("abc", r"^[A-Z]+$")
        >>> result.is_valid
        False
    """
    if re.match(pattern, value):
        return ValidationResult.success()
    
    desc = pattern_description or f"pattern '{pattern}'"
    return ValidationResult.failure(
        f"{label} '{value}' does not match {desc}"
    )


def validate_collection(
    values: Sequence[Any],
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    allow_empty: bool = True,
    unique: bool = False,
    label: str = "collection",
) -> ValidationResult:
    """Validate collection size and uniqueness constraints.
    
    Args:
        values: Collection to validate.
        min_length: Minimum required length.
        max_length: Maximum allowed length.
        allow_empty: Whether empty collections are permitted.
        unique: Whether all elements must be unique.
        label: Label for error messages.
        
    Returns:
        ValidationResult indicating success or failure.
        
    Example:
        >>> result = validate_collection([1, 2, 3], min_length=2)
        >>> result.is_valid
        True
        >>> result = validate_collection([1, 1, 2], unique=True)
        >>> result.is_valid
        False
    """
    errors: list[str] = []
    length = len(values)
    
    if not allow_empty and length == 0:
        errors.append(f"{label} cannot be empty")
    
    if min_length is not None and length < min_length:
        errors.append(f"{label} has {length} elements, minimum is {min_length}")
    
    if max_length is not None and length > max_length:
        errors.append(f"{label} has {length} elements, maximum is {max_length}")
    
    if unique:
        seen: set[Any] = set()
        duplicates: list[Any] = []
        for v in values:
            try:
                if v in seen:
                    duplicates.append(v)
                seen.add(v)
            except TypeError:
                # Unhashable type, skip uniqueness check for this element
                pass
        if duplicates:
            errors.append(f"{label} contains duplicate values: {duplicates[:3]}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_dataframe_schema(
    data: dict[str, list[Any]],
    schema: dict[str, type],
    *,
    allow_extra_columns: bool = False,
) -> ValidationResult:
    """Validate tabular data against expected schema.
    
    Args:
        data: Dictionary mapping column names to value lists.
        schema: Dictionary mapping column names to expected types.
        allow_extra_columns: Whether columns not in schema are permitted.
        
    Returns:
        ValidationResult indicating success or failure.
        
    Example:
        >>> data = {"name": ["Alice", "Bob"], "age": [30, 25]}
        >>> schema = {"name": str, "age": int}
        >>> result = validate_dataframe_schema(data, schema)
        >>> result.is_valid
        True
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    # Check for missing columns
    missing = set(schema.keys()) - set(data.keys())
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    # Check for extra columns
    if not allow_extra_columns:
        extra = set(data.keys()) - set(schema.keys())
        if extra:
            warnings.append(f"Unexpected columns: {extra}")
    
    # Check column types
    for col_name, expected_type in schema.items():
        if col_name not in data:
            continue
        
        values = data[col_name]
        for i, v in enumerate(values):
            if v is not None and not isinstance(v, expected_type):
                errors.append(
                    f"Column '{col_name}' row {i}: expected {expected_type.__name__}, "
                    f"got {type(v).__name__}"
                )
                break  # Report first error per column
    
    # Check consistent lengths
    if data:
        lengths = {col: len(vals) for col, vals in data.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            errors.append(f"Inconsistent column lengths: {lengths}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


# =============================================================================
# §3. NUMERICAL resilience
# =============================================================================


def safe_float_comparison(
    a: float,
    b: float,
    *,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    """Compare floats with tolerance for representational differences.
    
    These limitations motivate several defensive programming practices.
    First, we should avoid testing floating-point values for exact equality,
    instead using a tolerance value to accommodate small representational
    differences (Book §5).
    
    Args:
        a: First value.
        b: Second value.
        rel_tol: Relative tolerance.
        abs_tol: Absolute tolerance.
        
    Returns:
        True if values are approximately equal.
        
    Example:
        >>> safe_float_comparison(0.1 + 0.2, 0.3)
        True
        >>> 0.1 + 0.2 == 0.3  # Without safe comparison
        False
    """
    # Handle special cases
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)  # Same sign infinity
    if math.isinf(a) or math.isinf(b):
        return False
    
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def detect_numerical_instability(
    values: Sequence[float],
    *,
    expected_range: tuple[float, float] | None = None,
    max_nan_ratio: float = 0.0,
    max_inf_ratio: float = 0.0,
) -> ValidationResult:
    """Detect potential numerical instability in computed values.
    
    Identifies NaN values, infinities and out-of-range results that
    may indicate numerical problems.
    
    Args:
        values: Sequence of computed values to check.
        expected_range: Optional (min, max) expected value range.
        max_nan_ratio: Maximum allowed ratio of NaN values.
        max_inf_ratio: Maximum allowed ratio of infinite values.
        
    Returns:
        ValidationResult with stability assessment.
        
    Example:
        >>> result = detect_numerical_instability([1.0, 2.0, float('nan')])
        >>> result.is_valid
        False
        >>> 'NaN' in result.errors[0]
        True
    """
    errors: list[str] = []
    warnings: list[str] = []
    
    if not values:
        return ValidationResult.success()
    
    total = len(values)
    nan_count = sum(1 for v in values if math.isnan(v))
    inf_count = sum(1 for v in values if math.isinf(v))
    
    nan_ratio = nan_count / total
    inf_ratio = inf_count / total
    
    if nan_ratio > max_nan_ratio:
        if nan_ratio > 0:
            errors.append(
                f"Found {nan_count} NaN values ({nan_ratio:.1%}) indicating "
                "undefined operations"
            )
    
    if inf_ratio > max_inf_ratio:
        if inf_ratio > 0:
            warnings.append(
                f"Found {inf_count} infinite values ({inf_ratio:.1%}) indicating "
                "potential overflow"
            )
    
    if expected_range is not None:
        min_val, max_val = expected_range
        out_of_range = sum(
            1 for v in values
            if not math.isnan(v) and not math.isinf(v) and not (min_val <= v <= max_val)
        )
        if out_of_range > 0:
            warnings.append(
                f"{out_of_range} values outside expected range [{min_val}, {max_val}]"
            )
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)


def kahan_summation(values: Iterable[float]) -> float:
    """Sum floating-point values with compensation for rounding errors.
    
    Implements Kahan summation algorithm to maintain precision when
    summing many values of varying magnitudes. The algorithm tracks
    accumulated error and compensates in subsequent additions.
    
    Args:
        values: Iterable of floating-point values.
        
    Returns:
        Sum with improved numerical accuracy.
        
    Example:
        >>> # Standard sum loses precision with many small values
        >>> standard = sum([0.1] * 10)
        >>> kahan = kahan_summation([0.1] * 10)
        >>> abs(kahan - 1.0) < abs(standard - 1.0)
        True
    """
    total = 0.0
    compensation = 0.0  # Running compensation for lost low-order bits
    
    for value in values:
        # Add compensation to current value
        y = value - compensation
        # Perform addition
        t = total + y
        # Calculate new compensation: (what was lost in the addition)
        compensation = (t - total) - y
        total = t
    
    return total


def stable_mean(values: Sequence[float]) -> float:
    """Calculate mean with improved numerical stability.
    
    Uses Welford's online algorithm to compute mean without
    accumulating large intermediate sums.
    
    Args:
        values: Sequence of values.
        
    Returns:
        Mean value.
        
    Raises:
        ValueError: If values is empty.
        
    Example:
        >>> stable_mean([1.0, 2.0, 3.0])
        2.0
    """
    if not values:
        raise ValueError("Cannot compute mean of empty sequence")
    
    mean = 0.0
    for i, x in enumerate(values, 1):
        mean += (x - mean) / i
    
    return mean


def stable_variance(values: Sequence[float], *, ddof: int = 0) -> float:
    """Calculate variance with improved numerical stability.
    
    Uses Welford's online algorithm to compute variance without
    the numerical issues of the naive two-pass algorithm.
    
    Args:
        values: Sequence of values.
        ddof: Delta degrees of freedom (0 for population, 1 for sample).
        
    Returns:
        Variance value.
        
    Raises:
        ValueError: If insufficient values for given ddof.
        
    Example:
        >>> stable_variance([1.0, 2.0, 3.0])
        0.6666666666666666
        >>> stable_variance([1.0, 2.0, 3.0], ddof=1)
        1.0
    """
    n = len(values)
    if n <= ddof:
        raise ValueError(f"Need at least {ddof + 1} values for ddof={ddof}")
    
    mean = 0.0
    m2 = 0.0  # Sum of squared differences from mean
    
    for i, x in enumerate(values, 1):
        delta = x - mean
        mean += delta / i
        delta2 = x - mean
        m2 += delta * delta2
    
    return m2 / (n - ddof)


# =============================================================================
# §4. DEFENSIVE DATA PROCESSING
# =============================================================================


def safe_json_load(
    content: str | bytes,
    *,
    default: T | None = None,
    strict: bool = True,
) -> dict[str, Any] | list[Any] | T | None:
    """Safely load JSON with error handling.
    
    Args:
        content: JSON string or bytes.
        default: Value to return on parse error (if not strict).
        strict: Whether to raise on parse error.
        
    Returns:
        Parsed JSON data or default value.
        
    Raises:
        ValueError: If strict and JSON is invalid.
        
    Example:
        >>> safe_json_load('{"key": "value"}')
        {'key': 'value'}
        >>> safe_json_load('invalid', default={}, strict=False)
        {}
    """
    try:
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning("JSON parse error: %s", e)
        if strict:
            raise ValueError(f"Invalid JSON: {e}") from e
        return default


def resilient_csv_reader(
    file_path: Path,
    *,
    encoding: str = "utf-8",
    fallback_encodings: tuple[str, ...] = ("latin-1", "cp1252"),
    skip_errors: bool = False,
) -> list[dict[str, str]]:
    """Read CSV with consistent encoding and error handling.
    
    Attempts multiple encodings and optionally skips malformed rows.
    
    Args:
        file_path: Path to CSV file.
        encoding: Primary encoding to try.
        fallback_encodings: Encodings to try if primary fails.
        skip_errors: Whether to skip rows with errors.
        
    Returns:
        List of dictionaries (one per row).
        
    Raises:
        FileNotFoundError: If file does not exist.
        UnicodeDecodeError: If no encoding works.
        
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        ...     _ = f.write('name,age\\nAlice,30\\n')
        ...     path = Path(f.name)
        >>> rows = resilient_csv_reader(path)
        >>> rows[0]['name']
        'Alice'
        >>> path.unlink()
    """
    encodings_to_try = [encoding, *fallback_encodings]
    content: str | None = None
    
    for enc in encodings_to_try:
        try:
            content = file_path.read_text(encoding=enc)
            logger.debug("Successfully read %s with encoding %s", file_path, enc)
            break
        except UnicodeDecodeError:
            logger.debug("Encoding %s failed for %s", enc, file_path)
            continue
    
    if content is None:
        raise UnicodeDecodeError(
            encoding,
            b"",
            0,
            1,
            f"Could not decode {file_path} with any attempted encoding",
        )
    
    rows: list[dict[str, str]] = []
    reader = csv.DictReader(content.splitlines())
    
    for i, row in enumerate(reader):
        try:
            # Validate row has expected structure
            if any(v is None for v in row.values()):
                raise ValueError(f"Row {i} has missing fields")
            rows.append(row)
        except Exception as e:
            logger.warning("Error processing row %d: %s", i, e)
            if not skip_errors:
                raise
    
    return rows


@dataclass
class CheckpointManager:
    """Manage checkpoints for long-running computations.
    
    Enables saving and restoring computation state to handle
    interruptions without losing progress.
    
    Attributes:
        checkpoint_path: Path to checkpoint file.
        state: Current state dictionary.
        completed_items: Set of completed item identifiers.
        
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        ...     path = Path(f.name)
        >>> manager = CheckpointManager(path)
        >>> manager.mark_completed("item1")
        >>> manager.save()
        >>> manager2 = CheckpointManager(path)
        >>> manager2.load_if_exists()
        >>> "item1" in manager2.completed_items
        True
        >>> path.unlink()
    """
    
    checkpoint_path: Path
    state: dict[str, Any] = field(default_factory=dict)
    completed_items: set[str] = field(default_factory=set)
    _is_complete: bool = field(default=False, init=False, repr=False)
    
    @property
    def is_complete(self) -> bool:
        """Whether processing is marked complete."""
        return self._is_complete
    
    def save(self) -> None:
        """Save current state to checkpoint file."""
        checkpoint_data = {
            "state": self.state,
            "completed_items": self.completed_items,
            "is_complete": self._is_complete,
        }
        
        # Write to temporary file then rename for atomicity
        temp_path = self.checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        temp_path.rename(self.checkpoint_path)
        
        logger.info(
            "Checkpoint saved: %d completed items",
            len(self.completed_items),
        )
    
    def load_if_exists(self) -> bool:
        """Load checkpoint if it exists.
        
        Returns:
            True if checkpoint was loaded.
        """
        if not self.checkpoint_path.exists():
            return False
        
        try:
            with open(self.checkpoint_path, "rb") as f:
                data = pickle.load(f)
            
            self.state = data.get("state", {})
            self.completed_items = data.get("completed_items", set())
            self._is_complete = data.get("is_complete", False)
            
            logger.info(
                "Checkpoint loaded: %d completed items",
                len(self.completed_items),
            )
            return True
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
            return False
    
    def mark_completed(self, item_id: str) -> None:
        """Mark an item as completed.
        
        Args:
            item_id: Identifier for the completed item.
        """
        self.completed_items.add(item_id)
    
    def is_completed(self, item_id: str) -> bool:
        """Check if an item is already completed.
        
        Args:
            item_id: Identifier to check.
            
        Returns:
            True if item was previously completed.
        """
        return item_id in self.completed_items
    
    def mark_complete(self) -> None:
        """Mark the entire process as complete."""
        self._is_complete = True
        self.save()
    
    def clear(self) -> None:
        """Clear checkpoint data and delete file."""
        self.state.clear()
        self.completed_items.clear()
        self._is_complete = False
        
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
            logger.info("Checkpoint cleared")


def checkpoint_processor(
    items: Sequence[T],
    processor: Callable[[T], Any],
    checkpoint_path: Path,
    *,
    item_id_func: Callable[[T], str] = str,
    save_interval: int = 10,
) -> list[Any]:
    """Process items with checkpoint-based recovery.
    
    Enables resumption of long-running processing after failures.
    
    Args:
        items: Items to process.
        processor: Function to apply to each item.
        checkpoint_path: Path for checkpoint file.
        item_id_func: Function to get unique ID from item.
        save_interval: How often to save checkpoint.
        
    Returns:
        List of processing results.
        
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        ...     path = Path(f.name)
        >>> results = checkpoint_processor(
        ...     [1, 2, 3],
        ...     lambda x: x * 2,
        ...     path,
        ... )
        >>> results
        [2, 4, 6]
        >>> path.unlink()
    """
    manager = CheckpointManager(checkpoint_path)
    manager.load_if_exists()
    
    results: list[Any] = []
    
    try:
        for i, item in enumerate(items):
            item_id = item_id_func(item)
            
            if manager.is_completed(item_id):
                logger.debug("Skipping already completed: %s", item_id)
                continue
            
            result = processor(item)
            results.append(result)
            manager.mark_completed(item_id)
            
            if (i + 1) % save_interval == 0:
                manager.save()
        
        manager.mark_complete()
        
    except Exception:
        manager.save()
        logger.info("Progress saved after error")
        raise
    
    finally:
        # Clean up on success
        if manager.is_complete and checkpoint_path.exists():
            checkpoint_path.unlink()
    
    return results


# =============================================================================
# MODULE EXECUTION
# =============================================================================


def main() -> None:
    """Demonstrate defensive programming concepts."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting 09UNIT Lab 02 demonstrations")
    
    # §1: Design by Contract
    logger.info("§1: Design by Contract")
    
    @precondition(lambda x: x >= 0, "x must be non-negative")
    @postcondition(lambda r: r >= 0, "result must be non-negative")
    def safe_sqrt(x: float) -> float:
        return math.sqrt(x)
    
    logger.info("safe_sqrt(4) = %s", safe_sqrt(4))
    
    # §2: Input Validation
    logger.info("§2: Input Validation Patterns")
    
    result = validate_numeric_range(0.5, min_value=0, max_value=1)
    logger.info("Validate 0.5 in [0,1]: %s", result.is_valid)
    
    result = validate_numeric_range(-1, min_value=0)
    logger.info("Validate -1 >= 0: %s, errors: %s", result.is_valid, result.errors)
    
    # §3: Numerical resilience
    logger.info("§3: Numerical resilience")
    
    a, b = 0.1 + 0.2, 0.3
    logger.info("0.1 + 0.2 == 0.3 (direct): %s", a == b)
    logger.info("0.1 + 0.2 == 0.3 (safe): %s", safe_float_comparison(a, b))
    
    values = [0.1] * 10
    logger.info("Standard sum: %s", sum(values))
    logger.info("Kahan sum: %s", kahan_summation(values))
    
    # §4: Defensive Data Processing
    logger.info("§4: Defensive Data Processing")
    
    json_result = safe_json_load('{"key": "value"}')
    logger.info("Parsed JSON: %s", json_result)
    
    json_fallback = safe_json_load("invalid", default={}, strict=False)
    logger.info("Invalid JSON with fallback: %s", json_fallback)
    
    logger.info("09UNIT Lab 02 demonstrations complete")


if __name__ == "__main__":
    main()
