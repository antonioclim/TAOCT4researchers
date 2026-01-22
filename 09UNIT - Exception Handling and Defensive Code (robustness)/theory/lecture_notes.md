# 09UNIT: Lecture Notes

## Exception Handling and Defensive Programming

---

## §1 Introduction to Exception Handling

### 1.1 The Problem of Exceptional Conditions

Every computational system operates under assumptions about its environment: input data conforms to expected formats, external services respond within reasonable timeframes, computational resources remain available and numerical operations produce meaningful results. When these assumptions fail—and in scientific computing, they inevitably do—the system must respond gracefully rather than crashing or producing silently incorrect results.

Effective error handling is necessary for creating functions that fail gracefully and provide meaningful feedback when problems occur. Python's exception mechanism offers a structured approach to error handling, enabling functions to detect and respond to exceptional conditions (Beazley, 2009, p. 114). This lecture explores Python's exception system in depth, from fundamental mechanics through advanced patterns applicable to research software.

### 1.2 Historical Context

Exception handling mechanisms evolved from the limitations of earlier error-signalling approaches:

**Return Code Approach** (C, early languages):
- Functions return special values (e.g., -1, NULL) to indicate errors
- Callers must explicitly check every return value
- Error handling code intermixes with normal logic
- Easy to forget checks, leading to silent failures

**Global Error State** (errno pattern):
- Global variable holds error code from last operation
- Separates error indication from return value
- Suffers from thread-safety issues and easy oversight

**Exception Mechanism** (PL/I, CLU, Python):
- Errors propagate automatically until handled
- Separates normal flow from error handling
- Enforces attention to error conditions
- Enables structured cleanup through finally/context managers

Python's exception system derives from the termination model developed by Barbara Liskov for CLU in 1975. Unlike resumption models (where handlers can return control to the raising point), termination models treat exceptions as one-way transfers of control—once raised, an exception propagates until caught or terminates the program.

### 1.3 The Exception Hierarchy

Python's exception types form a class hierarchy rooted at `BaseException`:

```
BaseException
├── SystemExit          # sys.exit() calls
├── KeyboardInterrupt   # Ctrl+C signal
├── GeneratorExit       # Generator cleanup
└── Exception           # All other exceptions
    ├── StopIteration
    ├── ArithmeticError
    │   ├── ZeroDivisionError
    │   ├── OverflowError
    │   └── FloatingPointError
    ├── LookupError
    │   ├── IndexError
    │   └── KeyError
    ├── ValueError
    ├── TypeError
    ├── RuntimeError
    ├── OSError
    │   ├── FileNotFoundError
    │   ├── PermissionError
    │   └── ConnectionError
    └── [User-defined exceptions]
```

**Critical distinction**: Catching `Exception` (recommended) differs from catching `BaseException` (dangerous). The latter intercepts `SystemExit` (preventing clean shutdown), `KeyboardInterrupt` (ignoring user interrupts) and `GeneratorExit` (breaking generator cleanup). Always catch `Exception` unless you have specific reasons to handle these system-level signals.

---

## §2 Exception Handling Syntax

### 2.1 The try/except Statement

The basic exception handling construct pairs a guarded block (try) with one or more exception handlers (except):

```python
def safe_divide(numerator: float, denominator: float) -> float | None:
    """Perform division with error handling.
    
    Args:
        numerator: Dividend value.
        denominator: Divisor value.
        
    Returns:
        Quotient if division succeeds, None otherwise.
    """
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        logging.warning("Division by zero attempted: %s / %s", numerator, denominator)
        return None
    return result
```

**Handler selection**: When an exception occurs, Python searches handlers in order, selecting the first whose type matches the exception or is a supertype. This matching behaviour necessitates ordering handlers from most specific to most general:

```python
try:
    process_file(path)
except FileNotFoundError:
    # Specific: file does not exist
    logging.error("File not found: %s", path)
except PermissionError:
    # Specific: insufficient permissions
    logging.error("Permission denied: %s", path)
except OSError as e:
    # General: other OS-related errors
    logging.error("OS error processing %s: %s", path, e)
```

### 2.2 The else and finally Clauses

The complete try statement supports four clauses:

```python
try:
    # Code that may raise exceptions
    result = risky_operation()
except SpecificError as e:
    # Handle known error conditions
    handle_error(e)
else:
    # Execute ONLY if no exception occurred
    # Useful for code that should run on success but shouldn't be in try block
    process_result(result)
finally:
    # ALWAYS executes, regardless of exception status
    # necessary for cleanup operations
    release_resources()
```

**The else clause**: Code in the else block runs only if the try block completes without raising an exception. This is preferable to placing success code in the try block because:
1. It narrows the scope of exception handling, preventing accidental catching of exceptions from success code
2. It clearly delineates the "protected" code from subsequent operations

**The finally clause**: The finally block executes unconditionally—whether the try block succeeds, raises an exception, or exits via return/break/continue. This guarantee makes finally ideal for resource cleanup:

```python
def process_with_connection(host: str) -> dict[str, Any]:
    """Process data using external connection with guaranteed cleanup."""
    connection = establish_connection(host)
    try:
        data = connection.fetch_data()
        return process(data)
    finally:
        connection.close()  # Always closes, even if exception occurs
```

### 2.3 Raising Exceptions

The `raise` statement generates exceptions explicitly:

```python
def validate_probability(value: float) -> None:
    """Validate that value is a valid probability.
    
    Args:
        value: Numeric value to validate.
        
    Raises:
        ValueError: If value is not in [0, 1].
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"Expected numeric type, got {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {value}")
```

**Re-raising exceptions**: Within an except block, bare `raise` re-raises the current exception, preserving its traceback:

```python
try:
    process_data(data)
except ValueError:
    logging.error("Invalid data encountered")
    raise  # Re-raise with original traceback
```

### 2.4 Exception Chaining

Python 3 supports explicit exception chaining via `raise ... from`, preserving causal relationships:

```python
def load_configuration(path: Path) -> dict[str, Any]:
    """Load configuration from JSON file.
    
    Args:
        path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        ConfigurationError: If loading or parsing fails.
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigurationError(f"Cannot read config: {path}") from e
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config: {path}") from e
```

The `__cause__` attribute stores the explicitly chained exception (from `raise ... from`), whilst `__context__` stores implicitly chained exceptions (when raising within an except block without explicit `from`).

---

## §3 Custom Exception Hierarchies

### 3.1 Designing Domain-Specific Exceptions

Well-designed exception hierarchies communicate domain semantics through their structure. For scientific computing applications, consider organising exceptions by error category:

```python
class ResearchError(Exception):
    """Base exception for research computing errors.
    
    Attributes:
        message: Human-readable error description.
        details: Additional context for debugging.
    """
    
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataValidationError(ResearchError):
    """Raised when input data fails validation."""
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
    ) -> None:
        details = {"field": field, "value": value, "constraint": constraint}
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.constraint = constraint


class ComputationError(ResearchError):
    """Raised when computation fails or produces invalid results."""


class NumericalInstabilityError(ComputationError):
    """Raised when numerical computation becomes unstable."""
    
    def __init__(
        self,
        message: str,
        condition_number: float | None = None,
        operation: str | None = None,
    ) -> None:
        details = {"condition_number": condition_number, "operation": operation}
        super().__init__(message, details)
```

### 3.2 Exception Hierarchy Principles

**Single inheritance**: Each exception class should inherit from exactly one parent in your hierarchy. Multiple inheritance complicates handler matching and exception catching semantics.

**Meaningful granularity**: Create specific exceptions for conditions that callers might want to handle differently, but avoid excessive proliferation. A good heuristic: if callers would always handle two exception types identically, consider merging them.

**Informative attributes**: Include attributes that help callers understand and potentially recover from the error:

```python
class RateLimitError(NetworkError):
    """Raised when API rate limit is exceeded.
    
    Attributes:
        retry_after: Seconds to wait before retrying.
        limit: Rate limit that was exceeded.
    """
    
    def __init__(self, message: str, retry_after: float, limit: int) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.limit = limit
```

---

## §4 Context Managers and Resource Management

### 4.1 The Context Manager Protocol

Context managers implement the resource acquisition is initialisation (RAII) pattern in Python. They define `__enter__` and `__exit__` methods that bracket resource usage:

```python
class ManagedFile:
    """Context manager for file operations with automatic cleanup.
    
    Example:
        >>> with ManagedFile("data.txt", "r") as f:
        ...     content = f.read()
    """
    
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        self._file: IO[Any] | None = None
    
    def __enter__(self) -> IO[Any]:
        """Open the file and return the file object."""
        self._file = open(self.path, self.mode)
        return self._file
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Close the file, regardless of exceptions.
        
        Args:
            exc_type: Exception type if exception occurred, None otherwise.
            exc_val: Exception instance if exception occurred.
            exc_tb: Traceback if exception occurred.
            
        Returns:
            False to propagate exceptions, True to suppress.
        """
        if self._file is not None:
            self._file.close()
        return False  # Never suppress exceptions
```

**Exception handling in __exit__**: The `__exit__` method receives exception information when the with-block exits via exception. Returning `True` suppresses the exception (rarely appropriate); returning `False` (or None) allows it to propagate.

### 4.2 The contextlib Module

The `contextlib` module provides utilities for creating context managers without writing classes:

**@contextmanager decorator**:

```python
from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Create a temporary directory that is automatically cleaned up.
    
    Yields:
        Path to the temporary directory.
        
    Example:
        >>> with temporary_directory() as tmpdir:
        ...     (tmpdir / "test.txt").write_text("hello")
    """
    tmpdir = Path(tempfile.mkdtemp())
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
```

**ExitStack for dynamic context management**:

```python
from contextlib import ExitStack


def process_multiple_files(paths: list[Path]) -> list[str]:
    """Process multiple files with guaranteed cleanup.
    
    Uses ExitStack to manage dynamically-determined number of files.
    """
    results = []
    with ExitStack() as stack:
        files = [stack.enter_context(open(p)) for p in paths]
        for f in files:
            results.append(f.read())
    return results
```

### 4.3 Common Context Manager Applications

**Timing operations**:

```python
@contextmanager
def timer(label: str) -> Generator[None, None, None]:
    """Time a code block and log the duration."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info("%s completed in %.3f seconds", label, elapsed)
```

**Database connections**:

```python
@contextmanager
def database_connection(dsn: str) -> Generator[Connection, None, None]:
    """Provide database connection with automatic commit/rollback."""
    conn = connect(dsn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

---

## §5 Defensive Programming Principles

### 5.1 Design by Contract

Design by contract, formalised by Bertrand Meyer for Eiffel, specifies software components through preconditions, postconditions and invariants:

**Preconditions**: Obligations the caller must satisfy before invoking a function. If preconditions are violated, the function may behave arbitrarily.

**Postconditions**: Guarantees the function provides upon successful completion, assuming preconditions held.

**Invariants**: Properties that must hold before and after every public method of a class.

```python
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")


def precondition(pred: Callable[..., bool], message: str = "Precondition violated"):
    """Decorator that enforces a precondition on function arguments."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not pred(*args, **kwargs):
                raise ContractViolationError(message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def postcondition(pred: Callable[[T], bool], message: str = "Postcondition violated"):
    """Decorator that enforces a postcondition on return value."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            result = func(*args, **kwargs)
            if not pred(result):
                raise ContractViolationError(message)
            return result
        return wrapper
    return decorator
```

### 5.2 Input Validation Patterns

Scientific computing often involves complex data transformations where errors can be subtle and difficult to detect. Defensive programming—anticipating and handling potential errors and edge cases—helps create more consistent functions that fail gracefully when presented with unexpected inputs (Downey, 2015, p. 126).

**Validation result pattern**:

```python
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of a validation operation.
    
    Attributes:
        is_valid: Whether validation passed.
        errors: List of error messages if validation failed.
        warnings: List of warning messages.
    """
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            raise DataValidationError(
                f"Validation failed: {'; '.join(self.errors)}"
            )


def validate_numeric_range(
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    allow_nan: bool = False,
    allow_inf: bool = False,
) -> ValidationResult:
    """Validate a numeric value against constraints."""
    errors = []
    warnings = []
    
    if math.isnan(value):
        if not allow_nan:
            errors.append("NaN values not permitted")
    elif math.isinf(value):
        if not allow_inf:
            errors.append("Infinite values not permitted")
    else:
        if min_value is not None and value < min_value:
            errors.append(f"Value {value} below minimum {min_value}")
        if max_value is not None and value > max_value:
            errors.append(f"Value {value} above maximum {max_value}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
```

### 5.3 Fail-Fast Principle

The fail-fast principle advocates detecting errors as early as possible, close to their source. This approach:

1. **Simplifies debugging**: Errors are caught near their cause, not after propagating through layers of code
2. **Prevents cascading failures**: Invalid state is detected before it corrupts additional data
3. **Provides clearer error messages**: Context is available to generate informative diagnostics

```python
def process_experiment_data(
    measurements: Sequence[float],
    metadata: dict[str, Any],
) -> ExperimentResult:
    """Process experimental measurements with fail-fast validation.
    
    Validates all inputs before beginning expensive computation.
    """
    # Validate measurements immediately
    if not measurements:
        raise DataValidationError("Measurements cannot be empty")
    
    for i, m in enumerate(measurements):
        result = validate_numeric_range(m, min_value=0, allow_nan=False)
        if not result.is_valid:
            raise DataValidationError(
                f"Invalid measurement at index {i}: {result.errors[0]}"
            )
    
    # Validate metadata structure
    required_fields = {"experiment_id", "timestamp", "operator"}
    missing = required_fields - set(metadata.keys())
    if missing:
        raise DataValidationError(f"Missing required metadata: {missing}")
    
    # Now proceed with computation, confident in input validity
    return compute_results(measurements, metadata)
```

---

## §6 Resilience Patterns

### 6.1 Retry with Exponential Backoff

Transient failures—network timeouts, temporary resource unavailability, rate limiting—often resolve themselves if the operation is retried after a delay. Exponential backoff progressively increases the delay between retries, reducing load on struggling systems:

```python
def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (ConnectionError, TimeoutError),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries operations with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        delay = min(
                            base_delay * (exponential_base ** (attempt - 1)),
                            max_delay,
                        )
                        # Add jitter to prevent thundering herd
                        delay *= 0.5 + random.random()
                        logging.warning(
                            "Attempt %d/%d failed: %s. Retrying in %.2fs",
                            attempt, max_attempts, e, delay,
                        )
                        time.sleep(delay)
            
            raise last_exception  # type: ignore[misc]
        return wrapper
    return decorator
```

### 6.2 Circuit Breaker Pattern

The circuit breaker pattern prevents cascading failures by "opening" (failing fast) when an external service becomes unreliable:

```python
from enum import Enum, auto
from dataclasses import dataclass


class CircuitState(Enum):
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing fast
    HALF_OPEN = auto() # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault-tolerant external calls."""
    
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.reset_timeout:
                self._state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        self._failure_count = 0
        self._state = CircuitState.CLOSED
    
    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
```

### 6.3 Checkpoint-Based Recovery

Long-running computations benefit from periodic checkpointing, enabling recovery from failures without restarting from the beginning:

```python
@contextmanager
def checkpoint_processor(
    checkpoint_path: Path,
    save_interval: int = 100,
) -> Generator[CheckpointManager, None, None]:
    """Context manager for checkpoint-based processing.
    
    Automatically saves progress and enables resume on failure.
    """
    manager = CheckpointManager(checkpoint_path)
    manager.load_if_exists()
    
    try:
        yield manager
        manager.mark_complete()
    except Exception:
        manager.save()
        raise
    finally:
        if checkpoint_path.exists() and manager.is_complete:
            checkpoint_path.unlink()  # Clean up on success
```

---

## §7 Numerical resilience

### 7.1 Floating-Point Comparison

These limitations motivate several defensive programming practices. First, we should avoid testing floating-point values for exact equality, instead using a tolerance value to accommodate small representational differences (Book §5):

```python
def safe_float_comparison(
    a: float,
    b: float,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-12,
) -> bool:
    """Compare floats with tolerance for representational differences.
    
    Uses both relative and absolute tolerance to handle values
    across different magnitudes.
    """
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


def detect_numerical_instability(
    values: Sequence[float],
    expected_range: tuple[float, float] | None = None,
) -> ValidationResult:
    """Detect potential numerical instability in computed values."""
    errors = []
    warnings = []
    
    nan_count = sum(1 for v in values if math.isnan(v))
    inf_count = sum(1 for v in values if math.isinf(v))
    
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values indicating undefined operations")
    
    if inf_count > 0:
        warnings.append(f"Found {inf_count} infinite values indicating overflow")
    
    if expected_range is not None:
        out_of_range = sum(1 for v in values if not (expected_range[0] <= v <= expected_range[1]))
        if out_of_range > 0:
            warnings.append(f"{out_of_range} values outside expected range {expected_range}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
```

### 7.2 Kahan Summation

When summing many floating-point values, accumulated rounding errors can become significant. Kahan summation compensates for lost precision:

```python
def kahan_summation(values: Iterable[float]) -> float:
    """Sum floating-point values with compensation for rounding errors.
    
    Implements Kahan summation algorithm to maintain precision
    when summing many values of varying magnitudes.
    """
    total = 0.0
    compensation = 0.0  # Running compensation for lost low-order bits
    
    for value in values:
        y = value - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    
    return total
```

---

## §8 Summary and Key Takeaways

### 8.1 Core Principles

1. **Use exceptions for exceptional conditions**: Reserve exceptions for truly unexpected situations, not normal control flow
2. **Catch specific exceptions**: Avoid bare `except:` or catching `BaseException`
3. **Preserve exception context**: Use `raise ... from` to maintain causal chains
4. **Clean up deterministically**: Employ context managers for resource management
5. **Fail fast**: Validate inputs early, close to the error source
6. **Design consistent systems**: Implement retry, circuit breaker and checkpoint patterns for consistent operation

### 8.2 Mental Models

**Exception as control flow**: Think of exceptions as an alternative return path—they transfer control and information from the point of failure to the nearest appropriate handler.

**Context managers as brackets**: The `with` statement brackets resource usage with guaranteed setup and cleanup, like matched parentheses ensuring balanced operations.

**Defensive programming as insurance**: Validation and error handling are investments that pay dividends when (not if) unexpected conditions occur.

### 8.3 Looking Ahead

The exception handling and defensive programming techniques from this unit form the foundation for:

- **10UNIT**: File handling with consistent error recovery
- **12UNIT**: Network programming with transient failure handling
- **13UNIT**: Parallel computing with fault tolerance

---

## References

Beazley, D. (2009). *Python necessary Reference* (4th ed.). Addison-Wesley.

Dijkstra, E. W. (1976). *A Discipline of Programming*. Prentice Hall.

Downey, A. B. (2015). *Think Python* (2nd ed.). O'Reilly Media.

Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.

Meyer, B. (1988). *Object-Oriented Software Construction*. Prentice Hall.

Yourdon, E., & Constantine, L. L. (1979). *Structured Design*. Prentice Hall.
