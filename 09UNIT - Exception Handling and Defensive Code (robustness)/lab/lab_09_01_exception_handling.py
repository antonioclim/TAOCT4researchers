#!/usr/bin/env python3
"""09UNIT Lab 01: Exception Handling Fundamentals.

This laboratory module explores Python's exception handling mechanisms,
progressing from basic try/except blocks through custom exception hierarchies,
context managers and advanced exception patterns for research computing.

Effective error handling is necessary for creating functions that fail gracefully
and provide meaningful feedback when problems occur. Python's exception mechanism
offers a structured approach to error handling, enabling functions to detect and
respond to exceptional conditions (Beazley, 2009, p. 114).

Sections:
    §1. Exception Mechanism Fundamentals
    §2. Custom Exception Hierarchies
    §3. Context Managers
    §4. Exception Patterns
    §5. Logging Integration

Author: Course Team
Version: 1.0.0
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Callable, Generator, ParamSpec, TypeVar

# Configure module logger
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# §1. EXCEPTION MECHANISM FUNDAMENTALS
# =============================================================================


def demonstrate_exception_hierarchy() -> dict[str, list[str]]:
    """Demonstrate Python's exception class hierarchy.
    
    Explores the inheritance relationships between exception classes,
    illustrating how exception handling relies on type matching.
    
    Returns:
        Dictionary mapping exception categories to their members.
        
    Example:
        >>> hierarchy = demonstrate_exception_hierarchy()
        >>> "ArithmeticError" in hierarchy
        True
        >>> "ZeroDivisionError" in hierarchy["ArithmeticError"]
        True
    """
    hierarchy: dict[str, list[str]] = {}
    
    # Arithmetic errors
    hierarchy["ArithmeticError"] = [
        "ZeroDivisionError",
        "OverflowError",
        "FloatingPointError",
    ]
    
    # Lookup errors
    hierarchy["LookupError"] = [
        "IndexError",
        "KeyError",
    ]
    
    # OS errors (file system, network, etc.)
    hierarchy["OSError"] = [
        "FileNotFoundError",
        "FileExistsError",
        "PermissionError",
        "IsADirectoryError",
        "NotADirectoryError",
        "ConnectionError",
        "TimeoutError",
    ]
    
    # Value and type errors
    hierarchy["ValueError_TypeError"] = [
        "ValueError",
        "TypeError",
        "UnicodeError",
    ]
    
    # Demonstrate inheritance checking
    logger.debug("ZeroDivisionError is subclass of ArithmeticError: %s",
                 issubclass(ZeroDivisionError, ArithmeticError))
    logger.debug("FileNotFoundError is subclass of OSError: %s",
                 issubclass(FileNotFoundError, OSError))
    
    return hierarchy


def safe_divide(
    numerator: float,
    denominator: float,
    *,
    default: float | None = None,
) -> float | None:
    """Perform division with comprehensive error handling.
    
    Demonstrates basic try/except pattern with specific exception handling
    and optional default value on failure.
    
    Args:
        numerator: Dividend value.
        denominator: Divisor value.
        default: Value to return on division error (None raises exception).
        
    Returns:
        Quotient of numerator/denominator, or default if division fails.
        
    Raises:
        ZeroDivisionError: If denominator is zero and no default provided.
        TypeError: If arguments are not numeric.
        
    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0, default=0.0)
        0.0
        >>> safe_divide(10, 0)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: Division by zero: 10 / 0
    """
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        logger.warning("Division by zero attempted: %s / %s", numerator, denominator)
        if default is not None:
            return default
        raise ZeroDivisionError(f"Division by zero: {numerator} / {denominator}")
    except TypeError as e:
        logger.error("Type error in division: %s", e)
        raise TypeError(
            f"Expected numeric types, got {type(numerator).__name__} "
            f"and {type(denominator).__name__}"
        ) from e
    else:
        # This block runs only if no exception occurred
        logger.debug("Division successful: %s / %s = %s", numerator, denominator, result)
        return result


def parse_config(config_path: Path) -> dict[str, Any]:
    """Parse configuration from JSON file with layered error handling.
    
    Demonstrates exception chaining using 'raise ... from' syntax
    to preserve causal relationships between errors.
    
    Args:
        config_path: Path to JSON configuration file.
        
    Returns:
        Parsed configuration dictionary.
        
    Raises:
        ConfigurationError: If file cannot be read or parsed.
        
    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        ...     _ = f.write('{"key": "value"}')
        ...     temp_path = Path(f.name)
        >>> config = parse_config(temp_path)
        >>> config["key"]
        'value'
        >>> temp_path.unlink()
    """
    # Attempt to read file
    try:
        content = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", config_path)
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            details={"path": str(config_path), "error_type": "file_not_found"},
        ) from e
    except PermissionError as e:
        logger.error("Permission denied reading config: %s", config_path)
        raise ConfigurationError(
            f"Permission denied: {config_path}",
            details={"path": str(config_path), "error_type": "permission_denied"},
        ) from e
    except OSError as e:
        logger.error("OS error reading config: %s", e)
        raise ConfigurationError(
            f"Cannot read configuration: {config_path}",
            details={"path": str(config_path), "os_error": str(e)},
        ) from e
    
    # Attempt to parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config file: %s at line %d", e.msg, e.lineno)
        raise ConfigurationError(
            f"Invalid JSON in configuration: {e.msg}",
            details={
                "path": str(config_path),
                "line": e.lineno,
                "column": e.colno,
                "error_type": "json_parse_error",
            },
        ) from e


def exception_chaining_demo() -> None:
    """Demonstrate exception chaining mechanisms.
    
    Shows the difference between explicit chaining (raise ... from)
    and implicit chaining (raise within except block).
    """
    def explicit_chaining() -> None:
        """Explicit chaining preserves cause in __cause__."""
        try:
            int("not_a_number")
        except ValueError as original:
            raise RuntimeError("Failed to parse value") from original
    
    def implicit_chaining() -> None:
        """Implicit chaining stores context in __context__."""
        try:
            int("not_a_number")
        except ValueError:
            # New exception raised during handling
            raise RuntimeError("Failed to parse value")
    
    def suppress_context() -> None:
        """Suppress context with 'from None'."""
        try:
            int("not_a_number")
        except ValueError:
            raise RuntimeError("Parse failed") from None
    
    # Demonstrate each pattern
    for name, func in [
        ("explicit", explicit_chaining),
        ("implicit", implicit_chaining),
        ("suppressed", suppress_context),
    ]:
        try:
            func()
        except RuntimeError as e:
            logger.info(
                "%s chaining - __cause__: %s, __context__: %s",
                name,
                e.__cause__,
                e.__context__,
            )


# =============================================================================
# §2. CUSTOM EXCEPTION HIERARCHIES
# =============================================================================


class ResearchError(Exception):
    """Base exception for research computing errors.
    
    Provides a foundation for domain-specific exception hierarchies
    with structured error information for debugging and logging.
    
    Attributes:
        message: Human-readable error description.
        details: Additional context dictionary for debugging.
        
    Example:
        >>> raise ResearchError("Computation failed", details={"stage": "analysis"})
        Traceback (most recent call last):
            ...
        ResearchError: Computation failed
    """
    
    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialise research error with message and optional details.
        
        Args:
            message: Human-readable error description.
            details: Additional context for debugging.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return string representation including details if present."""
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(ResearchError):
    """Raised when configuration loading or validation fails.
    
    Used for errors related to configuration files, environment
    variables and other configuration sources.
    """


class DataValidationError(ResearchError):
    """Raised when input data fails validation.
    
    Provides structured information about validation failures
    including the field, invalid value and violated constraint.
    
    Attributes:
        field: Name of the field that failed validation.
        value: The invalid value.
        constraint: Description of the violated constraint.
    """
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        constraint: str | None = None,
    ) -> None:
        """Initialise data validation error.
        
        Args:
            message: Human-readable error description.
            field: Name of field that failed validation.
            value: The invalid value.
            constraint: Description of violated constraint.
        """
        details = {
            "field": field,
            "value": repr(value),
            "constraint": constraint,
        }
        super().__init__(message, details)
        self.field = field
        self.value = value
        self.constraint = constraint


class FileFormatError(ResearchError):
    """Raised when a file does not conform to expected format.
    
    Used for parsing errors in data files (CSV, JSON, FASTA, etc.)
    that prevent successful data extraction.
    
    Attributes:
        file_path: Path to the malformed file.
        expected_format: Description of expected format.
        line_number: Line where error was detected (if applicable).
    """
    
    def __init__(
        self,
        message: str,
        file_path: Path | str | None = None,
        expected_format: str | None = None,
        line_number: int | None = None,
    ) -> None:
        """Initialise file format error.
        
        Args:
            message: Human-readable error description.
            file_path: Path to the malformed file.
            expected_format: Description of expected format.
            line_number: Line number where error occurred.
        """
        details = {
            "file_path": str(file_path) if file_path else None,
            "expected_format": expected_format,
            "line_number": line_number,
        }
        super().__init__(message, details)
        self.file_path = Path(file_path) if file_path else None
        self.expected_format = expected_format
        self.line_number = line_number


class ComputationError(ResearchError):
    """Raised when computation fails or produces invalid results.
    
    Base class for computational failures including numerical
    instability, convergence failures and resource exhaustion.
    """


class NumericalInstabilityError(ComputationError):
    """Raised when numerical computation becomes unstable.
    
    Indicates conditions such as loss of precision, overflow,
    or ill-conditioned matrices that compromise result validity.
    
    Attributes:
        condition_number: Matrix condition number if applicable.
        operation: Description of the failing operation.
    """
    
    def __init__(
        self,
        message: str,
        condition_number: float | None = None,
        operation: str | None = None,
    ) -> None:
        """Initialise numerical instability error.
        
        Args:
            message: Human-readable error description.
            condition_number: Condition number indicating instability.
            operation: Description of the operation that failed.
        """
        details = {
            "condition_number": condition_number,
            "operation": operation,
        }
        super().__init__(message, details)
        self.condition_number = condition_number
        self.operation = operation


class ConvergenceError(ComputationError):
    """Raised when iterative computation fails to converge.
    
    Attributes:
        iterations: Number of iterations performed.
        final_error: Error metric at termination.
        tolerance: Required convergence tolerance.
    """
    
    def __init__(
        self,
        message: str,
        iterations: int | None = None,
        final_error: float | None = None,
        tolerance: float | None = None,
    ) -> None:
        """Initialise convergence error.
        
        Args:
            message: Human-readable error description.
            iterations: Number of iterations performed.
            final_error: Error at termination.
            tolerance: Required tolerance.
        """
        details = {
            "iterations": iterations,
            "final_error": final_error,
            "tolerance": tolerance,
        }
        super().__init__(message, details)
        self.iterations = iterations
        self.final_error = final_error
        self.tolerance = tolerance


class ContractViolationError(ResearchError):
    """Raised when a design-by-contract condition is violated.
    
    Used by precondition, postcondition and invariant decorators
    to signal contract violations during execution.
    
    Attributes:
        contract_type: Type of contract (precondition, postcondition, invariant).
        condition: Description of the violated condition.
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
            condition: The violated condition.
        """
        details = {
            "contract_type": contract_type,
            "condition": condition,
        }
        super().__init__(message, details)
        self.contract_type = contract_type
        self.condition = condition


class CircuitOpenError(ResearchError):
    """Raised when circuit breaker is open and calls are rejected.
    
    Attributes:
        service_name: Name of the protected service.
        reset_time: Estimated time until circuit reset.
    """
    
    def __init__(
        self,
        message: str,
        service_name: str | None = None,
        reset_time: float | None = None,
    ) -> None:
        """Initialise circuit open error.
        
        Args:
            message: Human-readable error description.
            service_name: Name of the service.
            reset_time: Seconds until reset attempt.
        """
        details = {
            "service_name": service_name,
            "reset_time": reset_time,
        }
        super().__init__(message, details)
        self.service_name = service_name
        self.reset_time = reset_time


# =============================================================================
# §3. CONTEXT MANAGERS
# =============================================================================


class ManagedFile:
    """Context manager for file operations with automatic cleanup.
    
    Demonstrates the class-based context manager protocol with
    __enter__ and __exit__ methods.
    
    Attributes:
        path: Path to the managed file.
        mode: File open mode.
        
    Example:
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as tf:
        ...     temp_path = Path(tf.name)
        >>> with ManagedFile(temp_path, "w") as f:
        ...     _ = f.write("test content")
        >>> temp_path.read_text()
        'test content'
        >>> temp_path.unlink()
    """
    
    def __init__(self, path: str | Path, mode: str = "r") -> None:
        """Initialise managed file.
        
        Args:
            path: Path to file.
            mode: File open mode (r, w, a, etc.).
        """
        self.path = Path(path)
        self.mode = mode
        self._file: IO[Any] | None = None
        logger.debug("ManagedFile created for %s (mode=%s)", self.path, self.mode)
    
    def __enter__(self) -> IO[Any]:
        """Open file and return file object.
        
        Returns:
            Open file object.
        """
        logger.debug("Opening file: %s", self.path)
        self._file = open(self.path, self.mode)
        return self._file
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Close file regardless of exception status.
        
        Args:
            exc_type: Exception type if exception occurred.
            exc_val: Exception instance if exception occurred.
            exc_tb: Traceback if exception occurred.
            
        Returns:
            False to propagate exceptions.
        """
        if self._file is not None:
            logger.debug("Closing file: %s", self.path)
            self._file.close()
        
        if exc_type is not None:
            logger.warning(
                "Exception during file operation on %s: %s",
                self.path,
                exc_val,
            )
        
        return False  # Never suppress exceptions


@contextmanager
def temporary_directory() -> Generator[Path, None, None]:
    """Create temporary directory with automatic cleanup.
    
    Uses the @contextmanager decorator to implement the context
    manager protocol via a generator function.
    
    Yields:
        Path to the temporary directory.
        
    Example:
        >>> with temporary_directory() as tmpdir:
        ...     test_file = tmpdir / "test.txt"
        ...     _ = test_file.write_text("hello")
        ...     test_file.exists()
        True
    """
    tmpdir = Path(tempfile.mkdtemp())
    logger.debug("Created temporary directory: %s", tmpdir)
    try:
        yield tmpdir
    finally:
        logger.debug("Removing temporary directory: %s", tmpdir)
        shutil.rmtree(tmpdir, ignore_errors=True)


class DatabaseConnection:
    """Simulated database connection with transaction management.
    
    Demonstrates context manager for transaction handling with
    automatic commit on success and rollback on exception.
    
    Attributes:
        dsn: Data source name (connection string).
        auto_commit: Whether to auto-commit on exit.
    """
    
    def __init__(self, dsn: str, *, auto_commit: bool = True) -> None:
        """Initialise database connection.
        
        Args:
            dsn: Data source name for connection.
            auto_commit: Auto-commit transactions on successful exit.
        """
        self.dsn = dsn
        self.auto_commit = auto_commit
        self._connected = False
        self._transaction_active = False
    
    def __enter__(self) -> DatabaseConnection:
        """Establish connection and begin transaction.
        
        Returns:
            Self for method chaining.
        """
        logger.info("Connecting to database: %s", self.dsn)
        self._connected = True
        self._transaction_active = True
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Handle transaction completion and close connection.
        
        Commits on success, rolls back on exception.
        
        Args:
            exc_type: Exception type if exception occurred.
            exc_val: Exception instance if exception occurred.
            exc_tb: Traceback if exception occurred.
            
        Returns:
            False to propagate exceptions.
        """
        if exc_type is not None:
            logger.warning("Rolling back transaction due to: %s", exc_val)
            self._rollback()
        elif self.auto_commit:
            logger.info("Committing transaction")
            self._commit()
        
        logger.info("Closing database connection")
        self._connected = False
        self._transaction_active = False
        
        return False
    
    def _commit(self) -> None:
        """Commit current transaction."""
        if self._transaction_active:
            logger.debug("Transaction committed")
            self._transaction_active = False
    
    def _rollback(self) -> None:
        """Rollback current transaction."""
        if self._transaction_active:
            logger.debug("Transaction rolled back")
            self._transaction_active = False
    
    def execute(self, query: str) -> list[dict[str, Any]]:
        """Execute query (simulated).
        
        Args:
            query: SQL query string.
            
        Returns:
            Simulated query results.
        """
        if not self._connected:
            raise RuntimeError("Not connected to database")
        logger.debug("Executing query: %s", query[:50])
        return []


@dataclass
class Timer:
    """Context manager for timing code blocks.
    
    Records elapsed time and optionally logs it.
    
    Attributes:
        label: Description of the timed operation.
        log_level: Logging level for timing output.
        elapsed: Elapsed time in seconds after completion.
        
    Example:
        >>> with Timer("test operation") as t:
        ...     _ = sum(range(1000))
        >>> t.elapsed > 0
        True
    """
    
    label: str = "Operation"
    log_level: int = logging.INFO
    elapsed: float = field(default=0.0, init=False)
    _start_time: float = field(default=0.0, init=False, repr=False)
    
    def __enter__(self) -> Timer:
        """Start timing.
        
        Returns:
            Self for accessing elapsed time.
        """
        self._start_time = time.perf_counter()
        logger.log(self.log_level, "Starting: %s", self.label)
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Stop timing and record elapsed.
        
        Args:
            exc_type: Exception type if exception occurred.
            exc_val: Exception instance if exception occurred.
            exc_tb: Traceback if exception occurred.
            
        Returns:
            False to propagate exceptions.
        """
        self.elapsed = time.perf_counter() - self._start_time
        status = "failed" if exc_type else "completed"
        logger.log(
            self.log_level,
            "%s %s in %.4f seconds",
            self.label,
            status,
            self.elapsed,
        )
        return False


def exitstack_demo(paths: list[Path]) -> list[str]:
    """Demonstrate ExitStack for managing multiple contexts.
    
    ExitStack enables dynamic management of an arbitrary number
    of context managers determined at runtime.
    
    Args:
        paths: List of file paths to read.
        
    Returns:
        List of file contents.
        
    Example:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as td:
        ...     p1 = Path(td) / "a.txt"
        ...     p2 = Path(td) / "b.txt"
        ...     _ = p1.write_text("content a")
        ...     _ = p2.write_text("content b")
        ...     results = exitstack_demo([p1, p2])
        >>> results
        ['content a', 'content b']
    """
    results: list[str] = []
    
    with ExitStack() as stack:
        # Dynamically enter context for each file
        files = [stack.enter_context(open(p, encoding="utf-8")) for p in paths]
        
        # Process all files while they're open
        for f in files:
            results.append(f.read())
    
    # All files automatically closed when ExitStack exits
    return results


# =============================================================================
# §4. EXCEPTION PATTERNS
# =============================================================================


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    ),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that retries operations with exponential backoff.
    
    Implements the retry pattern for transient failures with
    progressively increasing delays between attempts.
    
    Args:
        max_attempts: Maximum number of attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay between retries.
        exponential_base: Multiplier for exponential backoff.
        retryable_exceptions: Exception types to retry on.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @retry_with_backoff(max_attempts=2, base_delay=0.01)
        ... def flaky_operation():
        ...     return "success"
        >>> flaky_operation()
        'success'
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        # Calculate delay with jitter
                        delay = min(
                            base_delay * (exponential_base ** (attempt - 1)),
                            max_delay,
                        )
                        # Add jitter: 50-100% of calculated delay
                        delay *= 0.5 + random.random() * 0.5
                        
                        logger.warning(
                            "Attempt %d/%d for %s failed: %s. "
                            "Retrying in %.2f seconds",
                            attempt,
                            max_attempts,
                            func.__name__,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts for %s failed",
                            max_attempts,
                            func.__name__,
                        )
            
            # Re-raise last exception after all attempts exhausted
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected state: no exception captured")
        
        return wrapper
    return decorator


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault-tolerant external service calls.
    
    Implements the circuit breaker pattern to prevent cascading
    failures when external services become unreliable.
    
    States:
        CLOSED: Normal operation, calls pass through
        OPEN: Failing fast, calls rejected immediately
        HALF_OPEN: Testing recovery with limited calls
        
    Attributes:
        failure_threshold: Failures before opening circuit.
        reset_timeout: Seconds before attempting reset.
        
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)
        >>> def reliable_call():
        ...     return "success"
        >>> breaker.call(reliable_call)
        'success'
    """
    
    failure_threshold: int = 5
    reset_timeout: float = 30.0
    
    # Internal state (not exposed in __init__)
    _state: str = field(default="CLOSED", init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _success_count: int = field(default=0, init=False, repr=False)
    
    @property
    def state(self) -> str:
        """Current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Whether circuit is closed (normal operation)."""
        return self._state == "CLOSED"
    
    @property
    def is_open(self) -> bool:
        """Whether circuit is open (failing fast)."""
        return self._state == "OPEN"
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.
            
        Returns:
            Result of function call.
            
        Raises:
            CircuitOpenError: If circuit is open.
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self._state == "OPEN":
            time_since_failure = time.time() - self._last_failure_time
            if time_since_failure >= self.reset_timeout:
                logger.info("Circuit transitioning to HALF_OPEN")
                self._state = "HALF_OPEN"
                self._success_count = 0
            else:
                raise CircuitOpenError(
                    "Circuit breaker is open",
                    reset_time=self.reset_timeout - time_since_failure,
                )
        
        # Attempt the call
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == "HALF_OPEN":
            self._success_count += 1
            if self._success_count >= 3:  # Require 3 successes to close
                logger.info("Circuit transitioning to CLOSED")
                self._state = "CLOSED"
                self._failure_count = 0
        else:
            self._failure_count = 0
    
    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == "HALF_OPEN":
            logger.warning("Circuit reopening after failure in HALF_OPEN state")
            self._state = "OPEN"
        elif self._failure_count >= self.failure_threshold:
            logger.warning(
                "Circuit opening after %d failures",
                self._failure_count,
            )
            self._state = "OPEN"
    
    def reset(self) -> None:
        """Manually reset circuit to closed state."""
        logger.info("Circuit manually reset")
        self._state = "CLOSED"
        self._failure_count = 0
        self._success_count = 0


def graceful_degradation(
    fallback_value: T,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator providing graceful degradation with fallback value.
    
    Returns a fallback value when the decorated function raises
    specified exceptions, enabling continued operation with
    reduced functionality.
    
    Args:
        fallback_value: Value to return on failure.
        exceptions: Exception types to catch.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @graceful_degradation(fallback_value=0)
        ... def fetch_count():
        ...     raise ConnectionError("Service unavailable")
        >>> fetch_count()
        0
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.warning(
                    "%s failed with %s, returning fallback value: %s",
                    func.__name__,
                    type(e).__name__,
                    fallback_value,
                )
                return fallback_value
        return wrapper
    return decorator


def bulk_operation_with_partial_failure(
    items: list[T],
    operation: Callable[[T], Any],
    *,
    continue_on_error: bool = True,
) -> tuple[list[Any], list[tuple[T, Exception]]]:
    """Process items with optional continuation after failures.
    
    Enables processing of collections where some items may fail
    without aborting the entire operation.
    
    Args:
        items: Items to process.
        operation: Function to apply to each item.
        continue_on_error: Whether to continue after failures.
        
    Returns:
        Tuple of (successful_results, failed_items_with_exceptions).
        
    Example:
        >>> def process(x):
        ...     if x == 0:
        ...         raise ValueError("Zero not allowed")
        ...     return x * 2
        >>> results, failures = bulk_operation_with_partial_failure(
        ...     [1, 0, 2], process
        ... )
        >>> results
        [2, 4]
        >>> len(failures)
        1
    """
    results: list[Any] = []
    failures: list[tuple[T, Exception]] = []
    
    for item in items:
        try:
            result = operation(item)
            results.append(result)
        except Exception as e:
            logger.warning("Failed to process item %s: %s", item, e)
            failures.append((item, e))
            if not continue_on_error:
                raise
    
    if failures:
        logger.info(
            "Bulk operation completed: %d successes, %d failures",
            len(results),
            len(failures),
        )
    
    return results, failures


# =============================================================================
# §5. LOGGING INTEGRATION
# =============================================================================


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    *,
    include_timestamp: bool = True,
    include_module: bool = True,
) -> logging.Logger:
    """Configure logging for exception tracking.
    
    Sets up logging with appropriate formatters and handlers
    for capturing exception information during development
    and production.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path for file logging.
        include_timestamp: Include timestamp in log format.
        include_module: Include module name in log format.
        
    Returns:
        Configured root logger.
        
    Example:
        >>> root_logger = setup_logging(level=logging.DEBUG)
        >>> root_logger.info("Test message")
    """
    # Build format string
    format_parts: list[str] = []
    if include_timestamp:
        format_parts.append("%(asctime)s")
    format_parts.append("%(levelname)-8s")
    if include_module:
        format_parts.append("%(name)s")
    format_parts.append("%(message)s")
    
    log_format = " | ".join(format_parts)
    formatter = logging.Formatter(log_format)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


def logged_function(
    log_level: int = logging.DEBUG,
    *,
    log_args: bool = True,
    log_result: bool = True,
    log_exceptions: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that logs function entry, exit and exceptions.
    
    Provides automatic logging of function calls for debugging
    and monitoring exception behaviour.
    
    Args:
        log_level: Level for entry/exit logs.
        log_args: Whether to log function arguments.
        log_result: Whether to log return value.
        log_exceptions: Whether to log exceptions before re-raising.
        
    Returns:
        Decorator function.
        
    Example:
        >>> @logged_function(log_level=logging.INFO)
        ... def add(a, b):
        ...     return a + b
        >>> add(1, 2)
        3
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        func_logger = logging.getLogger(func.__module__)
        
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Log entry
            if log_args:
                func_logger.log(
                    log_level,
                    "Entering %s(args=%s, kwargs=%s)",
                    func.__name__,
                    args,
                    kwargs,
                )
            else:
                func_logger.log(log_level, "Entering %s", func.__name__)
            
            try:
                result = func(*args, **kwargs)
                
                # Log exit
                if log_result:
                    func_logger.log(
                        log_level,
                        "Exiting %s with result: %s",
                        func.__name__,
                        result,
                    )
                else:
                    func_logger.log(log_level, "Exiting %s", func.__name__)
                
                return result
                
            except Exception as e:
                if log_exceptions:
                    func_logger.exception(
                        "Exception in %s: %s",
                        func.__name__,
                        e,
                    )
                raise
        
        return wrapper
    return decorator


# =============================================================================
# MODULE EXECUTION
# =============================================================================


def main() -> None:
    """Demonstrate exception handling concepts.
    
    Runs through examples from each section to verify
    implementations and illustrate usage patterns.
    """
    setup_logging(level=logging.INFO)
    logger.info("Starting 09UNIT Lab 01 demonstrations")
    
    # §1: Exception mechanism
    logger.info("§1: Exception Mechanism Fundamentals")
    hierarchy = demonstrate_exception_hierarchy()
    logger.info("Exception hierarchy categories: %s", list(hierarchy.keys()))
    
    result = safe_divide(10, 2)
    logger.info("safe_divide(10, 2) = %s", result)
    
    result = safe_divide(10, 0, default=float("inf"))
    logger.info("safe_divide(10, 0, default=inf) = %s", result)
    
    # §2: Custom exceptions
    logger.info("§2: Custom Exception Hierarchies")
    try:
        raise DataValidationError(
            "Invalid temperature value",
            field="temperature",
            value=-500,
            constraint="must be >= -273.15 (absolute zero)",
        )
    except DataValidationError as e:
        logger.info("Caught DataValidationError: %s", e.message)
        logger.info("  Field: %s, Value: %s", e.field, e.value)
    
    # §3: Context managers
    logger.info("§3: Context Managers")
    with Timer("example operation") as t:
        time.sleep(0.1)
    logger.info("Timer recorded: %.4f seconds", t.elapsed)
    
    with temporary_directory() as tmpdir:
        test_file = tmpdir / "test.txt"
        test_file.write_text("Hello, World!")
        logger.info("Created temp file: %s", test_file)
    logger.info("Temp directory cleaned up automatically")
    
    # §4: Exception patterns
    logger.info("§4: Exception Patterns")
    
    @retry_with_backoff(max_attempts=3, base_delay=0.1)
    def sometimes_fails() -> str:
        if random.random() < 0.3:
            raise ConnectionError("Simulated failure")
        return "success"
    
    try:
        result = sometimes_fails()
        logger.info("Operation succeeded: %s", result)
    except ConnectionError:
        logger.info("Operation failed after retries")
    
    # Circuit breaker demo
    breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5.0)
    logger.info("Circuit breaker state: %s", breaker.state)
    
    # §5: Logging integration
    logger.info("§5: Logging Integration")
    
    @logged_function(log_level=logging.INFO)
    def compute_sum(values: list[float]) -> float:
        return sum(values)
    
    total = compute_sum([1.0, 2.0, 3.0])
    logger.info("Computed sum: %s", total)
    
    logger.info("09UNIT Lab 01 demonstrations complete")


if __name__ == "__main__":
    main()
