#!/usr/bin/env python3
"""Solutions for Lab 09-01: Exception Handling.

Complete reference implementations for all laboratory exercises.
These solutions demonstrate established conventions for exception handling
in Python scientific computing applications.

IMPORTANT: This file is for instructor reference only.
Students should attempt exercises independently before consulting.
"""

from __future__ import annotations

import json
import logging
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

T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# §1: EXCEPTION HIERARCHY SOLUTIONS
# =============================================================================


class ResearchError(Exception):
    """Base exception for all research-related errors.
    
    All domain-specific exceptions inherit from this class,
    enabling categorical exception handling.
    
    Attributes:
        message: Human-readable error description.
        details: Additional context as key-value pairs.
    """
    
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class DataValidationError(ResearchError):
    """Raised when data fails validation constraints."""
    
    def __init__(
        self,
        message: str,
        field_name: str,
        expected_type: type,
        actual_value: Any,
    ) -> None:
        details = {
            "field": field_name,
            "expected": expected_type.__name__,
            "actual": type(actual_value).__name__,
        }
        super().__init__(message, details)
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value


class FileFormatError(ResearchError):
    """Raised when file format is invalid or corrupted."""
    
    def __init__(
        self,
        message: str,
        file_path: Path,
        line_number: int,
        expected_format: str,
    ) -> None:
        details = {
            "file": str(file_path),
            "line": line_number,
            "expected_format": expected_format,
        }
        super().__init__(message, details)
        self.file_path = file_path
        self.line_number = line_number
        self.expected_format = expected_format


class ConfigurationError(ResearchError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str | None = None) -> None:
        details = {"key": config_key} if config_key else {}
        super().__init__(message, details)
        self.config_key = config_key


class ComputationError(ResearchError):
    """Raised when a computation fails."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        input_shape: tuple[int, ...] | None = None,
    ) -> None:
        details = {"operation": operation}
        if input_shape:
            details["input_shape"] = input_shape
        super().__init__(message, details)
        self.operation = operation
        self.input_shape = input_shape


class NumericalInstabilityError(ComputationError):
    """Raised when numerical instability is detected."""
    
    def __init__(
        self,
        message: str,
        value: float,
        threshold: float,
    ) -> None:
        super().__init__(message, "numerical_check")
        self.value = value
        self.threshold = threshold
        self.details.update({"value": value, "threshold": threshold})


class ConvergenceError(ComputationError):
    """Raised when iterative algorithm fails to converge."""
    
    def __init__(
        self,
        message: str,
        iterations: int,
        final_error: float,
        tolerance: float,
    ) -> None:
        super().__init__(message, "convergence_check")
        self.iterations = iterations
        self.final_error = final_error
        self.tolerance = tolerance
        self.details.update({
            "iterations": iterations,
            "final_error": final_error,
            "tolerance": tolerance,
        })


class ContractViolationError(ResearchError):
    """Raised when a design by contract assertion fails."""
    
    def __init__(self, message: str, contract_type: str) -> None:
        super().__init__(message, {"contract_type": contract_type})
        self.contract_type = contract_type


class CircuitOpenError(ResearchError):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, reset_time: float) -> None:
        super().__init__(message, {"reset_in_seconds": reset_time})
        self.reset_time = reset_time


# =============================================================================
# §2: SAFE OPERATIONS SOLUTIONS
# =============================================================================


def safe_divide(
    numerator: float,
    denominator: float,
    default: float | None = None,
) -> float | None:
    """Safely divide two numbers, handling division by zero.
    
    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value to return if division fails.
        
    Returns:
        Result of division or default value.
    """
    try:
        return numerator / denominator
    except ZeroDivisionError:
        logger.warning(
            "Division by zero attempted: %s / %s",
            numerator,
            denominator,
        )
        return default


def parse_config(config_path: Path) -> dict[str, Any] | None:
    """Parse JSON configuration file with error handling.
    
    Args:
        config_path: Path to JSON configuration file.
        
    Returns:
        Parsed configuration dictionary or None on error.
    """
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        return None
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in %s: %s", config_path, e)
        return None
    except PermissionError:
        logger.error("Permission denied reading: %s", config_path)
        return None


def demonstrate_exception_hierarchy() -> None:
    """Demonstrate the exception hierarchy with examples."""
    exceptions = [
        ResearchError("Base research error"),
        DataValidationError("Invalid temperature", "temp", float, "cold"),
        FileFormatError("Malformed CSV", Path("data.csv"), 42, "CSV"),
        ComputationError("Matrix singular", "inverse", (3, 3)),
        NumericalInstabilityError("Overflow", float("inf"), 1e308),
        ConvergenceError("Did not converge", 1000, 0.01, 0.001),
    ]
    
    for exc in exceptions:
        logger.info("Exception: %s", exc)


def exception_chaining_demo() -> None:
    """Demonstrate exception chaining."""
    try:
        try:
            raise FileNotFoundError("config.json not found")
        except FileNotFoundError as e:
            raise ConfigurationError("Failed to load configuration") from e
    except ConfigurationError:
        raise


# =============================================================================
# §3: CONTEXT MANAGER SOLUTIONS
# =============================================================================


class ManagedFile:
    """Context manager for file operations with logging.
    
    Ensures files are properly closed and logs all operations.
    """
    
    def __init__(self, path: Path | str, mode: str = "r") -> None:
        self.path = Path(path)
        self.mode = mode
        self._file: IO[Any] | None = None
    
    def __enter__(self) -> IO[Any]:
        logger.debug("Opening file: %s (mode=%s)", self.path, self.mode)
        self._file = open(self.path, self.mode)
        return self._file
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self._file:
            self._file.close()
            logger.debug("Closed file: %s", self.path)
        
        if exc_type is not None:
            logger.error(
                "Exception during file operation on %s: %s",
                self.path,
                exc_val,
            )
        
        return False  # Don't suppress exceptions


@contextmanager
def temporary_directory(
    prefix: str = "research_",
    cleanup: bool = True,
) -> Generator[Path, None, None]:
    """Create temporary directory with automatic cleanup.
    
    Args:
        prefix: Directory name prefix.
        cleanup: Whether to remove directory on exit.
        
    Yields:
        Path to temporary directory.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix))
    logger.debug("Created temporary directory: %s", temp_dir)
    
    try:
        yield temp_dir
    finally:
        if cleanup and temp_dir.exists():
            shutil.rmtree(temp_dir)
            logger.debug("Removed temporary directory: %s", temp_dir)


class DatabaseConnection:
    """Simulated database connection context manager."""
    
    def __init__(self, db_name: str) -> None:
        self.db_name = db_name
        self._connected = False
        self._queries: list[str] = []
    
    def __enter__(self) -> DatabaseConnection:
        self._connected = True
        logger.info("Connected to database: %s", self.db_name)
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            logger.warning("Rolling back transaction due to: %s", exc_val)
            self._rollback()
        else:
            logger.info("Committing transaction")
            self._commit()
        
        self._connected = False
        logger.info("Disconnected from database: %s", self.db_name)
        return False
    
    def execute(self, query: str) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to database")
        self._queries.append(query)
        logger.debug("Executed query: %s", query)
    
    def _commit(self) -> None:
        logger.debug("Committed %d queries", len(self._queries))
        self._queries.clear()
    
    def _rollback(self) -> None:
        logger.debug("Rolled back %d queries", len(self._queries))
        self._queries.clear()


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "operation") -> None:
        self.name = name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
    
    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        logger.debug("Starting timer: %s", self.name)
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.elapsed = time.perf_counter() - self.start_time
        logger.info("Timer %s: %.4f seconds", self.name, self.elapsed)
        return False


def exitstack_demo(file_paths: list[Path]) -> None:
    """Demonstrate ExitStack for managing multiple contexts."""
    with ExitStack() as stack:
        files = [
            stack.enter_context(open(path))
            for path in file_paths
            if path.exists()
        ]
        
        for f in files:
            logger.debug("Processing: %s", f.name)


# =============================================================================
# §4: RESILIENCE PATTERNS SOLUTIONS
# =============================================================================


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Execute function with exponential backoff retry.
    
    Args:
        func: Function to execute.
        max_attempts: Maximum retry attempts.
        base_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        retryable_exceptions: Exception types to retry.
        on_retry: Callback for each retry.
        
    Returns:
        Function result on success.
        
    Raises:
        Last exception if all attempts fail.
    """
    import random
    
    last_exception: Exception | None = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            
            if attempt == max_attempts:
                logger.error("All %d attempts failed", max_attempts)
                raise
            
            # Calculate delay with jitter
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 0.5 + random.random() * 0.5
            
            logger.warning(
                "Attempt %d/%d failed: %s. Retrying in %.2fs",
                attempt,
                max_attempts,
                e,
                delay,
            )
            
            if on_retry:
                on_retry(attempt, e)
            
            time.sleep(delay)
    
    # Should not reach here, but satisfy type checker
    raise last_exception  # type: ignore[misc]


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault-tolerant service calls.
    
    States:
        closed: Normal operation
        open: Failing fast (rejecting calls)
        half_open: Testing recovery
    """
    
    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout: float = 30.0
    
    _state: str = field(default="closed", init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    
    @property
    def state(self) -> str:
        return self._state
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker."""
        if self._state == "open":
            if self._should_attempt_reset():
                self._state = "half_open"
                logger.info("Circuit breaker transitioning to half-open")
            else:
                time_remaining = self.reset_timeout - (
                    time.time() - self._last_failure_time
                )
                raise CircuitOpenError(
                    "Circuit breaker is open",
                    max(0.0, time_remaining),
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        return time.time() - self._last_failure_time >= self.reset_timeout
    
    def _on_success(self) -> None:
        self._failure_count = 0
        
        if self._state == "half_open":
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state = "closed"
                self._success_count = 0
                logger.info("Circuit breaker closed")
    
    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        self._success_count = 0
        
        if self._state == "half_open":
            self._state = "open"
            logger.warning("Circuit breaker reopened")
        elif self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning("Circuit breaker opened after %d failures", self._failure_count)


def graceful_degradation(
    primary: Callable[[], T],
    fallback: Callable[[], T],
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute primary function with fallback on failure."""
    try:
        return primary()
    except exceptions as e:
        logger.warning("Primary failed (%s), using fallback", e)
        return fallback()


def bulk_operation_with_partial_failure(
    items: list[T],
    processor: Callable[[T], Any],
    continue_on_error: bool = True,
) -> dict[str, list[Any]]:
    """Process items, collecting successes and failures."""
    results: dict[str, list[Any]] = {
        "successful": [],
        "failed": [],
    }
    
    for item in items:
        try:
            result = processor(item)
            results["successful"].append({"item": item, "result": result})
        except Exception as e:
            results["failed"].append({"item": item, "error": str(e)})
            if not continue_on_error:
                raise
    
    return results


# =============================================================================
# §5: LOGGING INTEGRATION SOLUTIONS
# =============================================================================


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Configure logging with optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def logged_function(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that logs function calls and exceptions."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        func_logger = logging.getLogger(func.__module__)
        func_logger.debug("Calling %s", func.__name__)
        
        try:
            result = func(*args, **kwargs)
            func_logger.debug("%s completed successfully", func.__name__)
            return result
        except Exception as e:
            func_logger.exception("%s raised %s: %s", func.__name__, type(e).__name__, e)
            raise
    
    return wrapper
