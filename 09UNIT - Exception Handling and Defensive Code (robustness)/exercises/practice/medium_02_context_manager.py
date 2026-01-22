#!/usr/bin/env python3
"""Exercise 05: Context Managers (Medium).

This exercise develops skills in implementing context managers
for resource management and cleanup.

Learning Objectives:
    - Implement class-based context managers with __enter__/__exit__
    - Use @contextmanager decorator for generator-based managers
    - Handle exceptions in context manager exit

Estimated Time: 15 minutes
Difficulty: Medium (★★☆)

Instructions:
    1. Complete both class-based and decorator-based context managers
    2. Ensure proper cleanup in all scenarios
    3. Run tests to verify your implementation
"""

from __future__ import annotations

import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Generator

logger = logging.getLogger(__name__)


# =============================================================================
# TASK 1: Class-Based Context Manager
# =============================================================================

class LoggedOperation:
    """Context manager that logs operation start, end and duration.
    
    Logs when an operation starts and when it completes (with duration),
    regardless of whether the operation succeeds or fails.
    
    Attributes:
        operation_name: Name of the operation for logging.
        start_time: When the operation started.
        elapsed: Duration in seconds after completion.
        
    Example:
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> with LoggedOperation("data processing") as op:
        ...     time.sleep(0.1)
        ...     # Logs: "Starting operation: data processing"
        ...     # Logs: "Completed operation: data processing in 0.100s"
        >>> op.elapsed > 0.1
        True
    """
    
    def __init__(self, operation_name: str) -> None:
        """Initialise logged operation.
        
        Args:
            operation_name: Name for log messages.
        """
        # TODO: Store operation_name, initialise start_time and elapsed to 0.0
        pass
    
    def __enter__(self) -> LoggedOperation:
        """Start timing and log operation start.
        
        Returns:
            Self for accessing elapsed time later.
        """
        # TODO: Record start_time using time.perf_counter()
        # TODO: Log info message "Starting operation: {operation_name}"
        # TODO: Return self
        pass
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Stop timing and log completion.
        
        Args:
            exc_type: Exception type if exception occurred.
            exc_val: Exception instance if exception occurred.
            exc_tb: Traceback if exception occurred.
            
        Returns:
            False to propagate any exceptions.
        """
        # TODO: Calculate elapsed time
        # TODO: Log different messages for success vs failure
        # - Success: "Completed operation: {name} in {elapsed:.3f}s"
        # - Failure: "Failed operation: {name} in {elapsed:.3f}s - {exc_type.__name__}"
        # TODO: Return False (don't suppress exceptions)
        pass


class TemporaryFile:
    """Context manager for temporary file with automatic cleanup.
    
    Creates a temporary file that is automatically deleted when
    the context exits, even if an exception occurs.
    
    Attributes:
        path: Path to the temporary file.
        suffix: File suffix (e.g., ".txt", ".csv").
        
    Example:
        >>> with TemporaryFile(suffix=".txt") as tf:
        ...     tf.path.write_text("hello")
        ...     tf.path.exists()
        True
        >>> tf.path.exists()  # Cleaned up
        False
    """
    
    def __init__(self, suffix: str = ".tmp", prefix: str = "temp_") -> None:
        """Initialise temporary file manager.
        
        Args:
            suffix: File extension.
            prefix: File name prefix.
        """
        # TODO: Store suffix and prefix
        # TODO: Initialise path to None
        pass
    
    def __enter__(self) -> TemporaryFile:
        """Create temporary file and return manager.
        
        Returns:
            Self with path set to temporary file.
        """
        # TODO: Create temporary file using tempfile.NamedTemporaryFile
        # - Set delete=False (we'll handle deletion in __exit__)
        # - Use self.suffix and self.prefix
        # TODO: Store path, close the file handle (we just need the path)
        # TODO: Return self
        pass
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Delete temporary file.
        
        Returns:
            False to propagate any exceptions.
        """
        # TODO: Delete the file if it exists
        # TODO: Return False
        pass


# =============================================================================
# TASK 2: Decorator-Based Context Manager
# =============================================================================

@contextmanager
def working_directory(path: Path) -> Generator[Path, None, None]:
    """Context manager that temporarily changes working directory.
    
    Changes to the specified directory for the duration of the
    context, then restores the original directory.
    
    Args:
        path: Directory to change to.
        
    Yields:
        The path that was changed to.
        
    Example:
        >>> import os
        >>> original = Path.cwd()
        >>> with working_directory(Path("/tmp")) as wd:
        ...     Path.cwd() == Path("/tmp")
        True
        >>> Path.cwd() == original
        True
    """
    import os
    
    # TODO: Store original directory using os.getcwd()
    # TODO: Change to new directory using os.chdir(path)
    # TODO: try/finally to ensure original is restored
    # TODO: yield the path
    pass


@contextmanager
def suppress_exceptions(*exception_types: type[Exception]) -> Generator[list[Exception], None, None]:
    """Context manager that suppresses specified exception types.
    
    Catches specified exceptions and stores them instead of raising.
    Other exception types propagate normally.
    
    Args:
        *exception_types: Exception types to suppress.
        
    Yields:
        List that will contain any suppressed exceptions.
        
    Example:
        >>> with suppress_exceptions(ValueError, TypeError) as errors:
        ...     raise ValueError("test error")
        >>> len(errors)
        1
        >>> isinstance(errors[0], ValueError)
        True
    """
    # TODO: Create empty list for suppressed exceptions
    # TODO: try/except to catch specified types
    # TODO: Append caught exceptions to list
    # TODO: yield the exceptions list
    pass


@contextmanager
def transaction(connection: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """Context manager for simulated database transaction.
    
    Simulates a database transaction with commit on success
    and rollback on failure.
    
    Args:
        connection: Dictionary simulating database connection.
                   Should have 'committed' and 'rolled_back' keys.
        
    Yields:
        The connection dictionary.
        
    Example:
        >>> conn = {"committed": False, "rolled_back": False}
        >>> with transaction(conn):
        ...     pass  # Success
        >>> conn["committed"]
        True
        >>> conn = {"committed": False, "rolled_back": False}
        >>> try:
        ...     with transaction(conn):
        ...         raise ValueError("failed")
        ... except ValueError:
        ...     pass
        >>> conn["rolled_back"]
        True
    """
    # TODO: yield connection inside try/except
    # TODO: On success (no exception): set connection["committed"] = True
    # TODO: On exception: set connection["rolled_back"] = True, then re-raise
    pass


# =============================================================================
# TASK 3: Combining Context Managers
# =============================================================================

def process_with_logging_and_temp(
    data: str,
    operation_name: str,
) -> tuple[str, float]:
    """Process data using combined context managers.
    
    Uses LoggedOperation for timing and TemporaryFile for
    intermediate storage.
    
    Args:
        data: Input data string.
        operation_name: Name for logging.
        
    Returns:
        Tuple of (processed_data, elapsed_time).
        
    Example:
        >>> result, elapsed = process_with_logging_and_temp("hello", "test")
        >>> result
        'HELLO'
        >>> elapsed > 0
        True
    """
    # TODO: Use LoggedOperation and TemporaryFile together
    # 1. Create LoggedOperation context
    # 2. Create TemporaryFile context (nested)
    # 3. Write data to temp file
    # 4. Read data back and convert to uppercase
    # 5. Return (uppercase_data, operation.elapsed)
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing LoggedOperation...")
    with LoggedOperation("test operation") as op:
        time.sleep(0.05)
    assert op.elapsed >= 0.05
    
    try:
        with LoggedOperation("failing operation"):
            raise ValueError("intentional")
    except ValueError:
        pass  # Expected
    print("  ✓ LoggedOperation passed")
    
    print("Testing TemporaryFile...")
    with TemporaryFile(suffix=".txt") as tf:
        tf.path.write_text("test content")
        assert tf.path.exists()
        temp_path = tf.path
    assert not temp_path.exists()  # Should be cleaned up
    print("  ✓ TemporaryFile passed")
    
    print("Testing working_directory...")
    import os
    original = Path.cwd()
    with working_directory(Path(tempfile.gettempdir())) as wd:
        assert Path.cwd() != original
    assert Path.cwd() == original
    print("  ✓ working_directory passed")
    
    print("Testing suppress_exceptions...")
    with suppress_exceptions(ValueError, TypeError) as errors:
        raise ValueError("test")
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    
    # Non-suppressed exception should propagate
    try:
        with suppress_exceptions(ValueError) as errors:
            raise KeyError("not suppressed")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    print("  ✓ suppress_exceptions passed")
    
    print("Testing transaction...")
    conn = {"committed": False, "rolled_back": False}
    with transaction(conn):
        pass
    assert conn["committed"]
    assert not conn["rolled_back"]
    
    conn = {"committed": False, "rolled_back": False}
    try:
        with transaction(conn):
            raise ValueError("failed")
    except ValueError:
        pass
    assert not conn["committed"]
    assert conn["rolled_back"]
    print("  ✓ transaction passed")
    
    print("Testing process_with_logging_and_temp...")
    result, elapsed = process_with_logging_and_temp("hello world", "uppercase")
    assert result == "HELLO WORLD"
    assert elapsed > 0
    print("  ✓ process_with_logging_and_temp passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
