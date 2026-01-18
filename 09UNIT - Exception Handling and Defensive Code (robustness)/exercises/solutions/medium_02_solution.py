#!/usr/bin/env python3
"""Solution for Exercise 05: Context Managers.

This module provides reference implementations for class-based
and decorator-based context managers.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Generator

logger = logging.getLogger(__name__)


class LoggedOperation:
    """Context manager that logs operation start, end and duration."""
    
    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.start_time: float = 0.0
        self.elapsed: float = 0.0
    
    def __enter__(self) -> LoggedOperation:
        self.start_time = time.perf_counter()
        logger.info("Starting operation: %s", self.operation_name)
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        self.elapsed = time.perf_counter() - self.start_time
        
        if exc_type is not None:
            logger.error(
                "Failed operation: %s in %.3fs - %s",
                self.operation_name,
                self.elapsed,
                exc_type.__name__,
            )
        else:
            logger.info(
                "Completed operation: %s in %.3fs",
                self.operation_name,
                self.elapsed,
            )
        
        return False  # Don't suppress exceptions


class TemporaryFile:
    """Context manager for temporary file with automatic cleanup."""
    
    def __init__(self, suffix: str = ".tmp", prefix: str = "temp_") -> None:
        self.suffix = suffix
        self.prefix = prefix
        self.path: Path | None = None
    
    def __enter__(self) -> TemporaryFile:
        fd, path_str = tempfile.mkstemp(suffix=self.suffix, prefix=self.prefix)
        os.close(fd)  # Close the file descriptor, we just need the path
        self.path = Path(path_str)
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        if self.path is not None and self.path.exists():
            self.path.unlink()
        return False


@contextmanager
def working_directory(path: Path) -> Generator[Path, None, None]:
    """Context manager that temporarily changes working directory."""
    original = os.getcwd()
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(original)


@contextmanager
def suppress_exceptions(
    *exception_types: type[Exception],
) -> Generator[list[Exception], None, None]:
    """Context manager that suppresses specified exception types."""
    errors: list[Exception] = []
    try:
        yield errors
    except exception_types as e:
        errors.append(e)


@contextmanager
def transaction(connection: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
    """Context manager for simulated database transaction."""
    try:
        yield connection
        connection["committed"] = True
    except Exception:
        connection["rolled_back"] = True
        raise


def process_with_logging_and_temp(
    data: str,
    operation_name: str,
) -> tuple[str, float]:
    """Process data using combined context managers."""
    with LoggedOperation(operation_name) as op:
        with TemporaryFile(suffix=".txt") as tf:
            assert tf.path is not None
            tf.path.write_text(data)
            content = tf.path.read_text()
            result = content.upper()
    
    return result, op.elapsed


def main() -> None:
    """Verify solution implementations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing LoggedOperation...")
    with LoggedOperation("test operation") as op:
        time.sleep(0.05)
    assert op.elapsed >= 0.05
    
    try:
        with LoggedOperation("failing operation"):
            raise ValueError("intentional")
    except ValueError:
        pass
    print("  ✓ LoggedOperation passed")
    
    print("Testing TemporaryFile...")
    with TemporaryFile(suffix=".txt") as tf:
        assert tf.path is not None
        tf.path.write_text("test content")
        assert tf.path.exists()
        temp_path = tf.path
    assert not temp_path.exists()
    print("  ✓ TemporaryFile passed")
    
    print("Testing working_directory...")
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
    
    try:
        with suppress_exceptions(ValueError) as errors:
            raise KeyError("not suppressed")
        assert False
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
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
