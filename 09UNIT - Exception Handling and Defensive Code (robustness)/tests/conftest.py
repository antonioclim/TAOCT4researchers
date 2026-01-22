"""Pytest configuration and shared fixtures for 09UNIT tests.

This module provides common fixtures for testing exception handling
and defensive programming implementations.

Usage:
    pytest tests/
    pytest tests/ -v --cov=lab --cov-report=term-missing
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================


@pytest.fixture(autouse=True)
def configure_logging() -> Generator[None, None, None]:
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    yield


# =============================================================================
# TEMPORARY FILE FIXTURES
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary file path."""
    path = temp_dir / "test_file.txt"
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_json_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary JSON file path."""
    path = temp_dir / "test_config.json"
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def temp_csv_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary CSV file path."""
    path = temp_dir / "test_data.csv"
    yield path
    if path.exists():
        path.unlink()


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Provide sample configuration dictionary."""
    return {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db",
        },
        "logging": {
            "level": "INFO",
            "file": "/var/log/app.log",
        },
        "features": {
            "cache_enabled": True,
            "max_connections": 100,
        },
    }


@pytest.fixture
def valid_json_file(temp_json_file: Path, sample_config: dict[str, Any]) -> Path:
    """Create a valid JSON configuration file."""
    temp_json_file.write_text(json.dumps(sample_config, indent=2))
    return temp_json_file


@pytest.fixture
def invalid_json_file(temp_json_file: Path) -> Path:
    """Create an invalid JSON file."""
    temp_json_file.write_text("{ invalid json content")
    return temp_json_file


@pytest.fixture
def sample_csv_content() -> str:
    """Provide sample CSV content."""
    return """name,age,email
Alice,30,alice@example.com
Bob,25,bob@example.com
Charlie,35,charlie@example.com
"""


@pytest.fixture
def valid_csv_file(temp_csv_file: Path, sample_csv_content: str) -> Path:
    """Create a valid CSV file."""
    temp_csv_file.write_text(sample_csv_content)
    return temp_csv_file


@pytest.fixture
def malformed_csv_file(temp_csv_file: Path) -> Path:
    """Create a malformed CSV file."""
    content = """name,age,email
Alice,30
Bob,25,bob@example.com,extra_field
"""
    temp_csv_file.write_text(content)
    return temp_csv_file


# =============================================================================
# NUMERIC DATA FIXTURES
# =============================================================================


@pytest.fixture
def float_values() -> list[float]:
    """Provide list of float values for numerical tests."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.fixture
def values_with_nan() -> list[float]:
    """Provide list containing NaN values."""
    return [1.0, 2.0, float("nan"), 4.0, 5.0]


@pytest.fixture
def values_with_inf() -> list[float]:
    """Provide list containing infinite values."""
    return [1.0, float("inf"), 3.0, float("-inf"), 5.0]


# =============================================================================
# EXCEPTION TESTING HELPERS
# =============================================================================


@pytest.fixture
def failing_function() -> Any:
    """Provide a function that always fails."""
    def _fail() -> None:
        raise ValueError("Intentional failure")
    return _fail


@pytest.fixture
def flaky_function() -> Any:
    """Provide a function that fails intermittently."""
    call_count = [0]
    
    def _flaky() -> str:
        call_count[0] += 1
        if call_count[0] < 3:
            raise ConnectionError(f"Attempt {call_count[0]} failed")
        return "success"
    
    return _flaky


@pytest.fixture
def counting_function() -> Any:
    """Provide a function that counts its calls."""
    call_count = [0]
    
    def _count() -> int:
        call_count[0] += 1
        return call_count[0]
    
    return _count


# =============================================================================
# CONTEXT MANAGER TEST HELPERS
# =============================================================================


@pytest.fixture
def mock_database_connection() -> dict[str, Any]:
    """Provide a mock database connection object."""
    return {
        "connected": False,
        "committed": False,
        "rolled_back": False,
        "queries": [],
    }


# =============================================================================
# VALIDATION TEST DATA
# =============================================================================


@pytest.fixture
def valid_user_data() -> dict[str, Any]:
    """Provide valid user data for validation tests."""
    return {
        "name": "Alice Smith",
        "age": 30,
        "email": "alice@example.com",
        "active": True,
    }


@pytest.fixture
def invalid_user_data() -> dict[str, Any]:
    """Provide invalid user data for validation tests."""
    return {
        "name": "",  # Empty
        "age": -5,  # Negative
        "email": "invalid-email",  # No @
        "active": "yes",  # Wrong type
    }


# =============================================================================
# CHECKPOINT TEST HELPERS
# =============================================================================


@pytest.fixture
def checkpoint_path(temp_dir: Path) -> Path:
    """Provide a path for checkpoint files."""
    return temp_dir / "checkpoint.json"


@pytest.fixture
def items_to_process() -> list[int]:
    """Provide items for batch processing tests."""
    return list(range(1, 11))


# =============================================================================
# CUSTOM MARKERS
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test",
    )
