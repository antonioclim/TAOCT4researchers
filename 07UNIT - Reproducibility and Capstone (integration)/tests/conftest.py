#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7: Pytest Configuration and Shared Fixtures
═══════════════════════════════════════════════════════════════════════════════

This module provides shared fixtures and configuration for the Week 7 test
suite, demonstrating best practices for reproducible testing.

FIXTURES PROVIDED
─────────────────
- set_random_seeds: Auto-use fixture for reproducible randomness
- temp_project_dir: Temporary directory for project scaffolding tests
- sample_config: Sample configuration dictionary
- mock_git_repo: Mocked git repository for version control tests
- captured_logs: Fixture for capturing log output
- sample_test_data: Sample data for testing reproducibility tools

MARKERS
───────
- @pytest.mark.slow: Tests that take longer than 1 second
- @pytest.mark.integration: Integration tests requiring external resources
- @pytest.mark.reproducibility: Tests specifically for reproducibility features

DEPENDENCIES
────────────
pytest>=7.0
pytest-cov>=4.0
numpy>=1.24

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

# Attempt numpy import for numerical reproducibility
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

RANDOM_SEED = 42
TEST_PROJECT_NAME = "test_research_project"
TEST_AUTHOR = "Dr Test Author"
TEST_PYTHON_VERSION = "3.12"


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for the test suite."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers",
        "reproducibility: marks tests specifically for reproducibility features"
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item]
) -> None:
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Auto-mark slow tests
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Auto-mark integration tests
        if "integration" in item.name or "e2e" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark reproducibility tests
        if "reproducib" in item.name or "seed" in item.name:
            item.add_marker(pytest.mark.reproducibility)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CORE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def set_random_seeds() -> Generator[None, None, None]:
    """
    Set random seeds before each test for reproducibility.
    
    This fixture runs automatically before every test to ensure
    deterministic behaviour in tests involving randomness.
    
    Yields:
        None: Control returns to test after seed setting.
    
    Example:
        >>> # Seeds are automatically set - no action needed
        >>> random.random()  # Will always return same value
        0.6394267984578837
    """
    # Python random
    random.seed(RANDOM_SEED)
    
    # NumPy random (if available)
    if HAS_NUMPY:
        np.random.seed(RANDOM_SEED)
    
    # Environment variable for subprocess reproducibility
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    
    yield
    
    # Cleanup: restore environment
    if "PYTHONHASHSEED" in os.environ:
        del os.environ["PYTHONHASHSEED"]


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary directory for project scaffolding tests.
    
    This fixture provides an isolated directory that is automatically
    cleaned up after the test completes.
    
    Args:
        tmp_path: Built-in pytest fixture for temporary paths.
    
    Yields:
        Path: Path to the temporary project directory.
    
    Example:
        >>> def test_create_project(temp_project_dir):
        ...     project = temp_project_dir / "my_project"
        ...     project.mkdir()
        ...     assert project.exists()
    """
    project_dir = tmp_path / "projects"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Store original working directory
    original_cwd = Path.cwd()
    
    yield project_dir
    
    # Cleanup
    os.chdir(original_cwd)
    if project_dir.exists():
        shutil.rmtree(project_dir)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    Create a temporary output directory for test artifacts.
    
    Args:
        tmp_path: Built-in pytest fixture for temporary paths.
    
    Returns:
        Path: Path to the temporary output directory.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProjectConfig:
    """Configuration for a test project."""
    
    name: str
    author: str
    python_version: str
    include_tests: bool = True
    include_docs: bool = True
    include_cicd: bool = True
    include_docker: bool = False
    dependencies: list[str] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "name": self.name,
            "author": self.author,
            "python_version": self.python_version,
            "include_tests": self.include_tests,
            "include_docs": self.include_docs,
            "include_cicd": self.include_cicd,
            "include_docker": self.include_docker,
            "dependencies": self.dependencies or [],
        }


@pytest.fixture
def sample_config() -> ProjectConfig:
    """
    Provide a sample project configuration for testing.
    
    Returns:
        ProjectConfig: A fully populated test configuration.
    
    Example:
        >>> def test_scaffolder(sample_config):
        ...     assert sample_config.name == "test_research_project"
        ...     assert sample_config.include_tests is True
    """
    return ProjectConfig(
        name=TEST_PROJECT_NAME,
        author=TEST_AUTHOR,
        python_version=TEST_PYTHON_VERSION,
        include_tests=True,
        include_docs=True,
        include_cicd=True,
        include_docker=True,
        dependencies=["numpy", "pandas", "matplotlib"],
    )


@pytest.fixture
def minimal_config() -> ProjectConfig:
    """
    Provide a minimal project configuration for testing.
    
    Returns:
        ProjectConfig: A minimal test configuration with features disabled.
    """
    return ProjectConfig(
        name="minimal_project",
        author="Test Author",
        python_version="3.12",
        include_tests=False,
        include_docs=False,
        include_cicd=False,
        include_docker=False,
        dependencies=[],
    )


@pytest.fixture
def sample_config_dict(sample_config: ProjectConfig) -> dict[str, Any]:
    """
    Provide sample configuration as a dictionary.
    
    Args:
        sample_config: The sample ProjectConfig fixture.
    
    Returns:
        Dictionary representation of the configuration.
    """
    return sample_config.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MOCK FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_git_repo(temp_project_dir: Path) -> Generator[MagicMock, None, None]:
    """
    Provide a mocked git repository for version control tests.
    
    This fixture mocks git operations to avoid requiring an actual
    git installation during testing.
    
    Args:
        temp_project_dir: Temporary directory fixture.
    
    Yields:
        MagicMock: Mocked git repository object.
    
    Example:
        >>> def test_git_operations(mock_git_repo):
        ...     mock_git_repo.commit.return_value = "abc123"
        ...     result = mock_git_repo.commit("Test message")
        ...     assert result == "abc123"
    """
    mock_repo = MagicMock()
    mock_repo.working_dir = str(temp_project_dir)
    mock_repo.head.commit.hexsha = "a" * 40
    mock_repo.head.commit.message = "Initial commit"
    mock_repo.head.commit.committed_datetime = datetime.now()
    mock_repo.is_dirty.return_value = False
    mock_repo.untracked_files = []
    
    # Mock common git operations
    mock_repo.git.status.return_value = "On branch main\nnothing to commit"
    mock_repo.git.log.return_value = "commit aaaa...\nAuthor: Test\nDate: Today"
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        yield mock_repo


@pytest.fixture
def mock_subprocess() -> Generator[MagicMock, None, None]:
    """
    Mock subprocess calls for testing CLI interactions.
    
    Yields:
        MagicMock: Mocked subprocess.run function.
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Success",
            stderr=""
        )
        yield mock_run


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LOGGING FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def captured_logs(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """
    Capture log output during tests.
    
    Args:
        caplog: Built-in pytest log capture fixture.
    
    Returns:
        LogCaptureFixture: Configured log capture fixture.
    
    Example:
        >>> def test_logging(captured_logs):
        ...     logging.info("Test message")
        ...     assert "Test message" in captured_logs.text
    """
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def logger() -> logging.Logger:
    """
    Provide a configured logger for testing.
    
    Returns:
        Logger: A test logger with DEBUG level.
    """
    test_logger = logging.getLogger("test_week7")
    test_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    test_logger.handlers.clear()
    
    # Add a null handler to prevent "no handler" warnings
    test_logger.addHandler(logging.NullHandler())
    
    return test_logger


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_test_data() -> dict[str, Any]:
    """
    Provide sample data for testing reproducibility tools.
    
    Returns:
        Dictionary containing various test data types.
    
    Example:
        >>> def test_data_processing(sample_test_data):
        ...     assert len(sample_test_data["values"]) == 10
        ...     assert "metadata" in sample_test_data
    """
    return {
        "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "labels": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "matrix": [[1, 2], [3, 4], [5, 6]],
        "metadata": {
            "created": "2025-01-15T10:00:00Z",
            "version": "1.0.0",
            "seed": RANDOM_SEED,
        },
        "nested": {
            "level1": {
                "level2": {
                    "value": 42
                }
            }
        }
    }


@pytest.fixture
def sample_csv_content() -> str:
    """
    Provide sample CSV content for data manifest tests.
    
    Returns:
        String containing CSV data.
    """
    return """id,name,value,category
1,alpha,10.5,A
2,beta,20.3,B
3,gamma,15.7,A
4,delta,25.1,C
5,epsilon,18.9,B
"""


@pytest.fixture
def sample_json_content() -> str:
    """
    Provide sample JSON content for testing.
    
    Returns:
        String containing JSON data.
    """
    return json.dumps({
        "experiment": "test_001",
        "parameters": {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 100
        },
        "results": {
            "accuracy": 0.95,
            "loss": 0.05
        }
    }, indent=2)


@pytest.fixture
def sample_data_file(tmp_path: Path, sample_csv_content: str) -> Path:
    """
    Create a sample data file for testing.
    
    Args:
        tmp_path: Temporary path fixture.
        sample_csv_content: Sample CSV content fixture.
    
    Returns:
        Path to the created data file.
    """
    data_file = tmp_path / "sample_data.csv"
    data_file.write_text(sample_csv_content)
    return data_file


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: HASH AND CHECKSUM FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def file_hasher():
    """
    Provide a file hashing utility for testing.
    
    Returns:
        Callable that computes SHA-256 hash of a file.
    """
    def hash_file(filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    return hash_file


@pytest.fixture
def content_hasher():
    """
    Provide a content hashing utility for testing.
    
    Returns:
        Callable that computes SHA-256 hash of content.
    """
    def hash_content(content: str | bytes) -> str:
        """Compute SHA-256 hash of content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()
    
    return hash_content


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CI/CD FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_workflow_config() -> dict[str, Any]:
    """
    Provide a sample GitHub Actions workflow configuration.
    
    Returns:
        Dictionary containing workflow configuration.
    """
    return {
        "name": "CI",
        "on": {
            "push": {"branches": ["main"]},
            "pull_request": {"branches": ["main"]}
        },
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Python",
                        "uses": "actions/setup-python@v5",
                        "with": {"python-version": "3.12"}
                    },
                    {
                        "name": "Run tests",
                        "run": "pytest --cov"
                    }
                ]
            }
        }
    }


@pytest.fixture
def github_actions_dir(temp_project_dir: Path) -> Path:
    """
    Create a .github/workflows directory structure.
    
    Args:
        temp_project_dir: Temporary project directory fixture.
    
    Returns:
        Path to the workflows directory.
    """
    workflows_dir = temp_project_dir / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    return workflows_dir


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: PARAMETRISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Common test parameters for reproducibility tests
SEED_VALUES = [0, 1, 42, 123, 999, 2**31 - 1]

# Python versions for compatibility tests
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]

# Common project structures
PROJECT_STRUCTURES = [
    {"include_tests": True, "include_docs": True, "include_cicd": True},
    {"include_tests": True, "include_docs": False, "include_cicd": False},
    {"include_tests": False, "include_docs": True, "include_cicd": True},
]


@pytest.fixture(params=SEED_VALUES)
def various_seeds(request: pytest.FixtureRequest) -> int:
    """
    Parametrised fixture providing various seed values.
    
    Args:
        request: Pytest request object.
    
    Returns:
        A seed value for testing.
    """
    return request.param


@pytest.fixture(params=PROJECT_STRUCTURES)
def various_structures(request: pytest.FixtureRequest) -> dict[str, bool]:
    """
    Parametrised fixture providing various project structures.
    
    Args:
        request: Pytest request object.
    
    Returns:
        Dictionary of project structure options.
    """
    return request.param


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: ASSERTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def assert_file_exists():
    """
    Provide an assertion helper for file existence checks.
    
    Returns:
        Callable that asserts a file exists and optionally checks content.
    """
    def _assert_file_exists(
        filepath: Path,
        contains: str | None = None,
        not_contains: str | None = None
    ) -> None:
        """Assert file exists and optionally check content."""
        assert filepath.exists(), f"File does not exist: {filepath}"
        assert filepath.is_file(), f"Path is not a file: {filepath}"
        
        if contains or not_contains:
            content = filepath.read_text()
            if contains:
                assert contains in content, f"'{contains}' not found in {filepath}"
            if not_contains:
                assert not_contains not in content, f"'{not_contains}' found in {filepath}"
    
    return _assert_file_exists


@pytest.fixture
def assert_directory_structure():
    """
    Provide an assertion helper for directory structure checks.
    
    Returns:
        Callable that asserts expected directory structure exists.
    """
    def _assert_directory_structure(
        root: Path,
        expected: list[str]
    ) -> None:
        """Assert expected files and directories exist."""
        for path_str in expected:
            path = root / path_str
            assert path.exists(), f"Expected path does not exist: {path}"
    
    return _assert_directory_structure


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: CLEANUP FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def cleanup_environment() -> Generator[None, None, None]:
    """
    Clean up environment variables after each test.
    
    This fixture runs automatically and ensures test isolation
    by restoring the original environment state.
    
    Yields:
        None: Control returns to test.
    """
    # Store original environment
    original_env = os.environ.copy()
    
    yield
    
    # Restore environment
    # Remove any new variables
    for key in set(os.environ.keys()) - set(original_env.keys()):
        del os.environ[key]
    
    # Restore original values
    os.environ.update(original_env)


# ═══════════════════════════════════════════════════════════════════════════════
# END OF CONFTEST
# ═══════════════════════════════════════════════════════════════════════════════
