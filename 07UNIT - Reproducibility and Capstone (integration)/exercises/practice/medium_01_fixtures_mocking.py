#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Medium 01 - Fixtures and Mocking
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Real-world testing often requires setting up complex test data and isolating
code from external dependencies. This exercise introduces pytest fixtures for
reusable test setup and mocking for isolating units under test.

PREREQUISITES
─────────────
- Completed easy exercises on basic testing
- Understanding of pytest fundamentals
- Familiarity with classes and methods

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Create and use pytest fixtures
2. Use fixture scopes appropriately
3. Mock external dependencies with unittest.mock

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 40 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import tempfile
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# CODE TO TEST (Do not modify)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    seed: int
    parameters: dict[str, Any] = field(default_factory=dict)


class DataLoader:
    """Loads data from files."""

    def __init__(self, data_dir: Path) -> None:
        """Initialise with data directory."""
        self.data_dir = data_dir

    def load_json(self, filename: str) -> dict[str, Any]:
        """Load JSON file from data directory."""
        filepath = self.data_dir / filename
        with filepath.open() as f:
            return json.load(f)

    def list_files(self) -> list[str]:
        """List all JSON files in data directory."""
        return [f.name for f in self.data_dir.glob("*.json")]


class ExperimentRunner:
    """Runs experiments with external API calls."""

    def __init__(self, api_url: str) -> None:
        """Initialise with API URL."""
        self.api_url = api_url

    def fetch_data(self, experiment_id: str) -> dict[str, Any]:
        """Fetch experiment data from API (simulated)."""
        # In real code, this would make an HTTP request
        import urllib.request

        url = f"{self.api_url}/experiments/{experiment_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read())

    def run_experiment(self, config: ExperimentConfig) -> dict[str, Any]:
        """Run an experiment and return results."""
        # Simulated experiment
        return {
            "name": config.name,
            "seed": config.seed,
            "status": "completed",
            "metrics": {"accuracy": 0.95, "loss": 0.05},
        }


class ResultsCache:
    """Caches experiment results."""

    def __init__(self) -> None:
        """Initialise empty cache."""
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        """Get cached result or None."""
        return self._cache.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Cache a result."""
        self._cache[key] = value

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Create Basic Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_config() -> ExperimentConfig:
    """
    TODO: Create a fixture that returns a sample ExperimentConfig.

    The fixture should return an ExperimentConfig with:
    - name: "test_experiment"
    - seed: 42
    - parameters: {"learning_rate": 0.01, "epochs": 100}

    Example:
        def test_something(sample_config):
            assert sample_config.name == "test_experiment"
    """
    # TODO: Implement this fixture
    pass


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """
    TODO: Create a fixture that sets up a temporary data directory.

    The fixture should:
    1. Create a subdirectory called "data" in tmp_path
    2. Create a sample JSON file "experiment_001.json" with content:
       {"id": "001", "results": [1, 2, 3]}
    3. Return the data directory path

    Note: tmp_path is a built-in pytest fixture that provides a temporary
    directory unique to the test invocation.
    """
    # TODO: Implement this fixture
    pass


@pytest.fixture
def data_loader(temp_data_dir: Path) -> DataLoader:
    """
    TODO: Create a fixture that returns a DataLoader using temp_data_dir.

    This demonstrates fixture composition - using one fixture in another.
    """
    # TODO: Implement this fixture
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Fixture Scopes
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def expensive_resource() -> dict[str, Any]:
    """
    TODO: Create a module-scoped fixture for an "expensive" resource.

    Module-scoped fixtures are created once per test module and shared
    across all tests in that module. Use this for expensive setup.

    Return a dictionary simulating loaded configuration:
    {"database": "test_db", "connection_pool_size": 5}
    """
    # TODO: Implement this fixture
    # Add a print statement to see when it's called:
    # print("\n[Creating expensive resource]")
    pass


@pytest.fixture
def results_cache() -> ResultsCache:
    """
    TODO: Create a fixture that returns a fresh ResultsCache.

    Since this has function scope (default), each test gets a new cache.
    This ensures test isolation.
    """
    # TODO: Implement this fixture
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Mocking External Dependencies
# ═══════════════════════════════════════════════════════════════════════════════

def test_fetch_data_with_mock() -> None:
    """
    TODO: Test ExperimentRunner.fetch_data using mocking.

    Mock the urllib.request.urlopen to avoid actual HTTP calls.

    Steps:
    1. Create an ExperimentRunner
    2. Use @patch or patch() context manager to mock urlopen
    3. Configure the mock to return a response with JSON data
    4. Call fetch_data and verify the result

    Hint:
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"id": "123", "status": "ok"}'
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)
    """
    # TODO: Implement this test
    pass


def test_run_experiment_with_mock_cache() -> None:
    """
    TODO: Test that ExperimentRunner checks cache before running.

    This test should:
    1. Create a mock cache
    2. Configure mock to return cached result for a specific key
    3. Verify that when cache has data, it's returned without running

    Note: You may need to modify the ExperimentRunner class or create
    a version that accepts a cache parameter.
    """
    # TODO: Implement this test
    pass


@pytest.fixture
def mock_api_runner() -> ExperimentRunner:
    """
    TODO: Create a fixture that returns an ExperimentRunner with mocked API.

    Use patch to mock the fetch_data method so tests don't need network.
    """
    # TODO: Implement this fixture
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Parametrised Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(params=[10, 100, 1000])
def sample_size(request: pytest.FixtureRequest) -> int:
    """
    TODO: Create a parametrised fixture for different sample sizes.

    This fixture should return different sample sizes, and any test
    using it will run multiple times (once per parameter).

    The request.param contains the current parameter value.
    """
    # TODO: Implement this fixture
    # return request.param
    pass


@pytest.fixture(params=[
    {"name": "small", "seed": 1, "parameters": {}},
    {"name": "medium", "seed": 42, "parameters": {"epochs": 50}},
    {"name": "large", "seed": 123, "parameters": {"epochs": 200, "batch_size": 64}},
])
def config_variant(request: pytest.FixtureRequest) -> ExperimentConfig:
    """
    TODO: Create a parametrised fixture for different configurations.

    Return ExperimentConfig created from each parameter dict.
    """
    # TODO: Implement this fixture
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS USING YOUR FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

class TestWithFixtures:
    """Tests demonstrating fixture usage."""

    def test_sample_config_values(self, sample_config: ExperimentConfig) -> None:
        """Test that sample_config fixture provides correct values."""
        assert sample_config is not None
        assert sample_config.name == "test_experiment"
        assert sample_config.seed == 42

    def test_data_loader_loads_json(self, data_loader: DataLoader) -> None:
        """Test that data_loader fixture works with temp files."""
        assert data_loader is not None
        data = data_loader.load_json("experiment_001.json")
        assert data["id"] == "001"

    def test_cache_isolation(self, results_cache: ResultsCache) -> None:
        """Test that each test gets a fresh cache."""
        assert results_cache.get("any_key") is None
        results_cache.set("test_key", {"value": 42})
        assert results_cache.get("test_key") == {"value": 42}

    def test_with_sample_size(self, sample_size: int) -> None:
        """Test runs multiple times with different sample sizes."""
        assert sample_size in [10, 100, 1000]
        # Actual test logic would go here

    def test_with_config_variant(
        self,
        config_variant: ExperimentConfig,
    ) -> None:
        """Test runs multiple times with different configurations."""
        assert config_variant.name in ["small", "medium", "large"]


# ═══════════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Run tests with: pytest medium_01_fixtures_mocking.py -v")
    print("\nTo see fixture creation, add -s flag:")
    print("  pytest medium_01_fixtures_mocking.py -v -s")
