#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Medium Exercise 1 — Fixtures and Mocking
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for pytest fixtures and unittest.mock exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# CODE UNDER TEST
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
        self.data_dir = data_dir

    def load_json(self, filename: str) -> dict[str, Any]:
        """Load JSON data from a file."""
        filepath = self.data_dir / filename
        with open(filepath) as f:
            return json.load(f)

    def load_csv(self, filename: str) -> list[list[str]]:
        """Load CSV data from a file."""
        filepath = self.data_dir / filename
        with open(filepath) as f:
            return [line.strip().split(",") for line in f]


class ExperimentRunner:
    """Runs experiments with configuration."""

    def __init__(self, config: ExperimentConfig, loader: DataLoader) -> None:
        self.config = config
        self.loader = loader
        self.results: list[float] = []

    def run(self, data_file: str) -> dict[str, Any]:
        """Run the experiment on data."""
        data = self.loader.load_json(data_file)
        self.results = [v * self.config.seed for v in data.get("values", [])]
        return {
            "name": self.config.name,
            "results": self.results,
            "count": len(self.results)
        }


class ResultsCache:
    """Cache for experiment results with external API."""

    def __init__(self, api_url: str) -> None:
        self.api_url = api_url
        self._cache: dict[str, Any] = {}

    def fetch_data(self, experiment_id: str) -> dict[str, Any]:
        """Fetch data from external API (to be mocked)."""
        import urllib.request
        url = f"{self.api_url}/experiments/{experiment_id}"
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())

    def get_or_fetch(self, experiment_id: str) -> dict[str, Any]:
        """Get from cache or fetch from API."""
        if experiment_id not in self._cache:
            self._cache[experiment_id] = self.fetch_data(experiment_id)
        return self._cache[experiment_id]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: BASIC FIXTURES — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_config() -> ExperimentConfig:
    """
    Fixture providing a sample experiment configuration.

    Returns:
        An ExperimentConfig with test values.
    """
    return ExperimentConfig(
        name="test_experiment",
        seed=42,
        parameters={"learning_rate": 0.01, "epochs": 100}
    )


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """
    Fixture providing a temporary directory with test data files.

    Creates a temporary directory, populates it with test JSON and CSV files,
    yields the path, and cleans up automatically after the test.

    Yields:
        Path to the temporary directory containing test data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create test JSON file
        json_data = {"values": [1.0, 2.0, 3.0, 4.0, 5.0], "name": "test_data"}
        with open(data_dir / "test_data.json", "w") as f:
            json.dump(json_data, f)

        # Create test CSV file
        csv_content = "a,b,c\n1,2,3\n4,5,6\n"
        with open(data_dir / "test_data.csv", "w") as f:
            f.write(csv_content)

        yield data_dir


@pytest.fixture
def data_loader(temp_data_dir: Path) -> DataLoader:
    """
    Fixture providing a DataLoader configured with the temp directory.

    Args:
        temp_data_dir: The temporary data directory fixture.

    Returns:
        A configured DataLoader instance.
    """
    return DataLoader(temp_data_dir)


class TestBasicFixtures:
    """Tests demonstrating basic fixture usage."""

    def test_config_fixture_provides_values(
        self,
        sample_config: ExperimentConfig
    ) -> None:
        """Test that config fixture provides expected values."""
        assert sample_config.name == "test_experiment"
        assert sample_config.seed == 42
        assert "learning_rate" in sample_config.parameters

    def test_data_loader_loads_json(self, data_loader: DataLoader) -> None:
        """Test that data loader can load JSON files."""
        data = data_loader.load_json("test_data.json")
        assert data["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert data["name"] == "test_data"

    def test_data_loader_loads_csv(self, data_loader: DataLoader) -> None:
        """Test that data loader can load CSV files."""
        data = data_loader.load_csv("test_data.csv")
        assert len(data) == 3  # Header + 2 rows
        assert data[0] == ["a", "b", "c"]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: FIXTURE SCOPES — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def expensive_resource() -> Generator[dict[str, Any], None, None]:
    """
    Module-scoped fixture simulating an expensive resource.

    This fixture is created once per test module and shared across all tests
    in that module. Useful for database connections, loaded models, etc.

    Yields:
        A dictionary representing the expensive resource.
    """
    # Simulate expensive setup
    resource = {
        "connection_id": "conn_12345",
        "setup_count": 1,
        "data": list(range(1000))
    }
    print("\n[SETUP] Creating expensive resource")

    yield resource

    # Cleanup
    print("\n[TEARDOWN] Releasing expensive resource")


@pytest.fixture(scope="function")
def fresh_config() -> ExperimentConfig:
    """
    Function-scoped fixture providing a fresh config for each test.

    Returns:
        A new ExperimentConfig instance.
    """
    return ExperimentConfig(
        name=f"fresh_experiment",
        seed=123,
        parameters={}
    )


class TestFixtureScopes:
    """Tests demonstrating fixture scopes."""

    def test_expensive_resource_first_use(
        self,
        expensive_resource: dict[str, Any]
    ) -> None:
        """First test using the expensive resource."""
        assert expensive_resource["connection_id"] == "conn_12345"
        assert len(expensive_resource["data"]) == 1000

    def test_expensive_resource_second_use(
        self,
        expensive_resource: dict[str, Any]
    ) -> None:
        """Second test using same expensive resource (not recreated)."""
        # Same resource instance is used
        assert expensive_resource["setup_count"] == 1

    def test_fresh_config_is_independent(
        self,
        fresh_config: ExperimentConfig
    ) -> None:
        """Test that function-scoped fixture is fresh each time."""
        fresh_config.parameters["modified"] = True
        assert fresh_config.parameters["modified"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: MOCKING — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_api_response() -> dict[str, Any]:
    """Fixture providing a mock API response."""
    return {
        "experiment_id": "exp-001",
        "status": "completed",
        "results": [0.95, 0.87, 0.92],
        "metadata": {"runtime": 120.5}
    }


class TestMocking:
    """Tests demonstrating mocking techniques."""

    def test_fetch_data_with_mock(self, mock_api_response: dict[str, Any]) -> None:
        """Test ResultsCache.fetch_data with mocked urllib."""
        cache = ResultsCache("https://api.example.com")

        # Create mock response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_api_response).encode()
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = cache.fetch_data("exp-001")

        assert result["experiment_id"] == "exp-001"
        assert result["status"] == "completed"
        assert result["results"] == [0.95, 0.87, 0.92]

    def test_get_or_fetch_caches_result(
        self,
        mock_api_response: dict[str, Any]
    ) -> None:
        """Test that get_or_fetch uses cache on second call."""
        cache = ResultsCache("https://api.example.com")

        with patch.object(cache, "fetch_data", return_value=mock_api_response) as mock_fetch:
            # First call should fetch
            result1 = cache.get_or_fetch("exp-001")
            # Second call should use cache
            result2 = cache.get_or_fetch("exp-001")

            assert mock_fetch.call_count == 1  # Only called once
            assert result1 == result2

    def test_mock_data_loader(self) -> None:
        """Test ExperimentRunner with mocked DataLoader."""
        # Create mock loader
        mock_loader = Mock(spec=DataLoader)
        mock_loader.load_json.return_value = {"values": [1, 2, 3]}

        config = ExperimentConfig(name="test", seed=10, parameters={})
        runner = ExperimentRunner(config, mock_loader)

        result = runner.run("fake_file.json")

        mock_loader.load_json.assert_called_once_with("fake_file.json")
        assert result["results"] == [10, 20, 30]  # values * seed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: PARAMETRISED FIXTURES — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(params=[10, 100, 1000])
def sample_size(request: pytest.FixtureRequest) -> int:
    """
    Parametrised fixture providing different sample sizes.

    Args:
        request: pytest fixture request object.

    Returns:
        The current parameter value (10, 100, or 1000).
    """
    return request.param


@pytest.fixture(params=[
    {"name": "small", "seed": 1},
    {"name": "medium", "seed": 42},
    {"name": "large", "seed": 999}
])
def config_variant(request: pytest.FixtureRequest) -> ExperimentConfig:
    """
    Parametrised fixture providing different config variants.

    Args:
        request: pytest fixture request object.

    Returns:
        An ExperimentConfig with the current parameter values.
    """
    return ExperimentConfig(
        name=request.param["name"],
        seed=request.param["seed"],
        parameters={}
    )


class TestParametrisedFixtures:
    """Tests demonstrating parametrised fixtures."""

    def test_sample_sizes(self, sample_size: int) -> None:
        """Test runs once for each sample_size parameter."""
        data = list(range(sample_size))
        assert len(data) == sample_size
        assert sample_size in [10, 100, 1000]

    def test_config_variants(self, config_variant: ExperimentConfig) -> None:
        """Test runs once for each config variant."""
        assert config_variant.name in ["small", "medium", "large"]
        assert config_variant.seed in [1, 42, 999]

    def test_combined_parameters(
        self,
        sample_size: int,
        config_variant: ExperimentConfig
    ) -> None:
        """Test with combined fixtures (runs 3 × 3 = 9 times)."""
        # Each combination is tested
        data = [config_variant.seed] * sample_size
        assert len(data) == sample_size
        assert all(v == config_variant.seed for v in data)


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL TEST RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_tests() -> None:
    """Run all validation tests manually."""
    print("=" * 70)
    print("SOLUTION VALIDATION: Medium Exercise 1 — Fixtures and Mocking")
    print("=" * 70)

    # Test basic fixtures
    print("\n--- Exercise 1: Basic Fixtures ---")
    config = ExperimentConfig(
        name="test_experiment",
        seed=42,
        parameters={"learning_rate": 0.01}
    )
    assert config.name == "test_experiment"
    print("✓ Config fixture pattern works")

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        json_data = {"values": [1.0, 2.0, 3.0]}
        with open(data_dir / "test.json", "w") as f:
            json.dump(json_data, f)

        loader = DataLoader(data_dir)
        loaded = loader.load_json("test.json")
        assert loaded["values"] == [1.0, 2.0, 3.0]
        print("✓ DataLoader fixture pattern works")

    # Test mocking
    print("\n--- Exercise 3: Mocking ---")
    mock_loader = Mock(spec=DataLoader)
    mock_loader.load_json.return_value = {"values": [1, 2, 3]}

    runner = ExperimentRunner(
        ExperimentConfig("test", 10, {}),
        mock_loader
    )
    result = runner.run("fake.json")
    assert result["results"] == [10, 20, 30]
    print("✓ Mock DataLoader works correctly")

    # Test cache mocking
    cache = ResultsCache("https://api.example.com")
    mock_response = {"experiment_id": "exp-001", "status": "completed"}

    with patch.object(cache, "fetch_data", return_value=mock_response):
        result1 = cache.get_or_fetch("exp-001")
        result2 = cache.get_or_fetch("exp-001")
        assert result1 == result2
        print("✓ Cache mocking works correctly")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
