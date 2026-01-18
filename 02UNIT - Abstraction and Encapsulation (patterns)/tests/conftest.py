#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT: Test Configuration (conftest.py)
═══════════════════════════════════════════════════════════════════════════════

Shared pytest fixtures and configuration for 02UNIT tests.

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sir_initial_state() -> dict[str, float]:
    """Provide initial state for SIR model tests."""
    return {
        "susceptible": 0.99,
        "infected": 0.01,
        "recovered": 0.0,
    }


@pytest.fixture
def sir_parameters() -> dict[str, float]:
    """Provide standard SIR model parameters."""
    return {
        "beta": 0.3,    # Transmission rate
        "gamma": 0.1,   # Recovery rate
    }


@pytest.fixture
def nbody_initial_positions() -> np.ndarray:
    """Provide initial positions for N-body tests."""
    return np.array([
        [0.0, 0.0],     # Body 1 at origin
        [1.0, 0.0],     # Body 2 at (1, 0)
        [0.0, 1.0],     # Body 3 at (0, 1)
    ])


@pytest.fixture
def nbody_initial_velocities() -> np.ndarray:
    """Provide initial velocities for N-body tests."""
    return np.array([
        [0.0, 0.0],
        [0.0, 0.5],
        [-0.5, 0.0],
    ])


@pytest.fixture
def nbody_masses() -> np.ndarray:
    """Provide masses for N-body tests."""
    return np.array([1.0, 0.1, 0.1])


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_data() -> list[int]:
    """Provide sample data for pattern tests."""
    return [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]


@pytest.fixture
def sorted_data() -> list[int]:
    """Provide sorted version of sample data."""
    return [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]


@pytest.fixture
def empty_list() -> list[Any]:
    """Provide empty list for edge case tests."""
    return []


@pytest.fixture
def single_element() -> list[int]:
    """Provide single element list."""
    return [42]


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVER PATTERN FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockObserver:
    """Mock observer for testing observer pattern."""
    
    notifications: list[Any] = None
    
    def __post_init__(self) -> None:
        if self.notifications is None:
            self.notifications = []
    
    def update(self, value: Any) -> None:
        """Record notification."""
        self.notifications.append(value)
    
    @property
    def call_count(self) -> int:
        """Get number of notifications received."""
        return len(self.notifications)


@pytest.fixture
def mock_observer() -> MockObserver:
    """Provide a mock observer."""
    return MockObserver()


@pytest.fixture
def multiple_observers() -> list[MockObserver]:
    """Provide multiple mock observers."""
    return [MockObserver() for _ in range(3)]


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY PATTERN FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def shape_params() -> dict[str, dict[str, float]]:
    """Provide parameters for shape creation."""
    return {
        "circle": {"radius": 5.0},
        "rectangle": {"width": 3.0, "height": 4.0},
        "triangle": {"a": 3.0, "b": 4.0, "c": 5.0},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL COMPARISON HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def float_tolerance() -> float:
    """Provide standard tolerance for floating point comparisons."""
    return 1e-9


@pytest.fixture
def assert_float_equal() -> Callable[[float, float, float], None]:
    """Provide helper for floating point equality assertion."""
    def _assert(actual: float, expected: float, tol: float = 1e-9) -> None:
        assert abs(actual - expected) < tol, f"{actual} != {expected} (tol={tol})"
    return _assert


@pytest.fixture
def assert_array_equal() -> Callable[[np.ndarray, np.ndarray, float], None]:
    """Provide helper for numpy array equality assertion."""
    def _assert(actual: np.ndarray, expected: np.ndarray, tol: float = 1e-9) -> None:
        np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    return _assert


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def configure_logging() -> None:
    """Configure logging for tests to reduce noise."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORARY DIRECTORY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide temporary directory for test outputs."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRIZE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
