"""
Shared pytest fixtures for 14UNIT tests.
"""

import pytest
import numpy as np
from multiprocessing import cpu_count


@pytest.fixture
def sample_data():
    """Generate sample numerical data for testing."""
    np.random.seed(42)
    return np.random.randn(10000)


@pytest.fixture
def n_workers():
    """Return appropriate worker count for testing."""
    return min(4, cpu_count())


@pytest.fixture
def small_list():
    """Small list for quick tests."""
    return list(range(100))


@pytest.fixture
def medium_list():
    """Medium list for performance tests."""
    return list(range(10000))
