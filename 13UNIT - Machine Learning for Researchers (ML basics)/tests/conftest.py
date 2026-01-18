"""
Pytest configuration and fixtures for 13UNIT tests.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification


@pytest.fixture
def classification_data():
    """Generate synthetic classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate synthetic multiclass classification dataset."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate synthetic regression dataset."""
    np.random.seed(42)
    X = np.random.randn(200, 5)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(200) * 0.5
    return X, y


@pytest.fixture
def clustering_data():
    """Generate synthetic clustering dataset."""
    X, y = make_blobs(
        n_samples=300,
        centers=4,
        n_features=5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        weights=[0.95, 0.05],
        random_state=42,
    )
    return X, y


@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y
