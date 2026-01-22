"""Solution for Easy 01 - Train/Test Split Fundamentals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def perform_stratified_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Perform stratified train/test split preserving class proportions."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def verify_stratification(
    y_train: ArrayLike,
    y_test: ArrayLike,
    tolerance: float = 0.02,
) -> dict[str, float | bool]:
    """Verify that stratification preserved class proportions."""
    train_ratio = float(np.mean(y_train))
    test_ratio = float(np.mean(y_test))
    difference = abs(train_ratio - test_ratio)
    return {
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "difference": difference,
        "is_valid": difference <= tolerance,
    }


if __name__ == "__main__":
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = perform_stratified_split(X, y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Verification: {verify_stratification(y_train, y_test)}")
