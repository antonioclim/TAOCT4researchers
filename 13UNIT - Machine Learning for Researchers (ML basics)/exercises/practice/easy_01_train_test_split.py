"""
Exercise: Easy 01 - Train/Test Split Fundamentals

Learning Objective: LO3 (Validation Methodology)
Estimated Time: 15 minutes

Task: Implement proper train/test splitting with stratification.

Instructions:
1. Complete the functions below
2. Run the tests to verify your implementation
3. Ensure all type hints are present
"""

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
    """
    Perform stratified train/test split preserving class proportions.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        test_size: Proportion of data for test set (0.0 to 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Example:
        >>> X = np.random.randn(100, 5)
        >>> y = np.array([0]*50 + [1]*50)
        >>> X_train, X_test, y_train, y_test = perform_stratified_split(X, y)
        >>> len(X_train)
        80
    """
    # TODO: Implement this function
    # Hint: Use train_test_split with stratify parameter
    raise NotImplementedError("Complete this function")


def verify_stratification(
    y_train: ArrayLike,
    y_test: ArrayLike,
    tolerance: float = 0.02,
) -> dict[str, float | bool]:
    """
    Verify that stratification preserved class proportions.

    Args:
        y_train: Training labels.
        y_test: Test labels.
        tolerance: Maximum acceptable difference in proportions.

    Returns:
        Dictionary containing:
            - 'train_ratio': Proportion of positive class in training set
            - 'test_ratio': Proportion of positive class in test set
            - 'difference': Absolute difference between ratios
            - 'is_valid': Whether difference is within tolerance

    Example:
        >>> y_train = np.array([0, 0, 1, 1])
        >>> y_test = np.array([0, 1])
        >>> result = verify_stratification(y_train, y_test)
        >>> result['is_valid']
        True
    """
    # TODO: Implement this function
    # Hint: Calculate mean of each array to get positive class ratio
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    # Load data
    data = load_breast_cancer()
    X, y = data.data, data.target

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)}")

    # Perform split
    X_train, X_test, y_train, y_test = perform_stratified_split(X, y)

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Verify stratification
    verification = verify_stratification(y_train, y_test)
    print(f"\nStratification verification:")
    print(f"  Train ratio: {verification['train_ratio']:.4f}")
    print(f"  Test ratio: {verification['test_ratio']:.4f}")
    print(f"  Valid: {verification['is_valid']}")


if __name__ == "__main__":
    main()
