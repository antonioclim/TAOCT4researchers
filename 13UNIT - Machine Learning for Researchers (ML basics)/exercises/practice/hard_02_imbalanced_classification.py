"""
Exercise: Hard 02 - Imbalanced Classification

Learning Objective: LO5 (Pitfall Mitigation)
Estimated Time: 35 minutes

Task: Handle class imbalance through multiple strategies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def create_imbalanced_dataset(
    n_samples: int = 1000,
    imbalance_ratio: float = 0.05,
    random_state: int = 42,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Create dataset with specified class imbalance.
    
    Returns:
        Tuple of (X, y) with imbalanced classes.
    """
    # TODO: Create imbalanced dataset using make_classification
    raise NotImplementedError("Complete this function")


def baseline_classifier(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
) -> dict[str, float]:
    """
    Train baseline without imbalance handling.
    
    Returns:
        Metrics for minority class.
    """
    # TODO: Train standard LogisticRegression
    raise NotImplementedError("Complete this function")


def class_weight_classifier(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
) -> dict[str, float]:
    """
    Use class_weight='balanced'.
    """
    # TODO: Train with balanced class weights
    raise NotImplementedError("Complete this function")


def threshold_adjusted_classifier(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
) -> dict[str, float]:
    """
    Adjust decision threshold.
    """
    # TODO: Find optimal threshold for minority class F1
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    X, y = create_imbalanced_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Baseline:", baseline_classifier(X_train, y_train, X_test, y_test))
    print("Class weights:", class_weight_classifier(X_train, y_train, X_test, y_test))
    print("Threshold:", threshold_adjusted_classifier(X_train, y_train, X_test, y_test))


if __name__ == "__main__":
    main()
