"""
Exercise: Easy 02 - Simple Classifier

Learning Objective: LO2 (Pipeline Implementation)
Estimated Time: 20 minutes

Task: Train and evaluate a basic classification model.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def train_logistic_classifier(
    X_train: ArrayLike,
    y_train: ArrayLike,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """
    Train a logistic regression classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        C: Inverse regularisation strength.
        max_iter: Maximum iterations for solver.
        
    Returns:
        Fitted LogisticRegression model.
    """
    # TODO: Create and fit LogisticRegression model
    raise NotImplementedError("Complete this function")


def evaluate_classifier(
    model: LogisticRegression,
    X_test: ArrayLike,
    y_test: ArrayLike,
) -> dict[str, float | ArrayLike]:
    """
    Evaluate classifier on test data.
    
    Args:
        model: Fitted classifier.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary with 'accuracy', 'predictions', 'probabilities'.
    """
    # TODO: Generate predictions and calculate metrics
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = train_logistic_classifier(X_train, y_train)
    results = evaluate_classifier(model, X_test, y_test)
    
    print(f"Accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
