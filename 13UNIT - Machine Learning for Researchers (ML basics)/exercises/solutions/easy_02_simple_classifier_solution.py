"""Solution for Easy 02 - Simple Classifier."""

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
    """Train a logistic regression classifier."""
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(
    model: LogisticRegression,
    X_test: ArrayLike,
    y_test: ArrayLike,
) -> dict[str, float | ArrayLike]:
    """Evaluate classifier on test data."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "predictions": predictions,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = train_logistic_classifier(X_train, y_train)
    results = evaluate_classifier(model, X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
