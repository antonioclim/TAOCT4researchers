"""Solution for Hard 02 - Imbalanced Classification."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def create_imbalanced_dataset(
    n_samples: int = 1000, imbalance_ratio: float = 0.05, random_state: int = 42
) -> tuple[ArrayLike, ArrayLike]:
    """Create dataset with specified class imbalance."""
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_classes=2,
        weights=[1 - imbalance_ratio, imbalance_ratio], random_state=random_state
    )
    return X, y


def baseline_classifier(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike
) -> dict[str, float]:
    """Train baseline without imbalance handling."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    }


def class_weight_classifier(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike
) -> dict[str, float]:
    """Use class_weight='balanced'."""
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    }


def threshold_adjusted_classifier(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, y_test: ArrayLike
) -> dict[str, float]:
    """Adjust decision threshold."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    best_f1, best_threshold = 0.0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, thresh
    
    y_pred = (y_prob >= best_threshold).astype(int)
    return {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold": float(best_threshold),
    }


if __name__ == "__main__":
    X, y = create_imbalanced_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Baseline:", baseline_classifier(X_train, y_train, X_test, y_test))
    print("Weighted:", class_weight_classifier(X_train, y_train, X_test, y_test))
    print("Threshold:", threshold_adjusted_classifier(X_train, y_train, X_test, y_test))
