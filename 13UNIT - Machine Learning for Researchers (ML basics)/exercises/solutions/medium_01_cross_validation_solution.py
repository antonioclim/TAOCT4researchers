"""Solution for Medium 01 - Cross-Validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def kfold_cross_validation(
    model, X: ArrayLike, y: ArrayLike, n_splits: int = 5, random_state: int = 42
) -> dict[str, list | float]:
    """Perform stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    fold_sizes = [len(test_idx) for _, test_idx in cv.split(X, y)]
    return {
        "scores": list(scores),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "fold_sizes": fold_sizes,
    }


def compare_cv_strategies(model, X: ArrayLike, y: ArrayLike) -> dict[int, float]:
    """Compare 3-fold, 5-fold, and 10-fold CV."""
    results = {}
    for k in [3, 5, 10]:
        cv_result = kfold_cross_validation(model, X, y, n_splits=k)
        results[k] = cv_result["mean"]
    return results


if __name__ == "__main__":
    data = load_breast_cancer()
    model = LogisticRegression(max_iter=1000)
    results = kfold_cross_validation(model, data.data, data.target)
    print(f"5-fold CV: {results['mean']:.4f} ± {results['std']:.4f}")
    print(f"Comparison: {compare_cv_strategies(model, data.data, data.target)}")
