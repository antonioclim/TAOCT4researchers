"""Solution for Hard 01 - Nested Cross-Validation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def standard_cv_with_tuning(X: ArrayLike, y: ArrayLike, param_grid: dict) -> dict[str, float | dict]:
    """Standard CV with tuning (biased estimate)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=cv, scoring="accuracy")
    grid.fit(X, y)
    return {"best_score": float(grid.best_score_), "best_params": grid.best_params_}


def nested_cross_validation(
    X: ArrayLike, y: ArrayLike, param_grid: dict, outer_cv: int = 5, inner_cv: int = 3
) -> dict[str, ArrayLike | float | list]:
    """Nested CV for unbiased estimation."""
    outer_cv_strategy = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    inner_cv_strategy = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
    
    outer_scores, best_params_per_fold = [], []
    for train_idx, test_idx in outer_cv_strategy.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        inner_grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=inner_cv_strategy)
        inner_grid.fit(X_train, y_train)
        
        score = float(accuracy_score(y_test, inner_grid.best_estimator_.predict(X_test)))
        outer_scores.append(score)
        best_params_per_fold.append(inner_grid.best_params_)
    
    return {
        "outer_scores": np.array(outer_scores),
        "mean_score": float(np.mean(outer_scores)),
        "best_params_per_fold": best_params_per_fold,
    }


if __name__ == "__main__":
    data = load_breast_cancer()
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    print(f"Standard: {standard_cv_with_tuning(data.data, data.target, param_grid)['best_score']:.4f}")
    nested = nested_cross_validation(data.data, data.target, param_grid)
    print(f"Nested: {nested['mean_score']:.4f}")
