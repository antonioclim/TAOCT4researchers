"""
Exercise: Hard 01 - Nested Cross-Validation

Learning Objectives: LO3, LO5 (Validation, Pitfall Mitigation)
Estimated Time: 35 minutes

Task: Implement nested CV for unbiased hyperparameter selection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def standard_cv_with_tuning(
    X: ArrayLike,
    y: ArrayLike,
    param_grid: dict,
) -> dict[str, float | dict]:
    """
    Standard CV with tuning (biased estimate).
    
    Returns:
        Dictionary with 'best_score', 'best_params'.
    """
    # TODO: Perform GridSearchCV
    raise NotImplementedError("Complete this function")


def nested_cross_validation(
    X: ArrayLike,
    y: ArrayLike,
    param_grid: dict,
    outer_cv: int = 5,
    inner_cv: int = 3,
) -> dict[str, ArrayLike | float | list]:
    """
    Nested CV for unbiased estimation.
    
    Returns:
        Dictionary with 'outer_scores', 'mean_score', 'best_params_per_fold'.
    """
    # TODO: Implement nested CV with separate inner/outer loops
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    param_grid = {"C": [0.01, 0.1, 1, 10]}
    
    standard = standard_cv_with_tuning(X, y, param_grid)
    print(f"Standard CV (biased): {standard['best_score']:.4f}")
    
    nested = nested_cross_validation(X, y, param_grid)
    print(f"Nested CV (unbiased): {nested['mean_score']:.4f}")


if __name__ == "__main__":
    main()
