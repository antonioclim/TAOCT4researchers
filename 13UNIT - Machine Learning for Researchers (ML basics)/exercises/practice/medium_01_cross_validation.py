"""
Exercise: Medium 01 - Cross-Validation

Learning Objective: LO3 (Validation Methodology)
Estimated Time: 25 minutes

Task: Implement and compare cross-validation strategies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def kfold_cross_validation(
    model,
    X: ArrayLike,
    y: ArrayLike,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, list | float]:
    """
    Perform stratified k-fold cross-validation.
    
    Args:
        model: Unfitted estimator.
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of folds.
        random_state: Random seed.
        
    Returns:
        Dictionary with 'scores', 'mean', 'std', 'fold_sizes'.
    """
    # TODO: Implement stratified k-fold CV
    raise NotImplementedError("Complete this function")


def compare_cv_strategies(
    model,
    X: ArrayLike,
    y: ArrayLike,
) -> dict[int, float]:
    """
    Compare 3-fold, 5-fold, and 10-fold CV.
    
    Returns:
        Dictionary mapping fold count to mean score.
    """
    # TODO: Compare different fold strategies
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    data = load_breast_cancer()
    X, y = data.data, data.target
    model = LogisticRegression(max_iter=1000)
    
    results = kfold_cross_validation(model, X, y)
    print(f"5-fold CV: {results['mean']:.4f} ± {results['std']:.4f}")
    
    comparison = compare_cv_strategies(model, X, y)
    print(f"CV comparison: {comparison}")


if __name__ == "__main__":
    main()
