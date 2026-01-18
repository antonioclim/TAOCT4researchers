"""
Exercise: Easy 03 - Metrics Basics

Learning Objective: LO4 (Metric Interpretation)
Estimated Time: 15 minutes

Task: Compute and interpret classification metrics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayLike = NDArray[np.integer]


def compute_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, int]:
    """
    Compute confusion matrix components manually.
    
    Args:
        y_true: True labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        
    Returns:
        Dictionary with 'TP', 'TN', 'FP', 'FN' counts.
    """
    # TODO: Calculate TP, TN, FP, FN
    raise NotImplementedError("Complete this function")


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """
    Compute classification metrics from predictions.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary with 'accuracy', 'precision', 'recall', 'f1_score'.
    """
    # TODO: Use confusion matrix to compute metrics
    raise NotImplementedError("Complete this function")


def interpret_metrics(metrics: dict[str, float]) -> str:
    """
    Generate interpretation of metrics.
    
    Args:
        metrics: Dictionary of computed metrics.
        
    Returns:
        String describing model performance.
    """
    # TODO: Create meaningful interpretation
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix: {cm}")
    
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"Metrics: {metrics}")
    
    interpretation = interpret_metrics(metrics)
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
