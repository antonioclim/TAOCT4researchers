"""Solution for Easy 03 - Metrics Basics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayLike = NDArray[np.integer]


def compute_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, int]:
    """Compute confusion matrix components manually."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def compute_classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute classification metrics from predictions."""
    cm = compute_confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}


def interpret_metrics(metrics: dict[str, float]) -> str:
    """Generate interpretation of metrics."""
    if metrics["precision"] > metrics["recall"]:
        focus = "precision over recall (fewer false positives)"
    elif metrics["recall"] > metrics["precision"]:
        focus = "recall over precision (fewer false negatives)"
    else:
        focus = "balanced precision and recall"
    
    quality = "good" if metrics["f1_score"] > 0.7 else "moderate" if metrics["f1_score"] > 0.5 else "poor"
    return f"Model shows {quality} performance ({metrics['f1_score']:.2f} F1) with {focus}."


if __name__ == "__main__":
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0, 1, 1])
    print(f"CM: {compute_confusion_matrix(y_true, y_pred)}")
    metrics = compute_classification_metrics(y_true, y_pred)
    print(f"Metrics: {metrics}")
    print(f"Interpretation: {interpret_metrics(metrics)}")
