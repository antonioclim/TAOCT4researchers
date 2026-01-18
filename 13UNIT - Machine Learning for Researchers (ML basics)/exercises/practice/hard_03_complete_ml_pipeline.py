"""
Exercise: Hard 03 - Complete ML Pipeline

Learning Objectives: All LOs (Synthesis)
Estimated Time: 45 minutes

Task: Build production-quality ML pipeline following established conventions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


class MLPipeline:
    """Complete ML pipeline following established conventions."""

    def __init__(self, random_state: int = 42) -> None:
        """Initialise pipeline."""
        self.random_state = random_state
        self.X: ArrayLike | None = None
        self.y: ArrayLike | None = None
        self.model: Any = None

    def load_data(self, dataset_name: str) -> None:
        """Load specified sklearn dataset."""
        # TODO: Load data and store in self.X, self.y
        raise NotImplementedError("Complete this method")

    def explore_data(self) -> dict[str, Any]:
        """Return summary statistics."""
        # TODO: Return n_samples, n_features, class_distribution
        raise NotImplementedError("Complete this method")

    def train(
        self,
        model: Any,
        param_grid: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """Train with optional hyperparameter search."""
        # TODO: Train model, use nested CV if param_grid provided
        raise NotImplementedError("Complete this method")

    def evaluate(self) -> dict[str, float]:
        """Return comprehensive evaluation metrics."""
        # TODO: Return accuracy, precision, recall, f1
        raise NotImplementedError("Complete this method")

    def diagnose(self) -> dict[str, Any]:
        """Return overfitting diagnostics."""
        # TODO: Compare train vs test scores
        raise NotImplementedError("Complete this method")


def run_complete_experiment() -> dict[str, Any]:
    """
    Run complete ML experiment with multiple models.
    
    Returns:
        Comprehensive results report.
    """
    # TODO: Use MLPipeline to compare models
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    results = run_complete_experiment()
    print("Experiment Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
