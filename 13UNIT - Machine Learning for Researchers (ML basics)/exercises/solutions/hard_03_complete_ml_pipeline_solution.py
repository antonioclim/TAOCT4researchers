"""Solution for Hard 03 - Complete ML Pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


class MLPipeline:
    """Complete ML pipeline following established conventions."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.X: ArrayLike | None = None
        self.y: ArrayLike | None = None
        self.X_train: ArrayLike | None = None
        self.X_test: ArrayLike | None = None
        self.y_train: ArrayLike | None = None
        self.y_test: ArrayLike | None = None
        self.model: Any = None
        self.scaler = StandardScaler()

    def load_data(self, dataset_name: str) -> None:
        if dataset_name == "breast_cancer":
            data = load_breast_cancer()
            self.X, self.y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def explore_data(self) -> dict[str, Any]:
        return {
            "n_samples": self.X.shape[0],
            "n_features": self.X.shape[1],
            "class_distribution": dict(zip(*np.unique(self.y, return_counts=True))),
        }

    def train(self, model: Any, param_grid: dict[str, list] | None = None) -> dict[str, Any]:
        if param_grid:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            grid = GridSearchCV(model, param_grid, cv=cv, scoring="f1")
            grid.fit(self.X_train, self.y_train)
            self.model = grid.best_estimator_
            return {"best_params": grid.best_params_, "cv_score": float(grid.best_score_)}
        else:
            model.fit(self.X_train, self.y_train)
            self.model = model
            return {"model": type(model).__name__}

    def evaluate(self) -> dict[str, float]:
        y_pred = self.model.predict(self.X_test)
        return {
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred)),
            "recall": float(recall_score(self.y_test, y_pred)),
            "f1": float(f1_score(self.y_test, y_pred)),
        }

    def diagnose(self) -> dict[str, Any]:
        train_pred = self.model.predict(self.X_train)
        test_pred = self.model.predict(self.X_test)
        train_acc = float(accuracy_score(self.y_train, train_pred))
        test_acc = float(accuracy_score(self.y_test, test_pred))
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "overfit_gap": train_acc - test_acc,
            "is_overfitting": (train_acc - test_acc) > 0.1,
        }


def run_complete_experiment() -> dict[str, Any]:
    """Run complete ML experiment with multiple models."""
    pipeline = MLPipeline()
    pipeline.load_data("breast_cancer")
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(random_state=42),
    }
    
    results = {"data_info": pipeline.explore_data(), "model_results": {}}
    
    for name, model in models.items():
        pipeline.train(model)
        results["model_results"][name] = {
            "metrics": pipeline.evaluate(),
            "diagnostics": pipeline.diagnose(),
        }
    
    return results


if __name__ == "__main__":
    results = run_complete_experiment()
    print(f"Data: {results['data_info']}")
    for name, data in results["model_results"].items():
        print(f"\n{name}:")
        print(f"  F1: {data['metrics']['f1']:.4f}")
        print(f"  Overfit gap: {data['diagnostics']['overfit_gap']:.4f}")
