"""Solution for Medium 02 - Pipeline Construction."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ArrayLike = NDArray[np.floating] | NDArray[np.integer]


def create_preprocessing_pipeline(
    numerical_columns: list[int], categorical_columns: list[int]
) -> ColumnTransformer:
    """Create preprocessing pipeline for mixed feature types."""
    return ColumnTransformer([
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
    ])


def create_full_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    """Combine preprocessing and classification."""
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_pipeline_with_cv(
    pipeline: Pipeline, X: ArrayLike, y: ArrayLike, cv: int = 5
) -> ArrayLike:
    """Evaluate pipeline with cross-validation."""
    return cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.hstack([np.random.randn(n, 3), np.random.randint(0, 3, (n, 2))])
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    preprocessor = create_preprocessing_pipeline([0, 1, 2], [3, 4])
    pipeline = create_full_pipeline(preprocessor, LogisticRegression(max_iter=1000))
    scores = evaluate_pipeline_with_cv(pipeline, X, y)
    print(f"CV scores: {scores}\nMean: {np.mean(scores):.4f}")
