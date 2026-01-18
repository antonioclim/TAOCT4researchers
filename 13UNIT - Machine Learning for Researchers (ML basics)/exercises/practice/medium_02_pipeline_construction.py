"""
Exercise: Medium 02 - Pipeline Construction

Learning Objective: LO2 (Pipeline Implementation)
Estimated Time: 30 minutes

Task: Build a complete preprocessing and classification pipeline.
"""

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
    numerical_columns: list[int],
    categorical_columns: list[int],
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for mixed feature types.
    
    Args:
        numerical_columns: Indices of numerical columns.
        categorical_columns: Indices of categorical columns.
        
    Returns:
        Configured ColumnTransformer.
    """
    # TODO: Create ColumnTransformer with StandardScaler and OneHotEncoder
    raise NotImplementedError("Complete this function")


def create_full_pipeline(
    preprocessor: ColumnTransformer,
    classifier,
) -> Pipeline:
    """
    Combine preprocessing and classification.
    
    Returns:
        Complete Pipeline.
    """
    # TODO: Create Pipeline combining preprocessor and classifier
    raise NotImplementedError("Complete this function")


def evaluate_pipeline_with_cv(
    pipeline: Pipeline,
    X: ArrayLike,
    y: ArrayLike,
    cv: int = 5,
) -> ArrayLike:
    """
    Evaluate pipeline with cross-validation.
    
    Returns:
        Array of CV scores.
    """
    # TODO: Run cross-validation on the pipeline
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    # Create synthetic mixed data
    np.random.seed(42)
    n = 200
    X_num = np.random.randn(n, 3)
    X_cat = np.random.randint(0, 3, (n, 2))
    X = np.hstack([X_num, X_cat])
    y = (X_num[:, 0] + X_num[:, 1] > 0).astype(int)
    
    preprocessor = create_preprocessing_pipeline([0, 1, 2], [3, 4])
    pipeline = create_full_pipeline(preprocessor, LogisticRegression(max_iter=1000))
    scores = evaluate_pipeline_with_cv(pipeline, X, y)
    
    print(f"CV scores: {scores}")
    print(f"Mean: {np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
