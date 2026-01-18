"""
Lab 13_01 Solution: Supervised Learning Pipeline

Complete solution demonstrating all supervised learning concepts.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score,
    learning_curve, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute complete supervised learning workflow."""
    logger.info("Loading breast cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Pipeline with preprocessing
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    logger.info(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Evaluation
    logger.info("\n=== Test Set Evaluation ===")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
    logger.info(f"F1: {f1_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]):.4f}")
    
    logger.info("\nConfusion Matrix:")
    logger.info(f"\n{confusion_matrix(y_test, y_pred)}")
    
    # Hyperparameter tuning with nested CV
    logger.info("\n=== Nested Cross-Validation ===")
    param_grid = {"classifier__C": [0.01, 0.1, 1, 10]}
    
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    nested_scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_o, X_test_o = X[train_idx], X[test_idx]
        y_train_o, y_test_o = y[train_idx], y[test_idx]
        
        grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="accuracy")
        grid.fit(X_train_o, y_train_o)
        nested_scores.append(grid.score(X_test_o, y_test_o))
    
    logger.info(f"Nested CV: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
    
    # Demonstrate overfitting
    logger.info("\n=== Overfitting Demonstration ===")
    for depth in [1, 3, 5, 10, None]:
        tree = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(max_depth=depth, random_state=42))
        ])
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        test_acc = tree.score(X_test, y_test)
        logger.info(f"Depth {depth}: Train={train_acc:.4f}, Test={test_acc:.4f}, Gap={train_acc-test_acc:.4f}")
    
    # Data leakage demonstration
    logger.info("\n=== Data Leakage Demonstration ===")
    
    # WRONG: Scale before split
    scaler = StandardScaler()
    X_leaky = scaler.fit_transform(X)
    leaky_scores = cross_val_score(
        LogisticRegression(max_iter=1000), X_leaky, y, cv=cv
    )
    
    # CORRECT: Scale inside pipeline
    proper_scores = cross_val_score(pipeline, X, y, cv=cv)
    
    logger.info(f"Leaky approach: {np.mean(leaky_scores):.4f}")
    logger.info(f"Proper approach: {np.mean(proper_scores):.4f}")
    logger.info(f"Difference: {np.mean(leaky_scores) - np.mean(proper_scores):.4f}")
    
    logger.info("\nLab 13_01 complete!")


if __name__ == "__main__":
    main()
