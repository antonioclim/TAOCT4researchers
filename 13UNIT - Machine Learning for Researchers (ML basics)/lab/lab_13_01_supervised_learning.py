"""
Lab 13_01: Supervised Learning Pipeline

Complete supervised learning workflow covering data preparation,
classification, regression, model selection and pitfall demonstrations.

Learning Objectives:
    LO2: Implement classification and regression pipelines
    LO3: Apply proper validation protocols
    LO4: Select and interpret evaluation metrics
    LO5: Identify and mitigate common pitfalls

Estimated Duration: ~150 minutes
Lines: 550+

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, cross_validate,
    learning_curve, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ArrayLike = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]
TargetArray = NDArray[np.integer[Any]] | NDArray[np.floating[Any]]


# =============================================================================
# SECTION 1: Data Preparation
# =============================================================================

def load_and_explore(dataset_name: str = "breast_cancer") -> dict[str, Any]:
    """Load dataset and return exploration summary."""
    logger.info("Loading dataset: %s", dataset_name)
    
    if dataset_name == "breast_cancer":
        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target
        return {
            "X": X, "y": y,
            "feature_names": list(dataset.feature_names),
            "target_names": list(dataset.target_names),
            "n_samples": X.shape[0], "n_features": X.shape[1],
            "class_distribution": {dataset.target_names[i]: int(np.sum(y == i)) for i in range(2)},
        }
    elif dataset_name == "diabetes":
        dataset = load_diabetes()
        X, y = dataset.data, dataset.target
        return {
            "X": X, "y": y,
            "feature_names": list(dataset.feature_names),
            "n_samples": X.shape[0], "n_features": X.shape[1],
            "target_stats": {"mean": float(np.mean(y)), "std": float(np.std(y))},
        }
    elif dataset_name == "synthetic":
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        return {"X": X, "y": y, "n_samples": X.shape[0], "n_features": X.shape[1]}
    raise ValueError(f"Unknown dataset: {dataset_name}")


def preprocess_features(
    X: ArrayLike,
    numerical_columns: list[int] | None = None,
    categorical_columns: list[int] | None = None,
) -> tuple[ColumnTransformer, ArrayLike]:
    """Create and apply preprocessing pipeline."""
    if numerical_columns is None:
        numerical_columns = list(range(X.shape[1]))
    
    transformers: list[tuple[str, Any, list[int]]] = []
    if numerical_columns:
        transformers.append(("num", StandardScaler(), numerical_columns))
    if categorical_columns:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns))
    if not transformers:
        transformers.append(("num", StandardScaler(), list(range(X.shape[1]))))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Preprocessing complete. Shape: %s -> %s", X.shape, X_transformed.shape)
    return preprocessor, X_transformed


def train_test_stratified_split(
    X: ArrayLike, y: TargetArray, test_size: float = 0.2, random_state: int = 42
) -> tuple[ArrayLike, ArrayLike, TargetArray, TargetArray]:
    """Perform stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info("Split complete. Train: %d, Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


# =============================================================================
# SECTION 2: Classification Pipeline
# =============================================================================

def train_classifier(
    X_train: ArrayLike, y_train: TargetArray, classifier_type: str = "logistic", **kwargs: Any
) -> Any:
    """Train a classification model."""
    classifiers = {
        "logistic": LogisticRegression(max_iter=1000, random_state=42, **kwargs),
        "random_forest": RandomForestClassifier(random_state=42, **kwargs),
        "decision_tree": DecisionTreeClassifier(random_state=42, **kwargs),
        "knn": KNeighborsClassifier(**kwargs),
        "svc": SVC(probability=True, random_state=42, **kwargs),
    }
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier_type}")
    model = classifiers[classifier_type]
    model.fit(X_train, y_train)
    logger.info("Trained %s classifier", classifier_type)
    return model


def evaluate_classifier(model: Any, X_test: ArrayLike, y_test: TargetArray) -> dict[str, Any]:
    """Evaluate classification model with multiple metrics."""
    y_pred = model.predict(X_test)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    logger.info("Accuracy: %.3f, F1: %.3f", metrics["accuracy"], metrics["f1"])
    return metrics


def compare_classifiers(
    X_train: ArrayLike, y_train: TargetArray, X_test: ArrayLike, y_test: TargetArray,
    classifiers: list[str] | None = None,
) -> pd.DataFrame:
    """Train and compare multiple classifiers."""
    if classifiers is None:
        classifiers = ["logistic", "random_forest", "decision_tree", "knn", "svc"]
    results = []
    for clf_type in classifiers:
        model = train_classifier(X_train, y_train, clf_type)
        metrics = evaluate_classifier(model, X_test, y_test)
        results.append({
            "classifier": clf_type, "accuracy": metrics["accuracy"],
            "precision": metrics["precision"], "recall": metrics["recall"],
            "f1": metrics["f1"], "roc_auc": metrics.get("roc_auc", np.nan),
        })
    return pd.DataFrame(results).sort_values("f1", ascending=False)


def plot_confusion_matrix(
    y_true: TargetArray, y_pred: TargetArray, class_names: list[str] | None = None
) -> plt.Figure:
    """Visualise confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True label", xlabel="Predicted label")
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_roc_curves(models: dict[str, Any], X_test: ArrayLike, y_test: TargetArray) -> plt.Figure:
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# =============================================================================
# SECTION 3: Regression Pipeline
# =============================================================================

def train_regressor(
    X_train: ArrayLike, y_train: TargetArray, regressor_type: str = "linear", **kwargs: Any
) -> Any:
    """Train a regression model."""
    regressors = {
        "linear": LinearRegression(**kwargs),
        "ridge": Ridge(random_state=42, **kwargs),
        "random_forest": RandomForestRegressor(random_state=42, **kwargs),
    }
    if regressor_type not in regressors:
        raise ValueError(f"Unknown regressor: {regressor_type}")
    model = regressors[regressor_type]
    model.fit(X_train, y_train)
    logger.info("Trained %s regressor", regressor_type)
    return model


def evaluate_regressor(model: Any, X_test: ArrayLike, y_test: TargetArray) -> dict[str, float]:
    """Evaluate regression model."""
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    return {
        "mse": mse, "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def plot_predictions_vs_actual(y_true: TargetArray, y_pred: TargetArray) -> plt.Figure:
    """Scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)
    lims = [min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))]
    ax.plot(lims, lims, "r--", label="Perfect Prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_residuals(y_true: TargetArray, y_pred: TargetArray) -> plt.Figure:
    """Residual plot for regression diagnostics."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residuals")
    axes[1].hist(residuals, bins=30, edgecolor="black")
    axes[1].set_xlabel("Residual Value")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    return fig


# =============================================================================
# SECTION 4: Model Selection
# =============================================================================

def cross_validate_model(
    model: Any, X: ArrayLike, y: TargetArray, cv: int = 5, scoring: str = "accuracy"
) -> dict[str, Any]:
    """Perform cross-validation with detailed results."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=cv_strategy, scoring=scoring, return_train_score=True)
    results = {
        "test_scores": cv_results["test_score"],
        "train_scores": cv_results["train_score"],
        "mean_test_score": float(np.mean(cv_results["test_score"])),
        "std_test_score": float(np.std(cv_results["test_score"])),
        "mean_train_score": float(np.mean(cv_results["train_score"])),
    }
    logger.info("CV: %.3f (±%.3f)", results["mean_test_score"], results["std_test_score"])
    return results


def grid_search_cv(
    model: Any, param_grid: dict[str, list[Any]], X: ArrayLike, y: TargetArray,
    cv: int = 5, scoring: str = "accuracy"
) -> dict[str, Any]:
    """Grid search with cross-validation."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(model, param_grid, cv=cv_strategy, scoring=scoring, return_train_score=True, n_jobs=-1)
    grid.fit(X, y)
    logger.info("Best score: %.3f, params: %s", grid.best_score_, grid.best_params_)
    return {"best_params": grid.best_params_, "best_score": float(grid.best_score_),
            "cv_results": pd.DataFrame(grid.cv_results_), "best_estimator": grid.best_estimator_}


def nested_cross_validation(
    model: Any, param_grid: dict[str, list[Any]], X: ArrayLike, y: TargetArray,
    outer_cv: int = 5, inner_cv: int = 3, scoring: str = "accuracy"
) -> dict[str, Any]:
    """Nested CV for unbiased performance estimation."""
    outer_cv_strategy = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    inner_cv_strategy = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
    outer_scores, best_params_per_fold = [], []
    
    for train_idx, test_idx in outer_cv_strategy.split(X, y):
        X_train_o, X_test_o = X[train_idx], X[test_idx]
        y_train_o, y_test_o = y[train_idx], y[test_idx]
        inner_grid = GridSearchCV(model, param_grid, cv=inner_cv_strategy, scoring=scoring, n_jobs=-1)
        inner_grid.fit(X_train_o, y_train_o)
        score = float(accuracy_score(y_test_o, inner_grid.best_estimator_.predict(X_test_o)))
        outer_scores.append(score)
        best_params_per_fold.append(inner_grid.best_params_)
    
    logger.info("Nested CV: %.3f (±%.3f)", np.mean(outer_scores), np.std(outer_scores))
    return {"outer_scores": np.array(outer_scores), "mean_score": float(np.mean(outer_scores)),
            "std_score": float(np.std(outer_scores)), "best_params_per_fold": best_params_per_fold}


def learning_curve_analysis(
    model: Any, X: ArrayLike, y: TargetArray, cv: int = 5,
    train_sizes: ArrayLike | None = None
) -> tuple[plt.Figure, dict[str, Any]]:
    """Generate learning curves for bias/variance diagnosis."""
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1, random_state=42
    )
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    test_mean, test_std = np.mean(test_scores, axis=1), np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    ax.plot(train_sizes_abs, train_mean, "o-", color="blue", label="Training")
    ax.plot(train_sizes_abs, test_mean, "o-", color="orange", label="Validation")
    ax.set_xlabel("Training Size")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig, {"train_sizes": train_sizes_abs, "train_mean": train_mean, "test_mean": test_mean}


# =============================================================================
# SECTION 5: Pitfall Demonstrations
# =============================================================================

def demonstrate_overfitting(X: ArrayLike, y: TargetArray) -> tuple[plt.Figure, pd.DataFrame]:
    """Demonstrate overfitting with varying tree depth."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    depths = [1, 2, 3, 5, 7, 10, 15, 20, None]
    results = []
    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)
        train_acc = float(accuracy_score(y_train, model.predict(X_train)))
        test_acc = float(accuracy_score(y_test, model.predict(X_test)))
        results.append({"max_depth": str(depth) if depth else "None",
                        "train_accuracy": train_acc, "test_accuracy": test_acc, "gap": train_acc - test_acc})
    
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(depths)), df["train_accuracy"], "o-", label="Train", color="blue")
    ax.plot(range(len(depths)), df["test_accuracy"], "o-", label="Test", color="orange")
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([str(d) if d else "None" for d in depths])
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, df


def demonstrate_data_leakage(X: ArrayLike, y: TargetArray) -> dict[str, float]:
    """Demonstrate data leakage impact."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # WRONG: fit scaler on all data
    scaler = StandardScaler()
    X_leaky = scaler.fit_transform(X)
    leaky_scores = cross_val_score(LogisticRegression(max_iter=1000), X_leaky, y, cv=cv)
    
    # CORRECT: pipeline
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    proper_scores = cross_val_score(pipeline, X, y, cv=cv)
    
    results = {"leaky_score": float(np.mean(leaky_scores)), "proper_score": float(np.mean(proper_scores)),
               "difference": float(np.mean(leaky_scores) - np.mean(proper_scores))}
    logger.info("Leakage impact: %.4f", results["difference"])
    return results


def handle_class_imbalance(X: ArrayLike, y: TargetArray, imbalance_ratio: float = 0.1) -> dict[str, Any]:
    """Demonstrate class imbalance handling strategies."""
    # Create imbalanced data
    minority_idx = np.where(y == 1)[0][:int(np.sum(y == 1) * imbalance_ratio)]
    majority_idx = np.where(y == 0)[0]
    idx = np.concatenate([majority_idx, minority_idx])
    np.random.shuffle(idx)
    X_imb, y_imb = X[idx], y[idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb)
    results: dict[str, Any] = {}
    
    # Baseline
    m1 = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    p1 = m1.predict(X_test)
    results["baseline"] = {"f1": float(f1_score(y_test, p1, zero_division=0)),
                           "recall": float(recall_score(y_test, p1, zero_division=0))}
    
    # Class weights
    m2 = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced").fit(X_train, y_train)
    p2 = m2.predict(X_test)
    results["class_weights"] = {"f1": float(f1_score(y_test, p2, zero_division=0)),
                                 "recall": float(recall_score(y_test, p2, zero_division=0))}
    
    logger.info("Baseline F1: %.3f, Weighted F1: %.3f", results["baseline"]["f1"], results["class_weights"]["f1"])
    return results


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Run demonstration."""
    data = load_and_explore("breast_cancer")
    X, y = data["X"], data["y"]
    X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)
    
    preprocessor, X_train_s = preprocess_features(X_train)
    X_test_s = preprocessor.transform(X_test)
    
    model = train_classifier(X_train_s, y_train, "logistic")
    metrics = evaluate_classifier(model, X_test_s, y_test)
    logger.info("Metrics: %s", {k: v for k, v in metrics.items() if k != "classification_report"})
    
    cv_results = cross_validate_model(LogisticRegression(max_iter=1000), X, y)
    leakage = demonstrate_data_leakage(X, y)
    logger.info("Complete")


if __name__ == "__main__":
    main()


# =============================================================================
# Additional Utility Functions
# =============================================================================

def create_classification_pipeline(
    classifier: Any,
    numerical_columns: list[int] | None = None,
    categorical_columns: list[int] | None = None,
) -> Pipeline:
    """
    Create a complete classification pipeline with preprocessing.
    
    Args:
        classifier: Classifier instance to use.
        numerical_columns: Indices of numerical columns.
        categorical_columns: Indices of categorical columns.
        
    Returns:
        Configured Pipeline object.
    """
    transformers = []
    if numerical_columns:
        transformers.append(("num", StandardScaler(), numerical_columns))
    if categorical_columns:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns))
    
    if not transformers:
        # Default: scale all columns
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ])
    
    preprocessor = ColumnTransformer(transformers, remainder="passthrough")
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def feature_importance_analysis(
    model: Any,
    feature_names: list[str],
    top_n: int = 10,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Analyse and visualise feature importances.
    
    Args:
        model: Fitted model with feature_importances_ or coef_ attribute.
        feature_names: Names of features.
        top_n: Number of top features to display.
        
    Returns:
        Tuple of (Figure, DataFrame with importances).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        raise ValueError("Model has no feature_importances_ or coef_")
    
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_df = df.head(top_n)
    ax.barh(range(len(top_df)), top_df["importance"].values)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()
    
    return fig, df


def stratification_verification(
    y_train: TargetArray,
    y_test: TargetArray,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """
    Verify that stratification preserved class proportions.
    
    Args:
        y_train: Training labels.
        y_test: Test labels.
        tolerance: Maximum acceptable difference in proportions.
        
    Returns:
        Dictionary with verification results.
    """
    train_props = {int(c): float(np.mean(y_train == c)) for c in np.unique(y_train)}
    test_props = {int(c): float(np.mean(y_test == c)) for c in np.unique(y_test)}
    
    max_diff = max(abs(train_props[c] - test_props[c]) for c in train_props)
    
    return {
        "train_proportions": train_props,
        "test_proportions": test_props,
        "max_difference": float(max_diff),
        "stratification_valid": max_diff <= tolerance,
    }


def classification_threshold_analysis(
    model: Any,
    X_test: ArrayLike,
    y_test: TargetArray,
    thresholds: ArrayLike | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Analyse classification performance across different thresholds.
    
    Args:
        model: Fitted classifier with predict_proba method.
        X_test: Test features.
        y_test: Test labels.
        thresholds: Array of thresholds to evaluate.
        
    Returns:
        Tuple of (Figure, DataFrame with threshold metrics).
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    results = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        results.append({
            "threshold": float(thresh),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
        })
    
    df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["threshold"], df["precision"], "o-", label="Precision")
    ax.plot(df["threshold"], df["recall"], "o-", label="Recall")
    ax.plot(df["threshold"], df["f1"], "o-", label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Metrics vs Classification Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig, df


def model_comparison_report(
    models: dict[str, Any],
    X_train: ArrayLike,
    y_train: TargetArray,
    X_test: ArrayLike,
    y_test: TargetArray,
) -> pd.DataFrame:
    """
    Generate comprehensive comparison report for multiple models.
    
    Args:
        models: Dictionary mapping model names to unfitted estimators.
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        DataFrame with comparison metrics.
    """
    results = []
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)
        
        entry = {
            "model": name,
            "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted")),
            "recall": float(recall_score(y_test, y_pred, average="weighted")),
            "f1": float(f1_score(y_test, y_pred, average="weighted")),
        }
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                entry["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
        
        entry["overfit_gap"] = entry["train_accuracy"] - entry["test_accuracy"]
        results.append(entry)
    
    return pd.DataFrame(results).sort_values("f1", ascending=False)
