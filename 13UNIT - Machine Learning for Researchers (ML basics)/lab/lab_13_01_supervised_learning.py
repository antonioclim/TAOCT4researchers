from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RunConfig:
    outdir: Path
    random_state: int = 13
    test_size: float = 0.2


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def load_classification() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.target
    return X, y


def load_regression() -> Tuple[pd.DataFrame, pd.Series]:
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.target
    return X, y


def build_classification_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_features = feature_names
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=5000, solver="lbfgs")
    return Pipeline([("pre", pre), ("model", model)])


def build_regression_pipeline(feature_names: list[str]) -> Pipeline:
    numeric_features = feature_names
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )
    model = Ridge(alpha=1.0)
    return Pipeline([("pre", pre), ("model", model)])


def train_and_evaluate_classification(cfg: RunConfig) -> Dict[str, Any]:
    X, y = load_classification()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipe = build_classification_pipeline(list(X.columns))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    out = {
        "task": "classification",
        "accuracy": float(acc),
        "classification_report": report,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": cfg.random_state,
    }

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.figure_.suptitle("13UNIT: Confusion matrix (breast cancer)")
    disp.figure_.tight_layout()
    disp.figure_.savefig(cfg.outdir / "confusion_matrix.png", dpi=140)
    plt.close(disp.figure_)

    return out


def train_and_evaluate_regression(cfg: RunConfig) -> Dict[str, Any]:
    X, y = load_regression()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    pipe = build_regression_pipeline(list(X.columns))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    out = {
        "task": "regression",
        "mse": float(mse),
        "r2": float(r2),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "random_state": cfg.random_state,
    }
    return out


def tune_classification(cfg: RunConfig) -> Dict[str, Any]:
    X, y = load_classification()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    pipe = build_classification_pipeline(list(X.columns))
    param_grid = {
        "model__C": [0.05, 0.1, 0.5, 1.0, 2.0],
    }
    search = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
    search.fit(X_train, y_train)

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    out = {
        "task": "classification_tuning",
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "test_accuracy": float(acc),
    }
    return out


def plot_learning_curve(cfg: RunConfig) -> None:
    X, y = load_classification()
    pipe = build_classification_pipeline(list(X.columns))
    train_sizes, train_scores, valid_scores = learning_curve(
        pipe, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 8)
    )

    train_mean = train_scores.mean(axis=1)
    valid_mean = valid_scores.mean(axis=1)

    fig = plt.figure()
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, valid_mean, label="Validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("13UNIT: Learning curve (breast cancer, logistic regression)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(cfg.outdir / "learning_curve.png", dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="13UNIT supervised learning lab")
    p.add_argument("--outdir", default="output", help="Output directory")
    p.add_argument("--demo", action="store_true", help="Run a short demo")
    p.add_argument("--train", action="store_true", help="Train baseline models and write metrics")
    p.add_argument("--evaluate", action="store_true", help="Alias for --train")
    p.add_argument("--tune", action="store_true", help="Run a small hyperparameter search")
    p.add_argument("--plot-learning", action="store_true", help="Generate a learning curve plot")
    p.add_argument("--plot-confusion", action="store_true", help="Generate a confusion matrix plot")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(outdir=Path(args.outdir))
    ensure_outdir(cfg.outdir)

    results: Dict[str, Any] = {"unit": "13", "outputs": []}

    if args.demo or args.train or args.evaluate or args.plot_confusion:
        results["outputs"].append(train_and_evaluate_classification(cfg))
    if args.demo or args.train or args.evaluate:
        results["outputs"].append(train_and_evaluate_regression(cfg))
    if args.tune:
        results["outputs"].append(tune_classification(cfg))
    if args.plot_learning:
        plot_learning_curve(cfg)

    (cfg.outdir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
