#!/usr/bin/env python3
"""
14UNIT Exercise Solution: Parallel Machine Learning (Hard)
© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class CVResult:
    fold: int
    train_score: float
    val_score: float
    train_time: float


@dataclass
class GridSearchResult:
    best_params: dict[str, Any]
    best_score: float
    all_results: list[tuple[dict[str, Any], float]]
    total_time: float


def create_sample_data(
    n_samples: int = 1000,
    n_features: int = 20,
    random_state: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    weights = rng.standard_normal(n_features)
    y = (X @ weights > 0).astype(np.int64)
    return X, y


def simple_classifier_train(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    alpha: float = 1.0
) -> NDArray[np.float64]:
    n_features = X_train.shape[1]
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    weights = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
    return weights


def simple_classifier_predict(
    X: NDArray[np.float64],
    weights: NDArray[np.float64]
) -> NDArray[np.int64]:
    return (X @ weights > 0.5).astype(np.int64)


def accuracy_score(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    return float(np.mean(y_true == y_pred))


def train_fold(args: tuple[Any, ...]) -> CVResult:
    """Train and evaluate a single CV fold."""
    fold, X_train, y_train, X_val, y_val, alpha = args
    
    start = time.perf_counter()
    weights = simple_classifier_train(X_train, y_train, alpha)
    train_time = time.perf_counter() - start
    
    train_pred = simple_classifier_predict(X_train, weights)
    val_pred = simple_classifier_predict(X_val, weights)
    
    train_score = accuracy_score(y_train, train_pred)
    val_score = accuracy_score(y_val, val_pred)
    
    return CVResult(fold, train_score, val_score, train_time)


def create_cv_splits(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_folds: int = 5
) -> Iterator[tuple[int, NDArray, NDArray, NDArray, NDArray]]:
    n_samples = len(y)
    indices = np.arange(n_samples)
    fold_size = n_samples // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        yield fold, X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def parallel_cross_validation(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    alpha: float = 1.0,
    n_folds: int = 5,
    n_workers: int = 4
) -> list[CVResult]:
    """Perform K-fold cross-validation in parallel."""
    fold_args = [(*split, alpha) for split in create_cv_splits(X, y, n_folds)]
    
    with Pool(processes=n_workers) as pool:
        results = pool.map(train_fold, fold_args)
    
    return results


def sequential_cross_validation(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    alpha: float = 1.0,
    n_folds: int = 5
) -> list[CVResult]:
    return [train_fold((*split, alpha)) for split in create_cv_splits(X, y, n_folds)]


def parallel_grid_search(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    param_grid: dict[str, Sequence[Any]],
    n_folds: int = 3,
    n_workers: int = 4
) -> GridSearchResult:
    """Perform parallel grid search."""
    start = time.perf_counter()
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    
    all_results = []
    best_score = -1.0
    best_params = combinations[0]
    
    for params in combinations:
        cv_results = parallel_cross_validation(
            X, y, alpha=params.get('alpha', 1.0),
            n_folds=n_folds, n_workers=n_workers
        )
        mean_score = np.mean([r.val_score for r in cv_results])
        all_results.append((params, mean_score))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    total_time = time.perf_counter() - start
    return GridSearchResult(best_params, best_score, all_results, total_time)


def compare_parallelisation(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_folds: int = 10,
    worker_counts: Sequence[int] = (1, 2, 4)
) -> dict[str, Any]:
    """Compare sequential vs parallel CV performance."""
    # Sequential
    start = time.perf_counter()
    sequential_cross_validation(X, y, n_folds=n_folds)
    seq_time = time.perf_counter() - start
    
    # Parallel
    times = {}
    speedups = {}
    for n_workers in worker_counts:
        start = time.perf_counter()
        parallel_cross_validation(X, y, n_folds=n_folds, n_workers=n_workers)
        elapsed = time.perf_counter() - start
        times[n_workers] = elapsed
        speedups[n_workers] = seq_time / elapsed
    
    return {
        'sequential_time': seq_time,
        'parallel_times': times,
        'speedup': speedups
    }


if __name__ == '__main__':
    X, y = create_sample_data(1000)
    results = parallel_cross_validation(X, y, n_folds=5, n_workers=2)
    print(f"Mean accuracy: {np.mean([r.val_score for r in results]):.3f}")
