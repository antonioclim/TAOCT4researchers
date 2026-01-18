#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Parallel Machine Learning (Hard)
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ★★★★★ (Hard)
ESTIMATED TIME: 40 minutes
PREREQUISITES: Cross-validation, multiprocessing, scikit-learn basics

LEARNING OBJECTIVES
───────────────────
- LO2: Parallelise ML workflows effectively
- LO4: Use concurrent.futures for model training

PROBLEM DESCRIPTION
───────────────────
Implement parallel cross-validation and hyperparameter grid search for
machine learning models. These are classic embarrassingly parallel problems.

TASKS
─────
1. Implement `train_fold` - train and evaluate single CV fold
2. Implement `parallel_cross_validation` - K-fold CV with multiprocessing
3. Implement `parallel_grid_search` - grid search across hyperparameters
4. Implement `compare_parallelisation` - benchmark sequential vs parallel

HINTS
─────
- Hint 1: Each CV fold is independent - perfect for parallel execution
- Hint 2: Grid search evaluates independent hyperparameter combinations
- Hint 3: Use ProcessPoolExecutor for CPU-bound model training

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class CVResult:
    """Result from single cross-validation fold."""
    fold: int
    train_score: float
    val_score: float
    train_time: float


@dataclass
class GridSearchResult:
    """Result from grid search."""
    best_params: dict[str, Any]
    best_score: float
    all_results: list[tuple[dict[str, Any], float]]
    total_time: float


def create_sample_data(
    n_samples: int = 1000,
    n_features: int = 20,
    random_state: int = 42
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Create sample classification data.
    
    Args:
        n_samples: Number of samples.
        n_features: Number of features.
        random_state: Random seed.
        
    Returns:
        Tuple of (features, labels).
    """
    rng = np.random.default_rng(random_state)
    X = rng.standard_normal((n_samples, n_features))
    # Simple linear decision boundary with noise
    weights = rng.standard_normal(n_features)
    y = (X @ weights > 0).astype(np.int64)
    return X, y


def simple_classifier_train(
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    alpha: float = 1.0
) -> NDArray[np.float64]:
    """
    Train a simple ridge regression classifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        alpha: Regularisation parameter.
        
    Returns:
        Learned weights.
    """
    n_features = X_train.shape[1]
    # Ridge regression: w = (X'X + αI)^(-1) X'y
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    weights = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
    return weights


def simple_classifier_predict(
    X: NDArray[np.float64],
    weights: NDArray[np.float64]
) -> NDArray[np.int64]:
    """Predict using learned weights."""
    return (X @ weights > 0.5).astype(np.int64)


def accuracy_score(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
    """Calculate classification accuracy."""
    return float(np.mean(y_true == y_pred))


def train_fold(args: tuple[Any, ...]) -> CVResult:
    """
    Train and evaluate a single cross-validation fold.
    
    Args:
        args: Tuple of (fold_idx, X_train, y_train, X_val, y_val, alpha).
        
    Returns:
        CVResult with scores and timing.
        
    Example:
        >>> # Prepare data splits
        >>> result = train_fold((0, X_train, y_train, X_val, y_val, 1.0))
        >>> 0 <= result.val_score <= 1
        True
    """
    # TODO: Implement this function
    # 1. Unpack arguments
    # 2. Train model
    # 3. Evaluate on train and validation sets
    # 4. Return CVResult
    raise NotImplementedError("Implement train_fold")


def create_cv_splits(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_folds: int = 5
) -> Iterator[tuple[int, NDArray, NDArray, NDArray, NDArray]]:
    """
    Generate cross-validation splits.
    
    Yields:
        Tuples of (fold_idx, X_train, y_train, X_val, y_val).
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    fold_size = n_samples // n_folds
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_samples
        
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        yield (
            fold,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx]
        )


def parallel_cross_validation(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    alpha: float = 1.0,
    n_folds: int = 5,
    n_workers: int = 4
) -> list[CVResult]:
    """
    Perform K-fold cross-validation in parallel.
    
    Each fold is trained and evaluated independently, making this
    embarrassingly parallel.
    
    Args:
        X: Feature matrix.
        y: Label vector.
        alpha: Regularisation parameter.
        n_folds: Number of CV folds.
        n_workers: Number of parallel workers.
        
    Returns:
        List of CVResult for each fold.
        
    Example:
        >>> X, y = create_sample_data(500)
        >>> results = parallel_cross_validation(X, y, n_folds=5, n_workers=2)
        >>> len(results) == 5
        True
    """
    # TODO: Implement this function
    # 1. Create CV splits
    # 2. Add alpha to each split's arguments
    # 3. Use Pool.map to train folds in parallel
    raise NotImplementedError("Implement parallel_cross_validation")


def sequential_cross_validation(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    alpha: float = 1.0,
    n_folds: int = 5
) -> list[CVResult]:
    """Sequential cross-validation (baseline)."""
    results = []
    for split in create_cv_splits(X, y, n_folds):
        args = (*split, alpha)
        results.append(train_fold(args))
    return results


def parallel_grid_search(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    param_grid: dict[str, Sequence[Any]],
    n_folds: int = 3,
    n_workers: int = 4
) -> GridSearchResult:
    """
    Perform parallel grid search over hyperparameters.
    
    Args:
        X: Feature matrix.
        y: Label vector.
        param_grid: Dictionary of parameter names to values to try.
        n_folds: Number of CV folds per configuration.
        n_workers: Number of parallel workers.
        
    Returns:
        GridSearchResult with best parameters and all results.
        
    Example:
        >>> X, y = create_sample_data(500)
        >>> param_grid = {'alpha': [0.1, 1.0, 10.0]}
        >>> result = parallel_grid_search(X, y, param_grid, n_folds=3)
        >>> result.best_params['alpha'] in [0.1, 1.0, 10.0]
        True
    """
    # TODO: Implement this function
    # 1. Generate all parameter combinations
    # 2. For each combination, run parallel_cross_validation
    # 3. Track best configuration
    raise NotImplementedError("Implement parallel_grid_search")


def compare_parallelisation(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_folds: int = 10,
    worker_counts: Sequence[int] = (1, 2, 4)
) -> dict[str, Any]:
    """
    Compare sequential vs parallel cross-validation performance.
    
    Args:
        X: Feature matrix.
        y: Label vector.
        n_folds: Number of CV folds.
        worker_counts: Worker counts to benchmark.
        
    Returns:
        Dictionary with timing results and speedups.
        
    Example:
        >>> X, y = create_sample_data(1000)
        >>> results = compare_parallelisation(X, y, n_folds=5, worker_counts=[1, 2])
        >>> results['speedup'][2] > 1.0  # Parallel should be faster
        True
    """
    # TODO: Implement this function
    # 1. Time sequential version
    # 2. Time each parallel configuration
    # 3. Calculate speedups
    raise NotImplementedError("Implement compare_parallelisation")


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_train_fold() -> None:
    """Test single fold training."""
    X, y = create_sample_data(200)
    splits = list(create_cv_splits(X, y, n_folds=2))
    args = (*splits[0], 1.0)
    
    result = train_fold(args)
    assert result.fold == 0
    assert 0 <= result.train_score <= 1
    assert 0 <= result.val_score <= 1
    print("✓ train_fold tests passed")


def test_parallel_cross_validation() -> None:
    """Test parallel CV."""
    X, y = create_sample_data(500)
    results = parallel_cross_validation(X, y, n_folds=5, n_workers=2)
    
    assert len(results) == 5
    scores = [r.val_score for r in results]
    assert all(0 <= s <= 1 for s in scores)
    print(f"✓ parallel_cross_validation: mean score = {np.mean(scores):.3f}")


def test_parallel_grid_search() -> None:
    """Test parallel grid search."""
    X, y = create_sample_data(300)
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    
    result = parallel_grid_search(X, y, param_grid, n_folds=3, n_workers=2)
    
    assert result.best_params['alpha'] in param_grid['alpha']
    assert len(result.all_results) == 4
    print(f"✓ parallel_grid_search: best alpha = {result.best_params['alpha']}")


def test_compare_parallelisation() -> None:
    """Test parallelisation comparison."""
    X, y = create_sample_data(500)
    results = compare_parallelisation(X, y, n_folds=4, worker_counts=[1, 2])
    
    assert 'sequential_time' in results
    assert 'speedup' in results
    print(f"✓ compare_parallelisation: 2-worker speedup = {results['speedup'].get(2, 0):.2f}x")


def main() -> None:
    """Run all tests."""
    print("Running hard_03_parallel_ml tests...")
    print("-" * 50)
    
    try:
        test_train_fold()
        test_parallel_cross_validation()
        test_parallel_grid_search()
        test_compare_parallelisation()
        print("-" * 50)
        print("All tests passed! ✓")
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}")
    except AssertionError as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    main()
