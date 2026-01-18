"""
Exercise: Medium 03 - Clustering Evaluation

Learning Objective: LO6 (Unsupervised Implementation)
Estimated Time: 25 minutes

Task: Implement clustering with systematic evaluation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

ArrayLike = NDArray[np.floating]


def find_optimal_k(
    X: ArrayLike,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> dict[str, list | int]:
    """
    Find optimal k using silhouette score.
    
    Returns:
        Dictionary with 'k_values', 'inertias', 'silhouettes', 'optimal_k'.
    """
    # TODO: Try different k values and track metrics
    raise NotImplementedError("Complete this function")


def compare_clustering_algorithms(
    X: ArrayLike,
    n_clusters: int,
) -> dict[str, float]:
    """
    Compare k-means and hierarchical clustering.
    
    Returns:
        Dictionary mapping algorithm name to silhouette score.
    """
    # TODO: Compare different algorithms
    raise NotImplementedError("Complete this function")


def main() -> None:
    """Run demonstration."""
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    
    results = find_optimal_k(X)
    print(f"Optimal k: {results['optimal_k']}")
    
    comparison = compare_clustering_algorithms(X, n_clusters=4)
    print(f"Algorithm comparison: {comparison}")


if __name__ == "__main__":
    main()
