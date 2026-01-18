"""Solution for Medium 03 - Clustering Evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

ArrayLike = NDArray[np.floating]


def find_optimal_k(
    X: ArrayLike, k_range: range = range(2, 11), random_state: int = 42
) -> dict[str, list | int]:
    """Find optimal k using silhouette score."""
    k_values, inertias, silhouettes = [], [], []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        k_values.append(k)
        inertias.append(float(kmeans.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))
    optimal_k = k_values[np.argmax(silhouettes)]
    return {"k_values": k_values, "inertias": inertias, "silhouettes": silhouettes, "optimal_k": optimal_k}


def compare_clustering_algorithms(X: ArrayLike, n_clusters: int) -> dict[str, float]:
    """Compare k-means and hierarchical clustering."""
    results = {}
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    results["kmeans"] = float(silhouette_score(X, kmeans.fit_predict(X)))
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    results["hierarchical"] = float(silhouette_score(X, agg.fit_predict(X)))
    return results


if __name__ == "__main__":
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    results = find_optimal_k(X)
    print(f"Optimal k: {results['optimal_k']}")
    print(f"Comparison: {compare_clustering_algorithms(X, 4)}")
