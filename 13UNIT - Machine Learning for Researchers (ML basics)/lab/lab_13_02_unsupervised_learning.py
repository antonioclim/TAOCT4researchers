"""
Lab 13_02: Unsupervised Learning

This laboratory module implements unsupervised learning techniques
including clustering algorithms and dimensionality reduction methods.

Learning Objectives:
    LO6: Implement unsupervised learning techniques: clustering and
         dimensionality reduction

Estimated Duration: ~90 minutes

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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = NDArray[np.floating[Any]] | NDArray[np.integer[Any]]


# =============================================================================
# SECTION 1: Clustering Algorithms (~120 lines)
# =============================================================================


def kmeans_clustering(
    X: ArrayLike,
    n_clusters: int = 3,
    random_state: int = 42,
    n_init: int = 10,
) -> dict[str, Any]:
    """
    Perform k-means clustering with evaluation metrics.

    Args:
        X: Feature matrix.
        n_clusters: Number of clusters to find.
        random_state: Random seed for reproducibility.
        n_init: Number of initialisations to run.

    Returns:
        Dictionary containing:
            - 'labels': Cluster assignments
            - 'centroids': Cluster centres
            - 'inertia': Within-cluster sum of squares
            - 'silhouette': Silhouette score
            - 'model': Fitted KMeans object

    Example:
        >>> X, _ = make_blobs(n_samples=100, centers=3)
        >>> results = kmeans_clustering(X, n_clusters=3)
        >>> len(np.unique(results['labels'])) == 3
        True
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(X)

    results: dict[str, Any] = {
        "labels": labels,
        "centroids": kmeans.cluster_centers_,
        "inertia": float(kmeans.inertia_),
        "silhouette": float(silhouette_score(X, labels)),
        "model": kmeans,
    }

    logger.info(
        "k-Means complete. k=%d, Silhouette=%.3f, Inertia=%.1f",
        n_clusters,
        results["silhouette"],
        results["inertia"],
    )

    return results


def find_optimal_k(
    X: ArrayLike,
    k_range: range | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Find optimal number of clusters using elbow and silhouette methods.

    Args:
        X: Feature matrix.
        k_range: Range of k values to evaluate. Defaults to range(2, 11).
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'k_values': Evaluated k values
            - 'inertias': Inertia for each k
            - 'silhouettes': Silhouette score for each k
            - 'optimal_k_silhouette': k with highest silhouette
            - 'results_df': DataFrame with all results

    Example:
        >>> X, _ = make_blobs(n_samples=200, centers=4)
        >>> results = find_optimal_k(X)
        >>> results['optimal_k_silhouette'] > 1
        True
    """
    if k_range is None:
        k_range = range(2, 11)

    k_values = list(k_range)
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        inertias.append(float(kmeans.inertia_))
        silhouettes.append(float(silhouette_score(X, labels)))

    optimal_k = k_values[np.argmax(silhouettes)]

    results_df = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
        "silhouette": silhouettes,
    })

    logger.info("Optimal k (by silhouette): %d", optimal_k)

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k_silhouette": optimal_k,
        "results_df": results_df,
    }


def hierarchical_clustering(
    X: ArrayLike,
    n_clusters: int = 3,
    linkage_method: str = "ward",
) -> dict[str, Any]:
    """
    Perform agglomerative hierarchical clustering.

    Args:
        X: Feature matrix.
        n_clusters: Number of clusters to find.
        linkage_method: Linkage criterion. Options: 'ward', 'complete',
            'average', 'single'.

    Returns:
        Dictionary containing:
            - 'labels': Cluster assignments
            - 'silhouette': Silhouette score
            - 'linkage_matrix': Linkage matrix for dendrogram
            - 'model': Fitted AgglomerativeClustering object

    Example:
        >>> X, _ = make_blobs(n_samples=100, centers=3)
        >>> results = hierarchical_clustering(X, n_clusters=3)
        >>> len(np.unique(results['labels'])) == 3
        True
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage_method,
    )
    labels = model.fit_predict(X)

    # Compute linkage matrix for dendrogram
    linkage_matrix = linkage(X, method=linkage_method)

    results: dict[str, Any] = {
        "labels": labels,
        "silhouette": float(silhouette_score(X, labels)),
        "linkage_matrix": linkage_matrix,
        "model": model,
    }

    logger.info(
        "Hierarchical clustering complete. Silhouette=%.3f",
        results["silhouette"],
    )

    return results


def dbscan_clustering(
    X: ArrayLike,
    eps: float = 0.5,
    min_samples: int = 5,
) -> dict[str, Any]:
    """
    Perform DBSCAN density-based clustering.

    Args:
        X: Feature matrix.
        eps: Maximum distance between samples in same neighbourhood.
        min_samples: Minimum samples in neighbourhood for core point.

    Returns:
        Dictionary containing:
            - 'labels': Cluster assignments (-1 indicates noise)
            - 'n_clusters': Number of clusters found
            - 'n_noise': Number of noise points
            - 'silhouette': Silhouette score (if >1 cluster found)
            - 'model': Fitted DBSCAN object

    Example:
        >>> X, _ = make_moons(n_samples=200, noise=0.1)
        >>> results = dbscan_clustering(X, eps=0.3)
        >>> results['n_clusters'] >= 1
        True
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    results: dict[str, Any] = {
        "labels": labels,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "model": model,
    }

    # Silhouette requires at least 2 clusters and non-noise points
    if n_clusters > 1 and n_noise < len(labels):
        mask = labels != -1
        if np.sum(mask) > 0:
            results["silhouette"] = float(silhouette_score(X[mask], labels[mask]))
    else:
        results["silhouette"] = None

    logger.info(
        "DBSCAN complete. Clusters=%d, Noise points=%d",
        n_clusters,
        n_noise,
    )

    return results


def evaluate_clustering(
    X: ArrayLike,
    labels: ArrayLike,
) -> dict[str, float | None]:
    """
    Compute multiple clustering evaluation metrics.

    Args:
        X: Feature matrix.
        labels: Cluster assignments.

    Returns:
        Dictionary containing:
            - 'silhouette': Silhouette score (-1 to 1, higher better)
            - 'calinski_harabasz': Calinski-Harabasz index (higher better)
            - 'davies_bouldin': Davies-Bouldin index (lower better)

    Example:
        >>> X, labels = make_blobs(n_samples=100, centers=3)
        >>> metrics = evaluate_clustering(X, labels)
        >>> 'silhouette' in metrics
        True
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        logger.warning("Cannot evaluate clustering with < 2 clusters")
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    # Filter out noise points for metrics
    mask = labels != -1
    X_filtered = X[mask]
    labels_filtered = labels[mask]

    if len(np.unique(labels_filtered)) < 2:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}

    metrics: dict[str, float | None] = {
        "silhouette": float(silhouette_score(X_filtered, labels_filtered)),
        "calinski_harabasz": float(calinski_harabasz_score(X_filtered, labels_filtered)),
        "davies_bouldin": float(davies_bouldin_score(X_filtered, labels_filtered)),
    }

    logger.info(
        "Clustering metrics - Silhouette: %.3f, CH: %.1f, DB: %.3f",
        metrics["silhouette"],
        metrics["calinski_harabasz"],
        metrics["davies_bouldin"],
    )

    return metrics


# =============================================================================
# SECTION 2: Dimensionality Reduction (~100 lines)
# =============================================================================


def pca_analysis(
    X: ArrayLike,
    n_components: int | float | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Perform PCA with variance analysis.

    Args:
        X: Feature matrix.
        n_components: Number of components. If float (0-1), selects components
            to retain that proportion of variance. If None, keeps all.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'X_transformed': Transformed data
            - 'explained_variance_ratio': Variance explained by each component
            - 'cumulative_variance': Cumulative variance explained
            - 'n_components': Number of components retained
            - 'model': Fitted PCA object

    Example:
        >>> X = np.random.randn(100, 10)
        >>> results = pca_analysis(X, n_components=3)
        >>> results['X_transformed'].shape[1] == 3
        True
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_transformed = pca.fit_transform(X)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    results: dict[str, Any] = {
        "X_transformed": X_transformed,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cumulative_variance,
        "n_components": pca.n_components_,
        "components": pca.components_,
        "model": pca,
    }

    logger.info(
        "PCA complete. Components=%d, Total variance=%.3f",
        results["n_components"],
        cumulative_variance[-1] if len(cumulative_variance) > 0 else 0,
    )

    return results


def visualise_pca_loadings(
    pca_model: PCA,
    feature_names: list[str] | None = None,
    n_components: int = 2,
) -> plt.Figure:
    """
    Visualise PCA loadings (component weights).

    Args:
        pca_model: Fitted PCA model.
        feature_names: Names of original features.
        n_components: Number of components to visualise.

    Returns:
        Matplotlib Figure object.

    Example:
        >>> pca = PCA(n_components=2).fit(X)
        >>> fig = visualise_pca_loadings(pca)
        >>> isinstance(fig, plt.Figure)
        True
    """
    components = pca_model.components_[:n_components]
    n_features = components.shape[1]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    fig, axes = plt.subplots(1, n_components, figsize=(6 * n_components, 5))

    if n_components == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        loadings = components[i]
        indices = np.arange(len(loadings))

        ax.barh(indices, loadings)
        ax.set_yticks(indices)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Loading")
        ax.set_title(f"PC{i + 1} Loadings")
        ax.axvline(x=0, color="k", linestyle="--", linewidth=0.5)

    fig.tight_layout()

    return fig


def tsne_visualisation(
    X: ArrayLike,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    labels: ArrayLike | None = None,
) -> tuple[plt.Figure, ArrayLike]:
    """
    Perform t-SNE for 2D visualisation.

    Args:
        X: Feature matrix.
        perplexity: Perplexity parameter (typically 5-50).
        n_iter: Number of iterations.
        random_state: Random seed for reproducibility.
        labels: Optional cluster labels for colouring.

    Returns:
        Tuple containing:
            - Matplotlib Figure object
            - Transformed 2D coordinates

    Example:
        >>> X = np.random.randn(100, 10)
        >>> fig, X_2d = tsne_visualisation(X)
        >>> X_2d.shape[1] == 2
        True
    """
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
    )
    X_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.7,
            edgecolors="k",
            linewidth=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
    else:
        ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            alpha=0.7,
            edgecolors="k",
            linewidth=0.5,
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE Visualisation (perplexity={perplexity})")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    logger.info("t-SNE complete. Perplexity=%d", perplexity)

    return fig, X_2d


def compare_reduction_methods(
    X: ArrayLike,
    labels: ArrayLike | None = None,
    random_state: int = 42,
) -> tuple[plt.Figure, dict[str, ArrayLike]]:
    """
    Compare PCA and t-SNE dimensionality reduction.

    Args:
        X: Feature matrix.
        labels: Optional labels for colouring.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple containing:
            - Matplotlib Figure with side-by-side comparison
            - Dictionary with transformed data for each method

    Example:
        >>> X = np.random.randn(100, 10)
        >>> fig, data = compare_reduction_methods(X)
        >>> 'pca' in data and 'tsne' in data
        True
    """
    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    X_tsne = tsne.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, X_reduced, title in [
        (axes[0], X_pca, "PCA"),
        (axes[1], X_tsne, "t-SNE"),
    ]:
        if labels is not None:
            scatter = ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.7,
                edgecolors="k",
                linewidth=0.5,
            )
            plt.colorbar(scatter, ax=ax)
        else:
            ax.scatter(
                X_reduced[:, 0],
                X_reduced[:, 1],
                alpha=0.7,
                edgecolors="k",
                linewidth=0.5,
            )

        ax.set_xlabel(f"{title} 1")
        ax.set_ylabel(f"{title} 2")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, {"pca": X_pca, "tsne": X_tsne}


# =============================================================================
# SECTION 3: Practical Pipelines (~80 lines)
# =============================================================================


def preprocessing_clustering_pipeline(
    X: ArrayLike,
    n_clusters: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Create a pipeline combining preprocessing and clustering.

    Args:
        X: Feature matrix.
        n_clusters: Number of clusters.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'labels': Cluster assignments
            - 'pipeline': Fitted pipeline
            - 'silhouette': Silhouette score

    Example:
        >>> X = np.random.randn(100, 10)
        >>> results = preprocessing_clustering_pipeline(X, n_clusters=3)
        >>> 'labels' in results
        True
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)),
    ])

    labels = pipeline.fit_predict(X)
    X_scaled = pipeline.named_steps["scaler"].transform(X)

    results: dict[str, Any] = {
        "labels": labels,
        "pipeline": pipeline,
        "silhouette": float(silhouette_score(X_scaled, labels)),
    }

    logger.info("Pipeline clustering complete. Silhouette=%.3f", results["silhouette"])

    return results


def cluster_profiling(
    X: ArrayLike,
    labels: ArrayLike,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate cluster profiles showing mean feature values per cluster.

    Args:
        X: Feature matrix.
        labels: Cluster assignments.
        feature_names: Names of features.

    Returns:
        DataFrame with cluster profiles.

    Example:
        >>> X, labels = make_blobs(n_samples=100, centers=3)
        >>> profile = cluster_profiling(X, labels)
        >>> 'cluster' in profile.columns
        True
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)
    df["cluster"] = labels

    # Compute mean and std per cluster
    profile = df.groupby("cluster").agg(["mean", "std"])

    # Add cluster sizes
    cluster_sizes = df["cluster"].value_counts().sort_index()

    logger.info("Cluster sizes: %s", dict(cluster_sizes))

    return profile


def anomaly_detection_isolation_forest(
    X: ArrayLike,
    contamination: float = 0.1,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Detect anomalies using Isolation Forest.

    Args:
        X: Feature matrix.
        contamination: Expected proportion of outliers.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'labels': Predictions (1=normal, -1=anomaly)
            - 'scores': Anomaly scores (lower = more anomalous)
            - 'n_anomalies': Count of detected anomalies
            - 'anomaly_indices': Indices of anomalous samples
            - 'model': Fitted IsolationForest object

    Example:
        >>> X = np.random.randn(100, 5)
        >>> results = anomaly_detection_isolation_forest(X)
        >>> results['n_anomalies'] >= 0
        True
    """
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    labels = model.fit_predict(X)
    scores = model.decision_function(X)

    anomaly_mask = labels == -1
    n_anomalies = int(np.sum(anomaly_mask))
    anomaly_indices = np.where(anomaly_mask)[0]

    results: dict[str, Any] = {
        "labels": labels,
        "scores": scores,
        "n_anomalies": n_anomalies,
        "anomaly_indices": anomaly_indices,
        "model": model,
    }

    logger.info(
        "Anomaly detection complete. Found %d anomalies (%.1f%%)",
        n_anomalies,
        100 * n_anomalies / len(X),
    )

    return results


# =============================================================================
# SECTION 4: Research Applications (~50 lines)
# =============================================================================


def customer_segmentation_demo(
    n_customers: int = 500,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Demonstrate customer segmentation using clustering.

    Args:
        n_customers: Number of synthetic customers to generate.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'segments': Cluster labels
            - 'profile': Segment profiles
            - 'optimal_k': Optimal number of segments
            - 'data': Generated customer data

    Example:
        >>> results = customer_segmentation_demo(n_customers=200)
        >>> 'segments' in results
        True
    """
    np.random.seed(random_state)

    # Generate synthetic customer data
    data = pd.DataFrame({
        "annual_income": np.random.normal(50000, 20000, n_customers).clip(20000, 150000),
        "spending_score": np.random.uniform(1, 100, n_customers),
        "age": np.random.normal(40, 15, n_customers).clip(18, 80).astype(int),
        "purchase_frequency": np.random.poisson(12, n_customers),
    })

    X = data.values

    # Find optimal segments
    optimal_results = find_optimal_k(X, k_range=range(2, 8), random_state=random_state)
    optimal_k = optimal_results["optimal_k_silhouette"]

    # Final clustering
    clustering_results = preprocessing_clustering_pipeline(X, n_clusters=optimal_k, random_state=random_state)

    # Profile segments
    profile = cluster_profiling(X, clustering_results["labels"], list(data.columns))

    logger.info("Customer segmentation complete. %d segments identified", optimal_k)

    return {
        "segments": clustering_results["labels"],
        "profile": profile,
        "optimal_k": optimal_k,
        "data": data,
        "silhouette": clustering_results["silhouette"],
    }


def document_clustering_demo(
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Demonstrate document clustering using TF-IDF and k-means.

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
            - 'labels': Document cluster assignments
            - 'n_clusters': Number of clusters found
            - 'top_terms': Top terms per cluster

    Example:
        >>> results = document_clustering_demo()
        >>> 'labels' in results
        True
    """
    # Sample documents
    documents = [
        "machine learning algorithms data science",
        "neural networks deep learning artificial intelligence",
        "statistics probability regression analysis",
        "data mining clustering classification",
        "python programming software development",
        "web development javascript html css",
        "database sql nosql mongodb",
        "cloud computing aws azure deployment",
        "natural language processing text mining",
        "computer vision image recognition",
    ]

    # TF-IDF vectorisation
    vectoriser = TfidfVectorizer(stop_words="english")
    X = vectoriser.fit_transform(documents)

    # Find optimal clusters
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    # Get top terms per cluster
    feature_names = vectoriser.get_feature_names_out()
    top_terms: dict[int, list[str]] = {}

    for cluster_id in range(n_clusters):
        centroid = kmeans.cluster_centers_[cluster_id]
        top_indices = centroid.argsort()[-5:][::-1]
        top_terms[cluster_id] = [feature_names[i] for i in top_indices]

    logger.info("Document clustering complete. %d clusters", n_clusters)

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "top_terms": top_terms,
        "documents": documents,
    }


# =============================================================================
# Main execution for demonstration
# =============================================================================


if __name__ == "__main__":
    # Generate sample data
    X, true_labels = make_blobs(
        n_samples=300,
        centers=4,
        cluster_std=1.0,
        random_state=42,
    )

    logger.info("Generated sample data: %d samples, %d features", X.shape[0], X.shape[1])

    # K-Means clustering
    logger.info("\n--- K-Means Clustering ---")
    kmeans_results = kmeans_clustering(X, n_clusters=4)

    # Find optimal k
    logger.info("\n--- Finding Optimal k ---")
    optimal_k_results = find_optimal_k(X)
    logger.info("Optimal k: %d", optimal_k_results["optimal_k_silhouette"])

    # Hierarchical clustering
    logger.info("\n--- Hierarchical Clustering ---")
    hierarchical_results = hierarchical_clustering(X, n_clusters=4)

    # DBSCAN
    logger.info("\n--- DBSCAN Clustering ---")
    X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    dbscan_results = dbscan_clustering(X_moons, eps=0.3)

    # PCA
    logger.info("\n--- PCA Analysis ---")
    pca_results = pca_analysis(X, n_components=2)
    logger.info("Variance explained: %s", pca_results["explained_variance_ratio"])

    # t-SNE
    logger.info("\n--- t-SNE Visualisation ---")
    fig, X_tsne = tsne_visualisation(X, labels=true_labels)

    # Anomaly detection
    logger.info("\n--- Anomaly Detection ---")
    anomaly_results = anomaly_detection_isolation_forest(X)

    # Customer segmentation demo
    logger.info("\n--- Customer Segmentation Demo ---")
    customer_results = customer_segmentation_demo()

    # Document clustering demo
    logger.info("\n--- Document Clustering Demo ---")
    doc_results = document_clustering_demo()

    logger.info("\nLab 13_02 demonstration complete.")
