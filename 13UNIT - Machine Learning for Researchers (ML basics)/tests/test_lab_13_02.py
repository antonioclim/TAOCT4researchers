"""
Tests for Lab 13_02: Unsupervised Learning.

Coverage target: ≥80%
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lab.lab_13_02_unsupervised_learning import (
    anomaly_detection_isolation_forest,
    cluster_profiling,
    compare_reduction_methods,
    customer_segmentation_demo,
    dbscan_clustering,
    document_clustering_demo,
    evaluate_clustering,
    find_optimal_k,
    hierarchical_clustering,
    kmeans_clustering,
    pca_analysis,
    preprocessing_clustering_pipeline,
    tsne_visualisation,
    visualise_pca_loadings,
)


class TestClustering:
    """Tests for Section 1: Clustering Algorithms."""

    def test_kmeans_clustering(self, clustering_data):
        """Test k-means clustering."""
        X, _ = clustering_data
        results = kmeans_clustering(X, n_clusters=4)
        
        assert len(results["labels"]) == len(X)
        assert len(np.unique(results["labels"])) == 4
        assert results["centroids"].shape == (4, X.shape[1])
        assert -1 <= results["silhouette"] <= 1
        assert results["inertia"] > 0

    def test_kmeans_different_k(self, clustering_data):
        """Test k-means with different k values."""
        X, _ = clustering_data
        for k in [2, 3, 5]:
            results = kmeans_clustering(X, n_clusters=k)
            assert len(np.unique(results["labels"])) == k

    def test_find_optimal_k(self, clustering_data):
        """Test optimal k finding."""
        X, _ = clustering_data
        results = find_optimal_k(X, k_range=range(2, 7))
        
        assert len(results["k_values"]) == 5
        assert len(results["inertias"]) == 5
        assert len(results["silhouettes"]) == 5
        assert 2 <= results["optimal_k_silhouette"] <= 6
        
        # Inertia should decrease with more clusters
        assert results["inertias"][0] > results["inertias"][-1]

    def test_hierarchical_clustering(self, clustering_data):
        """Test hierarchical clustering."""
        X, _ = clustering_data
        results = hierarchical_clustering(X, n_clusters=4)
        
        assert len(results["labels"]) == len(X)
        assert len(np.unique(results["labels"])) == 4
        assert -1 <= results["silhouette"] <= 1
        assert results["linkage_matrix"] is not None

    def test_hierarchical_linkage_methods(self, clustering_data):
        """Test different linkage methods."""
        X, _ = clustering_data
        for method in ["ward", "complete", "average"]:
            results = hierarchical_clustering(X, n_clusters=3, linkage_method=method)
            assert len(np.unique(results["labels"])) == 3

    def test_dbscan_clustering(self, clustering_data):
        """Test DBSCAN clustering."""
        X, _ = clustering_data
        results = dbscan_clustering(X, eps=1.0, min_samples=5)
        
        assert len(results["labels"]) == len(X)
        assert results["n_clusters"] >= 1
        assert results["n_noise"] >= 0
        assert results["n_clusters"] + (1 if results["n_noise"] > 0 else 0) <= len(np.unique(results["labels"])) + 1

    def test_evaluate_clustering(self, clustering_data):
        """Test clustering evaluation metrics."""
        X, true_labels = clustering_data
        metrics = evaluate_clustering(X, true_labels)
        
        assert "silhouette" in metrics
        assert "calinski_harabasz" in metrics
        assert "davies_bouldin" in metrics
        
        assert -1 <= metrics["silhouette"] <= 1
        assert metrics["calinski_harabasz"] > 0
        assert metrics["davies_bouldin"] >= 0

    def test_evaluate_clustering_single_cluster(self):
        """Test evaluation with single cluster returns None."""
        X = np.random.randn(50, 3)
        labels = np.zeros(50, dtype=int)  # All same cluster
        metrics = evaluate_clustering(X, labels)
        
        assert metrics["silhouette"] is None


class TestDimensionalityReduction:
    """Tests for Section 2: Dimensionality Reduction."""

    def test_pca_analysis(self, clustering_data):
        """Test PCA analysis."""
        X, _ = clustering_data
        results = pca_analysis(X, n_components=2)
        
        assert results["X_transformed"].shape == (len(X), 2)
        assert len(results["explained_variance_ratio"]) == 2
        assert np.sum(results["explained_variance_ratio"]) <= 1.0
        assert len(results["cumulative_variance"]) == 2

    def test_pca_variance_threshold(self, clustering_data):
        """Test PCA with variance threshold."""
        X, _ = clustering_data
        results = pca_analysis(X, n_components=0.95)
        
        # Should retain components explaining 95% variance
        assert results["cumulative_variance"][-1] >= 0.95

    def test_visualise_pca_loadings(self, clustering_data):
        """Test PCA loadings visualisation."""
        X, _ = clustering_data
        pca_model = PCA(n_components=2).fit(X)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        fig = visualise_pca_loadings(pca_model, feature_names)
        
        assert fig is not None

    def test_tsne_visualisation(self, clustering_data):
        """Test t-SNE visualisation."""
        X, labels = clustering_data
        # Use smaller dataset for speed
        X_small = X[:100]
        labels_small = labels[:100]
        
        fig, X_2d = tsne_visualisation(X_small, labels=labels_small, perplexity=20)
        
        assert fig is not None
        assert X_2d.shape == (100, 2)

    def test_compare_reduction_methods(self, clustering_data):
        """Test comparison of reduction methods."""
        X, labels = clustering_data
        X_small = X[:100]
        labels_small = labels[:100]
        
        fig, data = compare_reduction_methods(X_small, labels=labels_small)
        
        assert fig is not None
        assert "pca" in data
        assert "tsne" in data
        assert data["pca"].shape == (100, 2)
        assert data["tsne"].shape == (100, 2)


class TestPracticalPipelines:
    """Tests for Section 3: Practical Pipelines."""

    def test_preprocessing_clustering_pipeline(self, clustering_data):
        """Test preprocessing + clustering pipeline."""
        X, _ = clustering_data
        results = preprocessing_clustering_pipeline(X, n_clusters=4)
        
        assert len(results["labels"]) == len(X)
        assert results["pipeline"] is not None
        assert -1 <= results["silhouette"] <= 1

    def test_cluster_profiling(self, clustering_data):
        """Test cluster profiling."""
        X, _ = clustering_data
        results = kmeans_clustering(X, n_clusters=3)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        profile = cluster_profiling(X, results["labels"], feature_names)
        
        assert isinstance(profile, pd.DataFrame)
        # Should have stats for 3 clusters
        assert len(profile) == 3

    def test_anomaly_detection(self, clustering_data):
        """Test anomaly detection with Isolation Forest."""
        X, _ = clustering_data
        results = anomaly_detection_isolation_forest(X, contamination=0.1)
        
        assert len(results["labels"]) == len(X)
        assert len(results["scores"]) == len(X)
        # With 10% contamination, expect ~10% anomalies
        expected_anomalies = int(len(X) * 0.1)
        assert abs(results["n_anomalies"] - expected_anomalies) < expected_anomalies * 0.5


class TestResearchApplications:
    """Tests for Section 4: Research Applications."""

    def test_customer_segmentation_demo(self):
        """Test customer segmentation demonstration."""
        results = customer_segmentation_demo(n_customers=200)
        
        assert "segments" in results
        assert "profile" in results
        assert "optimal_k" in results
        assert "data" in results
        
        assert len(results["segments"]) == 200
        assert 2 <= results["optimal_k"] <= 7

    def test_document_clustering_demo(self):
        """Test document clustering demonstration."""
        results = document_clustering_demo()
        
        assert "labels" in results
        assert "n_clusters" in results
        assert "top_terms" in results
        
        assert results["n_clusters"] == 3
        assert len(results["top_terms"]) == 3
        # Each cluster should have top terms
        for cluster_id in range(3):
            assert len(results["top_terms"][cluster_id]) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_kmeans_single_cluster(self):
        """Test k-means with k=1."""
        X = np.random.randn(50, 3)
        results = kmeans_clustering(X, n_clusters=1)
        assert len(np.unique(results["labels"])) == 1

    def test_pca_all_components(self):
        """Test PCA keeping all components."""
        X = np.random.randn(50, 5)
        results = pca_analysis(X, n_components=None)
        assert results["n_components"] == 5
        assert np.isclose(results["cumulative_variance"][-1], 1.0)

    def test_dbscan_all_noise(self):
        """Test DBSCAN when all points are noise."""
        # Sparse data with high eps requirement
        X = np.random.randn(20, 3) * 10
        results = dbscan_clustering(X, eps=0.1, min_samples=10)
        
        # Most or all points should be noise
        assert results["n_noise"] > 0

    def test_reproducibility(self, clustering_data):
        """Test that results are reproducible with same random state."""
        X, _ = clustering_data
        
        results1 = kmeans_clustering(X, n_clusters=3, random_state=42)
        results2 = kmeans_clustering(X, n_clusters=3, random_state=42)
        
        np.testing.assert_array_equal(results1["labels"], results2["labels"])
        np.testing.assert_array_almost_equal(results1["centroids"], results2["centroids"])
