"""
Lab 13_02 Solution: Unsupervised Learning

Complete solution demonstrating clustering and dimensionality reduction.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Execute complete unsupervised learning workflow."""
    # Generate data
    logger.info("Generating synthetic data with 4 clusters...")
    X, true_labels = make_blobs(
        n_samples=300, centers=4, cluster_std=1.0, random_state=42
    )
    
    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Data shape: {X.shape}")
    
    # K-Means
    logger.info("\n=== K-Means Clustering ===")
    
    # Find optimal k
    k_range = range(2, 10)
    silhouettes = []
    inertias = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        silhouettes.append(silhouette_score(X_scaled, labels))
        inertias.append(kmeans.inertia_)
    
    optimal_k = list(k_range)[np.argmax(silhouettes)]
    logger.info(f"Optimal k by silhouette: {optimal_k}")
    
    # Final k-means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    logger.info(f"Silhouette: {silhouette_score(X_scaled, kmeans_labels):.4f}")
    
    # Hierarchical
    logger.info("\n=== Hierarchical Clustering ===")
    hierarchical = AgglomerativeClustering(n_clusters=4, linkage="ward")
    hier_labels = hierarchical.fit_predict(X_scaled)
    logger.info(f"Silhouette: {silhouette_score(X_scaled, hier_labels):.4f}")
    
    # DBSCAN
    logger.info("\n=== DBSCAN Clustering ===")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = np.sum(dbscan_labels == -1)
    logger.info(f"Clusters found: {n_clusters}, Noise points: {n_noise}")
    
    # PCA
    logger.info("\n=== PCA Dimensionality Reduction ===")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"Explained variance: {pca.explained_variance_ratio_}")
    logger.info(f"Total variance: {sum(pca.explained_variance_ratio_):.4f}")
    
    # Keep 95% variance
    pca_95 = PCA(n_components=0.95)
    X_95 = pca_95.fit_transform(X_scaled)
    logger.info(f"Components for 95% variance: {pca_95.n_components_}")
    
    # t-SNE
    logger.info("\n=== t-SNE Visualisation ===")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    logger.info(f"t-SNE shape: {X_tsne.shape}")
    
    # Compare algorithms
    logger.info("\n=== Algorithm Comparison ===")
    algorithms = {
        "K-Means": KMeans(n_clusters=4, random_state=42, n_init=10),
        "Hierarchical (Ward)": AgglomerativeClustering(n_clusters=4, linkage="ward"),
        "Hierarchical (Complete)": AgglomerativeClustering(n_clusters=4, linkage="complete"),
    }
    
    for name, algo in algorithms.items():
        labels = algo.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        logger.info(f"{name}: Silhouette = {sil:.4f}")
    
    logger.info("\nLab 13_02 complete!")


if __name__ == "__main__":
    main()
