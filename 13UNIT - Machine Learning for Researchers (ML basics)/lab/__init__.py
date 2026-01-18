"""
13UNIT Laboratory Package: Machine Learning for Researchers

This package provides implementations for supervised and unsupervised
machine learning workflows using scikit-learn.

Modules:
    lab_13_01_supervised_learning: Classification, regression, validation
    lab_13_02_unsupervised_learning: Clustering, dimensionality reduction
"""

from lab.lab_13_01_supervised_learning import (
    load_and_explore,
    preprocess_features,
    train_test_stratified_split,
    train_classifier,
    evaluate_classifier,
    compare_classifiers,
    train_regressor,
    evaluate_regressor,
    cross_validate_model,
    grid_search_cv,
    nested_cross_validation,
    learning_curve_analysis,
    demonstrate_overfitting,
    demonstrate_data_leakage,
    handle_class_imbalance,
)

from lab.lab_13_02_unsupervised_learning import (
    kmeans_clustering,
    find_optimal_k,
    hierarchical_clustering,
    dbscan_clustering,
    evaluate_clustering,
    pca_analysis,
    visualise_pca_loadings,
    tsne_visualisation,
    compare_reduction_methods,
    preprocessing_clustering_pipeline,
    cluster_profiling,
    anomaly_detection_isolation_forest,
    customer_segmentation_demo,
    document_clustering_demo,
)

__all__ = [
    # Supervised Learning
    "load_and_explore",
    "preprocess_features",
    "train_test_stratified_split",
    "train_classifier",
    "evaluate_classifier",
    "compare_classifiers",
    "train_regressor",
    "evaluate_regressor",
    "cross_validate_model",
    "grid_search_cv",
    "nested_cross_validation",
    "learning_curve_analysis",
    "demonstrate_overfitting",
    "demonstrate_data_leakage",
    "handle_class_imbalance",
    # Unsupervised Learning
    "kmeans_clustering",
    "find_optimal_k",
    "hierarchical_clustering",
    "dbscan_clustering",
    "evaluate_clustering",
    "pca_analysis",
    "visualise_pca_loadings",
    "tsne_visualisation",
    "compare_reduction_methods",
    "preprocessing_clustering_pipeline",
    "cluster_profiling",
    "anomaly_detection_isolation_forest",
    "customer_segmentation_demo",
    "document_clustering_demo",
]

__version__ = "1.0.0"
__author__ = "Antonio Clim"
