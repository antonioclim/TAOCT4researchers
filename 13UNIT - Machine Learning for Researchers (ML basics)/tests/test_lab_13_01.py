"""
Tests for Lab 13_01: Supervised Learning Pipeline.

Coverage target: ≥80%
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from lab.lab_13_01_supervised_learning import (
    classification_threshold_analysis,
    compare_classifiers,
    create_classification_pipeline,
    cross_validate_model,
    demonstrate_data_leakage,
    demonstrate_overfitting,
    evaluate_classifier,
    evaluate_regressor,
    feature_importance_analysis,
    grid_search_cv,
    handle_class_imbalance,
    learning_curve_analysis,
    load_and_explore,
    model_comparison_report,
    nested_cross_validation,
    plot_confusion_matrix,
    plot_predictions_vs_actual,
    plot_residuals,
    plot_roc_curves,
    preprocess_features,
    stratification_verification,
    train_classifier,
    train_regressor,
    train_test_stratified_split,
)


class TestDataPreparation:
    """Tests for Section 1: Data Preparation."""

    def test_load_breast_cancer(self):
        """Test loading breast cancer dataset."""
        data = load_and_explore("breast_cancer")
        assert data["n_samples"] == 569
        assert data["n_features"] == 30
        assert "class_distribution" in data
        assert len(data["feature_names"]) == 30

    def test_load_diabetes(self):
        """Test loading diabetes dataset."""
        data = load_and_explore("diabetes")
        assert data["n_samples"] == 442
        assert data["n_features"] == 10
        assert "target_stats" in data

    def test_load_synthetic(self):
        """Test loading synthetic dataset."""
        data = load_and_explore("synthetic")
        assert data["n_samples"] == 1000
        assert data["n_features"] == 20

    def test_load_invalid_dataset(self):
        """Test error handling for invalid dataset."""
        with pytest.raises(ValueError):
            load_and_explore("invalid_dataset")

    def test_preprocess_features(self, classification_data):
        """Test feature preprocessing."""
        X, _ = classification_data
        preprocessor, X_transformed = preprocess_features(X)
        assert X_transformed.shape == X.shape
        # Check standardisation (mean ≈ 0, std ≈ 1)
        assert np.abs(np.mean(X_transformed)) < 0.1
        assert np.abs(np.std(X_transformed) - 1) < 0.1

    def test_stratified_split(self, classification_data):
        """Test stratified train/test split."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)
        
        assert len(X_train) == 160  # 80% of 200
        assert len(X_test) == 40    # 20% of 200
        
        # Check stratification
        train_ratio = np.mean(y_train)
        test_ratio = np.mean(y_test)
        assert abs(train_ratio - test_ratio) < 0.1

    def test_stratification_verification(self, classification_data):
        """Test stratification verification function."""
        X, y = classification_data
        _, _, y_train, y_test = train_test_stratified_split(X, y)
        result = stratification_verification(y_train, y_test)
        assert result["stratification_valid"] is True
        assert result["max_difference"] < 0.05


class TestClassification:
    """Tests for Section 2: Classification Pipeline."""

    def test_train_logistic_classifier(self, classification_data):
        """Test training logistic regression."""
        X, y = classification_data
        model = train_classifier(X, y, "logistic")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_random_forest(self, classification_data):
        """Test training random forest."""
        X, y = classification_data
        model = train_classifier(X, y, "random_forest")
        assert hasattr(model, "feature_importances_")

    def test_train_invalid_classifier(self, classification_data):
        """Test error for invalid classifier type."""
        X, y = classification_data
        with pytest.raises(ValueError):
            train_classifier(X, y, "invalid_classifier")

    def test_evaluate_classifier(self, classification_data):
        """Test classifier evaluation."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)
        model = train_classifier(X_train, y_train, "logistic")
        metrics = evaluate_classifier(model, X_test, y_test)
        
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert "roc_auc" in metrics
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_compare_classifiers(self, classification_data):
        """Test classifier comparison."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)
        comparison = compare_classifiers(
            X_train, y_train, X_test, y_test,
            classifiers=["logistic", "decision_tree"]
        )
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "f1" in comparison.columns

    def test_plot_confusion_matrix(self, classification_data):
        """Test confusion matrix plotting."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_stratified_split(X, y)
        model = train_classifier(X_train, y_train, "logistic")
        y_pred = model.predict(X_test)
        fig = plot_confusion_matrix(y_test, y_pred)
        assert fig is not None


class TestRegression:
    """Tests for Section 3: Regression Pipeline."""

    def test_train_linear_regressor(self, regression_data):
        """Test training linear regression."""
        X, y = regression_data
        model = train_regressor(X, y, "linear")
        assert hasattr(model, "coef_")

    def test_train_ridge_regressor(self, regression_data):
        """Test training ridge regression."""
        X, y = regression_data
        model = train_regressor(X, y, "ridge")
        assert hasattr(model, "alpha")

    def test_evaluate_regressor(self, regression_data):
        """Test regressor evaluation."""
        X, y = regression_data
        X_train, X_test = X[:160], X[160:]
        y_train, y_test = y[:160], y[160:]
        model = train_regressor(X_train, y_train, "linear")
        metrics = evaluate_regressor(model, X_test, y_test)
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))


class TestModelSelection:
    """Tests for Section 4: Model Selection."""

    def test_cross_validate_model(self, classification_data):
        """Test cross-validation."""
        X, y = classification_data
        model = LogisticRegression(max_iter=1000)
        results = cross_validate_model(model, X, y, cv=3)
        
        assert len(results["test_scores"]) == 3
        assert 0 <= results["mean_test_score"] <= 1
        assert results["std_test_score"] >= 0

    def test_grid_search_cv(self, classification_data):
        """Test grid search cross-validation."""
        X, y = classification_data
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.1, 1.0]}
        results = grid_search_cv(model, param_grid, X, y, cv=3)
        
        assert "best_params" in results
        assert "best_score" in results
        assert results["best_params"]["C"] in [0.1, 1.0]

    def test_nested_cross_validation(self, classification_data):
        """Test nested cross-validation."""
        X, y = classification_data
        model = LogisticRegression(max_iter=1000)
        param_grid = {"C": [0.1, 1.0]}
        results = nested_cross_validation(
            model, param_grid, X, y, outer_cv=3, inner_cv=2
        )
        
        assert len(results["outer_scores"]) == 3
        assert len(results["best_params_per_fold"]) == 3

    def test_learning_curve_analysis(self, classification_data):
        """Test learning curve generation."""
        X, y = classification_data
        model = LogisticRegression(max_iter=1000)
        fig, data = learning_curve_analysis(model, X, y, cv=3)
        
        assert fig is not None
        assert "train_mean" in data
        assert "test_mean" in data


class TestPitfallDemonstrations:
    """Tests for Section 5: Pitfall Demonstrations."""

    def test_demonstrate_overfitting(self, classification_data):
        """Test overfitting demonstration."""
        X, y = classification_data
        fig, df = demonstrate_overfitting(X, y)
        
        assert fig is not None
        assert "train_accuracy" in df.columns
        assert "test_accuracy" in df.columns
        assert "gap" in df.columns

    def test_demonstrate_data_leakage(self, classification_data):
        """Test data leakage demonstration."""
        X, y = classification_data
        results = demonstrate_data_leakage(X, y)
        
        assert "leaky_score" in results
        assert "proper_score" in results
        assert "difference" in results

    def test_handle_class_imbalance(self, classification_data):
        """Test class imbalance handling."""
        X, y = classification_data
        results = handle_class_imbalance(X, y, imbalance_ratio=0.2)
        
        assert "baseline" in results
        assert "class_weights" in results
        assert results["class_weights"]["recall"] >= results["baseline"]["recall"]


class TestUtilityFunctions:
    """Tests for additional utility functions."""

    def test_create_classification_pipeline(self, classification_data):
        """Test pipeline creation."""
        X, y = classification_data
        pipeline = create_classification_pipeline(LogisticRegression(max_iter=1000))
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)
        assert len(predictions) == len(y)

    def test_feature_importance_analysis(self, classification_data):
        """Test feature importance analysis."""
        X, y = classification_data
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        fig, df = feature_importance_analysis(model, feature_names)
        
        assert fig is not None
        assert len(df) == X.shape[1]
        assert df["importance"].sum() == pytest.approx(1.0, rel=0.01)
