# 13UNIT: Homework Exercises

## Machine Learning for Researchers

---

## Overview

This homework consolidates machine learning competencies through progressive exercises spanning classification, regression, validation and clustering. Complete the exercises in sequence—later exercises build upon earlier implementations.

**Total Exercises**: 9 (3 Easy, 3 Medium, 3 Hard)

**Estimated Time**: 3–4 hours

**Submission Requirements**:
- All code must include type hints for function parameters and return values
- All functions must have Google-style docstrings
- Code must pass `ruff check` without errors
- Code must pass `mypy --strict` without errors

---

## Easy Exercises

### Exercise 1: Train/Test Split Fundamentals

**File**: `practice/easy_01_train_test_split.py`

**Learning Objective**: LO3 (Validation Methodology)

**Estimated Time**: 15 minutes

**Task**: Implement proper train/test splitting with stratification for a classification problem.

**Specifications**:

1. Load the breast cancer dataset from scikit-learn
2. Implement function `perform_stratified_split()` that:
   - Accepts features X, targets y, test_size (default 0.2) and random_state
   - Returns X_train, X_test, y_train, y_test as a tuple
   - Preserves class proportions in both sets using stratification
3. Implement function `verify_stratification()` that:
   - Accepts y_train and y_test arrays
   - Returns a dictionary with keys 'train_ratio' and 'test_ratio' containing the proportion of positive class in each set
   - Ratios should be approximately equal (within 0.02)

**Validation Criteria**:
- Stratification preserves class ratios within tolerance
- Random state ensures reproducibility
- Function signatures match specification
- Type hints present for all parameters and returns

---

### Exercise 2: Simple Classifier

**File**: `practice/easy_02_simple_classifier.py`

**Learning Objective**: LO2 (Pipeline Implementation)

**Estimated Time**: 20 minutes

**Task**: Train and evaluate a basic classification model.

**Specifications**:

1. Implement function `train_logistic_classifier()` that:
   - Accepts X_train, y_train and optional hyperparameters (C, max_iter)
   - Returns a fitted LogisticRegression model
   - Uses default solver 'lbfgs' and specified or default max_iter=1000

2. Implement function `evaluate_classifier()` that:
   - Accepts a fitted model, X_test and y_test
   - Returns a dictionary containing:
     - 'accuracy': float
     - 'predictions': numpy array of predictions
     - 'probabilities': numpy array of probability predictions (shape n_samples × n_classes)

3. Implement function `classification_pipeline()` that:
   - Combines loading data, splitting, training and evaluation
   - Returns the evaluation metrics dictionary

**Validation Criteria**:
- Model achieves ≥90% accuracy on breast cancer dataset
- Probabilities sum to 1.0 for each sample
- Pipeline produces consistent results with fixed random state

---

### Exercise 3: Metrics Basics

**File**: `practice/easy_03_metrics_basics.py`

**Learning Objective**: LO4 (Metric Interpretation)

**Estimated Time**: 15 minutes

**Task**: Compute and interpret classification metrics from predictions.

**Specifications**:

1. Implement function `compute_confusion_matrix()` that:
   - Accepts y_true and y_pred arrays
   - Returns a dictionary with keys 'TP', 'TN', 'FP', 'FN' containing integer counts

2. Implement function `compute_classification_metrics()` that:
   - Accepts y_true and y_pred arrays
   - Returns a dictionary containing:
     - 'accuracy': float
     - 'precision': float
     - 'recall': float
     - 'f1_score': float
   - Handles edge cases (e.g., division by zero) by returning 0.0

3. Implement function `interpret_metrics()` that:
   - Accepts a metrics dictionary
   - Returns a string describing model performance in plain language
   - Mentions whether precision or recall is higher and what that implies

**Validation Criteria**:
- Metrics match scikit-learn's implementations
- Edge cases handled gracefully
- Interpretation is meaningful and accurate

---

## Medium Exercises

### Exercise 4: Cross-Validation

**File**: `practice/medium_01_cross_validation.py`

**Learning Objective**: LO3 (Validation Methodology)

**Estimated Time**: 25 minutes

**Task**: Implement and compare different cross-validation strategies.

**Specifications**:

1. Implement function `kfold_cross_validation()` that:
   - Accepts model, X, y, n_splits (default 5) and random_state
   - Uses StratifiedKFold for classification
   - Returns a dictionary containing:
     - 'scores': list of accuracy scores per fold
     - 'mean': mean accuracy
     - 'std': standard deviation
     - 'fold_sizes': list of test set sizes per fold

2. Implement function `compare_cv_strategies()` that:
   - Accepts model, X, y
   - Compares 3-fold, 5-fold and 10-fold cross-validation
   - Returns a dictionary mapping fold count to mean score

3. Implement function `analyse_cv_stability()` that:
   - Accepts model, X, y and n_repeats (default 10)
   - Runs 5-fold CV multiple times with different random states
   - Returns mean and std of the mean CV scores across repeats
   - Quantifies how stable the CV estimate is

**Validation Criteria**:
- Stratification preserves class ratios
- More folds generally yield more stable estimates
- Repeated CV demonstrates estimate variability

---

### Exercise 5: Pipeline Construction

**File**: `practice/medium_02_pipeline_construction.py`

**Learning Objective**: LO2 (Pipeline Implementation)

**Estimated Time**: 30 minutes

**Task**: Build a complete preprocessing and classification pipeline.

**Specifications**:

1. Implement function `create_preprocessing_pipeline()` that:
   - Accepts lists of numerical_columns and categorical_columns
   - Returns a ColumnTransformer that:
     - Applies StandardScaler to numerical columns
     - Applies OneHotEncoder (handle_unknown='ignore') to categorical columns

2. Implement function `create_full_pipeline()` that:
   - Accepts a preprocessor and a classifier
   - Returns a Pipeline combining preprocessing and classification

3. Implement function `evaluate_pipeline_with_cv()` that:
   - Accepts pipeline, X, y and cv (default 5)
   - Returns cross-validation scores using the complete pipeline
   - Demonstrates that preprocessing is correctly integrated into CV

4. Apply to the Adult Income dataset (fetch from OpenML or create synthetic):
   - Features include numerical (age, hours_per_week) and categorical (workclass, education)
   - Target is income bracket (>50K or ≤50K)

**Validation Criteria**:
- Pipeline handles mixed feature types correctly
- Preprocessing fitted only on training folds within CV
- No data leakage warnings or errors

---

### Exercise 6: Clustering Evaluation

**File**: `practice/medium_03_clustering_evaluation.py`

**Learning Objective**: LO6 (Unsupervised Implementation)

**Estimated Time**: 25 minutes

**Task**: Implement clustering with systematic evaluation of cluster quality.

**Specifications**:

1. Implement function `find_optimal_k()` that:
   - Accepts X, k_range (default range(2, 11)) and random_state
   - Runs k-means for each k value
   - Returns a dictionary mapping k to:
     - 'inertia': within-cluster sum of squares
     - 'silhouette': silhouette score
   - Identifies optimal k using silhouette score

2. Implement function `compare_clustering_algorithms()` that:
   - Accepts X and n_clusters
   - Compares k-means, agglomerative clustering (ward linkage) and DBSCAN
   - For DBSCAN, use eps=0.5 and min_samples=5
   - Returns comparison dictionary with silhouette scores (where applicable)

3. Implement function `visualise_clusters()` that:
   - Accepts X and cluster labels
   - Uses PCA to reduce to 2D if necessary
   - Creates a scatter plot coloured by cluster
   - Returns the matplotlib figure

**Validation Criteria**:
- Silhouette scores computed correctly
- DBSCAN handles noise points appropriately
- Visualisation clearly shows cluster structure

---

## Hard Exercises

### Exercise 7: Nested Cross-Validation

**File**: `practice/hard_01_nested_cv.py`

**Learning Objective**: LO3, LO5 (Validation, Pitfall Mitigation)

**Estimated Time**: 35 minutes

**Task**: Implement nested cross-validation for unbiased hyperparameter selection.

**Specifications**:

1. Implement function `standard_cv_with_tuning()` that:
   - Accepts X, y, model and param_grid
   - Performs GridSearchCV with 5-fold CV
   - Returns best score and best parameters
   - NOTE: This provides biased performance estimate

2. Implement function `nested_cross_validation()` that:
   - Accepts X, y, model and param_grid
   - Outer loop: 5-fold CV for performance estimation
   - Inner loop: 3-fold CV for hyperparameter selection
   - Returns a dictionary containing:
     - 'outer_scores': list of scores from outer folds
     - 'mean_score': unbiased performance estimate
     - 'best_params_per_fold': list of best parameters found in each outer fold

3. Implement function `compare_cv_approaches()` that:
   - Compares standard CV with tuning vs nested CV
   - Demonstrates that standard CV overestimates performance
   - Returns both estimates for comparison

4. Apply to a dataset where the difference is observable (e.g., small dataset with many hyperparameters)

**Validation Criteria**:
- Nested CV typically yields lower (more realistic) scores than standard CV with tuning
- Each outer fold has independently selected hyperparameters
- Implementation correctly separates inner and outer loops

---

### Exercise 8: Imbalanced Classification

**File**: `practice/hard_02_imbalanced_classification.py`

**Learning Objective**: LO5 (Pitfall Mitigation)

**Estimated Time**: 35 minutes

**Task**: Handle class imbalance through multiple strategies.

**Specifications**:

1. Implement function `create_imbalanced_dataset()` that:
   - Creates a synthetic dataset with 95% majority class, 5% minority class
   - Returns X, y with specified imbalance ratio

2. Implement function `baseline_classifier()` that:
   - Trains without any imbalance handling
   - Returns accuracy, precision, recall, F1 for minority class

3. Implement function `class_weight_classifier()` that:
   - Uses class_weight='balanced' in LogisticRegression
   - Returns the same metrics

4. Implement function `resampled_classifier()` that:
   - Applies SMOTE (from imblearn) to oversample minority class
   - Trains on resampled data
   - Returns metrics
   - Note: Install imblearn or implement simple random oversampling if unavailable

5. Implement function `threshold_adjusted_classifier()` that:
   - Trains standard classifier
   - Adjusts decision threshold to optimise F1 for minority class
   - Returns metrics and optimal threshold

6. Implement function `compare_imbalance_strategies()` that:
   - Runs all four approaches
   - Returns comparison dictionary showing metric improvements

**Validation Criteria**:
- Baseline shows high accuracy but poor minority class metrics
- Each strategy improves minority class recall
- Trade-offs between strategies are documented

---

### Exercise 9: Complete ML Pipeline

**File**: `practice/hard_03_complete_ml_pipeline.py`

**Learning Objective**: All LOs (Synthesis)

**Estimated Time**: 45 minutes

**Task**: Build a production-quality machine learning pipeline with all established methods.

**Specifications**:

1. Implement class `MLPipeline` with methods:
   - `__init__(self, random_state: int = 42)`: initialise with reproducibility
   - `load_data(self, dataset_name: str)`: load specified sklearn dataset
   - `explore_data(self) -> dict`: return summary statistics and class distribution
   - `preprocess(self, numerical_cols: list, categorical_cols: list)`: create preprocessor
   - `train(self, model, param_grid: dict | None = None)`: train with optional hyperparameter search
   - `evaluate(self) -> dict`: return comprehensive evaluation metrics
   - `diagnose(self) -> dict`: return overfitting diagnostics (train vs test scores)
   - `get_feature_importance(self) -> dict`: return feature importances if available

2. Implement function `run_complete_experiment()` that:
   - Uses MLPipeline class
   - Loads breast cancer dataset
   - Explores data
   - Creates preprocessing pipeline
   - Compares multiple classifiers (LogisticRegression, RandomForest, SVC)
   - Uses nested CV for fair comparison
   - Reports comprehensive results
   - Diagnoses any overfitting
   - Returns final report as structured dictionary

3. Document all decisions and trade-offs in docstrings

**Validation Criteria**:
- Code follows all style guidelines (type hints, docstrings)
- No data leakage in any step
- Multiple models compared fairly
- Diagnostics correctly identify any issues
- Report is comprehensive and interpretable

---

## Submission Checklist

Before submitting, verify:

- [ ] All 9 exercise files are present in `practice/` directory
- [ ] Each file runs without errors
- [ ] All functions have type hints
- [ ] All functions have Google-style docstrings
- [ ] `ruff check practice/` returns no errors
- [ ] `mypy practice/ --strict` returns no errors
- [ ] Test coverage ≥80% for implemented functions
- [ ] Random states set for reproducibility

---

## Grading Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Functions produce expected outputs |
| Code Quality | 25% | Type hints, docstrings, style compliance |
| Methodology | 25% | Appropriate validation, metric selection, pitfall handling |
| Documentation | 10% | Clear reasoning, interpretation of results |

---

## Resources

- scikit-learn documentation: https://scikit-learn.org/stable/
- Lab solutions for reference implementations
- Office hours for clarification questions

---
