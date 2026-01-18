# 13UNIT: Glossary

## Machine Learning for Researchers

---

## Core Terminology

### Accuracy
The proportion of correct predictions among all predictions made. Computed as (TP + TN) / (TP + TN + FP + FN) for binary classification. While intuitive, accuracy can be misleading for imbalanced datasets—a classifier predicting only the majority class achieves high accuracy despite providing no discriminative value.

### Bias (Statistical)
The systematic error introduced by approximating a complex real-world problem with a simplified model. High-bias models make strong assumptions about the data (e.g., linearity) and may underfit. Distinguished from variance, which measures sensitivity to training data fluctuations. The bias-variance trade-off represents a fundamental tension in model complexity selection.

### Classification
A supervised learning task where the model predicts discrete class labels from input features. Binary classification involves two classes (e.g., spam/legitimate); multiclass classification involves three or more mutually exclusive classes (e.g., species identification). Contrast with regression, which predicts continuous values.

### Clustering
An unsupervised learning technique that groups similar instances without predefined categories. The algorithm discovers structure in unlabelled data based on feature similarity. Common algorithms include k-means (centroid-based), hierarchical (tree-based) and DBSCAN (density-based).

### Confusion Matrix
A table summarising classification performance by showing counts of true positives (TP), true negatives (TN), false positives (FP) and false negatives (FN). Enables computation of various metrics and reveals specific error patterns. For multiclass problems, the matrix has dimensions equal to the number of classes.

### Cross-Validation
An evaluation technique that partitions data into multiple subsets, trains on some subsets and evaluates on others, then aggregates results. k-fold cross-validation divides data into k equal parts, using each part once for validation. Provides more reliable performance estimates than single train/test splits, especially with limited data.

### Data Leakage
A methodological error where information from the test set influences model training, producing overly optimistic performance estimates. Common causes include fitting preprocessing on the full dataset before splitting, using future information to predict past events, or including features derived from the target. Prevented by ensuring all data-dependent transformations occur within cross-validation folds.

### Dimensionality Reduction
Techniques that reduce the number of features whilst preserving essential information. Linear methods like Principal Component Analysis (PCA) find orthogonal directions of maximum variance. Nonlinear methods like t-SNE preserve local neighbourhood structure for visualisation. Reduces computational cost, mitigates curse of dimensionality and enables visualisation of high-dimensional data.

### Estimator
In scikit-learn, an object that implements the `fit()` method to learn from data. Estimators may be transformers (implementing `transform()` for data preprocessing), predictors (implementing `predict()` for generating predictions), or both. The consistent API enables algorithm interchangeability and pipeline construction.

### F1-Score
The harmonic mean of precision and recall, computed as 2 × (Precision × Recall) / (Precision + Recall). Provides a single metric balancing both types of classification errors. Ranges from 0 (worst) to 1 (best). Particularly useful when class distribution is uneven, as it considers both false positives and false negatives.

### Feature
An individual measurable property or characteristic used as input to a machine learning model. Features may be numerical (continuous or discrete) or categorical (nominal or ordinal). Feature engineering—creating informative features from raw data—often determines model success more than algorithm choice.

### Feature Importance
A measure of how much each feature contributes to model predictions. Tree-based models provide built-in importance scores based on reduction in impurity. Permutation importance measures performance degradation when feature values are shuffled. Understanding importance aids interpretation and guides feature selection.

### Generalisation
The ability of a trained model to perform well on previously unseen data. The fundamental goal of machine learning—models must extract patterns that transfer beyond training examples. Evaluated by measuring performance on held-out test data or through cross-validation.

### Hyperparameter
A model configuration setting determined before training, as opposed to parameters learned during training. Examples include regularisation strength (C in logistic regression), number of trees (n_estimators in random forest) and neighbourhood size (k in k-NN). Selected through cross-validation, grid search or randomised search.

### Inertia
In k-means clustering, the sum of squared distances from each point to its assigned cluster centroid. Lower inertia indicates tighter clusters. The elbow method plots inertia against cluster count to identify an optimal number of clusters where adding more clusters provides diminishing returns.

### Mean Squared Error (MSE)
A regression metric computed as the average of squared differences between predicted and actual values: (1/n) × Σ(y - ŷ)². Penalises large errors quadratically, making it sensitive to outliers. The square root (RMSE) provides error in the same units as the target variable.

### Model Selection
The process of choosing among candidate algorithms and hyperparameter configurations. Requires comparing generalisation performance estimates, typically via cross-validation. Nested cross-validation provides unbiased comparison when hyperparameters are tuned.

### Overfitting
When a model learns patterns specific to the training data that do not generalise to new data. Characterised by high training performance but poor test performance. Causes include excessive model complexity, insufficient training data, or too many features relative to samples. Mitigated through regularisation, cross-validation and early stopping.

### Pipeline
In scikit-learn, a sequence of data transformations followed by a final estimator. Pipelines ensure consistent preprocessing between training and prediction, prevent data leakage by keeping transformations within cross-validation folds, and simplify code by encapsulating the complete workflow.

### Precision
The proportion of positive predictions that are actually correct, computed as TP / (TP + FP). Answers: "Of all instances predicted positive, how many actually are?" Important when false positives are costly—e.g., in spam filtering, we want to avoid blocking legitimate emails.

### Principal Component Analysis (PCA)
A linear dimensionality reduction technique that finds orthogonal directions (principal components) of maximum variance in the data. The first component captures the most variance, subsequent components capture remaining variance orthogonally. Enables visualisation, noise reduction and feature extraction whilst preserving interpretable loadings.

### R² (Coefficient of Determination)
A regression metric measuring the proportion of target variance explained by the model, computed as 1 - (SS_residual / SS_total). Values range from negative infinity to 1, where 1 indicates perfect prediction, 0 indicates performance equivalent to predicting the mean, and negative values indicate worse-than-mean predictions.

### Recall (Sensitivity)
The proportion of actual positive instances correctly identified, computed as TP / (TP + FN). Answers: "Of all actually positive instances, how many did we detect?" Important when false negatives are costly—e.g., in disease screening, we want to avoid missing true cases.

### Regression
A supervised learning task where the model predicts continuous numerical values from input features. Examples include price prediction, temperature forecasting and survival time estimation. Evaluated using metrics like MSE, RMSE, MAE and R².

### Regularisation
Techniques that constrain model complexity to prevent overfitting. L1 regularisation (Lasso) adds the sum of absolute parameter values to the loss, encouraging sparsity. L2 regularisation (Ridge) adds the sum of squared parameters, shrinking coefficients toward zero. Controlled by a hyperparameter (alpha, C or lambda) balancing fit and complexity.

### ROC-AUC
The Area Under the Receiver Operating Characteristic curve, measuring classification performance across all possible decision thresholds. The ROC curve plots true positive rate against false positive rate. AUC ranges from 0.5 (random guessing) to 1.0 (perfect discrimination). Useful when optimal threshold is unknown or when comparing classifiers independent of threshold choice.

### Silhouette Score
A clustering evaluation metric measuring how similar each point is to its own cluster versus other clusters. For point i, s(i) = (b(i) - a(i)) / max(a(i), b(i)), where a(i) is mean intra-cluster distance and b(i) is mean nearest-cluster distance. Values range from -1 (poor clustering) to 1 (well-separated clusters).

### Stratification
A sampling technique that preserves class proportions when splitting data. Ensures that both training and test sets maintain the same class distribution as the original dataset. Essential for classification problems with imbalanced classes; implemented via the `stratify` parameter in train_test_split or StratifiedKFold cross-validation.

### Supervised Learning
A machine learning paradigm where models learn from labelled examples—training data includes both inputs (features) and corresponding outputs (targets). The model learns a mapping from inputs to outputs that generalises to new, unlabelled instances. Includes classification (discrete outputs) and regression (continuous outputs).

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
A nonlinear dimensionality reduction technique optimised for visualisation. Preserves local neighbourhood structure, making it effective for revealing cluster patterns in high-dimensional data. Unlike PCA, t-SNE cannot transform new data and does not preserve global distances—interpretation requires caution.

### Underfitting
When a model is too simple to capture underlying patterns in the data. Characterised by poor performance on both training and test data. Indicates insufficient model complexity, missing important features, or inadequate training. Addressed by using more complex models, adding features, or reducing regularisation.

### Unsupervised Learning
A machine learning paradigm where models discover patterns in unlabelled data—no target variable guides the learning process. Includes clustering (grouping similar instances) and dimensionality reduction (finding compact representations). Often used for exploratory analysis or as preprocessing for supervised tasks.

### Variance (Model)
The variability in model predictions across different training sets. High-variance models are sensitive to training data fluctuations and tend to overfit. Simple models have low variance but may have high bias. The bias-variance trade-off implies that reducing one typically increases the other.

---

## Cross-Reference Index

| Concept | Related Terms |
|---------|---------------|
| Model Quality | Accuracy, Precision, Recall, F1-Score, MSE, R² |
| Validation | Cross-Validation, Data Leakage, Generalisation |
| Complexity Control | Bias, Variance, Overfitting, Underfitting, Regularisation |
| Unsupervised | Clustering, Dimensionality Reduction, PCA, t-SNE, Silhouette |
| scikit-learn | Estimator, Pipeline, Hyperparameter |

---
