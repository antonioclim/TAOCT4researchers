# 13UNIT: Lecture Notes

## Machine Learning for Researchers

---

## Table of Contents

1. [Introduction to Machine Learning](#1-introduction-to-machine-learning)
2. [The Learning Paradigms](#2-the-learning-paradigms)
3. [The scikit-learn Ecosystem](#3-the-scikit-learn-ecosystem)
4. [Validation Methodology](#4-validation-methodology)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Common Pitfalls and Mitigations](#6-common-pitfalls-and-mitigations)
7. [Unsupervised Learning Foundations](#7-unsupervised-learning-foundations)
8. [Research Applications](#8-research-applications)

---

## 1. Introduction to Machine Learning

### 1.1 What is Machine Learning?

Machine learning represents a fundamental shift in how we construct computational models. Traditional programming requires explicit specification of rules: *if* this condition, *then* that action. Machine learning inverts this relationship—we provide examples of desired behaviour and the algorithm infers the underlying rules. This distinction carries profound implications for research practice.

Consider the task of classifying email as spam or legitimate correspondence. A rule-based approach would require enumerating all possible spam indicators: specific words, suspicious links, unusual sender patterns. Such rules become brittle as spammers adapt. Machine learning approaches instead present thousands of labelled examples—emails known to be spam or legitimate—and algorithms discover discriminative patterns automatically.

This example illustrates the core machine learning workflow:
1. **Collect** labelled examples (training data)
2. **Train** an algorithm to recognise patterns
3. **Evaluate** performance on held-out data
4. **Deploy** the model for prediction on new instances

The fourth step reveals a fundamental requirement: models must generalise beyond their training data. A model that merely memorises training examples provides no value—we already know those labels. The challenge lies in learning patterns that transfer to unseen instances.

### 1.2 Models as Purposeful Abstractions

Every machine learning model embodies decisions about what matters. A model predicting house prices might consider square footage, location and number of bedrooms whilst ignoring paint colour or owner's name. These choices constitute abstraction—retaining relevant features whilst discarding irrelevant details.

Effective abstraction requires domain knowledge. A geneticist building a disease prediction model knows which biomarkers correlate with outcomes. A economist modelling market behaviour understands which indicators drive prices. Technical machine learning expertise becomes most powerful when combined with substantive domain understanding.

This interplay between computational methods and domain knowledge characterises effective research applications. Machine learning provides the pattern recognition capability; domain expertise guides feature selection, model interpretation and result validation.

### 1.3 The Learning Problem Formally

Machine learning problems share a common structure. We observe pairs of inputs (features) and outputs (targets):

- **Features** (X): measurable attributes of instances (age, income, gene expression levels)
- **Targets** (y): outcomes we wish to predict (diagnosis, purchase decision, survival time)

The learning algorithm seeks a function f(X) that approximates y. Quality is measured by how closely predictions ŷ = f(X) match actual outcomes y on data not used during training.

This formulation reveals why validation is essential: the function f is chosen to minimise error on training data, but we care about performance on unseen data. Without rigorous validation, we cannot distinguish models that have learned generalisable patterns from those that have merely memorised training examples.

---

## 2. The Learning Paradigms

### 2.1 Supervised Learning

Supervised learning addresses problems where labelled examples are available. The term "supervised" references the presence of a teacher providing correct answers during training. Given sufficient examples of input-output pairs, algorithms learn mappings that generalise to new inputs.

**Classification** predicts discrete categories:
- Disease diagnosis (disease present/absent)
- Species identification (species A, B, or C)
- Sentiment analysis (positive, negative, neutral)

**Regression** predicts continuous quantities:
- House price estimation (monetary value)
- Temperature forecasting (degrees)
- Drug dosage optimisation (milligrams)

The boundary between classification and regression sometimes blurs. Predicting "high" vs "low" risk converts a continuous quantity into discrete categories. Predicting probability of class membership produces a continuous value interpretable as classification confidence.

### 2.2 Unsupervised Learning

Unsupervised learning operates without labelled examples. The algorithm must discover structure in the data without guidance about what patterns are interesting or relevant.

**Clustering** groups similar instances:
- Customer segmentation (identifying market segments)
- Gene expression profiling (co-expressed gene groups)
- Document organisation (topic clusters)

**Dimensionality Reduction** finds compact representations:
- Visualisation (projecting high-dimensional data to 2D)
- Feature extraction (learning informative combinations)
- Noise reduction (removing uninformative variation)

Unsupervised learning often serves as a precursor to supervised analysis. Clustering may reveal subgroups that merit separate modelling. Dimensionality reduction may extract features that improve classification performance.

### 2.3 Reinforcement Learning (Conceptual)

Reinforcement learning addresses sequential decision-making under uncertainty. An agent interacts with an environment, receives feedback (rewards or penalties) and learns behaviour that maximises cumulative reward.

While beyond this unit's practical scope, researchers should recognise reinforcement learning's applicability to problems involving:
- Sequential actions with delayed consequences
- Exploration-exploitation trade-offs
- Adaptive systems that improve through experience

Examples include robotics control, game playing, treatment regime optimisation and recommendation systems that adapt to user feedback.

### 2.4 Paradigm Selection

Given a research problem, which paradigm applies?

**Use supervised learning when:**
- Labelled examples are available
- The goal is prediction or classification
- Ground truth exists for evaluation

**Use unsupervised learning when:**
- No labels are available
- The goal is pattern discovery or exploration
- Structure in data is unknown

**Consider reinforcement learning when:**
- Decisions must be made sequentially
- Feedback arrives delayed
- The environment responds to actions

Many research applications combine paradigms. Unsupervised clustering might identify subgroups that inform supervised classification. Semi-supervised approaches use small labelled datasets augmented by large unlabelled collections.

---

## 3. The scikit-learn Ecosystem

### 3.1 The Estimator API

scikit-learn provides a unified interface for machine learning algorithms. Every algorithm follows the Estimator pattern:

```python
# All estimators follow this pattern
model = SomeAlgorithm(hyperparameters)
model.fit(X_train, y_train)       # Learn from training data
predictions = model.predict(X_test)  # Generate predictions
```

This consistency enables algorithm comparison with minimal code changes. Replacing `LogisticRegression` with `RandomForestClassifier` requires changing only the instantiation line—the `fit()` and `predict()` calls remain identical.

The API distinguishes three method types:
- **fit()**: Learn parameters from training data
- **predict()**: Generate predictions for new data
- **transform()**: Transform data (for preprocessing)

Some estimators support **fit_transform()**, combining fitting and transformation in a single call—useful for preprocessing where the transformation parameters are learned from data.

### 3.2 Pipeline Construction

Machine learning workflows typically involve multiple preprocessing steps before model training. scikit-learn's Pipeline constructs chain these steps:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Pipelines ensure consistent preprocessing between training and prediction. When `fit()` is called on the pipeline, each step fits to the (transformed) data in sequence. When `predict()` is called, each transformation applies before the final prediction.

This encapsulation prevents a common error: fitting preprocessing (e.g., scaling) on the full dataset before splitting. Such "data leakage" contaminates test set evaluation, producing optimistic performance estimates.

### 3.3 Handling Heterogeneous Features

Real datasets often contain mixed feature types: numerical measurements and categorical variables. The ColumnTransformer applies different transformations to different columns:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(), categorical_columns)
])
```

This preprocessor scales numerical features and one-hot encodes categorical features, then concatenates the results. Combined with Pipeline, it creates complete workflows from raw data to predictions.

---

## 4. Validation Methodology

### 4.1 The Generalisation Problem

A model's value lies in its predictions on new data. Training performance tells us only that the model can fit the data it has seen—not whether it will perform well on unseen instances. This distinction motivates rigorous validation protocols.

The fundamental strategy is hold-out validation: reserve some data for evaluation that is never used during training. The model cannot have memorised test examples because it never observed them. Test performance therefore estimates generalisation performance.

### 4.2 Train/Test Split

The simplest validation approach splits data into training and test sets:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Key considerations:
- **Test size**: Typically 20–30% of data
- **Stratification**: Preserves class proportions in both sets
- **Random state**: Ensures reproducibility

A single split provides one performance estimate. With limited data, this estimate may be unreliable—a different split might yield substantially different results.

### 4.3 Cross-Validation

Cross-validation addresses single-split instability by averaging across multiple splits. k-fold cross-validation divides data into k subsets (folds), trains on k-1 folds and evaluates on the held-out fold, rotating through all k possibilities:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
mean_score = scores.mean()
std_score = scores.std()
```

The mean across folds provides a more stable performance estimate. The standard deviation indicates estimate variability—high variability suggests unreliable estimates, potentially due to limited data or heterogeneous subgroups.

### 4.4 Stratified Cross-Validation

For classification problems, StratifiedKFold ensures each fold maintains the original class distribution:

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)
```

Stratification is particularly important with imbalanced classes. Without it, some folds might contain very few minority class examples, producing unreliable estimates.

### 4.5 Nested Cross-Validation

When hyperparameter tuning is involved, standard cross-validation produces biased estimates. The hyperparameters are chosen to maximise cross-validation performance, which inflates the estimate of generalisation performance.

Nested cross-validation addresses this bias with two loops:
- **Outer loop**: Evaluates final model performance
- **Inner loop**: Selects hyperparameters

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

# Inner loop: hyperparameter selection
inner_cv = StratifiedKFold(n_splits=3)
grid_search = GridSearchCV(model, param_grid, cv=inner_cv)

# Outer loop: performance estimation
outer_cv = StratifiedKFold(n_splits=5)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)
```

The outer loop provides unbiased performance estimates; the inner loop selects hyperparameters independently for each outer fold.

### 4.6 Data Leakage

Data leakage occurs when information from the test set influences model training. Common causes include:

- Fitting preprocessing on full data before splitting
- Using future information to predict past events
- Including features derived from the target variable
- Duplicated instances appearing in both train and test sets

Leakage produces overly optimistic performance estimates—models appear better than they will perform in deployment. Prevention requires careful pipeline construction ensuring all data-dependent steps fit only on training data.

---

## 5. Evaluation Metrics

### 5.1 Classification Metrics

**Confusion Matrix** tabulates prediction outcomes:

|                 | Predicted Positive | Predicted Negative |
|-----------------|-------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

From these counts, various metrics derive:

**Accuracy** = (TP + TN) / (TP + TN + FP + FN)
- Proportion of correct predictions
- Misleading with imbalanced classes

**Precision** = TP / (TP + FP)
- Of positive predictions, how many are correct?
- Important when false positives are costly

**Recall** (Sensitivity) = TP / (TP + FN)
- Of actual positives, how many are detected?
- Important when false negatives are costly

**F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
- Harmonic mean balancing precision and recall
- Useful when both error types matter

**ROC-AUC** measures discrimination across all classification thresholds. An AUC of 0.5 indicates random guessing; 1.0 indicates perfect separation.

### 5.2 Regression Metrics

**Mean Squared Error (MSE)** = (1/n) × Σ(y - ŷ)²
- Penalises large errors heavily
- Units are squared target units

**Root Mean Squared Error (RMSE)** = √MSE
- Same units as target variable
- More interpretable than MSE

**Mean Absolute Error (MAE)** = (1/n) × Σ|y - ŷ|
- Robust to outliers
- Easier to interpret

**R² (Coefficient of Determination)** = 1 - (SS_res / SS_tot)
- Proportion of variance explained
- 1.0 is perfect; 0.0 equals predicting the mean; negative values indicate worse-than-mean predictions

### 5.3 Metric Selection

Metric choice should reflect problem costs:

| Scenario | Preferred Metric | Rationale |
|----------|------------------|-----------|
| Balanced classes, equal error costs | Accuracy | Simple, intuitive |
| Imbalanced classes | F1, Precision, Recall | Accuracy misleads |
| Medical screening (don't miss disease) | Recall | Minimise false negatives |
| Spam filtering (don't block legitimate) | Precision | Minimise false positives |
| Ranking matters | ROC-AUC | Threshold-independent |

---

## 6. Common Pitfalls and Mitigations

### 6.1 Overfitting

**Symptoms**: High training accuracy, poor test accuracy. Learning curves show diverging train/test performance as model complexity increases.

**Causes**: Model too complex for available data. Insufficient regularisation. Too many features relative to samples.

**Mitigations**:
- Regularisation (L1, L2 penalties)
- Cross-validation for hyperparameter selection
- Feature selection to reduce dimensionality
- Ensemble methods that average predictions
- Early stopping during training

### 6.2 Underfitting

**Symptoms**: Poor performance on both training and test data. Learning curves plateau at low accuracy.

**Causes**: Model too simple for problem complexity. Important features missing. Insufficient training.

**Mitigations**:
- More complex models
- Feature engineering
- Longer training (more iterations)
- Reduced regularisation

### 6.3 Class Imbalance

**Symptoms**: High accuracy but poor minority class performance. Confusion matrix reveals most minority cases misclassified.

**Causes**: Unequal class frequencies in training data. Model optimises for majority class.

**Mitigations**:
- Resampling: oversample minority (SMOTE) or undersample majority
- Class weights: penalise minority class errors more heavily
- Threshold adjustment: lower decision threshold for minority class
- Appropriate metrics: use F1, precision/recall rather than accuracy

### 6.4 Data Leakage

**Symptoms**: Unusually high performance that fails in deployment. Model performance degrades on truly new data.

**Causes**: Test data information available during training. Preprocessing fitted on full dataset.

**Mitigations**:
- Use Pipelines for all preprocessing
- Split data before any exploration
- Verify no target-derived features
- Check for duplicate instances

---

## 7. Unsupervised Learning Foundations

### 7.1 Clustering

Clustering groups similar instances without predefined categories. The algorithm must infer group structure from data alone.

**k-Means** partitions data into k clusters, minimising within-cluster variance. The algorithm iteratively assigns points to nearest centroid and updates centroids. Limitations include sensitivity to initialisation and assumption of spherical clusters.

**Hierarchical Clustering** builds a tree of clusters through iterative merging (agglomerative) or splitting (divisive). The dendrogram visualisation reveals cluster structure at multiple granularities.

**DBSCAN** identifies dense regions separated by sparse areas. Unlike k-means, it discovers clusters of arbitrary shape and identifies outliers as noise. The eps (neighbourhood radius) and min_samples parameters control density definition.

### 7.2 Cluster Evaluation

Without ground truth labels, cluster quality assessment is challenging. Common approaches include:

**Silhouette Score** measures how similar points are to their own cluster versus other clusters. Values range from -1 to 1; higher indicates better-defined clusters.

**Elbow Method** plots within-cluster variance against cluster count. The "elbow" where adding clusters provides diminishing returns suggests optimal k.

### 7.3 Dimensionality Reduction

High-dimensional data creates challenges: visualisation is impossible, distances become less meaningful and many algorithms perform poorly.

**Principal Component Analysis (PCA)** finds orthogonal directions of maximum variance. The first principal component captures the most variance; subsequent components capture remaining variance orthogonally. PCA is linear, fast and provides interpretable loadings.

**t-SNE** (t-Distributed Stochastic Neighbor Embedding) preserves local neighbourhood structure in low dimensions. It excels at visualisation but does not preserve global distances and cannot transform new data.

---

## 8. Research Applications

### 8.1 Domain-Specific Considerations

Machine learning applications require domain adaptation:

**Biology**: Gene expression data involves thousands of features with few samples (p >> n). Regularisation and feature selection are essential. Domain knowledge guides feature engineering.

**Medicine**: Interpretability matters for clinical acceptance. Class imbalance is common (most patients don't have rare diseases). Cost asymmetry requires careful metric selection.

**Social Science**: Survey data includes missing values and categorical features. Text analysis requires preprocessing pipelines. Results require substantive interpretation.

**Economics**: Time series require temporal validation (train on past, test on future). Regime changes may invalidate historical patterns.

### 8.2 Reproducibility Requirements

Research applications demand reproducibility. Essential practices include:

- Setting random seeds for all stochastic operations
- Version control for code and data
- Documenting preprocessing decisions
- Archiving trained models and predictions
- Reporting performance with uncertainty estimates

These practices, introduced in 07UNIT, apply directly to machine learning workflows.

### 8.3 From Prediction to Understanding

Machine learning often prioritises predictive accuracy over interpretability. Research applications frequently require both. Strategies for understanding learned models include:

- Feature importance analysis
- Partial dependence plots
- SHAP (SHapley Additive exPlanations) values
- Simpler, interpretable models (decision trees, logistic regression)
- Post-hoc explanations for complex models

The trade-off between accuracy and interpretability shapes model selection for research applications.

---

## Summary

This unit has introduced machine learning as a systematic methodology for extracting patterns from data. Key concepts include:

1. **Paradigms**: Supervised, unsupervised and reinforcement learning address different problem types
2. **Validation**: Proper protocols prevent overfitting and provide reliable performance estimates
3. **Metrics**: Selection depends on problem characteristics and error costs
4. **Pitfalls**: Overfitting, leakage and imbalance require specific mitigations
5. **Unsupervised**: Clustering and dimensionality reduction enable pattern discovery

The laboratory exercises that follow will instantiate these concepts through practical implementation using scikit-learn.

---
