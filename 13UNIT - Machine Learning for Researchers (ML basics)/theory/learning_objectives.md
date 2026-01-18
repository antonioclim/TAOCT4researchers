# 13UNIT: Learning Objectives

## Machine Learning for Researchers

---

## Overview

This document specifies the measurable learning objectives for 13UNIT. Each objective is mapped to cognitive levels, assessment methods and curriculum alignment.

---

## Learning Objectives Matrix

| ID | Objective Statement | Cognitive Level | Verb | Assessment | Prerequisites |
|----|---------------------|-----------------|------|------------|---------------|
| LO1 | Distinguish supervised, unsupervised and reinforcement learning paradigms based on problem characteristics and data availability | Understand | Distinguish | Quiz Q1–Q3 | None |
| LO2 | Implement classification and regression pipelines using scikit-learn's Estimator API and Pipeline constructs | Apply | Implement | Lab 01 §1–§3 | 06UNIT |
| LO3 | Apply proper validation protocols including train/test split, k-fold cross-validation, stratification and nested cross-validation | Apply | Apply | Lab 01 §4, Exercises | 10UNIT |
| LO4 | Select and interpret evaluation metrics appropriate for classification (accuracy, precision, recall, F1, ROC-AUC) and regression (MSE, RMSE, R²) problems | Analyse | Select, Interpret | Lab 01 §2–§3, Quiz Q6–Q7 | 10UNIT |
| LO5 | Identify and mitigate common machine learning pitfalls: overfitting, underfitting, data leakage and class imbalance | Evaluate | Identify, Mitigate | Lab 01 §5, Hard exercises | LO1–LO4 |
| LO6 | Implement unsupervised learning techniques including k-means clustering, hierarchical clustering, DBSCAN, PCA and t-SNE visualisation | Apply | Implement | Lab 02 §1–§3 | 06UNIT |

---

## Detailed Objective Specifications

### LO1: Paradigm Distinction

**Full Statement**: Given a research problem description, correctly classify the appropriate machine learning paradigm (supervised classification, supervised regression, unsupervised clustering, unsupervised dimensionality reduction, reinforcement learning) and justify the selection based on data characteristics and desired outcomes.

**Enabling Knowledge**:
- Definition and characteristics of supervised learning (labelled training data)
- Definition and characteristics of unsupervised learning (pattern discovery in unlabelled data)
- Conceptual understanding of reinforcement learning (agent-environment interaction)
- Distinction between classification (discrete outputs) and regression (continuous outputs)
- Distinction between clustering and dimensionality reduction objectives

**Performance Criteria**:
- Correctly identifies paradigm for ≥80% of presented scenarios
- Provides valid justification referencing data characteristics
- Recognises edge cases where multiple paradigms might apply

**Assessment Evidence**:
- Quiz questions 1–3 (multiple choice with justification)
- Self-assessment scenario analysis

---

### LO2: Pipeline Implementation

**Full Statement**: Construct complete machine learning pipelines using scikit-learn that encompass data preprocessing (scaling, encoding), model training, prediction and evaluation, following the Estimator API conventions and Pipeline/ColumnTransformer constructs.

**Enabling Knowledge**:
- scikit-learn Estimator API: `fit()`, `predict()`, `transform()`, `fit_transform()`
- Pipeline construction for sequential transformations
- ColumnTransformer for heterogeneous feature processing
- Common preprocessing steps: StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
- Core algorithms: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, SVC, LinearRegression, Ridge, Lasso

**Performance Criteria**:
- Constructs syntactically correct pipelines
- Applies appropriate preprocessing to feature types
- Generates predictions on held-out data
- Produces well-structured, type-annotated code

**Assessment Evidence**:
- Lab 01 sections 1–3 (practical implementation)
- Easy exercises 1–2
- Medium exercise 2 (pipeline construction)

---

### LO3: Validation Methodology

**Full Statement**: Apply statistically sound validation protocols to estimate model generalisation performance, including train/test split with stratification, k-fold cross-validation, leave-one-out cross-validation and nested cross-validation for hyperparameter selection.

**Enabling Knowledge**:
- Purpose of validation: estimating generalisation error
- train_test_split with stratify parameter
- k-fold cross-validation: concept and implementation
- Stratified k-fold for imbalanced data
- Nested cross-validation: outer loop for evaluation, inner loop for hyperparameter selection
- Data leakage: definition, causes and prevention

**Performance Criteria**:
- Correctly implements train/test split with appropriate test size
- Applies stratification when class imbalance exists
- Implements cross-validation using cross_val_score or cross_validate
- Recognises and avoids data leakage scenarios
- Implements nested CV for unbiased hyperparameter selection

**Assessment Evidence**:
- Lab 01 section 4 (model selection)
- Easy exercise 1 (train/test split)
- Medium exercise 1 (cross-validation)
- Hard exercise 1 (nested CV)

---

### LO4: Metric Interpretation

**Full Statement**: Select evaluation metrics appropriate to problem characteristics (binary/multiclass classification, regression, imbalanced data) and interpret metric values to assess model quality, compare alternatives and diagnose issues.

**Enabling Knowledge**:
- Confusion matrix components: TP, TN, FP, FN
- Classification metrics: accuracy, precision, recall, F1-score, specificity
- Multi-class extensions: macro/micro/weighted averaging
- ROC curves and AUC interpretation
- Regression metrics: MSE, RMSE, MAE, R², adjusted R²
- Metric selection based on problem costs (precision vs recall trade-off)

**Performance Criteria**:
- Computes metrics correctly from predictions
- Selects metrics appropriate to problem characteristics
- Interprets metric values in domain context
- Compares models using appropriate criteria
- Identifies when accuracy is misleading (class imbalance)

**Assessment Evidence**:
- Lab 01 sections 2–3 (classification and regression evaluation)
- Easy exercise 3 (metrics basics)
- Quiz questions 6–7
- Medium exercise 3 (clustering evaluation)

---

### LO5: Pitfall Mitigation

**Full Statement**: Identify symptoms of common machine learning pathologies (overfitting, underfitting, data leakage, class imbalance) through diagnostic analysis and apply appropriate mitigation strategies.

**Enabling Knowledge**:
- Overfitting: high training accuracy, poor test accuracy; mitigation via regularisation, cross-validation, early stopping
- Underfitting: poor performance on both sets; mitigation via more complex models, feature engineering
- Bias-variance trade-off: conceptual understanding
- Data leakage: information from test set influencing training; common causes and prevention
- Class imbalance: definition, impact on metrics; mitigation via resampling (SMOTE, undersampling), class weights, threshold adjustment

**Performance Criteria**:
- Diagnoses overfitting from learning curves
- Identifies data leakage in pipeline code
- Detects class imbalance and assesses impact
- Applies appropriate mitigation strategies
- Evaluates effectiveness of interventions

**Assessment Evidence**:
- Lab 01 section 5 (pitfall demonstrations)
- Hard exercise 2 (imbalanced classification)
- Hard exercise 3 (complete pipeline with diagnostics)
- Quiz questions 8–10

---

### LO6: Unsupervised Implementation

**Full Statement**: Implement unsupervised learning algorithms for clustering (k-means, hierarchical, DBSCAN) and dimensionality reduction (PCA, t-SNE) with appropriate evaluation and visualisation.

**Enabling Knowledge**:
- k-means: algorithm, initialisation, elbow method for k selection
- Hierarchical clustering: agglomerative linkage methods, dendrograms
- DBSCAN: density-based clustering, eps and min_samples parameters
- Cluster evaluation: silhouette score, Davies-Bouldin index, inertia
- PCA: variance explanation, component interpretation, scree plots
- t-SNE: nonlinear visualisation, perplexity parameter, interpretation caveats

**Performance Criteria**:
- Implements clustering with appropriate algorithm selection
- Determines optimal cluster count using quantitative methods
- Implements PCA with variance retention criteria
- Creates effective visualisations of reduced data
- Interprets clustering and reduction results in domain context

**Assessment Evidence**:
- Lab 02 sections 1–3 (clustering and dimensionality reduction)
- Medium exercise 3 (clustering evaluation)
- Hard exercise 3 (complete pipeline including unsupervised components)

---

## Objective Interdependencies

```
LO1 (Paradigm Distinction)
 │
 ├──► LO2 (Pipeline Implementation)
 │     │
 │     ├──► LO4 (Metric Interpretation)
 │     │
 │     └──► LO6 (Unsupervised Implementation)
 │
 └──► LO3 (Validation Methodology)
       │
       └──► LO5 (Pitfall Mitigation)
             │
             └──► Synthesised competency (Hard exercise 3)
```

---

## Alignment with Course Outcomes

| UNIT Objective | Course-Level Outcome |
|----------------|---------------------|
| LO1 | CO3: Apply appropriate computational methods to research problems |
| LO2 | CO2: Implement well-structured, maintainable code |
| LO3 | CO4: Validate computational results rigorously |
| LO4 | CO5: Interpret and communicate computational findings |
| LO5 | CO4: Validate computational results rigorously |
| LO6 | CO3: Apply appropriate computational methods to research problems |

---

## Assessment Mapping

| Assessment Component | LO1 | LO2 | LO3 | LO4 | LO5 | LO6 |
|---------------------|-----|-----|-----|-----|-----|-----|
| Quiz Q1–Q3 | ✓ | | | | | |
| Quiz Q4–Q5 | | | ✓ | | | |
| Quiz Q6–Q7 | | | | ✓ | | |
| Quiz Q8–Q10 | | | | | ✓ | |
| Lab 01 §1 | | ✓ | | | | |
| Lab 01 §2 | | ✓ | | ✓ | | |
| Lab 01 §3 | | ✓ | | ✓ | | |
| Lab 01 §4 | | | ✓ | | | |
| Lab 01 §5 | | | | | ✓ | |
| Lab 02 §1–§2 | | | | | | ✓ |
| Lab 02 §3–§4 | | | | | | ✓ |
| Easy exercises | | ✓ | ✓ | ✓ | | |
| Medium exercises | | ✓ | ✓ | | | ✓ |
| Hard exercises | | | ✓ | | ✓ | |

---

## Mastery Indicators

### Emerging (Beginning)
- Recognises ML paradigm names but confuses characteristics
- Implements pipelines with syntax errors or logical flaws
- Uses validation incorrectly (e.g., fitting scaler on test data)
- Computes metrics but misinterprets values
- Cannot diagnose overfitting or leakage

### Developing (Intermediate)
- Correctly classifies most paradigm scenarios
- Implements working pipelines with minor issues
- Applies cross-validation correctly
- Interprets common metrics appropriately
- Recognises obvious overfitting symptoms

### Proficient (Target)
- Accurately distinguishes all paradigms with justification
- Constructs reliable pipelines following established methods
- Selects appropriate validation strategy for context
- Interprets full range of metrics and selects appropriately
- Diagnoses and mitigates common pitfalls

### Advanced (Exceeds)
- Identifies subtle paradigm edge cases
- Optimises pipeline efficiency and structure
- Implements nested CV for complex model selection
- Applies advanced metrics and threshold optimisation
- Prevents pitfalls proactively through design

---
