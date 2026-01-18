# 13UNIT: Self-Assessment Checklist

## Machine Learning for Researchers

---

## Instructions

Complete this self-assessment after finishing all laboratory exercises and before submitting homework. For each item, honestly evaluate your current competency level.

**Rating Scale**:
- ✅ **Confident**: I can do this independently and explain it to others
- 🔶 **Developing**: I can do this with reference materials
- ❌ **Need Review**: I struggle with this concept

---

## Learning Objective 1: Paradigm Distinction

*Can you distinguish supervised, unsupervised and reinforcement learning paradigms?*

| Competency | Self-Rating | Notes |
|------------|-------------|-------|
| Define supervised learning and identify when it applies | ⬜ | |
| Distinguish classification from regression problems | ⬜ | |
| Define unsupervised learning and its applications | ⬜ | |
| Explain when clustering vs dimensionality reduction applies | ⬜ | |
| Describe reinforcement learning conceptually | ⬜ | |
| Given a research problem, select the appropriate paradigm | ⬜ | |

**Reflection**: If you marked any items ❌, review lecture notes Section 2.

---

## Learning Objective 2: Pipeline Implementation

*Can you implement ML pipelines using scikit-learn?*

| Competency | Self-Rating | Notes |
|------------|-------------|-------|
| Use the Estimator API (fit, predict, transform) | ⬜ | |
| Create Pipeline objects for sequential transformations | ⬜ | |
| Use ColumnTransformer for heterogeneous features | ⬜ | |
| Apply StandardScaler, MinMaxScaler for numerical features | ⬜ | |
| Apply OneHotEncoder for categorical features | ⬜ | |
| Train classifiers: LogisticRegression, RandomForest, SVC | ⬜ | |
| Train regressors: LinearRegression, Ridge, Lasso | ⬜ | |
| Generate predictions and probability estimates | ⬜ | |

**Reflection**: If you marked any items ❌, redo Lab 01 Sections 1–3.

---

## Learning Objective 3: Validation Methodology

*Can you apply proper validation protocols?*

| Competency | Self-Rating | Notes |
|------------|-------------|-------|
| Perform train/test split with appropriate test size | ⬜ | |
| Apply stratification for classification problems | ⬜ | |
| Implement k-fold cross-validation | ⬜ | |
| Use StratifiedKFold for imbalanced data | ⬜ | |
| Interpret mean and std of CV scores | ⬜ | |
| Explain why nested CV provides unbiased estimates | ⬜ | |
| Implement nested cross-validation | ⬜ | |
| Identify data leakage scenarios | ⬜ | |
| Prevent leakage using Pipelines | ⬜ | |

**Reflection**: If you marked any items ❌, review Lab 01 Section 4 and lecture notes Section 4.

---

## Learning Objective 4: Metric Interpretation

*Can you select and interpret appropriate evaluation metrics?*

| Competency | Self-Rating | Notes |
|------------|-------------|-------|
| Construct and interpret confusion matrices | ⬜ | |
| Calculate accuracy from confusion matrix | ⬜ | |
| Calculate precision and explain its meaning | ⬜ | |
| Calculate recall and explain its meaning | ⬜ | |
| Calculate F1-score and when to use it | ⬜ | |
| Interpret ROC curves and AUC | ⬜ | |
| Calculate MSE, RMSE for regression | ⬜ | |
| Interpret R² and its limitations | ⬜ | |
| Select metrics based on problem costs | ⬜ | |
| Explain why accuracy is misleading for imbalanced data | ⬜ | |

**Reflection**: If you marked any items ❌, review lecture notes Section 5 and complete easy exercise 3.

---

## Learning Objective 5: Pitfall Mitigation

*Can you identify and address common ML pitfalls?*

| Competency | Self-Rating | Notes |
|------------|-------------|-------|
| Recognise overfitting from train/test performance gap | ⬜ | |
| Interpret learning curves for overfitting diagnosis | ⬜ | |
| Apply regularisation to reduce overfitting | ⬜ | |
| Recognise underfitting symptoms | ⬜ | |
| Explain bias-variance trade-off conceptually | ⬜ | |
| Identify data leakage in code | ⬜ | |
| Detect class imbalance from data exploration | ⬜ | |
| Apply class weights to address imbalance | ⬜ | |
| Use resampling techniques (SMOTE, undersampling) | ⬜ | |
| Adjust classification threshold for imbalanced data | ⬜ | |

**Reflection**: If you marked any items ❌, redo Lab 01 Section 5 and hard exercise 2.

---

## Learning Objective 6: Unsupervised Implementation

*Can you implement clustering and dimensionality reduction?*

| Competency | Self-Rating | Notes |
|------------|-------------|-------|
| Implement k-means clustering | ⬜ | |
| Use elbow method to select k | ⬜ | |
| Implement hierarchical clustering | ⬜ | |
| Interpret dendrograms | ⬜ | |
| Implement DBSCAN for density-based clustering | ⬜ | |
| Calculate and interpret silhouette scores | ⬜ | |
| Implement PCA for dimensionality reduction | ⬜ | |
| Interpret explained variance ratios | ⬜ | |
| Create scree plots | ⬜ | |
| Implement t-SNE for visualisation | ⬜ | |
| Understand t-SNE limitations | ⬜ | |

**Reflection**: If you marked any items ❌, redo Lab 02 and medium exercise 3.

---

## Code Quality Self-Check

| Requirement | Verified | Notes |
|-------------|----------|-------|
| All functions have type hints | ⬜ | |
| All functions have Google-style docstrings | ⬜ | |
| `ruff check` passes with no errors | ⬜ | |
| `mypy --strict` passes with no errors | ⬜ | |
| Random states set for reproducibility | ⬜ | |
| No hardcoded file paths | ⬜ | |
| No magic numbers (constants named) | ⬜ | |
| Logging used instead of print statements | ⬜ | |

---

## Conceptual Understanding Verification

Answer these questions without reference materials:

1. **What is the difference between precision and recall?**

   Your answer: _________________________________________________

2. **Why might a model with 99% accuracy still be useless?**

   Your answer: _________________________________________________

3. **What is data leakage and how do you prevent it?**

   Your answer: _________________________________________________

4. **When would you use nested cross-validation instead of standard CV?**

   Your answer: _________________________________________________

5. **How does PCA differ from t-SNE for dimensionality reduction?**

   Your answer: _________________________________________________

---

## Summary Assessment

Count your ratings:

| Rating | Count |
|--------|-------|
| ✅ Confident | ____ / 57 |
| 🔶 Developing | ____ / 57 |
| ❌ Need Review | ____ / 57 |

**Readiness Assessment**:
- ≥50 ✅: Ready to submit
- 40–49 ✅: Minor review recommended
- 30–39 ✅: Significant review needed
- <30 ✅: Complete additional practice before submission

---

## Action Plan

Based on your self-assessment, identify areas needing improvement:

1. **Priority 1**: _______________________________________________

2. **Priority 2**: _______________________________________________

3. **Priority 3**: _______________________________________________

**Planned Actions**:

- [ ] Review lecture notes sections: _______________
- [ ] Redo laboratory sections: _______________
- [ ] Complete additional exercises: _______________
- [ ] Seek help on: _______________

---

## Declaration

By submitting this self-assessment, I confirm that:

- [ ] I have honestly evaluated my competencies
- [ ] I have completed all required laboratory exercises
- [ ] My code meets the specified quality standards
- [ ] I understand the material well enough to explain it to a peer

**Signature**: _________________________ **Date**: _____________

---
