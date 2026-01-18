# 13UNIT: Assessment Quiz

## Machine Learning for Researchers

---

**Instructions**: Answer all 10 questions. Select the single best answer for each question. Passing score: 70% (7/10).

**Time Limit**: 20 minutes

---

### Question 1 (LO1: Paradigm Distinction)

A researcher has a dataset of customer transactions and wants to identify groups of customers with similar purchasing patterns. No labels indicate which group customers belong to. Which machine learning paradigm is most appropriate?

- [ ] A) Supervised classification
- [ ] B) Supervised regression
- [ ] C) Unsupervised clustering
- [ ] D) Reinforcement learning

---

### Question 2 (LO1: Paradigm Distinction)

A medical researcher wants to predict whether a tumour is malignant or benign based on cell measurements. Historical data includes confirmed diagnoses for each sample. This problem is best addressed by:

- [ ] A) Unsupervised dimensionality reduction
- [ ] B) Supervised classification
- [ ] C) Unsupervised clustering
- [ ] D) Supervised regression

---

### Question 3 (LO1: Paradigm Distinction)

An economist wants to predict house prices (in pounds) from features such as square footage, location and number of bedrooms. Training data includes actual sale prices. This is a:

- [ ] A) Classification problem requiring accuracy metric
- [ ] B) Regression problem requiring MSE or R² metric
- [ ] C) Clustering problem requiring silhouette score
- [ ] D) Dimensionality reduction problem

---

### Question 4 (LO3: Validation Methodology)

A data scientist fits a StandardScaler on the entire dataset before splitting into train and test sets, then trains a classifier on the scaled training set. What problem does this create?

- [ ] A) The model will underfit due to insufficient data
- [ ] B) Data leakage—test set statistics influenced the scaler
- [ ] C) The model will be too slow to train
- [ ] D) Class imbalance will not be handled properly

---

### Question 5 (LO3: Validation Methodology)

When using 5-fold cross-validation, how is the final performance estimate computed?

- [ ] A) The best score among the 5 folds
- [ ] B) The score from the last fold
- [ ] C) The mean of scores across all 5 folds
- [ ] D) The score on the complete training set

---

### Question 6 (LO4: Metric Interpretation)

A spam filter is evaluated on a test set. The confusion matrix shows:
- True Positives (spam correctly identified): 180
- False Positives (legitimate marked as spam): 20
- True Negatives (legitimate correctly passed): 780
- False Negatives (spam missed): 20

What is the precision for the spam class?

- [ ] A) 0.90
- [ ] B) 0.95
- [ ] C) 0.97
- [ ] D) 0.98

---

### Question 7 (LO4: Metric Interpretation)

A disease screening test achieves 99% accuracy on a dataset where only 1% of patients have the disease. A model that simply predicts "no disease" for everyone would achieve:

- [ ] A) 0% accuracy
- [ ] B) 1% accuracy
- [ ] C) 50% accuracy
- [ ] D) 99% accuracy

---

### Question 8 (LO5: Pitfall Mitigation)

A model achieves 98% accuracy on training data but only 72% on test data. Which phenomenon does this demonstrate?

- [ ] A) Underfitting
- [ ] B) Overfitting
- [ ] C) Class imbalance
- [ ] D) Data leakage

---

### Question 9 (LO5: Pitfall Mitigation)

When performing hyperparameter tuning with GridSearchCV and then reporting the best cross-validation score as the final performance estimate, what problem occurs?

- [ ] A) The estimate is unbiased
- [ ] B) The estimate is optimistically biased (too high)
- [ ] C) The estimate is pessimistically biased (too low)
- [ ] D) The model will fail on new data

---

### Question 10 (LO5: Pitfall Mitigation)

In a binary classification problem with 95% negative class and 5% positive class, which strategy would NOT help improve positive class detection?

- [ ] A) Using class_weight='balanced' in the classifier
- [ ] B) Oversampling the minority class with SMOTE
- [ ] C) Using accuracy as the evaluation metric
- [ ] D) Adjusting the classification threshold

---

## Answer Key

*(For instructor use only)*

| Question | Correct Answer | Explanation |
|----------|---------------|-------------|
| Q1 | C | No labels + finding groups = unsupervised clustering |
| Q2 | B | Labelled data + discrete output = supervised classification |
| Q3 | B | Labelled data + continuous output = regression |
| Q4 | B | Scaler learned test set statistics, causing leakage |
| Q5 | C | CV estimates by averaging across all folds |
| Q6 | A | Precision = TP/(TP+FP) = 180/200 = 0.90 |
| Q7 | D | Predicting majority class achieves majority proportion accuracy |
| Q8 | B | Large train-test gap indicates overfitting |
| Q9 | B | Selection bias inflates reported performance |
| Q10 | C | Accuracy is misleading with imbalanced classes |

---

## Score Interpretation

| Score | Interpretation | Recommended Action |
|-------|---------------|-------------------|
| 9–10 | Excellent understanding | Proceed to advanced topics |
| 7–8 | Satisfactory—passing | Review missed concepts |
| 5–6 | Partial understanding | Revisit lecture notes and labs |
| 0–4 | Needs significant review | Re-study theory, redo labs |

---
