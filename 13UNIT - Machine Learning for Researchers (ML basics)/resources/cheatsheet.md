# 13UNIT: scikit-learn Cheatsheet

## Machine Learning for Researchers

---

## Quick Reference Card

### Data Splitting

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # reproducibility
    stratify=y          # preserve class ratios
)
```

---

### Common Classifiers

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Logistic Regression (linear, probabilistic)
clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# Decision Tree (interpretable, prone to overfit)
clf = DecisionTreeClassifier(max_depth=5, random_state=42)

# Random Forest (robust, ensemble)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# k-Nearest Neighbours (instance-based)
clf = KNeighborsClassifier(n_neighbors=5)

# Support Vector Machine (kernel-based)
clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
```

---

### Common Regressors

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Linear Regression (baseline)
reg = LinearRegression()

# Ridge (L2 regularisation)
reg = Ridge(alpha=1.0)

# Lasso (L1 regularisation, feature selection)
reg = Lasso(alpha=0.1)

# Random Forest Regressor
reg = RandomForestRegressor(n_estimators=100, random_state=42)
```

---

### Preprocessing

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Standardisation (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Min-Max scaling (0 to 1)
scaler = MinMaxScaler()

# One-Hot Encoding (categorical)
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
```

---

### Pipeline Construction

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# ColumnTransformer for mixed types
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(), ['gender', 'country'])
])

# Full pipeline
full_pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Usage
full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

---

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold

# Simple CV
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")

# Stratified CV (for classification)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv)

# Multiple metrics
results = cross_validate(
    model, X, y, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1']
)
```

---

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__max_iter': [100, 500, 1000]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Randomised Search (for large search spaces)
from scipy.stats import uniform, randint
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 15)
}
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=20, cv=5, random_state=42
)
```

---

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
# [[TN, FP],
#  [FN, TP]]

# Full report
print(classification_report(y_test, y_pred))

# ROC-AUC (requires probabilities)
y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
```

---

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

### Clustering

```python
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

# k-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
silhouette = silhouette_score(X, labels)

# Hierarchical
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)

# DBSCAN (density-based)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)  # -1 indicates noise
```

---

### Dimensionality Reduction

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_}")

# PCA with variance threshold
pca = PCA(n_components=0.95)  # retain 95% variance
X_pca = pca.fit_transform(X)

# t-SNE (for visualisation only)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)
```

---

### Handling Class Imbalance

```python
# Method 1: Class weights
clf = LogisticRegression(class_weight='balanced')

# Method 2: Manual weights
clf = LogisticRegression(class_weight={0: 1, 1: 10})

# Method 3: SMOTE (requires imbalanced-learn)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

### Common Scoring Strings

| Task | Scoring String |
|------|----------------|
| Classification | 'accuracy', 'precision', 'recall', 'f1', 'roc_auc' |
| Regression | 'neg_mean_squared_error', 'neg_mean_absolute_error', 'r2' |
| Clustering | Use silhouette_score separately |

Note: Regression scorers are negated so that higher is always better.

---

### Formula Reference

| Metric | Formula |
|--------|---------|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 × (P × R) / (P + R) |
| MSE | (1/n) × Σ(y - ŷ)² |
| R² | 1 - SS_res / SS_tot |
| Silhouette | (b - a) / max(a, b) |

---

### Debugging Tips

```python
# Check for data leakage
# Ensure preprocessing is fitted ONLY on training data
scaler.fit(X_train)  # Correct
scaler.fit(X)        # WRONG - leaks test info

# Verify class distribution
print(pd.Series(y_train).value_counts(normalize=True))

# Inspect feature importance (tree-based models)
importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print("Top features:", [feature_names[i] for i in sorted_idx[:5]])

# Check for convergence warnings
import warnings
warnings.filterwarnings("error")  # Treat warnings as errors
```

---
