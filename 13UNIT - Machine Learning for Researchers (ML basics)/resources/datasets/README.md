# Datasets for 13UNIT

This unit uses built-in scikit-learn datasets for consistency and ease of use.

## Classification Datasets

### Breast Cancer Wisconsin
- **Samples**: 569
- **Features**: 30
- **Classes**: 2 (malignant, benign)
- **Load**: `from sklearn.datasets import load_breast_cancer`

### Iris
- **Samples**: 150
- **Features**: 4
- **Classes**: 3
- **Load**: `from sklearn.datasets import load_iris`

## Regression Datasets

### Diabetes
- **Samples**: 442
- **Features**: 10
- **Target**: Disease progression measure
- **Load**: `from sklearn.datasets import load_diabetes`

### California Housing
- **Samples**: 20,640
- **Features**: 8
- **Target**: Median house value
- **Load**: `from sklearn.datasets import fetch_california_housing`

## Synthetic Data Generation

For controlled experiments, use:

```python
from sklearn.datasets import make_classification, make_blobs, make_regression

# Classification
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Clustering
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Regression
X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
```

## Research Datasets

For domain-specific research, consider:
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/
- **Kaggle**: https://www.kaggle.com/datasets
- **OpenML**: https://www.openml.org/
