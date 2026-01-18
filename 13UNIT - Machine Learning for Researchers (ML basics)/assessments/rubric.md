# 13UNIT: Grading Rubric

## Machine Learning for Researchers

---

## Overview

This rubric defines assessment criteria for laboratory submissions and homework exercises. Each criterion is weighted according to its importance for demonstrating machine learning competency.

---

## Overall Weighting

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Algorithms produce expected outputs |
| Code Quality | 25% | Type hints, docstrings, style compliance |
| Methodology | 25% | Appropriate validation, metric selection |
| Documentation | 10% | Clear reasoning, result interpretation |

---

## Detailed Criteria

### 1. Correctness (40%)

#### Exemplary (36–40 points)
- All functions produce correct outputs for all test cases
- Edge cases handled appropriately (empty inputs, single samples)
- Numerical results match expected values within tolerance
- No runtime errors or warnings

#### Proficient (28–35 points)
- Core functionality correct for standard cases
- Minor issues with edge cases
- Results mostly accurate with small deviations
- No critical errors

#### Developing (16–27 points)
- Some functions produce correct outputs
- Multiple edge cases fail
- Noticeable inaccuracies in results
- Some runtime errors in specific scenarios

#### Beginning (0–15 points)
- Fundamental logic errors
- Most test cases fail
- Significant numerical errors
- Frequent runtime errors

---

### 2. Code Quality (25%)

#### Exemplary (23–25 points)
- Type hints on all function parameters and return values
- Google-style docstrings for all public functions
- Passes `ruff check` with no errors
- Passes `mypy --strict` with no errors
- Consistent naming conventions (snake_case for functions/variables)
- Appropriate use of constants (no magic numbers)
- Logical code organisation

#### Proficient (18–22 points)
- Type hints on most functions (>80%)
- Docstrings present but may lack detail
- Minor style issues (1–3 ruff warnings)
- Minor type issues (1–3 mypy warnings)
- Generally consistent naming

#### Developing (10–17 points)
- Type hints on some functions (50–80%)
- Incomplete or missing docstrings
- Multiple style issues (4–10 warnings)
- Type checking issues
- Inconsistent naming in places

#### Beginning (0–9 points)
- Few or no type hints
- Missing docstrings
- Many style violations (>10)
- Fails type checking significantly
- Inconsistent or unclear naming

---

### 3. Methodology (25%)

#### Exemplary (23–25 points)
- Validation protocol appropriate for problem (stratified splits for classification)
- No data leakage—preprocessing inside CV loops
- Metrics appropriate for problem characteristics
- Hyperparameter selection uses proper nested CV when required
- Class imbalance handled appropriately
- Random states set for reproducibility

#### Proficient (18–22 points)
- Validation generally correct with minor issues
- Preprocessing mostly correct
- Appropriate primary metric selected
- Basic hyperparameter tuning implemented
- Reproducibility mostly ensured

#### Developing (10–17 points)
- Validation present but with methodological issues
- Potential data leakage in preprocessing
- Metrics selected without clear justification
- Hyperparameter tuning may be biased
- Reproducibility inconsistent

#### Beginning (0–9 points)
- Inappropriate or missing validation
- Clear data leakage
- Inappropriate metrics for problem type
- No hyperparameter consideration
- Results not reproducible

---

### 4. Documentation (10%)

#### Exemplary (9–10 points)
- Clear explanation of methodological choices
- Results interpreted in domain context
- Trade-offs discussed thoughtfully
- Limitations acknowledged
- Well-organised presentation

#### Proficient (7–8 points)
- Adequate explanation of approach
- Basic result interpretation
- Some trade-off discussion
- Organised presentation

#### Developing (4–6 points)
- Minimal explanation
- Results stated without interpretation
- Trade-offs not discussed
- Disorganised presentation

#### Beginning (0–3 points)
- No explanation of approach
- No result interpretation
- No documentation
- Unclear presentation

---

## Laboratory-Specific Criteria

### Lab 13_01: Supervised Learning

| Section | Points | Key Assessment Criteria |
|---------|--------|------------------------|
| §1 Data Preparation | 15 | Correct loading, exploration, stratified splitting |
| §2 Classification | 20 | Pipeline construction, metric computation, visualisation |
| §3 Regression | 15 | Regression pipeline, diagnostic plots |
| §4 Model Selection | 25 | CV implementation, grid search, nested CV |
| §5 Pitfall Demos | 25 | Overfitting detection, leakage demonstration, imbalance handling |

### Lab 13_02: Unsupervised Learning

| Section | Points | Key Assessment Criteria |
|---------|--------|------------------------|
| §1 Clustering | 30 | Algorithm implementation, evaluation metrics |
| §2 Dimensionality | 25 | PCA analysis, t-SNE visualisation |
| §3 Pipelines | 25 | Preprocessing integration, anomaly detection |
| §4 Applications | 20 | Research domain application |

---

## Exercise-Specific Criteria

### Easy Exercises (Each worth 10 points)

| Criterion | Points |
|-----------|--------|
| Function signatures match specification | 2 |
| Core functionality correct | 4 |
| Type hints and docstrings | 2 |
| Style compliance | 2 |

### Medium Exercises (Each worth 15 points)

| Criterion | Points |
|-----------|--------|
| Function signatures match specification | 2 |
| Core functionality correct | 6 |
| Methodology appropriate | 3 |
| Type hints and docstrings | 2 |
| Style compliance | 2 |

### Hard Exercises (Each worth 20 points)

| Criterion | Points |
|-----------|--------|
| Function/class signatures match specification | 2 |
| Core functionality correct | 8 |
| Methodology exemplary | 4 |
| Documentation and interpretation | 2 |
| Type hints and docstrings | 2 |
| Style compliance | 2 |

---

## Common Deductions

| Issue | Deduction | Category |
|-------|-----------|----------|
| Missing type hint on function | -0.5 per function | Code Quality |
| Missing docstring on public function | -0.5 per function | Code Quality |
| Ruff warning | -0.25 per warning | Code Quality |
| Mypy error | -0.5 per error | Code Quality |
| Data leakage | -5 to -10 | Methodology |
| Inappropriate metric selection | -3 | Methodology |
| Missing stratification for imbalanced data | -2 | Methodology |
| Hardcoded random state without parameter | -1 | Methodology |
| No result interpretation | -2 | Documentation |
| Unclear code organisation | -1 | Code Quality |

---

## Grade Scale

| Percentage | Grade | Description |
|------------|-------|-------------|
| 90–100% | A | Excellent—exemplary work |
| 80–89% | B | Good—proficient understanding |
| 70–79% | C | Satisfactory—meets requirements |
| 60–69% | D | Marginal—needs improvement |
| <60% | F | Unsatisfactory—significant gaps |

---

## Appeals Process

Students may appeal grades by:
1. Submitting written explanation within 5 days
2. Identifying specific rubric criteria in question
3. Providing evidence supporting different assessment
4. Meeting with instructor to discuss

---
