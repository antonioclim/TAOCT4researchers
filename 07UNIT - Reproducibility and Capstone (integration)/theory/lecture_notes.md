# Week 7: Reproducibility and Capstone

## Lecture Notes

### Overview

This final week addresses one of the most pressing challenges in computational research: reproducibility. We explore the tools, techniques and cultural practices that enable other researchers—or your future self—to reliably reproduce your computational results.

---

## 1. The Reproducibility Crisis

### 1.1 Historical Context

The term "reproducibility crisis" gained prominence following a 2016 survey in *Nature* where more than 70% of researchers reported being unable to reproduce another scientist's experiments, and more than 50% had failed to reproduce their own experiments. While this survey primarily addressed experimental sciences, computational research faces analogous challenges.

### 1.2 Computational Reproducibility

In computational research, reproducibility encompasses several dimensions:

**Computational reproducibility** refers to the ability to obtain the same results using the same data and code. This should be achievable in principle but often fails due to:

- Undocumented dependencies
- Random seed management issues
- Platform-specific behaviour
- Missing data or preprocessing steps

**Empirical reproducibility** involves obtaining consistent results when the same analysis is applied to independent datasets. This is a higher bar that tests the generalisability of findings.

**Statistical reproducibility** concerns whether the same statistical conclusions can be drawn from the same data using different valid analytical approaches.

### 1.3 Consequences of Irreproducibility

The consequences of irreproducible research extend beyond academic concerns:

1. **Wasted resources**: Researchers spend significant time attempting to build upon work that cannot be replicated.
2. **Delayed progress**: Scientific advancement slows when foundational work cannot be verified.
3. **Erosion of trust**: Public and funder confidence in research diminishes.
4. **Career impacts**: Researchers may face retractions or reputational damage.

---

## 2. Foundations of Reproducible Research Software

### 2.1 Version Control

Version control is the cornerstone of reproducible computational work. Git, the most widely adopted system, provides:

- **Complete history**: Every change is recorded with author, timestamp and message.
- **Branching**: Parallel development tracks for features and experiments.
- **Collaboration**: Multiple contributors can work simultaneously.
- **Reversal**: Any previous state can be recovered.

Best practices for version control in research:

```
# Commit message format
<type>: <short description>

<optional detailed explanation>

Types: feat, fix, docs, test, refactor, data
```

### 2.2 Dependency Management

Modern Python projects should specify dependencies precisely:

```toml
# pyproject.toml
[project]
dependencies = [
    "numpy>=1.24,<2.0",  # Minimum with upper bound
    "pandas>=2.0",       # Minimum only
    "scipy==1.11.0",     # Exact version
]
```

The choice between minimum, maximum and exact version constraints involves trade-offs:

| Approach | Pros | Cons |
|----------|------|------|
| Exact versions | Maximum reproducibility | Difficult to update, security risks |
| Minimum only | Flexible, gets improvements | May break with updates |
| Min and max | Balanced approach | Requires testing across range |

### 2.3 Environment Isolation

Isolated environments prevent dependency conflicts and ensure reproducibility:

**Virtual environments** (venv):
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
```

**Containerisation** (Docker):
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install .
COPY src/ src/
CMD ["python", "-m", "mypackage"]
```

---

## 3. Random Seed Management

### 3.1 Sources of Randomness

Computational research frequently involves stochastic processes:

- Initialisation of neural network weights
- Monte Carlo simulations
- Data shuffling and sampling
- Stochastic optimisation algorithms

Each source of randomness must be controlled for reproducibility.

### 3.2 Seed Setting Strategies

A comprehensive seed management approach addresses all random sources:

```python
def set_all_seeds(seed: int) -> None:
    """Set seeds for all random number generators."""
    import random
    import os
    
    # Python's built-in random
    random.seed(seed)
    
    # Hash seed for dictionary ordering
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch (if used)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```

### 3.3 Documenting Stochasticity

When perfect reproducibility is impossible (e.g., parallel execution), document the expected variability:

- Report results as mean ± standard deviation over multiple runs
- Specify the number of independent runs
- Archive representative results alongside statistical summaries

---

## 4. Testing for Research Software

### 4.1 The Testing Pyramid

Research software benefits from a structured testing approach:

```
        /\
       /  \      End-to-End Tests
      /----\     (Few, expensive)
     /      \
    /--------\   Integration Tests
   /          \  (Moderate number)
  /------------\
 /              \ Unit Tests
/________________\ (Many, fast)
```

### 4.2 Unit Testing

Unit tests verify individual functions and methods in isolation:

```python
def test_normalise_vector():
    """Test vector normalisation."""
    vec = np.array([3.0, 4.0])
    result = normalise(vec)
    
    assert np.allclose(result, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(result), 1.0)
```

Properties of good unit tests:

1. **Fast**: Execute in milliseconds
2. **Isolated**: No external dependencies
3. **Deterministic**: Same result every time
4. **Focused**: Test one thing per test

### 4.3 Property-Based Testing

Property-based testing generates random inputs and verifies that properties hold:

```python
from hypothesis import given
from hypothesis import strategies as st

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_normalise_preserves_direction(x):
    """Normalisation preserves direction for non-zero vectors."""
    if abs(x) > 1e-10:
        vec = np.array([x, 0.0])
        result = normalise(vec)
        assert np.sign(result[0]) == np.sign(x)
```

This approach excels at finding edge cases that manual testing misses.

### 4.4 Regression Testing

Regression tests capture expected outputs and alert when they change:

```python
def test_model_output_regression():
    """Ensure model output hasn't changed unexpectedly."""
    model = load_model("baseline_v1")
    test_input = load_fixture("standard_input.json")
    
    result = model.predict(test_input)
    expected = load_fixture("expected_output.json")
    
    assert_outputs_equivalent(result, expected, tolerance=1e-6)
```

---

## 5. Continuous Integration and Deployment

### 5.1 CI/CD Concepts

Continuous Integration (CI) automatically runs tests and checks on every code change. Continuous Deployment (CD) extends this to automatically deploy passing changes.

Benefits for research:

- **Immediate feedback**: Know within minutes if changes break anything
- **Consistent quality**: Every contribution meets the same standards
- **Documentation**: The CI configuration documents the build process

### 5.2 GitHub Actions

GitHub Actions provides free CI/CD for public repositories:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.12'
    - run: pip install -e ".[dev]"
    - run: pytest --cov=src
```

### 5.3 Code Quality Automation

Modern tooling automates code quality enforcement:

| Tool | Purpose |
|------|---------|
| ruff | Linting and formatting |
| mypy | Static type checking |
| pytest-cov | Test coverage measurement |
| pre-commit | Git hook management |

---

## 6. Documentation

### 6.1 Documentation Levels

Effective documentation operates at multiple levels:

**Code-level**: Docstrings explain what functions do and how to use them.

```python
def compute_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    Args:
        distribution: Array of probabilities summing to 1.
        
    Returns:
        Entropy in bits (base 2).
        
    Raises:
        ValueError: If distribution doesn't sum to 1.
        
    Example:
        >>> compute_entropy(np.array([0.5, 0.5]))
        1.0
    """
```

**Module-level**: README files and module docstrings provide orientation.

**Project-level**: Comprehensive documentation sites (Sphinx, MkDocs) serve users and contributors.

### 6.2 The README

Every research software project needs a README covering:

1. **What**: Brief description and purpose
2. **Why**: Motivation and problem addressed
3. **How**: Installation and basic usage
4. **Where**: Links to documentation, paper, data

### 6.3 Data Documentation

Data documentation is often neglected but essential:

- **Provenance**: Where did the data come from?
- **Processing**: What transformations were applied?
- **Schema**: What do fields/columns represent?
- **Integrity**: How can correctness be verified?

---

## 7. Project Structure

### 7.1 Standard Layout

A well-organised project structure facilitates reproducibility:

```
project/
├── src/
│   └── package_name/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── conftest.py
│   └── test_core.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── MANIFEST.json
├── experiments/
│   └── exp001.json
├── notebooks/
│   └── exploration.ipynb
├── docs/
├── pyproject.toml
└── README.md
```

### 7.2 Separation of Concerns

Key principles:

- **Source code** in `src/`: Importable, tested, documented
- **Data** in `data/`: With clear raw/processed separation
- **Experiments** in `experiments/`: Logged with full configuration
- **Notebooks** for exploration, not production code

---

## 8. Capstone Project Integration

### 8.1 Applying Course Concepts

Your capstone project should demonstrate mastery across all seven weeks:

| Week | Concept | Capstone Application |
|------|---------|---------------------|
| 1 | Computation | Clear algorithmic foundation |
| 2 | Design patterns | Well-structured code architecture |
| 3 | Complexity | Performance analysis |
| 4 | Data structures | Appropriate structure selection |
| 5 | Scientific computing | Numerical methods where appropriate |
| 6 | Visualisation | Publication-quality figures |
| 7 | Reproducibility | Full reproducibility package |

### 8.2 Peer Review

Peer review is integral to research quality. Review criteria include:

- **Functionality**: Does the code work as claimed?
- **Reproducibility**: Can results be reproduced?
- **Code quality**: Is the code readable and maintainable?
- **Documentation**: Is usage clear?
- **Testing**: Are tests comprehensive?

---

## Summary

Reproducibility is not an afterthought but a fundamental aspect of computational research. By adopting the practices covered in this week—version control, dependency management, seed control, testing, CI/CD and documentation—you build research software that others can trust, verify and build upon.

The capstone project provides an opportunity to synthesise all course concepts into a coherent, reproducible research software package that demonstrates your computational thinking skills.

---

## References

1. Baker, M. (2016). 1,500 scientists lift the lid on reproducibility. *Nature*, 533(7604), 452-454.

2. Peng, R. D. (2011). Reproducible research in computational science. *Science*, 334(6060), 1226-1227.

3. Wilson, G., et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510.

4. Pineau, J., et al. (2021). Improving reproducibility in machine learning research. *Journal of Machine Learning Research*, 22, 1-20.

---

© 2025 Antonio Clim. All rights reserved.
