# Week 7: Reproducibility and Capstone — Cheatsheet

> **One-page reference** | Print-friendly A4 format | © 2025 Antonio Clim

---

## 🔑 Key Concepts at a Glance

| Concept | Definition | Why It Matters |
|---------|------------|----------------|
| **Reproducibility** | Same code + same data → same results | Scientific validity |
| **Replicability** | Different code → same conclusions | Scientific rigour |
| **Determinism** | Eliminating randomness sources | Debugging, verification |
| **CI/CD** | Automated test and deployment pipelines | Quality assurance |
| **Data Manifest** | Checksums + metadata for datasets | Data integrity |

---

## 🎲 Random Seed Management

```python
import random
import numpy as np

def set_global_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # For PyTorch: torch.manual_seed(seed)
    # For TensorFlow: tf.random.set_seed(seed)

# Context manager for temporary seed
from contextlib import contextmanager

@contextmanager
def temporary_seed(seed: int):
    state = random.getstate()
    np._state = np.random.get_state()
    set_global_seeds(seed)
    try:
        yield
    finally:
        random.setstate(state)
        np.random.set_state(np._state)
```

---

## 🧪 pytest Essentials

### Test Structure (AAA Pattern)

```python
def test_function_behaviour():
    # Arrange — set up test data
    data = [3, 1, 4, 1, 5]
    
    # Act — execute the function
    result = sort_data(data)
    
    # Assert — verify the outcome
    assert result == [1, 1, 3, 4, 5]
```

### Common Fixtures

```python
import pytest

@pytest.fixture
def sample_dataframe():
    """Provide a test DataFrame."""
    return pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory."""
    return tmp_path / "test_output"
```

### Parametrised Tests

```python
@pytest.mark.parametrize("input,expected", [
    ([1, 2, 3], 6),
    ([], 0),
    ([-1, 1], 0),
])
def test_sum(input, expected):
    assert sum(input) == expected
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch

@patch('module.external_api_call')
def test_with_mock(mock_api):
    mock_api.return_value = {'status': 'ok'}
    result = function_using_api()
    assert result == expected
    mock_api.assert_called_once()
```

---

## 🔄 GitHub Actions Quick Reference

### Minimal Workflow (.github/workflows/ci.yml)

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
      - run: pip install -e .[dev]
      - run: pytest --cov=src
```

### Matrix Testing

```yaml
strategy:
  matrix:
    python-version: ['3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest]
```

### Caching Dependencies

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

---

## 📁 Project Structure Template

```
my_research_project/
├── pyproject.toml          # Project metadata and dependencies
├── README.md               # Project documentation
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core.py         # Main functionality
│       └── utils.py        # Helper functions
├── tests/
│   ├── conftest.py         # Shared fixtures
│   └── test_core.py
├── data/
│   ├── raw/                # Original, immutable data
│   └── processed/          # Transformed data
├── notebooks/              # Exploratory analysis
└── .github/
    └── workflows/
        └── ci.yml          # CI/CD pipeline
```

---

## 📋 Data Manifest Format

```json
{
  "version": "1.0.0",
  "created": "2025-01-15T10:30:00Z",
  "files": [
    {
      "path": "data/raw/experiment_001.csv",
      "sha256": "a1b2c3d4...",
      "size_bytes": 1048576,
      "rows": 10000,
      "columns": ["id", "value", "timestamp"]
    }
  ]
}
```

---

## ⚠️ Common Mistakes to Avoid

| Mistake | Problem | Solution |
|---------|---------|----------|
| Forgetting seeds | Non-reproducible results | Set seeds at script start |
| Testing implementation | Brittle tests | Test behaviour, not code |
| Hardcoded paths | Fails on other machines | Use `pathlib.Path` + config |
| No `.gitignore` | Bloated repository | Ignore data, caches, outputs |
| Skipping CI | Bugs reach main branch | Always run tests on push |

---

## 🔗 Connections to Previous Weeks

| Week | Concept | Connection to Week 7 |
|------|---------|---------------------|
| 1 | Determinism | Turing machines are deterministic |
| 2 | Encapsulation | Test interfaces, not internals |
| 3 | Complexity | Benchmark tests for performance |
| 4 | Data structures | Test edge cases thoroughly |
| 5 | Simulations | Seed management critical |
| 6 | Visualisation | Reproducible figures |

---

## 📊 Testing Pyramid

```
         /\
        /  \      E2E Tests (few, slow)
       /----\
      /      \    Integration Tests
     /--------\
    /          \  Unit Tests (many, fast)
   /____________\
```

**Target coverage**: ≥80% for research code

---

## 🛠️ Essential Commands

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_core.py -v

# Run tests matching pattern
pytest -k "test_sort"

# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/ --strict

# Build documentation
sphinx-build -b html docs/ docs/_build/
```

---

## ✅ Pre-Submission Checklist

- [ ] All tests pass locally
- [ ] Coverage ≥80%
- [ ] No linting errors
- [ ] Type hints complete
- [ ] README updated
- [ ] Data manifest current
- [ ] Dependencies pinned
- [ ] CI/CD pipeline green

**Verification command sequence:**

```bash
make lint && make typecheck && make test && make validate
```

---

*© 2025 Antonio Clim. All rights reserved.*
