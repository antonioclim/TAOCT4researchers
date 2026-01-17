# Week 7 Self-Assessment Checklist

## 📋 Overview

Use this checklist to evaluate your understanding of Week 7 concepts before submitting assignments or taking assessments. Be honest with yourself—identifying gaps early allows time for review.

---

## Learning Objectives Self-Check

### Objective 1: Implement Comprehensive Testing with pytest and CI/CD

Rate your confidence: ☐ Not confident | ☐ Somewhat confident | ☐ Confident | ☐ Very confident

**Can you:**

- [ ] Write unit tests following the AAA (Arrange-Act-Assert) pattern
- [ ] Create reusable fixtures with appropriate scopes (function, class, module, session)
- [ ] Use `pytest.mark.parametrize` to test multiple input combinations
- [ ] Apply mocking with `unittest.mock` to isolate units under test
- [ ] Test exception handling with `pytest.raises`
- [ ] Measure code coverage and interpret coverage reports
- [ ] Write a GitHub Actions workflow that runs tests on push
- [ ] Configure matrix testing across Python versions

**Quick self-test:**

```python
# Can you write a test for this function?
def calculate_discount(price: float, percentage: float) -> float:
    """Calculate discounted price."""
    if percentage < 0 or percentage > 100:
        raise ValueError("Percentage must be between 0 and 100")
    return price * (1 - percentage / 100)
```

<details>
<summary>Check your answer</summary>

```python
import pytest

def test_calculate_discount_basic():
    assert calculate_discount(100, 20) == 80.0

def test_calculate_discount_zero():
    assert calculate_discount(100, 0) == 100.0

def test_calculate_discount_full():
    assert calculate_discount(100, 100) == 0.0

def test_calculate_discount_invalid_negative():
    with pytest.raises(ValueError):
        calculate_discount(100, -10)

def test_calculate_discount_invalid_over_100():
    with pytest.raises(ValueError):
        calculate_discount(100, 150)

@pytest.mark.parametrize("price,percentage,expected", [
    (100, 10, 90),
    (200, 50, 100),
    (50, 25, 37.5),
])
def test_calculate_discount_parametrized(price, percentage, expected):
    assert calculate_discount(price, percentage) == expected
```

</details>

---

### Objective 2: Build Reproducible Project Structure with Proper Documentation

Rate your confidence: ☐ Not confident | ☐ Somewhat confident | ☐ Confident | ☐ Very confident

**Can you:**

- [ ] Create a standard Python project structure with src/ layout
- [ ] Write a complete pyproject.toml with metadata and dependencies
- [ ] Document code with Google-style docstrings
- [ ] Set random seeds for reproducible results
- [ ] Create data manifests with cryptographic hashes
- [ ] Capture environment information (package versions, Python version)
- [ ] Write a comprehensive README with installation and usage sections
- [ ] Configure linting tools (ruff) and type checking (mypy)

**Quick self-test:**

What are the five essential sections every research project README should contain?

<details>
<summary>Check your answer</summary>

1. **Project title and description** — What the project does and why it matters
2. **Installation instructions** — How to set up the environment and install dependencies
3. **Usage examples** — Working code snippets demonstrating key functionality
4. **Requirements/dependencies** — Python version, packages with versions
5. **Licence and citation** — How others can use and cite the work

Bonus sections: Contributing guidelines, changelog, acknowledgements

</details>

---

### Objective 3: Conduct Peer Review Using Established Criteria

Rate your confidence: ☐ Not confident | ☐ Somewhat confident | ☐ Confident | ☐ Very confident

**Can you:**

- [ ] Evaluate code quality using defined criteria (style, documentation, testing)
- [ ] Assess reproducibility by attempting to run another's code
- [ ] Provide constructive feedback that is specific and actionable
- [ ] Distinguish between critical issues and minor suggestions
- [ ] Review documentation for completeness and clarity
- [ ] Check that tests are meaningful and not just "green for the sake of green"

**Quick self-test:**

You're reviewing a colleague's code and notice the following test:

```python
def test_process_data():
    result = process_data([1, 2, 3])
    assert result is not None
```

What feedback would you provide?

<details>
<summary>Check your answer</summary>

This test has several issues:

1. **Weak assertion**: `assert result is not None` doesn't verify correctness—only that something is returned
2. **No expected output**: The test should check that the result matches expected behaviour
3. **Missing edge cases**: What about empty lists, negative numbers or invalid inputs?
4. **No documentation**: What is `process_data` supposed to do?

Suggested improvement:

```python
def test_process_data_doubles_values():
    """Test that process_data doubles each input value."""
    result = process_data([1, 2, 3])
    assert result == [2, 4, 6]

def test_process_data_empty_list():
    """Test that empty input returns empty output."""
    assert process_data([]) == []

def test_process_data_preserves_order():
    """Test that output order matches input order."""
    result = process_data([3, 1, 2])
    assert result == [6, 2, 4]
```

</details>

---

## Concept Comprehension Check

For each concept, rate your understanding:

| Concept | 😕 Confused | 🤔 Partial | 😊 Understand | 🌟 Can Teach |
|---------|------------|-----------|---------------|--------------|
| Reproducibility crisis causes | ☐ | ☐ | ☐ | ☐ |
| Random seed management | ☐ | ☐ | ☐ | ☐ |
| Data manifests and hashing | ☐ | ☐ | ☐ | ☐ |
| Unit vs integration testing | ☐ | ☐ | ☐ | ☐ |
| pytest fixtures and scopes | ☐ | ☐ | ☐ | ☐ |
| Mocking and test doubles | ☐ | ☐ | ☐ | ☐ |
| Property-based testing | ☐ | ☐ | ☐ | ☐ |
| GitHub Actions workflows | ☐ | ☐ | ☐ | ☐ |
| Matrix testing strategies | ☐ | ☐ | ☐ | ☐ |
| Code coverage interpretation | ☐ | ☐ | ☐ | ☐ |
| Project structure standards | ☐ | ☐ | ☐ | ☐ |
| pyproject.toml configuration | ☐ | ☐ | ☐ | ☐ |

---

## Practical Skills Verification

### Can you complete these tasks without reference materials?

**Reproducibility:**

- [ ] Set seeds for Python random and NumPy in a single function
- [ ] Generate a SHA-256 hash of a file
- [ ] Create a requirements.txt from your current environment
- [ ] Capture the Python version programmatically

**Testing:**

- [ ] Write a pytest fixture that creates a temporary directory
- [ ] Use `@pytest.mark.parametrize` with multiple parameters
- [ ] Mock a function that makes HTTP requests
- [ ] Configure pytest.ini or pyproject.toml for pytest

**CI/CD:**

- [ ] Write a GitHub Actions workflow that runs on push to main
- [ ] Configure a matrix to test Python 3.10, 3.11 and 3.12
- [ ] Add a step that fails if code coverage drops below 80%
- [ ] Set up automatic deployment on tag creation

---

## Code Quality Checklist

Before submitting any code, verify:

### Type Hints
- [ ] All function parameters have type annotations
- [ ] All function return types are specified
- [ ] Complex types use `typing` module (List, Dict, Optional, etc.)
- [ ] Code passes `mypy --strict`

### Documentation
- [ ] All public functions have docstrings
- [ ] Docstrings follow Google style (Args, Returns, Raises, Example)
- [ ] Module-level docstring explains purpose
- [ ] README is complete and accurate

### Testing
- [ ] All public functions have at least one test
- [ ] Edge cases are tested (empty inputs, boundary values)
- [ ] Error conditions are tested with `pytest.raises`
- [ ] Coverage is ≥70% (preferably ≥80%)

### Style
- [ ] Code passes `ruff check`
- [ ] Code is formatted with `ruff format`
- [ ] No `print` statements (use logging)
- [ ] No hardcoded paths (use `pathlib`)

---

## Reflection Questions

Answer these questions honestly to identify areas for improvement:

1. **What concept from this week was most challenging?**

   _Your answer:_ ____________________________________________

2. **Which lab exercise did you find most valuable? Why?**

   _Your answer:_ ____________________________________________

3. **What would you do differently if starting a new research project tomorrow?**

   _Your answer:_ ____________________________________________

4. **What question do you still have that wasn't answered?**

   _Your answer:_ ____________________________________________

---

## Action Plan

Based on your self-assessment, identify areas needing improvement:

| Area for Improvement | Specific Action | Deadline |
|---------------------|-----------------|----------|
| | | |
| | | |
| | | |

---

## Resources for Further Study

If you identified gaps, these resources may help:

- **Testing**: [pytest documentation](https://docs.pytest.org/)
- **GitHub Actions**: [GitHub Actions documentation](https://docs.github.com/en/actions)
- **Reproducibility**: [The Turing Way](https://the-turing-way.netlify.app/)
- **Code Quality**: [Real Python - Python Code Quality](https://realpython.com/python-code-quality/)

---

## Completion Confirmation

☐ I have honestly assessed my understanding of all learning objectives  
☐ I have identified specific areas where I need additional practice  
☐ I have created an action plan to address knowledge gaps  
☐ I am ready for the Week 7 assessment

**Date completed:** _______________

**Confidence level for assessment:** ☐ Low | ☐ Medium | ☐ High

---

© 2025 Antonio Clim. All rights reserved.
