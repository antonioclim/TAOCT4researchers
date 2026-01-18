# Week 7 Homework: Capstone Project

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | See schedule below |
| **Total Points** | 100 |
| **Estimated Time** | 15-20 hours |
| **Difficulty** | ⭐⭐⭐⭐⭐ (5/5) |

## 🔗 Prerequisites

- [ ] Completed Lab 7.1: Reproducibility Toolkit
- [ ] Completed Lab 7.2: Testing and CI/CD
- [ ] Completed Lab 7.3: Project Scaffolder
- [ ] Read lecture notes on reproducibility
- [ ] Reviewed all materials from Weeks 1-6

## 🎯 Objectives Assessed

1. [Apply] Implement thorough testing with pytest and CI/CD pipelines
2. [Create] Build a reproducible project structure with proper documentation
3. [Evaluate] Conduct peer review using established criteria

---

## Deadlines

| Date | Deliverable |
|------|-------------|
| **Thursday, 18:00** | Final proposal (1 page) |
| **Friday, 23:59** | Code and documentation on GitHub |
| **Following week** | Presentations (15-20 min per team) |

---

## Part 1: Project Structure (25 points)

### Context

A well-organised project structure is fundamental to reproducibility. Your capstone project must demonstrate mastery of all concepts covered throughout this course whilst adhering to professional software engineering standards.

### Repository Structure

```
capstone-project-[name]/
├── README.md               # MANDATORY - principal documentation
├── LICENSE                 # MANDATORY - MIT recommended
├── pyproject.toml          # MANDATORY - dependencies and metadata
├── .gitignore              # MANDATORY - exclude generated files
│
├── src/                    # Main source code
│   └── [package_name]/
│       ├── __init__.py
│       ├── core.py         # Principal logic
│       ├── utils.py        # Utilities
│       └── visualisation.py
│
├── tests/                  # Automated tests
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_core.py
│   └── test_utils.py
│
├── notebooks/              # Jupyter notebooks for exploration
│   └── exploration.ipynb
│
├── data/                   # Data (or download instructions)
│   ├── README.md           # Data description + sources
│   └── MANIFEST.json       # Hashes for verification
│
├── experiments/            # Experiment results
│   └── exp001.json
│
├── docs/                   # Extended documentation
│   └── methodology.md
│
└── figures/                # Figures for report/presentation
    ├── figure1.pdf
    └── figure2.png
```

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 1.1 | 8 | All mandatory files present and correctly formatted |
| 1.2 | 7 | Logical separation of concerns (src/tests/data/docs) |
| 1.3 | 5 | Proper package structure with `__init__.py` files |
| 1.4 | 5 | Clean `.gitignore` (no generated files committed) |

### Test Cases

```bash
# Verify structure
ls README.md LICENSE pyproject.toml .gitignore
ls src/*/
ls tests/
ls data/README.md data/MANIFEST.json
```

### Hints

<details>
<summary>💡 Hint 1: Using the Scaffolder</summary>

Use `lab_7_03_project_scaffolder.py` to generate the initial structure:

```bash
python -m lab.lab_7_03_project_scaffolder --interactive
```

This creates all mandatory files with proper templates.
</details>

<details>
<summary>💡 Hint 2: gitignore Essentials</summary>

Your `.gitignore` should include:
```
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
```
</details>

---

## Part 2: README Documentation (20 points)

### Context

The README is the first impression of your project. It must be thorough yet concise, enabling any researcher to understand, install and reproduce your work.

### Required Sections

```markdown
# Project Title

One sentence describing what the project does.

![Demo](figures/demo.gif)  <!-- Optional but impressive -->

## Quick Start

git clone https://github.com/...
cd project
pip install -e .
python -m mypackage --demo

## Problem Statement

What problem does the project solve? Why is it important?

## Methodology

- What computational approach do you use?
- Which data structures/algorithms?
- Which techniques from the course do you apply?

## Results

- Representative screenshot/figure
- Key metrics (if relevant)

## Detailed Installation

### Requirements
- Python 3.12+
- [other dependencies]

### Steps
pip install -e ".[dev]"
pytest  # verify installation

## Usage

### Simple Example
from mypackage import Model
model = Model()
result = model.run(data)

### Advanced Configuration
...

## Project Structure

[tree structure]

## Reproducibility

python run.py --seed 42 --config config/default.yaml

All experiments are logged in `experiments/`.

## Tests

pytest --cov=src --cov-report=html

Current coverage: XX%

## References

- [Paper X](link)
- [Documentation Y](link)

## Authors

- Student Name (email)

## Licence

MIT Licence
```

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 2.1 | 6 | Quick Start that works within 3 commands |
| 2.2 | 5 | Clear problem statement and methodology |
| 2.3 | 4 | Working code examples |
| 2.4 | 3 | Reproducibility instructions with seed management |
| 2.5 | 2 | Proper references and attribution |

### Test Cases

```bash
# Verify README completeness
grep -q "Quick Start" README.md
grep -q "Problem" README.md
grep -q "Methodology" README.md
grep -q "Reproducibility" README.md
grep -q "pytest" README.md
```

---

## Part 3: Code Quality (25 points)

### 3.1 Style and Quality

All quality checks must pass without errors:

```bash
ruff check src/ tests/
ruff format src/ tests/ --check
mypy src/ --strict
```

### 3.2 Code Documentation

**Every public function** must have a docstring:

```python
def simulate_epidemic(
    population: int,
    initial_infected: int,
    beta: float,
    gamma: float,
    days: int,
) -> pd.DataFrame:
    """
    Simulate SIR epidemic model.

    Parameters
    ----------
    population : int
        Total population size (N).
    initial_infected : int
        Number of initially infected individuals.
    beta : float
        Transmission rate per contact per day.
    gamma : float
        Recovery rate (1/gamma = avg infectious period).
    days : int
        Number of days to simulate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['day', 'S', 'I', 'R'].

    Raises
    ------
    ValueError
        If initial_infected > population.

    Examples
    --------
    >>> df = simulate_epidemic(1000, 10, 0.3, 0.1, 100)
    >>> df.columns.tolist()
    ['day', 'S', 'I', 'R']
    """
```

### 3.3 Type Hints

```python
# YES - complete type hints
def process_data(
    raw_data: list[dict[str, Any]],
    threshold: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    ...

# NO - without type hints
def process_data(raw_data, threshold=0.5):
    ...
```

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 3.1 | 8 | `ruff check` passes with no errors |
| 3.2 | 7 | `mypy --strict` passes on main files |
| 3.3 | 5 | All public functions have docstrings |
| 3.4 | 5 | 100% type hint coverage on public API |

---

## Part 4: Testing (15 points)

### 4.1 Test Structure

```python
# tests/test_core.py
import pytest
from mypackage.core import simulate_epidemic

class TestSimulateEpidemic:
    """Tests for simulate_epidemic function."""

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        result = simulate_epidemic(1000, 10, 0.3, 0.1, 100)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self):
        """Should have correct column names."""
        result = simulate_epidemic(1000, 10, 0.3, 0.1, 100)
        assert list(result.columns) == ['day', 'S', 'I', 'R']

    def test_population_conservation(self):
        """Total population should be conserved."""
        result = simulate_epidemic(1000, 10, 0.3, 0.1, 100)
        for _, row in result.iterrows():
            assert row['S'] + row['I'] + row['R'] == pytest.approx(1000)

    def test_invalid_initial_infected_raises(self):
        """Should raise ValueError if initial > population."""
        with pytest.raises(ValueError, match="initial_infected"):
            simulate_epidemic(100, 200, 0.3, 0.1, 100)

    @pytest.mark.parametrize("beta,gamma,expected_r0", [
        (0.3, 0.1, 3.0),
        (0.2, 0.2, 1.0),
        (0.1, 0.2, 0.5),
    ])
    def test_r0_values(self, beta, gamma, expected_r0):
        """R0 should equal beta/gamma."""
        # Implementation test
        ...
```

### 4.2 Minimum Coverage

```bash
# Coverage target: 70%
pytest --cov=src --cov-report=term-missing --cov-fail-under=70
```

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 4.1 | 5 | All tests pass (`pytest` succeeds) |
| 4.2 | 5 | Coverage ≥ 70% |
| 4.3 | 3 | Tests are meaningful (not trivial) |
| 4.4 | 2 | Edge cases covered with parametrisation |

---

## Part 5: Presentation (15 points)

### 5.1 Structure (15-20 minutes)

| Time | Section | Content |
|------|---------|---------|
| 2 min | Introduction | Problem, motivation |
| 3 min | Background | Context, prior work |
| 5 min | Methodology | Your approach, algorithms, structures |
| 5 min | **Live Demo** | Code execution, visualisations |
| 3 min | Results | Achievements, metrics |
| 2 min | Conclusions | Limitations, future work |

### 5.2 Recommended Slides

1. **Title Slide**: Title, name, date
2. **Problem Statement**: What problem are you solving?
3. **Why It Matters**: Why is it important?
4. **Related Work**: What already exists? (optional)
5. **Methodology Overview**: Architecture diagram
6. **Key Algorithm/Structure**: Important technical detail
7. **Demo**: LIVE or video backup
8. **Results**: Graphs, tables, metrics
9. **Limitations & Future Work**: What does not work perfectly
10. **Conclusion**: Key takeaways
11. **Questions**: Contact information

### 5.3 Tips for Demo

- ✅ Prepare the exact command to run
- ✅ Test on another computer/container
- ✅ Have video backup for error cases
- ✅ Test data prepared (do not download live)
- ❌ Do not improvise

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 5.1 | 5 | Clear structure and timing |
| 5.2 | 5 | Working live demo or backup |
| 5.3 | 5 | Clear communication of technical concepts |

---

## Grading Rubric

### Total: 100 points

#### Technical Excellence (35 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Architecture | 10 | Clear structure, separation of concerns |
| Implementation | 10 | Correct, efficient, idiomatic Python |
| Testing | 8 | Coverage ≥70%, meaningful tests |
| Documentation | 7 | Complete README, docstrings, comments |

#### Research Rigour (30 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Problem Formulation | 10 | Clear, well-defined problem |
| Methodology | 10 | Justified approach, course concepts applied |
| Analysis | 10 | Results interpreted, limitations acknowledged |

#### Communication (25 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Written Report | 10 | Clear and complete README + docs |
| Presentation | 10 | Structure, clarity, timing |
| Demo | 5 | Functionality, presentation |

#### Originality & Impact (10 points)

| Level | Points | Description |
|-------|--------|-------------|
| Breakthrough | 9-10 | Novel idea, real potential |
| Novel | 7-8 | Creative combination of techniques |
| Solid | 5-6 | Good implementation of concepts |
| Derivative | 0-4 | Copies examples without extensions |

---

## Pre-Submission Checklist

### Repository

- [ ] README.md complete and formatted
- [ ] LICENSE file (MIT recommended)
- [ ] .gitignore includes `__pycache__`, `.pyc`, `.env`, `data/`
- [ ] pyproject.toml with all dependencies
- [ ] No large files (>10MB) in repository

### Code

- [ ] `ruff check` no errors
- [ ] `ruff format --check` no differences
- [ ] `mypy --strict` no errors on main files
- [ ] `pytest` all tests pass
- [ ] Coverage ≥ 70%

### Documentation

- [ ] Every public function has docstring
- [ ] README explains installation and execution
- [ ] Working usage examples
- [ ] Licence specified

### Presentation

- [ ] Slides finalised
- [ ] Demo tested on another computer
- [ ] Video backup prepared
- [ ] Timing verified (15-20 minutes)

---

## Resources

### Templates

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
- [Python Project Template](https://github.com/rochacbruno/python-project-template)

### Documentation

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)

### Scientific Presentations

- [How to Give a Great Research Talk](https://www.microsoft.com/en-us/research/academic-program/give-great-research-talk/)

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Week 7 — Capstone Project*

**** 🎓

---

© 2025 Antonio Clim. All rights reserved.
