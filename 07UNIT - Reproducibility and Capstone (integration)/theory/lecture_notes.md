# 07UNIT: Reproducibility and Capstone

## Lecture Notes

---

## 1. The Reproducibility Crisis: Origins and Implications

### 1.1 Historical Context

The term "reproducibility crisis" gained scientific prominence following a 2016 survey published in *Nature*, wherein more than 1,500 researchers reported their experiences with research replication. The findings proved sobering: 70% of respondents had failed to reproduce another scientist's experiments, whilst over 50% could not replicate their own work. Though this survey addressed experimental sciences broadly, computational research faces analogous—and in some respects more tractable—challenges.

The roots of reproducibility concerns trace to earlier work. Jon Claerbout, a computational geophysicist at Stanford, advocated for "reproducible research" as early as 1992, recognising that computational results without accompanying code and data remained in effect unverifiable. His insight—that scientific claims resting upon computation require computational verification—anticipated debates that would intensify two decades later.

The 2005 paper by Ioannidis, provocatively titled "Why Most Published Research Findings Are False," catalysed broader attention to replication failures. Although focused on medical research, its statistical arguments about publication bias, multiple testing and effect size inflation applied with equal force to computational disciplines. Subsequent systematic replication efforts, such as the Open Science Collaboration's 2015 attempt to reproduce 100 psychology studies (succeeding in only 36%), demonstrated that reproducibility failures were neither rare nor confined to any single field.

### 1.2 Dimensions of Computational Reproducibility

Computational research admits several distinct reproducibility standards, each progressively more demanding:

**Computational reproducibility** (strictest): obtaining bit-identical results using the same data and code. This standard appears achievable in principle—after all, computers are deterministic machines—yet fails frequently due to:

- Undocumented dependencies and version conflicts
- Inadequate random seed management
- Platform-specific numerical behaviour
- Missing preprocessing or data cleaning steps
- Implicit environmental assumptions

**Empirical reproducibility**: obtaining consistent conclusions when the same analysis is applied to independent datasets. This tests the generalisability of computational findings beyond their original context.

**Statistical reproducibility**: whether identical data subjected to different valid analytical approaches yields consistent conclusions. This addresses the "garden of forking paths" problem wherein flexible analytical choices can inflate false positive rates.

### 1.3 Consequences of Irreproducibility

The consequences of irreproducible computational research extend beyond mere academic inconvenience:

**Resource wastage**: Researchers invest substantial time attempting to build upon computational work that cannot be executed, verified or extended. A 2016 estimate suggested that preclinical research irreproducibility costs approximately $28 billion annually in the United States alone.

**Delayed scientific progress**: When foundational computational work proves irreproducible, the research edifice built upon it becomes suspect. Entire research programmes may require re-evaluation.

**Erosion of public trust**: High-profile replication failures—particularly in politically contentious areas such as climate modelling or public health—undermine confidence in computational science more broadly.

**Career implications**: Researchers whose computational findings cannot be reproduced face professional consequences ranging from informal reputational damage to formal retractions.

---

## 2. Foundations of Reproducible Research Software

### 2.1 Version Control as Scientific Methodology

Version control represents the foundation of reproducible computational practice. Git, the predominant system, provides capabilities essential to scientific work:

**Complete history**: Every modification to code, data or documentation is recorded with authorship attribution, timestamps and explanatory commit messages. This creates an auditable trail permitting reconstruction of the project's evolution.

**Branching and experimentation**: Parallel development tracks enable exploration of alternative approaches without disrupting established functionality. A researcher might maintain separate branches for different model specifications, merging successful experiments whilst preserving the ability to revisit abandoned approaches.

**Collaborative workflows**: Multiple contributors can work simultaneously without interference, reconciling their changes through structured merge processes. Pull requests and code review workflows introduce quality gates prior to integration.

**State recovery**: Any historical state of the codebase can be recovered precisely. If a bug emerges months after its introduction, one can bisect the commit history to identify exactly when problematic behaviour appeared.

Effective commit messages follow established conventions:

```
<type>: <short description>

<optional detailed explanation>

Types: feat, fix, docs, test, refactor, data
```

For instance: `fix: correct off-by-one error in bootstrap sampling`

### 2.2 Dependency Management and Environment Specification

Modern Python projects employ `pyproject.toml` for dependency specification, superseding the fragmented constellation of `setup.py`, `requirements.txt` and `setup.cfg`:

```toml
[project]
name = "reproducible-research-example"
version = "0.1.0"
dependencies = [
    "numpy>=1.24,<2.0",   # Minimum with upper bound
    "pandas>=2.0",        # Minimum only
    "scipy==1.11.0",      # Exact version pinning
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]
```

The choice between constraint strategies involves trade-offs:

| Strategy | Advantages | Disadvantages |
|----------|------------|---------------|
| Exact pinning (`==`) | Maximum reproducibility | Difficult updates, security vulnerabilities may persist |
| Minimum only (`>=`) | Flexible, receives improvements | May break with major updates |
| Bounded (`>=,<`) | Balanced approach | Requires testing across version range |

For research software prioritising reproducibility, exact pinning of direct dependencies combined with a lockfile (e.g., `pip-compile` output) provides the strongest guarantees.

### 2.3 Environment Isolation

Isolated execution environments prevent dependency conflicts and ensure reproducibility across machines:

**Virtual environments** create lightweight Python installations:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
```

**Containerisation** encapsulates the entire execution environment:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install .
COPY src/ src/
CMD ["python", "-m", "mypackage"]
```

Docker containers guarantee identical execution environments regardless of host system configuration—a property termed "build once, run anywhere."

---

## 3. Random Seed Management

### 3.1 Sources of Stochasticity in Computational Research

Computational research frequently involves stochastic processes:

- Neural network weight initialisation (random draws from distributions)
- Monte Carlo integration and simulation (pseudorandom sampling)
- Data shuffling and train/test splitting (permutation operations)
- Stochastic optimisation algorithms (SGD, Adam, genetic algorithms)
- Bootstrap and cross-validation procedures (resampling with randomness)
- Bayesian inference sampling (MCMC, variational methods)

Each source of randomness must be explicitly controlled for bit-identical reproducibility.

### 3.2 Implementing Seed Management

A thorough seed management approach addresses all random number generators present in a computational pipeline:

```python
def configure_reproducibility(seed: int) -> None:
    """
    Configure all random number generators for deterministic execution.
    
    Args:
        seed: Integer seed value propagated to all RNG sources.
    """
    import random
    import os
    
    # Python's built-in random module
    random.seed(seed)
    
    # Hash seed affects dictionary ordering (Python 3.3+)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy's random number generator
    import numpy as np
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
```

Note the PyTorch CUDA settings: `cudnn.deterministic = True` forces deterministic algorithms at a performance cost, whilst `cudnn.benchmark = False` disables runtime algorithm selection that can introduce variability.

### 3.3 Documenting Unavoidable Non-Determinism

Some computational situations resist perfect reproducibility:

- **Parallel execution**: Thread scheduling and GPU kernel execution order may vary between runs
- **Floating-point accumulation**: Reordering of floating-point operations (e.g., in parallel reductions) changes results due to non-associativity
- **External data sources**: Network fetches, database queries and API calls may return different results over time

When perfect reproducibility proves impossible, document the expected variability:

- Report results as mean ± standard deviation over multiple independent runs
- Specify the number of runs and seeds used
- Archive representative outputs alongside statistical summaries
- Establish tolerance thresholds for result comparison

---

## 4. Testing for Research Software

### 4.1 The Testing Pyramid

Research software benefits from a structured testing architecture. The canonical testing pyramid prescribes proportions:

```
          ╱╲
         ╱  ╲       End-to-End Tests (10%)
        ╱────╲      Few, expensive, full workflow
       ╱      ╲
      ╱────────╲    Integration Tests (20%)
     ╱          ╲   Moderate number, component interactions
    ╱────────────╲
   ╱              ╲ Unit Tests (70%)
  ╱────────────────╲ Many, fast, isolated functions
```

**Unit tests** verify individual functions and methods in isolation:

```python
def test_normalise_vector():
    """Verify vector normalisation produces unit length."""
    vec = np.array([3.0, 4.0])
    result = normalise(vec)
    
    np.testing.assert_allclose(result, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(result), 1.0)
```

Properties of effective unit tests:

1. **Fast**: Execute in milliseconds, enabling rapid feedback
2. **Isolated**: No external dependencies (databases, networks, filesystems)
3. **Deterministic**: Same result every execution (controlled seeds)
4. **Focused**: Test one logical unit per test function

**Integration tests** verify interactions between components:

```python
def test_pipeline_integration(tmp_path):
    """Verify full data pipeline produces expected outputs."""
    input_file = tmp_path / "input.csv"
    input_file.write_text("x,y\n1,2\n3,4")
    
    output_file = tmp_path / "output.csv"
    run_pipeline(input_file, output_file)
    
    result = pd.read_csv(output_file)
    assert len(result) == 2
    assert "prediction" in result.columns
```

**End-to-end tests** validate complete workflows against expected outputs:

```python
def test_full_analysis_workflow():
    """Verify entire analysis reproduces archived results."""
    result = run_analysis(config="baseline_v1.yaml")
    expected = load_archived_results("baseline_v1_expected.json")
    
    assert_results_equivalent(result, expected, tolerance=1e-6)
```

### 4.2 Property-Based Testing

Property-based testing complements example-based tests by generating random inputs and verifying that invariants hold:

```python
from hypothesis import given
from hypothesis import strategies as st

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=1e-10))
def test_normalise_preserves_direction(magnitude):
    """Normalisation preserves sign for non-zero scalars."""
    vec = np.array([magnitude, 0.0])
    result = normalise(vec)
    
    if magnitude > 0:
        assert result[0] > 0
    else:
        assert result[0] < 0
```

Hypothesis excels at discovering edge cases that manual test construction overlooks.

### 4.3 Fixtures and Test Organisation

pytest fixtures provide reusable test infrastructure:

```python
@pytest.fixture
def sample_dataset():
    """Provide a consistent test dataset."""
    return pd.DataFrame({
        'feature_1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5],
        'target': [0, 0, 1, 1, 1]
    })

@pytest.fixture
def trained_model(sample_dataset):
    """Provide a trained model for inference tests."""
    model = SimpleClassifier()
    model.fit(sample_dataset[['feature_1', 'feature_2']], 
              sample_dataset['target'])
    return model
```

Fixture scope controls resource lifecycle: `function` (default) recreates per test; `module` shares across a module; `session` persists for the entire test run.

---

## 5. Continuous Integration and Deployment

### 5.1 CI/CD Principles

**Continuous Integration** automatically executes tests and quality checks upon every code change. This provides:

- **Immediate feedback**: Developers know within minutes whether changes introduce regressions
- **Quality enforcement**: Every contribution meets identical standards
- **Documentation**: CI configuration documents the build and test process
- **Collaboration**: Multiple contributors can work simultaneously with confidence

**Continuous Deployment** extends CI to automatically deploy passing changes to production environments—less common in research contexts but valuable for maintaining public-facing tools, documentation sites and APIs.

### 5.2 GitHub Actions Configuration

GitHub Actions provides free CI for public repositories. A minimal workflow:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: pip install -e ".[dev]"
    
    - name: Lint with ruff
      run: ruff check src/
    
    - name: Type check with mypy
      run: mypy --strict src/
    
    - name: Test with pytest
      run: pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v4
```

### 5.3 Code Quality Automation

Modern tooling automates quality enforcement:

| Tool | Purpose | Command |
|------|---------|---------|
| ruff | Linting, formatting | `ruff check --fix src/` |
| mypy | Static type checking | `mypy --strict src/` |
| pytest-cov | Test coverage | `pytest --cov=src` |
| pre-commit | Git hook management | `pre-commit run --all-files` |

Pre-commit hooks execute checks before commit, preventing problematic code from entering the repository:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## 6. Documentation

### 6.1 Documentation Levels

Effective documentation operates at multiple granularities:

**Code-level documentation** (docstrings) explains function behaviour:

```python
def compute_entropy(distribution: np.ndarray) -> float:
    """
    Compute Shannon entropy of a probability distribution.
    
    The Shannon entropy quantifies uncertainty in a discrete 
    probability distribution, measured in bits when using base-2 
    logarithm.
    
    Args:
        distribution: Array of probabilities that must sum to 1.
            All elements must be non-negative.
    
    Returns:
        Entropy in bits. Returns 0 for deterministic distributions
        and approaches log2(n) for uniform distributions over n outcomes.
    
    Raises:
        ValueError: If distribution does not sum to 1 (within tolerance).
    
    Example:
        >>> compute_entropy(np.array([0.5, 0.5]))
        1.0
        >>> compute_entropy(np.array([1.0, 0.0]))
        0.0
    """
```

**Module-level documentation** provides orientation via README files and module docstrings.

**Project-level documentation** sites (Sphinx, MkDocs) serve broader audiences with installation guides, tutorials, API references and conceptual explanations.

### 6.2 The README as Scientific Communication

Every research software project requires a README addressing:

1. **What**: Concise description of the software's purpose and scope
2. **Why**: Scientific motivation and the problem being addressed
3. **How**: Installation instructions and basic usage examples
4. **Where**: Links to paper, data, extended documentation

A researcher encountering the repository should understand within two minutes whether this software addresses their needs and how to proceed.

### 6.3 Data Documentation

Data documentation proves essential yet frequently neglected:

- **Provenance**: Origin, collection methodology, legal constraints
- **Processing**: Transformations applied, filtering criteria, normalisation procedures
- **Schema**: Column definitions, data types, valid value ranges
- **Integrity**: Checksums enabling corruption detection

Data manifests in JSON format capture this information programmatically:

```json
{
    "version": "1.0",
    "created": "2025-01-17T12:00:00Z",
    "source": "University Hospital Clinical Database",
    "preprocessing": [
        "Removed records with missing age",
        "Normalised continuous variables to [0, 1]"
    ],
    "files": {
        "data/train.csv": "sha256:abc123...",
        "data/test.csv": "sha256:def456..."
    }
}
```

---

## 7. Test Coverage Metrics

### 7.1 Coverage Types

Test coverage quantifies which portions of code are exercised during testing:

**Line coverage** measures the proportion of source lines executed:

$$C_{\text{line}} = \frac{|\text{executed lines}|}{|\text{total lines}|} \times 100\%$$

**Branch coverage** tracks whether each conditional branch has been taken:

$$C_{\text{branch}} = \frac{|\text{executed branches}|}{|\text{total branches}|} \times 100\%$$

**Path coverage** considers all possible execution paths—exponentially expensive and rarely practical.

**Mutation coverage** introduces small changes ("mutants") to source code and verifies that tests detect them:

$$C_{\text{mutation}} = \frac{|\text{killed mutants}|}{|\text{total mutants}|} \times 100\%$$

### 7.2 Coverage Targets

Common coverage targets in research software:

- **Minimum acceptable**: 70% line coverage
- **Good practice**: 80% line coverage with branch coverage
- **High assurance**: 90%+ with mutation testing on critical components

Coverage metrics provide useful signals but do not guarantee correctness. A test suite achieving 100% coverage may still miss important behaviours if assertions are weak or edge cases unexamined.

### 7.3 Generating Coverage Reports

pytest-cov generates coverage reports:

```bash
pytest --cov=src --cov-report=html --cov-report=xml

# View HTML report
open htmlcov/index.html
```

The HTML report highlights uncovered lines in red, enabling targeted improvement of test suites.

---

## 8. Project Structure

### 8.1 Standard Layout

Well-organised project structure supports both reproducibility and collaboration:

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
│   └── exp001/
│       ├── config.yaml
│       └── results.json
├── notebooks/
│   └── exploration.ipynb
├── docs/
├── pyproject.toml
└── README.md
```

### 8.2 Principles

- **Source code** in `src/`: Importable, tested, documented library code
- **Data** in `data/`: Separate raw inputs from processed outputs; include manifests
- **Experiments** in `experiments/`: Logged configurations and results
- **Notebooks** for exploration only: Never production code; clear outputs before commit

---

## 9. Summary

Reproducibility constitutes a foundational requirement of computational scholarship, not an optional enhancement. By adopting version control, dependency management, seed control, structured testing, continuous integration and systematic documentation, researchers construct software that others can verify, extend and build upon.

The capstone project provides opportunity to synthesise these practices into a coherent, reproducible research software package demonstrating mastery across the full curriculum.

---

## References

1. Baker, M. (2016). 1,500 scientists lift the lid on reproducibility. *Nature*, 533(7604), 452-454.

2. Ioannidis, J. P. A. (2005). Why most published research findings are false. *PLOS Medicine*, 2(8), e124.

3. Open Science Collaboration (2015). Estimating the reproducibility of psychological science. *Science*, 349(6251), aac4716.

4. Peng, R. D. (2011). Reproducible research in computational science. *Science*, 334(6060), 1226-1227.

5. Sandve, G. K., Nekrutenko, A., Taylor, J., & Hovig, E. (2013). Ten simple rules for reproducible computational research. *PLOS Computational Biology*, 9(10), e1003285.

6. Wilson, G., et al. (2017). Good enough practices in scientific computing. *PLOS Computational Biology*, 13(6), e1005510.

---

© 2025 Antonio Clim. All rights reserved.
