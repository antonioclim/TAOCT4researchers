# Contributing to TAOCT4researchers

## The Art of Computational Thinking for Researchers

---

## Overview

Thank you for your interest in contributing to this educational curriculum. This document outlines the procedures and standards for proposing changes, reporting issues and submitting improvements to the repository.

**Important Notice:** Due to the restrictive licence governing these materials, contributions are accepted under specific conditions. Please read this document thoroughly before proceeding.

---

## Contribution Policy

### What We Accept

We welcome contributions that:

- **Fix errors** — Corrections to typographical, grammatical or factual errors
- **Improve clarity** — Rewordings that aid comprehension without changing meaning
- **Update dependencies** — Version updates for third-party libraries (with testing)
- **Improve accessibility** — Improvements to navigation, screen reader compatibility or colour contrast
- **Add test coverage** — Additional unit tests for existing laboratory exercises
- **Report bugs** — Well-documented issues with reproducible examples

### What Requires Prior Approval

The following require explicit written approval before submission:

- New exercises or laboratory components
- Modifications to learning objectives
- Changes to assessment materials
- Structural reorganisation of units
- New unit proposals
- Translations into other languages

### What We Do Not Accept

We cannot accept:

- Content that violates the restrictive licence terms
- Materials derived from other copyrighted sources without permission
- Contributions that introduce proprietary dependencies
- Changes that reduce pedagogical rigour or academic standards

---

## Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Read the Licence** — Understand the [LICENCE.md](LICENCE.md) terms fully
2. **Reviewed the Curriculum** — Familiarise yourself with the [curriculum structure](docs/curriculum.md)
3. **Configured Your Environment** — Follow the [installation guide](README.md#installation-and-environment)

### Development Environment

```bash
# Clone the repository
git clone https://github.com/antonioclim/TAOCT4researchers.git
cd TAOCT4researchers

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Verify installation
make check
```

---

## Contribution Workflow

### Step 1: Open an Issue

Before making changes, open an issue to discuss your proposal:

1. Go to the [Issues](https://github.com/antonioclim/TAOCT4researchers/issues) page
2. Check existing issues to avoid duplicates
3. Create a new issue using the appropriate template:
   - `Bug Report` — For errors or unexpected behaviour
   - `Feature Request` — For improvements or new features
   - `Documentation Update` — For clarifications or corrections
   - `Question` — For general enquiries

### Step 2: Fork and Branch

Once your issue is acknowledged:

```bash
# Fork the repository via GitHub interface, then:
git clone https://github.com/YOUR_USERNAME/TAOCT4researchers.git
cd TAOCT4researchers

# Add upstream remote
git remote add upstream https://github.com/antonioclim/TAOCT4researchers.git

# Create a feature branch
git checkout -b fix/issue-123-typo-correction
```

### Branch Naming Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `fix/` | Bug fixes and error corrections | `fix/issue-42-monte-carlo-test` |
| `docs/` | Documentation improvements | `docs/clarify-unit-03-objectives` |
| `feat/` | New features (requires approval) | `feat/add-unit-08-animations` |
| `test/` | Test coverage additions | `test/unit-05-lab-coverage` |
| `chore/` | Maintenance and housekeeping | `chore/update-dependencies` |

### Step 3: Make Changes

Follow these guidelines when making changes:

#### Code Style

- **Python**: Follow PEP 8; use `ruff` for linting
- **Type hints**: Required for all function signatures
- **Docstrings**: NumPy style for all public functions
- **Line length**: Maximum 88 characters
- **Imports**: Sorted with `isort`

```python
# Example function signature
def calculate_complexity(
    algorithm: Callable[[list[int]], list[int]],
    input_sizes: list[int],
    *,
    repetitions: int = 10,
) -> dict[str, float]:
    """
    Measure empirical time complexity of an algorithm.

    Parameters
    ----------
    algorithm : Callable[[list[int]], list[int]]
        The sorting or processing algorithm to benchmark.
    input_sizes : list[int]
        Sequence of input sizes to test.
    repetitions : int, optional
        Number of repetitions per size (default: 10).

    Returns
    -------
    dict[str, float]
        Mapping from input size to mean execution time.

    Examples
    --------
    >>> results = calculate_complexity(sorted, [100, 1000, 10000])
    >>> results[1000] < results[10000]
    True
    """
    ...
```

#### Documentation Style

- **Language**: British English (colour, serialisation, behaviour)
- **Oxford comma**: Not used (red, green and blue)
- **Voice**: Active voice preferred
- **Tense**: Present tense for descriptions
- **Person**: Second person for instructions ("Run the command...")

#### Commit Messages

Follow the Conventional Commits specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Test additions or corrections
- `style`: Formatting changes (no code change)
- `refactor`: Code restructuring (no behaviour change)
- `chore`: Maintenance tasks

**Examples:**

```
fix(unit-03): correct Big-O notation in exercise 2

The complexity was incorrectly stated as O(n²) when the
actual complexity is O(n log n) due to the sorting step.

Fixes #123
```

```
docs(readme): clarify Python version requirements

Added explicit mention of Python 3.10 as minimum version
and recommended 3.12 for optimal performance.
```

### Step 4: Test Your Changes

All contributions must pass the test suite:

```bash
# Run all tests
make test

# Run tests for specific unit
pytest "08UNIT - Recursion and Dynamic Programming (algorithms)/tests/" -v

# Check code style
make lint

# Validate unit structure
python scripts/validate_all_units.py
```

### Step 5: Submit Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin fix/issue-123-typo-correction
   ```

2. Open a Pull Request via GitHub interface

3. Complete the PR template:
   - Reference the related issue
   - Describe changes made
   - Confirm testing completed
   - Acknowledge licence terms

4. Await review (typically 5–10 working days)

---

## Quality Standards

### Unit Structure Validation

All units must pass structural validation:

```bash
python scripts/validate_all_units.py
```

The validator checks:

| Criterion | Requirement |
|-----------|-------------|
| Directory structure | All required folders present |
| README.md | Minimum 2,500 words |
| PlantUML diagrams | At least 3 per unit |
| SVG assets | At least 2 per unit |
| Test coverage | 100% for laboratory files |
| Type hints | All public functions annotated |
| Docstrings | All public functions documented |

### Documentation Standards

Documentation must adhere to:

- **Accuracy**: All technical claims must be verifiable
- **Completeness**: All options and parameters documented
- **Consistency**: Terminology consistent with glossary
- **Accessibility**: Clear structure with navigation aids

### Pedagogical Standards

Educational content must:

- **Align with objectives**: Support stated learning outcomes
- **Scaffold appropriately**: Respect prerequisite dependencies
- **Provide feedback**: Include automated testing where possible
- **Encourage transfer**: Connect to authentic research contexts

---

## Review Process

### What Reviewers Check

1. **Correctness** — Technical accuracy of changes
2. **Style** — Adherence to code and documentation standards
3. **Completeness** — All necessary updates (tests, docs) included
4. **Compatibility** — No breaking changes to existing functionality
5. **Pedagogy** — Educational appropriateness (for content changes)

### Response Times

| Issue Type | Initial Response | Resolution Target |
|------------|------------------|-------------------|
| Critical bug | 24 hours | 72 hours |
| Standard bug | 5 working days | 2 weeks |
| Feature request | 10 working days | Varies |
| Question | 5 working days | — |

---

## Recognition

Contributors who make significant improvements will be:

- Listed in the repository's ACKNOWLEDGEMENTS section
- Credited in release notes
- Invited to review related future contributions

---

## Code of Conduct

### Expected Behaviour

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what benefits the educational mission

### Unacceptable Behaviour

- Harassment or discriminatory language
- Personal attacks or trolling
- Publishing others' private information
- Other conduct inappropriate for an academic context

### Enforcement

Violations may result in:
1. Warning and request for correction
2. Temporary suspension from contribution
3. Permanent ban from the repository

Report concerns via the repository issue tracker with the `[CONDUCT]` tag.

---

## Contact

For contribution-related enquiries:

- **Issues**: https://github.com/antonioclim/TAOCT4researchers/issues
- **Discussions**: https://github.com/antonioclim/TAOCT4researchers/discussions

---

<div align="center">

*Thank you for helping improve computational education for researchers worldwide.*

**© 2019–2026 Antonio Clim. All rights reserved.**

</div>
