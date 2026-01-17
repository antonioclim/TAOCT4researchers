#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Medium 03 - CI/CD Configuration
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Continuous Integration and Continuous Deployment (CI/CD) automates testing and
deployment workflows. This exercise focuses on understanding and configuring
GitHub Actions workflows for Python research projects.

PREREQUISITES
─────────────
- Understanding of YAML syntax
- Familiarity with pytest and code quality tools
- Basic knowledge of Git workflows

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Configure GitHub Actions workflows
2. Set up matrix testing for multiple Python versions
3. Integrate code quality tools into CI pipelines

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 40 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Understanding Workflow Structure
# ═══════════════════════════════════════════════════════════════════════════════

# This is a basic GitHub Actions workflow. Study it and answer the questions
# in the docstrings below.

BASIC_WORKFLOW = """
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Run tests
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
"""


def question_1_triggers() -> list[str]:
    """
    Question: What events trigger this workflow?

    Examine the 'on:' section of BASIC_WORKFLOW.

    Returns:
        List of trigger events (e.g., ["push to main", "pull request to main"]).

    TODO: Return the correct list of triggers.
    """
    # TODO: Implement
    pass


def question_2_steps() -> int:
    """
    Question: How many steps are in the 'test' job?

    Count the items under 'steps:' in BASIC_WORKFLOW.

    Returns:
        Number of steps.

    TODO: Return the correct count.
    """
    # TODO: Implement
    pass


def question_3_python_version() -> str:
    """
    Question: Which Python version is used?

    Find the 'python-version' in BASIC_WORKFLOW.

    Returns:
        Python version string.

    TODO: Return the correct version.
    """
    # TODO: Implement
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Matrix Testing Configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MatrixConfig:
    """Configuration for matrix testing."""

    python_versions: list[str] = field(default_factory=lambda: ["3.11", "3.12"])
    operating_systems: list[str] = field(default_factory=lambda: ["ubuntu-latest"])
    exclude: list[dict[str, str]] = field(default_factory=list)


def generate_matrix_workflow(config: MatrixConfig) -> str:
    """
    Generate a GitHub Actions workflow with matrix testing.

    The workflow should test on all combinations of Python versions and
    operating systems defined in the config.

    Args:
        config: Matrix configuration.

    Returns:
        YAML string for the workflow.

    Example:
        >>> config = MatrixConfig(
        ...     python_versions=["3.11", "3.12"],
        ...     operating_systems=["ubuntu-latest", "windows-latest"]
        ... )
        >>> workflow = generate_matrix_workflow(config)
        >>> "matrix:" in workflow
        True

    Expected output structure:
        jobs:
          test:
            runs-on: ${{ matrix.os }}
            strategy:
              matrix:
                python-version: ['3.11', '3.12']
                os: [ubuntu-latest, windows-latest]
    """
    # TODO: Implement this function
    # Use string formatting or a YAML library
    pass


def add_matrix_exclusion(
    workflow: str,
    exclude_config: dict[str, str],
) -> str:
    """
    Add an exclusion to a matrix workflow.

    Matrix exclusions prevent specific combinations from running.
    For example, excluding Python 3.11 on Windows.

    Args:
        workflow: Existing workflow YAML string.
        exclude_config: Configuration to exclude, e.g.,
            {"python-version": "3.11", "os": "windows-latest"}

    Returns:
        Modified workflow YAML string.

    Example:
        >>> workflow = generate_matrix_workflow(MatrixConfig())
        >>> modified = add_matrix_exclusion(
        ...     workflow,
        ...     {"python-version": "3.11", "os": "windows-latest"}
        ... )
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Code Quality Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityConfig:
    """Configuration for code quality checks."""

    enable_ruff: bool = True
    enable_mypy: bool = True
    enable_pytest: bool = True
    coverage_threshold: int = 80
    fail_fast: bool = True


def generate_quality_workflow(config: QualityConfig) -> str:
    """
    Generate a comprehensive code quality workflow.

    The workflow should include separate jobs for:
    1. Linting with ruff (if enabled)
    2. Type checking with mypy (if enabled)
    3. Testing with pytest (if enabled)

    Args:
        config: Quality check configuration.

    Returns:
        YAML string for the workflow.

    Example:
        >>> config = QualityConfig(coverage_threshold=90)
        >>> workflow = generate_quality_workflow(config)
        >>> "ruff check" in workflow
        True
        >>> "--cov-fail-under=90" in workflow
        True

    Expected structure:
        jobs:
          lint:
            steps:
              - run: ruff check src/ tests/
              - run: ruff format --check src/ tests/

          typecheck:
            steps:
              - run: mypy src/

          test:
            steps:
              - run: pytest --cov=src --cov-fail-under=80
    """
    # TODO: Implement this function
    pass


def generate_pyproject_tool_config() -> str:
    """
    Generate pyproject.toml sections for ruff, mypy, and pytest.

    Returns:
        TOML configuration string.

    Expected output:
        [tool.ruff]
        line-length = 88
        target-version = "py312"

        [tool.ruff.lint]
        select = ["E", "F", "I", "N", "W", "UP"]

        [tool.mypy]
        python_version = "3.12"
        strict = true

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        addopts = "-v --cov=src"
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Complete CI/CD Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CICDConfig:
    """Complete CI/CD configuration."""

    project_name: str
    python_versions: list[str] = field(default_factory=lambda: ["3.12"])
    enable_docs: bool = False
    enable_release: bool = False
    test_coverage_threshold: int = 80


def generate_complete_pipeline(config: CICDConfig) -> dict[str, str]:
    """
    Generate a complete CI/CD pipeline for a research project.

    This should generate multiple workflow files:
    1. ci.yml - Main CI workflow (lint, test, typecheck)
    2. docs.yml - Documentation build (if enabled)
    3. release.yml - Release automation (if enabled)

    Args:
        config: Complete CI/CD configuration.

    Returns:
        Dictionary mapping filename to workflow content.

    Example:
        >>> config = CICDConfig(
        ...     project_name="my-research",
        ...     enable_docs=True
        ... )
        >>> workflows = generate_complete_pipeline(config)
        >>> "ci.yml" in workflows
        True
        >>> "docs.yml" in workflows
        True
    """
    # TODO: Implement this function
    pass


def save_workflows(
    workflows: dict[str, str],
    output_dir: Path,
) -> list[Path]:
    """
    Save workflow files to .github/workflows directory.

    Args:
        workflows: Dictionary mapping filename to content.
        output_dir: Project root directory.

    Returns:
        List of created file paths.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION AND EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

EXPECTED_CI_WORKFLOW = """
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: pip install ruff
      - name: Lint
        run: |
          ruff check src/ tests/
          ruff format --check src/ tests/

  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Type check
        run: mypy src/

  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Test
        run: pytest --cov=src --cov-fail-under=80 --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        if: matrix.python-version == '3.12'
"""


def run_exercises() -> None:
    """Run and validate exercises."""
    print("=" * 60)
    print("Exercise 1: Understanding Workflow Structure")
    print("=" * 60)

    triggers = question_1_triggers()
    print(f"Triggers: {triggers}")

    steps = question_2_steps()
    print(f"Number of steps: {steps}")

    version = question_3_python_version()
    print(f"Python version: {version}")

    print("\n" + "=" * 60)
    print("Exercise 2: Matrix Testing")
    print("=" * 60)

    config = MatrixConfig(
        python_versions=["3.11", "3.12"],
        operating_systems=["ubuntu-latest", "macos-latest"],
    )
    workflow = generate_matrix_workflow(config)
    print("Generated matrix workflow:")
    print(workflow[:500] if workflow else "Not implemented")

    print("\n" + "=" * 60)
    print("Exercise 3: Code Quality Pipeline")
    print("=" * 60)

    quality_config = QualityConfig(coverage_threshold=85)
    quality_workflow = generate_quality_workflow(quality_config)
    print("Generated quality workflow:")
    print(quality_workflow[:500] if quality_workflow else "Not implemented")

    print("\n" + "=" * 60)
    print("Exercise 4: Complete Pipeline")
    print("=" * 60)

    cicd_config = CICDConfig(
        project_name="research-project",
        python_versions=["3.11", "3.12"],
        enable_docs=True,
    )
    workflows = generate_complete_pipeline(cicd_config)
    if workflows:
        print(f"Generated {len(workflows)} workflow files:")
        for name in workflows:
            print(f"  - {name}")
    else:
        print("Not implemented")

    print("\n" + "=" * 60)
    print("Reference: Expected CI Workflow")
    print("=" * 60)
    print(EXPECTED_CI_WORKFLOW)


if __name__ == "__main__":
    run_exercises()
