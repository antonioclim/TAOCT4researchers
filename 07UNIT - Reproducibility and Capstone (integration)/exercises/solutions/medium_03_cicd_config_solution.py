#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Medium Exercise 3 — CI/CD Configuration Generation
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for GitHub Actions workflow generation exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: UNDERSTANDING WORKFLOW STRUCTURE — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

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
          pip install pytest

      - name: Run tests
        run: pytest
"""

# Questions and answers about the workflow
WORKFLOW_QUESTIONS = {
    "q1": {
        "question": "What event triggers this workflow?",
        "answer": "Push to main branch and pull requests targeting main"
    },
    "q2": {
        "question": "What operating system does the job run on?",
        "answer": "ubuntu-latest"
    },
    "q3": {
        "question": "What Python version is used?",
        "answer": "3.12"
    },
    "q4": {
        "question": "What action is used to checkout the repository?",
        "answer": "actions/checkout@v4"
    },
    "q5": {
        "question": "How many steps are in the test job?",
        "answer": "4 steps"
    }
}


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: MATRIX TESTING — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MatrixConfig:
    """
    Configuration for matrix testing in CI.

    Attributes:
        python_versions: List of Python versions to test.
        operating_systems: List of OS to test on.
        exclude: List of (python, os) tuples to exclude from matrix.
        include: List of additional combinations to include.
    """

    python_versions: list[str] = field(default_factory=lambda: ["3.11", "3.12"])
    operating_systems: list[str] = field(default_factory=lambda: ["ubuntu-latest"])
    exclude: list[tuple[str, str]] = field(default_factory=list)
    include: list[dict[str, str]] = field(default_factory=list)


def generate_matrix_workflow(config: MatrixConfig, name: str = "CI Matrix") -> str:
    """
    Generate a GitHub Actions workflow with matrix testing.

    Creates a workflow that tests across multiple Python versions and
    operating systems using GitHub Actions matrix strategy.

    Args:
        config: MatrixConfig specifying versions and platforms.
        name: Name for the workflow.

    Returns:
        YAML string of the complete workflow.

    Example:
        >>> config = MatrixConfig(
        ...     python_versions=["3.11", "3.12"],
        ...     operating_systems=["ubuntu-latest", "windows-latest"]
        ... )
        >>> workflow = generate_matrix_workflow(config)
    """
    workflow: dict[str, Any] = {
        "name": name,
        "on": {
            "push": {"branches": ["main"]},
            "pull_request": {"branches": ["main"]}
        },
        "jobs": {
            "test": {
                "runs-on": "${{ matrix.os }}",
                "strategy": {
                    "fail-fast": False,
                    "matrix": {
                        "python-version": config.python_versions,
                        "os": config.operating_systems
                    }
                },
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {
                        "name": "Set up Python ${{ matrix.python-version }}",
                        "uses": "actions/setup-python@v5",
                        "with": {
                            "python-version": "${{ matrix.python-version }}"
                        }
                    },
                    {
                        "name": "Install dependencies",
                        "run": "python -m pip install --upgrade pip\npip install -e \".[dev]\""
                    },
                    {
                        "name": "Run tests",
                        "run": "pytest --cov=src --cov-report=xml"
                    }
                ]
            }
        }
    }

    # Add exclusions if any
    if config.exclude:
        workflow["jobs"]["test"]["strategy"]["matrix"]["exclude"] = [
            {"python-version": py, "os": os}
            for py, os in config.exclude
        ]

    # Add inclusions if any
    if config.include:
        workflow["jobs"]["test"]["strategy"]["matrix"]["include"] = config.include

    return yaml.dump(workflow, sort_keys=False, default_flow_style=False)


def add_matrix_exclusion(
    config: MatrixConfig,
    python_version: str,
    os: str
) -> MatrixConfig:
    """
    Add an exclusion to the matrix configuration.

    Args:
        config: The MatrixConfig to modify.
        python_version: Python version to exclude.
        os: Operating system to exclude.

    Returns:
        The modified MatrixConfig (same instance).
    """
    config.exclude.append((python_version, os))
    return config


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: CODE QUALITY PIPELINE — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class QualityConfig:
    """
    Configuration for code quality checks.

    Attributes:
        enable_ruff: Enable ruff linting.
        enable_mypy: Enable mypy type checking.
        enable_pytest: Enable pytest testing.
        coverage_threshold: Minimum coverage percentage required.
        python_version: Python version to use.
    """

    enable_ruff: bool = True
    enable_mypy: bool = True
    enable_pytest: bool = True
    coverage_threshold: int = 80
    python_version: str = "3.12"


def generate_quality_workflow(
    config: QualityConfig,
    name: str = "Code Quality"
) -> str:
    """
    Generate a comprehensive code quality workflow.

    Creates a workflow with separate jobs for linting, type checking
    and testing, each running in parallel for faster feedback.

    Args:
        config: QualityConfig specifying which checks to enable.
        name: Name for the workflow.

    Returns:
        YAML string of the complete workflow.
    """
    workflow: dict[str, Any] = {
        "name": name,
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]}
        },
        "jobs": {}
    }

    # Common setup steps
    setup_steps = [
        {"uses": "actions/checkout@v4"},
        {
            "name": "Set up Python",
            "uses": "actions/setup-python@v5",
            "with": {"python-version": config.python_version}
        },
        {
            "name": "Install dependencies",
            "run": "python -m pip install --upgrade pip\npip install -e \".[dev]\""
        }
    ]

    if config.enable_ruff:
        workflow["jobs"]["lint"] = {
            "runs-on": "ubuntu-latest",
            "steps": setup_steps + [
                {
                    "name": "Run ruff linter",
                    "run": "ruff check src/ tests/"
                },
                {
                    "name": "Run ruff formatter check",
                    "run": "ruff format --check src/ tests/"
                }
            ]
        }

    if config.enable_mypy:
        workflow["jobs"]["type-check"] = {
            "runs-on": "ubuntu-latest",
            "steps": setup_steps + [
                {
                    "name": "Run mypy",
                    "run": "mypy src/ --strict"
                }
            ]
        }

    if config.enable_pytest:
        workflow["jobs"]["test"] = {
            "runs-on": "ubuntu-latest",
            "steps": setup_steps + [
                {
                    "name": "Run tests with coverage",
                    "run": f"pytest --cov=src --cov-report=xml --cov-fail-under={config.coverage_threshold}"
                },
                {
                    "name": "Upload coverage",
                    "uses": "codecov/codecov-action@v3",
                    "with": {
                        "files": "./coverage.xml",
                        "fail_ci_if_error": True
                    }
                }
            ]
        }

    return yaml.dump(workflow, sort_keys=False, default_flow_style=False)


def generate_pyproject_tool_config(config: QualityConfig) -> str:
    """
    Generate pyproject.toml configuration for quality tools.

    Args:
        config: QualityConfig specifying tool settings.

    Returns:
        TOML string for the [tool] sections.
    """
    sections = []

    if config.enable_ruff:
        sections.append("""[tool.ruff]
target-version = "py312"
line-length = 88
select = ["E", "F", "W", "I", "N", "D", "UP", "ANN", "B", "C4", "SIM"]
ignore = ["ANN101", "ANN102", "D100", "D104"]

[tool.ruff.pydocstyle]
convention = "google"

[tool.ruff.isort]
known-first-party = ["src"]""")

    if config.enable_mypy:
        sections.append("""[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true""")

    if config.enable_pytest:
        sections.append(f"""[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src --cov-report=term-missing"

[tool.coverage.run]
source = ["src"]
branch = true

[tool.coverage.report]
fail_under = {config.coverage_threshold}
exclude_lines = ["pragma: no cover", "if TYPE_CHECKING:"]""")

    return "\n\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: COMPLETE CI/CD PIPELINE — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CICDConfig:
    """
    Complete CI/CD configuration.

    Attributes:
        quality: Code quality configuration.
        matrix: Matrix testing configuration.
        enable_docs: Enable documentation building.
        enable_release: Enable automated releases.
        docs_tool: Documentation tool (sphinx or mkdocs).
    """

    quality: QualityConfig = field(default_factory=QualityConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    enable_docs: bool = True
    enable_release: bool = True
    docs_tool: str = "mkdocs"


def generate_complete_pipeline(config: CICDConfig) -> dict[str, str]:
    """
    Generate a complete CI/CD pipeline with multiple workflow files.

    Creates separate workflow files for CI, documentation and releases
    following established conventions for repository organisation.

    Args:
        config: Complete CICDConfig specifying all settings.

    Returns:
        Dictionary mapping filename to workflow content.
        Keys: "ci.yml", "docs.yml", "release.yml"
    """
    workflows: dict[str, str] = {}

    # Main CI workflow
    ci_workflow: dict[str, Any] = {
        "name": "CI",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]}
        },
        "jobs": {}
    }

    # Add quality jobs
    setup_steps = [
        {"uses": "actions/checkout@v4"},
        {
            "name": "Set up Python",
            "uses": "actions/setup-python@v5",
            "with": {"python-version": config.quality.python_version}
        },
        {
            "name": "Install dependencies",
            "run": "python -m pip install --upgrade pip\npip install -e \".[dev]\""
        }
    ]

    if config.quality.enable_ruff:
        ci_workflow["jobs"]["lint"] = {
            "runs-on": "ubuntu-latest",
            "steps": setup_steps + [
                {"name": "Lint", "run": "ruff check src/ tests/"}
            ]
        }

    if config.quality.enable_mypy:
        ci_workflow["jobs"]["type-check"] = {
            "runs-on": "ubuntu-latest",
            "steps": setup_steps + [
                {"name": "Type check", "run": "mypy src/ --strict"}
            ]
        }

    # Matrix test job
    ci_workflow["jobs"]["test"] = {
        "runs-on": "${{ matrix.os }}",
        "needs": ["lint", "type-check"] if config.quality.enable_ruff else [],
        "strategy": {
            "matrix": {
                "python-version": config.matrix.python_versions,
                "os": config.matrix.operating_systems
            }
        },
        "steps": [
            {"uses": "actions/checkout@v4"},
            {
                "name": "Set up Python",
                "uses": "actions/setup-python@v5",
                "with": {"python-version": "${{ matrix.python-version }}"}
            },
            {"name": "Install", "run": "pip install -e \".[dev]\""},
            {"name": "Test", "run": "pytest --cov=src"}
        ]
    }

    workflows["ci.yml"] = yaml.dump(ci_workflow, sort_keys=False)

    # Documentation workflow
    if config.enable_docs:
        docs_workflow: dict[str, Any] = {
            "name": "Documentation",
            "on": {
                "push": {"branches": ["main"], "paths": ["docs/**", "src/**"]}
            },
            "jobs": {
                "build-docs": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": "3.12"}
                        },
                        {
                            "name": "Install",
                            "run": f"pip install {config.docs_tool}"
                        },
                        {
                            "name": "Build docs",
                            "run": f"{config.docs_tool} build" if config.docs_tool == "mkdocs" else "sphinx-build docs docs/_build"
                        },
                        {
                            "name": "Deploy",
                            "uses": "peaceiris/actions-gh-pages@v3",
                            "with": {
                                "github_token": "${{ secrets.GITHUB_TOKEN }}",
                                "publish_dir": "./site" if config.docs_tool == "mkdocs" else "./docs/_build"
                            }
                        }
                    ]
                }
            }
        }
        workflows["docs.yml"] = yaml.dump(docs_workflow, sort_keys=False)

    # Release workflow
    if config.enable_release:
        release_workflow: dict[str, Any] = {
            "name": "Release",
            "on": {
                "push": {"tags": ["v*"]}
            },
            "jobs": {
                "release": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v4"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v5",
                            "with": {"python-version": "3.12"}
                        },
                        {
                            "name": "Install build",
                            "run": "pip install build twine"
                        },
                        {
                            "name": "Build package",
                            "run": "python -m build"
                        },
                        {
                            "name": "Publish to PyPI",
                            "env": {"TWINE_USERNAME": "__token__", "TWINE_PASSWORD": "${{ secrets.PYPI_TOKEN }}"},
                            "run": "twine upload dist/*"
                        }
                    ]
                }
            }
        }
        workflows["release.yml"] = yaml.dump(release_workflow, sort_keys=False)

    return workflows


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════


def run_tests() -> None:
    """Run all validation tests for the exercises."""
    print("=" * 70)
    print("SOLUTION VALIDATION: Medium Exercise 3 — CI/CD Configuration")
    print("=" * 70)

    # Test Exercise 1: Workflow understanding
    print("\n--- Exercise 1: Workflow Understanding ---")
    parsed = yaml.safe_load(BASIC_WORKFLOW)
    assert parsed["name"] == "CI"
    assert "push" in parsed["on"]
    assert parsed["jobs"]["test"]["runs-on"] == "ubuntu-latest"
    print("✓ Basic workflow structure understood")

    # Test Exercise 2: Matrix workflow
    print("\n--- Exercise 2: Matrix Workflow ---")
    matrix_config = MatrixConfig(
        python_versions=["3.11", "3.12"],
        operating_systems=["ubuntu-latest", "windows-latest"]
    )
    workflow = generate_matrix_workflow(matrix_config)
    parsed = yaml.safe_load(workflow)

    assert "strategy" in parsed["jobs"]["test"]
    assert parsed["jobs"]["test"]["strategy"]["matrix"]["python-version"] == ["3.11", "3.12"]
    print("✓ Matrix workflow generated correctly")

    # Test exclusions
    add_matrix_exclusion(matrix_config, "3.11", "windows-latest")
    workflow = generate_matrix_workflow(matrix_config)
    parsed = yaml.safe_load(workflow)
    assert "exclude" in parsed["jobs"]["test"]["strategy"]["matrix"]
    print("✓ Matrix exclusions work")

    # Test Exercise 3: Quality workflow
    print("\n--- Exercise 3: Quality Workflow ---")
    quality_config = QualityConfig(coverage_threshold=85)
    workflow = generate_quality_workflow(quality_config)
    parsed = yaml.safe_load(workflow)

    assert "lint" in parsed["jobs"]
    assert "type-check" in parsed["jobs"]
    assert "test" in parsed["jobs"]
    print("✓ Quality workflow generated correctly")

    # Test pyproject generation
    toml = generate_pyproject_tool_config(quality_config)
    assert "[tool.ruff]" in toml
    assert "[tool.mypy]" in toml
    assert "fail_under = 85" in toml
    print("✓ pyproject.toml config generated")

    # Test Exercise 4: Complete pipeline
    print("\n--- Exercise 4: Complete Pipeline ---")
    cicd_config = CICDConfig()
    workflows = generate_complete_pipeline(cicd_config)

    assert "ci.yml" in workflows
    assert "docs.yml" in workflows
    assert "release.yml" in workflows
    print("✓ Complete pipeline generated")

    # Validate CI workflow
    ci = yaml.safe_load(workflows["ci.yml"])
    assert "test" in ci["jobs"]
    print("✓ CI workflow is valid YAML")

    # Validate release workflow
    release = yaml.safe_load(workflows["release.yml"])
    assert release["on"]["push"]["tags"] == ["v*"]
    print("✓ Release workflow triggers on tags")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
