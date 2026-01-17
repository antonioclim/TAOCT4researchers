#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Lab 3: Project Scaffolder
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Good software is like a well-organised laboratory — everything has its place,
and you can find what you need when you need it."

Creating a proper project structure is often the first barrier to starting a
research software project. This lab provides tools to automatically generate
professional project scaffolds that follow best practices for reproducible
research.

PREREQUISITES
─────────────
- Week 7 Lab 1: Reproducibility Toolkit
- Week 7 Lab 2: Testing and CI/CD
- Python: Familiarity with pathlib and file operations

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Generate standardised project structures automatically
2. Create boilerplate files with proper templates
3. Configure projects for testing, linting and CI/CD
4. Customise scaffolds for different research domains

ESTIMATED TIME
──────────────
- Reading: 20 minutes
- Coding: 40 minutes
- Total: 60 minutes

DEPENDENCIES
────────────
- Python 3.12+
- Standard library: pathlib, string, json

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

# Configure module-level logger
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROJECT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProjectConfig:
    """
    Configuration for project scaffolding.

    Attributes:
        name: Project name (will be converted to valid Python identifier).
        description: Brief project description.
        author: Author name.
        email: Author email.
        licence: Licence type (MIT, Apache-2.0, GPL-3.0, etc.).
        python_version: Minimum Python version.
        include_tests: Whether to include test directory and configuration.
        include_docs: Whether to include documentation structure.
        include_ci: Whether to include CI/CD configuration.
        include_docker: Whether to include Docker configuration.
        research_domain: Optional domain for specialised templates.

    Example:
        >>> config = ProjectConfig(
        ...     name="protein-analysis",
        ...     description="Tools for protein structure analysis",
        ...     author="Jane Researcher",
        ...     email="jane@university.edu",
        ...     research_domain="bioinformatics"
        ... )
    """

    name: str
    description: str = "A computational research project"
    author: str = "Researcher"
    email: str = "researcher@example.com"
    licence: str = "MIT"
    python_version: str = "3.11"
    include_tests: bool = True
    include_docs: bool = True
    include_ci: bool = True
    include_docker: bool = False
    research_domain: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalise configuration."""
        # Convert name to valid Python package name
        self.package_name = self._to_package_name(self.name)
        self.slug = self._to_slug(self.name)

    @staticmethod
    def _to_package_name(name: str) -> str:
        """Convert project name to valid Python package name."""
        # Replace hyphens and spaces with underscores
        name = re.sub(r"[-\s]+", "_", name.lower())
        # Remove invalid characters
        name = re.sub(r"[^a-z0-9_]", "", name)
        # Ensure doesn't start with number
        if name and name[0].isdigit():
            name = "_" + name
        return name or "project"

    @staticmethod
    def _to_slug(name: str) -> str:
        """Convert project name to URL-friendly slug."""
        slug = re.sub(r"[^\w\s-]", "", name.lower())
        return re.sub(r"[-\s]+", "-", slug).strip("-")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template substitution."""
        return {
            "name": self.name,
            "package_name": self.package_name,
            "slug": self.slug,
            "description": self.description,
            "author": self.author,
            "email": self.email,
            "licence": self.licence,
            "python_version": self.python_version,
            "year": datetime.now().year,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FILE TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════


README_TEMPLATE = Template("""# $name

$description

## Installation

```bash
git clone https://github.com/username/$slug.git
cd $slug
pip install -e ".[dev]"
```

## Quick Start

```python
from $package_name import main

main.run()
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/
```

## Project Structure

```
$slug/
├── src/
│   └── $package_name/
│       ├── __init__.py
│       ├── core.py
│       └── utils.py
├── tests/
│   ├── conftest.py
│   └── test_core.py
├── notebooks/
│   └── exploration.ipynb
├── data/
│   └── README.md
└── docs/
    └── README.md
```

## Licence

$licence © $year $author
""")


PYPROJECT_TEMPLATE = Template('''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "$slug"
version = "0.1.0"
description = "$description"
readme = "README.md"
requires-python = ">=$python_version"
license = {text = "$licence"}
authors = [
    {name = "$author", email = "$email"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: $python_version",
]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]

[tool.mypy]
python_version = "$python_version"
warn_return_any = true
strict = true
''')


INIT_TEMPLATE = Template('''"""
$name - $description

© $year $author
"""

__version__ = "0.1.0"
__author__ = "$author"
''')


CORE_TEMPLATE = Template('''"""
Core functionality for $name.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point."""
    logger.info("$name started")
    # Add your code here


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
''')


UTILS_TEMPLATE = Template('''"""
Utility functions for $name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load configuration from file."""
    import json

    with open(path) as f:
        return json.load(f)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path
''')


CONFTEST_TEMPLATE = Template('''"""
Pytest configuration and fixtures.
"""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory fixture."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_config() -> dict:
    """Sample configuration fixture."""
    return {
        "name": "test",
        "version": "0.1.0",
    }
''')


TEST_CORE_TEMPLATE = Template('''"""
Tests for core module.
"""

import pytest
from $package_name import core


class TestCore:
    """Tests for core functionality."""

    def test_main_runs(self) -> None:
        """Test that main function runs without error."""
        # This is a placeholder test
        # Replace with actual tests for your code
        assert True

    def test_import(self) -> None:
        """Test that module can be imported."""
        from $package_name import __version__
        assert __version__ == "0.1.0"
''')


GITIGNORE_TEMPLATE = """# Byte-compiled / optimised / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/
.eggs/

# Virtual environments
venv/
.venv/
env/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# Jupyter
.ipynb_checkpoints/

# Data (add specific patterns as needed)
*.csv
*.parquet
!data/README.md

# OS
.DS_Store
Thumbs.db
"""


GITHUB_CI_TEMPLATE = Template("""name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['$python_version', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python $${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: $${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Lint
      run: ruff check .

    - name: Type check
      run: mypy src/

    - name: Test
      run: pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
""")


DOCKERFILE_TEMPLATE = Template("""FROM python:$python_version-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "$package_name"]
""")


DATA_README_TEMPLATE = Template("""# Data Directory

This directory contains data files for $name.

## Structure

```
data/
├── raw/           # Original, immutable data
├── processed/     # Cleaned, transformed data
└── MANIFEST.json  # Data integrity checksums
```

## Data Sources

Document your data sources here.

## Integrity Verification

To verify data integrity:

```python
from $package_name.reproducibility import DataManifest

manifest = DataManifest.load("data/MANIFEST.json")
results = manifest.verify_all()
assert all(results.values()), "Data integrity check failed"
```
""")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PROJECT SCAFFOLDER
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ProjectScaffolder:
    """
    Generates complete project structure from configuration.

    Attributes:
        config: Project configuration.
        output_dir: Base directory for project creation.

    Example:
        >>> config = ProjectConfig(name="my-research")
        >>> scaffolder = ProjectScaffolder(config, Path("/projects"))
        >>> scaffolder.create()
        >>> # Creates /projects/my-research/ with full structure
    """

    config: ProjectConfig
    output_dir: Path

    def __post_init__(self) -> None:
        """Initialise derived attributes."""
        self.project_dir = self.output_dir / self.config.slug
        self.src_dir = self.project_dir / "src" / self.config.package_name
        self.tests_dir = self.project_dir / "tests"
        self.docs_dir = self.project_dir / "docs"
        self.data_dir = self.project_dir / "data"
        self.notebooks_dir = self.project_dir / "notebooks"
        self._template_vars = self.config.to_dict()

    def create(self, overwrite: bool = False) -> Path:
        """
        Create the complete project structure.

        Args:
            overwrite: If True, overwrite existing files.

        Returns:
            Path to created project directory.

        Raises:
            FileExistsError: If project exists and overwrite is False.
        """
        if self.project_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Project directory already exists: {self.project_dir}"
            )

        logger.info(f"Creating project: {self.config.name}")

        self._create_directories()
        self._create_core_files()

        if self.config.include_tests:
            self._create_test_files()

        if self.config.include_docs:
            self._create_docs_files()

        if self.config.include_ci:
            self._create_ci_files()

        if self.config.include_docker:
            self._create_docker_files()

        self._create_data_structure()
        self._create_notebooks()

        logger.info(f"Project created at: {self.project_dir}")
        return self.project_dir

    def _create_directories(self) -> None:
        """Create directory structure."""
        directories = [
            self.src_dir,
            self.tests_dir,
            self.docs_dir,
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.notebooks_dir,
            self.project_dir / "experiments",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")

    def _create_core_files(self) -> None:
        """Create core project files."""
        # README.md
        self._write_template(
            self.project_dir / "README.md",
            README_TEMPLATE,
        )

        # pyproject.toml
        self._write_template(
            self.project_dir / "pyproject.toml",
            PYPROJECT_TEMPLATE,
        )

        # .gitignore
        self._write_file(
            self.project_dir / ".gitignore",
            GITIGNORE_TEMPLATE,
        )

        # Source package
        self._write_template(
            self.src_dir / "__init__.py",
            INIT_TEMPLATE,
        )

        self._write_template(
            self.src_dir / "core.py",
            CORE_TEMPLATE,
        )

        self._write_template(
            self.src_dir / "utils.py",
            UTILS_TEMPLATE,
        )

    def _create_test_files(self) -> None:
        """Create test files."""
        self._write_file(
            self.tests_dir / "__init__.py",
            "",
        )

        self._write_template(
            self.tests_dir / "conftest.py",
            CONFTEST_TEMPLATE,
        )

        self._write_template(
            self.tests_dir / "test_core.py",
            TEST_CORE_TEMPLATE,
        )

    def _create_docs_files(self) -> None:
        """Create documentation files."""
        docs_readme = f"""# Documentation for {self.config.name}

## Contents

- [Getting Started](getting_started.md)
- [API Reference](api.md)
- [Examples](examples.md)

## Building Documentation

```bash
# With MkDocs
pip install mkdocs
mkdocs serve

# With Sphinx
pip install sphinx
sphinx-build docs/ docs/_build/
```
"""
        self._write_file(self.docs_dir / "README.md", docs_readme)

    def _create_ci_files(self) -> None:
        """Create CI/CD configuration files."""
        workflows_dir = self.project_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        self._write_template(
            workflows_dir / "ci.yml",
            GITHUB_CI_TEMPLATE,
        )

    def _create_docker_files(self) -> None:
        """Create Docker configuration files."""
        self._write_template(
            self.project_dir / "Dockerfile",
            DOCKERFILE_TEMPLATE,
        )

        compose = f"""services:
  app:
    build: .
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
"""
        self._write_file(
            self.project_dir / "docker-compose.yml",
            compose,
        )

    def _create_data_structure(self) -> None:
        """Create data directory with README."""
        self._write_template(
            self.data_dir / "README.md",
            DATA_README_TEMPLATE,
        )

        # Create empty MANIFEST.json
        manifest = {
            "files": {},
            "created_at": datetime.now().isoformat(),
            "algorithm": "sha256",
        }
        with open(self.data_dir / "MANIFEST.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _create_notebooks(self) -> None:
        """Create Jupyter notebook templates."""
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {self.config.name} - Exploration\n",
                        "\n",
                        "This notebook is for data exploration and prototyping.",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        f"from {self.config.package_name} import core",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.11.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        with open(self.notebooks_dir / "exploration.ipynb", "w") as f:
            json.dump(notebook, f, indent=2)

    def _write_template(self, path: Path, template: Template) -> None:
        """Write a template file with substitution."""
        content = template.safe_substitute(self._template_vars)
        self._write_file(path, content)

    def _write_file(self, path: Path, content: str) -> None:
        """Write content to file."""
        path.write_text(content)
        logger.debug(f"Created file: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: INTERACTIVE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


def prompt_config() -> ProjectConfig:
    """
    Interactively prompt for project configuration.

    Returns:
        Configured ProjectConfig instance.
    """
    print("\n" + "=" * 60)
    print("  PROJECT SCAFFOLDER - Interactive Configuration")
    print("=" * 60 + "\n")

    name = input("Project name: ").strip() or "my-research"
    description = (
        input("Description [A computational research project]: ").strip()
        or "A computational research project"
    )
    author = input("Author name [Researcher]: ").strip() or "Researcher"
    email = (
        input("Author email [researcher@example.com]: ").strip()
        or "researcher@example.com"
    )

    print("\nOptional features (y/n):")
    include_tests = input("  Include tests? [Y/n]: ").lower() != "n"
    include_docs = input("  Include docs? [Y/n]: ").lower() != "n"
    include_ci = input("  Include CI/CD? [Y/n]: ").lower() != "n"
    include_docker = input("  Include Docker? [y/N]: ").lower() == "y"

    return ProjectConfig(
        name=name,
        description=description,
        author=author,
        email=email,
        include_tests=include_tests,
        include_docs=include_docs,
        include_ci=include_ci,
        include_docker=include_docker,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_scaffolder() -> None:
    """Demonstrate project scaffolding."""
    logger.info("=" * 60)
    logger.info("DEMO: Project Scaffolder")
    logger.info("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ProjectConfig(
            name="demo-research-project",
            description="A demonstration of the project scaffolder",
            author="Demo Author",
            email="demo@example.com",
            include_tests=True,
            include_docs=True,
            include_ci=True,
            include_docker=True,
        )

        scaffolder = ProjectScaffolder(config, Path(tmpdir))
        project_path = scaffolder.create()

        # List created structure
        logger.info(f"\nCreated project structure at: {project_path}")
        logger.info("\nDirectory tree:")

        for path in sorted(project_path.rglob("*")):
            relative = path.relative_to(project_path)
            indent = "  " * len(relative.parts)
            if path.is_dir():
                logger.info(f"{indent}{path.name}/")
            else:
                logger.info(f"{indent}{path.name}")


def run_all_demos() -> None:
    """Execute all demonstrations."""
    demo_scaffolder()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Week 7 Lab 3: Project Scaffolder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python lab_7_03_project_scaffolder.py --interactive

  # Create project with defaults
  python lab_7_03_project_scaffolder.py --name "my-research" --output ./projects

  # Create with all options
  python lab_7_03_project_scaffolder.py \\
      --name "protein-analysis" \\
      --description "Tools for protein structure analysis" \\
      --author "Jane Researcher" \\
      --email "jane@university.edu" \\
      --docker \\
      --output ./projects

  # Run demo
  python lab_7_03_project_scaffolder.py --demo
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Interactive configuration mode",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Project name",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="A computational research project",
        help="Project description",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="Researcher",
        help="Author name",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="researcher@example.com",
        help="Author email",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=".",
        help="Output directory",
    )
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Include Docker configuration",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip test structure",
    )
    parser.add_argument(
        "--no-docs",
        action="store_true",
        help="Skip documentation structure",
    )
    parser.add_argument(
        "--no-ci",
        action="store_true",
        help="Skip CI/CD configuration",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.demo:
        run_all_demos()
    elif args.interactive:
        config = prompt_config()
        output_dir = Path(
            input(f"\nOutput directory [{args.output}]: ").strip() or args.output
        )
        scaffolder = ProjectScaffolder(config, output_dir)
        scaffolder.create()
        print(f"\n✓ Project created at: {scaffolder.project_dir}")
    elif args.name:
        config = ProjectConfig(
            name=args.name,
            description=args.description,
            author=args.author,
            email=args.email,
            include_tests=not args.no_tests,
            include_docs=not args.no_docs,
            include_ci=not args.no_ci,
            include_docker=args.docker,
        )
        scaffolder = ProjectScaffolder(config, Path(args.output))
        scaffolder.create()
        logger.info(f"Project created at: {scaffolder.project_dir}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
