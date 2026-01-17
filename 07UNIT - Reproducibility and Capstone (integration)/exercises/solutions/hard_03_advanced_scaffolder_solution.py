#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Hard Exercise 3 — Advanced Project Scaffolder
═══════════════════════════════════════════════════════════════════════════════

This solution implements an advanced project scaffolding system including:
1. Custom template engine with conditionals, loops and filters
2. Plugin architecture for extensibility
3. Project validation framework
4. Integrated scaffolder with preview capability

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS AND DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TemplateRenderer(Protocol):
    """Protocol for template rendering engines."""

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render a template with the given context."""
        ...


class FileGenerator(Protocol):
    """Protocol for file generators."""

    def generate(self, spec: "FileSpec", context: dict[str, Any]) -> str:
        """Generate file content from a specification."""
        ...


@dataclass
class FileSpec:
    """Specification for a file to generate.

    Attributes:
        path: Relative path for the file.
        template: Template string for content.
        condition: Optional condition for generation.
        mode: File mode (e.g. 0o644).
    """

    path: str
    template: str
    condition: str | None = None
    mode: int = 0o644


@dataclass
class DirectorySpec:
    """Specification for a directory to create.

    Attributes:
        path: Relative path for the directory.
        condition: Optional condition for creation.
    """

    path: str
    condition: str | None = None


@dataclass
class ProjectTemplate:
    """Complete project template specification.

    Attributes:
        name: Template name.
        description: Template description.
        directories: List of directory specifications.
        files: List of file specifications.
        variables: Default variable values.
    """

    name: str
    description: str
    directories: list[DirectorySpec] = field(default_factory=list)
    files: list[FileSpec] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: SIMPLE TEMPLATE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTemplateEngine:
    """Custom template engine with conditionals, loops and filters.

    Supports:
    - Variable substitution: ${variable}
    - Conditionals: {% if condition %}...{% endif %}
    - Loops: {% for item in items %}...{% endfor %}
    - Comments: {# comment #}
    - Filters: ${variable|filter}

    Available filters:
    - upper: Convert to uppercase
    - lower: Convert to lowercase
    - title: Convert to title case
    - snake_case: Convert to snake_case
    - kebab_case: Convert to kebab-case
    - strip: Remove whitespace

    Example:
        >>> engine = SimpleTemplateEngine()
        >>> template = "Hello, ${name|title}!"
        >>> engine.render(template, {"name": "world"})
        'Hello, World!'
    """

    def __init__(self) -> None:
        """Initialise the template engine with default filters."""
        self.filters: dict[str, Callable[[str], str]] = {
            "upper": str.upper,
            "lower": str.lower,
            "title": str.title,
            "strip": str.strip,
            "snake_case": self._to_snake_case,
            "kebab_case": self._to_kebab_case,
        }

    def _to_snake_case(self, value: str) -> str:
        """Convert string to snake_case.

        Args:
            value: Input string.

        Returns:
            snake_case version of the string.
        """
        # Handle camelCase and PascalCase
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Replace spaces and hyphens with underscores
        s3 = re.sub(r"[\s\-]+", "_", s2)
        return s3.lower()

    def _to_kebab_case(self, value: str) -> str:
        """Convert string to kebab-case.

        Args:
            value: Input string.

        Returns:
            kebab-case version of the string.
        """
        snake = self._to_snake_case(value)
        return snake.replace("_", "-")

    def register_filter(self, name: str, func: Callable[[str], str]) -> None:
        """Register a custom filter.

        Args:
            name: Filter name.
            func: Filter function.
        """
        self.filters[name] = func

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template: Template string.
            context: Dictionary of variables.

        Returns:
            Rendered string.
        """
        result = template

        # Remove comments first
        result = self._remove_comments(result)

        # Process conditionals
        result = self._process_conditionals(result, context)

        # Process loops
        result = self._process_loops(result, context)

        # Substitute variables (with filters)
        result = self._substitute_variables(result, context)

        return result

    def _remove_comments(self, template: str) -> str:
        """Remove comment blocks from template.

        Args:
            template: Template string.

        Returns:
            Template with comments removed.
        """
        return re.sub(r"\{#.*?#\}", "", template, flags=re.DOTALL)

    def _process_conditionals(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """Process conditional blocks.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Template with conditionals resolved.
        """
        # Pattern for {% if condition %}...{% endif %}
        # Also supports {% if condition %}...{% else %}...{% endif %}
        pattern = r"\{%\s*if\s+(.+?)\s*%\}(.*?)(?:\{%\s*else\s*%\}(.*?))?\{%\s*endif\s*%\}"

        def replace_conditional(match: re.Match) -> str:
            condition = match.group(1).strip()
            if_content = match.group(2)
            else_content = match.group(3) or ""

            # Evaluate condition
            try:
                # Simple evaluation: check if variable is truthy
                # Support: variable, not variable, variable == value
                if condition.startswith("not "):
                    var_name = condition[4:].strip()
                    result = not self._get_value(var_name, context)
                elif "==" in condition:
                    left, right = condition.split("==")
                    left_val = self._get_value(left.strip(), context)
                    right_val = right.strip().strip("'\"")
                    result = str(left_val) == right_val
                elif "!=" in condition:
                    left, right = condition.split("!=")
                    left_val = self._get_value(left.strip(), context)
                    right_val = right.strip().strip("'\"")
                    result = str(left_val) != right_val
                else:
                    result = bool(self._get_value(condition, context))

                return if_content if result else else_content
            except (KeyError, ValueError):
                return else_content

        # Process from innermost to outermost
        max_iterations = 10
        for _ in range(max_iterations):
            new_template = re.sub(pattern, replace_conditional, template, flags=re.DOTALL)
            if new_template == template:
                break
            template = new_template

        return template

    def _process_loops(self, template: str, context: dict[str, Any]) -> str:
        """Process loop blocks.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Template with loops expanded.
        """
        # Pattern for {% for item in items %}...{% endfor %}
        pattern = r"\{%\s*for\s+(\w+)\s+in\s+(\w+)\s*%\}(.*?)\{%\s*endfor\s*%\}"

        def replace_loop(match: re.Match) -> str:
            item_var = match.group(1)
            list_var = match.group(2)
            loop_content = match.group(3)

            items = self._get_value(list_var, context)
            if not isinstance(items, (list, tuple)):
                return ""

            result_parts = []
            for i, item in enumerate(items):
                # Create loop context
                loop_context = {
                    **context,
                    item_var: item,
                    "loop": {
                        "index": i + 1,
                        "index0": i,
                        "first": i == 0,
                        "last": i == len(items) - 1,
                    },
                }
                # Recursively process the loop content
                processed = self._substitute_variables(loop_content, loop_context)
                processed = self._process_conditionals(processed, loop_context)
                result_parts.append(processed)

            return "".join(result_parts)

        # Process loops (potentially nested)
        max_iterations = 10
        for _ in range(max_iterations):
            new_template = re.sub(pattern, replace_loop, template, flags=re.DOTALL)
            if new_template == template:
                break
            template = new_template

        return template

    def _substitute_variables(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """Substitute variables in template.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Template with variables substituted.
        """
        # Pattern for ${variable} or ${variable|filter}
        pattern = r"\$\{([^}|]+)(?:\|([^}]+))?\}"

        def replace_variable(match: re.Match) -> str:
            var_path = match.group(1).strip()
            filter_chain = match.group(2)

            try:
                value = self._get_value(var_path, context)
                value_str = str(value)

                # Apply filters
                if filter_chain:
                    for filter_name in filter_chain.split("|"):
                        filter_name = filter_name.strip()
                        if filter_name in self.filters:
                            value_str = self.filters[filter_name](value_str)

                return value_str
            except (KeyError, TypeError):
                return match.group(0)  # Return original if not found

        return re.sub(pattern, replace_variable, template)

    def _get_value(self, path: str, context: dict[str, Any]) -> Any:
        """Get a value from context using dot notation.

        Args:
            path: Dot-separated path (e.g. "user.name").
            context: Variable context.

        Returns:
            Value at the path.

        Raises:
            KeyError: If path not found.
        """
        parts = path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                raise KeyError(f"Cannot resolve '{part}' in path '{path}'")

        return value


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: PLUGIN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ScaffolderPlugin(ABC):
    """Abstract base class for scaffolder plugins.

    Plugins can add directories, files and post-generation hooks
    to the scaffolding process.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description."""
        ...

    @abstractmethod
    def get_directories(self, context: dict[str, Any]) -> list[DirectorySpec]:
        """Get directories to create.

        Args:
            context: Project context.

        Returns:
            List of directory specifications.
        """
        ...

    @abstractmethod
    def get_files(self, context: dict[str, Any]) -> list[FileSpec]:
        """Get files to create.

        Args:
            context: Project context.

        Returns:
            List of file specifications.
        """
        ...

    def post_generate(self, project_path: Path, context: dict[str, Any]) -> None:
        """Hook called after generation completes.

        Args:
            project_path: Path to generated project.
            context: Project context.
        """
        pass


class PythonPackagePlugin(ScaffolderPlugin):
    """Plugin for Python package structure.

    Adds standard Python package files and directories including
    __init__.py files, setup configuration and type stubs.
    """

    @property
    def name(self) -> str:
        return "python_package"

    @property
    def description(self) -> str:
        return "Adds Python package structure with __init__.py files"

    def get_directories(self, context: dict[str, Any]) -> list[DirectorySpec]:
        """Get Python package directories."""
        package_name = context.get("package_name", "package")
        return [
            DirectorySpec(path=f"src/{package_name}"),
            DirectorySpec(path=f"src/{package_name}/utils"),
        ]

    def get_files(self, context: dict[str, Any]) -> list[FileSpec]:
        """Get Python package files."""
        package_name = context.get("package_name", "package")
        version = context.get("version", "0.1.0")

        init_template = '''"""
${project_name}
${"=" * len(project_name)}

${description}
"""

__version__ = "${version}"
__author__ = "${author}"
'''

        utils_init = '''"""Utility modules for ${project_name}."""
'''

        py_typed = "# Marker file for PEP 561\n"

        return [
            FileSpec(
                path=f"src/{package_name}/__init__.py",
                template=init_template,
            ),
            FileSpec(
                path=f"src/{package_name}/utils/__init__.py",
                template=utils_init,
            ),
            FileSpec(
                path=f"src/{package_name}/py.typed",
                template=py_typed,
            ),
        ]


class DocumentationPlugin(ScaffolderPlugin):
    """Plugin for documentation structure.

    Adds documentation files and directories for Sphinx or MkDocs.
    """

    @property
    def name(self) -> str:
        return "documentation"

    @property
    def description(self) -> str:
        return "Adds documentation structure (Sphinx/MkDocs compatible)"

    def get_directories(self, context: dict[str, Any]) -> list[DirectorySpec]:
        """Get documentation directories."""
        return [
            DirectorySpec(path="docs"),
            DirectorySpec(path="docs/api"),
            DirectorySpec(path="docs/guides"),
        ]

    def get_files(self, context: dict[str, Any]) -> list[FileSpec]:
        """Get documentation files."""
        index_template = """# ${project_name}

${description}

## Installation

```bash
pip install ${package_name}
```

## Quick Start

```python
import ${package_name}

# Your code here
```

## Contents

- [API Reference](api/index.md)
- [User Guides](guides/index.md)
"""

        api_index = """# API Reference

This section contains the API documentation for ${project_name}.

## Modules

{% for module in modules %}
- [${module}](${module}.md)
{% endfor %}
"""

        guides_index = """# User Guides

This section contains guides for using ${project_name}.

## Available Guides

- [Getting Started](getting_started.md)
- [Configuration](configuration.md)
"""

        mkdocs_config = """site_name: ${project_name}
site_description: ${description}
site_author: ${author}

theme:
  name: material
  palette:
    primary: indigo
    accent: indigo

nav:
  - Home: index.md
  - API Reference: api/index.md
  - Guides: guides/index.md

plugins:
  - search
  - mkdocstrings

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences
"""

        return [
            FileSpec(path="docs/index.md", template=index_template),
            FileSpec(path="docs/api/index.md", template=api_index),
            FileSpec(path="docs/guides/index.md", template=guides_index),
            FileSpec(path="mkdocs.yml", template=mkdocs_config),
        ]


class CICDPlugin(ScaffolderPlugin):
    """Plugin for CI/CD configuration.

    Adds GitHub Actions workflows for testing, linting and deployment.
    """

    @property
    def name(self) -> str:
        return "cicd"

    @property
    def description(self) -> str:
        return "Adds GitHub Actions CI/CD workflows"

    def get_directories(self, context: dict[str, Any]) -> list[DirectorySpec]:
        """Get CI/CD directories."""
        return [
            DirectorySpec(path=".github"),
            DirectorySpec(path=".github/workflows"),
        ]

    def get_files(self, context: dict[str, Any]) -> list[FileSpec]:
        """Get CI/CD workflow files."""
        ci_workflow = """name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

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

      - name: Run linting
        run: |
          ruff check src tests
          ruff format --check src tests

      - name: Run type checking
        run: mypy src

      - name: Run tests
        run: pytest --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
"""

        release_workflow = """name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install build tools
        run: pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: $${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
"""

        return [
            FileSpec(
                path=".github/workflows/ci.yml",
                template=ci_workflow,
            ),
            FileSpec(
                path=".github/workflows/release.yml",
                template=release_workflow,
                condition="include_release",
            ),
        ]


class PluginRegistry:
    """Registry for scaffolder plugins.

    Manages registration, enabling and retrieval of plugins.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(PythonPackagePlugin())
        >>> registry.enable("python_package")
        >>> plugins = registry.get_enabled_plugins()
    """

    def __init__(self) -> None:
        """Initialise an empty plugin registry."""
        self._plugins: dict[str, ScaffolderPlugin] = {}
        self._enabled: set[str] = set()

    def register(self, plugin: ScaffolderPlugin) -> None:
        """Register a plugin.

        Args:
            plugin: Plugin to register.
        """
        self._plugins[plugin.name] = plugin
        logger.debug(f"Registered plugin: {plugin.name}")

    def enable(self, name: str) -> None:
        """Enable a registered plugin.

        Args:
            name: Plugin name.

        Raises:
            KeyError: If plugin not registered.
        """
        if name not in self._plugins:
            raise KeyError(f"Plugin '{name}' not registered")
        self._enabled.add(name)
        logger.debug(f"Enabled plugin: {name}")

    def disable(self, name: str) -> None:
        """Disable a plugin.

        Args:
            name: Plugin name.
        """
        self._enabled.discard(name)

    def get_enabled_plugins(self) -> list[ScaffolderPlugin]:
        """Get all enabled plugins.

        Returns:
            List of enabled plugin instances.
        """
        return [self._plugins[name] for name in self._enabled if name in self._plugins]

    def list_plugins(self) -> list[tuple[str, str, bool]]:
        """List all registered plugins.

        Returns:
            List of (name, description, enabled) tuples.
        """
        return [
            (name, plugin.description, name in self._enabled)
            for name, plugin in self._plugins.items()
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: PROJECT VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationRule:
    """A validation rule for projects.

    Attributes:
        name: Rule name.
        description: What the rule checks.
        validator: Function that returns (passed, message).
    """

    name: str
    description: str
    validator: Callable[[Path], tuple[bool, str]]


@dataclass
class ValidationResult:
    """Result of project validation.

    Attributes:
        rule_name: Name of the rule.
        passed: Whether the rule passed.
        message: Description or error message.
    """

    rule_name: str
    passed: bool
    message: str


class ProjectValidator:
    """Validator for generated projects.

    Checks that generated projects meet quality standards
    including required files, correct structure and content.

    Example:
        >>> validator = ProjectValidator()
        >>> results = validator.validate(Path("./my_project"))
        >>> all(r.passed for r in results)
        True
    """

    def __init__(self) -> None:
        """Initialise validator with default rules."""
        self.rules: list[ValidationRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        self.add_rule(ValidationRule(
            name="readme_exists",
            description="README.md file exists",
            validator=self.validate_readme,
        ))
        self.add_rule(ValidationRule(
            name="pyproject_valid",
            description="pyproject.toml is valid",
            validator=self.validate_pyproject,
        ))
        self.add_rule(ValidationRule(
            name="tests_exist",
            description="Tests directory exists with test files",
            validator=self.validate_tests_exist,
        ))

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule.

        Args:
            rule: Rule to add.
        """
        self.rules.append(rule)

    def validate(self, project_path: Path) -> list[ValidationResult]:
        """Validate a project against all rules.

        Args:
            project_path: Path to the project.

        Returns:
            List of validation results.
        """
        results = []
        for rule in self.rules:
            try:
                passed, message = rule.validator(project_path)
            except Exception as e:
                passed = False
                message = f"Error during validation: {e}"

            results.append(ValidationResult(
                rule_name=rule.name,
                passed=passed,
                message=message,
            ))

            status = "✓" if passed else "✗"
            logger.info(f"  {status} {rule.name}: {message}")

        return results

    def validate_readme(self, project_path: Path) -> tuple[bool, str]:
        """Validate README.md exists and has content.

        Args:
            project_path: Path to the project.

        Returns:
            Tuple of (passed, message).
        """
        readme = project_path / "README.md"
        if not readme.exists():
            return False, "README.md not found"

        content = readme.read_text()
        if len(content) < 100:
            return False, "README.md is too short (< 100 characters)"

        # Check for required sections
        required_sections = ["#", "Installation", "Usage"]
        missing = [s for s in required_sections if s not in content]
        if missing:
            return False, f"README.md missing sections: {missing}"

        return True, "README.md is valid"

    def validate_pyproject(self, project_path: Path) -> tuple[bool, str]:
        """Validate pyproject.toml exists and is valid.

        Args:
            project_path: Path to the project.

        Returns:
            Tuple of (passed, message).
        """
        pyproject = project_path / "pyproject.toml"
        if not pyproject.exists():
            return False, "pyproject.toml not found"

        content = pyproject.read_text()

        # Check for required sections
        required = ["[project]", "name", "version"]
        missing = [r for r in required if r not in content]
        if missing:
            return False, f"pyproject.toml missing: {missing}"

        # Try to parse TOML
        try:
            import tomllib
            tomllib.loads(content)
        except ImportError:
            # Python < 3.11, skip TOML validation
            pass
        except Exception as e:
            return False, f"pyproject.toml parse error: {e}"

        return True, "pyproject.toml is valid"

    def validate_tests_exist(self, project_path: Path) -> tuple[bool, str]:
        """Validate tests directory exists with test files.

        Args:
            project_path: Path to the project.

        Returns:
            Tuple of (passed, message).
        """
        tests_dir = project_path / "tests"
        if not tests_dir.exists():
            return False, "tests/ directory not found"

        if not tests_dir.is_dir():
            return False, "tests is not a directory"

        test_files = list(tests_dir.glob("test_*.py"))
        if not test_files:
            return False, "No test files (test_*.py) found"

        return True, f"Found {len(test_files)} test file(s)"


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: ADVANCED SCAFFOLDER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffolderConfig:
    """Configuration for project scaffolding.

    Attributes:
        project_name: Name of the project.
        package_name: Python package name.
        description: Project description.
        author: Author name.
        author_email: Author email.
        version: Initial version.
        python_version: Minimum Python version.
        include_docs: Whether to include documentation.
        include_cicd: Whether to include CI/CD.
        include_release: Whether to include release workflow.
        extra_context: Additional context variables.
    """

    project_name: str
    package_name: str = ""
    description: str = "A Python project"
    author: str = "Author"
    author_email: str = "author@example.com"
    version: str = "0.1.0"
    python_version: str = "3.10"
    include_docs: bool = True
    include_cicd: bool = True
    include_release: bool = False
    extra_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default package name from project name."""
        if not self.package_name:
            # Convert project name to valid Python package name
            self.package_name = re.sub(r"[^a-zA-Z0-9]+", "_", self.project_name).lower()


# Templates for core files

README_TEMPLATE = """# ${project_name}

${description}

## Installation

```bash
pip install ${package_name}
```

## Usage

```python
import ${package_name}

# Your code here
```

{% if include_docs %}
## Documentation

Documentation is available at [docs/](docs/).
{% endif %}

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

## Licence

MIT
"""

PYPROJECT_TEMPLATE = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "${package_name}"
version = "${version}"
description = "${description}"
readme = "README.md"
requires-python = ">=${python_version}"
license = "MIT"
authors = [
    { name = "${author}", email = "${author_email}" },
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = []

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]
{% if include_docs %}
docs = [
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.24",
]
{% endif %}

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "C4"]

[tool.mypy]
python_version = "${python_version}"
strict = true
"""

GITIGNORE_TEMPLATE = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# Docs
site/
"""

CONFTEST_TEMPLATE = '''"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_data() -> list[int]:
    """Provide sample data for tests."""
    return [1, 2, 3, 4, 5]
'''

TEST_TEMPLATE = '''"""Tests for ${package_name}."""

import ${package_name}


def test_version() -> None:
    """Test version is set."""
    assert hasattr(${package_name}, "__version__")
    assert ${package_name}.__version__ == "${version}"
'''


class AdvancedScaffolder:
    """Advanced project scaffolder with plugins and validation.

    Combines template engine, plugins and validation to create
    complete project structures.

    Example:
        >>> config = ScaffolderConfig(project_name="my_project")
        >>> scaffolder = AdvancedScaffolder()
        >>> scaffolder.scaffold(Path("./output"), config)
    """

    def __init__(self) -> None:
        """Initialise scaffolder with default plugins."""
        self.engine = SimpleTemplateEngine()
        self.validator = ProjectValidator()
        self.registry = PluginRegistry()

        # Register default plugins
        self.registry.register(PythonPackagePlugin())
        self.registry.register(DocumentationPlugin())
        self.registry.register(CICDPlugin())

    def scaffold(
        self,
        output_path: Path,
        config: ScaffolderConfig,
        validate: bool = True,
    ) -> bool:
        """Generate a complete project structure.

        Args:
            output_path: Where to create the project.
            config: Project configuration.
            validate: Whether to validate after generation.

        Returns:
            True if successful (and validation passed if enabled).
        """
        logger.info(f"Scaffolding project '{config.project_name}' at {output_path}")

        # Build context
        context = self._build_context(config)

        # Enable plugins based on config
        if config.include_docs:
            self.registry.enable("documentation")
        if config.include_cicd:
            self.registry.enable("cicd")
        self.registry.enable("python_package")

        # Collect all specs
        directories, files = self._collect_specs(context)

        # Create project directory
        project_path = output_path / config.project_name
        if project_path.exists():
            logger.warning(f"Removing existing directory: {project_path}")
            shutil.rmtree(project_path)
        project_path.mkdir(parents=True)

        # Create directories
        for dir_spec in directories:
            if self._should_include(dir_spec.condition, context):
                dir_path = project_path / dir_spec.path
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_spec.path}")

        # Create files
        for file_spec in files:
            if self._should_include(file_spec.condition, context):
                file_path = project_path / file_spec.path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                content = self.engine.render(file_spec.template, context)
                file_path.write_text(content)
                logger.debug(f"Created file: {file_spec.path}")

        # Run plugin post-generation hooks
        for plugin in self.registry.get_enabled_plugins():
            plugin.post_generate(project_path, context)

        # Validate if requested
        if validate:
            logger.info("Validating generated project...")
            results = self.validator.validate(project_path)
            passed = all(r.passed for r in results)
            if passed:
                logger.info("✓ All validations passed")
            else:
                logger.warning("✗ Some validations failed")
            return passed

        return True

    def preview(self, config: ScaffolderConfig) -> str:
        """Preview what would be generated without creating files.

        Args:
            config: Project configuration.

        Returns:
            String representation of the project structure.
        """
        context = self._build_context(config)

        # Enable plugins based on config
        if config.include_docs:
            self.registry.enable("documentation")
        if config.include_cicd:
            self.registry.enable("cicd")
        self.registry.enable("python_package")

        directories, files = self._collect_specs(context)

        lines = [
            f"Project: {config.project_name}",
            "=" * 50,
            "",
            "Directories:",
        ]

        for dir_spec in sorted(directories, key=lambda d: d.path):
            if self._should_include(dir_spec.condition, context):
                lines.append(f"  📁 {dir_spec.path}/")

        lines.extend(["", "Files:"])

        for file_spec in sorted(files, key=lambda f: f.path):
            if self._should_include(file_spec.condition, context):
                lines.append(f"  📄 {file_spec.path}")

        lines.extend([
            "",
            "Enabled plugins:",
        ])
        for plugin in self.registry.get_enabled_plugins():
            lines.append(f"  • {plugin.name}: {plugin.description}")

        return "\n".join(lines)

    def _build_context(self, config: ScaffolderConfig) -> dict[str, Any]:
        """Build the template context from configuration.

        Args:
            config: Project configuration.

        Returns:
            Context dictionary.
        """
        context = {
            "project_name": config.project_name,
            "package_name": config.package_name,
            "description": config.description,
            "author": config.author,
            "author_email": config.author_email,
            "version": config.version,
            "python_version": config.python_version,
            "include_docs": config.include_docs,
            "include_cicd": config.include_cicd,
            "include_release": config.include_release,
            "modules": [config.package_name],  # For documentation
        }
        context.update(config.extra_context)
        return context

    def _collect_specs(
        self,
        context: dict[str, Any],
    ) -> tuple[list[DirectorySpec], list[FileSpec]]:
        """Collect all directory and file specs from plugins and core.

        Args:
            context: Template context.

        Returns:
            Tuple of (directories, files).
        """
        directories: list[DirectorySpec] = [
            DirectorySpec(path="src"),
            DirectorySpec(path="tests"),
        ]

        files: list[FileSpec] = [
            FileSpec(path="README.md", template=README_TEMPLATE),
            FileSpec(path="pyproject.toml", template=PYPROJECT_TEMPLATE),
            FileSpec(path=".gitignore", template=GITIGNORE_TEMPLATE),
            FileSpec(path="tests/__init__.py", template=""),
            FileSpec(path="tests/conftest.py", template=CONFTEST_TEMPLATE),
            FileSpec(
                path=f"tests/test_{context['package_name']}.py",
                template=TEST_TEMPLATE,
            ),
        ]

        # Add from plugins
        for plugin in self.registry.get_enabled_plugins():
            directories.extend(plugin.get_directories(context))
            files.extend(plugin.get_files(context))

        return directories, files

    def _should_include(
        self,
        condition: str | None,
        context: dict[str, Any],
    ) -> bool:
        """Check if an item should be included based on condition.

        Args:
            condition: Condition string or None.
            context: Template context.

        Returns:
            True if item should be included.
        """
        if condition is None:
            return True

        # Simple condition evaluation
        if condition.startswith("not "):
            var_name = condition[4:].strip()
            return not context.get(var_name, False)
        else:
            return bool(context.get(condition, False))


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_template_engine() -> None:
    """Demonstrate the template engine."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING TEMPLATE ENGINE")
    logger.info("=" * 60)

    engine = SimpleTemplateEngine()

    # Simple variable substitution
    template1 = "Hello, ${name}!"
    result1 = engine.render(template1, {"name": "World"})
    logger.info(f"Variable substitution: {result1}")

    # Filters
    template2 = "Project: ${name|upper}, Package: ${name|snake_case}"
    result2 = engine.render(template2, {"name": "My Project"})
    logger.info(f"With filters: {result2}")

    # Conditionals
    template3 = """
{% if include_tests %}
Running tests...
{% else %}
Skipping tests.
{% endif %}
"""
    result3a = engine.render(template3, {"include_tests": True})
    result3b = engine.render(template3, {"include_tests": False})
    logger.info(f"Conditional (True): {result3a.strip()}")
    logger.info(f"Conditional (False): {result3b.strip()}")

    # Loops
    template4 = """
Dependencies:
{% for dep in dependencies %}
- ${dep}
{% endfor %}
"""
    result4 = engine.render(template4, {"dependencies": ["numpy", "pandas", "scipy"]})
    logger.info(f"Loop result:\n{result4}")


def demonstrate_scaffolding() -> None:
    """Demonstrate the full scaffolding process."""
    import tempfile

    logger.info("=" * 60)
    logger.info("DEMONSTRATING PROJECT SCAFFOLDING")
    logger.info("=" * 60)

    scaffolder = AdvancedScaffolder()

    config = ScaffolderConfig(
        project_name="example_research_project",
        description="An example research project",
        author="Researcher",
        author_email="researcher@university.edu",
        include_docs=True,
        include_cicd=True,
    )

    # Preview
    preview = scaffolder.preview(config)
    print("\nProject Preview:")
    print(preview)

    # Create in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        success = scaffolder.scaffold(Path(tmpdir), config, validate=True)
        logger.info(f"Scaffolding {'succeeded' if success else 'failed'}")

        # Show created structure
        project_path = Path(tmpdir) / config.project_name
        logger.info("\nCreated files:")
        for file in sorted(project_path.rglob("*")):
            if file.is_file():
                rel_path = file.relative_to(project_path)
                logger.info(f"  {rel_path}")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demonstrate_template_engine()
    print()
    demonstrate_scaffolding()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Advanced Project Scaffolder - Solution"
    )
    parser.add_argument(
        "--demo",
        choices=["template", "scaffold", "all"],
        help="Run specific demonstration",
    )
    parser.add_argument(
        "--scaffold",
        metavar="NAME",
        help="Scaffold a new project with given name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Output directory for scaffolding",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.scaffold:
        scaffolder = AdvancedScaffolder()
        config = ScaffolderConfig(project_name=args.scaffold)
        scaffolder.scaffold(args.output, config)
    elif args.demo == "template":
        demonstrate_template_engine()
    elif args.demo == "scaffold":
        demonstrate_scaffolding()
    elif args.demo == "all":
        run_all_demos()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
