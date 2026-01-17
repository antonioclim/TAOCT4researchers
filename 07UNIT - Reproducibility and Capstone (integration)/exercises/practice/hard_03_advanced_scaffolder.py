#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Hard 03 - Advanced Project Scaffolder
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Building on the basic scaffolder from the lab, this exercise challenges you
to create an advanced project scaffolder with plugin support, templates and
validation. This mirrors tools like Cookiecutter and Yeoman.

PREREQUISITES
─────────────
- Completed Lab 7.3: Project Scaffolder
- Understanding of design patterns (Factory, Strategy)
- Familiarity with template engines

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Design extensible scaffolding systems
2. Implement template-based code generation
3. Create validation frameworks for project structure

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 120 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import re
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from string import Template
from typing import Any
from typing import Protocol


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ABSTRACTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TemplateRenderer(Protocol):
    """Protocol for template rendering."""

    def render(self, template: str, context: dict[str, Any]) -> str:
        """Render a template with context."""
        ...


class FileGenerator(Protocol):
    """Protocol for generating files."""

    def generate(self, path: Path, content: str) -> None:
        """Generate a file with content."""
        ...


@dataclass
class FileSpec:
    """Specification for a file to generate."""

    path: str  # Relative path, may contain {variables}
    template: str
    condition: str | None = None  # Condition for including file


@dataclass
class DirectorySpec:
    """Specification for a directory structure."""

    path: str
    condition: str | None = None


@dataclass
class ProjectTemplate:
    """Complete project template specification."""

    name: str
    description: str
    variables: dict[str, Any]
    directories: list[DirectorySpec]
    files: list[FileSpec]
    post_hooks: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Template Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleTemplateEngine:
    """
    TODO: Implement a template engine with conditionals and loops.

    Support the following syntax:
    - ${variable} - Variable substitution
    - {% if condition %}...{% endif %} - Conditionals
    - {% for item in items %}...{% endfor %} - Loops
    - {# comment #} - Comments (removed from output)

    Example template:
        # ${project_name}

        {% if include_tests %}
        ## Testing
        Run tests with `pytest`.
        {% endif %}

        ## Features
        {% for feature in features %}
        - ${feature}
        {% endfor %}
    """

    def __init__(self) -> None:
        """Initialise template engine."""
        self._filters: dict[str, Any] = {
            "upper": str.upper,
            "lower": str.lower,
            "title": str.title,
            "snake_case": self._to_snake_case,
            "kebab_case": self._to_kebab_case,
        }

    def _to_snake_case(self, s: str) -> str:
        """Convert string to snake_case."""
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
        return s.replace("-", "_").lower()

    def _to_kebab_case(self, s: str) -> str:
        """Convert string to kebab-case."""
        return self._to_snake_case(s).replace("_", "-")

    def add_filter(self, name: str, func: Any) -> None:
        """Add a custom filter."""
        self._filters[name] = func

    def render(self, template: str, context: dict[str, Any]) -> str:
        """
        TODO: Render template with context.

        Args:
            template: Template string.
            context: Variable context.

        Returns:
            Rendered string.
        """
        # TODO: Implement
        # 1. Remove comments
        # 2. Process conditionals
        # 3. Process loops
        # 4. Substitute variables
        pass

    def _process_conditionals(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """
        TODO: Process {% if %}...{% endif %} blocks.

        Handle:
        - Simple conditions: {% if variable %}
        - Negation: {% if not variable %}
        - Comparison: {% if variable == "value" %}
        """
        # TODO: Implement
        pass

    def _process_loops(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """
        TODO: Process {% for %}...{% endfor %} blocks.

        Handle:
        - List iteration: {% for item in items %}
        - Dict iteration: {% for key, value in dict.items() %}
        """
        # TODO: Implement
        pass

    def _substitute_variables(
        self,
        template: str,
        context: dict[str, Any],
    ) -> str:
        """
        TODO: Substitute ${variable} and ${variable|filter}.

        Handle:
        - Simple variables: ${name}
        - Filtered variables: ${name|upper}
        - Nested access: ${config.name}
        """
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Plugin System
# ═══════════════════════════════════════════════════════════════════════════════

class ScaffolderPlugin(ABC):
    """Base class for scaffolder plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        ...

    @abstractmethod
    def get_files(self, config: dict[str, Any]) -> list[FileSpec]:
        """Get files to generate."""
        ...

    @abstractmethod
    def get_directories(self, config: dict[str, Any]) -> list[DirectorySpec]:
        """Get directories to create."""
        ...

    def post_generate(self, project_path: Path, config: dict[str, Any]) -> None:
        """Hook called after generation."""
        pass


class PythonPackagePlugin(ScaffolderPlugin):
    """
    TODO: Implement plugin for Python package structure.

    Generates:
    - src/{package_name}/__init__.py
    - src/{package_name}/core.py
    - src/{package_name}/utils.py
    - tests/__init__.py
    - tests/conftest.py
    - tests/test_core.py
    """

    @property
    def name(self) -> str:
        return "python_package"

    def get_files(self, config: dict[str, Any]) -> list[FileSpec]:
        """TODO: Implement."""
        pass

    def get_directories(self, config: dict[str, Any]) -> list[DirectorySpec]:
        """TODO: Implement."""
        pass


class DocumentationPlugin(ScaffolderPlugin):
    """
    TODO: Implement plugin for documentation structure.

    Generates:
    - docs/index.md
    - docs/installation.md
    - docs/usage.md
    - docs/api.md
    - mkdocs.yml or conf.py (depending on config)
    """

    @property
    def name(self) -> str:
        return "documentation"

    def get_files(self, config: dict[str, Any]) -> list[FileSpec]:
        """TODO: Implement."""
        pass

    def get_directories(self, config: dict[str, Any]) -> list[DirectorySpec]:
        """TODO: Implement."""
        pass


class CICDPlugin(ScaffolderPlugin):
    """
    TODO: Implement plugin for CI/CD configuration.

    Generates:
    - .github/workflows/ci.yml
    - .github/workflows/release.yml (optional)
    - .pre-commit-config.yaml (optional)
    """

    @property
    def name(self) -> str:
        return "cicd"

    def get_files(self, config: dict[str, Any]) -> list[FileSpec]:
        """TODO: Implement."""
        pass

    def get_directories(self, config: dict[str, Any]) -> list[DirectorySpec]:
        """TODO: Implement."""
        pass


class PluginRegistry:
    """
    TODO: Implement plugin registry.

    Features:
    - Register plugins by name
    - Enable/disable plugins
    - Resolve plugin dependencies
    """

    def __init__(self) -> None:
        """Initialise registry."""
        self._plugins: dict[str, ScaffolderPlugin] = {}
        self._enabled: set[str] = set()

    def register(self, plugin: ScaffolderPlugin) -> None:
        """Register a plugin."""
        # TODO: Implement
        pass

    def enable(self, name: str) -> None:
        """Enable a plugin."""
        # TODO: Implement
        pass

    def disable(self, name: str) -> None:
        """Disable a plugin."""
        # TODO: Implement
        pass

    def get_enabled_plugins(self) -> list[ScaffolderPlugin]:
        """Get all enabled plugins."""
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Project Validator
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationRule:
    """A validation rule for project structure."""

    name: str
    description: str
    check: Any  # Callable[[Path], bool]
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Result of validation."""

    rule: str
    passed: bool
    message: str
    severity: str
    path: Path | None = None


class ProjectValidator:
    """
    TODO: Implement project structure validator.

    Validates:
    - Required files exist
    - File contents meet criteria
    - Dependencies are consistent
    - Configuration is valid
    """

    def __init__(self) -> None:
        """Initialise validator."""
        self.rules: list[ValidationRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # TODO: Implement default rules:
        # - README.md exists
        # - pyproject.toml is valid
        # - No circular imports
        # - Tests exist for modules
        pass

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.rules.append(rule)

    def validate(self, project_path: Path) -> list[ValidationResult]:
        """
        Validate a project.

        Args:
            project_path: Path to project root.

        Returns:
            List of validation results.
        """
        # TODO: Implement
        pass

    def validate_readme(self, project_path: Path) -> ValidationResult:
        """
        TODO: Validate README.md exists and has required sections.

        Required sections:
        - Title (h1)
        - Installation
        - Usage
        """
        # TODO: Implement
        pass

    def validate_pyproject(self, project_path: Path) -> ValidationResult:
        """
        TODO: Validate pyproject.toml is valid TOML and has required fields.

        Required fields:
        - project.name
        - project.version
        - project.dependencies
        """
        # TODO: Implement
        pass

    def validate_tests_exist(self, project_path: Path) -> ValidationResult:
        """
        TODO: Validate that test files exist for source modules.

        For each module in src/, there should be a corresponding test file.
        """
        # TODO: Implement
        pass

    def generate_report(
        self,
        results: list[ValidationResult],
    ) -> str:
        """Generate validation report."""
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Advanced Scaffolder
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScaffolderConfig:
    """Configuration for project scaffolding."""

    project_name: str
    author: str
    email: str
    description: str = ""
    python_version: str = "3.12"
    plugins: list[str] = field(default_factory=lambda: ["python_package"])
    extra_vars: dict[str, Any] = field(default_factory=dict)


class AdvancedScaffolder:
    """
    TODO: Implement advanced project scaffolder.

    Features:
    - Plugin-based architecture
    - Template engine integration
    - Pre/post generation hooks
    - Validation after generation
    - Dry-run mode
    """

    def __init__(self) -> None:
        """Initialise scaffolder."""
        self.template_engine = SimpleTemplateEngine()
        self.plugin_registry = PluginRegistry()
        self.validator = ProjectValidator()

        # Register default plugins
        self._register_default_plugins()

    def _register_default_plugins(self) -> None:
        """Register default plugins."""
        self.plugin_registry.register(PythonPackagePlugin())
        self.plugin_registry.register(DocumentationPlugin())
        self.plugin_registry.register(CICDPlugin())

    def scaffold(
        self,
        config: ScaffolderConfig,
        output_dir: Path,
        dry_run: bool = False,
    ) -> list[Path]:
        """
        Scaffold a new project.

        Args:
            config: Scaffolder configuration.
            output_dir: Output directory.
            dry_run: If True, don't actually create files.

        Returns:
            List of created file paths.
        """
        # TODO: Implement
        # 1. Enable requested plugins
        # 2. Collect all file and directory specs
        # 3. Build template context
        # 4. Create directories
        # 5. Render and create files
        # 6. Run post-generation hooks
        # 7. Validate result
        pass

    def _build_context(self, config: ScaffolderConfig) -> dict[str, Any]:
        """Build template context from config."""
        # TODO: Implement
        pass

    def _collect_specs(
        self,
        config: ScaffolderConfig,
    ) -> tuple[list[DirectorySpec], list[FileSpec]]:
        """Collect all specs from enabled plugins."""
        # TODO: Implement
        pass

    def preview(self, config: ScaffolderConfig) -> str:
        """
        Preview what would be generated.

        Args:
            config: Scaffolder configuration.

        Returns:
            Tree-like preview of project structure.
        """
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

README_TEMPLATE = """# ${project_name}

${description}

## Installation

```bash
pip install ${project_name|kebab_case}
```

{% if include_dev_install %}
### Development Installation

```bash
git clone https://github.com/${author}/${project_name|kebab_case}.git
cd ${project_name|kebab_case}
pip install -e ".[dev]"
```
{% endif %}

## Usage

```python
from ${project_name|snake_case} import main

main()
```

{% if features %}
## Features

{% for feature in features %}
- ${feature}
{% endfor %}
{% endif %}

## Licence

MIT Licence © ${year} ${author}
"""

PYPROJECT_TEMPLATE = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "${project_name|kebab_case}"
version = "0.1.0"
description = "${description}"
readme = "README.md"
requires-python = ">=${python_version}"
authors = [
    { name = "${author}", email = "${email}" }
]
dependencies = [
{% for dep in dependencies %}
    "${dep}",
{% endfor %}
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
]

[tool.ruff]
line-length = 88
target-version = "py${python_version|replace('.', '')}"

[tool.mypy]
python_version = "${python_version}"
strict = true
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_template_engine() -> None:
    """Demonstrate the template engine."""
    print("=" * 60)
    print("Template Engine Demo")
    print("=" * 60)

    engine = SimpleTemplateEngine()

    template = """
# ${project_name}

{% if include_badges %}
![Build](https://github.com/${author}/${project_name}/workflows/CI/badge.svg)
{% endif %}

## Features
{% for feature in features %}
- ${feature}
{% endfor %}

Author: ${author|title}
Package: ${project_name|snake_case}
"""

    context = {
        "project_name": "MyAwesomeProject",
        "author": "john doe",
        "include_badges": True,
        "features": ["Fast", "Reliable", "Easy to use"],
    }

    result = engine.render(template, context)
    print(result)


def demonstrate_scaffolder() -> None:
    """Demonstrate the advanced scaffolder."""
    print("\n" + "=" * 60)
    print("Advanced Scaffolder Demo")
    print("=" * 60)

    scaffolder = AdvancedScaffolder()

    config = ScaffolderConfig(
        project_name="research_toolkit",
        author="Jane Researcher",
        email="jane@example.com",
        description="A toolkit for reproducible research",
        plugins=["python_package", "documentation", "cicd"],
    )

    # Preview
    preview = scaffolder.preview(config)
    print("\nProject Structure Preview:")
    print(preview)

    # Dry run
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        files = scaffolder.scaffold(
            config,
            Path(tmpdir),
            dry_run=False,
        )

        print(f"\nGenerated {len(files)} files")
        for f in files[:10]:
            print(f"  - {f}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more")


if __name__ == "__main__":
    demonstrate_template_engine()
    demonstrate_scaffolder()
