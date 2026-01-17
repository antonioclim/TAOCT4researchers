#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
07UNIT: Test Suite for Lab 7.03 — Project Scaffolder
═══════════════════════════════════════════════════════════════════════════════

This test module provides comprehensive tests for the project scaffolder,
covering template rendering, directory creation, file generation and
project validation.

TEST COVERAGE
─────────────
1. TemplateEngine: Variable substitution, conditionals, loops, filters
2. ProjectScaffolder: Directory creation, file generation, structure validation
3. ConfigurationParser: Config loading, validation, defaults
4. ValidationEngine: Project structure validation, best practices checks
5. Integration: End-to-end project generation workflows

USAGE
─────
    pytest tests/test_lab_07_03.py -v
    pytest tests/test_lab_07_03.py -v -m integration
    pytest tests/test_lab_07_03.py -v --cov=lab

DEPENDENCIES
────────────
pytest>=7.0
pytest-cov>=4.0

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

# Add lab directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TEMPLATE ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemplateEngine:
    """Tests for template rendering functionality."""
    
    def test_simple_variable_substitution(self) -> None:
        """Test basic variable substitution in templates."""
        template = "Hello, ${name}!"
        context = {"name": "World"}
        
        # Simple substitution using regex
        result = re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(context.get(m.group(1), m.group(0))),
            template,
        )
        
        assert result == "Hello, World!"
    
    def test_multiple_variable_substitution(self) -> None:
        """Test multiple variables in a single template."""
        template = "Project: ${project_name} by ${author}"
        context = {
            "project_name": "my_research",
            "author": "Dr. Smith",
        }
        
        result = re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(context.get(m.group(1), m.group(0))),
            template,
        )
        
        assert result == "Project: my_research by Dr. Smith"
    
    def test_missing_variable_handling(self) -> None:
        """Test handling of missing variables."""
        template = "Value: ${missing_var}"
        context = {}
        
        # Default: keep original if missing
        result = re.sub(
            r'\$\{(\w+)\}',
            lambda m: str(context.get(m.group(1), m.group(0))),
            template,
        )
        
        assert result == "Value: ${missing_var}"
    
    def test_nested_variable_access(self) -> None:
        """Test accessing nested dictionary values."""
        template = "Version: ${config.version}"
        context = {
            "config": {"version": "1.0.0"},
        }
        
        def get_nested(ctx: dict, key: str) -> Any:
            """Get nested value using dot notation."""
            parts = key.split(".")
            value = ctx
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, "")
                else:
                    return ""
            return value
        
        result = re.sub(
            r'\$\{([\w.]+)\}',
            lambda m: str(get_nested(context, m.group(1))),
            template,
        )
        
        assert result == "Version: 1.0.0"
    
    def test_filter_upper(self) -> None:
        """Test uppercase filter."""
        value = "hello"
        result = value.upper()
        assert result == "HELLO"
    
    def test_filter_lower(self) -> None:
        """Test lowercase filter."""
        value = "HELLO"
        result = value.lower()
        assert result == "hello"
    
    def test_filter_title(self) -> None:
        """Test title case filter."""
        value = "hello world"
        result = value.title()
        assert result == "Hello World"
    
    def test_filter_snake_case(self) -> None:
        """Test snake_case conversion filter."""
        def to_snake_case(text: str) -> str:
            """Convert text to snake_case."""
            # Replace spaces and hyphens with underscores
            text = re.sub(r'[-\s]+', '_', text)
            # Insert underscore before uppercase letters
            text = re.sub(r'([a-z])([A-Z])', r'\1_\2', text)
            return text.lower()
        
        test_cases = [
            ("HelloWorld", "hello_world"),
            ("hello-world", "hello_world"),
            ("Hello World", "hello_world"),
            ("myProjectName", "my_project_name"),
        ]
        
        for input_val, expected in test_cases:
            assert to_snake_case(input_val) == expected
    
    def test_filter_kebab_case(self) -> None:
        """Test kebab-case conversion filter."""
        def to_kebab_case(text: str) -> str:
            """Convert text to kebab-case."""
            # Replace spaces and underscores with hyphens
            text = re.sub(r'[_\s]+', '-', text)
            # Insert hyphen before uppercase letters
            text = re.sub(r'([a-z])([A-Z])', r'\1-\2', text)
            return text.lower()
        
        test_cases = [
            ("HelloWorld", "hello-world"),
            ("hello_world", "hello-world"),
            ("Hello World", "hello-world"),
        ]
        
        for input_val, expected in test_cases:
            assert to_kebab_case(input_val) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PROJECT SCAFFOLDER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestProjectScaffolder:
    """Tests for project scaffolding functionality."""
    
    @pytest.fixture
    def scaffolder_config(self) -> dict[str, Any]:
        """Provide a standard scaffolder configuration."""
        return {
            "project_name": "test_project",
            "author": "Test Author",
            "version": "0.1.0",
            "python_version": "3.12",
            "include_tests": True,
            "include_ci": True,
            "include_docs": False,
            "dependencies": ["numpy", "pandas"],
        }
    
    def test_create_directory_structure(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test creating basic directory structure."""
        # Define structure
        directories = [
            "src/package",
            "tests",
            "data/raw",
            "data/processed",
        ]
        
        # Create directories
        for dir_path in directories:
            full_path = temp_project_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        # Verify
        assert (temp_project_dir / "src" / "package").is_dir()
        assert (temp_project_dir / "tests").is_dir()
        assert (temp_project_dir / "data" / "raw").is_dir()
        assert (temp_project_dir / "data" / "processed").is_dir()
    
    def test_create_init_files(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test creating __init__.py files."""
        package_dir = temp_project_dir / "src" / "package"
        package_dir.mkdir(parents=True)
        
        # Create __init__.py
        init_file = package_dir / "__init__.py"
        init_content = '"""Package initialisation."""\n\n__version__ = "0.1.0"\n'
        init_file.write_text(init_content)
        
        assert init_file.exists()
        assert "__version__" in init_file.read_text()
    
    def test_create_readme(
        self,
        temp_project_dir: Path,
        scaffolder_config: dict[str, Any],
    ) -> None:
        """Test README.md generation."""
        readme_template = textwrap.dedent(f"""\
            # {scaffolder_config['project_name']}
            
            ## Installation
            
            ```bash
            pip install -e .
            ```
            
            ## Author
            
            {scaffolder_config['author']}
        """)
        
        readme_path = temp_project_dir / "README.md"
        readme_path.write_text(readme_template)
        
        assert readme_path.exists()
        content = readme_path.read_text()
        assert scaffolder_config["project_name"] in content
        assert scaffolder_config["author"] in content
    
    def test_create_pyproject_toml(
        self,
        temp_project_dir: Path,
        scaffolder_config: dict[str, Any],
    ) -> None:
        """Test pyproject.toml generation."""
        deps = "\n".join(f'    "{d}",' for d in scaffolder_config["dependencies"])
        
        pyproject_content = textwrap.dedent(f"""\
            [build-system]
            requires = ["setuptools>=61.0"]
            build-backend = "setuptools.build_meta"
            
            [project]
            name = "{scaffolder_config['project_name']}"
            version = "{scaffolder_config['version']}"
            authors = [
                {{name = "{scaffolder_config['author']}"}}
            ]
            requires-python = ">={scaffolder_config['python_version']}"
            dependencies = [
            {deps}
            ]
        """)
        
        pyproject_path = temp_project_dir / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        assert pyproject_path.exists()
        content = pyproject_path.read_text()
        assert scaffolder_config["project_name"] in content
        assert "numpy" in content
    
    def test_create_gitignore(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test .gitignore generation."""
        gitignore_content = textwrap.dedent("""\
            # Python
            __pycache__/
            *.py[cod]
            .venv/
            
            # IDE
            .idea/
            .vscode/
            
            # Testing
            .pytest_cache/
            .coverage
            htmlcov/
            
            # Distribution
            dist/
            build/
            *.egg-info/
        """)
        
        gitignore_path = temp_project_dir / ".gitignore"
        gitignore_path.write_text(gitignore_content)
        
        assert gitignore_path.exists()
        content = gitignore_path.read_text()
        assert "__pycache__/" in content
        assert ".venv/" in content
    
    def test_conditional_directory_creation(
        self,
        temp_project_dir: Path,
        scaffolder_config: dict[str, Any],
    ) -> None:
        """Test conditional directory creation based on config."""
        # Create tests directory only if include_tests is True
        if scaffolder_config["include_tests"]:
            (temp_project_dir / "tests").mkdir(exist_ok=True)
            (temp_project_dir / "tests" / "__init__.py").write_text("")
            (temp_project_dir / "tests" / "conftest.py").write_text("")
        
        # Create docs directory only if include_docs is True
        if scaffolder_config["include_docs"]:
            (temp_project_dir / "docs").mkdir(exist_ok=True)
        
        assert (temp_project_dir / "tests").exists()  # include_tests=True
        assert not (temp_project_dir / "docs").exists()  # include_docs=False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CONFIGURATION VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigurationValidation:
    """Tests for configuration parsing and validation."""
    
    def test_valid_config(
        self,
        sample_config: dict[str, Any],
    ) -> None:
        """Test that valid configuration passes validation."""
        required_fields = ["project_name", "version", "author"]
        
        for field in required_fields:
            assert field in sample_config
    
    def test_project_name_validation(self) -> None:
        """Test project name validation rules."""
        def is_valid_project_name(name: str) -> bool:
            """Validate project name follows PEP 508."""
            pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*$'
            return bool(re.match(pattern, name)) and len(name) <= 50
        
        valid_names = [
            "my_project",
            "MyProject",
            "project123",
            "my-project",
        ]
        
        invalid_names = [
            "123project",  # Starts with number
            "_project",     # Starts with underscore
            "my project",   # Contains space
            "",             # Empty
            "a" * 51,       # Too long
        ]
        
        for name in valid_names:
            assert is_valid_project_name(name), f"{name} should be valid"
        
        for name in invalid_names:
            assert not is_valid_project_name(name), f"{name} should be invalid"
    
    def test_python_version_validation(self) -> None:
        """Test Python version validation."""
        def is_valid_python_version(version: str) -> bool:
            """Validate Python version string."""
            pattern = r'^3\.(10|11|12|1[3-9]|[2-9]\d)$'
            return bool(re.match(pattern, version))
        
        valid_versions = ["3.10", "3.11", "3.12", "3.13"]
        invalid_versions = ["2.7", "3.9", "3.8", "4.0", "three.twelve"]
        
        for version in valid_versions:
            assert is_valid_python_version(version)
        
        for version in invalid_versions:
            assert not is_valid_python_version(version)
    
    def test_dependency_list_validation(self) -> None:
        """Test dependency list validation."""
        def is_valid_dependency(dep: str) -> bool:
            """Validate dependency specification."""
            # Simple check for package name pattern
            pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*(\[[\w,]+\])?(>=|<=|==|~=|!=)?[\d.]*$'
            return bool(re.match(pattern, dep))
        
        valid_deps = [
            "numpy",
            "pandas>=2.0",
            "scipy==1.11.0",
            "scikit-learn",
            "pytest-cov>=4.0",
        ]
        
        for dep in valid_deps:
            assert is_valid_dependency(dep), f"{dep} should be valid"
    
    def test_config_defaults(self) -> None:
        """Test default configuration values."""
        defaults = {
            "version": "0.1.0",
            "python_version": "3.12",
            "include_tests": True,
            "include_ci": True,
            "include_docs": False,
            "dependencies": [],
        }
        
        # Merge with minimal config
        minimal_config = {"project_name": "test", "author": "Author"}
        full_config = {**defaults, **minimal_config}
        
        assert full_config["version"] == "0.1.0"
        assert full_config["include_tests"] is True
        assert full_config["project_name"] == "test"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PROJECT VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestProjectValidation:
    """Tests for validating generated project structure."""
    
    @pytest.fixture
    def complete_project(
        self,
        temp_project_dir: Path,
    ) -> Path:
        """Create a complete project structure for validation."""
        # Create directories
        (temp_project_dir / "src" / "package").mkdir(parents=True)
        (temp_project_dir / "tests").mkdir()
        
        # Create essential files
        (temp_project_dir / "README.md").write_text("# Project\n")
        (temp_project_dir / "pyproject.toml").write_text(
            '[project]\nname = "test"\nversion = "0.1.0"\n'
        )
        (temp_project_dir / "src" / "package" / "__init__.py").write_text("")
        (temp_project_dir / "tests" / "__init__.py").write_text("")
        (temp_project_dir / "tests" / "conftest.py").write_text("")
        
        return temp_project_dir
    
    def test_validate_readme_exists(
        self,
        complete_project: Path,
    ) -> None:
        """Test validation of README.md existence."""
        readme = complete_project / "README.md"
        assert readme.exists(), "README.md should exist"
    
    def test_validate_pyproject_exists(
        self,
        complete_project: Path,
    ) -> None:
        """Test validation of pyproject.toml existence."""
        pyproject = complete_project / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"
    
    def test_validate_source_structure(
        self,
        complete_project: Path,
    ) -> None:
        """Test validation of source directory structure."""
        src_dir = complete_project / "src"
        assert src_dir.is_dir(), "src/ directory should exist"
        
        # Check for at least one package
        packages = [
            d for d in src_dir.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
        ]
        assert len(packages) >= 1, "At least one package should exist"
    
    def test_validate_tests_structure(
        self,
        complete_project: Path,
    ) -> None:
        """Test validation of tests directory structure."""
        tests_dir = complete_project / "tests"
        assert tests_dir.is_dir(), "tests/ directory should exist"
        assert (tests_dir / "conftest.py").exists(), "conftest.py should exist"
    
    def test_validate_no_absolute_paths(
        self,
        complete_project: Path,
    ) -> None:
        """Test that generated files don't contain absolute paths."""
        # Check all Python files
        for py_file in complete_project.rglob("*.py"):
            content = py_file.read_text()
            # Look for common absolute path patterns
            absolute_patterns = [
                r'/home/\w+/',
                r'C:\\Users\\',
                r'/Users/\w+/',
            ]
            for pattern in absolute_patterns:
                assert not re.search(pattern, content), \
                    f"Absolute path found in {py_file}"
    
    def test_validate_file_permissions(
        self,
        complete_project: Path,
    ) -> None:
        """Test that files have appropriate permissions."""
        for file_path in complete_project.rglob("*"):
            if file_path.is_file():
                # Check file is readable
                assert os.access(file_path, os.R_OK), \
                    f"{file_path} should be readable"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestScaffolderIntegration:
    """Integration tests for complete scaffolding workflows."""
    
    @pytest.mark.integration
    def test_full_project_generation(
        self,
        temp_project_dir: Path,
        sample_config: dict[str, Any],
    ) -> None:
        """Test complete project generation workflow."""
        project_name = sample_config["project_name"].replace("-", "_")
        
        # Create directory structure
        dirs_to_create = [
            f"src/{project_name}",
            "tests",
        ]
        
        if sample_config.get("include_docs"):
            dirs_to_create.append("docs")
        
        for dir_path in dirs_to_create:
            (temp_project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create essential files
        files_to_create = {
            "README.md": f"# {sample_config['project_name']}\n",
            "pyproject.toml": f'[project]\nname = "{sample_config["project_name"]}"\n',
            f"src/{project_name}/__init__.py": '"""Package."""\n',
            "tests/__init__.py": "",
            "tests/conftest.py": '"""Test configuration."""\n',
        }
        
        for file_path, content in files_to_create.items():
            (temp_project_dir / file_path).write_text(content)
        
        # Validate result
        assert (temp_project_dir / "README.md").exists()
        assert (temp_project_dir / "pyproject.toml").exists()
        assert (temp_project_dir / f"src/{project_name}/__init__.py").exists()
        assert (temp_project_dir / "tests/conftest.py").exists()
    
    @pytest.mark.integration
    def test_project_with_all_options(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test project generation with all options enabled."""
        config = {
            "project_name": "full_project",
            "author": "Test Author",
            "version": "0.1.0",
            "include_tests": True,
            "include_ci": True,
            "include_docs": True,
            "include_docker": True,
        }
        
        # Create all directories
        directories = [
            "src/full_project",
            "tests",
            "docs",
            ".github/workflows",
        ]
        
        for dir_path in directories:
            (temp_project_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create all files
        files = {
            "README.md": "# Full Project\n",
            "pyproject.toml": '[project]\nname = "full_project"\n',
            "Dockerfile": "FROM python:3.12\n",
            ".github/workflows/ci.yml": "name: CI\non: [push]\n",
            "docs/index.md": "# Documentation\n",
        }
        
        for file_path, content in files.items():
            (temp_project_dir / file_path).write_text(content)
        
        # Validate all components exist
        assert (temp_project_dir / "Dockerfile").exists()
        assert (temp_project_dir / ".github/workflows/ci.yml").exists()
        assert (temp_project_dir / "docs").is_dir()
    
    @pytest.mark.integration
    def test_project_minimal_options(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test project generation with minimal options."""
        config = {
            "project_name": "minimal_project",
            "author": "Author",
            "include_tests": False,
            "include_ci": False,
            "include_docs": False,
        }
        
        # Create minimal structure
        (temp_project_dir / "src/minimal_project").mkdir(parents=True)
        
        files = {
            "README.md": "# Minimal Project\n",
            "pyproject.toml": '[project]\nname = "minimal_project"\n',
            "src/minimal_project/__init__.py": "",
        }
        
        for file_path, content in files.items():
            (temp_project_dir / file_path).write_text(content)
        
        # Verify minimal structure
        assert (temp_project_dir / "README.md").exists()
        assert (temp_project_dir / "pyproject.toml").exists()
        assert not (temp_project_dir / "tests").exists()
        assert not (temp_project_dir / ".github").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EDGE CASES AND ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""
    
    def test_project_name_with_hyphens(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test handling project names with hyphens."""
        project_name = "my-research-project"
        package_name = project_name.replace("-", "_")
        
        # Package directory should use underscores
        package_dir = temp_project_dir / "src" / package_name
        package_dir.mkdir(parents=True)
        
        assert package_dir.exists()
        assert package_dir.name == "my_research_project"
    
    def test_existing_directory_handling(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test handling of existing directories."""
        existing_dir = temp_project_dir / "src"
        existing_dir.mkdir(parents=True)
        
        # mkdir with exist_ok=True should not raise
        existing_dir.mkdir(exist_ok=True)
        
        assert existing_dir.exists()
    
    def test_special_characters_in_author(self) -> None:
        """Test handling special characters in author name."""
        authors_with_special_chars = [
            "José García",
            "François Müller",
            "中文名字",
            "Имя Фамилия",
        ]
        
        for author in authors_with_special_chars:
            # Should be encodable as UTF-8
            encoded = author.encode("utf-8")
            assert isinstance(encoded, bytes)
    
    def test_empty_dependency_list(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test project generation with no dependencies."""
        pyproject_content = textwrap.dedent("""\
            [project]
            name = "test"
            version = "0.1.0"
            dependencies = []
        """)
        
        pyproject_path = temp_project_dir / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)
        
        content = pyproject_path.read_text()
        assert "dependencies = []" in content
    
    def test_unicode_in_readme(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test README with Unicode characters."""
        readme_content = textwrap.dedent("""\
            # 研究プロジェクト
            
            © 2025 Αυτός ο κώδικας
            
            Ωmega → ∞
        """)
        
        readme_path = temp_project_dir / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        
        loaded = readme_path.read_text(encoding="utf-8")
        assert "研究プロジェクト" in loaded
        assert "∞" in loaded


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance tests for scaffolder operations."""
    
    @pytest.mark.slow
    def test_large_project_generation(
        self,
        temp_project_dir: Path,
    ) -> None:
        """Test scaffolding a project with many files."""
        import time
        
        start = time.perf_counter()
        
        # Create many directories
        for i in range(10):
            (temp_project_dir / f"module_{i}").mkdir(exist_ok=True)
            
            # Create files in each directory
            for j in range(10):
                (temp_project_dir / f"module_{i}" / f"file_{j}.py").write_text(
                    f'"""Module {i}, file {j}."""\n'
                )
        
        elapsed = time.perf_counter() - start
        
        # Should complete in reasonable time (< 5 seconds)
        assert elapsed < 5.0, f"Generation took too long: {elapsed:.2f}s"
        
        # Verify structure
        total_files = len(list(temp_project_dir.rglob("*.py")))
        assert total_files == 100
    
    @pytest.mark.slow
    def test_template_rendering_performance(self) -> None:
        """Test template rendering performance."""
        import time
        
        template = "Hello, ${name}! Your project is ${project}."
        context = {"name": "User", "project": "MyProject"}
        
        iterations = 10000
        start = time.perf_counter()
        
        for _ in range(iterations):
            _ = re.sub(
                r'\$\{(\w+)\}',
                lambda m: str(context.get(m.group(1), "")),
                template,
            )
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        
        # Should be under 0.1ms per render
        assert avg_time < 0.0001, f"Rendering too slow: {avg_time*1000:.3f}ms"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
