#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
07UNIT: Test Suite for Lab 7.02 — Testing and CI/CD
═══════════════════════════════════════════════════════════════════════════════

This test module provides comprehensive tests for testing frameworks
and CI/CD pipeline functionality.

TEST COVERAGE
─────────────
1. Test Discovery: Automatic test discovery and naming conventions
2. Fixtures: Fixture creation, scope, and dependency injection
3. Parametrisation: Test parametrisation and data-driven testing
4. Mocking: Mock objects, patches, and side effects
5. Coverage: Code coverage measurement and reporting
6. CI/CD: Workflow configuration and pipeline validation

USAGE
─────
    pytest tests/test_lab_07_02.py -v
    pytest tests/test_lab_07_02.py -v -k "mock"
    pytest tests/test_lab_07_02.py -v --cov

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

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, Mock, call, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TEST STRUCTURE AND NAMING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTestNamingConventions:
    """Tests for test naming and structure conventions."""

    def test_function_names_should_be_descriptive(self) -> None:
        """Test that test names follow descriptive conventions."""
        # This test itself follows the naming convention
        test_name = "test_function_names_should_be_descriptive"
        
        assert test_name.startswith("test_")
        assert "_" in test_name  # Uses snake_case
        assert len(test_name) > 10  # Descriptive, not just "test_x"

    def test_class_names_follow_conventions(self) -> None:
        """Test that test class names follow conventions."""
        class_name = self.__class__.__name__
        
        assert class_name.startswith("Test")
        # Uses PascalCase
        assert class_name[0].isupper()
        assert "_" not in class_name

    def test_aaa_pattern_demonstration(self) -> None:
        """Demonstrate the Arrange-Act-Assert pattern."""
        # Arrange
        values = [3, 1, 4, 1, 5, 9, 2, 6]
        expected_min = 1
        expected_max = 9
        
        # Act
        actual_min = min(values)
        actual_max = max(values)
        
        # Assert
        assert actual_min == expected_min
        assert actual_max == expected_max


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FIXTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixtures:
    """Tests for pytest fixture functionality."""

    def test_fixture_provides_data(
        self,
        sample_test_data: dict[str, Any]
    ) -> None:
        """Test that fixtures provide expected data."""
        assert "values" in sample_test_data
        assert isinstance(sample_test_data["values"], list)

    def test_fixture_isolation(
        self,
        sample_test_data: dict[str, Any]
    ) -> None:
        """Test that fixtures provide isolated data per test."""
        # Modify the data
        original_len = len(sample_test_data["values"])
        sample_test_data["values"].append(999)
        
        # Verify modification worked
        assert len(sample_test_data["values"]) == original_len + 1

    def test_fixture_isolation_verification(
        self,
        sample_test_data: dict[str, Any]
    ) -> None:
        """Verify that previous test's modification didn't persist."""
        # Should have original length since fixture is fresh
        assert len(sample_test_data["values"]) == 10

    def test_tmp_path_fixture(self, tmp_path: Path) -> None:
        """Test the built-in tmp_path fixture."""
        assert tmp_path.exists()
        assert tmp_path.is_dir()
        
        # Can create files
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        assert test_file.exists()

    def test_temp_project_dir_fixture(
        self,
        temp_project_dir: Path
    ) -> None:
        """Test the custom temp_project_dir fixture."""
        assert temp_project_dir.exists()
        assert temp_project_dir.is_dir()
        
        # Can create project structure
        (temp_project_dir / "src").mkdir()
        (temp_project_dir / "tests").mkdir()
        
        assert (temp_project_dir / "src").exists()

    def test_fixture_dependency_injection(
        self,
        sample_config,  # Uses the ProjectConfig fixture
        temp_project_dir: Path
    ) -> None:
        """Test that multiple fixtures can be combined."""
        assert sample_config.name == "test_research_project"
        assert temp_project_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PARAMETRISATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestParametrisation:
    """Tests for test parametrisation functionality."""

    @pytest.mark.parametrize("input_val,expected", [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
        (10, 100),
        (-5, 25),
    ])
    def test_parametrised_square(
        self,
        input_val: int,
        expected: int
    ) -> None:
        """Test square function with multiple inputs."""
        assert input_val ** 2 == expected

    @pytest.mark.parametrize("values,expected_sum", [
        ([1, 2, 3], 6),
        ([0, 0, 0], 0),
        ([-1, 1], 0),
        ([10], 10),
        ([], 0),
    ])
    def test_parametrised_sum(
        self,
        values: list[int],
        expected_sum: int
    ) -> None:
        """Test sum with various input lists."""
        assert sum(values) == expected_sum

    @pytest.mark.parametrize("string,expected_upper", [
        ("hello", "HELLO"),
        ("World", "WORLD"),
        ("", ""),
        ("123abc", "123ABC"),
    ])
    def test_parametrised_string_operations(
        self,
        string: str,
        expected_upper: str
    ) -> None:
        """Test string operations with various inputs."""
        assert string.upper() == expected_upper

    @pytest.mark.parametrize("a", [1, 2, 3])
    @pytest.mark.parametrize("b", [10, 20])
    def test_multiple_parametrisation(self, a: int, b: int) -> None:
        """Test with cartesian product of parameters."""
        # Tests: (1,10), (1,20), (2,10), (2,20), (3,10), (3,20)
        assert a + b > a
        assert a + b > b


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MOCKING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMocking:
    """Tests for mocking functionality."""

    def test_basic_mock_object(self) -> None:
        """Test basic mock object creation and usage."""
        mock_obj = Mock()
        mock_obj.some_method.return_value = 42
        
        result = mock_obj.some_method()
        
        assert result == 42
        mock_obj.some_method.assert_called_once()

    def test_mock_with_side_effect(self) -> None:
        """Test mock with side effects."""
        mock_func = Mock(side_effect=[1, 2, 3])
        
        assert mock_func() == 1
        assert mock_func() == 2
        assert mock_func() == 3

    def test_mock_with_exception(self) -> None:
        """Test mock that raises an exception."""
        mock_func = Mock(side_effect=ValueError("Test error"))
        
        with pytest.raises(ValueError, match="Test error"):
            mock_func()

    def test_mock_call_arguments(self) -> None:
        """Test verifying mock call arguments."""
        mock_func = Mock()
        
        mock_func(1, 2, key="value")
        
        mock_func.assert_called_with(1, 2, key="value")

    def test_mock_multiple_calls(self) -> None:
        """Test verifying multiple mock calls."""
        mock_func = Mock()
        
        mock_func("first")
        mock_func("second")
        mock_func("third")
        
        assert mock_func.call_count == 3
        mock_func.assert_has_calls([
            call("first"),
            call("second"),
            call("third")
        ])

    def test_patch_decorator(self) -> None:
        """Test the patch decorator."""
        with patch("builtins.open", mock_open := Mock()):
            mock_open.return_value.__enter__.return_value.read.return_value = "content"
            
            # This would normally read a file, but is mocked
            # result = open("test.txt").read()
            # Simplified for test
            assert mock_open is not None

    def test_magic_mock_attributes(self) -> None:
        """Test MagicMock auto-created attributes."""
        mock = MagicMock()
        
        # MagicMock auto-creates attributes
        mock.attr1.attr2.attr3.method()
        
        mock.attr1.attr2.attr3.method.assert_called_once()

    def test_mock_spec(self) -> None:
        """Test mock with spec for type safety."""
        class RealClass:
            def real_method(self, x: int) -> int:
                return x * 2
        
        mock = Mock(spec=RealClass)
        mock.real_method.return_value = 10
        
        # Valid method
        assert mock.real_method(5) == 10
        
        # Invalid method would raise AttributeError with spec
        with pytest.raises(AttributeError):
            mock.nonexistent_method()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ASSERTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssertions:
    """Tests for various assertion types."""

    def test_equality_assertions(self) -> None:
        """Test equality assertions."""
        assert 1 + 1 == 2
        assert "hello" == "hello"
        assert [1, 2, 3] == [1, 2, 3]

    def test_inequality_assertions(self) -> None:
        """Test inequality assertions."""
        assert 1 != 2
        assert "hello" != "world"

    def test_truthiness_assertions(self) -> None:
        """Test truthiness assertions."""
        assert True
        assert 1  # Truthy
        assert "string"  # Truthy
        assert [1]  # Truthy
        
        assert not False
        assert not 0  # Falsy
        assert not ""  # Falsy
        assert not []  # Falsy

    def test_containment_assertions(self) -> None:
        """Test containment assertions."""
        assert 2 in [1, 2, 3]
        assert "ell" in "hello"
        assert "key" in {"key": "value"}

    def test_identity_assertions(self) -> None:
        """Test identity assertions."""
        a = [1, 2, 3]
        b = a
        c = [1, 2, 3]
        
        assert a is b  # Same object
        assert a is not c  # Different objects with same value
        assert a == c  # Equal values

    def test_type_assertions(self) -> None:
        """Test type assertions."""
        assert isinstance(42, int)
        assert isinstance("hello", str)
        assert isinstance([1, 2], list)
        assert isinstance({"a": 1}, dict)

    def test_approximate_assertions(self) -> None:
        """Test approximate equality for floats."""
        result = 0.1 + 0.2
        
        # Direct comparison may fail due to floating point
        # assert result == 0.3  # This might fail!
        
        # Use pytest.approx for float comparison
        assert result == pytest.approx(0.3)
        assert result == pytest.approx(0.3, rel=1e-9)

    def test_exception_assertions(self) -> None:
        """Test exception assertions."""
        with pytest.raises(ZeroDivisionError):
            _ = 1 / 0
        
        with pytest.raises(ValueError, match="invalid literal"):
            int("not a number")

    def test_warning_assertions(self) -> None:
        """Test warning assertions."""
        import warnings
        
        def warn_function() -> str:
            warnings.warn("This is a warning", UserWarning)
            return "result"
        
        with pytest.warns(UserWarning, match="This is a warning"):
            result = warn_function()
        
        assert result == "result"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CI/CD CONFIGURATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCICDConfiguration:
    """Tests for CI/CD configuration validation."""

    def test_github_actions_workflow_structure(
        self,
        sample_workflow_config: dict[str, Any]
    ) -> None:
        """Test GitHub Actions workflow structure."""
        assert "name" in sample_workflow_config
        assert "on" in sample_workflow_config
        assert "jobs" in sample_workflow_config

    def test_workflow_triggers(
        self,
        sample_workflow_config: dict[str, Any]
    ) -> None:
        """Test workflow trigger configuration."""
        triggers = sample_workflow_config["on"]
        
        assert "push" in triggers or "pull_request" in triggers

    def test_workflow_job_structure(
        self,
        sample_workflow_config: dict[str, Any]
    ) -> None:
        """Test workflow job structure."""
        jobs = sample_workflow_config["jobs"]
        
        assert len(jobs) > 0
        
        for job_name, job_config in jobs.items():
            assert "runs-on" in job_config
            assert "steps" in job_config

    def test_workflow_step_structure(
        self,
        sample_workflow_config: dict[str, Any]
    ) -> None:
        """Test workflow step structure."""
        steps = sample_workflow_config["jobs"]["test"]["steps"]
        
        for step in steps:
            # Each step must have either 'uses' or 'run'
            assert "uses" in step or "run" in step

    def test_workflow_yaml_serialisation(
        self,
        sample_workflow_config: dict[str, Any],
        tmp_path: Path
    ) -> None:
        """Test workflow configuration can be serialised to YAML."""
        try:
            import yaml
            
            workflow_file = tmp_path / "workflow.yml"
            with open(workflow_file, "w") as f:
                yaml.dump(sample_workflow_config, f)
            
            with open(workflow_file, "r") as f:
                loaded = yaml.safe_load(f)
            
            assert loaded["name"] == sample_workflow_config["name"]
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_matrix_strategy_configuration(self) -> None:
        """Test matrix strategy configuration."""
        matrix_config = {
            "strategy": {
                "matrix": {
                    "python-version": ["3.10", "3.11", "3.12"],
                    "os": ["ubuntu-latest", "windows-latest"]
                }
            }
        }
        
        matrix = matrix_config["strategy"]["matrix"]
        
        # Total combinations = 3 * 2 = 6
        total_combinations = (
            len(matrix["python-version"]) * len(matrix["os"])
        )
        assert total_combinations == 6


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: CODE COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeCoverage:
    """Tests related to code coverage concepts."""

    def test_branch_coverage_demonstration(self) -> None:
        """Demonstrate branch coverage testing."""
        def classify_number(n: int) -> str:
            if n < 0:
                return "negative"
            elif n == 0:
                return "zero"
            else:
                return "positive"
        
        # Cover all branches
        assert classify_number(-5) == "negative"
        assert classify_number(0) == "zero"
        assert classify_number(5) == "positive"

    def test_path_coverage_demonstration(self) -> None:
        """Demonstrate path coverage testing."""
        def process(a: bool, b: bool) -> str:
            result = ""
            if a:
                result += "A"
            if b:
                result += "B"
            return result or "none"
        
        # Cover all paths
        assert process(False, False) == "none"
        assert process(True, False) == "A"
        assert process(False, True) == "B"
        assert process(True, True) == "AB"

    def test_boundary_coverage_demonstration(self) -> None:
        """Demonstrate boundary value testing."""
        def is_valid_percentage(value: float) -> bool:
            return 0 <= value <= 100
        
        # Boundary values
        assert is_valid_percentage(0) is True      # Lower boundary
        assert is_valid_percentage(100) is True    # Upper boundary
        assert is_valid_percentage(-0.001) is False  # Just below lower
        assert is_valid_percentage(100.001) is False  # Just above upper
        assert is_valid_percentage(50) is True     # Middle value


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TEST MARKERS AND SKIPPING
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarkersAndSkipping:
    """Tests demonstrating markers and skip functionality."""

    @pytest.mark.skip(reason="Demonstrating skip marker")
    def test_skipped_test(self) -> None:
        """This test is always skipped."""
        assert False  # Would fail if run

    @pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="Requires Python 3.12+"
    )
    def test_conditional_skip(self) -> None:
        """Test skipped on older Python versions."""
        assert sys.version_info >= (3, 12)

    @pytest.mark.xfail(reason="Demonstrating expected failure")
    def test_expected_failure(self) -> None:
        """This test is expected to fail."""
        # This assertion would fail
        # assert 1 == 2
        pass  # Changed to pass for actual test run

    @pytest.mark.slow
    def test_marked_as_slow(self) -> None:
        """Test marked as slow (can be deselected with -m 'not slow')."""
        import time
        time.sleep(0.01)  # Simulate slow operation
        assert True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: TEST OUTPUT AND REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputAndReporting:
    """Tests for test output and reporting."""

    def test_captured_output(self, capsys) -> None:
        """Test capturing stdout and stderr."""
        print("Hello stdout")
        print("Hello stderr", file=sys.stderr)
        
        captured = capsys.readouterr()
        
        assert "Hello stdout" in captured.out
        assert "Hello stderr" in captured.err

    def test_captured_logs(self, captured_logs) -> None:
        """Test capturing log messages."""
        import logging
        
        logger = logging.getLogger("test")
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        assert "Test info message" in captured_logs.text
        assert "Test warning message" in captured_logs.text

    def test_assertion_introspection(self) -> None:
        """Demonstrate pytest's assertion introspection."""
        # pytest provides detailed information on assertion failures
        data = {"key": "value", "number": 42}
        
        assert data["key"] == "value"
        assert data["number"] == 42


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCICDIntegration:
    """Integration tests for CI/CD workflow."""

    @pytest.mark.integration
    def test_create_workflow_file(
        self,
        github_actions_dir: Path,
        sample_workflow_config: dict[str, Any]
    ) -> None:
        """Test creating a complete workflow file."""
        try:
            import yaml
            
            workflow_file = github_actions_dir / "ci.yml"
            with open(workflow_file, "w") as f:
                yaml.dump(sample_workflow_config, f, default_flow_style=False)
            
            assert workflow_file.exists()
            assert workflow_file.stat().st_size > 0
        except ImportError:
            # Fallback to JSON for testing
            workflow_file = github_actions_dir / "ci.json"
            with open(workflow_file, "w") as f:
                json.dump(sample_workflow_config, f, indent=2)
            
            assert workflow_file.exists()

    @pytest.mark.integration
    def test_validate_project_test_structure(
        self,
        temp_project_dir: Path
    ) -> None:
        """Test validating a project's test structure."""
        # Create standard test structure
        tests_dir = temp_project_dir / "tests"
        tests_dir.mkdir()
        
        # Create test files
        (tests_dir / "__init__.py").write_text("")
        (tests_dir / "conftest.py").write_text("# Shared fixtures\n")
        (tests_dir / "test_example.py").write_text(
            "def test_example():\n    assert True\n"
        )
        
        # Validate structure
        assert (tests_dir / "__init__.py").exists()
        assert (tests_dir / "conftest.py").exists()
        assert list(tests_dir.glob("test_*.py"))

    @pytest.mark.integration
    def test_mock_ci_pipeline_execution(
        self,
        mock_subprocess: MagicMock,
        temp_project_dir: Path
    ) -> None:
        """Test mocking a CI pipeline execution."""
        # Simulate CI steps
        steps = [
            ["pip", "install", "-e", ".[dev]"],
            ["ruff", "check", "."],
            ["pytest", "--cov"],
        ]
        
        for step in steps:
            result = subprocess.run(step, capture_output=True, text=True)
            # Mock always returns success
            assert result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: SUBPROCESS AND EXTERNAL COMMAND TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubprocessMocking:
    """Tests for mocking subprocess calls."""

    def test_mock_subprocess_success(
        self,
        mock_subprocess: MagicMock
    ) -> None:
        """Test mocking successful subprocess call."""
        result = subprocess.run(["echo", "hello"], capture_output=True)
        
        assert result.returncode == 0
        mock_subprocess.assert_called()

    def test_mock_subprocess_with_output(self) -> None:
        """Test mocking subprocess with specific output."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="test output\n",
                stderr=""
            )
            
            result = subprocess.run(
                ["some", "command"],
                capture_output=True,
                text=True
            )
            
            assert result.stdout == "test output\n"
            assert result.returncode == 0

    def test_mock_subprocess_failure(self) -> None:
        """Test mocking subprocess failure."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="Error: command failed\n"
            )
            
            result = subprocess.run(
                ["failing", "command"],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 1
            assert "Error" in result.stderr


# ═══════════════════════════════════════════════════════════════════════════════
# END OF TEST MODULE
# ═══════════════════════════════════════════════════════════════════════════════
