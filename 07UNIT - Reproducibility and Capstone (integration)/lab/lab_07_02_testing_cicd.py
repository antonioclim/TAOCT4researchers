#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
07UNIT, Lab 2: Testing Best Practices and CI/CD
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Code without tests is broken by design." — Jacob Kaplan-Moss

Automated testing and continuous integration/continuous deployment (CI/CD) are
essential for producing reliable research software. They prevent regressions,
document expected behaviour, facilitate refactoring and enable effective team
collaboration.

PREREQUISITES
─────────────
- 07UNIT Lab 1: Reproducibility Toolkit
- Python: Intermediate proficiency with classes and decorators
- Libraries: unittest, pytest (conceptual understanding)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Write effective unit tests following the Arrange-Act-Assert pattern
2. Use property-based testing with Hypothesis for comprehensive coverage
3. Apply mocking and fixtures to isolate test concerns
4. Configure CI/CD pipelines using GitHub Actions

ESTIMATED TIME
──────────────
- Reading: 40 minutes
- Coding: 80 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
- Python 3.12+
- pytest ≥7.0
- pytest-cov ≥4.0
- hypothesis ≥6.0 (optional, for property-based testing)

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
import math
import random
import tempfile
import unittest
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar
from unittest.mock import MagicMock, Mock, patch

# Configure module-level logger
logger = logging.getLogger(__name__)

# Type variable for generic tests
T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: UNIT TESTING FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Calculator:
    """
    Simple class for demonstrating testing concepts.

    This example illustrates key testing principles:
    - Test-driven development (TDD)
    - Edge case handling
    - Error handling and validation

    Attributes:
        precision: Number of decimal places for results.

    Example:
        >>> calc = Calculator(precision=6)
        >>> calc.add(2.5, 3.7)
        6.2
    """

    precision: int = 10

    def add(self, a: float, b: float) -> float:
        """Addition with controlled precision."""
        return round(a + b, self.precision)

    def subtract(self, a: float, b: float) -> float:
        """Subtraction with controlled precision."""
        return round(a - b, self.precision)

    def multiply(self, a: float, b: float) -> float:
        """Multiplication with controlled precision."""
        return round(a * b, self.precision)

    def divide(self, a: float, b: float) -> float:
        """
        Division with zero handling.

        Raises:
            ZeroDivisionError: If b equals zero.
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return round(a / b, self.precision)

    def power(self, base: float, exponent: float) -> float:
        """
        Exponentiation.

        Raises:
            ValueError: For mathematically invalid cases.
        """
        if base < 0 and not exponent.is_integer():
            raise ValueError("Cannot raise negative number to non-integer power")
        return round(base**exponent, self.precision)

    def sqrt(self, x: float) -> float:
        """
        Square root.

        Raises:
            ValueError: For negative numbers.
        """
        if x < 0:
            raise ValueError("Cannot compute square root of negative number")
        return round(math.sqrt(x), self.precision)


class TestCalculator(unittest.TestCase):
    """
    Unit tests for Calculator class.

    Demonstrates testing principles:
    1. Arrange-Act-Assert (AAA) pattern
    2. One assertion per test (when practical)
    3. Edge case testing
    4. Descriptive naming conventions
    """

    def setUp(self) -> None:
        """Set up executed before EACH test."""
        self.calc = Calculator(precision=6)

    def tearDown(self) -> None:
        """Cleanup executed after EACH test."""
        pass  # Add cleanup if necessary

    # ─────────────────────────────────────────────────────────────────────────
    # Basic Operation Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_add_positive_numbers(self) -> None:
        """Adding positive numbers returns correct sum."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)

    def test_add_negative_numbers(self) -> None:
        """Adding negative numbers returns correct sum."""
        result = self.calc.add(-2, -3)
        self.assertEqual(result, -5)

    def test_add_mixed_signs(self) -> None:
        """Adding numbers with different signs works correctly."""
        result = self.calc.add(-2, 3)
        self.assertEqual(result, 1)

    def test_add_with_zero(self) -> None:
        """Zero acts as identity element for addition."""
        self.assertEqual(self.calc.add(5, 0), 5)
        self.assertEqual(self.calc.add(0, 5), 5)
        self.assertEqual(self.calc.add(0, 0), 0)

    def test_add_floats(self) -> None:
        """Floating point addition respects precision."""
        result = self.calc.add(0.1, 0.2)
        self.assertAlmostEqual(result, 0.3, places=6)

    # ─────────────────────────────────────────────────────────────────────────
    # Division Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_divide_positive_numbers(self) -> None:
        """Division of positive numbers works correctly."""
        result = self.calc.divide(10, 2)
        self.assertEqual(result, 5)

    def test_divide_by_zero_raises(self) -> None:
        """Division by zero raises ZeroDivisionError."""
        with self.assertRaises(ZeroDivisionError):
            self.calc.divide(10, 0)

    def test_divide_by_zero_message(self) -> None:
        """Division by zero has descriptive error message."""
        with self.assertRaises(ZeroDivisionError) as context:
            self.calc.divide(10, 0)
        self.assertIn("zero", str(context.exception).lower())

    # ─────────────────────────────────────────────────────────────────────────
    # Square Root Tests
    # ─────────────────────────────────────────────────────────────────────────

    def test_sqrt_positive(self) -> None:
        """Square root of positive numbers works correctly."""
        self.assertEqual(self.calc.sqrt(4), 2)
        self.assertEqual(self.calc.sqrt(9), 3)

    def test_sqrt_zero(self) -> None:
        """Square root of zero is zero."""
        self.assertEqual(self.calc.sqrt(0), 0)

    def test_sqrt_negative_raises(self) -> None:
        """Square root of negative number raises ValueError."""
        with self.assertRaises(ValueError):
            self.calc.sqrt(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PYTEST STYLE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


# pytest fixtures would normally be in conftest.py
# Here we simulate the pattern for educational purposes


def calculator_fixture() -> Calculator:
    """Fixture that provides a Calculator instance."""
    return Calculator(precision=8)


class TestCalculatorPytest:
    """
    Pytest-style tests for Calculator.

    Demonstrates pytest idioms:
    - Class-based test organisation
    - Fixtures (simulated)
    - Parametrised tests (conceptual)
    - Assert statements without self
    """

    def setup_method(self) -> None:
        """Set up for each test method."""
        self.calc = calculator_fixture()

    def test_addition_commutative(self) -> None:
        """Addition is commutative: a + b == b + a."""
        a, b = 5, 3
        assert self.calc.add(a, b) == self.calc.add(b, a)

    def test_multiplication_commutative(self) -> None:
        """Multiplication is commutative: a * b == b * a."""
        a, b = 7, 4
        assert self.calc.multiply(a, b) == self.calc.multiply(b, a)

    def test_subtraction_not_commutative(self) -> None:
        """Subtraction is NOT commutative in general."""
        a, b = 5, 3
        assert self.calc.subtract(a, b) != self.calc.subtract(b, a)

    def test_distributive_property(self) -> None:
        """Tests a(b + c) == ab + ac."""
        a, b, c = 2, 3, 4
        left = self.calc.multiply(a, self.calc.add(b, c))
        right = self.calc.add(
            self.calc.multiply(a, b), self.calc.multiply(a, c)
        )
        assert abs(left - right) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MOCKING AND PATCHING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataService:
    """
    Service that fetches data from external source.

    Used to demonstrate mocking patterns.
    """

    base_url: str = "https://api.example.com"

    def fetch_data(self, endpoint: str) -> dict[str, Any]:
        """
        Fetch data from API endpoint.

        In real implementation, this would make HTTP requests.
        """
        # Simulated implementation
        raise NotImplementedError("Would fetch from API")

    def process_data(self, data: dict[str, Any]) -> list[float]:
        """Process raw data into numerical values."""
        return [float(v) for v in data.get("values", [])]


@dataclass
class DataAnalyser:
    """
    Analyser that uses DataService.

    Demonstrates dependency injection for testability.
    """

    service: DataService

    def get_average(self, endpoint: str) -> float:
        """Fetch data and compute average."""
        data = self.service.fetch_data(endpoint)
        values = self.service.process_data(data)
        if not values:
            return 0.0
        return sum(values) / len(values)


class TestMocking(unittest.TestCase):
    """
    Tests demonstrating mocking patterns.

    Mocking allows testing components in isolation by replacing
    dependencies with controlled substitutes.
    """

    def test_mock_basic(self) -> None:
        """Basic Mock object usage."""
        mock_service = Mock(spec=DataService)
        mock_service.fetch_data.return_value = {"values": [1, 2, 3, 4, 5]}
        mock_service.process_data.return_value = [1.0, 2.0, 3.0, 4.0, 5.0]

        analyser = DataAnalyser(service=mock_service)
        result = analyser.get_average("/test")

        self.assertEqual(result, 3.0)
        mock_service.fetch_data.assert_called_once_with("/test")

    def test_mock_side_effect(self) -> None:
        """Mock with side effects for sequential calls."""
        mock_service = Mock(spec=DataService)
        mock_service.fetch_data.side_effect = [
            {"values": [1, 2, 3]},
            {"values": [4, 5, 6]},
        ]
        mock_service.process_data.side_effect = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]

        analyser = DataAnalyser(service=mock_service)

        self.assertEqual(analyser.get_average("/first"), 2.0)
        self.assertEqual(analyser.get_average("/second"), 5.0)

    def test_mock_exception(self) -> None:
        """Mock that raises exception."""
        mock_service = Mock(spec=DataService)
        mock_service.fetch_data.side_effect = ConnectionError("Network error")

        analyser = DataAnalyser(service=mock_service)

        with self.assertRaises(ConnectionError):
            analyser.get_average("/test")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PROPERTY-BASED TESTING CONCEPTS
# ═══════════════════════════════════════════════════════════════════════════════


def property_test_addition_commutative(
    a: float, b: float, calc: Calculator
) -> bool:
    """Property: addition is commutative."""
    return calc.add(a, b) == calc.add(b, a)


def property_test_addition_associative(
    a: float, b: float, c: float, calc: Calculator
) -> bool:
    """Property: addition is associative."""
    left = calc.add(calc.add(a, b), c)
    right = calc.add(a, calc.add(b, c))
    return abs(left - right) < 1e-10


def property_test_identity(a: float, calc: Calculator) -> bool:
    """Property: zero is identity element for addition."""
    return calc.add(a, 0) == a and calc.add(0, a) == a


class PropertyTestRunner:
    """
    Simple property-based test runner.

    Demonstrates the concept of property-based testing without
    requiring the Hypothesis library.

    In production, use Hypothesis instead:
        from hypothesis import given
        from hypothesis import strategies as st

        @given(st.floats(), st.floats())
        def test_add_commutative(a, b):
            calc = Calculator()
            assert calc.add(a, b) == calc.add(b, a)
    """

    def __init__(self, iterations: int = 100) -> None:
        self.iterations = iterations
        self.calculator = Calculator(precision=10)

    def run_property(
        self, prop_func: Callable[..., bool], *arg_generators: Callable[[], Any]
    ) -> tuple[bool, list[Any] | None]:
        """
        Run a property test with generated inputs.

        Returns:
            Tuple of (all_passed, failing_args or None).
        """
        for _ in range(self.iterations):
            args = [gen() for gen in arg_generators]
            try:
                if not prop_func(*args, self.calculator):
                    return False, args
            except Exception:
                return False, args
        return True, None


def demo_property_testing() -> None:
    """Demonstrate property-based testing concepts."""
    logger.info("=" * 60)
    logger.info("DEMO: Property-Based Testing")
    logger.info("=" * 60)

    runner = PropertyTestRunner(iterations=100)

    # Test commutative property
    def gen_float() -> float:
        return random.uniform(-1000, 1000)

    passed, failing = runner.run_property(
        property_test_addition_commutative, gen_float, gen_float
    )
    logger.info(f"Commutative property: {'PASS' if passed else 'FAIL'}")

    # Test associative property
    passed, failing = runner.run_property(
        property_test_addition_associative, gen_float, gen_float, gen_float
    )
    logger.info(f"Associative property: {'PASS' if passed else 'FAIL'}")

    # Test identity property
    passed, failing = runner.run_property(property_test_identity, gen_float)
    logger.info(f"Identity property: {'PASS' if passed else 'FAIL'}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TEST FIXTURES AND SETUP
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TestDatabase:
    """
    In-memory database for testing.

    Demonstrates fixture patterns for database testing.
    """

    data: dict[str, Any] = field(default_factory=dict)

    def insert(self, key: str, value: Any) -> None:
        """Insert a value."""
        self.data[key] = value

    def get(self, key: str) -> Any:
        """Get a value."""
        return self.data.get(key)

    def delete(self, key: str) -> bool:
        """Delete a value."""
        if key in self.data:
            del self.data[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all data."""
        self.data.clear()


class DatabaseTestCase(unittest.TestCase):
    """
    Test case with database fixture.

    Demonstrates proper setup/teardown patterns.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """One-time setup for all tests in class."""
        cls.shared_db = TestDatabase()
        cls.shared_db.insert("config", {"version": "1.0"})
        logger.debug("setUpClass: shared database initialised")

    @classmethod
    def tearDownClass(cls) -> None:
        """One-time cleanup after all tests in class."""
        cls.shared_db.clear()
        logger.debug("tearDownClass: shared database cleared")

    def setUp(self) -> None:
        """Setup before each test."""
        self.db = TestDatabase()
        # Pre-populate with test data
        self.db.insert("user1", {"name": "Alice", "age": 30})
        self.db.insert("user2", {"name": "Bob", "age": 25})

    def tearDown(self) -> None:
        """Cleanup after each test."""
        self.db.clear()

    def test_insert_and_get(self) -> None:
        """Test basic insert and get operations."""
        self.db.insert("user3", {"name": "Charlie"})
        result = self.db.get("user3")
        self.assertEqual(result["name"], "Charlie")

    def test_delete(self) -> None:
        """Test delete operation."""
        self.assertTrue(self.db.delete("user1"))
        self.assertIsNone(self.db.get("user1"))

    def test_delete_nonexistent(self) -> None:
        """Delete of non-existent key returns False."""
        self.assertFalse(self.db.delete("nonexistent"))

    def test_fixture_isolation(self) -> None:
        """Each test gets fresh fixture."""
        # Modify the database
        self.db.delete("user1")
        self.db.delete("user2")
        # Next test will have fresh data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CI/CD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


GITHUB_ACTIONS_TEMPLATE = """
# GitHub Actions CI/CD Configuration
# Save as: .github/workflows/ci.yml

name: CI

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
        python-version: ['3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Lint with ruff
      run: |
        ruff check .
        ruff format --check .

    - name: Type check with mypy
      run: |
        mypy src/

    - name: Test with pytest
      run: |
        pytest --cov=src --cov-report=xml --cov-report=term

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: Build package
      run: |
        pip install build
        python -m build
"""


PYPROJECT_TEMPLATE = '''
# pyproject.toml template
# Modern Python project configuration

[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "your-project"
version = "0.1.0"
description = "A computational research project"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "scipy>=1.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "ruff>=0.1",
    "mypy>=1.0",
    "hypothesis>=6.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "D", "UP"]
ignore = ["D100", "D104"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
strict = true
'''


def generate_ci_config(output_dir: Path) -> dict[str, Path]:
    """
    Generate CI/CD configuration files.

    Args:
        output_dir: Directory to write configuration files.

    Returns:
        Dictionary mapping config names to file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    workflows_dir = output_dir / ".github" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)

    files = {}

    # GitHub Actions workflow
    ci_path = workflows_dir / "ci.yml"
    ci_path.write_text(GITHUB_ACTIONS_TEMPLATE.strip())
    files["github_actions"] = ci_path

    # pyproject.toml
    pyproject_path = output_dir / "pyproject.toml"
    pyproject_path.write_text(PYPROJECT_TEMPLATE.strip())
    files["pyproject"] = pyproject_path

    logger.info(f"Generated CI/CD configuration in {output_dir}")
    return files


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: TEST COVERAGE AND REPORTING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CoverageReport:
    """
    Simple coverage tracking for demonstration.

    In production, use pytest-cov:
        pytest --cov=src --cov-report=html
    """

    total_lines: int = 0
    covered_lines: int = 0
    uncovered_lines: list[int] = field(default_factory=list)

    @property
    def percentage(self) -> float:
        """Calculate coverage percentage."""
        if self.total_lines == 0:
            return 0.0
        return (self.covered_lines / self.total_lines) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_lines": self.total_lines,
            "covered_lines": self.covered_lines,
            "uncovered_lines": self.uncovered_lines,
            "percentage": round(self.percentage, 2),
        }


def demo_testing() -> None:
    """Demonstrate testing concepts."""
    logger.info("=" * 60)
    logger.info("DEMO: Unit Testing")
    logger.info("=" * 60)

    # Run Calculator tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCalculator)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")


def demo_mocking() -> None:
    """Demonstrate mocking patterns."""
    logger.info("=" * 60)
    logger.info("DEMO: Mocking")
    logger.info("=" * 60)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestMocking)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


def demo_ci_generation() -> None:
    """Demonstrate CI/CD configuration generation."""
    logger.info("=" * 60)
    logger.info("DEMO: CI/CD Configuration")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        files = generate_ci_config(Path(tmpdir))
        for name, path in files.items():
            logger.info(f"Generated: {name} -> {path}")
            logger.info(f"  Content preview: {path.read_text()[:100]}...")


def run_all_demos() -> None:
    """Execute all demonstrations."""
    demo_testing()
    print()
    demo_mocking()
    print()
    demo_property_testing()
    print()
    demo_ci_generation()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="07UNIT Lab 2: Testing Best Practices and CI/CD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_7_02_testing_cicd.py --demo
  python lab_7_02_testing_cicd.py --generate-ci /path/to/project
  python lab_7_02_testing_cicd.py --verbose

Testing Commands (with pytest installed):
  pytest tests/                    # Run all tests
  pytest --cov=src                 # Run with coverage
  pytest -k "test_add"             # Run specific tests
  pytest --tb=short                # Short traceback
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations",
    )
    parser.add_argument(
        "--generate-ci",
        type=str,
        metavar="DIR",
        help="Generate CI/CD configuration in specified directory",
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

    if args.generate_ci:
        generate_ci_config(Path(args.generate_ci))
    elif args.demo:
        run_all_demos()
    else:
        logger.info("07UNIT Lab 2: Testing Best Practices and CI/CD")
        logger.info("Use --demo to run demonstrations")
        logger.info("Use --generate-ci DIR to generate CI configuration")
        logger.info("Use --help for more options")


if __name__ == "__main__":
    main()
