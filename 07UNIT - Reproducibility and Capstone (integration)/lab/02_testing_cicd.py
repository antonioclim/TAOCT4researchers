#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Lab 2: Testing Best Practices și CI/CD
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Code without tests is broken by design." — Jacob Kaplan-Moss

Testarea automatizată și CI/CD sunt esențiale pentru:
- Prevenirea regresiilor
- Documentarea comportamentului așteptat
- Facilitarea refactorizării
- Colaborare eficientă în echipă

OBIECTIVE
─────────
1. Scrierea de teste unitare eficiente
2. Property-based testing cu Hypothesis
3. Mocking și fixtures
4. Configurare CI/CD

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import random
import tempfile
import json
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar, Generic
from pathlib import Path
from functools import wraps
import unittest
from unittest.mock import Mock, patch, MagicMock

# Type variable for generic tests
T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: UNIT TESTING FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Calculator:
    """
    Clasă simplă pentru demonstrarea testării.
    
    Acest exemplu ilustrează concepte cheie:
    - Test-driven development (TDD)
    - Edge cases
    - Error handling
    """
    precision: int = 10
    
    def add(self, a: float, b: float) -> float:
        """Adunare cu precizie controlată."""
        return round(a + b, self.precision)
    
    def subtract(self, a: float, b: float) -> float:
        """Scădere cu precizie controlată."""
        return round(a - b, self.precision)
    
    def multiply(self, a: float, b: float) -> float:
        """Înmulțire cu precizie controlată."""
        return round(a * b, self.precision)
    
    def divide(self, a: float, b: float) -> float:
        """
        Împărțire cu handling pentru zero.
        
        Raises:
            ZeroDivisionError: Dacă b == 0
        """
        if b == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return round(a / b, self.precision)
    
    def power(self, base: float, exponent: float) -> float:
        """
        Ridicare la putere.
        
        Raises:
            ValueError: Pentru cazuri matematice invalide
        """
        if base < 0 and not exponent.is_integer():
            raise ValueError("Cannot raise negative number to non-integer power")
        return round(base ** exponent, self.precision)
    
    def sqrt(self, x: float) -> float:
        """
        Rădăcină pătrată.
        
        Raises:
            ValueError: Pentru numere negative
        """
        if x < 0:
            raise ValueError("Cannot compute square root of negative number")
        return round(math.sqrt(x), self.precision)


# Exemple de teste pentru Calculator
class TestCalculator(unittest.TestCase):
    """
    Exemple de teste unitare pentru Calculator.
    
    Principii demonstrate:
    1. Arrange-Act-Assert (AAA) pattern
    2. Un assert per test (când posibil)
    3. Testare edge cases
    4. Denumire descriptivă
    """
    
    def setUp(self):
        """Setup executat înainte de FIECARE test."""
        self.calc = Calculator(precision=6)
    
    def tearDown(self):
        """Cleanup executat după FIECARE test."""
        pass  # Cleanup dacă e necesar
    
    # ─────────────────────────────────────────────────────────────────────────
    # Teste de bază
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_add_positive_numbers(self):
        """Adunarea numerelor pozitive."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Adunarea numerelor negative."""
        result = self.calc.add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_add_mixed_signs(self):
        """Adunarea numerelor cu semne diferite."""
        result = self.calc.add(-2, 3)
        self.assertEqual(result, 1)
    
    def test_add_with_zero(self):
        """Adunarea cu zero (element neutru)."""
        self.assertEqual(self.calc.add(5, 0), 5)
        self.assertEqual(self.calc.add(0, 5), 5)
        self.assertEqual(self.calc.add(0, 0), 0)
    
    def test_add_floats(self):
        """Adunarea numerelor cu virgulă."""
        result = self.calc.add(0.1, 0.2)
        self.assertAlmostEqual(result, 0.3, places=6)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Teste pentru excepții
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_divide_by_zero_raises(self):
        """Împărțirea la zero aruncă ZeroDivisionError."""
        with self.assertRaises(ZeroDivisionError) as context:
            self.calc.divide(10, 0)
        
        self.assertIn("Cannot divide by zero", str(context.exception))
    
    def test_sqrt_negative_raises(self):
        """Sqrt de număr negativ aruncă ValueError."""
        with self.assertRaises(ValueError):
            self.calc.sqrt(-1)
    
    def test_power_negative_base_non_integer_exp_raises(self):
        """Bază negativă la exponent non-întreg aruncă ValueError."""
        with self.assertRaises(ValueError):
            self.calc.power(-2, 0.5)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Edge cases
    # ─────────────────────────────────────────────────────────────────────────
    
    def test_divide_very_small_numbers(self):
        """Împărțirea numerelor foarte mici."""
        result = self.calc.divide(1e-10, 1e-5)
        self.assertAlmostEqual(result, 1e-5, places=10)
    
    def test_multiply_by_zero(self):
        """Înmulțirea cu zero returnează zero."""
        self.assertEqual(self.calc.multiply(1000000, 0), 0)
    
    def test_power_zero_exponent(self):
        """Orice număr la puterea 0 este 1."""
        self.assertEqual(self.calc.power(5, 0), 1)
        self.assertEqual(self.calc.power(-5, 0), 1)
        self.assertEqual(self.calc.power(0.5, 0), 1)
    
    def test_sqrt_of_zero(self):
        """Sqrt de zero este zero."""
        self.assertEqual(self.calc.sqrt(0), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: FIXTURES ȘI MOCKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataProcessor:
    """Procesor de date care depinde de servicii externe."""
    
    api_client: Any = None  # Dependență externă
    cache: dict[str, Any] = field(default_factory=dict)
    
    def fetch_data(self, endpoint: str) -> dict[str, Any]:
        """Fetch date de la API (pentru demonstrație)."""
        if self.api_client is None:
            raise RuntimeError("API client not configured")
        
        if endpoint in self.cache:
            return self.cache[endpoint]
        
        data = self.api_client.get(endpoint)
        self.cache[endpoint] = data
        return data
    
    def process_records(self, records: list[dict]) -> list[dict]:
        """Procesează o listă de înregistrări."""
        return [
            {**record, 'processed': True, 'id_hash': hash(record.get('id', ''))}
            for record in records
        ]
    
    def save_results(self, filepath: str, data: Any) -> bool:
        """Salvează rezultatele într-un fișier."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f)
            return True
        except Exception:
            return False


class TestDataProcessorWithMocking(unittest.TestCase):
    """
    Demonstrație: Mocking pentru teste izolate.
    
    Mocking permite:
    - Izolarea codului testat de dependențe externe
    - Controlul comportamentului dependențelor
    - Testarea scenariilor dificil de reprodus
    """
    
    def test_fetch_data_calls_api(self):
        """Verifică că API-ul este apelat corect."""
        # Arrange
        mock_client = Mock()
        mock_client.get.return_value = {'data': [1, 2, 3]}
        
        processor = DataProcessor(api_client=mock_client)
        
        # Act
        result = processor.fetch_data('/test')
        
        # Assert
        mock_client.get.assert_called_once_with('/test')
        self.assertEqual(result, {'data': [1, 2, 3]})
    
    def test_fetch_data_uses_cache(self):
        """Verifică că cache-ul este folosit."""
        mock_client = Mock()
        processor = DataProcessor(api_client=mock_client)
        processor.cache['/cached'] = {'cached': True}
        
        result = processor.fetch_data('/cached')
        
        # API-ul nu trebuie apelat
        mock_client.get.assert_not_called()
        self.assertEqual(result, {'cached': True})
    
    def test_process_records_adds_metadata(self):
        """Verifică adăugarea de metadata."""
        processor = DataProcessor()
        records = [{'id': 'abc', 'value': 1}]
        
        result = processor.process_records(records)
        
        self.assertTrue(result[0]['processed'])
        self.assertIn('id_hash', result[0])
    
    @patch('builtins.open', create=True)
    def test_save_results_handles_error(self, mock_open):
        """Verifică handling-ul erorilor la salvare."""
        mock_open.side_effect = IOError("Disk full")
        processor = DataProcessor()
        
        result = processor.save_results('/fake/path', {'data': 'test'})
        
        self.assertFalse(result)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: PYTEST FIXTURES (DOCUMENTAȚIE)
# ═══════════════════════════════════════════════════════════════════════════════

PYTEST_FIXTURES_EXAMPLE = '''
"""
Exemple de pytest fixtures pentru proiecte reale.

Fixtures oferă:
- Setup/teardown reutilizabil
- Dependency injection
- Parametrizare
- Scope control (function, class, module, session)
"""

import pytest
import tempfile
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Basic Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Fixture simplu care returnează date de test."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def calculator():
    """Fixture pentru Calculator cu configurație standard."""
    return Calculator(precision=6)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures cu setup/teardown
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_file():
    """Fixture care creează un fișier temporar și îl șterge după test."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("test content")
        filepath = f.name
    
    yield Path(filepath)  # Test-ul rulează aici
    
    # Cleanup
    Path(filepath).unlink(missing_ok=True)


@pytest.fixture
def temp_directory(tmp_path):
    """Fixture care creează o structură de directoare."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    (data_dir / "input.csv").write_text("a,b,c\\n1,2,3")
    (data_dir / "config.json").write_text('{"key": "value"}')
    
    yield data_dir
    
    # tmp_path se curăță automat de pytest


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures parametrizate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=[1, 2, 3])
def sample_number(request):
    """Fixture care rulează test-ul cu multiple valori."""
    return request.param


@pytest.fixture(params=['csv', 'json', 'parquet'])
def file_format(request):
    """Fixture pentru testarea multiple formate."""
    return request.param


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures cu scope
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def database_connection():
    """
    Fixture cu scope session - se creează o singură dată.
    Util pentru resurse costisitoare (conexiuni DB, modele ML).
    """
    print("Creating database connection...")
    connection = {"connected": True, "host": "localhost"}
    
    yield connection
    
    print("Closing database connection...")
    connection["connected"] = False


@pytest.fixture(scope="module")
def expensive_model():
    """Fixture cu scope module - se creează o dată per modul."""
    print("Loading expensive model...")
    model = {"loaded": True, "weights": [0.1, 0.2, 0.3]}
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures compuse
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def processor_with_mock_client():
    """Fixture care combină DataProcessor cu un client mock."""
    from unittest.mock import Mock
    
    mock_client = Mock()
    mock_client.get.return_value = {"success": True}
    
    processor = DataProcessor(api_client=mock_client)
    
    return processor, mock_client


# ─────────────────────────────────────────────────────────────────────────────
# conftest.py (fixtures partajate)
# ─────────────────────────────────────────────────────────────────────────────
"""
Fixtures definite în conftest.py sunt disponibile automat
în toate testele din acel director și subdirectoare.

# conftest.py
import pytest

@pytest.fixture(autouse=True)
def reset_random_seed():
    \"\"\"Auto-aplicat pentru reproducibilitate.\"\"\"
    import random
    random.seed(42)
    yield

@pytest.fixture
def shared_resource():
    \"\"\"Disponibil în toate testele.\"\"\"
    return {"shared": True}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Exemple de teste cu fixtures
# ─────────────────────────────────────────────────────────────────────────────

def test_with_sample_data(sample_data):
    """Test care folosește fixture sample_data."""
    assert len(sample_data) == 5
    assert sum(sample_data) == 15


def test_calculator_operations(calculator):
    """Test care folosește fixture calculator."""
    assert calculator.add(2, 3) == 5
    assert calculator.multiply(4, 5) == 20


def test_with_temp_file(temp_file):
    """Test care folosește un fișier temporar."""
    content = temp_file.read_text()
    assert "test content" in content


def test_parametrized(sample_number):
    """Rulează de 3 ori cu valori diferite."""
    assert sample_number > 0
    assert sample_number < 10


class TestWithFixtures:
    """Clasă de test cu fixtures."""
    
    def test_combined(self, calculator, sample_data):
        """Test cu multiple fixtures."""
        result = calculator.add(sample_data[0], sample_data[-1])
        assert result == 6
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: PROPERTY-BASED TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def property_test(func: Callable[[T], bool]) -> Callable[[], None]:
    """
    Decorator simplu pentru property-based testing.
    
    În practică, folosiți biblioteca Hypothesis!
    """
    @wraps(func)
    def wrapper():
        for _ in range(100):  # Hypothesis face mult mai multe
            # Generăm input random
            test_input = random.randint(-1000, 1000)
            try:
                assert func(test_input), f"Property failed for input: {test_input}"
            except AssertionError:
                raise
            except Exception as e:
                raise AssertionError(f"Exception for input {test_input}: {e}")
        print(f"Property '{func.__name__}' passed for 100 inputs")
    return wrapper


# Exemple de proprietăți
@property_test
def prop_addition_commutative(x: int) -> bool:
    """Adunarea este comutativă: a + b = b + a"""
    calc = Calculator()
    y = random.randint(-1000, 1000)
    return calc.add(x, y) == calc.add(y, x)


@property_test
def prop_addition_identity(x: int) -> bool:
    """Zero este element neutru: a + 0 = a"""
    calc = Calculator()
    return calc.add(x, 0) == x


@property_test
def prop_multiply_by_zero(x: int) -> bool:
    """Înmulțirea cu zero dă zero: a * 0 = 0"""
    calc = Calculator()
    return calc.multiply(x, 0) == 0


HYPOTHESIS_EXAMPLE = '''
"""
Exemple de property-based testing cu Hypothesis.

Hypothesis generează automat sute de cazuri de test,
inclusiv edge cases pe care nu le-ai fi considerat.

pip install hypothesis
"""

from hypothesis import given, strategies as st, assume, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

# ─────────────────────────────────────────────────────────────────────────────
# Basic Properties
# ─────────────────────────────────────────────────────────────────────────────

@given(st.integers(), st.integers())
def test_addition_is_commutative(a, b):
    """Adunarea este comutativă."""
    calc = Calculator()
    assert calc.add(a, b) == calc.add(b, a)


@given(st.integers())
def test_addition_identity(a):
    """Zero este element neutru."""
    calc = Calculator()
    assert calc.add(a, 0) == a


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_sqrt_of_square(x):
    """sqrt(x²) = |x|"""
    calc = Calculator()
    assume(abs(x) < 1e6)  # Evităm overflow
    
    squared = calc.power(x, 2)
    result = calc.sqrt(squared)
    
    assert abs(result - abs(x)) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Custom Strategies
# ─────────────────────────────────────────────────────────────────────────────

positive_floats = st.floats(min_value=0.01, max_value=1e6, allow_nan=False)
records = st.lists(
    st.fixed_dictionaries({
        'id': st.text(min_size=1, max_size=10),
        'value': st.integers(min_value=0, max_value=100)
    }),
    min_size=1,
    max_size=100
)


@given(positive_floats)
def test_sqrt_positive_result(x):
    """Sqrt de pozitiv este pozitiv."""
    calc = Calculator()
    result = calc.sqrt(x)
    assert result >= 0


@given(records)
def test_process_records_preserves_count(data):
    """Procesarea păstrează numărul de înregistrări."""
    processor = DataProcessor()
    result = processor.process_records(data)
    assert len(result) == len(data)


# ─────────────────────────────────────────────────────────────────────────────
# Stateful Testing
# ─────────────────────────────────────────────────────────────────────────────

class CalculatorStateMachine(RuleBasedStateMachine):
    """
    Testare stateful - verifică secvențe de operații.
    
    Hypothesis generează secvențe aleatorii de acțiuni
    și verifică invariantele la fiecare pas.
    """
    
    def __init__(self):
        super().__init__()
        self.calc = Calculator()
        self.value = 0.0
    
    @rule(x=st.floats(min_value=-100, max_value=100, allow_nan=False))
    def add_value(self, x):
        """Adaugă o valoare."""
        self.value = self.calc.add(self.value, x)
    
    @rule(x=st.floats(min_value=-100, max_value=100, allow_nan=False))
    def subtract_value(self, x):
        """Scade o valoare."""
        self.value = self.calc.subtract(self.value, x)
    
    @invariant()
    def value_is_finite(self):
        """Valoarea trebuie să fie finită."""
        assert not (math.isinf(self.value) or math.isnan(self.value))


# Pentru a rula: TestCalculatorState = CalculatorStateMachine.TestCase
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: CI/CD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GITHUB_ACTIONS_TEMPLATE = '''
# .github/workflows/ci.yml
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
      fail-fast: false
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Install dependencies
        run: |
          uv venv
          uv pip install -e ".[dev]"

      - name: Lint with Ruff
        run: |
          source .venv/bin/activate
          ruff check src/ tests/
          ruff format --check src/ tests/

      - name: Type check with MyPy
        run: |
          source .venv/bin/activate
          mypy src/ --ignore-missing-imports

      - name: Run tests
        run: |
          source .venv/bin/activate
          pytest tests/ -v --cov=src --cov-report=xml --cov-report=term

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
          fail_ci_if_error: false

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Security scan with Bandit
        run: |
          pip install bandit
          bandit -r src/ -ll

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build docs
        run: |
          pip install mkdocs mkdocs-material
          mkdocs build --strict
'''


PRE_COMMIT_CONFIG = '''
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-json
      - id: detect-private-key
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA VI: TEST COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CoverageReport:
    """Raport simplificat de coverage."""
    total_lines: int = 0
    covered_lines: int = 0
    missing_lines: list[int] = field(default_factory=list)
    
    @property
    def coverage_percent(self) -> float:
        if self.total_lines == 0:
            return 100.0
        return (self.covered_lines / self.total_lines) * 100
    
    def summary(self) -> str:
        return (
            f"Coverage: {self.coverage_percent:.1f}% "
            f"({self.covered_lines}/{self.total_lines} lines)\n"
            f"Missing lines: {self.missing_lines[:10]}..."
            if len(self.missing_lines) > 10 else
            f"Coverage: {self.coverage_percent:.1f}% "
            f"({self.covered_lines}/{self.total_lines} lines)\n"
            f"Missing lines: {self.missing_lines}"
        )


COVERAGE_CONFIG = '''
# pyproject.toml - Coverage configuration
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/__init__.py",
    "*/tests/*",
    "*/_version.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:",
    "if __name__ == .__main__.:",
    "@abstractmethod",
]
fail_under = 80
show_missing = true

[tool.coverage.html]
directory = "htmlcov"
'''


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_unit_tests() -> None:
    """Demonstrație: rulare teste unitare."""
    print("=" * 60)
    print("DEMO: Unit Tests")
    print("=" * 60)
    print()
    
    # Rulăm testele
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCalculator)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()


def demo_mocking() -> None:
    """Demonstrație: mocking."""
    print("=" * 60)
    print("DEMO: Mocking")
    print("=" * 60)
    print()
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDataProcessorWithMocking)
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    print()


def demo_property_tests() -> None:
    """Demonstrație: property-based testing."""
    print("=" * 60)
    print("DEMO: Property-Based Testing")
    print("=" * 60)
    print()
    
    print("Testing mathematical properties...")
    prop_addition_commutative()
    prop_addition_identity()
    prop_multiply_by_zero()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 7 LAB 2: TESTING & CI/CD")
    print("═" * 60 + "\n")
    
    demo_unit_tests()
    demo_mocking()
    demo_property_tests()
    
    print("=" * 60)
    print("Configurations available:")
    print("  - GITHUB_ACTIONS_TEMPLATE")
    print("  - PRE_COMMIT_CONFIG")
    print("  - PYTEST_FIXTURES_EXAMPLE")
    print("  - HYPOTHESIS_EXAMPLE")
    print("  - COVERAGE_CONFIG")
    print("=" * 60)
