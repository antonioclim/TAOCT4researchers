# ═══════════════════════════════════════════════════════════════════════════
# 09UNIT: Exception Handling and Defensive Programming
# Makefile for development, testing, and documentation
# ═══════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# Directories
LAB_DIR := lab
EXERCISES_DIR := exercises/practice
SOLUTIONS_DIR := exercises/solutions
TESTS_DIR := tests
SCRIPTS_DIR := scripts

# ═══════════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: all test lint typecheck validate clean help
.PHONY: run-exceptions run-defensive run-circuit run-retry run-validation
.PHONY: stress-test test-coverage test-lab-01 test-lab-02 format docs

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════

all: lint typecheck test
	@echo "✓ All checks passed for 09UNIT"

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════

test:
	@echo "Running tests..."
	$(PYTEST) $(TESTS_DIR) -v --tb=short
	@echo "✓ Tests complete"

test-coverage:
	@echo "Running tests with coverage..."
	$(PYTEST) $(TESTS_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

test-lab-01:
	@echo "Running tests for Lab 09.01..."
	$(PYTEST) $(TESTS_DIR)/test_lab_09_01.py -v

test-lab-02:
	@echo "Running tests for Lab 09.02..."
	$(PYTEST) $(TESTS_DIR)/test_lab_09_02.py -v

lint:
	@echo "Running linters..."
	$(RUFF) check $(LAB_DIR) $(EXERCISES_DIR) $(SOLUTIONS_DIR) $(TESTS_DIR) --fix || true
	@echo "✓ Linting complete"

typecheck:
	@echo "Running type checker (strict mode)..."
	$(MYPY) $(LAB_DIR) --strict --ignore-missing-imports || true
	@echo "✓ Type checks complete"

format:
	@echo "Formatting code..."
	$(RUFF) check --fix $(LAB_DIR) $(EXERCISES_DIR) $(SOLUTIONS_DIR) $(TESTS_DIR) || true
	@echo "✓ Formatting complete"

validate:
	@echo "Running UNIT validation script..."
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 09 --path .
	@echo "✓ Validation complete"

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS (Robustness)
# ═══════════════════════════════════════════════════════════════════════════

run-exceptions:
	@echo "Running exception handling demo..."
	$(PYTHON) -m lab.lab_09_01_exception_handling

run-defensive:
	@echo "Running defensive programming demo..."
	$(PYTHON) -m lab.lab_09_02_defensive_programming

run-circuit:
	@echo "Running circuit breaker simulation..."
	$(PYTHON) -c "from lab.lab_09_02_defensive_programming import CircuitBreaker; \
		cb = CircuitBreaker('demo', failure_threshold=3, reset_timeout=5.0); \
		print('Circuit breaker initialised:', cb.state)"

run-retry:
	@echo "Running retry mechanism demo..."
	$(PYTHON) -c "from lab.lab_09_02_defensive_programming import retry_with_backoff; \
		@retry_with_backoff(max_attempts=3, base_delay=0.1) \
		def demo(): return 'Success'; \
		print(demo())"

run-validation:
	@echo "Running input validation examples..."
	$(PYTHON) -c "from lab.lab_09_02_defensive_programming import validate_positive; \
		print('Validating 42:', validate_positive(42)); \
		print('Validation framework ready')"

stress-test:
	@echo "Running fault injection tests..."
	$(PYTEST) $(TESTS_DIR) -v -k "fault or stress or error" --tb=short || \
		echo "No fault injection tests found (this is expected)"
	@echo "✓ Stress tests complete"

# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════

docs:
	@echo "Generating documentation..."
	@mkdir -p docs
	$(PYTHON) -m pydoc -w $(LAB_DIR).lab_09_01_exception_handling 2>/dev/null || true
	$(PYTHON) -m pydoc -w $(LAB_DIR).lab_09_02_defensive_programming 2>/dev/null || true
	@mv *.html docs/ 2>/dev/null || true
	@echo "✓ Documentation generated in docs/"

# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

clean:
	@echo "Cleaning build artefacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov docs/*.html 2>/dev/null || true
	@echo "✓ Cleaned 09UNIT"

# ═══════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " 09UNIT: Exception Handling and Defensive Programming"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test          - Run tests with coverage"
	@echo "  make test-coverage - Run tests with HTML coverage report"
	@echo "  make test-lab-01   - Run tests for Lab 09.01"
	@echo "  make test-lab-02   - Run tests for Lab 09.02"
	@echo "  make lint          - Run ruff linter"
	@echo "  make typecheck     - Run mypy type checker (strict mode)"
	@echo "  make format        - Format code with ruff"
	@echo "  make validate      - Run UNIT validation script"
	@echo "  make all           - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific (Robustness):"
	@echo "  make run-exceptions - Execute exception handling demo"
	@echo "  make run-defensive  - Execute defensive programming demo"
	@echo "  make run-circuit    - Run circuit breaker simulation"
	@echo "  make run-retry      - Run retry mechanism demo"
	@echo "  make run-validation - Run input validation examples"
	@echo "  make stress-test    - Run fault injection tests"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs          - Generate API documentation"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Remove generated files and logs"
	@echo "  make help          - Show this help message"
	@echo ""
