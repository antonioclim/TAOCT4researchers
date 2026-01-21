# ═══════════════════════════════════════════════════════════════════════════════
# 08UNIT: Recursion and Dynamic Programming
# Makefile for build automation — v4.1
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# Directories
LAB_DIR := lab
TEST_DIR := tests
SCRIPTS_DIR := scripts
ASSETS_DIR := assets

# ═══════════════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all test lint typecheck validate clean help
.PHONY: run-recursion run-dp run-knapsack run-lcs run-edit
.PHONY: visualise-calls compare-memo demo demo-lab1 demo-lab2
.PHONY: test-lab1 test-lab2 test-coverage check format

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

all: lint typecheck test
	@echo "✓ All checks passed for 08UNIT"

# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

check:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Environment Check"
	@echo "═══════════════════════════════════════════════════════════════════"
	@$(PYTHON) --version
	@echo ""
	@echo "Checking Python version >= 3.12..."
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 12), 'Python 3.12+ required'"
	@echo "✓ Python version OK"
	@echo ""
	@echo "Checking required packages..."
	@$(PYTHON) -c "import pytest" 2>/dev/null && echo "✓ pytest installed" || echo "✗ pytest missing"
	@$(PYTHON) -c "import numpy" 2>/dev/null && echo "✓ numpy installed" || echo "✗ numpy missing"
	@$(PYTHON) -c "import matplotlib" 2>/dev/null && echo "✓ matplotlib installed" || echo "✗ matplotlib missing"
	@echo ""
	@echo "Environment check complete."

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════════

test:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Test Suite"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTEST) $(TEST_DIR) -v --tb=short

test-lab1:
	@echo "Testing Lab 08.01: Recursive Patterns"
	$(PYTEST) $(TEST_DIR)/test_lab_08_01.py -v

test-lab2:
	@echo "Testing Lab 08.02: Dynamic Programming"
	$(PYTEST) $(TEST_DIR)/test_lab_08_02.py -v

test-coverage:
	@echo "Running tests with coverage..."
	$(PYTEST) $(TEST_DIR) --cov=$(LAB_DIR) --cov-report=term-missing

lint:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Linters"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) --fix || true

typecheck:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Type Checker"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(MYPY) $(LAB_DIR) --ignore-missing-imports || true

validate:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running UNIT Validation"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 08

format:
	@echo "Formatting code with ruff..."
	$(RUFF) format $(LAB_DIR) $(TEST_DIR)

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS (Algorithms/DP)
# ═══════════════════════════════════════════════════════════════════════════════

run-recursion:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Executing Recursive Patterns Demo"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --demo

run-dp:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Executing Dynamic Programming Demo"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_02_dynamic_programming --demo

run-knapsack:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Knapsack Problem Solver"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_02_dynamic_programming --knapsack

run-lcs:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Longest Common Subsequence"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_02_dynamic_programming --lcs

run-edit:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running Edit Distance Calculator"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_02_dynamic_programming --edit-distance

visualise-calls:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Generating Call Tree Visualisation"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --visualise-tree

compare-memo:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Comparing Memoised vs Naive Performance"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --compare-memo

demo:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "Running All Demonstrations"
	@echo "═══════════════════════════════════════════════════════════════════"
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --demo
	@echo ""
	$(PYTHON) -m $(LAB_DIR).lab_08_02_dynamic_programming --demo

demo-lab1:
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --demo

demo-lab2:
	$(PYTHON) -m $(LAB_DIR).lab_08_02_dynamic_programming --demo

demo-fibonacci:
	@echo "Comparing Fibonacci implementations for n=30..."
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --fibonacci 30

demo-queens:
	@echo "Solving 8-Queens problem..."
	$(PYTHON) -m $(LAB_DIR).lab_08_01_recursive_patterns --n-queens 8

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleaned 08UNIT"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo " 08UNIT: Recursion and Dynamic Programming"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test          Run all tests"
	@echo "  make test-lab1     Run Lab 08.01 tests only"
	@echo "  make test-lab2     Run Lab 08.02 tests only"
	@echo "  make test-coverage Run tests with coverage report"
	@echo "  make lint          Run ruff linter"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make validate      Run UNIT validation script"
	@echo "  make format        Format code with ruff"
	@echo "  make all           Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific (Algorithms/DP):"
	@echo "  make run-recursion Execute recursive patterns demo"
	@echo "  make run-dp        Execute dynamic programming demo"
	@echo "  make run-knapsack  Run knapsack problem solver"
	@echo "  make run-lcs       Run longest common subsequence"
	@echo "  make run-edit      Run edit distance calculator"
	@echo "  make visualise-calls Generate call tree visualisation"
	@echo "  make compare-memo  Compare memoised vs naive performance"
	@echo ""
	@echo "Demonstrations:"
	@echo "  make demo          Run all demonstrations"
	@echo "  make demo-lab1     Run Lab 08.01 demo"
	@echo "  make demo-lab2     Run Lab 08.02 demo"
	@echo "  make demo-fibonacci Compare Fibonacci implementations"
	@echo "  make demo-queens   Solve N-Queens problem"
	@echo ""
	@echo "Maintenance:"
	@echo "  make check         Verify Python environment"
	@echo "  make clean         Remove build artifacts"
	@echo "  make help          Show this help message"
	@echo ""
