# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# Week 1: The Epistemology of Computation
# Makefile
# ═══════════════════════════════════════════════════════════════════════════════
#
# LICENCE
# ───────
# © 2025 Antonio Clim. All rights reserved.
# See README.md for full licence terms.
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all help run-labs test lint format check clean validate install

# Default Python interpreter
PYTHON ?= python3

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises

# Lab files
LAB_1_01 := $(LAB_DIR)/lab_1_01_turing_machine.py
LAB_1_02 := $(LAB_DIR)/lab_1_02_lambda_calculus.py
LAB_1_03 := $(LAB_DIR)/lab_1_03_ast_interpreter.py

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

all: check test
	@echo "✓ All checks passed"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "Week 1: The Epistemology of Computation — Makefile"
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Run all checks and tests (default)"
	@echo "  help       Show this help message"
	@echo "  run-labs   Run all lab demonstrations"
	@echo "  test       Run pytest test suite"
	@echo "  lint       Run linting checks (ruff)"
	@echo "  format     Format code (ruff format)"
	@echo "  check      Run type checking (mypy) and linting"
	@echo "  clean      Remove generated files and caches"
	@echo "  validate   Full validation (lint + type check + test)"
	@echo "  install    Install dependencies"
	@echo ""
	@echo "Lab-specific targets:"
	@echo "  run-lab-1  Run Turing machine demonstrations"
	@echo "  run-lab-2  Run Lambda calculus demonstrations"
	@echo "  run-lab-3  Run AST interpreter demonstrations"
	@echo "  repl       Start the AST interpreter REPL"
	@echo ""
	@echo "Examples:"
	@echo "  make run-labs          # Run all demonstrations"
	@echo "  make test              # Run test suite"
	@echo "  make validate          # Full validation before submission"
	@echo "═══════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN LABS
# ═══════════════════════════════════════════════════════════════════════════════

run-labs: run-lab-1 run-lab-2 run-lab-3
	@echo ""
	@echo "✓ All lab demonstrations completed"

run-lab-1:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "Lab 1: Turing Machine Simulator"
	@echo "═══════════════════════════════════════════════════════════════════════"
	$(PYTHON) $(LAB_1_01) --demo

run-lab-2:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "Lab 2: Lambda Calculus Basics"
	@echo "═══════════════════════════════════════════════════════════════════════"
	$(PYTHON) $(LAB_1_02) --demo

run-lab-3:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "Lab 3: AST Interpreter"
	@echo "═══════════════════════════════════════════════════════════════════════"
	$(PYTHON) $(LAB_1_03) --demo

repl:
	@echo "Starting AST Interpreter REPL..."
	@echo "Type ':help' for commands, ':quit' to exit"
	$(PYTHON) $(LAB_1_03) --repl

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

test:
	@echo ""
	@echo "Running pytest test suite..."
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short

test-verbose:
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=long

test-coverage:
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing

test-fast:
	$(PYTHON) -m pytest $(TEST_DIR) -v -m "not slow"

# ═══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

lint:
	@echo ""
	@echo "Running ruff linter..."
	$(PYTHON) -m ruff check $(LAB_DIR) $(TEST_DIR)

lint-fix:
	@echo ""
	@echo "Running ruff linter with auto-fix..."
	$(PYTHON) -m ruff check $(LAB_DIR) $(TEST_DIR) --fix

format:
	@echo ""
	@echo "Formatting code with ruff..."
	$(PYTHON) -m ruff format $(LAB_DIR) $(TEST_DIR)

format-check:
	@echo ""
	@echo "Checking code formatting..."
	$(PYTHON) -m ruff format --check $(LAB_DIR) $(TEST_DIR)

typecheck:
	@echo ""
	@echo "Running mypy type checker..."
	$(PYTHON) -m mypy $(LAB_DIR) --ignore-missing-imports

check: lint typecheck
	@echo ""
	@echo "✓ All code quality checks passed"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION (Full pre-submission check)
# ═══════════════════════════════════════════════════════════════════════════════

validate: format-check lint typecheck test
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "✓ VALIDATION COMPLETE"
	@echo "  All code quality checks passed"
	@echo "  All tests passed"
	@echo "  Ready for submission"
	@echo "═══════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

install:
	@echo "Installing dependencies..."
	pip install --break-system-packages pytest pytest-cov ruff mypy numpy

install-dev:
	@echo "Installing development dependencies..."
	pip install --break-system-packages pytest pytest-cov ruff mypy numpy hypothesis

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__
	rm -rf $(LAB_DIR)/__pycache__
	rm -rf $(TEST_DIR)/__pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf *.egg-info
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

docs:
	@echo "Generating documentation..."
	@echo "(Documentation generation not yet configured)"

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Quick smoke test
smoke:
	$(PYTHON) -c "from lab import *; print('✓ All imports successful')"

# Show lab file line counts
stats:
	@echo "Line counts:"
	@wc -l $(LAB_1_01) $(LAB_1_02) $(LAB_1_03)

# Verify British English spelling in comments
spelling:
	@echo "Checking for American spellings..."
	@grep -rn "color\|behavior\|analyze\|recognize\|organize" $(LAB_DIR) || echo "✓ No American spellings found"
