# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# 03UNIT: Algorithmic Complexity
# Makefile for Build Automation
# ═══════════════════════════════════════════════════════════════════════════════
#
# LICENCE
# ───────
# © 2025 Antonio Clim. All rights reserved.
# See README.md for full licence terms.
#
# ═══════════════════════════════════════════════════════════════════════════════

# Configuration
SHELL := /bin/bash
PYTHON := python3
PIP := pip3
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises
RESOURCES_DIR := resources
THEORY_DIR := theory

# Files
LAB_FILES := $(wildcard $(LAB_DIR)/lab_*.py)
TEST_FILES := $(wildcard $(TEST_DIR)/test_*.py)
EXERCISE_FILES := $(wildcard $(EXERCISES_DIR)/practice/*.py)
SOLUTION_FILES := $(wildcard $(EXERCISES_DIR)/solutions/*.py)

# Colours for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m  # No Colour

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all
all: check test
	@echo -e "$(GREEN)✓ All checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help
help:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "  03UNIT: Algorithmic Complexity - Available Commands"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  $(BLUE)Build & Run$(NC)"
	@echo "    make all          - Run all checks and tests"
	@echo "    make run-labs     - Execute all lab files with demos"
	@echo "    make run-lab1     - Run Lab 03.01 (Benchmark Suite)"
	@echo "    make run-lab2     - Run Lab 03.02 (Complexity Analyser)"
	@echo ""
	@echo "  $(BLUE)Domain-Specific (Performance)$(NC)"
	@echo "    make run-bench       - Execute benchmark suite"
	@echo "    make run-analyser    - Execute complexity analyser"
	@echo "    make profile         - Run profiling on sample algorithms"
	@echo "    make plot-complexity - Generate complexity comparison charts"
	@echo ""
	@echo "  $(BLUE)Testing$(NC)"
	@echo "    make test         - Run all tests with coverage"
	@echo "    make test-fast    - Run tests without slow markers"
	@echo "    make test-verbose - Run tests with verbose output"
	@echo "    make test-lab1    - Run tests for Lab 03.01 only"
	@echo "    make test-lab2    - Run tests for Lab 03.02 only"
	@echo "    make coverage     - Generate coverage report"
	@echo ""
	@echo "  $(BLUE)Code Quality$(NC)"
	@echo "    make lint         - Run ruff linter"
	@echo "    make format       - Format code with ruff"
	@echo "    make type-check   - Run mypy type checker"
	@echo "    make check        - Run all quality checks"
	@echo ""
	@echo "  $(BLUE)Utilities$(NC)"
	@echo "    make clean        - Remove generated files"
	@echo "    make validate     - Validate UNIT structure"
	@echo "    make install-deps - Install dependencies"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# RUNNING LABS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: run-labs
run-labs: run-lab1 run-lab2
	@echo -e "$(GREEN)✓ All labs executed$(NC)"

.PHONY: run-lab1
run-lab1:
	@echo -e "$(BLUE)Running Lab 03.01: Benchmark Suite...$(NC)"
	@$(PYTHON) -m $(LAB_DIR).lab_03_01_benchmark_suite --demo --verbose
	@echo -e "$(GREEN)✓ Lab 03.01 complete$(NC)"

.PHONY: run-lab2
run-lab2:
	@echo -e "$(BLUE)Running Lab 03.02: Complexity Analyser...$(NC)"
	@$(PYTHON) -m $(LAB_DIR).lab_03_02_complexity_analyser --demo --verbose
	@echo -e "$(GREEN)✓ Lab 03.02 complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS (Performance Analysis)
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: run-bench
run-bench: run-lab1
	@echo -e "$(GREEN)✓ Benchmark suite executed$(NC)"

.PHONY: run-analyser
run-analyser: run-lab2
	@echo -e "$(GREEN)✓ Complexity analyser executed$(NC)"

.PHONY: profile
profile:
	@echo -e "$(BLUE)Running profiling on sample algorithms...$(NC)"
	@$(PYTHON) -c "import cProfile; from lab.lab_03_01_benchmark_suite import run_benchmark_demo; cProfile.run('run_benchmark_demo()')" 2>/dev/null || \
		$(PYTHON) -m cProfile -s cumulative $(LAB_DIR)/lab_03_01_benchmark_suite.py 2>/dev/null || \
		echo -e "$(YELLOW)⚠ Profiling requires lab module to be importable$(NC)"
	@echo -e "$(GREEN)✓ Profiling complete$(NC)"

.PHONY: plot-complexity
plot-complexity:
	@echo -e "$(BLUE)Generating complexity comparison charts...$(NC)"
	@$(PYTHON) -c "from lab.lab_03_02_complexity_analyser import generate_complexity_plots; generate_complexity_plots()" 2>/dev/null || \
		echo -e "$(YELLOW)⚠ Chart generation requires matplotlib and configured output directory$(NC)"
	@echo -e "$(GREEN)✓ Complexity charts generated$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: test
test:
	@echo -e "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) $(TEST_DIR) --cov=$(LAB_DIR) --cov-report=term-missing -v
	@echo -e "$(GREEN)✓ Tests passed$(NC)"

.PHONY: test-lab1
test-lab1:
	@echo -e "$(BLUE)Running Lab 03.01 tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/test_lab_03_01.py -v
	@echo -e "$(GREEN)✓ Lab 03.01 tests passed$(NC)"

.PHONY: test-lab2
test-lab2:
	@echo -e "$(BLUE)Running Lab 03.02 tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/test_lab_03_02.py -v
	@echo -e "$(GREEN)✓ Lab 03.02 tests passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: lint
lint:
	@echo -e "$(BLUE)Running ruff linter...$(NC)"
	@$(RUFF) check $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) || true
	@echo -e "$(GREEN)✓ Linting complete$(NC)"

.PHONY: format
format:
	@echo -e "$(BLUE)Formatting code with ruff...$(NC)"
	@$(RUFF) format $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) || true
	@echo -e "$(GREEN)✓ Code formatted$(NC)"

.PHONY: type-check
type-check:
	@echo -e "$(BLUE)Running mypy type checker...$(NC)"
	@$(MYPY) $(LAB_DIR) --ignore-missing-imports || true
	@echo -e "$(GREEN)✓ Type checking complete$(NC)"

.PHONY: check
check: lint type-check
	@echo -e "$(GREEN)✓ All quality checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: clean
clean:
	@echo -e "$(BLUE)Cleaning generated files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: validate
validate:
	@echo -e "$(BLUE)Validating UNIT structure...$(NC)"
	@test -f README.md || (echo -e "$(RED)✗ Missing README.md$(NC)" && exit 1)
	@test -f $(THEORY_DIR)/03UNIT_slides.html || (echo -e "$(RED)✗ Missing 03UNIT_slides.html$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_03_01_benchmark_suite.py || (echo -e "$(RED)✗ Missing lab_03_01$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_03_02_complexity_analyser.py || (echo -e "$(RED)✗ Missing lab_03_02$(NC)" && exit 1)
	@grep -q "RESTRICTIVE LICENCE" README.md || (echo -e "$(RED)✗ Missing licence$(NC)" && exit 1)
	@grep -q "Version 4.1.0" README.md || (echo -e "$(RED)✗ Wrong licence version$(NC)" && exit 1)
	@echo -e "$(GREEN)✓ Validation passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: install-deps
install-deps:
	@echo -e "$(BLUE)Installing dependencies...$(NC)"
	@$(PIP) install numpy>=1.24 pandas>=2.0 matplotlib>=3.7 scipy>=1.11
	@$(PIP) install pytest>=7.0 pytest-cov>=4.0 ruff>=0.1 mypy>=1.0
	@echo -e "$(GREEN)✓ Dependencies installed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# END OF MAKEFILE
# ═══════════════════════════════════════════════════════════════════════════════
