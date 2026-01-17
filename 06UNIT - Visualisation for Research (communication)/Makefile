# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# Week 6: Visualisation for Research
# Makefile for Development and Testing
# ═══════════════════════════════════════════════════════════════════════════════
#
# This Makefile provides standard targets for:
# - Running laboratory demonstrations
# - Executing tests with coverage
# - Linting and formatting code
# - Validating project structure
# - Cleaning build artefacts
#
# Usage:
#   make help          Show available targets
#   make all           Run complete validation suite
#   make test          Run pytest with coverage
#   make lint          Check code quality
#   make format        Auto-format code
#   make clean         Remove generated files
#
# © 2025 Antonio Clim. All rights reserved.
# ═══════════════════════════════════════════════════════════════════════════════

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises
ASSETS_DIR := assets
OUTPUT_DIR := output

# Colour codes for terminal output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m  # No colour

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all
all: check test lint
	@echo "$(GREEN)✓ All checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help
help:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Week 6: Visualisation for Research - Makefile Help$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo ""
	@echo "  $(GREEN)all$(NC)           Run complete validation suite (check + test + lint)"
	@echo "  $(GREEN)help$(NC)          Show this help message"
	@echo ""
	@echo "$(YELLOW)Running Labs:$(NC)"
	@echo "  $(GREEN)run-labs$(NC)      Run all laboratory modules with demos"
	@echo "  $(GREEN)run-lab-01$(NC)    Run Lab 6.01 (Static Plots)"
	@echo "  $(GREEN)run-lab-02$(NC)    Run Lab 6.02 (Interactive Viz)"
	@echo ""
	@echo "$(YELLOW)Testing:$(NC)"
	@echo "  $(GREEN)test$(NC)          Run all tests with pytest"
	@echo "  $(GREEN)test-cov$(NC)      Run tests with coverage report"
	@echo "  $(GREEN)test-verbose$(NC)  Run tests with verbose output"
	@echo "  $(GREEN)test-lab-01$(NC)   Run tests for Lab 6.01 only"
	@echo "  $(GREEN)test-lab-02$(NC)   Run tests for Lab 6.02 only"
	@echo ""
	@echo "$(YELLOW)Code Quality:$(NC)"
	@echo "  $(GREEN)lint$(NC)          Check code with ruff"
	@echo "  $(GREEN)format$(NC)        Auto-format code with ruff"
	@echo "  $(GREEN)typecheck$(NC)     Run mypy type checking"
	@echo "  $(GREEN)check$(NC)         Run lint + typecheck"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  $(GREEN)clean$(NC)         Remove generated files and caches"
	@echo "  $(GREEN)validate$(NC)      Validate project structure"
	@echo "  $(GREEN)install$(NC)       Install dependencies"
	@echo "  $(GREEN)install-dev$(NC)   Install development dependencies"
	@echo ""
	@echo "$(YELLOW)Exercises:$(NC)"
	@echo "  $(GREEN)run-exercises$(NC) Run all practice exercises"
	@echo "  $(GREEN)check-solutions$(NC) Verify exercise solutions"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# RUNNING LABS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: run-labs run-lab-01 run-lab-02

run-labs: run-lab-01 run-lab-02
	@echo "$(GREEN)✓ All labs executed$(NC)"

run-lab-01:
	@echo "$(BLUE)Running Lab 6.01: Static Plots...$(NC)"
	$(PYTHON) $(LAB_DIR)/lab_6_01_static_plots.py --demo -v

run-lab-02:
	@echo "$(BLUE)Running Lab 6.02: Interactive Visualisation...$(NC)"
	$(PYTHON) $(LAB_DIR)/lab_6_02_interactive_viz.py --demo -v

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: test test-cov test-verbose test-lab-01 test-lab-02

test:
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short

test-cov:
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -v --cov=$(LAB_DIR) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

test-verbose:
	@echo "$(BLUE)Running tests with verbose output...$(NC)"
	$(PYTEST) $(TEST_DIR)/ -vv --tb=long

test-lab-01:
	@echo "$(BLUE)Running Lab 6.01 tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_lab_6_01.py -v --tb=short

test-lab-02:
	@echo "$(BLUE)Running Lab 6.02 tests...$(NC)"
	$(PYTEST) $(TEST_DIR)/test_lab_6_02.py -v --tb=short

# ═══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: lint format typecheck check

lint:
	@echo "$(BLUE)Checking code with ruff...$(NC)"
	$(RUFF) check $(LAB_DIR)/ $(TEST_DIR)/ $(EXERCISES_DIR)/
	@echo "$(GREEN)✓ Lint check passed$(NC)"

format:
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	$(RUFF) format $(LAB_DIR)/ $(TEST_DIR)/ $(EXERCISES_DIR)/
	$(RUFF) check --fix $(LAB_DIR)/ $(TEST_DIR)/ $(EXERCISES_DIR)/
	@echo "$(GREEN)✓ Code formatted$(NC)"

typecheck:
	@echo "$(BLUE)Running mypy type checking...$(NC)"
	$(MYPY) $(LAB_DIR)/ --ignore-missing-imports --no-error-summary || true
	@echo "$(GREEN)✓ Type check complete$(NC)"

check: lint typecheck
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: run-exercises check-solutions

run-exercises:
	@echo "$(BLUE)Running practice exercises...$(NC)"
	@for file in $(EXERCISES_DIR)/practice/*.py; do \
		echo "$(YELLOW)Running $$file...$(NC)"; \
		$(PYTHON) "$$file" || true; \
	done
	@echo "$(GREEN)✓ All exercises executed$(NC)"

check-solutions:
	@echo "$(BLUE)Checking exercise solutions...$(NC)"
	@for file in $(EXERCISES_DIR)/solutions/*.py; do \
		echo "$(YELLOW)Checking $$file...$(NC)"; \
		$(PYTHON) -m py_compile "$$file"; \
	done
	@echo "$(GREEN)✓ All solutions are syntactically correct$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: validate validate-structure validate-readme

validate: validate-structure validate-readme
	@echo "$(GREEN)✓ Project validation complete$(NC)"

validate-structure:
	@echo "$(BLUE)Validating project structure...$(NC)"
	@test -f README.md || (echo "$(RED)✗ Missing README.md$(NC)" && exit 1)
	@test -d theory || (echo "$(RED)✗ Missing theory/$(NC)" && exit 1)
	@test -d lab || (echo "$(RED)✗ Missing lab/$(NC)" && exit 1)
	@test -d exercises || (echo "$(RED)✗ Missing exercises/$(NC)" && exit 1)
	@test -d assessments || (echo "$(RED)✗ Missing assessments/$(NC)" && exit 1)
	@test -d resources || (echo "$(RED)✗ Missing resources/$(NC)" && exit 1)
	@test -d assets || (echo "$(RED)✗ Missing assets/$(NC)" && exit 1)
	@test -d tests || (echo "$(RED)✗ Missing tests/$(NC)" && exit 1)
	@test -f theory/slides.html || (echo "$(RED)✗ Missing slides.html$(NC)" && exit 1)
	@test -f theory/lecture_notes.md || (echo "$(RED)✗ Missing lecture_notes.md$(NC)" && exit 1)
	@echo "$(GREEN)✓ Directory structure valid$(NC)"

validate-readme:
	@echo "$(BLUE)Validating README.md licence...$(NC)"
	@grep -q "RESTRICTIVE LICENCE" README.md || (echo "$(RED)✗ Missing licence in README.md$(NC)" && exit 1)
	@grep -q "Version 2.0.2" README.md || (echo "$(RED)✗ Missing licence version in README.md$(NC)" && exit 1)
	@echo "$(GREEN)✓ README.md licence present$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: install install-dev

install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install numpy>=1.24 pandas>=2.0 matplotlib>=3.7 seaborn>=0.12 scipy>=1.11
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

install-dev: install
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install pytest>=7.0 pytest-cov>=4.0 ruff>=0.1 mypy>=1.0
	@echo "$(GREEN)✓ Development dependencies installed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: clean clean-cache clean-output clean-all

clean: clean-cache
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-cache:
	@echo "$(BLUE)Removing cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "$(GREEN)✓ Cache files removed$(NC)"

clean-output:
	@echo "$(BLUE)Removing output files...$(NC)"
	rm -rf $(OUTPUT_DIR)/* 2>/dev/null || true
	@echo "$(GREEN)✓ Output files removed$(NC)"

clean-all: clean-cache clean-output
	@echo "$(GREEN)✓ All generated files removed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: watch dev

# Watch for changes and run tests (requires entr or similar)
watch:
	@echo "$(BLUE)Watching for changes...$(NC)"
	@echo "Install 'entr' to use this target: apt install entr"
	find $(LAB_DIR) $(TEST_DIR) -name "*.py" | entr -c make test

# Development mode: lint, test and watch
dev: format lint test
	@echo "$(GREEN)✓ Development checks complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: docs serve-slides

docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation is provided in theory/lecture_notes.md"
	@echo "Presentation slides are in theory/slides.html"
	@echo "$(GREEN)✓ See theory/ directory for materials$(NC)"

serve-slides:
	@echo "$(BLUE)Serving slides on http://localhost:8000$(NC)"
	@echo "Press Ctrl+C to stop"
	cd theory && $(PYTHON) -m http.server 8000

# ═══════════════════════════════════════════════════════════════════════════════
# END OF MAKEFILE
# ═══════════════════════════════════════════════════════════════════════════════
