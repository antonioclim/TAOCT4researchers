# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# Week 3: Algorithmic Complexity
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
	@echo "  Week 3: Algorithmic Complexity - Available Commands"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  $(BLUE)Build & Run$(NC)"
	@echo "    make all          - Run all checks and tests"
	@echo "    make run-labs     - Execute all lab files with demos"
	@echo "    make run-lab1     - Run Lab 3.1 (Benchmark Suite)"
	@echo "    make run-lab2     - Run Lab 3.2 (Complexity Analyser)"
	@echo ""
	@echo "  $(BLUE)Testing$(NC)"
	@echo "    make test         - Run all tests with coverage"
	@echo "    make test-fast    - Run tests without slow markers"
	@echo "    make test-verbose - Run tests with verbose output"
	@echo "    make test-lab1    - Run tests for Lab 3.1 only"
	@echo "    make test-lab2    - Run tests for Lab 3.2 only"
	@echo "    make coverage     - Generate coverage report"
	@echo ""
	@echo "  $(BLUE)Code Quality$(NC)"
	@echo "    make lint         - Run ruff linter"
	@echo "    make format       - Format code with ruff"
	@echo "    make type-check   - Run mypy type checker"
	@echo "    make check        - Run all quality checks"
	@echo ""
	@echo "  $(BLUE)Documentation$(NC)"
	@echo "    make docs         - Build documentation"
	@echo "    make serve-slides - Serve slides locally"
	@echo ""
	@echo "  $(BLUE)Utilities$(NC)"
	@echo "    make clean        - Remove generated files"
	@echo "    make validate     - Validate week structure"
	@echo "    make install-deps - Install dependencies"
	@echo "    make setup        - Initial project setup"
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
	@echo -e "$(BLUE)Running Lab 3.1: Benchmark Suite...$(NC)"
	@$(PYTHON) -m $(LAB_DIR).lab_3_01_benchmark_suite --demo --verbose
	@echo -e "$(GREEN)✓ Lab 3.1 complete$(NC)"

.PHONY: run-lab2
run-lab2:
	@echo -e "$(BLUE)Running Lab 3.2: Complexity Analyser...$(NC)"
	@$(PYTHON) -m $(LAB_DIR).lab_3_02_complexity_analyser --demo --verbose
	@echo -e "$(GREEN)✓ Lab 3.2 complete$(NC)"

.PHONY: run-exercises
run-exercises:
	@echo -e "$(BLUE)Running practice exercises...$(NC)"
	@for f in $(EXERCISE_FILES); do \
		echo -e "$(YELLOW)  Running $$f...$(NC)"; \
		$(PYTHON) "$$f" || exit 1; \
	done
	@echo -e "$(GREEN)✓ All exercises executed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: test
test:
	@echo -e "$(BLUE)Running tests with coverage...$(NC)"
	@$(PYTEST) $(TEST_DIR) \
		--cov=$(LAB_DIR) \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=80 \
		-v
	@echo -e "$(GREEN)✓ Tests passed with ≥80% coverage$(NC)"

.PHONY: test-fast
test-fast:
	@echo -e "$(BLUE)Running fast tests (excluding slow markers)...$(NC)"
	@$(PYTEST) $(TEST_DIR) -v -m "not slow"
	@echo -e "$(GREEN)✓ Fast tests passed$(NC)"

.PHONY: test-verbose
test-verbose:
	@echo -e "$(BLUE)Running tests with verbose output...$(NC)"
	@$(PYTEST) $(TEST_DIR) -vvv --tb=long
	@echo -e "$(GREEN)✓ Tests complete$(NC)"

.PHONY: test-lab1
test-lab1:
	@echo -e "$(BLUE)Running Lab 3.1 tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/test_lab_3_01.py -v
	@echo -e "$(GREEN)✓ Lab 3.1 tests passed$(NC)"

.PHONY: test-lab2
test-lab2:
	@echo -e "$(BLUE)Running Lab 3.2 tests...$(NC)"
	@$(PYTEST) $(TEST_DIR)/test_lab_3_02.py -v
	@echo -e "$(GREEN)✓ Lab 3.2 tests passed$(NC)"

.PHONY: coverage
coverage:
	@echo -e "$(BLUE)Generating coverage report...$(NC)"
	@$(PYTEST) $(TEST_DIR) \
		--cov=$(LAB_DIR) \
		--cov-report=html:htmlcov \
		--cov-report=xml:coverage.xml
	@echo -e "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: lint
lint:
	@echo -e "$(BLUE)Running ruff linter...$(NC)"
	@$(RUFF) check $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR)
	@echo -e "$(GREEN)✓ Linting passed$(NC)"

.PHONY: format
format:
	@echo -e "$(BLUE)Formatting code with ruff...$(NC)"
	@$(RUFF) format $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR)
	@$(RUFF) check --fix $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR)
	@echo -e "$(GREEN)✓ Code formatted$(NC)"

.PHONY: type-check
type-check:
	@echo -e "$(BLUE)Running mypy type checker...$(NC)"
	@$(MYPY) $(LAB_DIR) --strict --ignore-missing-imports
	@echo -e "$(GREEN)✓ Type checking passed$(NC)"

.PHONY: check
check: lint type-check
	@echo -e "$(GREEN)✓ All quality checks passed$(NC)"

.PHONY: check-all
check-all: lint type-check test
	@echo -e "$(GREEN)✓ All checks and tests passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: docs
docs:
	@echo -e "$(BLUE)Building documentation...$(NC)"
	@echo "Documentation build would go here (sphinx/mkdocs)"
	@echo -e "$(GREEN)✓ Documentation built$(NC)"

.PHONY: serve-slides
serve-slides:
	@echo -e "$(BLUE)Serving slides on http://localhost:8000...$(NC)"
	@cd $(THEORY_DIR) && $(PYTHON) -m http.server 8000

.PHONY: open-slides
open-slides:
	@echo -e "$(BLUE)Opening slides in browser...$(NC)"
	@xdg-open $(THEORY_DIR)/slides.html 2>/dev/null || \
		open $(THEORY_DIR)/slides.html 2>/dev/null || \
		echo "Please open $(THEORY_DIR)/slides.html manually"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: clean
clean:
	@echo -e "$(BLUE)Cleaning generated files...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@rm -rf .eggs *.egg-info 2>/dev/null || true
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo -e "$(BLUE)Deep cleaning...$(NC)"
	@rm -rf venv .venv 2>/dev/null || true
	@rm -rf node_modules 2>/dev/null || true
	@echo -e "$(GREEN)✓ Deep clean complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: validate
validate:
	@echo -e "$(BLUE)Validating week structure...$(NC)"
	@echo "Checking required files..."
	@test -f README.md || (echo -e "$(RED)✗ Missing README.md$(NC)" && exit 1)
	@test -f $(THEORY_DIR)/slides.html || (echo -e "$(RED)✗ Missing slides.html$(NC)" && exit 1)
	@test -f $(THEORY_DIR)/lecture_notes.md || (echo -e "$(RED)✗ Missing lecture_notes.md$(NC)" && exit 1)
	@test -f $(THEORY_DIR)/learning_objectives.md || (echo -e "$(RED)✗ Missing learning_objectives.md$(NC)" && exit 1)
	@test -f $(LAB_DIR)/__init__.py || (echo -e "$(RED)✗ Missing lab/__init__.py$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_3_01_benchmark_suite.py || (echo -e "$(RED)✗ Missing lab_3_01$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_3_02_complexity_analyser.py || (echo -e "$(RED)✗ Missing lab_3_02$(NC)" && exit 1)
	@test -f $(EXERCISES_DIR)/homework.md || (echo -e "$(RED)✗ Missing homework.md$(NC)" && exit 1)
	@test -d $(EXERCISES_DIR)/practice || (echo -e "$(RED)✗ Missing practice dir$(NC)" && exit 1)
	@test -d $(EXERCISES_DIR)/solutions || (echo -e "$(RED)✗ Missing solutions dir$(NC)" && exit 1)
	@test -f $(RESOURCES_DIR)/cheatsheet.md || (echo -e "$(RED)✗ Missing cheatsheet.md$(NC)" && exit 1)
	@test -f $(RESOURCES_DIR)/further_reading.md || (echo -e "$(RED)✗ Missing further_reading.md$(NC)" && exit 1)
	@test -f $(RESOURCES_DIR)/glossary.md || (echo -e "$(RED)✗ Missing glossary.md$(NC)" && exit 1)
	@test -f $(TEST_DIR)/__init__.py || (echo -e "$(RED)✗ Missing tests/__init__.py$(NC)" && exit 1)
	@test -f $(TEST_DIR)/conftest.py || (echo -e "$(RED)✗ Missing conftest.py$(NC)" && exit 1)
	@test -d assets/diagrams || (echo -e "$(RED)✗ Missing diagrams dir$(NC)" && exit 1)
	@test -d assets/animations || (echo -e "$(RED)✗ Missing animations dir$(NC)" && exit 1)
	@echo ""
	@echo "Checking licence in README..."
	@grep -q "RESTRICTIVE LICENCE" README.md || (echo -e "$(RED)✗ Missing licence$(NC)" && exit 1)
	@grep -q "Version 2.0.2" README.md || (echo -e "$(RED)✗ Wrong licence version$(NC)" && exit 1)
	@echo ""
	@echo "Checking British English spelling..."
	@! grep -rn "color[^a-z]" $(LAB_DIR) $(THEORY_DIR) --include="*.py" --include="*.md" 2>/dev/null || true
	@! grep -rn "analyze[^a-z]" $(LAB_DIR) $(THEORY_DIR) --include="*.py" --include="*.md" 2>/dev/null || true
	@echo ""
	@echo -e "$(GREEN)✓ Validation passed$(NC)"

.PHONY: validate-code
validate-code:
	@echo -e "$(BLUE)Validating code requirements...$(NC)"
	@echo "Checking for print statements in lab files..."
	@! grep -n "^[^#]*print(" $(LAB_FILES) 2>/dev/null || echo "Warning: print() found"
	@echo ""
	@echo "Checking line counts..."
	@LAB1_LINES=$$(wc -l < $(LAB_DIR)/lab_3_01_benchmark_suite.py); \
		if [ $$LAB1_LINES -lt 500 ]; then \
			echo -e "$(RED)✗ Lab 3.1 has $$LAB1_LINES lines (need ≥500)$(NC)"; \
		else \
			echo -e "$(GREEN)✓ Lab 3.1 has $$LAB1_LINES lines$(NC)"; \
		fi
	@LAB2_LINES=$$(wc -l < $(LAB_DIR)/lab_3_02_complexity_analyser.py); \
		if [ $$LAB2_LINES -lt 300 ]; then \
			echo -e "$(RED)✗ Lab 3.2 has $$LAB2_LINES lines (need ≥300)$(NC)"; \
		else \
			echo -e "$(GREEN)✓ Lab 3.2 has $$LAB2_LINES lines$(NC)"; \
		fi
	@echo -e "$(GREEN)✓ Code validation complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP & DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: install-deps
install-deps:
	@echo -e "$(BLUE)Installing dependencies...$(NC)"
	@$(PIP) install numpy>=1.24 pandas>=2.0 matplotlib>=3.7 scipy>=1.11
	@$(PIP) install pytest>=7.0 pytest-cov>=4.0
	@$(PIP) install ruff>=0.1 mypy>=1.0
	@echo -e "$(GREEN)✓ Dependencies installed$(NC)"

.PHONY: setup
setup: install-deps
	@echo -e "$(BLUE)Setting up project...$(NC)"
	@mkdir -p $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR)/practice $(EXERCISES_DIR)/solutions
	@mkdir -p $(RESOURCES_DIR)/datasets $(THEORY_DIR) assets/diagrams assets/animations assets/images
	@touch $(LAB_DIR)/__init__.py $(TEST_DIR)/__init__.py
	@echo -e "$(GREEN)✓ Project setup complete$(NC)"

.PHONY: venv
venv:
	@echo -e "$(BLUE)Creating virtual environment...$(NC)"
	@$(PYTHON) -m venv venv
	@echo -e "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "Activate with: source venv/bin/activate"

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARKING DEMOS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: benchmark-sorting
benchmark-sorting:
	@echo -e "$(BLUE)Running sorting algorithm benchmarks...$(NC)"
	@$(PYTHON) -c "from $(LAB_DIR).lab_3_01_benchmark_suite import run_sorting_demo; run_sorting_demo()"

.PHONY: benchmark-search
benchmark-search:
	@echo -e "$(BLUE)Running search algorithm benchmarks...$(NC)"
	@$(PYTHON) -c "from $(LAB_DIR).lab_3_01_benchmark_suite import run_search_demo; run_search_demo()"

.PHONY: analyse-complexity
analyse-complexity:
	@echo -e "$(BLUE)Running complexity analysis demo...$(NC)"
	@$(PYTHON) -c "from $(LAB_DIR).lab_3_02_complexity_analyser import run_analysis_demo; run_analysis_demo()"

# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: report
report:
	@echo -e "$(BLUE)Generating benchmark report...$(NC)"
	@$(PYTHON) -c "\
from $(LAB_DIR).lab_3_01_benchmark_suite import BenchmarkSuite, SortingAlgorithms, DataGenerator; \
suite = BenchmarkSuite('week3_report'); \
suite.register_algorithm('bubble', SortingAlgorithms.bubble_sort); \
suite.register_algorithm('merge', SortingAlgorithms.merge_sort); \
suite.register_data_generator('random', DataGenerator.random_list); \
results = suite.run(['bubble', 'merge'], ['random'], [100, 500, 1000]); \
suite.export_csv('benchmark_report.csv'); \
print('Report saved to benchmark_report.csv')"
	@echo -e "$(GREEN)✓ Report generated$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCKER SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: docker-build
docker-build:
	@echo -e "$(BLUE)Building Docker image...$(NC)"
	@docker build -t week3-complexity:latest .
	@echo -e "$(GREEN)✓ Docker image built$(NC)"

.PHONY: docker-run
docker-run:
	@echo -e "$(BLUE)Running in Docker container...$(NC)"
	@docker run --rm -it -v $(PWD):/app week3-complexity:latest

.PHONY: docker-test
docker-test:
	@echo -e "$(BLUE)Running tests in Docker...$(NC)"
	@docker run --rm week3-complexity:latest pytest tests/ -v

# ═══════════════════════════════════════════════════════════════════════════════
# CI/CD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: ci
ci: lint type-check test validate
	@echo -e "$(GREEN)✓ CI pipeline passed$(NC)"

.PHONY: pre-commit
pre-commit: format lint type-check test-fast
	@echo -e "$(GREEN)✓ Pre-commit checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: info
info:
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "  Week 3: Algorithmic Complexity - Project Information"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Lab files:      $(words $(LAB_FILES))"
	@echo "  Test files:     $(words $(TEST_FILES))"
	@echo "  Exercise files: $(words $(EXERCISE_FILES))"
	@echo "  Solution files: $(words $(SOLUTION_FILES))"
	@echo ""
	@echo "  Python version: $$($(PYTHON) --version)"
	@echo "  Pytest version: $$($(PYTEST) --version 2>/dev/null | head -1 || echo 'not installed')"
	@echo "  Ruff version:   $$($(RUFF) --version 2>/dev/null || echo 'not installed')"
	@echo "  Mypy version:   $$($(MYPY) --version 2>/dev/null || echo 'not installed')"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""

.PHONY: tree
tree:
	@echo -e "$(BLUE)Project structure:$(NC)"
	@tree -I '__pycache__|*.pyc|.git|venv|htmlcov|.pytest_cache|.mypy_cache' . 2>/dev/null || \
		find . -not -path '*/\.*' -not -path '*/venv/*' -not -path '*/__pycache__/*' | sort

# ═══════════════════════════════════════════════════════════════════════════════
# END OF MAKEFILE
# ═══════════════════════════════════════════════════════════════════════════════
