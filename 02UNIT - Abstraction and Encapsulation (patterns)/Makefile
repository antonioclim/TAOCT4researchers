# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# Week 2: Abstraction and Encapsulation
# Makefile
# ═══════════════════════════════════════════════════════════════════════════════
#
# © 2025 Antonio Clim. All rights reserved.
#
# Usage:
#   make help       - Show this help message
#   make all        - Run full validation pipeline
#   make test       - Run all tests
#   make lint       - Check code style
#   make format     - Format code
#   make run-labs   - Run all lab demonstrations
#
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all help run-labs test lint format check clean validate install

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
RUFF := ruff
MYPY := mypy

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises/practice
SOLUTIONS_DIR := exercises/solutions

# Files
LAB_FILES := $(wildcard $(LAB_DIR)/lab_*.py)
TEST_FILES := $(wildcard $(TEST_DIR)/test_*.py)
EXERCISE_FILES := $(wildcard $(EXERCISES_DIR)/*.py)
SOLUTION_FILES := $(wildcard $(SOLUTIONS_DIR)/*.py)
ALL_PY_FILES := $(LAB_FILES) $(TEST_FILES) $(EXERCISE_FILES) $(SOLUTION_FILES)

# Colours for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
BLUE := \033[0;34m
NC := \033[0m # No Colour

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

all: lint check test
	@echo "$(GREEN)✓ All checks passed!$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════$(NC)"
	@echo "$(BLUE)  Week 2: Abstraction and Encapsulation - Makefile$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo ""
	@echo "  $(GREEN)make all$(NC)        - Run full validation pipeline (lint, check, test)"
	@echo "  $(GREEN)make help$(NC)       - Show this help message"
	@echo "  $(GREEN)make install$(NC)    - Install dependencies"
	@echo ""
	@echo "  $(GREEN)make run-labs$(NC)   - Run all lab demonstrations"
	@echo "  $(GREEN)make lab1$(NC)       - Run Lab 2.01 (Simulation Framework)"
	@echo "  $(GREEN)make lab2$(NC)       - Run Lab 2.02 (Design Patterns)"
	@echo ""
	@echo "  $(GREEN)make test$(NC)       - Run all tests with pytest"
	@echo "  $(GREEN)make test-cov$(NC)   - Run tests with coverage report"
	@echo "  $(GREEN)make test-v$(NC)     - Run tests with verbose output"
	@echo ""
	@echo "  $(GREEN)make lint$(NC)       - Check code style with ruff"
	@echo "  $(GREEN)make format$(NC)     - Format code with ruff"
	@echo "  $(GREEN)make check$(NC)      - Type check with mypy"
	@echo ""
	@echo "  $(GREEN)make validate$(NC)   - Validate week structure"
	@echo "  $(GREEN)make clean$(NC)      - Remove generated files"
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

install:
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install numpy matplotlib pytest pytest-cov ruff mypy
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN LABS
# ═══════════════════════════════════════════════════════════════════════════════

run-labs: lab1 lab2
	@echo "$(GREEN)✓ All labs executed$(NC)"

lab1:
	@echo "$(BLUE)Running Lab 2.01: Simulation Framework$(NC)"
	@echo "────────────────────────────────────────"
	$(PYTHON) $(LAB_DIR)/lab_02_01_simulation_framework.py --demo
	@echo ""

lab2:
	@echo "$(BLUE)Running Lab 2.02: Design Patterns$(NC)"
	@echo "────────────────────────────────────────"
	$(PYTHON) $(LAB_DIR)/lab_02_02_design_patterns.py --demo
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

test:
	@echo "$(YELLOW)Running tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-cov:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-v:
	@echo "$(YELLOW)Running tests (verbose)...$(NC)"
	$(PYTEST) $(TEST_DIR) -vv --tb=long

test-quick:
	@echo "$(YELLOW)Running quick tests...$(NC)"
	$(PYTEST) $(TEST_DIR) -v -x -q

# ═══════════════════════════════════════════════════════════════════════════════
# LINTING AND FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

lint:
	@echo "$(YELLOW)Checking code style...$(NC)"
	$(RUFF) check $(LAB_DIR)/*.py $(TEST_DIR)/*.py
	@echo "$(GREEN)✓ Code style OK$(NC)"

format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(RUFF) format $(LAB_DIR)/*.py $(TEST_DIR)/*.py
	$(RUFF) check --fix $(LAB_DIR)/*.py $(TEST_DIR)/*.py
	@echo "$(GREEN)✓ Code formatted$(NC)"

check:
	@echo "$(YELLOW)Type checking...$(NC)"
	$(MYPY) $(LAB_DIR)/*.py --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking passed$(NC)"

check-strict:
	@echo "$(YELLOW)Strict type checking...$(NC)"
	$(MYPY) $(LAB_DIR)/*.py --strict --ignore-missing-imports
	@echo "$(GREEN)✓ Strict type checking passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

validate:
	@echo "$(YELLOW)Validating week structure...$(NC)"
	@echo "Checking required files..."
	@test -f README.md || (echo "$(RED)✗ Missing README.md$(NC)" && exit 1)
	@test -f theory/slides.html || (echo "$(RED)✗ Missing theory/slides.html$(NC)" && exit 1)
	@test -f theory/lecture_notes.md || (echo "$(RED)✗ Missing theory/lecture_notes.md$(NC)" && exit 1)
	@test -f theory/learning_objectives.md || (echo "$(RED)✗ Missing theory/learning_objectives.md$(NC)" && exit 1)
	@test -f exercises/homework.md || (echo "$(RED)✗ Missing exercises/homework.md$(NC)" && exit 1)
	@test -f assessments/quiz.md || (echo "$(RED)✗ Missing assessments/quiz.md$(NC)" && exit 1)
	@test -f assessments/rubric.md || (echo "$(RED)✗ Missing assessments/rubric.md$(NC)" && exit 1)
	@test -f resources/cheatsheet.md || (echo "$(RED)✗ Missing resources/cheatsheet.md$(NC)" && exit 1)
	@test -d assets/diagrams || (echo "$(RED)✗ Missing assets/diagrams/$(NC)" && exit 1)
	@test -d tests || (echo "$(RED)✗ Missing tests/$(NC)" && exit 1)
	@echo "Checking licence in README..."
	@grep -q "RESTRICTIVE LICENCE" README.md || (echo "$(RED)✗ Missing licence in README.md$(NC)" && exit 1)
	@grep -q "Version 2.0.2" README.md || (echo "$(RED)✗ Missing licence version 2.0.2$(NC)" && exit 1)
	@echo "$(GREEN)✓ Week structure valid$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf .mypy_cache */.mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov .coverage
	rm -rf *.egg-info
	rm -rf dist build
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".DS_Store" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

run-exercises:
	@echo "$(YELLOW)Running practice exercises...$(NC)"
	@for f in $(EXERCISE_FILES); do \
		echo "$(BLUE)Running $$f$(NC)"; \
		$(PYTHON) $$f || true; \
		echo ""; \
	done

run-solutions:
	@echo "$(YELLOW)Running solutions...$(NC)"
	@for f in $(SOLUTION_FILES); do \
		echo "$(BLUE)Running $$f$(NC)"; \
		$(PYTHON) $$f || true; \
		echo ""; \
	done

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

docs:
	@echo "$(YELLOW)Opening documentation...$(NC)"
	@echo "Opening slides..."
	@command -v xdg-open > /dev/null && xdg-open theory/slides.html || \
		command -v open > /dev/null && open theory/slides.html || \
		echo "Please open theory/slides.html manually"

serve-slides:
	@echo "$(YELLOW)Serving slides on http://localhost:8000$(NC)"
	cd theory && $(PYTHON) -m http.server 8000

# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT
# ═══════════════════════════════════════════════════════════════════════════════

dev: format lint check test
	@echo "$(GREEN)✓ Development checks passed$(NC)"

watch:
	@echo "$(YELLOW)Watching for changes...$(NC)"
	@echo "Install watchexec for file watching: cargo install watchexec"
	watchexec -e py "make lint check"
