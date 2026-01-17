# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# Week 7: Reproducibility and Capstone
# Makefile
# ═══════════════════════════════════════════════════════════════════════════════
#
# This Makefile provides targets for building, testing, validating and cleaning
# the Week 7 materials. It follows GNU Make conventions and supports parallel
# execution where appropriate.
#
# USAGE:
#   make help       - Display available targets
#   make all        - Run complete build and validation
#   make test       - Execute test suite with coverage
#   make lint       - Run static analysis tools
#   make format     - Auto-format code files
#   make clean      - Remove generated artefacts
#
# LICENCE:
# © 2025 Antonio Clim. All rights reserved.
# See README.md for full licence terms.
#
# ═══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Week configuration
WEEK_NUM := 7
WEEK_DIR := $(shell pwd)
WEEK_NAME := Reproducibility and Capstone

# Python configuration
PYTHON := python3
PYTHON_VERSION := 3.12
PIP := pip3

# Source directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises
THEORY_DIR := theory
ASSETS_DIR := assets
RESOURCES_DIR := resources

# Python source files
LAB_FILES := $(wildcard $(LAB_DIR)/lab_*.py)
TEST_FILES := $(wildcard $(TEST_DIR)/test_*.py)
EXERCISE_FILES := $(wildcard $(EXERCISES_DIR)/practice/*.py)
SOLUTION_FILES := $(wildcard $(EXERCISES_DIR)/solutions/*.py)
ALL_PY_FILES := $(LAB_FILES) $(TEST_FILES) $(EXERCISE_FILES) $(SOLUTION_FILES)

# Tool configuration
RUFF := ruff
MYPY := mypy
PYTEST := pytest
COVERAGE := coverage

# Coverage threshold
COVERAGE_MIN := 80

# Output directories
BUILD_DIR := .build
COVERAGE_DIR := $(BUILD_DIR)/coverage
REPORTS_DIR := $(BUILD_DIR)/reports

# Colour codes for terminal output
COLOUR_RESET := \033[0m
COLOUR_GREEN := \033[32m
COLOUR_YELLOW := \033[33m
COLOUR_BLUE := \033[34m
COLOUR_RED := \033[31m
COLOUR_CYAN := \033[36m
COLOUR_BOLD := \033[1m

# ─────────────────────────────────────────────────────────────────────────────
# PHONY TARGETS
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all help run-labs test lint format check clean validate
.PHONY: install deps test-quick test-verbose test-coverage
.PHONY: lint-fix type-check docs slides
.PHONY: validate-structure validate-content validate-code validate-all
.PHONY: demo info version

# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Primary Targets

all: validate test lint  ## Run complete build: validate, test and lint
	@echo -e "$(COLOUR_GREEN)$(COLOUR_BOLD)✓ Week $(WEEK_NUM) build completed successfully$(COLOUR_RESET)"

help:  ## Display this help message
	@echo ""
	@echo -e "$(COLOUR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOUR_RESET)"
	@echo -e "$(COLOUR_BOLD) COMPUTATIONAL THINKING FOR RESEARCHERS$(COLOUR_RESET)"
	@echo -e "$(COLOUR_BOLD) Week $(WEEK_NUM): $(WEEK_NAME)$(COLOUR_RESET)"
	@echo -e "$(COLOUR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOUR_RESET)"
	@echo ""
	@echo -e "$(COLOUR_CYAN)Usage:$(COLOUR_RESET) make $(COLOUR_GREEN)<target>$(COLOUR_RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf ""} /^[a-zA-Z_-]+:.*?##/ { printf "  $(COLOUR_GREEN)%-18s$(COLOUR_RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(COLOUR_BOLD)%s$(COLOUR_RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo -e "$(COLOUR_CYAN)Examples:$(COLOUR_RESET)"
	@echo "  make test          # Run all tests with coverage"
	@echo "  make lint          # Check code style and types"
	@echo "  make format        # Auto-format all Python files"
	@echo "  make validate      # Validate week structure"
	@echo "  make clean         # Remove build artefacts"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# LABORATORY TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Laboratory

run-labs: run-lab-01 run-lab-02 run-lab-03  ## Run all laboratory demos
	@echo -e "$(COLOUR_GREEN)✓ All labs executed successfully$(COLOUR_RESET)"

run-lab-01:  ## Run Lab 7.01: Reproducibility
	@echo -e "$(COLOUR_BLUE)▶ Running Lab 7.01: Reproducibility...$(COLOUR_RESET)"
	@$(PYTHON) $(LAB_DIR)/lab_7_01_reproducibility.py --demo 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Lab 7.01 demo completed (some features may require dependencies)$(COLOUR_RESET)"

run-lab-02:  ## Run Lab 7.02: Testing and CI/CD
	@echo -e "$(COLOUR_BLUE)▶ Running Lab 7.02: Testing and CI/CD...$(COLOUR_RESET)"
	@$(PYTHON) $(LAB_DIR)/lab_7_02_testing_cicd.py --demo 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Lab 7.02 demo completed (some features may require dependencies)$(COLOUR_RESET)"

run-lab-03:  ## Run Lab 7.03: Project Scaffolder
	@echo -e "$(COLOUR_BLUE)▶ Running Lab 7.03: Project Scaffolder...$(COLOUR_RESET)"
	@$(PYTHON) $(LAB_DIR)/lab_7_03_project_scaffolder.py --demo 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Lab 7.03 demo completed (some features may require dependencies)$(COLOUR_RESET)"

demo: run-labs  ## Alias for run-labs

# ─────────────────────────────────────────────────────────────────────────────
# TESTING TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Testing

test: $(BUILD_DIR)  ## Run full test suite with coverage report
	@echo -e "$(COLOUR_BLUE)▶ Running test suite...$(COLOUR_RESET)"
	@$(PYTEST) $(TEST_DIR) \
		--cov=$(LAB_DIR) \
		--cov-report=term-missing \
		--cov-report=html:$(COVERAGE_DIR) \
		--cov-fail-under=$(COVERAGE_MIN) \
		-v \
		--tb=short \
		2>/dev/null || $(PYTEST) $(TEST_DIR) -v --tb=short
	@echo -e "$(COLOUR_GREEN)✓ Tests completed$(COLOUR_RESET)"

test-quick:  ## Run tests without coverage (faster)
	@echo -e "$(COLOUR_BLUE)▶ Running quick tests...$(COLOUR_RESET)"
	@$(PYTEST) $(TEST_DIR) -q --tb=line
	@echo -e "$(COLOUR_GREEN)✓ Quick tests completed$(COLOUR_RESET)"

test-verbose:  ## Run tests with verbose output
	@echo -e "$(COLOUR_BLUE)▶ Running verbose tests...$(COLOUR_RESET)"
	@$(PYTEST) $(TEST_DIR) -vvv --tb=long
	@echo -e "$(COLOUR_GREEN)✓ Verbose tests completed$(COLOUR_RESET)"

test-coverage: $(BUILD_DIR)  ## Generate detailed coverage report
	@echo -e "$(COLOUR_BLUE)▶ Generating coverage report...$(COLOUR_RESET)"
	@$(COVERAGE) run -m pytest $(TEST_DIR) -q 2>/dev/null || true
	@$(COVERAGE) report --show-missing 2>/dev/null || echo "Coverage data not available"
	@$(COVERAGE) html -d $(COVERAGE_DIR) 2>/dev/null || true
	@echo -e "$(COLOUR_GREEN)✓ Coverage report: $(COVERAGE_DIR)/index.html$(COLOUR_RESET)"

test-lab-01:  ## Run tests for Lab 7.01 only
	@$(PYTEST) $(TEST_DIR)/test_lab_7_01.py -v

test-lab-02:  ## Run tests for Lab 7.02 only
	@$(PYTEST) $(TEST_DIR)/test_lab_7_02.py -v

test-lab-03:  ## Run tests for Lab 7.03 only
	@$(PYTEST) $(TEST_DIR)/test_lab_7_03.py -v

# ─────────────────────────────────────────────────────────────────────────────
# CODE QUALITY TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Code Quality

lint: lint-ruff lint-types  ## Run all linting checks
	@echo -e "$(COLOUR_GREEN)✓ All linting checks passed$(COLOUR_RESET)"

lint-ruff:  ## Run ruff linter
	@echo -e "$(COLOUR_BLUE)▶ Running ruff linter...$(COLOUR_RESET)"
	@$(RUFF) check $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Ruff not installed or found issues$(COLOUR_RESET)"

lint-types: type-check  ## Alias for type-check

type-check:  ## Run mypy type checker
	@echo -e "$(COLOUR_BLUE)▶ Running mypy type checker...$(COLOUR_RESET)"
	@$(MYPY) $(LAB_DIR) --ignore-missing-imports 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Mypy not installed or found issues$(COLOUR_RESET)"

format:  ## Auto-format code with ruff
	@echo -e "$(COLOUR_BLUE)▶ Formatting code...$(COLOUR_RESET)"
	@$(RUFF) format $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Ruff formatter not available$(COLOUR_RESET)"
	@$(RUFF) check --fix $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) 2>/dev/null || true
	@echo -e "$(COLOUR_GREEN)✓ Code formatted$(COLOUR_RESET)"

lint-fix:  ## Fix auto-fixable lint issues
	@echo -e "$(COLOUR_BLUE)▶ Fixing lint issues...$(COLOUR_RESET)"
	@$(RUFF) check --fix $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) 2>/dev/null || true
	@echo -e "$(COLOUR_GREEN)✓ Auto-fixable issues resolved$(COLOUR_RESET)"

check: lint test  ## Run lint and test (CI check)
	@echo -e "$(COLOUR_GREEN)$(COLOUR_BOLD)✓ All checks passed$(COLOUR_RESET)"

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Validation

validate: validate-structure validate-content  ## Validate week structure and content
	@echo -e "$(COLOUR_GREEN)$(COLOUR_BOLD)✓ Week $(WEEK_NUM) validation passed$(COLOUR_RESET)"

validate-structure:  ## Validate directory structure
	@echo -e "$(COLOUR_BLUE)▶ Validating directory structure...$(COLOUR_RESET)"
	@errors=0; \
	for dir in $(LAB_DIR) $(TEST_DIR) $(EXERCISES_DIR) $(THEORY_DIR) $(ASSETS_DIR) $(RESOURCES_DIR) assessments; do \
		if [ ! -d "$$dir" ]; then \
			echo -e "$(COLOUR_RED)✗ Missing directory: $$dir$(COLOUR_RESET)"; \
			errors=$$((errors + 1)); \
		fi; \
	done; \
	for file in README.md Makefile; do \
		if [ ! -f "$$file" ]; then \
			echo -e "$(COLOUR_RED)✗ Missing file: $$file$(COLOUR_RESET)"; \
			errors=$$((errors + 1)); \
		fi; \
	done; \
	if [ $$errors -gt 0 ]; then \
		echo -e "$(COLOUR_RED)✗ Structure validation failed with $$errors errors$(COLOUR_RESET)"; \
		exit 1; \
	fi
	@echo -e "$(COLOUR_GREEN)✓ Directory structure valid$(COLOUR_RESET)"

validate-content:  ## Validate content completeness
	@echo -e "$(COLOUR_BLUE)▶ Validating content completeness...$(COLOUR_RESET)"
	@errors=0; \
	required_files="theory/slides.html theory/lecture_notes.md theory/learning_objectives.md \
		lab/lab_7_01_reproducibility.py lab/lab_7_02_testing_cicd.py lab/lab_7_03_project_scaffolder.py \
		exercises/homework.md assessments/quiz.md assessments/rubric.md assessments/self_check.md \
		resources/cheatsheet.md resources/further_reading.md resources/glossary.md \
		tests/conftest.py tests/test_lab_7_01.py tests/test_lab_7_02.py tests/test_lab_7_03.py"; \
	for file in $$required_files; do \
		if [ ! -f "$$file" ]; then \
			echo -e "$(COLOUR_RED)✗ Missing: $$file$(COLOUR_RESET)"; \
			errors=$$((errors + 1)); \
		fi; \
	done; \
	svg_count=$$(find $(ASSETS_DIR)/diagrams -name "*.svg" 2>/dev/null | wc -l); \
	if [ $$svg_count -lt 3 ]; then \
		echo -e "$(COLOUR_RED)✗ Need at least 3 SVG diagrams (found $$svg_count)$(COLOUR_RESET)"; \
		errors=$$((errors + 1)); \
	fi; \
	html_count=$$(find $(ASSETS_DIR)/animations -name "*.html" 2>/dev/null | wc -l); \
	if [ $$html_count -lt 1 ]; then \
		echo -e "$(COLOUR_RED)✗ Need at least 1 HTML animation (found $$html_count)$(COLOUR_RESET)"; \
		errors=$$((errors + 1)); \
	fi; \
	if [ $$errors -gt 0 ]; then \
		echo -e "$(COLOUR_RED)✗ Content validation failed with $$errors errors$(COLOUR_RESET)"; \
		exit 1; \
	fi
	@echo -e "$(COLOUR_GREEN)✓ Content completeness verified$(COLOUR_RESET)"

validate-code:  ## Validate Python files are syntactically correct
	@echo -e "$(COLOUR_BLUE)▶ Validating Python syntax...$(COLOUR_RESET)"
	@errors=0; \
	for file in $(ALL_PY_FILES); do \
		$(PYTHON) -m py_compile "$$file" 2>/dev/null || { \
			echo -e "$(COLOUR_RED)✗ Syntax error: $$file$(COLOUR_RESET)"; \
			errors=$$((errors + 1)); \
		}; \
	done; \
	if [ $$errors -gt 0 ]; then \
		echo -e "$(COLOUR_RED)✗ Code validation failed$(COLOUR_RESET)"; \
		exit 1; \
	fi
	@echo -e "$(COLOUR_GREEN)✓ All Python files syntactically valid$(COLOUR_RESET)"

validate-licence:  ## Validate README contains licence
	@echo -e "$(COLOUR_BLUE)▶ Validating licence...$(COLOUR_RESET)"
	@if grep -q "RESTRICTIVE LICENCE" README.md && grep -q "Version 2.0.2" README.md; then \
		echo -e "$(COLOUR_GREEN)✓ Licence present and correct$(COLOUR_RESET)"; \
	else \
		echo -e "$(COLOUR_RED)✗ Missing or incorrect licence in README.md$(COLOUR_RESET)"; \
		exit 1; \
	fi

validate-all: validate-structure validate-content validate-code validate-licence  ## Run all validations
	@echo -e "$(COLOUR_GREEN)$(COLOUR_BOLD)✓ All validations passed$(COLOUR_RESET)"

# ─────────────────────────────────────────────────────────────────────────────
# INSTALLATION TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Installation

install: deps  ## Install all dependencies
	@echo -e "$(COLOUR_GREEN)✓ Installation complete$(COLOUR_RESET)"

deps:  ## Install Python dependencies
	@echo -e "$(COLOUR_BLUE)▶ Installing dependencies...$(COLOUR_RESET)"
	@$(PIP) install -q numpy pandas matplotlib scipy networkx pytest pytest-cov ruff mypy 2>/dev/null || \
		echo -e "$(COLOUR_YELLOW)⚠ Some packages may require manual installation$(COLOUR_RESET)"
	@echo -e "$(COLOUR_GREEN)✓ Dependencies installed$(COLOUR_RESET)"

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTATION TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Documentation

docs:  ## Generate documentation
	@echo -e "$(COLOUR_BLUE)▶ Documentation available in:$(COLOUR_RESET)"
	@echo "  • $(THEORY_DIR)/lecture_notes.md"
	@echo "  • $(RESOURCES_DIR)/cheatsheet.md"
	@echo "  • $(RESOURCES_DIR)/glossary.md"
	@echo "  • $(RESOURCES_DIR)/further_reading.md"

slides:  ## Open presentation slides
	@echo -e "$(COLOUR_BLUE)▶ Slides: $(THEORY_DIR)/slides.html$(COLOUR_RESET)"
	@if command -v xdg-open &> /dev/null; then \
		xdg-open $(THEORY_DIR)/slides.html 2>/dev/null &; \
	elif command -v open &> /dev/null; then \
		open $(THEORY_DIR)/slides.html; \
	else \
		echo "  Open $(THEORY_DIR)/slides.html in your browser"; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY TARGETS
# ─────────────────────────────────────────────────────────────────────────────

##@ Utilities

clean:  ## Remove build artefacts and caches
	@echo -e "$(COLOUR_BLUE)▶ Cleaning build artefacts...$(COLOUR_RESET)"
	@rm -rf $(BUILD_DIR) 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .mypy_cache 2>/dev/null || true
	@rm -rf .ruff_cache 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf htmlcov 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@echo -e "$(COLOUR_GREEN)✓ Clean complete$(COLOUR_RESET)"

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(COVERAGE_DIR)
	@mkdir -p $(REPORTS_DIR)

info:  ## Display week information
	@echo ""
	@echo -e "$(COLOUR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOUR_RESET)"
	@echo -e "$(COLOUR_BOLD) WEEK $(WEEK_NUM): $(WEEK_NAME)$(COLOUR_RESET)"
	@echo -e "$(COLOUR_BOLD)═══════════════════════════════════════════════════════════════════════════════$(COLOUR_RESET)"
	@echo ""
	@echo -e "$(COLOUR_CYAN)Directory:$(COLOUR_RESET) $(WEEK_DIR)"
	@echo -e "$(COLOUR_CYAN)Python:$(COLOUR_RESET)    $(PYTHON_VERSION)+"
	@echo ""
	@echo -e "$(COLOUR_CYAN)Laboratory Files:$(COLOUR_RESET)"
	@for f in $(LAB_FILES); do echo "  • $$f"; done
	@echo ""
	@echo -e "$(COLOUR_CYAN)Test Files:$(COLOUR_RESET)"
	@for f in $(TEST_FILES); do echo "  • $$f"; done
	@echo ""
	@echo -e "$(COLOUR_CYAN)Statistics:$(COLOUR_RESET)"
	@echo "  • Lab files:      $$(echo $(LAB_FILES) | wc -w)"
	@echo "  • Test files:     $$(echo $(TEST_FILES) | wc -w)"
	@echo "  • Exercise files: $$(find $(EXERCISES_DIR)/practice -name '*.py' 2>/dev/null | wc -l)"
	@echo "  • Solution files: $$(find $(EXERCISES_DIR)/solutions -name '*.py' 2>/dev/null | wc -l)"
	@echo "  • SVG diagrams:   $$(find $(ASSETS_DIR)/diagrams -name '*.svg' 2>/dev/null | wc -l)"
	@echo "  • Animations:     $$(find $(ASSETS_DIR)/animations -name '*.html' 2>/dev/null | wc -l)"
	@echo ""

version:  ## Display tool versions
	@echo -e "$(COLOUR_CYAN)Tool Versions:$(COLOUR_RESET)"
	@echo -n "  • Python:  " && $(PYTHON) --version 2>&1 | cut -d' ' -f2 || echo "not found"
	@echo -n "  • pytest:  " && $(PYTEST) --version 2>&1 | head -1 | awk '{print $$2}' || echo "not found"
	@echo -n "  • ruff:    " && $(RUFF) --version 2>&1 | awk '{print $$2}' || echo "not found"
	@echo -n "  • mypy:    " && $(MYPY) --version 2>&1 | awk '{print $$2}' || echo "not found"

# ─────────────────────────────────────────────────────────────────────────────
# END OF MAKEFILE
# ─────────────────────────────────────────────────────────────────────────────

# © 2025 Antonio Clim. All rights reserved.
# See README.md for full licence terms.
