# =============================================================================
# TAOCT4researchers — Makefile
# The Art of Computational Thinking for Researchers
# Version 5.0.0
# =============================================================================
#
# Usage:
#   make help          Show available targets
#   make install       Install all dependencies
#   make test          Run complete test suite
#   make lint          Run code quality checks
#   make clean         Remove generated files
#
# =============================================================================

.PHONY: help install install-dev test test-unit lint format check validate clean docs

# Default target
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PIP := pip

# Directories
VENV := .venv
DOCS_DIR := docs
SCRIPTS_DIR := scripts

# Unit directories (in order)
UNIT_DIRS := \
	"01UNIT - The Epistemology of Computation (foundations)" \
	"02UNIT - Abstraction and Encapsulation (patterns)" \
	"03UNIT - Algorithmic Complexity (performance)" \
	"04UNIT - Advanced Data Structures (design)" \
	"05UNIT - Scientific Computing (simulations)" \
	"06UNIT - Visualisation for Research (communication)" \
	"07UNIT - Reproducibility and Capstone (integration)" \
	"08UNIT - Recursion and Dynamic Programming (algorithms)" \
	"09UNIT - Exception Handling and Defensive Code (robustness)" \
	"10UNIT - Data Persistence and Serialisation (storage)" \
	"11UNIT - Text Processing and NLP Fundamentals (text analysis)" \
	"12UNIT - Web APIs and Data Acquisition (web integration)" \
	"13UNIT - Machine Learning for Researchers (ML basics)" \
	"14UNIT - Parallel Computing and Scalability (performance+)"

# Colours for terminal output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m  # No colour

# =============================================================================
# Help
# =============================================================================

help:  ## Display this help message
	@echo ""
	@echo "$(BLUE)TAOCT4researchers — Build Automation$(NC)"
	@echo "$(BLUE)=====================================$(NC)"
	@echo ""
	@echo "$(GREEN)Installation:$(NC)"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo "  make install-unit   Install dependencies for specific unit (UNIT=01)"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test           Run complete test suite"
	@echo "  make test-unit      Run tests for specific unit (UNIT=01)"
	@echo "  make test-fast      Run tests excluding slow markers"
	@echo "  make coverage       Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint           Run all linters"
	@echo "  make format         Auto-format code"
	@echo "  make typecheck      Run type checking"
	@echo "  make check          Run all quality checks"
	@echo ""
	@echo "$(GREEN)Validation:$(NC)"
	@echo "  make validate       Validate all unit structures"
	@echo "  make validate-unit  Validate specific unit (UNIT=01)"
	@echo ""
	@echo "$(GREEN)Maintenance:$(NC)"
	@echo "  make clean          Remove generated files"
	@echo "  make clean-cache    Remove Python cache files"
	@echo "  make clean-all      Remove all generated content"
	@echo ""
	@echo "$(GREEN)Documentation:$(NC)"
	@echo "  make docs           Generate documentation"
	@echo "  make badges         Generate unit badges"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make test-unit UNIT=08"
	@echo "  make install-unit UNIT=13"
	@echo ""

# =============================================================================
# Installation
# =============================================================================

install:  ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Installation complete.$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)Development installation complete.$(NC)"

install-unit:  ## Install dependencies for specific unit (UNIT=01)
ifndef UNIT
	$(error UNIT is not set. Usage: make install-unit UNIT=01)
endif
	@echo "$(BLUE)Installing dependencies for Unit $(UNIT)...$(NC)"
	@UNIT_DIR=$$(ls -d *$(UNIT)UNIT* 2>/dev/null | head -1); \
	if [ -z "$$UNIT_DIR" ]; then \
		echo "$(RED)Error: Unit $(UNIT) not found$(NC)"; \
		exit 1; \
	fi; \
	if [ -f "$$UNIT_DIR/requirements.txt" ]; then \
		$(PIP) install -r "$$UNIT_DIR/requirements.txt"; \
	fi
	@echo "$(GREEN)Unit $(UNIT) dependencies installed.$(NC)"

# =============================================================================
# Testing
# =============================================================================

test:  ## Run complete test suite
	@echo "$(BLUE)Running complete test suite...$(NC)"
	$(PYTHON) -m pytest -v --tb=short
	@echo "$(GREEN)All tests completed.$(NC)"

test-unit:  ## Run tests for specific unit (UNIT=01)
ifndef UNIT
	$(error UNIT is not set. Usage: make test-unit UNIT=01)
endif
	@echo "$(BLUE)Running tests for Unit $(UNIT)...$(NC)"
	@UNIT_DIR=$$(ls -d *$(UNIT)UNIT* 2>/dev/null | head -1); \
	if [ -z "$$UNIT_DIR" ]; then \
		echo "$(RED)Error: Unit $(UNIT) not found$(NC)"; \
		exit 1; \
	fi; \
	$(PYTHON) -m pytest "$$UNIT_DIR/tests/" -v --tb=short
	@echo "$(GREEN)Unit $(UNIT) tests completed.$(NC)"

test-fast:  ## Run tests excluding slow markers
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTHON) -m pytest -v -m "not slow" --tb=short
	@echo "$(GREEN)Fast tests completed.$(NC)"

coverage:  ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint:  ## Run all linters
	@echo "$(BLUE)Running linters...$(NC)"
	$(PYTHON) -m ruff check .
	@echo "$(GREEN)Linting complete.$(NC)"

format:  ## Auto-format code
	@echo "$(BLUE)Formatting code...$(NC)"
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
	@echo "$(GREEN)Formatting complete.$(NC)"

typecheck:  ## Run type checking
	@echo "$(BLUE)Running type checker...$(NC)"
	$(PYTHON) -m mypy --ignore-missing-imports .
	@echo "$(GREEN)Type checking complete.$(NC)"

check: lint typecheck  ## Run all quality checks
	@echo "$(GREEN)All quality checks passed.$(NC)"

# =============================================================================
# Validation
# =============================================================================

validate:  ## Validate all unit structures
	@echo "$(BLUE)Validating all units...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/validate_all_units.py
	@echo "$(GREEN)Validation complete.$(NC)"

validate-unit:  ## Validate specific unit (UNIT=01)
ifndef UNIT
	$(error UNIT is not set. Usage: make validate-unit UNIT=01)
endif
	@echo "$(BLUE)Validating Unit $(UNIT)...$(NC)"
	@UNIT_DIR=$$(ls -d *$(UNIT)UNIT* 2>/dev/null | head -1); \
	if [ -z "$$UNIT_DIR" ]; then \
		echo "$(RED)Error: Unit $(UNIT) not found$(NC)"; \
		exit 1; \
	fi; \
	$(PYTHON) "$$UNIT_DIR/scripts/validate_unit.py" $(UNIT)
	@echo "$(GREEN)Unit $(UNIT) validation complete.$(NC)"

# =============================================================================
# Documentation
# =============================================================================

docs:  ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "Documentation generation not yet implemented"
	@echo "$(YELLOW)See docs/ directory for manual documentation$(NC)"

badges:  ## Generate unit badges
	@echo "$(BLUE)Generating badges...$(NC)"
	$(PYTHON) $(SCRIPTS_DIR)/generate_badges.py
	@echo "$(GREEN)Badges generated in assets/badges/$(NC)"

# =============================================================================
# Cleaning
# =============================================================================

clean-cache:  ## Remove Python cache files
	@echo "$(BLUE)Cleaning cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Cache cleaned.$(NC)"

clean: clean-cache  ## Remove generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	rm -rf htmlcov/ 2>/dev/null || true
	rm -rf .coverage 2>/dev/null || true
	rm -rf *.egg-info 2>/dev/null || true
	rm -rf dist/ build/ 2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(NC)"

clean-all: clean  ## Remove all generated content including venv
	@echo "$(BLUE)Cleaning all generated content...$(NC)"
	rm -rf $(VENV) 2>/dev/null || true
	@echo "$(GREEN)Full clean complete.$(NC)"

# =============================================================================
# Environment Setup
# =============================================================================

venv:  ## Create virtual environment
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	$(PYTHON) -m venv $(VENV)
	@echo "$(GREEN)Virtual environment created at $(VENV)/$(NC)"
	@echo "Activate with: source $(VENV)/bin/activate"

setup: venv  ## Complete environment setup
	@echo "$(BLUE)Setting up development environment...$(NC)"
	. $(VENV)/bin/activate && $(MAKE) install-dev
	@echo "$(GREEN)Setup complete.$(NC)"

# =============================================================================
# Shortcuts
# =============================================================================

t: test  ## Alias for test
l: lint  ## Alias for lint
f: format  ## Alias for format
c: check  ## Alias for check
v: validate  ## Alias for validate

# =============================================================================
# Information
# =============================================================================

info:  ## Display project information
	@echo ""
	@echo "$(BLUE)TAOCT4researchers$(NC)"
	@echo "The Art of Computational Thinking for Researchers"
	@echo "Version: 5.0.0"
	@echo ""
	@echo "$(GREEN)Units:$(NC) 14"
	@echo "$(GREEN)Python:$(NC) 3.10+"
	@echo "$(GREEN)Repository:$(NC) https://github.com/antonioclim/TAOCT4researchers"
	@echo ""
	@echo "$(GREEN)Available units:$(NC)"
	@for dir in $(UNIT_DIRS); do echo "  - $$dir"; done
	@echo ""
