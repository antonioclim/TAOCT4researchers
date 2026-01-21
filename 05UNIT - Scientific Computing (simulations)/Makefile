# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# 05UNIT: Scientific Computing — Makefile
# ═══════════════════════════════════════════════════════════════════════════════
#
# © 2025 Antonio Clim. All rights reserved.
#
# Usage:
#   make help       Show available targets
#   make all        Run all checks and tests
#   make test       Run test suite
#   make lint       Check code quality
#
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all help run-labs test lint format check clean validate install

# Default Python interpreter
PYTHON := python3
PIP := pip3

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISE_DIR := exercises

# Colours for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No colour

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

all: check test
	@echo "$(GREEN)✓ All checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo "  05UNIT: Scientific Computing — Available Targets"
	@echo "═══════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  $(GREEN)Development:$(NC)"
	@echo "    make install      Install dependencies"
	@echo "    make run-labs     Run all lab demonstrations"
	@echo "    make run-mc       Run Monte Carlo lab demo"
	@echo "    make run-ode      Run ODE solvers lab demo"
	@echo "    make run-abm      Run ABM lab demo"
	@echo ""
	@echo "  $(GREEN)Quality:$(NC)"
	@echo "    make test         Run pytest test suite"
	@echo "    make coverage     Run tests with coverage report"
	@echo "    make lint         Run ruff linter"
	@echo "    make format       Format code with ruff"
	@echo "    make typecheck    Run mypy type checker"
	@echo "    make check        Run all quality checks"
	@echo ""
	@echo "  $(GREEN)Maintenance:$(NC)"
	@echo "    make clean        Remove generated files"
	@echo "    make validate     Validate week structure"
	@echo "    make all          Run check + test"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

install:
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	$(PIP) install numpy>=1.24 scipy>=1.11 matplotlib>=3.7 pytest>=7.0 pytest-cov>=4.0 ruff>=0.1 mypy>=1.0
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# RUN LABS
# ═══════════════════════════════════════════════════════════════════════════════

run-labs: run-mc run-ode run-abm
	@echo "$(GREEN)✓ All labs completed$(NC)"

run-mc:
	@echo "$(YELLOW)Running Monte Carlo lab...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_01_monte_carlo --demo
	@echo "$(GREEN)✓ Monte Carlo demo complete$(NC)"

run-ode:
	@echo "$(YELLOW)Running ODE solvers lab...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_02_ode_solvers --demo
	@echo "$(GREEN)✓ ODE solvers demo complete$(NC)"

run-abm:
	@echo "$(YELLOW)Running ABM lab...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_03_agent_based_modelling --demo
	@echo "$(GREEN)✓ ABM demo complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

test:
	@echo "$(YELLOW)Running test suite...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v --tb=short
	@echo "$(GREEN)✓ Tests passed$(NC)"

coverage:
	@echo "$(YELLOW)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

test-fast:
	@echo "$(YELLOW)Running fast tests...$(NC)"
	$(PYTHON) -m pytest $(TEST_DIR) -v -x --tb=short -q
	@echo "$(GREEN)✓ Fast tests passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

lint:
	@echo "$(YELLOW)Running linter...$(NC)"
	$(PYTHON) -m ruff check $(LAB_DIR) $(TEST_DIR) $(EXERCISE_DIR)
	@echo "$(GREEN)✓ Lint passed$(NC)"

format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(PYTHON) -m ruff format $(LAB_DIR) $(TEST_DIR) $(EXERCISE_DIR)
	$(PYTHON) -m ruff check --fix $(LAB_DIR) $(TEST_DIR) $(EXERCISE_DIR)
	@echo "$(GREEN)✓ Code formatted$(NC)"

typecheck:
	@echo "$(YELLOW)Running type checker...$(NC)"
	$(PYTHON) -m mypy $(LAB_DIR) --exclude 'lab/solutions' --ignore-missing-imports
	@echo "$(GREEN)✓ Type check passed$(NC)"

check: lint typecheck
	@echo "$(GREEN)✓ All quality checks passed$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	@echo "$(YELLOW)Cleaning generated files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

validate:
	@echo "$(YELLOW)Validating week structure...$(NC)"
	@test -f README.md || (echo "$(RED)✗ Missing README.md$(NC)" && exit 1)
	@test -f theory/05UNIT_slides.html || (echo "$(RED)✗ Missing theory/05UNIT_slides.html$(NC)" && exit 1)
	@test -f theory/lecture_notes.md || (echo "$(RED)✗ Missing theory/lecture_notes.md$(NC)" && exit 1)
	@test -f theory/learning_objectives.md || (echo "$(RED)✗ Missing theory/learning_objectives.md$(NC)" && exit 1)
	@test -f $(LAB_DIR)/__init__.py || (echo "$(RED)✗ Missing $(LAB_DIR)/__init__.py$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_05_01_monte_carlo.py || (echo "$(RED)✗ Missing Monte Carlo lab$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_05_02_ode_solvers.py || (echo "$(RED)✗ Missing ODE solvers lab$(NC)" && exit 1)
	@test -f $(LAB_DIR)/lab_05_03_agent_based_modelling.py || (echo "$(RED)✗ Missing ABM lab$(NC)" && exit 1)
	@test -f exercises/homework.md || (echo "$(RED)✗ Missing exercises/homework.md$(NC)" && exit 1)
	@test -f assessments/quiz.md || (echo "$(RED)✗ Missing assessments/quiz.md$(NC)" && exit 1)
	@test -f assessments/rubric.md || (echo "$(RED)✗ Missing assessments/rubric.md$(NC)" && exit 1)
	@test -f assessments/self_check.md || (echo "$(RED)✗ Missing assessments/self_check.md$(NC)" && exit 1)
	@test -f resources/cheatsheet.md || (echo "$(RED)✗ Missing resources/cheatsheet.md$(NC)" && exit 1)
	@test -f resources/further_reading.md || (echo "$(RED)✗ Missing resources/further_reading.md$(NC)" && exit 1)
	@test -f resources/glossary.md || (echo "$(RED)✗ Missing resources/glossary.md$(NC)" && exit 1)
	@test -d $(TEST_DIR) || (echo "$(RED)✗ Missing tests directory$(NC)" && exit 1)
	@test -f $(TEST_DIR)/conftest.py || (echo "$(RED)✗ Missing tests/conftest.py$(NC)" && exit 1)
	@grep -q "RESTRICTIVE LICENCE" README.md || (echo "$(RED)✗ Missing licence in README.md$(NC)" && exit 1)
	@echo "$(GREEN)✓ 05UNIT structure validated$(NC)"

# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION SHORTCUTS
# ═══════════════════════════════════════════════════════════════════════════════

demo-pi:
	@echo "$(YELLOW)Estimating π using Monte Carlo...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_01_monte_carlo --estimate-pi --seed 42

demo-convergence:
	@echo "$(YELLOW)Running ODE convergence analysis...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_02_ode_solvers --convergence

demo-schelling:
	@echo "$(YELLOW)Running Schelling segregation model...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_03_agent_based_modelling --schelling --grid-size 30

demo-boids:
	@echo "$(YELLOW)Running Boids flocking simulation...$(NC)"
	$(PYTHON) -m $(LAB_DIR).lab_05_03_agent_based_modelling --boids --n-boids 50
