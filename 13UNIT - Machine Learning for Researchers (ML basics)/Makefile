# ═══════════════════════════════════════════════════════════════════════════
# 13UNIT: Machine Learning for Researchers
# Makefile for 13UNIT
# ═══════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

LAB_DIR := lab
TEST_DIR := tests
SCRIPTS_DIR := scripts

.PHONY: all test lint typecheck validate clean help
.PHONY: run-supervised run-unsupervised train evaluate tune plot-learning plot-confusion

all: lint typecheck test
	@echo "✓ All checks passed for 13UNIT"

test:
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing

lint:
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) $(SCRIPTS_DIR) --fix

typecheck:
	$(MYPY) $(LAB_DIR) --ignore-missing-imports

validate:
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 13

run-supervised:
	$(PYTHON) $(LAB_DIR)/lab_13_01_supervised_learning.py --demo

run-unsupervised:
	$(PYTHON) $(LAB_DIR)/lab_13_02_unsupervised_learning.py --demo

train:
	$(PYTHON) $(LAB_DIR)/lab_13_01_supervised_learning.py --train --outdir output

evaluate:
	$(PYTHON) $(LAB_DIR)/lab_13_01_supervised_learning.py --evaluate --outdir output

tune:
	$(PYTHON) $(LAB_DIR)/lab_13_01_supervised_learning.py --tune --outdir output

plot-learning:
	$(PYTHON) $(LAB_DIR)/lab_13_01_supervised_learning.py --plot-learning --outdir output

plot-confusion:
	$(PYTHON) $(LAB_DIR)/lab_13_01_supervised_learning.py --plot-confusion --outdir output

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf output 2>/dev/null || true
	@echo "✓ Cleaned 13UNIT"

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " 13UNIT: Machine Learning for Researchers"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test          - Run tests with coverage"
	@echo "  make lint          - Run ruff linter"
	@echo "  make typecheck     - Run mypy type checker"
	@echo "  make validate      - Run UNIT validation script"
	@echo "  make all           - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific:"
	@echo "  make run-supervised   - Supervised learning demo"
	@echo "  make run-unsupervised - Unsupervised learning demo"
	@echo "  make train            - Train baseline model"
	@echo "  make evaluate         - Evaluate baseline model"
	@echo "  make tune             - Hyperparameter search"
	@echo "  make plot-learning    - Generate learning curve plot"
	@echo "  make plot-confusion   - Generate confusion matrix plot"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean         - Remove generated files"
	@echo "  make help          - Show this help message"
	@echo ""
