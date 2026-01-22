# ═══════════════════════════════════════════════════════════════════════════
# 11UNIT: Text Processing and NLP Fundamentals
# Makefile for 11UNIT
# ═══════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

LAB_DIR := lab
TEST_DIR := tests
SCRIPTS_DIR := scripts

.PHONY: all test lint typecheck validate clean help
.PHONY: run-regex run-nlp analyse-corpus

all: lint typecheck test
	@echo "✓ All checks passed for 11UNIT"

test:
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing

lint:
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) $(SCRIPTS_DIR) --fix

typecheck:
	$(MYPY) $(LAB_DIR) --ignore-missing-imports

validate:
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 11

run-regex:
	$(PYTHON) -m lab.lab_11_01_regex_string_ops --demo

run-nlp:
	$(PYTHON) -m lab.lab_11_02_nlp_fundamentals --demo

analyse-corpus:
	$(PYTHON) -m lab.lab_11_02_nlp_fundamentals --analyse resources/datasets/sample_corpus.csv --out output

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleaned 11UNIT"

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " 11UNIT: Text Processing and NLP Fundamentals"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test           - Run tests with coverage"
	@echo "  make lint           - Run ruff linter"
	@echo "  make typecheck      - Run mypy type checker"
	@echo "  make validate       - Run UNIT validation script"
	@echo "  make all            - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific:"
	@echo "  make run-regex      - Run regex and parsing demo"
	@echo "  make run-nlp        - Run tokenisation and TF–IDF demo"
	@echo "  make analyse-corpus - Run corpus analysis pipeline"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean          - Remove generated files"
	@echo "  make help           - Show this help message"
