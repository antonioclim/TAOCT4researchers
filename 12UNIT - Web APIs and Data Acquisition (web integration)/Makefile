# ═══════════════════════════════════════════════════════════════════════════
# 12UNIT: Web APIs and Data Acquisition
# Makefile for 12UNIT
# ═══════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

LAB_DIR := lab
TEST_DIR := tests
SCRIPTS_DIR := scripts

.PHONY: all test lint typecheck validate clean help
.PHONY: run-api run-scraper run-flask run-pagination run-auth test-api

all: lint typecheck test
	@echo "✓ All checks passed for 12UNIT"

test:
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing

test-api:
	$(PYTEST) $(TEST_DIR) -v -k api --cov=$(LAB_DIR) --cov-report=term-missing

lint:
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) --fix

typecheck:
	$(MYPY) $(LAB_DIR) --ignore-missing-imports

validate:
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 12

run-api:
	$(PYTHON) $(LAB_DIR)/lab_12_01_api_consumption.py --demo

run-pagination:
	$(PYTHON) $(LAB_DIR)/lab_12_01_api_consumption.py --demo pagination

run-auth:
	$(PYTHON) $(LAB_DIR)/lab_12_01_api_consumption.py --demo auth

run-scraper:
	$(PYTHON) $(LAB_DIR)/lab_12_02_web_scraping_flask.py --demo scrape

run-flask:
	$(PYTHON) $(LAB_DIR)/lab_12_02_web_scraping_flask.py --serve --host 127.0.0.1 --port 5000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleaned 12UNIT"

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " 12UNIT: Web APIs and Data Acquisition"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test       - Run tests with coverage"
	@echo "  make test-api   - Run API-related tests"
	@echo "  make lint       - Run ruff linter"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make validate   - Run UNIT validation script"
	@echo "  make all        - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific:"
	@echo "  make run-api        - Run API client demos"
	@echo "  make run-pagination - Run pagination demo"
	@echo "  make run-auth       - Run authentication demo"
	@echo "  make run-scraper    - Run scraping demo"
	@echo "  make run-flask      - Start Flask demo server"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove generated files"
	@echo "  make help       - Show this help message"
	@echo ""
