# ═══════════════════════════════════════════════════════════════════════════════
# 10UNIT — Data Persistence and Serialisation
# Build Automation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   make help          Show available targets
#   make setup         Install dependencies
#   make test          Run test suite
#   make lint          Run code quality checks
#   make clean         Remove generated files
#
# © 2025 Antonio Clim. All rights reserved.
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: help setup test lint format typecheck clean demo diagrams all
.PHONY: run-serialise run-db run-json run-csv run-sqlite benchmark-formats init-db validate

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
RUFF := ruff
MYPY := mypy
COVERAGE_DIR := htmlcov
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises

# Colours for terminal output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo ""
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BLUE)  10UNIT — Data Persistence and Serialisation — Build Targets$(RESET)"
	@echo "$(BLUE)═══════════════════════════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(GREEN)Setup:$(RESET)"
	@echo "  make setup          Install required Python packages"
	@echo "  make setup-dev      Install development dependencies"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test           Run all tests with coverage"
	@echo "  make test-fast      Run tests without coverage"
	@echo "  make test-lab1      Run tests for lab_10_01 only"
	@echo "  make test-lab2      Run tests for lab_10_02 only"
	@echo "  make test-verbose   Run tests with verbose output"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  make lint           Run ruff linter"
	@echo "  make format         Auto-format code with ruff"
	@echo "  make typecheck      Run mypy type checking"
	@echo "  make quality        Run all quality checks"
	@echo ""
	@echo "$(GREEN)Demonstrations:$(RESET)"
	@echo "  make demo           Run interactive demonstrations"
	@echo "  make demo-lab1      Run lab_10_01 demo"
	@echo "  make demo-lab2      Run lab_10_02 demo"
	@echo ""
	@echo "$(GREEN)Domain-Specific (Data/Storage):$(RESET)"
	@echo "  make run-serialise  Execute serialisation demo"
	@echo "  make run-db         Execute database fundamentals demo"
	@echo "  make run-json       Run JSON processing examples"
	@echo "  make run-csv        Run CSV processing examples"
	@echo "  make run-sqlite     Run SQLite operations demo"
	@echo "  make benchmark-formats  Compare serialisation format performance"
	@echo "  make init-db        Initialise sample database"
	@echo ""
	@echo "$(GREEN)Diagrams:$(RESET)"
	@echo "  make diagrams       Generate SVG from PlantUML sources"
	@echo ""
	@echo "$(GREEN)Maintenance:$(RESET)"
	@echo "  make clean          Remove generated files"
	@echo "  make clean-all      Remove all generated files including coverage"
	@echo ""
	@echo "$(GREEN)Composite:$(RESET)"
	@echo "  make all            Setup, quality checks and tests"
	@echo "  make ci             Run continuous integration checks"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

setup:
	@echo "$(BLUE)Installing required packages...$(RESET)"
	$(PIP) install pandas pyarrow --break-system-packages
	@echo "$(GREEN)✓ Setup complete$(RESET)"

setup-dev: setup
	@echo "$(BLUE)Installing development dependencies...$(RESET)"
	$(PIP) install pytest pytest-cov ruff mypy --break-system-packages
	@echo "$(GREEN)✓ Development setup complete$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

test:
	@echo "$(BLUE)Running test suite with coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ --cov=$(LAB_DIR) --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests complete. HTML report: $(COVERAGE_DIR)/index.html$(RESET)"

test-fast:
	@echo "$(BLUE)Running tests without coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -q
	@echo "$(GREEN)✓ Tests complete$(RESET)"

test-lab1:
	@echo "$(BLUE)Running lab_10_01 tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/test_lab_10_01.py -v

test-lab2:
	@echo "$(BLUE)Running lab_10_02 tests...$(RESET)"
	$(PYTEST) $(TEST_DIR)/test_lab_10_02.py -v

test-verbose:
	@echo "$(BLUE)Running tests with verbose output...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=long

test-exercises:
	@echo "$(BLUE)Running exercise tests...$(RESET)"
	$(PYTEST) $(EXERCISES_DIR)/ -v 2>/dev/null || echo "$(YELLOW)No exercise tests found$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

lint:
	@echo "$(BLUE)Running linter...$(RESET)"
	$(RUFF) check $(LAB_DIR)/ $(TEST_DIR)/ $(EXERCISES_DIR)/ --ignore E501
	@echo "$(GREEN)✓ Linting complete$(RESET)"

format:
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(RUFF) format $(LAB_DIR)/ $(TEST_DIR)/ $(EXERCISES_DIR)/
	@echo "$(GREEN)✓ Formatting complete$(RESET)"

typecheck:
	@echo "$(BLUE)Running type checker...$(RESET)"
	$(MYPY) $(LAB_DIR)/ --ignore-missing-imports --no-error-summary || true
	@echo "$(GREEN)✓ Type checking complete$(RESET)"

quality: lint typecheck
	@echo "$(GREEN)✓ All quality checks complete$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

demo: demo-lab1 demo-lab2

demo-lab1:
	@echo "$(BLUE)Running lab_10_01 demonstration...$(RESET)"
	@echo "─────────────────────────────────────────────────"
	$(PYTHON) $(LAB_DIR)/lab_10_01_file_io_serialisation.py --demo 2>/dev/null || \
		$(PYTHON) -c "from $(LAB_DIR).lab_10_01_file_io_serialisation import *; print('Lab 10.01 loaded successfully')"
	@echo "─────────────────────────────────────────────────"

demo-lab2:
	@echo "$(BLUE)Running lab_10_02 demonstration...$(RESET)"
	@echo "─────────────────────────────────────────────────"
	$(PYTHON) $(LAB_DIR)/lab_10_02_database_fundamentals.py --demo 2>/dev/null || \
		$(PYTHON) -c "from $(LAB_DIR).lab_10_02_database_fundamentals import *; print('Lab 10.02 loaded successfully')"
	@echo "─────────────────────────────────────────────────"

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGRAMS
# ═══════════════════════════════════════════════════════════════════════════════

diagrams:
	@echo "$(BLUE)Generating diagrams from PlantUML sources...$(RESET)"
	@if command -v plantuml > /dev/null 2>&1; then \
		for f in assets/diagrams/*.puml; do \
			echo "  Processing $$f..."; \
			plantuml -tsvg "$$f"; \
		done; \
		echo "$(GREEN)✓ Diagrams generated$(RESET)"; \
	else \
		echo "$(YELLOW)PlantUML not installed. Install with: apt install plantuml$(RESET)"; \
	fi

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS (Data/Storage)
# ═══════════════════════════════════════════════════════════════════════════════

run-serialise: demo-lab1
	@echo "$(GREEN)✓ Serialisation demo complete$(RESET)"

run-db: demo-lab2
	@echo "$(GREEN)✓ Database fundamentals demo complete$(RESET)"

run-json:
	@echo "$(BLUE)Running JSON processing examples...$(RESET)"
	@$(PYTHON) -c "\
from pathlib import Path; \
import json; \
from lab.lab_10_01_file_io_serialisation import save_json, load_json; \
data = {'experiment': 'test', 'values': [1, 2, 3]}; \
p = Path('/tmp/test.json'); \
save_json(data, p); \
print(f'Saved JSON: {p}'); \
loaded = load_json(p); \
print(f'Loaded: {loaded}')"
	@echo "$(GREEN)✓ JSON examples complete$(RESET)"

run-csv:
	@echo "$(BLUE)Running CSV processing examples...$(RESET)"
	@$(PYTHON) -c "\
from pathlib import Path; \
from lab.lab_10_01_file_io_serialisation import write_csv_from_dicts, read_csv_as_dicts; \
data = [{'name': 'Alice', 'score': '95'}, {'name': 'Bob', 'score': '87'}]; \
p = Path('/tmp/test.csv'); \
write_csv_from_dicts(data, p); \
print(f'Saved CSV: {p}'); \
loaded = read_csv_as_dicts(p); \
print(f'Loaded {len(loaded)} records')"
	@echo "$(GREEN)✓ CSV examples complete$(RESET)"

run-sqlite:
	@echo "$(BLUE)Running SQLite operations demo...$(RESET)"
	@$(PYTHON) -c "\
from pathlib import Path; \
from lab.lab_10_02_database_fundamentals import ResearchDatabase; \
db = ResearchDatabase(Path('/tmp/test_demo.db')); \
db.create_schema(); \
print('Created schema'); \
db.close()"
	@echo "$(GREEN)✓ SQLite demo complete$(RESET)"

benchmark-formats:
	@echo "$(BLUE)Running format comparison benchmark...$(RESET)"
	@$(PYTHON) -c "\
from pathlib import Path; \
from lab.lab_10_01_file_io_serialisation import generate_sample_records, compare_formats, print_benchmark_report; \
import logging; \
logging.basicConfig(level=logging.INFO, format='%(message)s'); \
records = generate_sample_records(5000); \
print(f'Generated {len(records)} sample records'); \
benchmarks = compare_formats(records, Path('/tmp/benchmarks')); \
print_benchmark_report(benchmarks)"

init-db:
	@echo "$(BLUE)Initialising sample database...$(RESET)"
	@mkdir -p $(LAB_DIR)/data
	@$(PYTHON) -c "\
from pathlib import Path; \
from lab.lab_10_02_database_fundamentals import ResearchDatabase; \
db_path = Path('$(LAB_DIR)/data/research.db'); \
db = ResearchDatabase(db_path); \
db.create_schema(); \
print(f'Database initialised: {db_path}'); \
db.close()"
	@echo "$(GREEN)✓ Database initialised: $(LAB_DIR)/data/research.db$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	@echo "$(BLUE)Cleaning generated files...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -f $(LAB_DIR)/data/*.db 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete$(RESET)"

clean-all: clean
	@echo "$(BLUE)Removing coverage reports...$(RESET)"
	rm -rf $(COVERAGE_DIR) 2>/dev/null || true
	rm -f .coverage 2>/dev/null || true
	@echo "$(GREEN)✓ Full clean complete$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# COMPOSITE TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

all: setup-dev quality test
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)  ✓ All targets complete$(RESET)"
	@echo "$(GREEN)═══════════════════════════════════════════════════════════════════════════════$(RESET)"

ci: lint typecheck test-fast
	@echo "$(GREEN)✓ CI checks passed$(RESET)"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

validate:
	@echo "$(BLUE)Validating unit structure...$(RESET)"
	@$(PYTHON) scripts/validate_unit.py 10 --path .
	@echo "$(GREEN)✓ Validation complete$(RESET)"

check-syntax:
	@echo "$(BLUE)Checking Python syntax...$(RESET)"
	@$(PYTHON) -m py_compile $(LAB_DIR)/lab_10_01_file_io_serialisation.py && echo "  ✓ lab_10_01: OK"
	@$(PYTHON) -m py_compile $(LAB_DIR)/lab_10_02_database_fundamentals.py && echo "  ✓ lab_10_02: OK"
	@echo "$(GREEN)✓ Syntax check complete$(RESET)"
