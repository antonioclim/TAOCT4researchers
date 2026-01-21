# ═══════════════════════════════════════════════════════════════════════════
# 04UNIT: Advanced Data Structures
# Makefile for 04UNIT
# ═══════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

LAB_DIR := lab
TEST_DIR := tests
SCRIPTS_DIR := scripts

.PHONY: all test lint typecheck validate clean help
.PHONY: run-graph run-bloom run-dijkstra run-bfs-dfs

all: lint typecheck test
	@echo "✓ All checks passed for 04UNIT"

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTEST) $(TEST_DIR) -v 

lint:
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) --fix

typecheck:
	$(MYPY) $(LAB_DIR) --ignore-missing-imports

validate:
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 04 --strict

run-graph:
	$(PYTHON) $(LAB_DIR)/lab_04_01_graph_library.py --demo

run-bloom:
	$(PYTHON) $(LAB_DIR)/lab_04_02_probabilistic_ds.py --demo --section 2

run-dijkstra:
	$(PYTHON) $(LAB_DIR)/lab_04_01_graph_library.py --demo --section 3

run-bfs-dfs:
	$(PYTHON) $(LAB_DIR)/lab_04_01_graph_library.py --demo --section 2

clean:
	find . -type d -name "__pycache__" -exec rm -rf {{}} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {{}} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {{}} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {{}} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Cleaned 04UNIT"

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " 04UNIT: Advanced Data Structures"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test       - Run tests with coverage"
	@echo "  make lint       - Run ruff linter"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make validate   - Run UNIT validator (strict)"
	@echo "  make all        - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific:"
	@echo "  make run-graph   - Graph library demo"
	@echo "  make run-bloom   - Probabilistic DS demo"
	@echo "  make run-dijkstra- Dijkstra demo"
	@echo "  make run-bfs-dfs - Traversal comparison"
	@echo ""
