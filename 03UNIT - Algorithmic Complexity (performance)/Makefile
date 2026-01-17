# ==============================================================================
# 03UNIT — Algorithmic Complexity
# Makefile for development, verification and routine execution
# ==============================================================================

SHELL := /bin/bash
.ONESHELL:

PYTHON ?= python3
UNIT ?= 03

# The lab modules are importable when executed from the UNIT root directory.
LAB_01 := lab.lab_03_01_benchmark_suite
LAB_02 := lab.lab_03_02_complexity_analyser

RUFF ?= ruff
MYPY ?= mypy
PYTEST ?= pytest

.PHONY: all help run-labs test lint format type check clean validate

all: check

help:
	@printf "\n03UNIT — Algorithmic Complexity\n\n"
	@printf "Targets:\n"
	@printf "  make run-labs   Run both labs in demo mode\n"
	@printf "  make test       Run pytest with coverage\n"
	@printf "  make lint       Run ruff static checks\n"
	@printf "  make format     Format with ruff\n"
	@printf "  make type       Run mypy (strict)\n"
	@printf "  make validate   Validate UNIT structure\n"
	@printf "  make check      Run lint, type, test and validate\n"
	@printf "  make clean      Remove caches and build artefacts\n\n"

run-labs:
	$(PYTHON) -m $(LAB_01) --demo
	$(PYTHON) -m $(LAB_02) --demo

test:
	$(PYTEST) -q --cov=lab --cov-report=term-missing

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

type:
	$(MYPY) --strict lab tests

validate:
	@if [ -f ../scripts/validate_unit.py ]; then \
		$(PYTHON) ../scripts/validate_unit.py $(UNIT); \
	elif [ -f ../$(UNIT)UNIT/scripts/validate_unit.py ]; then \
		(cd .. && $(PYTHON) $(UNIT)UNIT/scripts/validate_unit.py $(UNIT)); \
	else \
		printf "ERROR: validate_unit.py not found.\n"; \
		printf "Searched: ../scripts/validate_unit.py and ../$(UNIT)UNIT/scripts/validate_unit.py\n\n"; \
		exit 2; \
	fi


check: lint type test validate

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov .coverage
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
