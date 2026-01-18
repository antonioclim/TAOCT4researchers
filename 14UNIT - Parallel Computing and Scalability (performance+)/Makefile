# ═══════════════════════════════════════════════════════════════════════════════
# MAKEFILE: 14UNIT — Parallel Computing and Scalability
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   make install      - Install dependencies
#   make test         - Run test suite
#   make lint         - Run linting checks
#   make format       - Format code with ruff
#   make typecheck    - Run mypy type checking
#   make validate     - Validate UNIT structure
#   make benchmark    - Run performance benchmarks
#   make profile      - Profile lab code
#   make clean        - Remove generated files
#   make docs         - Generate diagrams from PlantUML
#   make all          - Run all checks
#
# © 2025 Antonio Clim. All rights reserved.
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all install test lint format typecheck validate benchmark profile clean docs help
.PHONY: run-mp run-dask run-profile benchmark-parallel benchmark-scaling visualise-speedup

# Python interpreter
PYTHON := python3
PIP := $(PYTHON) -m pip

# Directories
LAB_DIR := lab
TEST_DIR := tests
SCRIPTS_DIR := scripts
EXERCISES_DIR := exercises
ASSETS_DIR := assets

# Unit number
UNIT := 14

# Default target
.DEFAULT_GOAL := help

# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────

help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "  14UNIT: Parallel Computing and Scalability — Makefile"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Setup:"
	@echo "    make install         Install dependencies"
	@echo ""
	@echo "  Quality Assurance:"
	@echo "    make test            Run test suite with coverage"
	@echo "    make lint            Run ruff linting"
	@echo "    make format          Format code with ruff"
	@echo "    make typecheck       Run mypy type checking"
	@echo "    make validate        Validate UNIT structure"
	@echo "    make all             Run lint, typecheck, test, validate"
	@echo ""
	@echo "  Domain-Specific (Parallel Computing):"
	@echo "    make run-mp          Execute multiprocessing demo"
	@echo "    make run-dask        Execute Dask demo"
	@echo "    make run-profile     Profile sample workload"
	@echo "    make benchmark-parallel  Parallel vs sequential benchmark"
	@echo "    make benchmark-scaling   Scaling benchmark (1-N cores)"
	@echo "    make visualise-speedup   Generate speedup plots"
	@echo ""
	@echo "  Performance:"
	@echo "    make benchmark       Run comprehensive benchmarks"
	@echo "    make profile         Profile lab code"
	@echo "    make profile-memory  Memory profiling"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make docs            Generate diagrams from PlantUML"
	@echo "    make clean           Remove generated files"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────────────────────────────────────
# INSTALLATION
# ─────────────────────────────────────────────────────────────────────────────

install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install numpy>=1.24 pandas>=2.0 dask>=2024.1 matplotlib>=3.7
	$(PIP) install pytest>=7.0 pytest-cov>=4.0 pytest-timeout>=2.0
	$(PIP) install ruff>=0.1 mypy>=1.0
	$(PIP) install line-profiler>=4.0 memory-profiler>=0.60
	@echo "Installation complete."

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install ipython jupyter notebook
	@echo "Development installation complete."

# ─────────────────────────────────────────────────────────────────────────────
# TESTING
# ─────────────────────────────────────────────────────────────────────────────

test:
	@echo "Running test suite..."
	$(PYTHON) -m pytest $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing --cov-report=html:coverage_html
	@echo "Test suite complete. Coverage report: coverage_html/index.html"

test-quick:
	@echo "Running quick test suite..."
	$(PYTHON) -m pytest $(TEST_DIR) -v --timeout=30
	@echo "Quick test complete."

test-lab01:
	@echo "Testing Lab 01: Multiprocessing..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_lab_14_01.py -v

test-lab02:
	@echo "Testing Lab 02: Dask and Profiling..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_lab_14_02.py -v

# ─────────────────────────────────────────────────────────────────────────────
# CODE QUALITY
# ─────────────────────────────────────────────────────────────────────────────

lint:
	@echo "Running linting checks..."
	$(PYTHON) -m ruff check $(LAB_DIR) $(TEST_DIR) $(SCRIPTS_DIR) $(EXERCISES_DIR)
	@echo "Linting complete."

format:
	@echo "Formatting code..."
	$(PYTHON) -m ruff format $(LAB_DIR) $(TEST_DIR) $(SCRIPTS_DIR) $(EXERCISES_DIR)
	$(PYTHON) -m ruff check --fix $(LAB_DIR) $(TEST_DIR) $(SCRIPTS_DIR) $(EXERCISES_DIR)
	@echo "Formatting complete."

typecheck:
	@echo "Running type checking..."
	$(PYTHON) -m mypy $(LAB_DIR) --ignore-missing-imports --no-error-summary
	@echo "Type checking complete."

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

validate:
	@echo "Validating UNIT structure..."
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py $(UNIT)
	@echo "Validation complete."

validate-install:
	@echo "Validating installation..."
	$(PYTHON) $(SCRIPTS_DIR)/validate_installation.py
	@echo "Installation validation complete."

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────

benchmark:
	@echo "Running benchmarks..."
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmarks.py
	@echo "Benchmarks complete."

profile:
	@echo "Profiling lab code..."
	$(PYTHON) -m cProfile -s cumulative $(LAB_DIR)/lab_14_01_multiprocessing.py --demo 2>/dev/null || true
	@echo "Profiling complete."

profile-memory:
	@echo "Memory profiling..."
	$(PYTHON) -m memory_profiler $(LAB_DIR)/lab_14_01_multiprocessing.py --demo 2>/dev/null || true
	@echo "Memory profiling complete."

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC TARGETS (Parallel Computing)
# ─────────────────────────────────────────────────────────────────────────────

run-mp:
	@echo "Executing multiprocessing demonstration..."
	$(PYTHON) $(LAB_DIR)/lab_14_01_multiprocessing.py --demo
	@echo "Multiprocessing demo complete."

run-dask:
	@echo "Executing Dask demonstration..."
	$(PYTHON) $(LAB_DIR)/lab_14_02_dask_profiling.py --demo
	@echo "Dask demo complete."

run-profile:
	@echo "Running profiling on sample workload..."
	$(PYTHON) -m cProfile -s cumulative -o profile.prof $(LAB_DIR)/lab_14_01_multiprocessing.py --demo --quiet
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.prof'); p.sort_stats('cumulative').print_stats(20)"
	@echo "Profiling complete. Results saved to profile.prof"

benchmark-parallel:
	@echo "Running parallel vs sequential benchmark..."
	$(PYTHON) -c "from lab.lab_14_01_multiprocessing import benchmark_scaling; benchmark_scaling(2_000_000, [1])"
	@echo "Sequential baseline established."
	$(PYTHON) -c "from lab.lab_14_01_multiprocessing import benchmark_scaling; benchmark_scaling(2_000_000, [1, 2, 4])"
	@echo "Parallel benchmark complete."

benchmark-scaling:
	@echo "Running scaling benchmark (1 to N cores)..."
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmarks.py
	@echo "Scaling benchmark complete."

visualise-speedup:
	@echo "Generating speedup visualisation..."
	$(PYTHON) -c "\
import matplotlib.pyplot as plt; \
import numpy as np; \
workers = [1, 2, 4, 8]; \
theoretical = workers; \
typical = [1, 1.8, 3.2, 5.5]; \
plt.figure(figsize=(10, 6)); \
plt.plot(workers, theoretical, 'b--', label='Ideal Linear'); \
plt.plot(workers, typical, 'ro-', label='Typical Observed'); \
plt.xlabel('Worker Count'); \
plt.ylabel('Speedup Factor'); \
plt.title('Parallel Scaling: Theoretical vs Observed'); \
plt.legend(); \
plt.grid(True, alpha=0.3); \
plt.savefig('assets/images/speedup_plot.png', dpi=150, bbox_inches='tight'); \
print('Saved: assets/images/speedup_plot.png')"
	@echo "Speedup visualisation complete."

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────

docs:
	@echo "Generating diagrams from PlantUML..."
	@if command -v plantuml >/dev/null 2>&1; then \
		for puml in $(ASSETS_DIR)/diagrams/*.puml; do \
			echo "Processing $$puml..."; \
			plantuml -tsvg "$$puml"; \
		done; \
		echo "Diagram generation complete."; \
	else \
		echo "PlantUML not installed. Install with: brew install plantuml (macOS) or apt install plantuml (Ubuntu)"; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────────────────────────────────────────

clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache */.pytest_cache
	rm -rf .mypy_cache */.mypy_cache
	rm -rf .ruff_cache */.ruff_cache
	rm -rf coverage_html .coverage
	rm -rf *.prof *.lprof
	rm -rf *.egg-info build dist
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete."

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE TARGETS
# ─────────────────────────────────────────────────────────────────────────────

all: lint typecheck test validate
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "  All checks passed for 14UNIT: Parallel Computing and Scalability"
	@echo "═══════════════════════════════════════════════════════════════════"

check: lint typecheck
	@echo "Code quality checks passed."

ci: install all
	@echo "CI pipeline complete."

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

verify-exercises:
	@echo "Verifying exercise syntax..."
	@for f in $(EXERCISES_DIR)/practice/*.py; do \
		echo "Checking $$f..."; \
		$(PYTHON) -m py_compile "$$f"; \
	done
	@echo "All exercises have valid syntax."

verify-solutions:
	@echo "Verifying solution syntax..."
	@for f in $(EXERCISES_DIR)/solutions/*.py; do \
		echo "Checking $$f..."; \
		$(PYTHON) -m py_compile "$$f"; \
	done
	@echo "All solutions have valid syntax."

# ═══════════════════════════════════════════════════════════════════════════════
# END OF MAKEFILE
# ═══════════════════════════════════════════════════════════════════════════════
