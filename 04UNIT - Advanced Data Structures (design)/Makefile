# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTATIONAL THINKING FOR RESEARCHERS
# Week 4: Advanced Data Structures
# Makefile
# ═══════════════════════════════════════════════════════════════════════════════
#
# Usage:
#   make help     - Show this help message
#   make all      - Run all quality checks and tests
#   make run-labs - Run all lab demos
#   make test     - Run pytest suite
#   make lint     - Run linting with ruff
#   make format   - Format code with ruff
#   make check    - Run type checking with mypy
#   make clean    - Remove generated files
#   make validate - Validate week structure
#
# © 2025 Antonio Clim. All rights reserved.
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all help run-labs test lint format check clean validate coverage docs

# Default target
all: lint check test

# Help target
help:
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo "Week 4: Advanced Data Structures - Makefile Targets"
	@echo "═══════════════════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  all        - Run lint, check, and test (default)"
	@echo "  lint       - Run ruff linter"
	@echo "  format     - Format code with ruff"
	@echo "  check      - Run mypy type checker"
	@echo "  test       - Run pytest test suite"
	@echo "  coverage   - Run tests with coverage report"
	@echo ""
	@echo "Execution:"
	@echo "  run-labs   - Run all lab demos"
	@echo "  run-lab1   - Run Lab 4.1 (Graph Library) demo"
	@echo "  run-lab2   - Run Lab 4.2 (Probabilistic DS) demo"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean      - Remove generated files"
	@echo "  validate   - Validate week structure"
	@echo "  docs       - Generate documentation"
	@echo ""
	@echo "═══════════════════════════════════════════════════════════════════════════════"

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Run linting
lint:
	@echo "Running ruff linter..."
	@ruff check lab/ tests/ exercises/practice/ --ignore E501
	@echo "✓ Linting passed"

# Format code
format:
	@echo "Formatting code with ruff..."
	@ruff format lab/ tests/ exercises/practice/
	@ruff check --fix lab/ tests/ exercises/practice/ --ignore E501
	@echo "✓ Formatting complete"

# Type checking
check:
	@echo "Running mypy type checker..."
	@mypy lab/ --ignore-missing-imports --no-error-summary || true
	@echo "✓ Type checking complete"

# Run tests
test:
	@echo "Running pytest..."
	@pytest tests/ -v --tb=short
	@echo "✓ Tests complete"

# Run tests with coverage
coverage:
	@echo "Running tests with coverage..."
	@pytest tests/ -v --cov=lab --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Run all lab demos
run-labs: run-lab1 run-lab2
	@echo "✓ All demos complete"

# Run Lab 4.1 demo
run-lab1:
	@echo "Running Lab 4.1: Graph Library Demo..."
	@echo "────────────────────────────────────────"
	@python lab/lab_4_01_graph_library.py --demo
	@echo ""

# Run Lab 4.2 demo
run-lab2:
	@echo "Running Lab 4.2: Probabilistic Data Structures Demo..."
	@echo "────────────────────────────────────────"
	@python lab/lab_4_02_probabilistic_ds.py --demo
	@echo ""

# Run specific exercise (usage: make run-exercise FILE=easy_01_graph_construction.py)
run-exercise:
	@echo "Running exercise: $(FILE)..."
	@python exercises/practice/$(FILE) --demo 2>/dev/null || python exercises/practice/$(FILE)

# ═══════════════════════════════════════════════════════════════════════════════
# MAINTENANCE TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✓ Clean complete"

# Validate week structure
validate:
	@echo "Validating week structure..."
	@echo "────────────────────────────────────────"
	@echo "Checking required files..."
	@test -f README.md && echo "  ✓ README.md" || echo "  ✗ README.md missing"
	@test -f theory/slides.html && echo "  ✓ theory/slides.html" || echo "  ✗ theory/slides.html missing"
	@test -f theory/lecture_notes.md && echo "  ✓ theory/lecture_notes.md" || echo "  ✗ theory/lecture_notes.md missing"
	@test -f theory/learning_objectives.md && echo "  ✓ theory/learning_objectives.md" || echo "  ✗ theory/learning_objectives.md missing"
	@test -f lab/__init__.py && echo "  ✓ lab/__init__.py" || echo "  ✗ lab/__init__.py missing"
	@test -f lab/lab_4_01_graph_library.py && echo "  ✓ lab/lab_4_01_graph_library.py" || echo "  ✗ lab/lab_4_01_graph_library.py missing"
	@test -f lab/lab_4_02_probabilistic_ds.py && echo "  ✓ lab/lab_4_02_probabilistic_ds.py" || echo "  ✗ lab/lab_4_02_probabilistic_ds.py missing"
	@test -f exercises/homework.md && echo "  ✓ exercises/homework.md" || echo "  ✗ exercises/homework.md missing"
	@test -f assessments/quiz.md && echo "  ✓ assessments/quiz.md" || echo "  ✗ assessments/quiz.md missing"
	@test -f assessments/rubric.md && echo "  ✓ assessments/rubric.md" || echo "  ✗ assessments/rubric.md missing"
	@test -f assessments/self_check.md && echo "  ✓ assessments/self_check.md" || echo "  ✗ assessments/self_check.md missing"
	@test -f resources/cheatsheet.md && echo "  ✓ resources/cheatsheet.md" || echo "  ✗ resources/cheatsheet.md missing"
	@test -f resources/further_reading.md && echo "  ✓ resources/further_reading.md" || echo "  ✗ resources/further_reading.md missing"
	@test -f resources/glossary.md && echo "  ✓ resources/glossary.md" || echo "  ✗ resources/glossary.md missing"
	@test -f tests/conftest.py && echo "  ✓ tests/conftest.py" || echo "  ✗ tests/conftest.py missing"
	@test -f tests/test_lab_4_01.py && echo "  ✓ tests/test_lab_4_01.py" || echo "  ✗ tests/test_lab_4_01.py missing"
	@test -f tests/test_lab_4_02.py && echo "  ✓ tests/test_lab_4_02.py" || echo "  ✗ tests/test_lab_4_02.py missing"
	@echo "────────────────────────────────────────"
	@echo "Checking practice exercises..."
	@ls exercises/practice/easy_*.py 2>/dev/null | wc -l | xargs -I {} echo "  Easy exercises: {}"
	@ls exercises/practice/medium_*.py 2>/dev/null | wc -l | xargs -I {} echo "  Medium exercises: {}"
	@ls exercises/practice/hard_*.py 2>/dev/null | wc -l | xargs -I {} echo "  Hard exercises: {}"
	@echo "────────────────────────────────────────"
	@echo "Checking licence in README.md..."
	@grep -q "RESTRICTIVE LICENCE" README.md && echo "  ✓ Licence section present" || echo "  ✗ Licence section missing"
	@grep -q "Version 2.0.2" README.md && echo "  ✓ Licence version 2.0.2" || echo "  ✗ Wrong licence version"
	@echo "────────────────────────────────────────"
	@echo "✓ Validation complete"

# Generate documentation (placeholder)
docs:
	@echo "Documentation generation not yet implemented"
	@echo "Consider using Sphinx or MkDocs for comprehensive documentation"

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -q numpy pandas matplotlib networkx pytest pytest-cov ruff mypy
	@echo "✓ Dependencies installed"

# Create virtual environment
venv:
	@echo "Creating virtual environment..."
	@python -m venv .venv
	@echo "✓ Virtual environment created"
	@echo "Activate with: source .venv/bin/activate"

# ═══════════════════════════════════════════════════════════════════════════════
# DEVELOPMENT TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

# Quick check (faster than full suite)
quick:
	@echo "Running quick checks..."
	@ruff check lab/ --select E,F --ignore E501
	@pytest tests/ -x -q
	@echo "✓ Quick check passed"

# Watch mode (requires pytest-watch)
watch:
	@ptw tests/ -- -v

# Profile lab execution
profile:
	@echo "Profiling Lab 4.1..."
	@python -m cProfile -s cumtime lab/lab_4_01_graph_library.py --demo 2>/dev/null | head -30
