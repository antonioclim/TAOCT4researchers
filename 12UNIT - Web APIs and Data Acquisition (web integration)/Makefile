# ═══════════════════════════════════════════════════════════════════════════════
# 12UNIT: Web APIs and Data Acquisition
# Makefile for Development, Testing and Documentation
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PLANTUML := plantuml

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISE_DIR := exercises
SOLUTION_DIR := exercises/solutions
DIAGRAM_DIR := assets/diagrams
SCRIPTS_DIR := scripts

# ═══════════════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all test lint typecheck validate clean help
.PHONY: run-api run-scraper run-flask run-pagination run-auth test-api
.PHONY: install install-dev diagrams stats

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

all: lint typecheck test
	@echo "✓ All checks passed for 12UNIT"

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

install:
	$(PYTHON) -m pip install -r requirements.txt --break-system-packages

install-dev: install
	$(PYTHON) -m pip install black httpx --break-system-packages

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════════

test:
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing

test-cov:
	$(PYTEST) $(TEST_DIR) --cov=$(LAB_DIR) --cov-report=html --cov-report=term

test-api:
	$(PYTEST) $(TEST_DIR)/test_lab_12_01.py -v -m "not integration"

test-scraper:
	$(PYTEST) $(TEST_DIR)/test_lab_12_02.py -v -m "not integration"

lint:
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) $(EXERCISE_DIR) --fix

typecheck:
	$(MYPY) $(LAB_DIR) --ignore-missing-imports

validate:
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 12 --path .

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS (Web/APIs)
# ═══════════════════════════════════════════════════════════════════════════════

run-api:
	@echo "Running Lab 12.01: API Consumption demonstrations..."
	$(PYTHON) -c "from lab.lab_12_01_api_consumption import demonstrate_http_methods; print(demonstrate_http_methods())"

run-scraper:
	@echo "Running Lab 12.02: Web Scraping demonstrations..."
	$(PYTHON) -c "from lab.lab_12_02_web_scraping_flask import EthicalScraper; s = EthicalScraper('https://example.com'); print('Scraper initialised with base URL:', s.base_url)"

run-flask:
	@echo "Starting Flask development server on port 5012..."
	$(PYTHON) -c "from lab.lab_12_02_web_scraping_flask import create_api_app; app = create_api_app(); app.run(debug=True, port=5012)"

run-pagination:
	@echo "Running pagination handling demonstration..."
	$(PYTHON) -c "from lab.lab_12_01_api_consumption import paginate_api; print('Pagination example: Use paginate_api() with cursor-based APIs')"

run-auth:
	@echo "Running authentication examples..."
	$(PYTHON) -c "from lab.lab_12_01_api_consumption import AuthenticatedClient; print('Authentication patterns: API Key, Bearer Token, Basic Auth, OAuth2')"

run-research-api:
	@echo "Starting Research Data API on port 5013..."
	$(PYTHON) -c "from lab.lab_12_02_web_scraping_flask import create_research_api; from pathlib import Path; app = create_research_api(Path('resources/datasets/sample_works.json')); app.run(debug=True, port=5013)"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

diagrams:
	@echo "Generating PlantUML diagrams..."
	@if command -v $(PLANTUML) > /dev/null; then \
		$(PLANTUML) -tsvg $(DIAGRAM_DIR)/plantuml/*.puml -o ../svg/generated/; \
	else \
		echo "PlantUML not installed. Install with: apt install plantuml"; \
	fi

stats:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo "  12UNIT: Web APIs and Data Acquisition - Statistics"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Markdown files:"
	@find . -name "*.md" -exec wc -w {} + | tail -1
	@echo ""
	@echo "Python files:"
	@find . -name "*.py" -exec wc -l {} + | tail -1
	@echo ""
	@echo "HTML files:"
	@find . -name "*.html" -exec wc -l {} + | tail -1
	@echo ""
	@echo "Total files:"
	@find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.html" -o -name "*.json" \) | wc -l

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ coverage.xml 2>/dev/null || true
	@echo "✓ Cleaned 12UNIT"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo "═══════════════════════════════════════════════════════════════"
	@echo " 12UNIT: Web APIs and Data Acquisition"
	@echo "═══════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Installation:"
	@echo "  make install      - Install Python dependencies"
	@echo "  make install-dev  - Install development dependencies"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test         - Run tests with coverage"
	@echo "  make test-api     - Run API tests with mocked responses"
	@echo "  make test-scraper - Run scraper tests"
	@echo "  make lint         - Run ruff linter"
	@echo "  make typecheck    - Run mypy type checker"
	@echo "  make validate     - Run UNIT validation script"
	@echo "  make all          - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific (Web/APIs):"
	@echo "  make run-api         - Execute API consumption demo"
	@echo "  make run-scraper     - Execute web scraping demo"
	@echo "  make run-flask       - Launch Flask demo server"
	@echo "  make run-pagination  - Run pagination handling demo"
	@echo "  make run-auth        - Run authentication examples"
	@echo "  make run-research-api - Start Research Data API"
	@echo ""
	@echo "Documentation:"
	@echo "  make diagrams     - Generate PlantUML diagrams"
	@echo "  make stats        - Show file statistics"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove generated files"
	@echo "  make help         - Show this help message"
	@echo ""
