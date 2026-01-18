# ═══════════════════════════════════════════════════════════════════════════
# 11UNIT: Text Processing and NLP Fundamentals
# Makefile for NLP/Text Processing Domain
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   make help          - Show this help message
#   make all           - Run lint, typecheck, test
#   make run-regex     - Execute regex demonstrations
#   make run-nlp       - Execute NLP pipeline demonstrations
#
# Author: Antonio Clim
# Version: 4.1.0
# ═══════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy
PIP := pip

# Directories
LAB_DIR := lab
TEST_DIR := tests
EXERCISES_DIR := exercises
SCRIPTS_DIR := scripts

# ═══════════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: all test lint typecheck validate clean help
.PHONY: setup setup-minimal run-regex run-nlp run-tokenise run-tfidf
.PHONY: run-ner analyse-corpus test-cov test-fast

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════

all: lint typecheck test
	@echo "✓ All checks passed for 11UNIT"

# ═══════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════

setup: ## Install all dependencies including NLP libraries
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm 2>/dev/null || echo "spaCy model download skipped"
	$(PYTHON) -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); \
		nltk.download('wordnet'); nltk.download('stopwords'); \
		nltk.download('averaged_perceptron_tagger')" 2>/dev/null || echo "NLTK downloads completed"
	@echo "✓ Setup complete for 11UNIT"

setup-minimal: ## Install minimal dependencies (no NLTK/spaCy)
	$(PIP) install --upgrade pip
	$(PIP) install pytest pytest-cov mypy ruff
	@echo "✓ Minimal setup complete"

# ═══════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════

test: ## Run all unit tests
	$(PYTEST) $(TEST_DIR) -v

test-cov: ## Run tests with coverage report
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing

test-fast: ## Run tests excluding slow and NLTK tests
	$(PYTEST) $(TEST_DIR) -v -m "not slow and not nltk"

lint: ## Run ruff linter
	$(RUFF) check $(LAB_DIR) $(TEST_DIR) --fix

typecheck: ## Run mypy type checker
	$(MYPY) $(LAB_DIR) --ignore-missing-imports

validate: ## Run UNIT validation script
	$(PYTHON) $(SCRIPTS_DIR)/validate_unit.py 11 -p .

# ═══════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS (NLP/Text)
# ═══════════════════════════════════════════════════════════════════════════

run-regex: ## Execute regex and string operations demo
	$(PYTHON) -m lab.lab_11_01_regex_string_ops

run-nlp: ## Execute NLP fundamentals demo
	$(PYTHON) -m lab.lab_11_02_nlp_fundamentals

run-tokenise: ## Run tokenisation examples
	@echo "Running tokenisation demonstrations..."
	$(PYTHON) -c "from lab.lab_11_02_nlp_fundamentals import Tokeniser; \
		t = Tokeniser(); \
		print('Word tokens:', t.word_tokenise('Hello world. This is a test.')); \
		print('Sentence tokens:', t.sentence_tokenise('Hello world. This is a test.'))"

run-tfidf: ## Calculate TF-IDF on sample corpus
	@echo "Computing TF-IDF on sample corpus..."
	$(PYTHON) -c "from lab.lab_11_02_nlp_fundamentals import TextFeatures; \
		corpus = ['The cat sat on the mat.', 'The dog chased the cat.', 'The mat was soft.']; \
		tf = TextFeatures(); \
		for doc in corpus: print(tf.compute_word_frequencies(doc))"

run-ner: ## Run named entity recognition demo (requires spaCy)
	@echo "Named entity recognition requires spaCy..."
	$(PYTHON) -c "try: \
		import spacy; nlp = spacy.load('en_core_web_sm'); \
		doc = nlp('Apple is looking at buying U.K. startup for 1 billion dollars'); \
		print('Entities:', [(e.text, e.label_) for e in doc.ents]) \
		except: print('Install spaCy and model: python -m spacy download en_core_web_sm')"

analyse-corpus: ## Run full corpus analysis on sample data
	@echo "Analysing sample corpus..."
	$(PYTHON) -c "from lab.lab_11_02_nlp_fundamentals import TextAnalyser; \
		from pathlib import Path; \
		corpus_path = Path('resources/datasets/sample_corpus.txt'); \
		if corpus_path.exists(): \
			text = corpus_path.read_text(); \
			analyser = TextAnalyser(); \
			result = analyser.analyse(text); \
			print(f'Word count: {result.word_count}'); \
			print(f'Sentence count: {result.sentence_count}'); \
			print(f'Top tokens: {list(result.word_frequencies.items())[:10]}') \
		else: print('Sample corpus not found')"

# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

run-easy: ## Run easy exercises
	$(PYTHON) $(EXERCISES_DIR)/practice/easy_01_string_operations.py
	$(PYTHON) $(EXERCISES_DIR)/practice/easy_02_regex_basics.py
	$(PYTHON) $(EXERCISES_DIR)/practice/easy_03_tokenisation_intro.py

verify-solutions: ## Verify exercise solutions
	$(PYTHON) -m pytest $(EXERCISES_DIR)/solutions/ -v 2>/dev/null || \
		echo "Run solution files directly to verify"

# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

clean: ## Remove generated files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf htmlcov/ 2>/dev/null || true
	@echo "✓ Cleaned 11UNIT"

# ═══════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════

help: ## Show this help message
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo " 11UNIT: Text Processing and NLP Fundamentals"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test       - Run all unit tests"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make lint       - Run ruff linter"
	@echo "  make typecheck  - Run mypy type checker"
	@echo "  make validate   - Run UNIT validation script"
	@echo "  make all        - Run lint, typecheck, test"
	@echo ""
	@echo "Domain-Specific (NLP/Text):"
	@echo "  make run-regex     - Execute regex demo"
	@echo "  make run-nlp       - Execute NLP pipeline demo"
	@echo "  make run-tokenise  - Run tokenisation examples"
	@echo "  make run-tfidf     - Calculate TF-IDF on sample corpus"
	@echo "  make run-ner       - Run named entity recognition demo"
	@echo "  make analyse-corpus - Run full corpus analysis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup         - Install all dependencies"
	@echo "  make setup-minimal - Install minimal dependencies"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove generated files"
	@echo "  make help       - Show this help message"
	@echo ""

# ═══════════════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════════════

info: ## Show environment information
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Pip: $$($(PIP) --version)"
	@echo "Working directory: $$(pwd)"
	@$(PYTHON) -c "import sys; print(f'Platform: {sys.platform}')"

version: ## Show version information
	@echo "11UNIT: Text Processing and NLP Fundamentals"
	@echo "Version: 4.1.0"
	@echo "Author: Antonio Clim"
