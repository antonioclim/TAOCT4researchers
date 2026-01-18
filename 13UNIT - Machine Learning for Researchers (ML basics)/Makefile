# ═══════════════════════════════════════════════════════════════════════════════
# 13UNIT: Machine Learning for Researchers
# Makefile for development, testing, and demonstrations
# ═══════════════════════════════════════════════════════════════════════════════

PYTHON := python3
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff
MYPY := $(PYTHON) -m mypy

# Directories
LAB_DIR := lab
TEST_DIR := tests
PRACTICE_DIR := exercises/practice
SCRIPTS_DIR := scripts

# ═══════════════════════════════════════════════════════════════════════════════
# PHONY TARGETS
# ═══════════════════════════════════════════════════════════════════════════════

.PHONY: all test lint typecheck validate clean help
.PHONY: install coverage format
.PHONY: run-supervised run-unsupervised train evaluate tune
.PHONY: plot-learning plot-confusion run-lab-01 run-lab-02

# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT TARGET
# ═══════════════════════════════════════════════════════════════════════════════

all: lint typecheck test
	@echo "✓ All checks passed for 13UNIT"

# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════════

install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSURANCE
# ═══════════════════════════════════════════════════════════════════════════════

test:
	$(PYTEST) $(TEST_DIR) -v --cov=$(LAB_DIR) --cov-report=term-missing
	@echo "✓ Tests complete"

coverage:
	$(PYTEST) $(TEST_DIR) --cov=$(LAB_DIR) --cov=$(PRACTICE_DIR) --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/index.html"

lint:
	$(RUFF) check $(LAB_DIR) $(PRACTICE_DIR) $(TEST_DIR) --fix
	@echo "✓ Linting complete"

typecheck:
	$(MYPY) $(LAB_DIR) --ignore-missing-imports
	@echo "✓ Type checking complete"

format:
	$(RUFF) format $(LAB_DIR) $(PRACTICE_DIR) $(TEST_DIR)
	@echo "✓ Formatting complete"

validate:
	@echo "Checking required files..."
	@test -f README.md || (echo "✗ Missing README.md" && exit 1)
	@test -f requirements.txt || (echo "✗ Missing requirements.txt" && exit 1)
	@test -f theory/lecture_notes.md || (echo "✗ Missing lecture_notes.md" && exit 1)
	@test -f theory/learning_objectives.md || (echo "✗ Missing learning_objectives.md" && exit 1)
	@test -f lab/__init__.py || (echo "✗ Missing lab/__init__.py" && exit 1)
	@test -f lab/lab_13_01_supervised_learning.py || (echo "✗ Missing lab_13_01" && exit 1)
	@test -f lab/lab_13_02_unsupervised_learning.py || (echo "✗ Missing lab_13_02" && exit 1)
	@test -f exercises/homework.md || (echo "✗ Missing homework.md" && exit 1)
	@test -f assessments/quiz.md || (echo "✗ Missing quiz.md" && exit 1)
	@test -f resources/glossary.md || (echo "✗ Missing glossary.md" && exit 1)
	@echo "✓ All required files present"

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-SPECIFIC TARGETS: Machine Learning
# ═══════════════════════════════════════════════════════════════════════════════

run-supervised:
	$(PYTHON) -m lab.lab_13_01_supervised_learning
	@echo "✓ Supervised learning demo complete"

run-unsupervised:
	$(PYTHON) -m lab.lab_13_02_unsupervised_learning
	@echo "✓ Unsupervised learning demo complete"

train:
	@echo "Training sample model with cross-validation..."
	$(PYTHON) -c "\
from sklearn.datasets import load_breast_cancer; \
from sklearn.ensemble import RandomForestClassifier; \
from sklearn.model_selection import cross_val_score; \
X, y = load_breast_cancer(return_X_y=True); \
scores = cross_val_score(RandomForestClassifier(random_state=42), X, y, cv=5); \
print(f'CV Scores: {scores}'); \
print(f'Mean: {scores.mean():.4f} (+/- {scores.std()*2:.4f})')"
	@echo "✓ Training complete"

evaluate:
	@echo "Evaluating model on test set..."
	$(PYTHON) -c "\
from sklearn.datasets import load_breast_cancer; \
from sklearn.ensemble import RandomForestClassifier; \
from sklearn.model_selection import train_test_split; \
from sklearn.metrics import classification_report; \
X, y = load_breast_cancer(return_X_y=True); \
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42); \
clf = RandomForestClassifier(random_state=42).fit(X_train, y_train); \
print(classification_report(y_test, clf.predict(X_test)))"
	@echo "✓ Evaluation complete"

tune:
	@echo "Running hyperparameter tuning..."
	$(PYTHON) -c "\
from sklearn.datasets import load_breast_cancer; \
from sklearn.ensemble import RandomForestClassifier; \
from sklearn.model_selection import GridSearchCV; \
X, y = load_breast_cancer(return_X_y=True); \
param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}; \
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3); \
grid.fit(X, y); \
print(f'Best params: {grid.best_params_}'); \
print(f'Best score: {grid.best_score_:.4f}')"
	@echo "✓ Hyperparameter tuning complete"

plot-learning:
	@echo "Generating learning curves..."
	$(PYTHON) -c "\
import matplotlib; matplotlib.use('Agg'); \
import matplotlib.pyplot as plt; \
from sklearn.datasets import load_breast_cancer; \
from sklearn.ensemble import RandomForestClassifier; \
from sklearn.model_selection import learning_curve; \
import numpy as np; \
X, y = load_breast_cancer(return_X_y=True); \
sizes, train_scores, test_scores = learning_curve(RandomForestClassifier(random_state=42), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)); \
plt.figure(figsize=(10, 6)); \
plt.plot(sizes, train_scores.mean(axis=1), 'o-', label='Training'); \
plt.plot(sizes, test_scores.mean(axis=1), 'o-', label='Validation'); \
plt.xlabel('Training Set Size'); plt.ylabel('Score'); \
plt.title('Learning Curves'); plt.legend(); plt.grid(True, alpha=0.3); \
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight'); \
print('Saved: learning_curves.png')"
	@echo "✓ Learning curves generated"

plot-confusion:
	@echo "Generating confusion matrix..."
	$(PYTHON) -c "\
import matplotlib; matplotlib.use('Agg'); \
import matplotlib.pyplot as plt; \
from sklearn.datasets import load_breast_cancer; \
from sklearn.ensemble import RandomForestClassifier; \
from sklearn.model_selection import train_test_split; \
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay; \
X, y = load_breast_cancer(return_X_y=True); \
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42); \
clf = RandomForestClassifier(random_state=42).fit(X_train, y_train); \
cm = confusion_matrix(y_test, clf.predict(X_test)); \
disp = ConfusionMatrixDisplay(cm, display_labels=['Malignant', 'Benign']); \
fig, ax = plt.subplots(figsize=(8, 6)); disp.plot(ax=ax, cmap='Blues'); \
plt.title('Confusion Matrix'); \
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight'); \
print('Saved: confusion_matrix.png')"
	@echo "✓ Confusion matrix generated"

# Aliases for lab execution
run-lab-01: run-supervised
run-lab-02: run-unsupervised

# ═══════════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.png" -path "./*.png" -delete 2>/dev/null || true
	@echo "✓ Cleaned 13UNIT"

# ═══════════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════════

help:
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo "  13UNIT: Machine Learning for Researchers"
	@echo "═══════════════════════════════════════════════════════════════════"
	@echo ""
	@echo "Quality Assurance:"
	@echo "  make test        - Run tests with coverage"
	@echo "  make coverage    - Generate HTML coverage report"
	@echo "  make lint        - Run ruff linter"
	@echo "  make typecheck   - Run mypy type checker"
	@echo "  make format      - Format code with ruff"
	@echo "  make validate    - Check unit structure"
	@echo "  make all         - Run lint, typecheck, test"
	@echo ""
	@echo "Machine Learning:"
	@echo "  make run-supervised   - Execute supervised learning demo"
	@echo "  make run-unsupervised - Execute unsupervised learning demo"
	@echo "  make train            - Train sample model with cross-validation"
	@echo "  make evaluate         - Evaluate model on test set"
	@echo "  make tune             - Run hyperparameter tuning"
	@echo "  make plot-learning    - Generate learning curves"
	@echo "  make plot-confusion   - Generate confusion matrix"
	@echo ""
	@echo "Maintenance:"
	@echo "  make install     - Install dependencies"
	@echo "  make clean       - Remove generated files"
	@echo "  make help        - Show this help message"
	@echo ""
