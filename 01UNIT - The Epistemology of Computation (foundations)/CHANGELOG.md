# CHANGELOG — 01UNIT Enhancement

## Version 3.1.0 (January 2025)

### Overview

This document catalogues all modifications applied during the v3.1.0 enhancement of 01UNIT: The Epistemology of Computation.

---

## Nomenclature Corrections

### File Renaming

| Original | Renamed | Rationale |
|----------|---------|-----------|
| `theory/01UNITslides.html` | `theory/01UNIT_slides.html` | Underscore separator per specification |
| `assets/animations/turing_machine_animation.html` | `assets/animations/01UNIT_turing_visualiser.html` | UNIT prefix requirement |
| `lab/lab_1_01_turing_machine.py` | `lab/lab_01_01_turing_machine.py` | Two-digit unit number |
| `lab/lab_1_02_lambda_calculus.py` | `lab/lab_01_02_lambda_calculus.py` | Two-digit unit number |
| `lab/lab_1_03_ast_interpreter.py` | `lab/lab_01_03_ast_interpreter.py` | Two-digit unit number |
| `lab/solutions/lab_1_01_solutions.py` | `lab/solutions/lab_01_01_solutions.py` | Consistency |
| `lab/solutions/lab_1_02_solutions.py` | `lab/solutions/lab_01_02_solutions.py` | Consistency |
| `lab/solutions/lab_1_03_solutions.py` | `lab/solutions/lab_01_03_solutions.py` | Consistency |
| `tests/test_lab_1_01.py` | `tests/test_lab_01_01.py` | Consistency |
| `tests/test_lab_1_02.py` | `tests/test_lab_01_02.py` | Consistency |
| `tests/test_lab_1_03.py` | `tests/test_lab_01_03.py` | Consistency |

---

## README.md Enhancements

### Metrics Comparison

| Metric | Before | After | Requirement |
|--------|--------|-------|-------------|
| Word count | 869 | 2,019 | ≥1,500 |
| PlantUML diagrams | 0 | 3 | ≥3 |
| SVG references | 2 | 5 | ≥2 |
| Licence version | 2.0.2 | 3.1.0 | 3.1.0 |
| Sections | 10 | 15 | ≥12 |

### New Sections Added

1. **📊 UNIT Architecture** — PlantUML mindmap of unit structure
2. **🔗 Prerequisites and Progression** — PlantUML diagram showing curriculum flow
3. **📐 Mathematical Foundations** — Formal definitions with LaTeX notation
4. **📚 Learning Path** — PlantUML activity diagram
5. **💻 Key Algorithms** — Pseudocode and Python implementations
6. **📚 Key Concepts Summary** — Tabular concept definitions

### AI Fingerprint Removal

| Removed Pattern | Replacement |
|-----------------|-------------|
| "This week establishes" | "This unit establishes" |
| "explore the very nature" | "examining the foundations" |
| "Let's explore" | Declarative statements |
| "exciting" | Removed |
| "discover how" | "articulate the relationship" |
| Excessive exclamation marks | Period termination |
| "Welcome to" | Direct description |
| Oxford commas | Removed per British English standard |

---

## New Assets Created

### PlantUML Source Files

| File | Description |
|------|-------------|
| `assets/diagrams/turing_machine_architecture.puml` | Component diagram of TM structure |
| `assets/diagrams/ast_tree_structure.puml` | Class diagram of AST hierarchy |
| `assets/diagrams/computation_paradigms.puml` | Historical mindmap of computation models |

### SVG Diagrams

| File | Description |
|------|-------------|
| `assets/diagrams/church_turing_thesis.svg` | Equivalence visualisation of computational models |
| `assets/diagrams/lambda_reduction.svg` | Step-by-step beta reduction illustration |

---

## Compliance Verification

### Section 2: Global Requirements

- [x] British English throughout (colour, analyse, behaviour)
- [x] No Oxford comma usage
- [x] Technology stack specified (Python 3.12+, reveal.js 5.0, etc.)
- [x] Licence v3.1.0 present

### Section 3: Directory Structure

- [x] README.md with licence
- [x] theory/01UNIT_slides.html (renamed)
- [x] theory/lecture_notes.md
- [x] theory/learning_objectives.md
- [x] lab/__init__.py
- [x] lab/lab_01_01_*.py (renamed)
- [x] lab/lab_01_02_*.py (renamed)
- [x] lab/lab_01_03_*.py (renamed)
- [x] lab/solutions/
- [x] exercises/homework.md
- [x] exercises/practice/ (9 exercises)
- [x] assessments/quiz.md
- [x] assessments/rubric.md
- [x] assessments/self_check.md
- [x] resources/cheatsheet.md
- [x] resources/further_reading.md
- [x] resources/glossary.md
- [x] assets/diagrams/*.puml (3 files)
- [x] assets/diagrams/*.svg (5 files)
- [x] assets/animations/01UNIT_*.html (renamed)
- [x] tests/
- [x] Makefile

### Section 7: HTML Verification

- [x] All HTML files prefixed with 01UNIT_
- [x] Viewport meta tag present
- [x] Responsive CSS media queries present

### Section 8: README Extension

- [x] ≥1,500 words (achieved: 2,019)
- [x] ≥3 PlantUML diagrams (achieved: 3)
- [x] ≥2 SVG references (achieved: 5)
- [x] Mathematical notation (LaTeX/KaTeX)
- [x] Pseudocode with consistent keywords
- [x] Complete code examples

### Section 9: AI Fingerprint Removal

- [x] Vocabulary blacklist compliance
- [x] Structural pattern avoidance
- [x] Dense academic prose style
- [x] Formal register throughout
- [x] No excessive exclamation marks
- [x] Third person / first person plural voice

---

## Files Modified

1. `README.md` — Complete rewrite
2. `theory/01UNIT_slides.html` — Renamed only
3. `assets/animations/01UNIT_turing_visualiser.html` — Renamed only
4. `lab/lab_01_01_turing_machine.py` — Renamed only
5. `lab/lab_01_02_lambda_calculus.py` — Renamed only
6. `lab/lab_01_03_ast_interpreter.py` — Renamed only
7. `lab/solutions/lab_01_01_solutions.py` — Renamed only
8. `lab/solutions/lab_01_02_solutions.py` — Renamed only
9. `lab/solutions/lab_01_03_solutions.py` — Renamed only
10. `tests/test_lab_01_01.py` — Renamed only
11. `tests/test_lab_01_02.py` — Renamed only
12. `tests/test_lab_01_03.py` — Renamed only

## Files Created

1. `assets/diagrams/turing_machine_architecture.puml`
2. `assets/diagrams/ast_tree_structure.puml`
3. `assets/diagrams/computation_paradigms.puml`
4. `assets/diagrams/church_turing_thesis.svg`
5. `assets/diagrams/lambda_reduction.svg`
6. `CHANGELOG.md` (this file)

---

© 2025 Antonio Clim. All rights reserved.
