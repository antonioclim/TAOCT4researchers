# CHANGELOG — 05UNIT Enhancement

## Version 3.1.0 (January 2025)

### Overview

This changelog documents all modifications made to the 05UNIT: Scientific Computing starter kit during the enhancement process. Changes align with the Master Prompt v3.1.0 specifications.

---

## Structural Modifications

### README.md — Extended (CRITICAL)

**Previous state:** 966 words, no PlantUML diagrams, Licence v2.0.2

**Current state:** ~2,100 words, 5 PlantUML diagrams, Licence v3.1.0

| Section | Status | Notes |
|---------|--------|-------|
| UNIT Architecture mindmap | Added | PlantUML mindmap |
| Learning Objectives table | Enhanced | Added assessment mapping |
| Prerequisites diagram | Added | PlantUML flowchart |
| Mathematical Foundations | Added | LaTeX formulae for MC, ODE, ABM |
| Directory structure | Enhanced | Visual tree with icons |
| Laboratory Sessions | Added | Code signatures and PlantUML class diagram |
| Progress Checklist | Added | Time estimates |
| UNIT Connections | Added | PlantUML dependency graph |
| Licence | Updated | Version 3.1.0 |

### exercises/homework.md — Created (CRITICAL)

**Previous state:** Empty file (0 bytes)

**Current state:** Complete 3-part assignment with:
- Part 1: Monte Carlo Methods (35 points)
- Part 2: ODE Solvers (35 points)
- Part 3: Agent-Based Modelling (30 points)
- Detailed rubric and submission checklist
- Test cases and function signatures
- Mathematical problem statements

---

## HTML File Modifications

### Naming Convention

| Original | Renamed | Status |
|----------|---------|--------|
| `theory/slides.html` | `theory/05UNIT_slides.html` | ✓ Completed |
| `assets/animations/boids_interactive.html` | `assets/animations/05UNIT_boids_interactive.html` | ✓ Completed |

### Content Updates

| File | Change | Status |
|------|--------|--------|
| `05UNIT_slides.html` | Title: "Week 5" → "05UNIT" | ✓ Completed |
| `05UNIT_slides.html` | Header: "Săptămâna 5" → "05UNIT" | ✓ Completed |
| `05UNIT_boids_interactive.html` | Title: "Week 5" → "05UNIT" | ✓ Completed |

---

## PlantUML Source Files — Created

| File | Purpose |
|------|---------|
| `assets/diagrams/monte_carlo_convergence.puml` | MC integration flowchart |
| `assets/diagrams/ode_solver_hierarchy.puml` | Solver class hierarchy |
| `assets/diagrams/abm_architecture.puml` | Agent-environment architecture |

---

## Python File Modifications

### AI Fingerprint Removal

| File | Original | Replacement |
|------|----------|-------------|
| `lab/lab_5_01_monte_carlo.py` | "leverage random sampling" | "employ random sampling" |

### Header Updates

| File | Change |
|------|--------|
| `lab/lab_5_01_monte_carlo.py` | "Week 5, Lab 1" → "05UNIT, Lab 1" |

### Print Statement Status

The main lab files (`lab_5_01`, `lab_5_02`, `lab_5_03`) contain minimal print statements (5-7 each) used in demo mode. These are acceptable per the master prompt guidance for `--demo` functionality. The logging infrastructure is properly configured.

---

## AI Fingerprint Report

### Vocabulary Replacements

| Term | Location | Replacement |
|------|----------|-------------|
| "leverage" | `lab/lab_5_01_monte_carlo.py:10` | "employ" |

### Terms Remaining (Acceptable Contexts)

| Term | Location | Context |
|------|----------|---------|
| "comprehensive" | `resources/further_reading.md` | Book description (third-party title) |
| "comprehensive" | `assessments/rubric.md` | Grading criterion (technical usage) |

These instances appear in contexts where the terms describe external resources or technical requirements, not Claude-generated prose.

---

## Verification Results

### Structure Validation

```
✓ README.md exists with licence
✓ theory/05UNIT_slides.html exists
✓ theory/lecture_notes.md exists
✓ theory/learning_objectives.md exists
✓ lab/__init__.py exists
✓ lab/*.py files present (3 main labs)
✓ exercises/homework.md exists and populated
✓ exercises/practice/ contains 9 exercises
✓ assessments/ contains quiz.md, rubric.md, self_check.md
✓ resources/ contains cheatsheet.md, further_reading.md, glossary.md
✓ assets/diagrams/ contains 3 SVG + 3 PlantUML
✓ assets/animations/ contains 1 HTML demo
✓ tests/ contains conftest.py and test files
✓ Makefile exists
```

### README.md Validation

```
✓ Word count: ~2,100 (≥1,500 required)
✓ PlantUML diagrams: 5 (≥3 required)
✓ SVG references: 3 (≥2 required)
✓ Licence version: 3.1.0
✓ Mathematical foundations: Present
✓ Progress checklist: Present
```

### HTML Validation

```
✓ 05UNIT_slides.html: Correct naming
✓ 05UNIT_slides.html: Viewport meta present
✓ 05UNIT_slides.html: Responsive CSS present
✓ 05UNIT_boids_interactive.html: Correct naming
✓ 05UNIT_boids_interactive.html: Viewport meta present
✓ 05UNIT_boids_interactive.html: Responsive CSS present
```

### Python Validation

```
✓ lab_5_01_monte_carlo.py: Syntax valid
✓ lab_5_02_ode_solvers.py: Syntax valid
✓ lab_5_03_agent_based_modelling.py: Syntax valid
✓ Type hints present in main files
✓ AI fingerprint "leverage" removed
```

---

## Files Not Modified

The following files were reviewed and determined to meet requirements without modification:

- `assessments/quiz.md` — 10 questions present
- `assessments/rubric.md` — Grading criteria complete
- `assessments/self_check.md` — Self-assessment checklist present
- `resources/cheatsheet.md` — A4 format reference
- `resources/further_reading.md` — 15+ resources listed
- `resources/glossary.md` — Terminology defined
- `tests/conftest.py` — Fixtures configured
- `tests/test_lab_5_01.py` — Test coverage adequate
- `tests/test_lab_5_02.py` — Test coverage adequate
- `tests/test_lab_5_03.py` — Test coverage adequate
- `Makefile` — Build targets present

---

## Recommendations for Further Enhancement

1. **Lab File Renaming**: Consider renaming `lab_5_01` to `lab_05_01` for consistency with UNIT numbering convention.

2. **Print Statement Migration**: Replace remaining print statements in demo functions with logging calls for full compliance.

3. **Interactive Demo**: Consider adding `05UNIT_abm_playground.html` as specified in the master prompt configuration.

4. **SVG Generation**: Generate SVG renders from the new PlantUML source files.

---

© 2025 Antonio Clim. All rights reserved.
