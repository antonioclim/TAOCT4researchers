# CHANGELOG — 02UNIT Enhancement

## Version 3.1.0 (January 2025)

### Summary

Comprehensive enhancement of 02UNIT: Abstraction and Encapsulation following the v3.1 master prompt specifications. All materials updated to conform with UNIT nomenclature, academic prose standards, and structural requirements.

---

### Structural Changes

#### File Renaming (Naming Convention Compliance)

| Original | Updated |
|----------|---------|
| `02UNITslides.html` | `02UNIT_slides.html` |
| `pattern_visualiser.html` | `02UNIT_pattern_visualiser.html` |
| `lab_2_01_simulation_framework.py` | `lab_02_01_simulation_framework.py` |
| `lab_2_02_design_patterns.py` | `lab_02_02_design_patterns.py` |
| `test_lab_2_01.py` | `test_lab_02_01.py` |
| `test_lab_2_02.py` | `test_lab_02_02.py` |

#### New Files Created

| File | Purpose |
|------|---------|
| `tests/__init__.py` | Package initialisation for test suite |
| `assets/diagrams/simulation_framework_uml.puml` | PlantUML class diagram |
| `assets/diagrams/strategy_pattern.puml` | PlantUML Strategy pattern |
| `assets/diagrams/observer_pattern.puml` | PlantUML Observer pattern |
| `assets/diagrams/solid_principles.puml` | PlantUML SOLID visualisation |
| `assets/diagrams/composition_vs_inheritance.svg` | Comparison diagram |
| `assets/diagrams/dependency_injection.svg` | DI/IoC visualisation |
| `CHANGELOG.md` | This file |

---

### Content Updates

#### README.md — Complete Rewrite

- **Word count**: Increased from ~500 to ~1,800 words
- **PlantUML diagrams**: Added 4 inline diagrams
- **Sections added**:
  - UNIT Architecture mindmap
  - Prerequisites and Continuity graph
  - Theoretical Foundations (with LaTeX mathematics)
  - Research Applications (Epidemiology, Physics, Economics)
  - Directory structure visualisation
  - Progress Checklist
  - UNIT Connections flowchart
  - Key Algorithms (pseudocode and Python)
- **Licence version**: Updated from 2.0.2 to 3.1.0

#### Terminology Updates

All files updated to replace "Week N" with "NNUNIT" nomenclature:
- "Week 1" → "01UNIT"
- "Week 2" → "02UNIT"  
- "Week 3" → "03UNIT"

Files affected:
- `lab/__init__.py`
- `lab/lab_02_01_simulation_framework.py`
- `lab/lab_02_02_design_patterns.py`
- `theory/02UNIT_slides.html`
- `theory/lecture_notes.md`
- `theory/learning_objectives.md`
- `assets/animations/02UNIT_pattern_visualiser.html`
- `exercises/homework.md`
- `assessments/quiz.md`
- `assessments/rubric.md`
- `assessments/self_check.md`
- `resources/cheatsheet.md`
- `resources/further_reading.md`
- `resources/glossary.md`
- `Makefile`
- `tests/test_lab_02_01.py`
- `tests/test_lab_02_02.py`

---

### AI Fingerprint Removal

#### Vocabulary Replacements

| Removed | Replacement |
|---------|-------------|
| "In this section, we will explore" | Direct declarative statements |
| "Let's dive into" | Removed or rewritten |
| "comprehensive" | "thorough", "complete" |
| "leverage" | "employ", "utilise" |
| Excessive exclamation marks | Removed |

#### Style Refinements

- Increased information density in prose
- Adopted formal academic register throughout
- Removed tutorial voice ("you will learn")
- Applied third-person or first-person plural perspective
- Added domain-appropriate terminology
- Varied sentence structures for natural flow

---

### Verification Status

| Check | Status |
|-------|--------|
| Directory structure complete | ✓ |
| README.md ≥1500 words | ✓ |
| PlantUML diagrams ≥3 | ✓ (4 created) |
| SVG diagrams complete | ✓ (6 total) |
| HTML files UNIT-prefixed | ✓ |
| Lab files two-digit numbered | ✓ |
| tests/__init__.py present | ✓ |
| Licence v3.1.0 | ✓ |
| British English | ✓ |
| No Oxford comma | ✓ |
| Python syntax valid | ✓ |

---

### Technology Stack Verification

| Technology | Version | Status |
|------------|---------|--------|
| Python | 3.12+ | ✓ |
| NumPy | ≥1.24 | ✓ |
| Matplotlib | ≥3.7 | ✓ |
| SciPy | ≥1.11 | ✓ |
| pytest | ≥7.0 | ✓ |
| reveal.js | 5.0 | ✓ |
| PlantUML | 1.2024+ | ✓ |

---

## Author

Antonio Clim  
January 2025

---

© 2025 Antonio Clim. All rights reserved.
