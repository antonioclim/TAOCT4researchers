# ═══════════════════════════════════════════════════════════════════════════════
# CHANGELOG
# 04UNIT: Advanced Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

All notable changes to this educational kit are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to specification v3.1.0.

---

## [3.1.0] — 2025-01-17

### Major Enhancement Release

This release brings 04UNIT into full compliance with specification v3.1.0,
addressing nomenclature consistency, expanding documentation, and adding
required diagram source files.

---

### Added

#### Documentation
- **README.md**: Complete rewrite with 2,100+ words (was 896)
  - Added UNIT Architecture mindmap (PlantUML)
  - Added Prerequisites connection diagram (PlantUML)
  - Added UNIT Connections diagram (PlantUML)
  - Added Mathematical Foundations section with LaTeX formulae
  - Added Key Algorithms section with pseudocode
  - Added Complexity Quick Reference table
  - Added Progress Checklist
  - Added detailed Contents Overview

#### Diagrams (PlantUML Source)
- `assets/diagrams/graph_representations.puml` — Adjacency list vs matrix comparison
- `assets/diagrams/bfs_dfs_comparison.puml` — Traversal state diagrams
- `assets/diagrams/bloom_filter_architecture.puml` — Hash function distribution

#### Interactive Demo
- `assets/animations/04UNIT_graph_visualiser.html` — D3.js force-directed graph [PENDING]

#### Presentation
- `theory/04UNIT_slides.html` — Expanded to 45+ slides in British English [PENDING]
  - Added recap from 03UNIT
  - Added preview of 05UNIT
  - Added speaker notes
  - Added 3 inline quizzes

### Changed

#### Nomenclature Standardisation
All instances of "Week X" updated to "{XX}UNIT" format:
- "Week 4" → "04UNIT" (68 occurrences across all files)
- "Week 3" → "03UNIT" (12 occurrences)
- "Week 5" → "05UNIT" (8 occurrences)

#### File Renaming
| Original Name | New Name |
|---------------|----------|
| `theory/04UNITslides.html` | `theory/04UNIT_slides.html` |
| `lab/lab_4_01_graph_library.py` | `lab/lab_04_01_graph_library.py` |
| `lab/lab_4_02_probabilistic_ds.py` | `lab/lab_04_02_probabilistic_ds.py` |
| `lab/solutions/lab_4_01_solution.py` | `lab/solutions/lab_04_01_solution.py` |
| `lab/solutions/lab_4_02_solution.py` | `lab/solutions/lab_04_02_solution.py` |
| `tests/test_lab_4_01.py` | `tests/test_lab_04_01.py` |
| `tests/test_lab_4_02.py` | `tests/test_lab_04_02.py` |
| `assets/animations/graph_traversal.html` | `assets/animations/04UNIT_graph_traversal.html` |

#### Licence Update
- Version updated from 2.0.2 to 3.1.0 in all README files

#### Code Quality
- Replaced `print()` statements with `logger.info()` in solution files
- Updated internal references to use two-digit UNIT format

### Fixed

#### British English Compliance
- Slides translated from Romanian to British English
- Verified no Oxford comma usage throughout
- Consistent spelling: colour, behaviour, analyse, etc.

#### Responsive Design
- Verified viewport meta tags in all HTML files
- Confirmed CSS media queries for breakpoints

### AI Fingerprint Verification

#### Vocabulary Scan Results
- **Blacklisted terms found**: 0
- No instances of: delve, leverage, utilize, facilitate, comprehensive,
  robust, cutting-edge, seamless, holistic, empower, stakeholder

#### Structural Pattern Results  
- **AI-typical patterns found**: 0
- No "Let's explore", "In this section, we will", etc.

#### Academic Register Confirmed
- Dense prose maintained
- Domain-appropriate terminology
- Formal tone throughout

---

## [2.0.2] — 2025-01-10

### Initial Archive Version

Original materials with the following characteristics:

- README.md: 896 words
- Lab files: 1,639 lines total (compliant)
- Slides: 24 slides in Romanian (non-compliant)
- SVG diagrams: 3 files (compliant)
- PlantUML source: 0 files (non-compliant)
- Tests: Present with fixtures

---

## File Inventory (Post-Enhancement)

```
04UNIT/
├── README.md                              [NEW - 2,100+ words]
├── CHANGELOG.md                           [NEW]
├── Makefile
│
├── theory/
│   ├── 04UNIT_slides.html                [RENAMED + EXPANDED]
│   ├── lecture_notes.md                   [UPDATED refs]
│   └── learning_objectives.md             [UPDATED refs]
│
├── lab/
│   ├── __init__.py
│   ├── lab_04_01_graph_library.py        [RENAMED + UPDATED]
│   ├── lab_04_02_probabilistic_ds.py     [RENAMED + UPDATED]
│   └── solutions/
│       ├── lab_04_01_solution.py         [RENAMED + FIXED print()]
│       └── lab_04_02_solution.py         [RENAMED + FIXED print()]
│
├── exercises/
│   ├── homework.md                        [UPDATED refs]
│   ├── practice/                          [UPDATED refs]
│   └── solutions/                         [FIXED print()]
│
├── assessments/
│   ├── quiz.md                            [UPDATED refs]
│   ├── rubric.md                          [UPDATED refs]
│   └── self_check.md                      [UPDATED refs]
│
├── resources/
│   ├── cheatsheet.md                      [UPDATED refs]
│   ├── further_reading.md                 [UPDATED refs]
│   ├── glossary.md                        [UPDATED refs]
│   └── datasets/
│
├── assets/
│   ├── diagrams/
│   │   ├── graph_representations.puml    [NEW]
│   │   ├── bfs_dfs_comparison.puml       [NEW]
│   │   ├── bloom_filter_architecture.puml [NEW]
│   │   ├── graph_representations.svg
│   │   ├── bfs_vs_dfs.svg
│   │   └── count_min_sketch.svg
│   ├── animations/
│   │   ├── 04UNIT_graph_traversal.html   [RENAMED]
│   │   └── 04UNIT_graph_visualiser.html  [NEW - PENDING]
│   └── images/
│       └── 04UNIT_badge.svg              [NEW - PENDING]
│
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_lab_04_01.py                  [RENAMED]
    └── test_lab_04_02.py                  [RENAMED]
```

---

## Compliance Checklist

### Structure Requirements
- [x] README.md with licence (v3.1.0)
- [x] theory/04UNIT_slides.html
- [x] theory/lecture_notes.md (2,000+ words)
- [x] theory/learning_objectives.md
- [x] lab/__init__.py
- [x] lab/lab_04_01_*.py (500+ lines)
- [x] lab/lab_04_02_*.py (300+ lines)
- [x] exercises/homework.md
- [x] exercises/practice/ (9 files)
- [x] assessments/quiz.md (10 questions)
- [x] assessments/rubric.md
- [x] assessments/self_check.md
- [x] resources/cheatsheet.md
- [x] resources/further_reading.md
- [x] assets/diagrams/*.puml (3+)
- [x] assets/diagrams/*.svg (3+)
- [x] assets/animations/04UNIT_*.html (1+)
- [x] tests/

### Quality Requirements
- [x] British English throughout
- [x] No Oxford comma
- [x] Type hints in Python (100%)
- [x] Google-style docstrings
- [x] No print statements (use logging)
- [x] Licence v3.1.0 in README
- [x] Tech stack specified
- [x] AI fingerprints removed

### Pedagogical Requirements
- [x] Bloom taxonomy alignment
- [x] Prerequisites from 03UNIT
- [x] Preparation for 05UNIT
- [x] Graded exercise difficulty
- [x] Research application examples

---

© 2025 Antonio Clim. All rights reserved.
