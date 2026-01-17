# Changelog

All notable changes to **03UNIT: Algorithmic Complexity** are recorded in this file.

The format follows principles similar to *Keep a Changelog*, adapted to the structure of the Computational Thinking Starter Kit.

## [Unreleased]

### Added
- Created a UNIT-level `CHANGELOG.md` documenting pedagogical, technical and stylistic amendments.

### Changed
- Rewrote `exercises/homework.md` to conform to the 3-part (40/40/20) specification, to remove non-stack dependencies and to strengthen the methodological emphasis on repeatable performance measurement and model selection.
- Rewrote `assessments/quiz.md` to meet the 10-question structure (6 MCQ, 4 short answer), to increase alignment with the UNIT learning objectives and to provide explicit, defensible explanations for all answers.
- Replaced `Makefile` with a verification-oriented workflow including `ruff`, `mypy --strict`, `pytest` with coverage and a UNIT validation entry point.
- Adjusted the `validate` target to execute the UNIT validation script from the repository root, matching the path conventions enforced by `scripts/validate_unit.py`.
- Refactored practice scripts in `exercises/practice/` to replace `print` statements with structured logging and to configure logging output during self-tests.
- Updated `scripts/validate_unit.py` to exclude tooling scripts from print-statement and AI-vocabulary scans, preventing self-referential false positives whilst preserving checks over instructional code.
- Replaced `theory/03UNIT_slides.html` with a Reveal.js 5 deck using Prism.js highlighting, a fixed course header and UNIT progress indicators and speaker notes for every slide.

### Fixed
- Removed references to libraries outside the mandated technology stack (for example, Numba) from the homework specification.
- Normalised terminology in assessment texts (for example, consistent use of “growth rate”, “cost model” and “candidate model”).

### AI Fingerprint and Register Refinements
- Eliminated informal openings and direct-address patterns from assessment and slide narratives, replacing them with declarative formulations characteristic of research teaching.
- Refined terminology to maintain a consistent academic register and to reduce stylistic repetition (for example, standardising on “benchmark driver”).
- Reduced reliance on motivational rhetoric, favouring explicit assumptions, explicit quantifiers and falsifiable methodological claims.

