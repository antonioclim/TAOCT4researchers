#!/usr/bin/env python3
"""
TAOCT4researchers — Unit Validation Script
The Art of Computational Thinking for Researchers
Version 5.0.0

This script validates the structural integrity and content quality of all
instructional units in the repository.

Usage:
    python scripts/validate_all_units.py [OPTIONS]

Options:
    --unit N        Validate specific unit only (01-14)
    --verbose, -v   Enable verbose output
    --strict        Treat warnings as errors
    --json          Output results as JSON
    --help          Display help message
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

UNIT_NAMES = {
    "01": "The Epistemology of Computation (foundations)",
    "02": "Abstraction and Encapsulation (patterns)",
    "03": "Algorithmic Complexity (performance)",
    "04": "Advanced Data Structures (design)",
    "05": "Scientific Computing (simulations)",
    "06": "Visualisation for Research (communication)",
    "07": "Reproducibility and Capstone (integration)",
    "08": "Recursion and Dynamic Programming (algorithms)",
    "09": "Exception Handling and Defensive Code (robustness)",
    "10": "Data Persistence and Serialisation (storage)",
    "11": "Text Processing and NLP Fundamentals (text analysis)",
    "12": "Web APIs and Data Acquisition (web integration)",
    "13": "Machine Learning for Researchers (ML basics)",
    "14": "Parallel Computing and Scalability (performance+)",
}

# Required directory structure
REQUIRED_DIRS = [
    "theory",
    "lab",
    "exercises",
    "exercises/practice",
    "exercises/solutions",
    "assessments",
    "resources",
    "assets",
    "assets/diagrams",
    "tests",
    "scripts",
]

# Required files
REQUIRED_FILES = [
    "README.md",
    "Makefile",
    "requirements.txt",
    "lab/__init__.py",
    "tests/__init__.py",
    "tests/conftest.py",
    "theory/lecture_notes.md",
    "theory/learning_objectives.md",
    "assessments/quiz.md",
    "assessments/rubric.md",
    "assessments/self_check.md",
    "resources/cheatsheet.md",
    "resources/glossary.md",
    "resources/further_reading.md",
]

# Content requirements
MIN_README_WORDS = 2500
MIN_LECTURE_NOTES_WORDS = 1500
MIN_PLANTUML_DIAGRAMS = 2
MIN_SVG_ASSETS = 2
MIN_EXERCISES_PER_DIFFICULTY = 3

# Vocabulary to flag (potential AI patterns)
FLAGGED_VOCABULARY = [
    "delve",
    "tapestry",
    "multifaceted",
    "utilize",
    "leverage",
    "cutting-edge",
    "game-changing",
    "revolutionary",
    "unlock",
    "empower",
    "realm",
    "embark",
    "crucial",
    "essential",
]


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check: str
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info


@dataclass
class UnitValidation:
    """Validation results for a single unit."""

    unit_num: str
    unit_name: str
    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all error-level validations passed."""
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def error_count(self) -> int:
        """Count of failed error-level validations."""
        return sum(1 for r in self.results if not r.passed and r.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warnings."""
        return sum(
            1 for r in self.results if not r.passed and r.severity == "warning"
        )


# -----------------------------------------------------------------------------
# Validation Functions
# -----------------------------------------------------------------------------


def find_unit_dir(unit_num: str) -> Path | None:
    """Find the directory for a given unit number."""
    pattern = f"{unit_num}UNIT*"
    matches = list(REPO_ROOT.glob(pattern))
    return matches[0] if matches else None


def validate_directory_structure(unit_dir: Path) -> list[ValidationResult]:
    """Validate that all required directories exist."""
    results = []

    for dir_path in REQUIRED_DIRS:
        full_path = unit_dir / dir_path
        exists = full_path.is_dir()
        results.append(
            ValidationResult(
                check=f"Directory: {dir_path}",
                passed=exists,
                message=f"{'Found' if exists else 'Missing'}: {dir_path}",
            )
        )

    return results


def validate_required_files(unit_dir: Path) -> list[ValidationResult]:
    """Validate that all required files exist."""
    results = []

    for file_path in REQUIRED_FILES:
        full_path = unit_dir / file_path
        exists = full_path.is_file()
        results.append(
            ValidationResult(
                check=f"File: {file_path}",
                passed=exists,
                message=f"{'Found' if exists else 'Missing'}: {file_path}",
            )
        )

    return results


def validate_readme_content(unit_dir: Path) -> list[ValidationResult]:
    """Validate README.md content requirements."""
    results = []
    readme_path = unit_dir / "README.md"

    if not readme_path.is_file():
        return [
            ValidationResult(
                check="README content",
                passed=False,
                message="README.md not found",
            )
        ]

    content = readme_path.read_text(encoding="utf-8")
    word_count = len(content.split())

    # Word count check
    results.append(
        ValidationResult(
            check="README word count",
            passed=word_count >= MIN_README_WORDS,
            message=f"Word count: {word_count} (minimum: {MIN_README_WORDS})",
        )
    )

    # PlantUML diagrams
    plantuml_count = content.count("```plantuml")
    results.append(
        ValidationResult(
            check="PlantUML diagrams",
            passed=plantuml_count >= MIN_PLANTUML_DIAGRAMS,
            message=f"PlantUML blocks: {plantuml_count} (minimum: {MIN_PLANTUML_DIAGRAMS})",
            severity="warning",
        )
    )

    # Check for flagged vocabulary
    flagged_found = []
    content_lower = content.lower()
    for word in FLAGGED_VOCABULARY:
        if word in content_lower:
            flagged_found.append(word)

    if flagged_found:
        results.append(
            ValidationResult(
                check="Vocabulary review",
                passed=False,
                message=f"Flagged terms found: {', '.join(flagged_found)}",
                severity="warning",
            )
        )
    else:
        results.append(
            ValidationResult(
                check="Vocabulary review",
                passed=True,
                message="No flagged vocabulary detected",
                severity="info",
            )
        )

    return results


def validate_assets(unit_dir: Path) -> list[ValidationResult]:
    """Validate asset files."""
    results = []
    assets_dir = unit_dir / "assets"
    diagrams_dir = assets_dir / "diagrams"

    # SVG files
    svg_files = list(diagrams_dir.glob("*.svg")) if diagrams_dir.exists() else []
    results.append(
        ValidationResult(
            check="SVG assets",
            passed=len(svg_files) >= MIN_SVG_ASSETS,
            message=f"SVG files: {len(svg_files)} (minimum: {MIN_SVG_ASSETS})",
            severity="warning",
        )
    )

    # PlantUML files
    puml_files = list(diagrams_dir.glob("*.puml")) if diagrams_dir.exists() else []
    results.append(
        ValidationResult(
            check="PlantUML files",
            passed=len(puml_files) >= MIN_PLANTUML_DIAGRAMS,
            message=f"PlantUML files: {len(puml_files)} (minimum: {MIN_PLANTUML_DIAGRAMS})",
            severity="warning",
        )
    )

    # HTML animations
    animations_dir = assets_dir / "animations"
    html_files = (
        list(animations_dir.glob("*.html")) if animations_dir.exists() else []
    )
    results.append(
        ValidationResult(
            check="Interactive animations",
            passed=len(html_files) >= 1,
            message=f"HTML animations: {len(html_files)}",
            severity="warning",
        )
    )

    return results


def validate_exercises(unit_dir: Path) -> list[ValidationResult]:
    """Validate exercise structure."""
    results = []
    practice_dir = unit_dir / "exercises" / "practice"
    solutions_dir = unit_dir / "exercises" / "solutions"

    if not practice_dir.exists():
        return [
            ValidationResult(
                check="Exercises",
                passed=False,
                message="exercises/practice directory not found",
            )
        ]

    # Count exercises by difficulty
    easy_count = len(list(practice_dir.glob("easy_*.py"))) + len(
        list(practice_dir.glob("easy_*.md"))
    )
    medium_count = len(list(practice_dir.glob("medium_*.py"))) + len(
        list(practice_dir.glob("medium_*.md"))
    )
    hard_count = len(list(practice_dir.glob("hard_*.py"))) + len(
        list(practice_dir.glob("hard_*.md"))
    )

    for difficulty, count in [
        ("easy", easy_count),
        ("medium", medium_count),
        ("hard", hard_count),
    ]:
        results.append(
            ValidationResult(
                check=f"Exercises ({difficulty})",
                passed=count >= MIN_EXERCISES_PER_DIFFICULTY,
                message=f"{difficulty.capitalize()}: {count} (minimum: {MIN_EXERCISES_PER_DIFFICULTY})",
                severity="warning",
            )
        )

    # Check for solutions
    solution_files = list(solutions_dir.glob("*.py")) if solutions_dir.exists() else []
    results.append(
        ValidationResult(
            check="Exercise solutions",
            passed=len(solution_files) > 0,
            message=f"Solution files: {len(solution_files)}",
            severity="warning",
        )
    )

    return results


def validate_tests(unit_dir: Path) -> list[ValidationResult]:
    """Validate test files."""
    results = []
    tests_dir = unit_dir / "tests"

    if not tests_dir.exists():
        return [
            ValidationResult(
                check="Tests",
                passed=False,
                message="tests directory not found",
            )
        ]

    # Count test files
    test_files = list(tests_dir.glob("test_*.py"))
    results.append(
        ValidationResult(
            check="Test files",
            passed=len(test_files) >= 2,
            message=f"Test files: {len(test_files)} (minimum: 2)",
        )
    )

    # Check for conftest.py
    conftest = tests_dir / "conftest.py"
    results.append(
        ValidationResult(
            check="Test configuration",
            passed=conftest.is_file(),
            message=f"conftest.py: {'present' if conftest.is_file() else 'missing'}",
        )
    )

    return results


def validate_lab_files(unit_dir: Path) -> list[ValidationResult]:
    """Validate laboratory files."""
    results = []
    lab_dir = unit_dir / "lab"

    if not lab_dir.exists():
        return [
            ValidationResult(
                check="Laboratory",
                passed=False,
                message="lab directory not found",
            )
        ]

    # Find lab files
    unit_num = unit_dir.name[:2]
    lab_pattern = f"lab_{unit_num}_*.py"
    lab_files = list(lab_dir.glob(lab_pattern))

    results.append(
        ValidationResult(
            check="Laboratory files",
            passed=len(lab_files) >= 2,
            message=f"Lab files matching {lab_pattern}: {len(lab_files)} (minimum: 2)",
        )
    )

    # Check for solutions
    solutions_dir = lab_dir / "solutions"
    solution_files = (
        list(solutions_dir.glob("*.py")) if solutions_dir.exists() else []
    )
    results.append(
        ValidationResult(
            check="Lab solutions",
            passed=len(solution_files) >= 1,
            message=f"Solution files: {len(solution_files)}",
            severity="warning",
        )
    )

    return results


def validate_unit(unit_num: str) -> UnitValidation:
    """Perform full validation of a single unit."""
    unit_name = UNIT_NAMES.get(unit_num, f"Unit {unit_num}")
    unit_dir = find_unit_dir(unit_num)

    validation = UnitValidation(unit_num=unit_num, unit_name=unit_name)

    if unit_dir is None:
        validation.results.append(
            ValidationResult(
                check="Unit directory",
                passed=False,
                message=f"Unit {unit_num} directory not found",
            )
        )
        return validation

    # Run all validations
    validation.results.extend(validate_directory_structure(unit_dir))
    validation.results.extend(validate_required_files(unit_dir))
    validation.results.extend(validate_readme_content(unit_dir))
    validation.results.extend(validate_assets(unit_dir))
    validation.results.extend(validate_exercises(unit_dir))
    validation.results.extend(validate_tests(unit_dir))
    validation.results.extend(validate_lab_files(unit_dir))

    return validation


# -----------------------------------------------------------------------------
# Output Functions
# -----------------------------------------------------------------------------


def print_validation_result(result: ValidationResult, verbose: bool = False) -> None:
    """Print a single validation result."""
    if result.passed:
        symbol = "✓"
        colour = "\033[92m"  # Green
    elif result.severity == "warning":
        symbol = "⚠"
        colour = "\033[93m"  # Yellow
    else:
        symbol = "✗"
        colour = "\033[91m"  # Red

    reset = "\033[0m"

    if verbose or not result.passed:
        print(f"  {colour}{symbol}{reset} {result.check}: {result.message}")


def print_unit_validation(validation: UnitValidation, verbose: bool = False) -> None:
    """Print validation results for a unit."""
    status = "PASSED" if validation.passed else "FAILED"
    colour = "\033[92m" if validation.passed else "\033[91m"
    reset = "\033[0m"

    print(f"\n{colour}[{status}]{reset} {validation.unit_num}UNIT: {validation.unit_name}")
    print(f"  Errors: {validation.error_count}, Warnings: {validation.warning_count}")

    for result in validation.results:
        print_validation_result(result, verbose)


def output_json(validations: list[UnitValidation]) -> None:
    """Output results as JSON."""
    output: dict[str, Any] = {
        "summary": {
            "total_units": len(validations),
            "passed": sum(1 for v in validations if v.passed),
            "failed": sum(1 for v in validations if not v.passed),
        },
        "units": [],
    }

    for v in validations:
        unit_data = {
            "unit_num": v.unit_num,
            "unit_name": v.unit_name,
            "passed": v.passed,
            "error_count": v.error_count,
            "warning_count": v.warning_count,
            "results": [
                {
                    "check": r.check,
                    "passed": r.passed,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in v.results
            ],
        }
        output["units"].append(unit_data)

    print(json.dumps(output, indent=2))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate TAOCT4researchers unit structure and content"
    )
    parser.add_argument(
        "--unit",
        type=str,
        help="Validate specific unit only (01-14)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Determine which units to validate
    if args.unit:
        unit_nums = [args.unit.zfill(2)]
    else:
        unit_nums = list(UNIT_NAMES.keys())

    # Run validations
    validations = [validate_unit(num) for num in unit_nums]

    # Output results
    if args.json:
        output_json(validations)
    else:
        print("\n" + "=" * 70)
        print(" TAOCT4researchers — Unit Validation Report")
        print("=" * 70)

        for validation in validations:
            print_unit_validation(validation, args.verbose)

        # Summary
        passed = sum(1 for v in validations if v.passed)
        failed = len(validations) - passed
        total_errors = sum(v.error_count for v in validations)
        total_warnings = sum(v.warning_count for v in validations)

        print("\n" + "=" * 70)
        print(f" Summary: {passed}/{len(validations)} units passed")
        print(f" Total errors: {total_errors}, Total warnings: {total_warnings}")
        print("=" * 70 + "\n")

    # Determine exit code
    if args.strict:
        all_passed = all(
            v.passed and v.warning_count == 0 for v in validations
        )
    else:
        all_passed = all(v.passed for v in validations)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
