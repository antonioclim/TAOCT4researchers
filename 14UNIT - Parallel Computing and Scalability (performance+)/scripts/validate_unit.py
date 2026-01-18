#!/usr/bin/env python3
"""
14UNIT Validation Script

Validates the completeness and correctness of Unit 14 materials.

Usage:
    python scripts/validate_unit.py [--verbose] [--fix]

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple


class ValidationResult(NamedTuple):
    """Result of a validation check."""
    name: str
    passed: bool
    message: str


def check_file_exists(path: Path) -> ValidationResult:
    """Check if a required file exists."""
    exists = path.exists()
    return ValidationResult(
        name=f"File: {path.name}",
        passed=exists,
        message="Found" if exists else f"Missing: {path}"
    )


def check_python_syntax(path: Path) -> ValidationResult:
    """Check Python file for syntax errors."""
    try:
        with open(path, 'r') as f:
            compile(f.read(), path, 'exec')
        return ValidationResult(
            name=f"Syntax: {path.name}",
            passed=True,
            message="Valid Python syntax"
        )
    except SyntaxError as e:
        return ValidationResult(
            name=f"Syntax: {path.name}",
            passed=False,
            message=f"Syntax error: {e}"
        )


def check_imports(path: Path) -> ValidationResult:
    """Check if Python file can be imported."""
    try:
        result = subprocess.run(
            [sys.executable, '-c', f'import sys; sys.path.insert(0, "."); exec(open("{path}").read())'],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0:
            return ValidationResult(
                name=f"Import: {path.name}",
                passed=True,
                message="Imports successfully"
            )
        else:
            return ValidationResult(
                name=f"Import: {path.name}",
                passed=False,
                message=f"Import failed: {result.stderr.decode()[:100]}"
            )
    except subprocess.TimeoutExpired:
        return ValidationResult(
            name=f"Import: {path.name}",
            passed=False,
            message="Import timed out"
        )


def validate_unit(verbose: bool = False) -> tuple[int, int]:
    """Run all validation checks."""
    unit_dir = Path(__file__).parent.parent
    
    required_files = [
        'README.md',
        'theory/learning_objectives.md',
        'theory/lecture_notes.md',
        'lab/lab_14_01_multiprocessing.py',
        'lab/lab_14_02_dask_profiling.py',
        'exercises/homework.md',
        'assessments/quiz.md',
        'assessments/rubric.md',
        'resources/glossary.md',
        'resources/cheatsheet.md',
    ]
    
    python_files = list(unit_dir.glob('**/*.py'))
    
    results: list[ValidationResult] = []
    
    # Check required files
    print("Checking required files...")
    for file_path in required_files:
        results.append(check_file_exists(unit_dir / file_path))
    
    # Check Python syntax
    print("Checking Python syntax...")
    for py_file in python_files:
        if '__pycache__' not in str(py_file):
            results.append(check_python_syntax(py_file))
    
    # Print results
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    for result in results:
        status = "✓" if result.passed else "✗"
        if verbose or not result.passed:
            print(f"{status} {result.name}: {result.message}")
    
    print("=" * 60)
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    return passed, failed


def main():
    parser = argparse.ArgumentParser(description='Validate 14UNIT materials')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all results')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues')
    args = parser.parse_args()
    
    passed, failed = validate_unit(verbose=args.verbose)
    
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
