#!/usr/bin/env python3
"""
UNIT VALIDATION SCRIPT — v4.1.0

Comprehensive validation for UNIT kits including structure verification,
word count validation, AI fingerprint detection and syntax verification.

This module implements the three-tier AI fingerprint detection system
specified in the enhancement workflow, scanning all text files for
vocabulary patterns that indicate machine-generated content.

Usage:
    python validate_unit.py <unit_number>
    python validate_unit.py 01
    python validate_unit.py 01 --strict

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


VALID_UNITS = {f"{i:02d}" for i in range(1, 15)}


AI_FINGERPRINTS_TIER1: list[str] = [
    "delve", "dive into", "crucial", "pivotal", "straightforward",
    "leverage", "robust", "seamless", "cutting-edge", "best practices",
    "game-changer", "empower", "harness", "landscape", "paradigm",
    "synergy", "holistic", "ecosystem", "utilize", "facilitate",
    "bolster", "foster", "underscore", "realm", "myriad", "plethora",
    "cornerstone", "encompasses", "intricacies",
]


AI_FINGERPRINTS_TIER2: list[str] = [
    r"\blet's explore\b", r"\bin this section\b", r"\bwe will discuss\b",
    r"\bit's worth noting\b", r"\binterestingly\b", r"\bnotably\b",
    r"\bimportantly\b", r"\bessentially\b", r"\bbasically\b",
    r"\bsimply put\b", r"\bin essence\b", r"\bat its core\b",
    r"\bwhen it comes to\b", r"\bmoving forward\b",
]


AI_FINGERPRINTS_TIER3: list[str] = [
    r"^welcome to", r"in this comprehensive", r"let me explain",
    r"happy coding", r"good luck!", r"have fun!", r"stay tuned",
    r"feel free to", r"don't hesitate", r"i hope this helps",
    r"thanks for reading",
]


MIN_WORD_COUNTS: dict[str, int] = {
    "README.md": 2500,
    "lecture_notes.md": 2500,
    "homework.md": 1500,
    "cheatsheet.md": 800,
    "quiz.md": 600,
    "glossary.md": 600,
    "further_reading.md": 400,
    "learning_objectives.md": 300,
    "self_check.md": 400,
    "rubric.md": 500,
}


REQUIRED_ROOT_FILES: list[str] = ["README.md", "Makefile", "requirements.txt"]


class ValidationResult(NamedTuple):
    """Represents the outcome of a single validation check."""

    category: str
    check: str
    passed: bool
    message: str
    severity: str


IGNORED_AI_SCAN_FILES = {"AI_FINGERPRINT_REPORT.md", "ANALYSIS_REPORT.md", "CHANGELOG.md", "VERIFICATION_REPORT.md", "validate_unit.py"}


class UnitValidator:
    """Comprehensive UNIT validator with AI fingerprint detection.

    This class implements the validation workflow specified in the UNIT
    enhancement protocol, checking structure, word counts, AI fingerprints
    and Python syntax across all kit files.
    """

    def __init__(self, unit: str, base_path: Path, strict: bool = False) -> None:
        """Initialise validator for a specific UNIT.

        Args:
            unit: Two-digit UNIT identifier (01-14).
            base_path: Root directory of the UNIT kit.
            strict: If True, treat AI fingerprints as errors rather than warnings.
        """
        self.unit = unit
        self.base_path = base_path
        self.strict = strict
        self.results: list[ValidationResult] = []
        self.ai_findings: dict[str, list[str]] = {}

    def _iter_text_files(self) -> Iterator[Path]:
        """Yield all text files eligible for AI fingerprint scanning."""
        extensions = ("*.md", "*.py", "*.html")
        for ext in extensions:
            yield from self.base_path.rglob(ext)

    def scan_ai_fingerprints(self, content: str, filename: str) -> list[str]:
        """Scan content for AI fingerprint patterns across all three tiers.

        Args:
            content: Text content to scan.
            filename: Name of the source file for reporting.

        Returns:
            List of detected AI fingerprint descriptions.
        """
        findings: list[str] = []
        content_lower = content.lower()

        for term in AI_FINGERPRINTS_TIER1:
            if term.lower() in content_lower:
                findings.append(f"Tier1: '{term}'")

        for pattern in AI_FINGERPRINTS_TIER2:
            if re.search(pattern, content_lower):
                findings.append(f"Tier2: pattern '{pattern}'")

        for pattern in AI_FINGERPRINTS_TIER3:
            if re.search(pattern, content_lower, re.MULTILINE):
                findings.append(f"Tier3: pattern '{pattern}'")

        if findings:
            self.ai_findings[filename] = findings

        return findings

    def validate_root_files(self) -> None:
        """Verify that all required root files exist."""
        for filename in REQUIRED_ROOT_FILES:
            path = self.base_path / filename
            exists = path.exists()
            self.results.append(ValidationResult(
                category="root_files",
                check=f"Root: {filename}",
                passed=exists,
                message="✓ Present" if exists else "✗ MISSING",
                severity="info" if exists else "error",
            ))

    def validate_structure(self) -> None:
        """Verify that all required directories exist."""
        required_dirs = [
            "theory", "lab", "exercises", "assessments",
            "resources", "assets", "tests", "scripts",
        ]
        for dir_name in required_dirs:
            path = self.base_path / dir_name
            exists = path.is_dir()
            self.results.append(ValidationResult(
                category="structure",
                check=f"Directory: {dir_name}/",
                passed=exists,
                message="✓ Present" if exists else "✗ MISSING",
                severity="info" if exists else "error",
            ))

    def validate_word_counts(self) -> None:
        """Verify that Markdown files meet minimum word count requirements."""
        for filename, min_words in MIN_WORD_COUNTS.items():
            matches = list(self.base_path.rglob(filename))
            for path in matches:
                try:
                    content = path.read_text(encoding="utf-8")
                    word_count = len(content.split())
                    passed = word_count >= min_words
                    self.results.append(ValidationResult(
                        category="word_count",
                        check=f"Words: {path.name}",
                        passed=passed,
                        message=f"{word_count}/{min_words} words",
                        severity="info" if passed else "warning",
                    ))
                except OSError as e:
                    self.results.append(ValidationResult(
                        category="word_count",
                        check=f"Words: {path.name}",
                        passed=False,
                        message=f"Read error: {e}",
                        severity="error",
                    ))

    def validate_ai_fingerprints(self) -> None:
        """Scan all text files for AI fingerprint patterns."""
        total_findings = 0
        for path in self._iter_text_files():
            if path.name in IGNORED_AI_SCAN_FILES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
                findings = self.scan_ai_fingerprints(content, str(path.name))
                total_findings += len(findings)
                if findings:
                    severity = "error" if self.strict else "warning"
                    self.results.append(ValidationResult(
                        category="ai_fingerprints",
                        check=f"AI: {path.name}",
                        passed=False,
                        message=f"⚠ {len(findings)} fingerprints found",
                        severity=severity,
                    ))
            except OSError:
                pass

        if total_findings == 0:
            self.results.append(ValidationResult(
                category="ai_fingerprints",
                check="AI Scan",
                passed=True,
                message="✓ ZERO fingerprints detected",
                severity="info",
            ))

    def validate_python_syntax(self) -> None:
        """Verify that all Python files have valid syntax."""
        py_files = list(self.base_path.rglob("*.py"))
        for path in py_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                passed = result.returncode == 0
                self.results.append(ValidationResult(
                    category="python",
                    check=f"Syntax: {path.name}",
                    passed=passed,
                    message="✓ Valid" if passed else "✗ Syntax error",
                    severity="info" if passed else "error",
                ))
            except OSError as e:
                self.results.append(ValidationResult(
                    category="python",
                    check=f"Syntax: {path.name}",
                    passed=False,
                    message=f"Error: {e}",
                    severity="error",
                ))

    def validate_html_naming(self) -> None:
        """Verify that HTML files follow the UNIT prefix convention."""
        html_files = list(self.base_path.rglob("*.html"))
        for path in html_files:
            has_prefix = path.name.startswith(f"{self.unit}UNIT_")
            self.results.append(ValidationResult(
                category="html",
                check=f"Naming: {path.name}",
                passed=has_prefix,
                message=(
                    "✓ Has UNIT prefix"
                    if has_prefix
                    else f"⚠ Missing {self.unit}UNIT_ prefix"
                ),
                severity="info" if has_prefix else "warning",
            ))

    def run_all(self) -> bool:
        """Execute all validation checks.

        Returns:
            True if no errors were found, False otherwise.
        """
        self.validate_root_files()
        self.validate_structure()
        self.validate_word_counts()
        self.validate_ai_fingerprints()
        self.validate_python_syntax()
        self.validate_html_naming()

        error_count = sum(
            1 for r in self.results
            if not r.passed and r.severity == "error"
        )
        return error_count == 0

    def print_report(self) -> None:
        """Print a formatted validation report to stdout."""
        border = "═" * 70
        print(f"\n{border}")
        print(f"  VALIDATION REPORT: {self.unit}UNIT")
        print(f"{border}\n")

        categories: dict[str, list[ValidationResult]] = {}
        for result in self.results:
            categories.setdefault(result.category, []).append(result)

        for category, results in categories.items():
            print(f"┌{'─' * 68}┐")
            print(f"│ {category.upper():^66} │")
            print(f"├{'─' * 68}┤")
            for r in results:
                status = "✓" if r.passed else ("⚠" if r.severity == "warning" else "✗")
                check_col = f"{status} {r.check[:35]:<35}"
                msg_col = f"{r.message[:28]:<28}"
                print(f"│ {check_col} {msg_col} │")
            print(f"└{'─' * 68}┘\n")

        if self.ai_findings:
            print(f"{'─' * 70}")
            print("AI FINGERPRINT DETAILS:")
            print(f"{'─' * 70}")
            for filename, findings in self.ai_findings.items():
                print(f"\n  {filename}:")
                for finding in findings[:5]:
                    print(f"    - {finding}")
                if len(findings) > 5:
                    print(f"    ... and {len(findings) - 5} more")

        passed = sum(1 for r in self.results if r.passed)
        errors = sum(
            1 for r in self.results
            if not r.passed and r.severity == "error"
        )
        warnings = sum(
            1 for r in self.results
            if not r.passed and r.severity == "warning"
        )

        print(f"\n{border}")
        if errors == 0:
            print(f"  ✓ {self.unit}UNIT VALIDATION PASSED")
        else:
            print(f"  ✗ {self.unit}UNIT VALIDATION FAILED")
        print(f"  {passed} passed | {errors} errors | {warnings} warnings")
        print(f"{border}\n")


def find_unit_directory(unit: str, search_path: Path) -> Path | None:
    """Locate the UNIT directory from a base search path.

    Args:
        unit: Two-digit UNIT identifier.
        search_path: Directory in which to search.

    Returns:
        Path to the UNIT directory, or None if not found.
    """
    candidates = list(search_path.glob(f"{unit}UNIT*"))
    if candidates:
        return candidates[0]
    if search_path.name.startswith(f"{unit}UNIT"):
        return search_path
    return None


def main() -> None:
    """Parse arguments and run validation."""
    parser = argparse.ArgumentParser(
        description="Validate UNIT kit structure and content (01-14).",
    )
    parser.add_argument("unit", help="UNIT number (01-14)")
    parser.add_argument(
        "-s", "--strict",
        action="store_true",
        help="Treat AI fingerprints as errors instead of warnings",
    )
    parser.add_argument(
        "-p", "--path",
        default=".",
        help="Base path to search for UNIT directory",
    )

    args = parser.parse_args()

    if args.unit not in VALID_UNITS:
        print(f"Error: Invalid UNIT '{args.unit}'. Valid values: 01-14")
        sys.exit(1)

    base_path = Path(args.path)
    unit_path = find_unit_directory(args.unit, base_path)

    if unit_path is None:
        print(f"Error: Cannot locate {args.unit}UNIT directory in {base_path}")
        sys.exit(1)

    validator = UnitValidator(args.unit, unit_path, strict=args.strict)
    passed = validator.run_all()
    validator.print_report()

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
