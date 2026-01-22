#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
UNIT VALIDATION SCRIPT — v4.1
═══════════════════════════════════════════════════════════════════════════════

Comprehensive validation for UNIT kits including:
- Structure verification
- Word count validation
- AI fingerprint detection
- Syntax verification
- Root file checks (README.md, Makefile, requirements.txt)

Supports all 14 UNITs with flexible structure detection.

© 2025 Antonio Clim. All rights reserved.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

VALID_UNITS = {f"{i:02d}" for i in range(1, 15)}

AI_FINGERPRINTS_TIER1 = [
    "delve", "dive into", "crucial", "pivotal", "straightforward",
    "leverage", "robust", "seamless", "cutting-edge", "best practices",
    "game-changer", "empower", "harness", "landscape", "paradigm",
    "synergy", "holistic", "ecosystem", "utilize", "facilitate",
    "bolster", "foster", "underscore", "realm", "myriad", "plethora",
    "cornerstone", "encompasses", "intricacies"
]

AI_FINGERPRINTS_TIER2 = [
    r"\blet's explore\b", r"\bin this section\b", r"\bwe will discuss\b",
    r"\bit's worth noting\b", r"\binterestingly\b", r"\bnotably\b",
    r"\bimportantly\b", r"\bessentially\b", r"\bbasically\b",
    r"\bsimply put\b", r"\bin essence\b", r"\bat its core\b",
    r"\bwhen it comes to\b", r"\bmoving forward\b"
]

AI_FINGERPRINTS_TIER3 = [
    r"^welcome to", r"^in this comprehensive", r"^let me explain",
    r"happy coding", r"good luck!", r"have fun!", r"stay tuned",
    r"feel free to", r"don't hesitate", r"i hope this helps",
    r"thanks for reading"
]

MIN_WORD_COUNTS = {
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

REQUIRED_ROOT_FILES = ["README.md", "Makefile", "requirements.txt"]


class ValidationResult(NamedTuple):
    category: str
    check: str
    passed: bool
    message: str
    severity: str


class UnitValidator:
    def __init__(self, unit: str, base_path: Path, strict: bool = False):
        self.unit = unit
        self.base_path = base_path
        self.strict = strict
        self.results: list[ValidationResult] = []
        self.ai_findings: dict[str, list[str]] = {}

    def scan_ai_fingerprints(self, content: str, filename: str) -> list[str]:
        findings = []
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
        for filename in REQUIRED_ROOT_FILES:
            path = self.base_path / filename
            if path.exists():
                self.results.append(ValidationResult("root_files", f"Root: {filename}", True, "✓ Present", "info"))
            else:
                self.results.append(ValidationResult("root_files", f"Root: {filename}", False, "✗ MISSING", "error"))

    def validate_structure(self) -> None:
        required_dirs = ["theory", "lab", "exercises", "assessments", "resources", "assets", "tests", "scripts"]
        for dir_name in required_dirs:
            path = self.base_path / dir_name
            exists = path.is_dir()
            self.results.append(
                ValidationResult("structure", f"Directory: {dir_name}/", exists, "✓ Present" if exists else "✗ MISSING", "info" if exists else "error")
            )

    def validate_word_counts(self) -> None:
        for filename, min_words in MIN_WORD_COUNTS.items():
            matches = list(self.base_path.rglob(filename))
            for path in matches:
                try:
                    content = path.read_text(encoding="utf-8")
                    wc = len(content.split())
                    passed = wc >= min_words
                    self.results.append(
                        ValidationResult("word_count", f"Words: {path.name}", passed, f"{wc}/{min_words} words", "info" if passed else "warning")
                    )
                except Exception as e:
                    self.results.append(ValidationResult("word_count", f"Words: {path.name}", False, f"Error: {e}", "error"))

    def validate_ai_fingerprints(self) -> None:
        text_files = list(self.base_path.rglob("*.md")) + list(self.base_path.rglob("*.py")) + list(self.base_path.rglob("*.html"))
        total = 0
        for path in text_files:
            try:
                content = path.read_text(encoding="utf-8")
                findings = self.scan_ai_fingerprints(content, path.name)
                total += len(findings)
                if findings:
                    self.results.append(
                        ValidationResult("ai_fingerprints", f"AI: {path.name}", False, f"⚠ {len(findings)} fingerprints found", "warning" if not self.strict else "error")
                    )
            except Exception:
                continue
        if total == 0:
            self.results.append(ValidationResult("ai_fingerprints", "AI Scan", True, "✓ ZERO fingerprints detected", "info"))

    def validate_python_syntax(self) -> None:
        for path in self.base_path.rglob("*.py"):
            result = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True)
            passed = result.returncode == 0
            self.results.append(
                ValidationResult("python", f"Syntax: {path.name}", passed, "✓ Valid" if passed else "✗ Syntax error", "info" if passed else "error")
            )

    def validate_html_naming(self) -> None:
        for path in self.base_path.rglob("*.html"):
            has_prefix = path.name.startswith(f"{self.unit}UNIT_")
            self.results.append(
                ValidationResult("html", f"Naming: {path.name}", has_prefix, "✓ Has UNIT prefix" if has_prefix else f"⚠ Missing {self.unit}UNIT_ prefix", "info" if has_prefix else "warning")
            )

    def run_all(self) -> bool:
        self.validate_root_files()
        self.validate_structure()
        self.validate_word_counts()
        self.validate_ai_fingerprints()
        self.validate_python_syntax()
        self.validate_html_naming()
        errors = sum(1 for r in self.results if (not r.passed and r.severity == "error"))
        return errors == 0

    def print_report(self) -> None:
        print(f"\n{'═' * 70}")
        print(f"  VALIDATION REPORT: {self.unit}UNIT")
        print(f"{'═' * 70}\n")
        categories: dict[str, list[ValidationResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)
        for category, results in categories.items():
            print(f"┌{'─' * 68}┐")
            print(f"│ {category.upper():^66} │")
            print(f"├{'─' * 68}┤")
            for r in results:
                status = "✓" if r.passed else ("⚠" if r.severity == "warning" else "✗")
                print(f"│ {status} {r.check[:35]:<35} {r.message[:28]:<28} │")
            print(f"└{'─' * 68}┘\n")
        if self.ai_findings:
            print(f"{'─' * 70}\nAI FINGERPRINT DETAILS:\n{'─' * 70}")
            for fn, findings in self.ai_findings.items():
                print(f"\n  {fn}:")
                for f in findings[:5]:
                    print(f"    - {f}")
                if len(findings) > 5:
                    print(f"    ... and {len(findings)-5} more")
        passed = sum(1 for r in self.results if r.passed)
        errors = sum(1 for r in self.results if (not r.passed and r.severity == "error"))
        warnings = sum(1 for r in self.results if (not r.passed and r.severity == "warning"))
        print(f"\n{'═' * 70}")
        print(f"  {'✓' if errors==0 else '✗'} {self.unit}UNIT {'PASSED' if errors==0 else 'FAILED'}")
        print(f"  {passed} passed | {errors} errors | {warnings} warnings")
        print(f"{'═' * 70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UNIT (01-14)")
    parser.add_argument("unit", help="UNIT number (01-14)")
    parser.add_argument("-s", "--strict", action="store_true", help="Treat AI fingerprints as errors")
    parser.add_argument("-p", "--path", default=".", help="Base path")
    args = parser.parse_args()

    if args.unit not in VALID_UNITS:
        print(f"Error: Invalid UNIT '{args.unit}'. Valid: 01-14")
        raise SystemExit(1)

    base_path = Path(args.path)
    unit_dirs = list(base_path.glob(f"{args.unit}UNIT*"))

    if unit_dirs:
        unit_path = unit_dirs[0]
    elif base_path.name.startswith(f"{args.unit}UNIT"):
        unit_path = base_path
    else:
        print(f"Error: Cannot find {args.unit}UNIT directory")
        raise SystemExit(1)

    validator = UnitValidator(args.unit, unit_path, args.strict)
    ok = validator.run_all()
    validator.print_report()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
