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

Usage:
    python validate_unit.py <unit_number>
    python validate_unit.py 08
    python validate_unit.py 08 --strict
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple
from collections import Counter

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VALID_UNITS = {f"{i:02d}" for i in range(1, 15)}

# AI Fingerprint terms (Tier 1 - immediate flags)
AI_FINGERPRINTS_TIER1 = [
    "delve", "dive into", "crucial", "pivotal", "straightforward",
    "leverage", "robust", "seamless", "cutting-edge", "best practices",
    "game-changer", "empower", "harness", "landscape", "paradigm",
    "synergy", "holistic", "ecosystem", "utilize", "facilitate",
    "bolster", "foster", "underscore", "realm", "myriad", "plethora",
    "cornerstone", "encompasses", "intricacies"
]

# AI Fingerprint patterns (Tier 2 - contextual)
AI_FINGERPRINTS_TIER2 = [
    r"\blet's explore\b", r"\bin this section\b", r"\bwe will discuss\b",
    r"\bit's worth noting\b", r"\binterestingly\b", r"\bnotably\b",
    r"\bimportantly\b", r"\bessentially\b", r"\bbasically\b",
    r"\bsimply put\b", r"\bin essence\b", r"\bat its core\b",
    r"\bwhen it comes to\b", r"\bmoving forward\b"
]

# AI Fingerprint openings/closings (Tier 3)
AI_FINGERPRINTS_TIER3 = [
    r"^welcome to", r"^in this comprehensive", r"^let me explain",
    r"happy coding", r"good luck!", r"have fun!", r"stay tuned",
    r"feel free to", r"don't hesitate", r"i hope this helps",
    r"thanks for reading"
]

# Minimum word counts
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

# Required root files
REQUIRED_ROOT_FILES = ["README.md", "Makefile", "requirements.txt"]


class ValidationResult(NamedTuple):
    category: str
    check: str
    passed: bool
    message: str
    severity: str


class UnitValidator:
    """Comprehensive UNIT validator with AI fingerprint detection."""
    
    def __init__(self, unit: str, base_path: Path, strict: bool = False):
        self.unit = unit
        self.base_path = base_path
        self.strict = strict
        self.results: list[ValidationResult] = []
        self.ai_findings: dict[str, list[str]] = {}
    
    def scan_ai_fingerprints(self, content: str, filename: str) -> list[str]:
        """Scan content for AI fingerprints."""
        findings = []
        content_lower = content.lower()
        
        # Tier 1 - skip "paradigm" as it's valid CS terminology
        for term in AI_FINGERPRINTS_TIER1:
            if term.lower() == "paradigm":
                continue  # Valid technical term in CS context
            if term.lower() in content_lower:
                findings.append(f"Tier1: '{term}'")
        
        # Tier 2
        for pattern in AI_FINGERPRINTS_TIER2:
            if re.search(pattern, content_lower):
                findings.append(f"Tier2: pattern '{pattern}'")
        
        # Tier 3
        for pattern in AI_FINGERPRINTS_TIER3:
            if re.search(pattern, content_lower, re.MULTILINE):
                findings.append(f"Tier3: pattern '{pattern}'")
        
        if findings:
            self.ai_findings[filename] = findings
        
        return findings
    
    def validate_root_files(self) -> None:
        """Check that required root files exist."""
        for filename in REQUIRED_ROOT_FILES:
            path = self.base_path / filename
            if path.exists():
                self.results.append(ValidationResult(
                    "root_files", f"Root: {filename}",
                    True, "✓ Present", "info"
                ))
            else:
                self.results.append(ValidationResult(
                    "root_files", f"Root: {filename}",
                    False, "✗ MISSING", "error"
                ))
    
    def validate_structure(self) -> None:
        """Validate directory structure."""
        required_dirs = ["theory", "lab", "exercises", "assessments", 
                        "resources", "assets", "tests", "scripts"]
        
        for dir_name in required_dirs:
            path = self.base_path / dir_name
            exists = path.is_dir()
            self.results.append(ValidationResult(
                "structure", f"Directory: {dir_name}/",
                exists, "✓ Present" if exists else "✗ MISSING",
                "info" if exists else "error"
            ))
    
    def validate_word_counts(self) -> None:
        """Validate minimum word counts for Markdown files."""
        for filename, min_words in MIN_WORD_COUNTS.items():
            matches = list(self.base_path.rglob(filename))
            
            for path in matches:
                try:
                    content = path.read_text(encoding="utf-8")
                    word_count = len(content.split())
                    passed = word_count >= min_words
                    
                    self.results.append(ValidationResult(
                        "word_count", f"Words: {path.name}",
                        passed,
                        f"{word_count}/{min_words} words",
                        "info" if passed else "warning"
                    ))
                except Exception as e:
                    self.results.append(ValidationResult(
                        "word_count", f"Words: {path.name}",
                        False, f"Error: {e}", "error"
                    ))
    
    def validate_ai_fingerprints(self) -> None:
        """Scan all text files for AI fingerprints."""
        text_files = (
            list(self.base_path.rglob("*.md")) +
            list(self.base_path.rglob("*.py")) +
            list(self.base_path.rglob("*.html"))
        )
        
        total_findings = 0
        for path in text_files:
            try:
                content = path.read_text(encoding="utf-8")
                findings = self.scan_ai_fingerprints(content, str(path.name))
                total_findings += len(findings)
                
                if findings:
                    self.results.append(ValidationResult(
                        "ai_fingerprints", f"AI: {path.name}",
                        False,
                        f"⚠ {len(findings)} fingerprints found",
                        "warning" if not self.strict else "error"
                    ))
            except Exception:
                pass
        
        if total_findings == 0:
            self.results.append(ValidationResult(
                "ai_fingerprints", "AI Scan",
                True, "✓ ZERO fingerprints detected", "info"
            ))
    
    def validate_python_syntax(self) -> None:
        """Validate Python files for syntax errors."""
        py_files = list(self.base_path.rglob("*.py"))
        
        for path in py_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(path)],
                    capture_output=True, text=True
                )
                passed = result.returncode == 0
                self.results.append(ValidationResult(
                    "python", f"Syntax: {path.name}",
                    passed,
                    "✓ Valid" if passed else "✗ Syntax error",
                    "info" if passed else "error"
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    "python", f"Syntax: {path.name}",
                    False, f"Error: {e}", "error"
                ))
    
    def validate_html_naming(self) -> None:
        """Verify HTML files have UNIT prefix."""
        html_files = list(self.base_path.rglob("*.html"))
        
        for path in html_files:
            has_prefix = path.name.startswith(f"{self.unit}UNIT_")
            self.results.append(ValidationResult(
                "html", f"Naming: {path.name}",
                has_prefix,
                "✓ Has UNIT prefix" if has_prefix else f"⚠ Missing {self.unit}UNIT_ prefix",
                "info" if has_prefix else "warning"
            ))
    
    def run_all(self) -> bool:
        """Run all validations."""
        self.validate_root_files()
        self.validate_structure()
        self.validate_word_counts()
        self.validate_ai_fingerprints()
        self.validate_python_syntax()
        self.validate_html_naming()
        
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")
        return errors == 0
    
    def print_report(self) -> None:
        """Print validation report."""
        print(f"\n{'═' * 70}")
        print(f"  VALIDATION REPORT: {self.unit}UNIT")
        print(f"{'═' * 70}\n")
        
        categories = {}
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
        
        # AI Fingerprint details
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
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        errors = sum(1 for r in self.results if not r.passed and r.severity == "error")
        warnings = sum(1 for r in self.results if not r.passed and r.severity == "warning")
        
        print(f"\n{'═' * 70}")
        if errors == 0:
            print(f"  ✓ {self.unit}UNIT PASSED")
        else:
            print(f"  ✗ {self.unit}UNIT FAILED")
        print(f"  {passed} passed | {errors} errors | {warnings} warnings")
        print(f"{'═' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Validate UNIT (01-14)")
    parser.add_argument("unit", help="UNIT number (01-14)")
    parser.add_argument("-s", "--strict", action="store_true",
                       help="Treat AI fingerprints as errors")
    parser.add_argument("-p", "--path", default=".", help="Base path")
    
    args = parser.parse_args()
    
    if args.unit not in VALID_UNITS:
        print(f"Error: Invalid UNIT '{args.unit}'. Valid: 01-14")
        sys.exit(1)
    
    base_path = Path(args.path)
    unit_dirs = list(base_path.glob(f"{args.unit}UNIT*"))
    
    if unit_dirs:
        unit_path = unit_dirs[0]
    elif base_path.name.startswith(f"{args.unit}UNIT"):
        unit_path = base_path
    else:
        unit_path = base_path
    
    validator = UnitValidator(args.unit, unit_path, args.strict)
    passed = validator.run_all()
    validator.print_report()
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
