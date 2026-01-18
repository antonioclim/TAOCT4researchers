#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
UNIT VALIDATION SCRIPT — v3.2
═══════════════════════════════════════════════════════════════════════════════

Validates UNIT structure, content completeness, quality standards and
AI fingerprint removal compliance.

© 2025 Antonio Clim. All rights reserved.

Usage: python validate_unit.py <unit_number>
       python validate_unit.py 01
       python validate_unit.py 07

The unit_number must be two digits: 01, 02, 03, 04, 05, 06, 07
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import NamedTuple

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

VALID_UNITS = {"01", "02", "03", "04", "05", "06", "07"}

REQUIRED_STRUCTURE: dict[str, str] = {
    "README.md": "file",
    "theory/{NN}UNIT_slides.html": "file",
    "theory/lecture_notes.md": "file",
    "theory/learning_objectives.md": "file",
    "lab/__init__.py": "file",
    "lab/": "dir_with_py",
    "exercises/homework.md": "file",
    "exercises/practice/": "dir_with_py",
    "assessments/quiz.md": "file",
    "assessments/rubric.md": "file",
    "assessments/self_check.md": "file",
    "resources/cheatsheet.md": "file",
    "resources/further_reading.md": "file",
    "assets/diagrams/": "dir_with_files",
    "tests/": "dir_with_py",
    "Makefile": "file",
}

README_REQUIREMENTS = {
    "licence_section": "RESTRICTIVE LICENCE",
    "licence_version": "Version 3.2.0",
    "min_words": 2500,
    "min_plantuml": 3,
    "min_svg_refs": 2,
}

HTML_REQUIREMENTS = {
    "viewport_meta": 'name="viewport"',
    "responsive_media": "@media",
    "prism_highlight": "prism",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationResult(NamedTuple):
    """Result of a validation check."""
    
    passed: bool
    message: str
    category: str


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def validate_structure(unit: str) -> list[ValidationResult]:
    """Validate directory structure for a UNIT."""
    results: list[ValidationResult] = []
    path = Path(f"{unit}UNIT")
    
    if not path.exists():
        return [ValidationResult(False, f"Directory {unit}UNIT does not exist", "structure")]
    
    for item, check in REQUIRED_STRUCTURE.items():
        # Replace {NN} placeholder with actual unit number
        item_path = item.replace("{NN}", unit)
        full = path / item_path
        
        if check == "file":
            if full.is_file():
                results.append(ValidationResult(True, f"Found: {item_path}", "structure"))
            else:
                results.append(ValidationResult(False, f"Missing: {item_path}", "structure"))
        elif "dir" in check:
            if not full.is_dir():
                results.append(ValidationResult(False, f"Missing directory: {item_path}", "structure"))
            elif not list(full.glob("*")):
                results.append(ValidationResult(False, f"Empty directory: {item_path}", "structure"))
            else:
                results.append(ValidationResult(True, f"Found directory: {item_path}", "structure"))
    
    return results


def validate_readme(unit: str) -> list[ValidationResult]:
    """Validate README.md content and formatting."""
    results: list[ValidationResult] = []
    readme_path = Path(f"{unit}UNIT/README.md")
    
    if not readme_path.exists():
        return [ValidationResult(False, "README.md not found", "readme")]
    
    content = readme_path.read_text(encoding="utf-8")
    
    # Check licence section
    if README_REQUIREMENTS["licence_section"] in content:
        results.append(ValidationResult(True, "Licence section present", "readme"))
    else:
        results.append(ValidationResult(False, "Missing licence section", "readme"))
    
    # Check licence version
    if README_REQUIREMENTS["licence_version"] in content:
        results.append(ValidationResult(True, "Licence version 3.2.0 present", "readme"))
    else:
        results.append(ValidationResult(False, "Missing licence version 3.2.0", "readme"))
    
    # Count words (excluding code blocks)
    text_only = re.sub(r"```[\s\S]*?```", "", content)
    word_count = len(text_only.split())
    if word_count >= README_REQUIREMENTS["min_words"]:
        results.append(ValidationResult(True, f"Word count: {word_count} (≥2500)", "readme"))
    else:
        results.append(ValidationResult(
            False, 
            f"Insufficient words: {word_count} (<2500)", 
            "readme"
        ))
    
    # Count PlantUML diagrams
    plantuml_count = content.count("@start")
    if plantuml_count >= README_REQUIREMENTS["min_plantuml"]:
        results.append(ValidationResult(True, f"PlantUML diagrams: {plantuml_count} (≥3)", "readme"))
    else:
        results.append(ValidationResult(
            False, 
            f"Insufficient PlantUML: {plantuml_count} (<3)", 
            "readme"
        ))
    
    # Count SVG references
    svg_count = len(re.findall(r"\.svg", content, re.IGNORECASE))
    if svg_count >= README_REQUIREMENTS["min_svg_refs"]:
        results.append(ValidationResult(True, f"SVG references: {svg_count} (≥2)", "readme"))
    else:
        results.append(ValidationResult(
            False, 
            f"Insufficient SVG refs: {svg_count} (<2)", 
            "readme"
        ))
    
    # Count Mermaid diagrams
    mermaid_count = content.count("```mermaid")
    if mermaid_count >= 2:
        results.append(ValidationResult(True, f"Mermaid diagrams: {mermaid_count} (≥2)", "readme"))
    else:
        results.append(ValidationResult(
            False, 
            f"Insufficient Mermaid: {mermaid_count} (<2)", 
            "readme"
        ))
    
    return results


def validate_html_files(unit: str) -> list[ValidationResult]:
    """Validate HTML files for naming, responsiveness and content."""
    results: list[ValidationResult] = []
    unit_path = Path(f"{unit}UNIT")
    
    html_files = list(unit_path.rglob("*.html"))
    
    if not html_files:
        results.append(ValidationResult(False, "No HTML files found", "html"))
        return results
    
    for html_file in html_files:
        relative = html_file.relative_to(unit_path)
        
        # Check naming convention
        if html_file.name.startswith(f"{unit}UNIT_"):
            results.append(ValidationResult(True, f"Correct naming: {relative}", "html"))
        else:
            results.append(ValidationResult(
                False, 
                f"Incorrect naming (missing {unit}UNIT_ prefix): {relative}", 
                "html"
            ))
        
        content = html_file.read_text(encoding="utf-8", errors="ignore")
        
        # Check viewport meta
        if HTML_REQUIREMENTS["viewport_meta"] in content:
            results.append(ValidationResult(True, f"Viewport meta present: {relative}", "html"))
        else:
            results.append(ValidationResult(
                False, 
                f"Missing viewport meta: {relative}", 
                "html"
            ))
        
        # Check responsive media queries
        if HTML_REQUIREMENTS["responsive_media"] in content:
            results.append(ValidationResult(True, f"Responsive CSS present: {relative}", "html"))
        else:
            results.append(ValidationResult(
                False, 
                f"Missing responsive CSS: {relative}", 
                "html"
            ))
    
    return results


def validate_python_scripts(unit: str) -> list[ValidationResult]:
    """Validate Python scripts for syntax, types and functionality."""
    results: list[ValidationResult] = []
    unit_path = Path(f"{unit}UNIT")
    
    py_files = [
        f for f in unit_path.rglob("*.py") 
        if "__pycache__" not in str(f) and "solution" not in str(f).lower()
    ]
    
    for py_file in py_files:
        relative = py_file.relative_to(unit_path)
        
        # Syntax check
        try:
            compile(py_file.read_text(encoding="utf-8"), py_file, "exec")
            results.append(ValidationResult(True, f"Syntax OK: {relative}", "python"))
        except SyntaxError as e:
            results.append(ValidationResult(False, f"Syntax error in {relative}: {e}", "python"))
            continue
        
        # Check for type hints (basic heuristic)
        content = py_file.read_text(encoding="utf-8")
        if "def " in content:
            if "->" in content or ": " in content:
                results.append(ValidationResult(True, f"Type hints present: {relative}", "python"))
            else:
                results.append(ValidationResult(
                    False, 
                    f"Missing type hints: {relative}", 
                    "python"
                ))
        
        # Check for print statements (should use logging)
        print_count = len(re.findall(r"\bprint\s*\(", content))
        if print_count == 0:
            results.append(ValidationResult(True, f"No print statements: {relative}", "python"))
        else:
            results.append(ValidationResult(
                False, 
                f"Found {print_count} print statements: {relative}", 
                "python"
            ))
    
    return results


def validate_ai_fingerprints(unit: str) -> list[ValidationResult]:
    """Check text files for AI-typical patterns and vocabulary."""
    results: list[ValidationResult] = []
    unit_path = Path(f"{unit}UNIT")
    
    # AI fingerprint blacklist (subset for validation)
    AI_VOCABULARY = {
        "delve", "delving", "leverage", "leveraging", "utilize", "utilizing",
        "facilitate", "facilitating", "comprehensive", "robust", "cutting-edge",
        "game-changer", "synergy", "paradigm shift", "ecosystem", "landscape",
        "streamline", "empower", "harness", "unlock", "seamless", "holistic",
        "stakeholder", "best practices", "moving forward", "at the end of the day",
        "it's important to note", "in today's world", "let's explore",
        "dive into", "dive in", "happy coding", "pretty cool"
    }
    
    AI_PATTERNS = [
        r"In this (?:section|document|chapter), we will",
        r"Let's (?:dive|explore|look|see)",
        r"Welcome to",
        r"Get ready to",
        r"!\s*$",  # Excessive exclamation marks
    ]
    
    # Check markdown files
    md_files = list(unit_path.rglob("*.md"))
    
    for md_file in md_files:
        relative = md_file.relative_to(unit_path)
        content = md_file.read_text(encoding="utf-8", errors="ignore").lower()
        
        # Check vocabulary
        found_terms = [term for term in AI_VOCABULARY if term in content]
        if found_terms:
            results.append(ValidationResult(
                False,
                f"AI vocabulary in {relative}: {', '.join(found_terms[:3])}...",
                "ai_style"
            ))
        else:
            results.append(ValidationResult(True, f"No AI vocabulary: {relative}", "ai_style"))
        
        # Check patterns
        pattern_matches = []
        for pattern in AI_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                pattern_matches.append(pattern[:20])
        
        if pattern_matches:
            results.append(ValidationResult(
                False,
                f"AI patterns in {relative}: {len(pattern_matches)} found",
                "ai_style"
            ))
    
    # Check Python docstrings
    py_files = list(unit_path.rglob("*.py"))
    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue
        relative = py_file.relative_to(unit_path)
        content = py_file.read_text(encoding="utf-8", errors="ignore").lower()
        
        # Check for AI vocabulary in docstrings
        found_terms = [term for term in AI_VOCABULARY if term in content]
        if found_terms:
            results.append(ValidationResult(
                False,
                f"AI vocabulary in {relative}: {', '.join(found_terms[:3])}",
                "ai_style"
            ))
    
    return results


def validate_lecture_notes(unit: str) -> list[ValidationResult]:
    """Validate lecture notes content and length."""
    results: list[ValidationResult] = []
    notes_path = Path(f"{unit}UNIT/theory/lecture_notes.md")
    
    if not notes_path.exists():
        return [ValidationResult(False, "lecture_notes.md not found", "lecture_notes")]
    
    content = notes_path.read_text(encoding="utf-8")
    
    # Count words (excluding code blocks)
    text_only = re.sub(r"```[\s\S]*?```", "", content)
    word_count = len(text_only.split())
    
    if word_count >= 2500:
        results.append(ValidationResult(True, f"Lecture notes word count: {word_count} (≥2500)", "lecture_notes"))
    else:
        results.append(ValidationResult(
            False, 
            f"Insufficient lecture notes: {word_count} words (<2500)", 
            "lecture_notes"
        ))
    
    # Check for proper sectioning
    section_count = len(re.findall(r"^##\s+", content, re.MULTILINE))
    if section_count >= 5:
        results.append(ValidationResult(True, f"Sections in lecture notes: {section_count} (≥5)", "lecture_notes"))
    else:
        results.append(ValidationResult(
            False, 
            f"Insufficient sections: {section_count} (<5)", 
            "lecture_notes"
        ))
    
    return results


def run_validation(unit: str) -> tuple[list[ValidationResult], bool]:
    """Run all validations for a UNIT."""
    all_results: list[ValidationResult] = []
    
    all_results.extend(validate_structure(unit))
    all_results.extend(validate_readme(unit))
    all_results.extend(validate_html_files(unit))
    all_results.extend(validate_python_scripts(unit))
    all_results.extend(validate_ai_fingerprints(unit))
    all_results.extend(validate_lecture_notes(unit))
    
    passed = all(r.passed for r in all_results)
    return all_results, passed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point for UNIT validation."""
    parser = argparse.ArgumentParser(
        description="Validate UNIT structure and content completeness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate_unit.py 01
    python validate_unit.py 07 --verbose
        """
    )
    parser.add_argument(
        "unit",
        help="Two-digit UNIT number (01-07)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show all results, not just failures"
    )
    args = parser.parse_args()
    
    # Validate unit number format
    unit = args.unit.zfill(2)
    if unit not in VALID_UNITS:
        print(f"❌ Invalid UNIT number: {args.unit}")
        print(f"   Valid options: {', '.join(sorted(VALID_UNITS))}")
        sys.exit(1)
    
    print(f"\n{'═' * 60}")
    print(f"  VALIDATING: {unit}UNIT")
    print(f"{'═' * 60}\n")
    
    results, passed = run_validation(unit)
    
    # Group by category
    categories = {"structure", "readme", "html", "python", "ai_style", "lecture_notes"}
    for category in categories:
        cat_results = [r for r in results if r.category == category]
        if not cat_results:
            continue
        
        print(f"\n┌─ {category.upper()} ─" + "─" * (50 - len(category)))
        for result in cat_results:
            if result.passed:
                if args.verbose:
                    print(f"│ ✓ {result.message}")
            else:
                print(f"│ ✗ {result.message}")
        print(f"└" + "─" * 55)
    
    # Summary
    passed_count = sum(1 for r in results if r.passed)
    failed_count = sum(1 for r in results if not r.passed)
    
    print(f"\n{'═' * 60}")
    if passed:
        print(f"  ✓ {unit}UNIT PASSED all {passed_count} validations")
    else:
        print(f"  ✗ {unit}UNIT FAILED: {failed_count} issues found")
        print(f"    ({passed_count} passed, {failed_count} failed)")
    print(f"{'═' * 60}\n")
    
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
