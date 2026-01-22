#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

VALID_UNITS = {f"{i:02d}" for i in range(1, 15)}

AI_TIER1 = [
    "delve", "dive into", "crucial", "pivotal", "straightforward",
    "leverage", "robust", "seamless", "cutting-edge", "best practices",
    "game-changer", "empower", "harness", "landscape", "paradigm",
    "synergy", "holistic", "ecosystem", "utilize", "facilitate",
    "bolster", "foster", "underscore", "realm", "myriad", "plethora",
    "cornerstone", "encompasses", "intricacies",
]

AI_TIER2 = [
    r"\blet's explore\b", r"\bin this section\b", r"\bwe will discuss\b",
    r"\bit's worth noting\b", r"\binterestingly\b", r"\bnotably\b",
    r"\bimportantly\b", r"\bessentially\b", r"\bbasically\b",
    r"\bsimply put\b", r"\bin essence\b", r"\bat its core\b",
    r"\bwhen it comes to\b", r"\bmoving forward\b",
]

AI_TIER3 = [
    r"^welcome to", r"^in this comprehensive", r"^let me explain",
    "happy coding", "good luck!", "have fun!", "stay tuned",
    "feel free to", "don't hesitate", "i hope this helps", "thanks for reading",
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

    def scan_ai(self, content: str, filename: str) -> list[str]:
        findings: list[str] = []
        lower = content.lower()

        for term in AI_TIER1:
            if term in lower:
                findings.append(f"Tier1: '{term}'")

        for pat in AI_TIER2:
            if re.search(pat, lower):
                findings.append(f"Tier2: pattern '{pat}'")

        for pat in AI_TIER3:
            if re.search(pat, lower, re.MULTILINE):
                findings.append(f"Tier3: pattern '{pat}'")

        if findings:
            self.ai_findings[filename] = findings
        return findings

    def validate_root_files(self) -> None:
        for fn in REQUIRED_ROOT_FILES:
            p = self.base_path / fn
            self.results.append(
                ValidationResult("root_files", f"Root: {fn}", p.exists(), "✓ Present" if p.exists() else "✗ MISSING", "info" if p.exists() else "error")
            )

    def validate_structure(self) -> None:
        for dn in ["theory", "lab", "exercises", "assessments", "resources", "assets", "tests"]:
            p = self.base_path / dn
            self.results.append(
                ValidationResult("structure", f"Directory: {dn}/", p.is_dir(), "✓ Present" if p.is_dir() else "✗ MISSING", "info" if p.is_dir() else "error")
            )

    def validate_word_counts(self) -> None:
        for fn, minw in MIN_WORD_COUNTS.items():
            for p in self.base_path.rglob(fn):
                txt = p.read_text(encoding="utf-8", errors="ignore")
                wc = len(txt.split())
                ok = wc >= minw
                self.results.append(ValidationResult("word_count", f"Words: {p.name}", ok, f"{wc}/{minw} words", "info" if ok else "warning"))

    def validate_ai_fingerprints(self) -> None:
        files = list(self.base_path.rglob("*.md")) + list(self.base_path.rglob("*.py")) + list(self.base_path.rglob("*.html"))
        total = 0
        for p in files:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            f = self.scan_ai(txt, p.name)
            total += len(f)
            if f:
                self.results.append(ValidationResult("ai_fingerprints", f"AI: {p.name}", False, f"⚠ {len(f)} fingerprints found", "warning" if not self.strict else "error"))
        if total == 0:
            self.results.append(ValidationResult("ai_fingerprints", "AI Scan", True, "✓ ZERO fingerprints detected", "info"))

    def validate_python_syntax(self) -> None:
        for p in self.base_path.rglob("*.py"):
            r = subprocess.run([sys.executable, "-m", "py_compile", str(p)], capture_output=True, text=True)
            ok = r.returncode == 0
            self.results.append(ValidationResult("python", f"Syntax: {p.name}", ok, "✓ Valid" if ok else "✗ Syntax error", "info" if ok else "error"))

    def validate_html_naming(self) -> None:
        for p in self.base_path.rglob("*.html"):
            ok = p.name.startswith(f"{self.unit}UNIT_")
            self.results.append(ValidationResult("html", f"Naming: {p.name}", ok, "✓ Has UNIT prefix" if ok else f"⚠ Missing {self.unit}UNIT_ prefix", "info" if ok else "warning"))

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
        print("═" * 70)
        print(f"VALIDATION REPORT: {self.unit}UNIT")
        print("═" * 70)
        for r in self.results:
            status = "✓" if r.passed else ("⚠" if r.severity == "warning" else "✗")
            print(f"{status} [{r.category}] {r.check} — {r.message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("unit")
    parser.add_argument("-s", "--strict", action="store_true")
    parser.add_argument("-p", "--path", default=".")
    args = parser.parse_args()

    if args.unit not in VALID_UNITS:
        raise SystemExit(1)

    base = Path(args.path)
    unit_dirs = list(base.glob(f"{args.unit}UNIT*"))
    unit_path = unit_dirs[0] if unit_dirs else base

    v = UnitValidator(args.unit, unit_path, args.strict)
    ok = v.run_all()
    v.print_report()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
