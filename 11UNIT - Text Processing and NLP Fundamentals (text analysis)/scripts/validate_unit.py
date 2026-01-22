#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

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

AI_TIER1 = [
    "delve", "dive into", "crucial", "pivotal", "straightforward",
    "leverage", "robust", "seamless", "cutting-edge", "best practices",
    "game-changer", "empower", "harness", "landscape", "paradigm",
    "synergy", "holistic", "ecosystem", "utilize", "facilitate",
    "bolster", "foster", "underscore", "realm", "myriad", "plethora",
    "cornerstone", "encompasses", "intricacies",
]

def count_words(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    return len(re.findall(r"\b\w+\b", text))

def scan_ai_terms(text: str) -> list[str]:
    low = text.lower()
    return [t for t in AI_TIER1 if t in low]

def main() -> int:
    p = argparse.ArgumentParser(description="Validate 11UNIT structure and content")
    p.add_argument("unit", help="UNIT number, expected 11")
    p.add_argument("-p", "--path", default=".", help="Base path")
    p.add_argument("--strict", action="store_true", help="Treat AI term hits as errors")
    args = p.parse_args()

    if args.unit != "11":
        print("Only UNIT 11 is supported by this validator instance.")
        return 2

    base_path = Path(args.path)
    unit_path = base_path / "11UNIT" if (base_path / "11UNIT").is_dir() else base_path

    required_dirs = ["theory", "lab", "exercises", "assessments", "resources", "assets", "tests", "scripts"]
    required_files = ["README.md", "Makefile", "requirements.txt"]

    ok = True
    for f in required_files:
        if not (unit_path / f).exists():
            print(f"[ERROR] Missing root file: {f}")
            ok = False

    for d in required_dirs:
        if not (unit_path / d).is_dir():
            print(f"[ERROR] Missing directory: {d}/")
            ok = False

    for filename, min_words in MIN_WORD_COUNTS.items():
        matches = list(unit_path.rglob(filename))
        if not matches:
            print(f"[ERROR] Missing required document: {filename}")
            ok = False
            continue
        for fp in matches:
            wc = count_words(fp)
            if wc < min_words:
                print(f"[WARN] {fp.as_posix()} word count {wc} < {min_words}")
                ok = False

    text_files = list(unit_path.rglob("*.md")) + list(unit_path.rglob("*.py")) + list(unit_path.rglob("*.html"))
    ai_hits = 0
    for fp in text_files:
        hits = scan_ai_terms(fp.read_text(encoding="utf-8"))
        if hits:
            ai_hits += len(hits)
            sev = "ERROR" if args.strict else "WARN"
            print(f"[{sev}] AI-term hits in {fp.name}: {sorted(set(hits))}")

    if ai_hits == 0:
        print("[OK] AI term scan: zero hits")
    else:
        ok = ok and (not args.strict)

    print("[OK]" if ok else "[FAIL]")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
