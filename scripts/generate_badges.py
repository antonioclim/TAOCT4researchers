#!/usr/bin/env python3
"""
TAOCT4researchers — Badge Generation Script
The Art of Computational Thinking for Researchers
Version 5.0.0

This script generates SVG badges for each instructional unit, displaying
unit number, difficulty level and estimated duration.

Usage:
    python scripts/generate_badges.py [OPTIONS]

Options:
    --unit N        Generate badge for specific unit only (01-14)
    --output DIR    Output directory (default: assets/badges)
    --help          Display help message
"""

from __future__ import annotations

import argparse
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent

UNIT_CONFIG = {
    "01": {
        "name": "Epistemology",
        "subtitle": "Foundations",
        "difficulty": 2,
        "hours": 13,
        "colour": "#4a90d9",
    },
    "02": {
        "name": "Abstraction",
        "subtitle": "Patterns",
        "difficulty": 3,
        "hours": 8,
        "colour": "#50b87d",
    },
    "03": {
        "name": "Complexity",
        "subtitle": "Performance",
        "difficulty": 3,
        "hours": 4,
        "colour": "#e67e22",
    },
    "04": {
        "name": "Data Structures",
        "subtitle": "Design",
        "difficulty": 4,
        "hours": 10,
        "colour": "#9b59b6",
    },
    "05": {
        "name": "Scientific Computing",
        "subtitle": "Simulations",
        "difficulty": 4,
        "hours": 7,
        "colour": "#3498db",
    },
    "06": {
        "name": "Visualisation",
        "subtitle": "Communication",
        "difficulty": 4,
        "hours": 14,
        "colour": "#1abc9c",
    },
    "07": {
        "name": "Reproducibility",
        "subtitle": "Integration",
        "difficulty": 5,
        "hours": 11,
        "colour": "#e74c3c",
    },
    "08": {
        "name": "Recursion & DP",
        "subtitle": "Algorithms",
        "difficulty": 4,
        "hours": 10,
        "colour": "#f39c12",
    },
    "09": {
        "name": "Exceptions",
        "subtitle": "Robustness",
        "difficulty": 3,
        "hours": 8,
        "colour": "#27ae60",
    },
    "10": {
        "name": "Persistence",
        "subtitle": "Storage",
        "difficulty": 4,
        "hours": 10,
        "colour": "#8e44ad",
    },
    "11": {
        "name": "Text & NLP",
        "subtitle": "Analysis",
        "difficulty": 4,
        "hours": 10,
        "colour": "#16a085",
    },
    "12": {
        "name": "Web APIs",
        "subtitle": "Integration",
        "difficulty": 4,
        "hours": 12,
        "colour": "#2980b9",
    },
    "13": {
        "name": "Machine Learning",
        "subtitle": "ML Basics",
        "difficulty": 5,
        "hours": 14,
        "colour": "#c0392b",
    },
    "14": {
        "name": "Parallel Computing",
        "subtitle": "Scalability",
        "difficulty": 5,
        "hours": 12,
        "colour": "#d35400",
    },
}


# -----------------------------------------------------------------------------
# SVG Templates
# -----------------------------------------------------------------------------

BADGE_TEMPLATE = '''<svg xmlns="http://www.w3.org/2000/svg" width="200" height="80" viewBox="0 0 200 80">
  <defs>
    <linearGradient id="bg_{unit_num}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{colour};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{colour_dark};stop-opacity:1" />
    </linearGradient>
    <filter id="shadow_{unit_num}" x="-10%" y="-10%" width="120%" height="120%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.3"/>
    </filter>
  </defs>
  
  <!-- Background -->
  <rect x="2" y="2" width="196" height="76" rx="8" ry="8" 
        fill="url(#bg_{unit_num})" filter="url(#shadow_{unit_num})"/>
  
  <!-- Unit number circle -->
  <circle cx="35" cy="40" r="22" fill="#ffffff" fill-opacity="0.2"/>
  <text x="35" y="47" font-family="Arial, sans-serif" font-size="18" 
        font-weight="bold" fill="#ffffff" text-anchor="middle">{unit_num}</text>
  
  <!-- Unit name -->
  <text x="70" y="30" font-family="Arial, sans-serif" font-size="14" 
        font-weight="bold" fill="#ffffff">{name}</text>
  
  <!-- Subtitle -->
  <text x="70" y="46" font-family="Arial, sans-serif" font-size="10" 
        fill="#ffffff" fill-opacity="0.8">{subtitle}</text>
  
  <!-- Difficulty stars -->
  <text x="70" y="62" font-family="Arial, sans-serif" font-size="12" 
        fill="#ffffff">{stars}</text>
  
  <!-- Duration -->
  <text x="140" y="62" font-family="Arial, sans-serif" font-size="10" 
        fill="#ffffff" fill-opacity="0.9">{hours}h</text>
</svg>'''

SUMMARY_BADGE_TEMPLATE = '''<svg xmlns="http://www.w3.org/2000/svg" width="280" height="60" viewBox="0 0 280 60">
  <defs>
    <linearGradient id="bg_summary" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#1a1a2e;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#16213e;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- Background -->
  <rect x="0" y="0" width="280" height="60" rx="6" ry="6" fill="url(#bg_summary)"/>
  
  <!-- Divider -->
  <rect x="140" y="10" width="1" height="40" fill="#58a6ff" fill-opacity="0.3"/>
  
  <!-- Units count -->
  <text x="70" y="28" font-family="Arial, sans-serif" font-size="20" 
        font-weight="bold" fill="#ffffff" text-anchor="middle">14</text>
  <text x="70" y="45" font-family="Arial, sans-serif" font-size="11" 
        fill="#58a6ff" text-anchor="middle">UNITS</text>
  
  <!-- Hours -->
  <text x="210" y="28" font-family="Arial, sans-serif" font-size="20" 
        font-weight="bold" fill="#ffffff" text-anchor="middle">143+</text>
  <text x="210" y="45" font-family="Arial, sans-serif" font-size="11" 
        fill="#58a6ff" text-anchor="middle">HOURS</text>
</svg>'''


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def darken_colour(hex_colour: str, factor: float = 0.8) -> str:
    """Darken a hex colour by a factor."""
    hex_colour = hex_colour.lstrip("#")
    r = int(int(hex_colour[0:2], 16) * factor)
    g = int(int(hex_colour[2:4], 16) * factor)
    b = int(int(hex_colour[4:6], 16) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_stars(difficulty: int) -> str:
    """Generate star string for difficulty level."""
    filled = "★" * difficulty
    empty = "☆" * (5 - difficulty)
    return filled + empty


def generate_unit_badge(unit_num: str, config: dict, output_dir: Path) -> Path:
    """Generate badge SVG for a single unit."""
    stars = generate_stars(config["difficulty"])
    colour_dark = darken_colour(config["colour"])

    svg_content = BADGE_TEMPLATE.format(
        unit_num=unit_num,
        name=config["name"],
        subtitle=config["subtitle"],
        stars=stars,
        hours=config["hours"],
        colour=config["colour"],
        colour_dark=colour_dark,
    )

    output_path = output_dir / f"{unit_num}UNIT_badge.svg"
    output_path.write_text(svg_content, encoding="utf-8")

    return output_path


def generate_summary_badge(output_dir: Path) -> Path:
    """Generate summary badge showing total units and hours."""
    output_path = output_dir / "summary_badge.svg"
    output_path.write_text(SUMMARY_BADGE_TEMPLATE, encoding="utf-8")
    return output_path


def generate_all_badges(output_dir: Path) -> list[Path]:
    """Generate badges for all units."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for unit_num, config in UNIT_CONFIG.items():
        path = generate_unit_badge(unit_num, config, output_dir)
        generated.append(path)
        print(f"  Generated: {path.name}")

    # Generate summary badge
    summary_path = generate_summary_badge(output_dir)
    generated.append(summary_path)
    print(f"  Generated: {summary_path.name}")

    return generated


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate SVG badges for TAOCT4researchers units"
    )
    parser.add_argument(
        "--unit",
        type=str,
        help="Generate badge for specific unit only (01-14)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="assets/badges",
        help="Output directory (default: assets/badges)",
    )

    args = parser.parse_args()
    output_dir = REPO_ROOT / args.output

    print("\nTAOCT4researchers — Badge Generation")
    print("=" * 40)

    if args.unit:
        unit_num = args.unit.zfill(2)
        if unit_num not in UNIT_CONFIG:
            print(f"Error: Unknown unit {unit_num}")
            return 1

        output_dir.mkdir(parents=True, exist_ok=True)
        path = generate_unit_badge(unit_num, UNIT_CONFIG[unit_num], output_dir)
        print(f"  Generated: {path}")
    else:
        paths = generate_all_badges(output_dir)
        print(f"\nGenerated {len(paths)} badges in {output_dir}")

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    exit(main())
