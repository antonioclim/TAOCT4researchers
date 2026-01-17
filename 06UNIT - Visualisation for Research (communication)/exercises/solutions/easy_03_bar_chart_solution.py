#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Easy Exercise 3 — Bar Chart with Error Bars
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates bar chart creation with error bars using Tufte's
principles of maximising data-ink ratio and colourblind-friendly styling.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Wong colourblind-safe palette
WONG_PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CategoryData:
    """Container for categorical data with error measurements."""

    categories: list[str]
    values: np.ndarray
    errors: np.ndarray

    def __post_init__(self) -> None:
        """Validate data consistency."""
        n = len(self.categories)
        if len(self.values) != n or len(self.errors) != n:
            raise ValueError("All arrays must have the same length")


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_sample_data(seed: int = 42) -> CategoryData:
    """
    Generate sample experimental data with categories and errors.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    CategoryData
        Container with categories, mean values and standard errors.
    """
    rng = np.random.default_rng(seed)

    categories = ["Control", "Treatment A", "Treatment B", "Treatment C", "Treatment D"]
    # Simulate experimental means
    values = np.array([10.2, 15.8, 12.4, 18.1, 14.3])
    # Simulate standard errors
    errors = rng.uniform(0.8, 2.0, len(categories))

    return CategoryData(categories=categories, values=values, errors=errors)


def create_bar_chart_with_errors(
    data: CategoryData,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    Create bar chart with error bars following Tufte's principles.

    Tufte's principles applied:
    1. Maximise data-ink ratio (no chartjunk)
    2. Remove non-data ink (minimal gridlines)
    3. Avoid redundant data encoding
    4. Use clear, direct labels

    Parameters
    ----------
    data : CategoryData
        Container with categories, values and errors.
    output_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    # Create x positions
    x = np.arange(len(data.categories))

    # Use colourblind-friendly colour
    bar_colour = WONG_PALETTE["blue"]
    error_colour = WONG_PALETTE["vermillion"]

    # Create bars
    bars = ax.bar(
        x,
        data.values,
        color=bar_colour,
        edgecolor="white",
        linewidth=1,
        width=0.6,
        alpha=0.85,
        label="Mean value",
    )

    # Add error bars
    ax.errorbar(
        x,
        data.values,
        yerr=data.errors,
        fmt="none",
        ecolor=error_colour,
        elinewidth=2,
        capsize=5,
        capthick=2,
        label="Standard error",
    )

    # Add value labels on bars
    for bar, val, err in zip(bars, data.values, data.errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + err + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Configure axes (Tufte: direct labels, minimal decoration)
    ax.set_xticks(x)
    ax.set_xticklabels(data.categories, fontsize=10)
    ax.set_ylabel("Response Value (units)", fontsize=11)
    ax.set_title(
        "Experimental Results by Treatment Group",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Tufte: Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Subtle y-axis grid only (horizontal reference lines)
    ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Set y-axis to start from 0 (no truncation)
    ax.set_ylim(0, max(data.values + data.errors) * 1.15)

    # Legend
    ax.legend(loc="upper left", framealpha=0.9, fontsize=10)

    # Tight layout
    fig.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved to %s", output_path)

    return fig, ax


def test_solution() -> None:
    """Test the solution implementation."""
    logger.info("Testing bar chart with error bars...")

    # Test 1: Data generation
    data = generate_sample_data(seed=42)
    assert len(data.categories) == 5
    assert len(data.values) == 5
    assert len(data.errors) == 5
    logger.info("✓ Data generation test passed")

    # Test 2: Data validation
    try:
        invalid_data = CategoryData(
            categories=["A", "B"],
            values=np.array([1, 2, 3]),  # Wrong length
            errors=np.array([0.1, 0.2]),
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("✓ Data validation test passed")

    # Test 3: Plot creation
    fig, ax = create_bar_chart_with_errors(data)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
    logger.info("✓ Plot creation test passed")

    # Test 4: Spines removed (Tufte principle)
    fig, ax = create_bar_chart_with_errors(data)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    plt.close(fig)
    logger.info("✓ Tufte principles test passed")

    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point demonstrating the complete solution."""
    import argparse

    parser = argparse.ArgumentParser(description="Bar chart with error bars")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        test_solution()
        return

    # Generate data
    data = generate_sample_data(seed=42)

    # Create plot
    fig, ax = create_bar_chart_with_errors(data, output_path=args.output)

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
