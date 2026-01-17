#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Medium Exercise 2 — Small Multiples Time Series
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates Tufte's small multiples principle using seaborn
FacetGrid for comparative time series visualisation.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from seaborn import FacetGrid

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Wong colourblind-safe palette
WONG_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_time_series_data(
    n_series: int = 6,
    n_points: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate multiple time series with different patterns.

    Parameters
    ----------
    n_series : int
        Number of distinct time series to generate.
    n_points : int
        Number of time points per series.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns: time, value, series.
    """
    rng = np.random.default_rng(seed)

    series_names = [f"Sensor {chr(65 + i)}" for i in range(n_series)]
    time = np.arange(n_points)

    records = []
    for i, name in enumerate(series_names):
        # Different patterns for each series
        base_freq = 0.05 + i * 0.02
        amplitude = 1.0 + i * 0.3
        phase = i * np.pi / 6
        trend = (i - n_series / 2) * 0.01

        # Generate signal: sinusoidal + trend + noise
        signal = (
            amplitude * np.sin(2 * np.pi * base_freq * time + phase)
            + trend * time
            + rng.normal(0, 0.3, n_points)
        )

        for t, v in zip(time, signal):
            records.append({"time": t, "value": v, "series": name})

    return pd.DataFrame(records)


def create_small_multiples(
    df: pd.DataFrame,
    output_path: Path | None = None,
) -> FacetGrid:
    """
    Create small multiples visualisation using FacetGrid.

    Tufte's small multiples principles:
    - Same scale across all panels for comparison
    - Minimal redundancy (shared axes)
    - High data density
    - Direct labelling

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame with time, value, series columns.
    output_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    FacetGrid
        Seaborn FacetGrid object.
    """
    # Set up the style
    sns.set_style("whitegrid")

    # Create FacetGrid with 2 rows x 3 columns
    g = sns.FacetGrid(
        df,
        col="series",
        col_wrap=3,
        height=2.5,
        aspect=1.5,
        sharex=True,
        sharey=True,
    )

    # Map line plots to each facet
    g.map_dataframe(
        sns.lineplot,
        x="time",
        y="value",
        color=WONG_COLOURS[0],
        linewidth=1.5,
    )

    # Add rolling mean to each panel
    def add_rolling_mean(data: pd.DataFrame, **kwargs) -> None:
        """Add rolling mean line to each facet."""
        rolling = data.sort_values("time")["value"].rolling(window=10).mean()
        plt.plot(
            data.sort_values("time")["time"],
            rolling,
            color=WONG_COLOURS[1],
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )

    g.map_dataframe(add_rolling_mean)

    # Configure titles and labels
    g.set_titles(col_template="{col_name}", fontsize=11, fontweight="bold")
    g.set_axis_labels("Time (samples)", "Sensor Value", fontsize=10)

    # Add overall title
    g.figure.suptitle(
        "Small Multiples: Sensor Time Series Comparison",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Add legend to first panel
    handles = [
        plt.Line2D([0], [0], color=WONG_COLOURS[0], linewidth=1.5, label="Raw signal"),
        plt.Line2D(
            [0], [0], color=WONG_COLOURS[1], linewidth=2, linestyle="--",
            label="10-pt rolling mean",
        ),
    ]
    g.axes[0].legend(handles=handles, loc="upper right", fontsize=8)

    # Adjust spacing
    g.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        g.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved to %s", output_path)

    return g


def test_solution() -> None:
    """Test the solution implementation."""
    logger.info("Testing small multiples visualisation...")

    # Test 1: Data generation
    df = generate_time_series_data(n_series=6, n_points=50, seed=42)
    assert len(df) == 6 * 50
    assert set(df.columns) == {"time", "value", "series"}
    assert df["series"].nunique() == 6
    logger.info("✓ Data generation test passed")

    # Test 2: FacetGrid creation
    g = create_small_multiples(df)
    assert g is not None
    assert len(g.axes) == 6  # 6 panels
    plt.close("all")
    logger.info("✓ FacetGrid creation test passed")

    # Test 3: Shared axes (Tufte principle)
    g = create_small_multiples(df)
    # All panels should have same y-limits
    y_limits = [ax.get_ylim() for ax in g.axes.flat]
    assert all(yl == y_limits[0] for yl in y_limits)
    plt.close("all")
    logger.info("✓ Shared axes test passed")

    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point demonstrating the complete solution."""
    import argparse

    parser = argparse.ArgumentParser(description="Small multiples time series")
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
    df = generate_time_series_data(n_series=6, n_points=100, seed=42)
    logger.info("Generated %d data points across %d series", len(df), 6)

    # Create visualisation
    g = create_small_multiples(df, output_path=args.output)

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
