#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Hard Exercise 2 — Publication-Ready Multi-Panel Figure
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates creating publication-ready multi-panel figures
with journal-specific styles (Nature, Science, IEEE), panel labels and
multi-format export with automatic caption generation.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# JOURNAL STYLE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


class JournalStyle(Enum):
    """Supported journal styles."""

    NATURE = "nature"
    SCIENCE = "science"
    IEEE = "ieee"
    PLOS = "plos"
    DEFAULT = "default"


@dataclass
class JournalConfig:
    """Configuration for journal-specific figure styles."""

    name: str
    column_width_mm: float
    double_column_width_mm: float
    font_family: str
    font_size: int
    dpi: int
    line_width: float = 1.0
    marker_size: float = 4.0

    @property
    def column_width_inches(self) -> float:
        """Convert column width to inches."""
        return self.column_width_mm / 25.4

    @property
    def double_column_width_inches(self) -> float:
        """Convert double column width to inches."""
        return self.double_column_width_mm / 25.4


JOURNAL_CONFIGS: dict[JournalStyle, JournalConfig] = {
    JournalStyle.NATURE: JournalConfig(
        name="Nature",
        column_width_mm=89,
        double_column_width_mm=183,
        font_family="Arial",
        font_size=7,
        dpi=300,
        line_width=0.75,
    ),
    JournalStyle.SCIENCE: JournalConfig(
        name="Science",
        column_width_mm=85,
        double_column_width_mm=174,
        font_family="Helvetica",
        font_size=7,
        dpi=300,
        line_width=0.75,
    ),
    JournalStyle.IEEE: JournalConfig(
        name="IEEE",
        column_width_mm=88,
        double_column_width_mm=181,
        font_family="Times New Roman",
        font_size=8,
        dpi=600,
        line_width=1.0,
    ),
    JournalStyle.PLOS: JournalConfig(
        name="PLOS",
        column_width_mm=84,
        double_column_width_mm=173,
        font_family="Arial",
        font_size=8,
        dpi=300,
        line_width=1.0,
    ),
    JournalStyle.DEFAULT: JournalConfig(
        name="Default",
        column_width_mm=100,
        double_column_width_mm=180,
        font_family="sans-serif",
        font_size=10,
        dpi=150,
        line_width=1.5,
    ),
}


# Wong colourblind-safe palette
WONG_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9", "#D55E00"]


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLICATION FIGURE CLASS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class PanelSpec:
    """Specification for a single panel."""

    row: int
    col: int
    label: str
    title: str = ""
    plot_func: Callable[[Axes], None] | None = None
    rowspan: int = 1
    colspan: int = 1


@dataclass
class PublicationFigure:
    """
    Class for creating publication-ready multi-panel figures.

    Attributes
    ----------
    journal : JournalStyle
        Target journal style.
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    double_column : bool
        Whether to use double column width.
    panels : list of PanelSpec
        Panel specifications.
    figure_number : int
        Figure number for caption.
    caption_prefix : str
        Prefix for figure caption.
    """

    journal: JournalStyle = JournalStyle.DEFAULT
    n_rows: int = 2
    n_cols: int = 2
    double_column: bool = False
    panels: list[PanelSpec] = field(default_factory=list)
    figure_number: int = 1
    caption_prefix: str = ""
    _fig: Figure | None = field(default=None, repr=False)
    _axes: dict[str, Axes] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Initialise configuration."""
        self.config = JOURNAL_CONFIGS[self.journal]

    def add_panel(
        self,
        row: int,
        col: int,
        label: str,
        title: str = "",
        plot_func: Callable[[Axes], None] | None = None,
        rowspan: int = 1,
        colspan: int = 1,
    ) -> PublicationFigure:
        """Add a panel specification."""
        panel = PanelSpec(
            row=row,
            col=col,
            label=label,
            title=title,
            plot_func=plot_func,
            rowspan=rowspan,
            colspan=colspan,
        )
        self.panels.append(panel)
        return self

    def _apply_journal_style(self) -> None:
        """Apply journal-specific matplotlib rcParams."""
        plt.rcParams.update({
            "font.family": self.config.font_family,
            "font.size": self.config.font_size,
            "axes.linewidth": self.config.line_width,
            "lines.linewidth": self.config.line_width,
            "lines.markersize": self.config.marker_size,
            "xtick.major.width": self.config.line_width * 0.8,
            "ytick.major.width": self.config.line_width * 0.8,
            "axes.labelsize": self.config.font_size,
            "axes.titlesize": self.config.font_size + 1,
            "legend.fontsize": self.config.font_size - 1,
            "figure.dpi": self.config.dpi,
            "savefig.dpi": self.config.dpi,
        })

    def create(self) -> tuple[Figure, dict[str, Axes]]:
        """
        Create the figure with all panels.

        Returns
        -------
        tuple
            (fig, axes_dict) where axes_dict maps panel labels to Axes.
        """
        self._apply_journal_style()

        # Determine figure size
        if self.double_column:
            width = self.config.double_column_width_inches
        else:
            width = self.config.column_width_inches

        # Height based on aspect ratio (golden ratio ≈ 1.618)
        height = width / 1.618 * (self.n_rows / self.n_cols)

        # Create figure and GridSpec
        self._fig = plt.figure(figsize=(width, height), constrained_layout=True)
        gs = GridSpec(self.n_rows, self.n_cols, figure=self._fig)

        # Create panels
        self._axes = {}
        for panel in self.panels:
            ax = self._fig.add_subplot(
                gs[
                    panel.row : panel.row + panel.rowspan,
                    panel.col : panel.col + panel.colspan,
                ]
            )
            self._axes[panel.label] = ax

            # Add panel label
            self._add_panel_label(ax, panel.label)

            # Set title if specified
            if panel.title:
                ax.set_title(panel.title, fontweight="bold")

            # Execute plot function if specified
            if panel.plot_func is not None:
                panel.plot_func(ax)

        return self._fig, self._axes

    def _add_panel_label(
        self,
        ax: Axes,
        label: str,
        loc: str = "upper left",
    ) -> None:
        """Add panel label (a, b, c, etc.) to axes."""
        # Position based on location
        if loc == "upper left":
            x, y = -0.15, 1.05
        elif loc == "upper right":
            x, y = 1.05, 1.05
        else:
            x, y = -0.15, 1.05

        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=self.config.font_size + 2,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    def save(
        self,
        output_dir: Path,
        basename: str = "figure",
        formats: list[str] | None = None,
    ) -> list[Path]:
        """
        Save figure in multiple formats.

        Parameters
        ----------
        output_dir : Path
            Directory to save files.
        basename : str
            Base filename (without extension).
        formats : list of str
            Formats to save (default: ['png', 'pdf', 'svg']).

        Returns
        -------
        list of Path
            Paths to saved files.
        """
        if self._fig is None:
            raise RuntimeError("Must call create() before save()")

        if formats is None:
            formats = ["png", "pdf", "svg"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for fmt in formats:
            path = output_dir / f"{basename}.{fmt}"
            self._fig.savefig(
                path,
                format=fmt,
                dpi=self.config.dpi,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            saved_paths.append(path)
            logger.info("Saved %s", path)

        return saved_paths

    def generate_caption(self, panel_descriptions: dict[str, str]) -> str:
        """
        Generate figure caption with panel descriptions.

        Parameters
        ----------
        panel_descriptions : dict
            Mapping of panel labels to descriptions.

        Returns
        -------
        str
            Formatted figure caption.
        """
        caption_parts = []

        if self.caption_prefix:
            caption_parts.append(self.caption_prefix)

        for panel in self.panels:
            if panel.label in panel_descriptions:
                caption_parts.append(
                    f"({panel.label}) {panel_descriptions[panel.label]}"
                )

        return f"Figure {self.figure_number}. " + " ".join(caption_parts)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_time_series(ax: Axes) -> None:
    """Plot sample time series data."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 10, 100)
    y = np.sin(t) + rng.normal(0, 0.2, 100)

    ax.plot(t, y, color=WONG_PALETTE[0], label="Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", framealpha=0.9)


def plot_bar_comparison(ax: Axes) -> None:
    """Plot sample bar chart."""
    categories = ["A", "B", "C", "D"]
    values = [23, 45, 32, 58]
    errors = [3, 5, 4, 6]

    x = np.arange(len(categories))
    ax.bar(x, values, yerr=errors, capsize=3, color=WONG_PALETTE[0], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel("Group")
    ax.set_ylabel("Response")


def plot_scatter_correlation(ax: Axes) -> None:
    """Plot sample scatter with correlation."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 50)
    y = 2 * x + rng.normal(0, 2, 50)

    ax.scatter(x, y, c=WONG_PALETTE[0], alpha=0.7, s=20, edgecolors="white", linewidths=0.5)

    # Add regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color=WONG_PALETTE[1], linestyle="--", label=f"r² = 0.85")

    ax.set_xlabel("X variable")
    ax.set_ylabel("Y variable")
    ax.legend(loc="lower right", framealpha=0.9)


def plot_histogram(ax: Axes) -> None:
    """Plot sample histogram."""
    rng = np.random.default_rng(42)
    data = rng.normal(50, 10, 200)

    ax.hist(data, bins=20, color=WONG_PALETTE[0], alpha=0.8, edgecolor="white")
    ax.axvline(np.mean(data), color=WONG_PALETTE[1], linestyle="--", label="Mean")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", framealpha=0.9)


def test_solution() -> None:
    """Test the solution implementation."""
    logger.info("Testing publication figure generator...")

    # Test 1: Journal config
    config = JOURNAL_CONFIGS[JournalStyle.NATURE]
    assert abs(config.column_width_inches - 89 / 25.4) < 0.01
    logger.info("✓ Journal config test passed")

    # Test 2: Figure creation
    fig_gen = PublicationFigure(
        journal=JournalStyle.NATURE,
        n_rows=2,
        n_cols=2,
        double_column=True,
    )
    fig_gen.add_panel(0, 0, "a", "Time Series", plot_time_series)
    fig_gen.add_panel(0, 1, "b", "Comparison", plot_bar_comparison)
    fig_gen.add_panel(1, 0, "c", "Correlation", plot_scatter_correlation)
    fig_gen.add_panel(1, 1, "d", "Distribution", plot_histogram)

    fig, axes = fig_gen.create()
    assert fig is not None
    assert len(axes) == 4
    assert all(label in axes for label in ["a", "b", "c", "d"])
    plt.close(fig)
    logger.info("✓ Figure creation test passed")

    # Test 3: Caption generation
    caption = fig_gen.generate_caption({
        "a": "Time series showing signal variation.",
        "b": "Bar chart comparing groups.",
        "c": "Scatter plot with correlation.",
        "d": "Histogram of distribution.",
    })
    assert "Figure 1." in caption
    assert "(a)" in caption and "(d)" in caption
    logger.info("✓ Caption generation test passed")

    # Test 4: File saving
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, _ = fig_gen.create()
        paths = fig_gen.save(Path(tmpdir), "test_fig", formats=["png"])
        assert len(paths) == 1
        assert paths[0].exists()
        plt.close(fig)
    logger.info("✓ File saving test passed")

    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point demonstrating the complete solution."""
    import argparse

    parser = argparse.ArgumentParser(description="Publication figure generator")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--output", type=Path, default=Path("."), help="Output directory")
    parser.add_argument(
        "--journal",
        type=str,
        choices=["nature", "science", "ieee", "plos"],
        default="nature",
        help="Journal style",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        test_solution()
        return

    # Create figure
    journal = JournalStyle(args.journal)
    fig_gen = PublicationFigure(
        journal=journal,
        n_rows=2,
        n_cols=2,
        double_column=True,
        figure_number=1,
        caption_prefix="Multi-panel figure showing experimental results.",
    )

    # Add panels
    fig_gen.add_panel(0, 0, "a", "Time Series", plot_time_series)
    fig_gen.add_panel(0, 1, "b", "Comparison", plot_bar_comparison)
    fig_gen.add_panel(1, 0, "c", "Correlation", plot_scatter_correlation)
    fig_gen.add_panel(1, 1, "d", "Distribution", plot_histogram)

    # Create and save
    fig, axes = fig_gen.create()
    paths = fig_gen.save(args.output, f"figure_{args.journal}")

    # Generate caption
    caption = fig_gen.generate_caption({
        "a": "Time series showing signal variation over 10 seconds.",
        "b": "Bar chart comparing response across treatment groups.",
        "c": "Scatter plot demonstrating positive correlation (r² = 0.85).",
        "d": "Histogram showing normal distribution of values.",
    })
    logger.info("Caption: %s", caption)

    # Save caption to file
    caption_path = args.output / f"figure_{args.journal}_caption.txt"
    caption_path.write_text(caption)
    logger.info("Caption saved to %s", caption_path)


if __name__ == "__main__":
    main()
