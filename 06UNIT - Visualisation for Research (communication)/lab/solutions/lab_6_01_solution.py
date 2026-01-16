#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Lab 1: Static Visualisation Toolkit — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

This solution file demonstrates complete implementations of all exercises
from lab_6_01_static_plots.py with detailed explanations.

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
from typing import Any

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: COLOUR PALETTES (COLOURBLIND-FRIENDLY)
# ═══════════════════════════════════════════════════════════════════════════════

# Wong palette — optimised for colourblind accessibility
WONG_PALETTE = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'pink': '#CC79A7',
    'yellow': '#F0E442',
    'light_blue': '#56B4E9',
    'vermillion': '#D55E00',
    'black': '#000000',
}

# As a list for sequential use
WONG_COLOURS = list(WONG_PALETTE.values())


def get_colourblind_palette(n: int = 8) -> list[str]:
    """Get colourblind-friendly colour palette.
    
    Args:
        n: Number of colours needed (max 8)
        
    Returns:
        List of hex colour codes
    """
    if n > 8:
        logger.warning(f"Requested {n} colours, but Wong palette has 8. Using cycling.")
        return [WONG_COLOURS[i % 8] for i in range(n)]
    return WONG_COLOURS[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: JOURNAL STYLE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JournalStyle:
    """Configuration for journal-specific figure requirements.
    
    Attributes:
        name: Journal identifier
        column_width_mm: Single column width in millimetres
        double_column_mm: Double column width in millimetres
        max_height_mm: Maximum figure height in millimetres
        font_family: Required font family
        font_size_pt: Base font size in points
        dpi: Required resolution (dots per inch)
        formats: Accepted file formats
    """
    name: str
    column_width_mm: float
    double_column_mm: float
    max_height_mm: float
    font_family: str
    font_size_pt: int
    dpi: int
    formats: tuple[str, ...]


# Pre-configured journal styles
JOURNAL_STYLES = {
    'nature': JournalStyle(
        name='Nature',
        column_width_mm=89.0,
        double_column_mm=183.0,
        max_height_mm=247.0,
        font_family='Arial',
        font_size_pt=7,
        dpi=300,
        formats=('pdf', 'eps', 'tiff')
    ),
    'science': JournalStyle(
        name='Science',
        column_width_mm=85.0,
        double_column_mm=178.0,
        max_height_mm=230.0,
        font_family='Helvetica',
        font_size_pt=7,
        dpi=300,
        formats=('pdf', 'eps')
    ),
    'ieee': JournalStyle(
        name='IEEE',
        column_width_mm=88.0,
        double_column_mm=181.0,
        max_height_mm=240.0,
        font_family='Times New Roman',
        font_size_pt=8,
        dpi=600,
        formats=('pdf', 'eps', 'png')
    ),
    'plos': JournalStyle(
        name='PLOS ONE',
        column_width_mm=140.0,
        double_column_mm=190.0,
        max_height_mm=225.0,
        font_family='Arial',
        font_size_pt=8,
        dpi=300,
        formats=('tiff', 'eps', 'pdf')
    ),
}


def mm_to_inches(mm: float) -> float:
    """Convert millimetres to inches."""
    return mm / 25.4


def apply_journal_style(style: str | JournalStyle) -> dict[str, Any]:
    """Apply journal-specific matplotlib rcParams.
    
    Args:
        style: Journal name string or JournalStyle instance
        
    Returns:
        Dictionary of applied rcParams for reference
    """
    if isinstance(style, str):
        style = JOURNAL_STYLES[style.lower()]
    
    rcparams = {
        'font.family': style.font_family,
        'font.size': style.font_size_pt,
        'axes.labelsize': style.font_size_pt,
        'axes.titlesize': style.font_size_pt + 1,
        'xtick.labelsize': style.font_size_pt - 1,
        'ytick.labelsize': style.font_size_pt - 1,
        'legend.fontsize': style.font_size_pt - 1,
        'figure.dpi': style.dpi,
        'savefig.dpi': style.dpi,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'patch.linewidth': 0.5,
    }
    
    for key, value in rcparams.items():
        plt.rcParams[key] = value
    
    logger.info(f"Applied {style.name} journal style")
    return rcparams


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PUBLICATION-READY FIGURE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class PublicationFigure:
    """Helper class for creating publication-ready figures.
    
    This class encapsulates best practices for scientific figures:
    - Journal-compliant dimensions
    - Colourblind-friendly palettes
    - Proper font sizing
    - Multi-format export
    """
    
    def __init__(
        self,
        journal: str = 'nature',
        columns: int = 1,
        aspect_ratio: float = 0.75,
        palette: str = 'wong'
    ):
        """Initialise publication figure.
        
        Args:
            journal: Target journal ('nature', 'science', 'ieee', 'plos')
            columns: Column span (1 or 2)
            aspect_ratio: Height/width ratio
            palette: Colour palette name
        """
        self.style = JOURNAL_STYLES[journal.lower()]
        
        # Calculate figure size
        width_mm = self.style.column_width_mm if columns == 1 else self.style.double_column_mm
        height_mm = min(width_mm * aspect_ratio, self.style.max_height_mm)
        
        self.figsize = (mm_to_inches(width_mm), mm_to_inches(height_mm))
        
        # Apply style
        apply_journal_style(self.style)
        
        # Set palette
        self.colours = get_colourblind_palette(8)
        
        # Create figure
        self.fig: Figure | None = None
        self.axes: list[Axes] = []
    
    def create(self, nrows: int = 1, ncols: int = 1) -> tuple[Figure, np.ndarray | Axes]:
        """Create the figure with subplots.
        
        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            
        Returns:
            Tuple of (figure, axes)
        """
        self.fig, axes = plt.subplots(
            nrows, ncols,
            figsize=self.figsize,
            constrained_layout=True
        )
        
        if nrows * ncols == 1:
            self.axes = [axes]
        else:
            self.axes = axes.flatten().tolist()
        
        return self.fig, axes
    
    def add_panel_labels(
        self,
        labels: list[str] | None = None,
        fontweight: str = 'bold',
        loc: tuple[float, float] = (-0.15, 1.05)
    ) -> None:
        """Add panel labels (A, B, C...) to subplots.
        
        Args:
            labels: Custom labels or None for auto (A, B, C...)
            fontweight: Label font weight
            loc: Label position in axes coordinates
        """
        if labels is None:
            labels = [chr(65 + i) for i in range(len(self.axes))]
        
        for ax, label in zip(self.axes, labels):
            ax.text(
                loc[0], loc[1],
                label,
                transform=ax.transAxes,
                fontsize=self.style.font_size_pt + 2,
                fontweight=fontweight,
                va='top',
                ha='left'
            )
    
    def apply_tufte_style(self, ax: Axes | None = None) -> None:
        """Apply Tufte's minimalist style principles.
        
        - Remove top and right spines
        - Use minimal tick marks
        - Remove unnecessary gridlines
        
        Args:
            ax: Target axes or None for all axes
        """
        axes_list = [ax] if ax is not None else self.axes
        
        for ax in axes_list:
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Thin remaining spines
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            
            # Outward ticks
            ax.tick_params(direction='out', length=3, width=0.5)
            
            # Remove grid
            ax.grid(False)
    
    def save(
        self,
        path: Path | str,
        formats: tuple[str, ...] | None = None,
        transparent: bool = False
    ) -> list[Path]:
        """Save figure in multiple formats.
        
        Args:
            path: Base path (extension will be replaced)
            formats: Output formats or None for journal defaults
            transparent: Use transparent background
            
        Returns:
            List of saved file paths
        """
        if self.fig is None:
            raise ValueError("No figure created. Call create() first.")
        
        path = Path(path)
        base = path.parent / path.stem
        
        if formats is None:
            formats = self.style.formats
        
        saved = []
        for fmt in formats:
            output = base.with_suffix(f'.{fmt}')
            self.fig.savefig(
                output,
                format=fmt,
                dpi=self.style.dpi,
                transparent=transparent,
                bbox_inches='tight',
                pad_inches=0.02
            )
            saved.append(output)
            logger.info(f"Saved: {output}")
        
        return saved


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMMON PLOT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_line_with_uncertainty(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray | None = None,
    colour: str | None = None,
    label: str | None = None,
    fill_alpha: float = 0.2
) -> None:
    """Plot line with shaded uncertainty region.
    
    Args:
        ax: Target axes
        x: X values
        y: Y values (mean or central estimate)
        yerr: Y error (symmetric) or None
        colour: Line colour
        label: Legend label
        fill_alpha: Opacity of uncertainty region
    """
    colour = colour or WONG_COLOURS[0]
    
    # Plot line
    line, = ax.plot(x, y, color=colour, label=label, linewidth=1.5)
    
    # Add uncertainty region
    if yerr is not None:
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=colour,
            alpha=fill_alpha,
            linewidth=0
        )


def plot_scatter_with_regression(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    colour: str | None = None,
    show_regression: bool = True,
    show_r2: bool = True
) -> dict[str, float]:
    """Plot scatter with linear regression line.
    
    Args:
        ax: Target axes
        x: X values
        y: Y values
        colour: Point colour
        show_regression: Show regression line
        show_r2: Show R² annotation
        
    Returns:
        Dictionary with regression statistics (slope, intercept, r2)
    """
    from scipy import stats
    
    colour = colour or WONG_COLOURS[0]
    
    # Scatter plot
    ax.scatter(x, y, c=colour, s=20, alpha=0.7, edgecolors='white', linewidths=0.5)
    
    stats_dict = {}
    
    if show_regression:
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        stats_dict = {
            'slope': slope,
            'intercept': intercept,
            'r2': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err
        }
        
        # Regression line
        x_line = np.array([x.min(), x.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=WONG_COLOURS[1], linewidth=1.5, linestyle='--')
        
        if show_r2:
            ax.text(
                0.95, 0.05,
                f'$R^2 = {r_value ** 2:.3f}$',
                transform=ax.transAxes,
                fontsize=7,
                ha='right',
                va='bottom'
            )
    
    return stats_dict


def plot_grouped_bar(
    ax: Axes,
    categories: list[str],
    groups: dict[str, list[float]],
    errors: dict[str, list[float]] | None = None,
    colours: list[str] | None = None
) -> None:
    """Plot grouped bar chart.
    
    Args:
        ax: Target axes
        categories: Category labels
        groups: Dictionary mapping group names to values
        errors: Dictionary mapping group names to error bars
        colours: Colours for each group
    """
    n_categories = len(categories)
    n_groups = len(groups)
    
    colours = colours or get_colourblind_palette(n_groups)
    
    # Bar positioning
    bar_width = 0.8 / n_groups
    x = np.arange(n_categories)
    
    for i, (group_name, values) in enumerate(groups.items()):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        error = errors.get(group_name) if errors else None
        
        ax.bar(
            x + offset,
            values,
            bar_width * 0.9,
            label=group_name,
            color=colours[i],
            yerr=error,
            capsize=2,
            error_kw={'linewidth': 0.5}
        )
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=False)


def plot_heatmap(
    ax: Axes,
    data: np.ndarray,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cmap: str = 'RdBu_r',
    annotate: bool = True,
    fmt: str = '.2f'
) -> None:
    """Plot annotated heatmap.
    
    Args:
        ax: Target axes
        data: 2D array of values
        row_labels: Row labels
        col_labels: Column labels
        cmap: Colourmap name
        annotate: Add value annotations
        fmt: Annotation number format
    """
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # Add colourbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.5)
    
    # Labels
    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
    
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
    
    # Annotations
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # Choose text colour based on background
                bg_val = (data[i, j] - data.min()) / (data.max() - data.min())
                text_colour = 'white' if 0.3 < bg_val < 0.7 else 'black'
                
                ax.text(
                    j, i,
                    format(data[i, j], fmt),
                    ha='center',
                    va='center',
                    fontsize=6,
                    color=text_colour
                )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_publication_figure() -> None:
    """Demonstrate publication figure creation."""
    logger.info("Creating publication figure demonstration...")
    
    # Create figure for Nature
    pub_fig = PublicationFigure(journal='nature', columns=2, aspect_ratio=0.5)
    fig, axes = pub_fig.create(nrows=1, ncols=3)
    
    # Generate sample data
    np.random.seed(42)
    
    # Panel A: Line plot with uncertainty
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x / 5)
    yerr = 0.1 + 0.05 * np.random.randn(100)
    
    plot_line_with_uncertainty(axes[0], x, y, np.abs(yerr), WONG_COLOURS[0], 'Signal')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time Series')
    
    # Panel B: Scatter with regression
    x_scatter = np.random.uniform(0, 10, 50)
    y_scatter = 2 * x_scatter + 3 + np.random.normal(0, 2, 50)
    
    plot_scatter_with_regression(axes[1], x_scatter, y_scatter, WONG_COLOURS[1])
    axes[1].set_xlabel('X Variable')
    axes[1].set_ylabel('Y Variable')
    axes[1].set_title('Correlation')
    
    # Panel C: Bar chart
    categories = ['A', 'B', 'C', 'D']
    groups = {
        'Control': [3.2, 4.1, 2.8, 3.9],
        'Treatment': [4.5, 5.2, 4.1, 5.0]
    }
    errors = {
        'Control': [0.3, 0.4, 0.2, 0.3],
        'Treatment': [0.4, 0.3, 0.3, 0.4]
    }
    
    plot_grouped_bar(axes[2], categories, groups, errors)
    axes[2].set_xlabel('Category')
    axes[2].set_ylabel('Value')
    axes[2].set_title('Comparison')
    
    # Apply styling
    for ax in axes:
        pub_fig.apply_tufte_style(ax)
    
    pub_fig.add_panel_labels()
    
    plt.tight_layout()
    plt.savefig('demo_publication_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved: demo_publication_figure.png")


def demo_heatmap() -> None:
    """Demonstrate heatmap visualisation."""
    logger.info("Creating heatmap demonstration...")
    
    # Create correlation matrix
    np.random.seed(42)
    n_vars = 6
    data = np.random.randn(100, n_vars)
    data[:, 1] = data[:, 0] * 0.8 + data[:, 1] * 0.2  # Add correlation
    data[:, 3] = -data[:, 2] * 0.7 + data[:, 3] * 0.3
    
    corr = np.corrcoef(data.T)
    labels = ['Var A', 'Var B', 'Var C', 'Var D', 'Var E', 'Var F']
    
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_heatmap(ax, corr, labels, labels, cmap='RdBu_r', fmt='.2f')
    ax.set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('demo_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Saved: demo_heatmap.png")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_publication_figure()
    demo_heatmap()
    logger.info("All demonstrations completed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Static visualisation toolkit solution demonstrations"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        run_all_demos()
    else:
        print("Run with --demo to generate example figures")
        print("See source code for complete implementations")


if __name__ == "__main__":
    import argparse
    main()
