#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
06UNIT, Lab 1: Static Visualisation Toolkit
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Effective data visualisation is a critical competency in research. This
laboratory provides a comprehensive toolkit for creating visualisations that
are publication-ready (conforming to journal standards), reproducible
(scriptable and version-controlled), and accessible (colourblind-friendly).

We apply Tufte's principles throughout: maximising data-ink ratio, eliminating
chartjunk, maintaining honest proportions, and using small multiples for
comparison.

PREREQUISITES
─────────────
- 05UNIT: Scientific Computing (simulation data, numerical methods)
- Python: Intermediate proficiency with NumPy and Pandas
- Libraries: matplotlib, seaborn, scipy

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Configure matplotlib for publication-quality output
2. Create figures conforming to Nature, Science and IEEE standards
3. Apply colourblind-friendly palettes and accessibility conventions
4. Export figures in multiple formats (PDF, PNG, SVG, EPS)

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 90 minutes
- Total: 2 hours

DEPENDENCIES
────────────
matplotlib>=3.7
numpy>=1.24
seaborn>=0.12
scipy>=1.11

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports with graceful degradation
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not available. Install with: pip install numpy")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("seaborn not available. Install with: pip install seaborn")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: COLOUR PALETTES
# ═══════════════════════════════════════════════════════════════════════════════

# Colourblind-friendly palettes following accessibility guidelines
PALETTES: dict[str, list[str]] = {
    # Wong palette - optimised for colourblindness
    'colorblind': [
        '#0072B2',  # Blue
        '#E69F00',  # Orange
        '#009E73',  # Teal
        '#CC79A7',  # Pink
        '#F0E442',  # Yellow
        '#56B4E9',  # Sky blue
        '#D55E00',  # Vermillion
        '#000000',  # Black
    ],
    # Default matplotlib-inspired
    'default': [
        '#4C72B0', '#DD8452', '#55A868', '#C44E52',
        '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
    ],
    # Greyscale for print
    'grayscale': [
        '#000000', '#404040', '#808080', '#BFBFBF',
        '#D9D9D9', '#F2F2F2', '#1A1A1A', '#666666',
    ],
    # Nature journal style
    'nature': [
        '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
        '#F39B7F', '#8491B4', '#91D1C2', '#DC0000',
    ],
    # Viridis-inspired categorical
    'viridis': [
        '#440154', '#31688E', '#35B779', '#FDE725',
        '#21918C', '#5DC863', '#482878', '#1F968B',
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: JOURNAL STYLE CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Configuration dictionaries for major academic journals
JOURNAL_STYLES: dict[str, dict[str, Any]] = {
    'nature': {
        'figure.figsize': (3.5, 2.5),  # Single column: 89mm ≈ 3.5"
        'font.size': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.5,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'savefig.dpi': 300,
    },
    'science': {
        'figure.figsize': (3.35, 2.5),  # Single column: 85mm ≈ 3.35"
        'font.size': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 0.5,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'lines.linewidth': 0.75,
        'lines.markersize': 3,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'savefig.dpi': 300,
    },
    'ieee': {
        'figure.figsize': (3.5, 2.5),  # Single column: 88mm ≈ 3.5"
        'font.size': 8,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.linewidth': 0.5,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'savefig.dpi': 600,  # IEEE requires higher resolution
    },
    'thesis': {
        'figure.figsize': (6, 4),
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.linewidth': 0.8,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'savefig.dpi': 300,
    },
    'presentation': {
        'figure.figsize': (10, 6),
        'font.size': 14,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.linewidth': 1.5,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'savefig.dpi': 150,
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PLOT STYLE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlotStyle:
    """
    Configuration class for plot styling.
    
    This dataclass encapsulates all visual parameters needed to create
    consistent, publication-ready figures. It supports preset journal
    styles and custom configurations.
    
    Attributes:
        figsize: Tuple of (width, height) in inches
        dpi: Dots per inch for raster output
        font_family: Primary font family ('serif' or 'sans-serif')
        font_size: Base font size in points
        palette: Name of colour palette from PALETTES dict
        
    Example:
        >>> style = PlotStyle.for_journal('nature')
        >>> style.apply()
        >>> fig, ax = plt.subplots()
    """
    
    # Dimensions
    figsize: tuple[float, float] = (6, 4)
    dpi: int = 300
    
    # Typography
    font_family: str = 'serif'
    font_size: float = 10
    title_size: float = 12
    label_size: float = 10
    tick_size: float = 9
    legend_size: float = 9
    
    # Lines and markers
    line_width: float = 1.5
    marker_size: float = 5
    axes_linewidth: float = 0.8
    
    # Colours
    palette: str = 'colorblind'
    background: str = 'white'
    grid_colour: str = '#CCCCCC'
    grid_alpha: float = 0.5
    
    # Grid configuration
    show_grid: bool = True
    grid_style: str = '--'
    
    # Spine visibility (Tufte recommends hiding top and right)
    spine_top: bool = False
    spine_right: bool = False
    spine_left: bool = True
    spine_bottom: bool = True
    
    def apply(self) -> None:
        """
        Apply this style configuration to matplotlib rcParams.
        
        This method updates the global matplotlib configuration. Call
        this before creating figures to ensure consistent styling.
        
        Raises:
            ImportError: If matplotlib is not available.
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib required. Install with: pip install matplotlib")
        
        logger.info(f"Applying plot style: {self.font_family}, {self.font_size}pt")
        
        mpl.rcParams.update({
            # Figure
            'figure.figsize': self.figsize,
            'figure.dpi': self.dpi,
            'figure.facecolor': self.background,
            'figure.edgecolor': self.background,
            
            # Saving
            'savefig.dpi': self.dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.facecolor': self.background,
            
            # Typography
            'font.family': self.font_family,
            'font.size': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'legend.fontsize': self.legend_size,
            
            # Lines and markers
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            'axes.linewidth': self.axes_linewidth,
            
            # Colours
            'axes.facecolor': self.background,
            
            # Grid
            'axes.grid': self.show_grid,
            'grid.color': self.grid_colour,
            'grid.alpha': self.grid_alpha,
            'grid.linestyle': self.grid_style,
            
            # Spines
            'axes.spines.top': self.spine_top,
            'axes.spines.right': self.spine_right,
            'axes.spines.left': self.spine_left,
            'axes.spines.bottom': self.spine_bottom,
        })
    
    def get_colours(self, n: int | None = None) -> list[str]:
        """
        Retrieve colours from the configured palette.
        
        Args:
            n: Number of colours needed. If None, returns full palette.
               If n exceeds palette length, colours cycle.
               
        Returns:
            List of hex colour strings.
        """
        colours = PALETTES.get(self.palette, PALETTES['colorblind'])
        if n is None:
            return colours.copy()
        return [colours[i % len(colours)] for i in range(n)]
    
    @classmethod
    def for_journal(cls, journal: str) -> 'PlotStyle':
        """
        Create a PlotStyle configured for a specific journal.
        
        Args:
            journal: Journal name ('nature', 'science', 'ieee', 'thesis',
                    'presentation').
                    
        Returns:
            Configured PlotStyle instance.
            
        Raises:
            ValueError: If journal name is not recognised.
            
        Example:
            >>> style = PlotStyle.for_journal('nature')
            >>> style.apply()
        """
        if journal not in JOURNAL_STYLES:
            available = ', '.join(JOURNAL_STYLES.keys())
            raise ValueError(f"Unknown journal: {journal}. Available: {available}")
        
        config = JOURNAL_STYLES[journal]
        logger.info(f"Creating style for journal: {journal}")
        
        return cls(
            figsize=config['figure.figsize'],
            dpi=config['savefig.dpi'],
            font_family=config['font.family'],
            font_size=config['font.size'],
            label_size=config['axes.labelsize'],
            title_size=config['axes.titlesize'],
            tick_size=config['xtick.labelsize'],
            legend_size=config['legend.fontsize'],
            line_width=config['lines.linewidth'],
            marker_size=config['lines.markersize'],
            axes_linewidth=config['axes.linewidth'],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FIGURE CREATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    style: PlotStyle | None = None,
    **kwargs: Any
) -> tuple['Figure', 'Axes | list[Axes]']:
    """
    Create a matplotlib figure with optional style application.
    
    This is a convenience wrapper around plt.subplots that optionally
    applies a PlotStyle before creating the figure.
    
    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        style: PlotStyle instance to apply. If None, uses current settings.
        **kwargs: Additional arguments passed to plt.subplots.
        
    Returns:
        Tuple of (Figure, Axes or array of Axes).
        
    Raises:
        ImportError: If matplotlib is not available.
        
    Example:
        >>> style = PlotStyle.for_journal('nature')
        >>> fig, ax = create_figure(style=style)
        >>> ax.plot([1, 2, 3], [1, 4, 9])
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    if style is not None:
        style.apply()
    
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    logger.debug(f"Created figure with {nrows}x{ncols} subplots")
    
    return fig, axes


def save_publication_figure(
    fig: 'Figure',
    filename: str,
    formats: list[str] | None = None,
    dpi: int = 300,
    transparent: bool = False
) -> list[Path]:
    """
    Save a figure in multiple formats suitable for publication.
    
    This function exports the figure to several formats commonly required
    by journals: PDF (vector), PNG (raster), and optionally SVG and EPS.
    
    Args:
        fig: The matplotlib Figure to save.
        filename: Base filename without extension.
        formats: List of formats to export. Defaults to ['pdf', 'png'].
        dpi: Resolution for raster formats.
        transparent: Whether to use transparent background.
        
    Returns:
        List of Path objects for saved files.
        
    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 9])
        >>> paths = save_publication_figure(fig, 'results/figure_1')
        >>> print(paths)
        [PosixPath('results/figure_1.pdf'), PosixPath('results/figure_1.png')]
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    if formats is None:
        formats = ['pdf', 'png']
    
    saved_paths: list[Path] = []
    base_path = Path(filename)
    
    # Create directory if needed
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        output_path = base_path.with_suffix(f'.{fmt}')
        
        # Vector formats don't need DPI
        save_dpi = None if fmt in ['pdf', 'svg', 'eps'] else dpi
        
        fig.savefig(
            output_path,
            format=fmt,
            dpi=save_dpi,
            bbox_inches='tight',
            pad_inches=0.05,
            facecolor=fig.get_facecolor() if not transparent else 'none',
            edgecolor='none',
            transparent=transparent
        )
        
        saved_paths.append(output_path)
        logger.info(f"Saved: {output_path}")
    
    return saved_paths


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: ANNOTATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def add_annotation(
    ax: 'Axes',
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float] | None = None,
    fontsize: int = 9,
    arrowprops: dict[str, Any] | None = None
) -> None:
    """
    Add a text annotation to an axes with optional arrow.
    
    Args:
        ax: The matplotlib Axes to annotate.
        text: Annotation text.
        xy: Point being annotated (data coordinates).
        xytext: Position of text. If None, places at xy.
        fontsize: Font size for annotation text.
        arrowprops: Dictionary of arrow properties.
    """
    if arrowprops is None and xytext is not None:
        arrowprops = {
            'arrowstyle': '->',
            'connectionstyle': 'arc3,rad=0.2',
            'color': '#333333',
            'linewidth': 0.8,
        }
    
    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        arrowprops=arrowprops,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                  edgecolor='#CCCCCC', alpha=0.9)
    )


def add_significance_bar(
    ax: 'Axes',
    x1: float,
    x2: float,
    y: float,
    text: str = '*',
    height: float = 0.02
) -> None:
    """
    Add a significance bar between two x positions.
    
    Common in biological research to indicate statistical significance
    between experimental conditions.
    
    Args:
        ax: The matplotlib Axes.
        x1: Left x position.
        x2: Right x position.
        y: Y position (bottom of bar, in data coordinates).
        text: Significance marker ('*', '**', '***', 'ns').
        height: Height of vertical lines as fraction of y-range.
    """
    # Get y-axis range for scaling
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bar_height = height * y_range
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], 
            [y, y + bar_height, y + bar_height, y],
            color='black', linewidth=0.8)
    
    # Add text
    ax.text((x1 + x2) / 2, y + bar_height, text,
            ha='center', va='bottom', fontsize=10)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COMMON PLOT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_with_error_band(
    ax: 'Axes',
    x: list[float],
    y: list[float],
    yerr: list[float],
    colour: str | None = None,
    label: str | None = None,
    alpha: float = 0.2
) -> None:
    """
    Plot a line with a shaded error band.
    
    This is a common representation for time series with uncertainty,
    such as model predictions with confidence intervals.
    
    Args:
        ax: The matplotlib Axes.
        x: X coordinates.
        y: Y coordinates (central line).
        yerr: Error values (symmetric).
        colour: Line and band colour.
        label: Legend label.
        alpha: Transparency of error band.
    """
    if HAS_NUMPY:
        x_arr = np.array(x)
        y_arr = np.array(y)
        yerr_arr = np.array(yerr)
        y_lower = y_arr - yerr_arr
        y_upper = y_arr + yerr_arr
    else:
        x_arr = x
        y_arr = y
        y_lower = [yi - ei for yi, ei in zip(y, yerr)]
        y_upper = [yi + ei for yi, ei in zip(y, yerr)]
    
    line = ax.plot(x_arr, y_arr, color=colour, label=label)[0]
    ax.fill_between(x_arr, y_lower, y_upper, 
                    color=line.get_color(), alpha=alpha, linewidth=0)


def scatter_with_regression(
    ax: 'Axes',
    x: list[float],
    y: list[float],
    colour: str | None = None,
    show_equation: bool = True,
    show_r2: bool = True
) -> tuple[float, float, float]:
    """
    Create a scatter plot with linear regression line.
    
    Args:
        ax: The matplotlib Axes.
        x: X coordinates.
        y: Y coordinates.
        colour: Point and line colour.
        show_equation: Whether to display regression equation.
        show_r2: Whether to display R² value.
        
    Returns:
        Tuple of (slope, intercept, r_squared).
    """
    if colour is None:
        colour = PALETTES['colorblind'][0]
    
    # Calculate regression
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_xx = sum(xi * xi for xi in x)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate R²
    y_pred = [slope * xi + intercept for xi in x]
    y_mean = sum_y / n
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Plot scatter
    ax.scatter(x, y, c=colour, alpha=0.6, s=30, 
               edgecolors='white', linewidth=0.5)
    
    # Plot regression line
    x_line = [min(x), max(x)]
    y_line = [slope * xi + intercept for xi in x_line]
    ax.plot(x_line, y_line, c=colour, linestyle='--', linewidth=1.5)
    
    # Add annotation
    if show_equation or show_r2:
        text_parts = []
        if show_equation:
            sign = '+' if intercept >= 0 else '−'
            text_parts.append(f'y = {slope:.3f}x {sign} {abs(intercept):.3f}')
        if show_r2:
            text_parts.append(f'R² = {r_squared:.3f}')
        
        ax.text(0.05, 0.95, '\n'.join(text_parts),
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return slope, intercept, r_squared


def create_heatmap(
    ax: 'Axes',
    data: list[list[float]],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cmap: str = 'viridis',
    annotate: bool = True,
    fmt: str = '.2f',
    cbar_label: str | None = None
) -> None:
    """
    Create a heatmap with optional annotations.
    
    Useful for correlation matrices, confusion matrices, and
    any 2D tabular data.
    
    Args:
        ax: The matplotlib Axes.
        data: 2D list of values.
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        cmap: Colour map name.
        annotate: Whether to show values in cells.
        fmt: Format string for annotations.
        cbar_label: Label for colour bar.
    """
    if HAS_NUMPY:
        data_arr = np.array(data)
    else:
        data_arr = data
    
    n_rows = len(data)
    n_cols = len(data[0]) if data else 0
    
    # Create heatmap
    im = ax.imshow(data_arr, cmap=cmap, aspect='auto')
    
    # Add colour bar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va='bottom')
    
    # Set tick labels
    if row_labels:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
    if col_labels:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
    
    # Add annotations
    if annotate:
        data_flat = [val for row in data for val in row]
        vmin, vmax = min(data_flat), max(data_flat)
        midpoint = (vmin + vmax) / 2
        
        for i in range(n_rows):
            for j in range(n_cols):
                value = data[i][j]
                colour = 'white' if value > midpoint else 'black'
                ax.text(j, i, f'{value:{fmt}}', 
                        ha='center', va='center', 
                        color=colour, fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: D3.JS EXPORT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_d3_json(
    data: dict[str, Any],
    filename: str | Path
) -> Path:
    """
    Export data to JSON format suitable for D3.js visualisation.
    
    Args:
        data: Dictionary to export.
        filename: Output path (with or without .json extension).
        
    Returns:
        Path to saved file.
    """
    path = Path(filename)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Exported D3 data to: {path}")
    return path


def generate_d3_line_chart_html(
    data: list[dict[str, float]],
    x_key: str,
    y_keys: list[str],
    title: str = "Line Chart",
    width: int = 800,
    height: int = 400
) -> str:
    """
    Generate a standalone HTML file containing a D3.js line chart.
    
    This function creates a complete HTML document with embedded
    D3.js code for an interactive line chart.
    
    Args:
        data: List of dictionaries with data points.
        x_key: Key for x-axis values.
        y_keys: List of keys for y-axis lines.
        title: Chart title.
        width: Chart width in pixels.
        height: Chart height in pixels.
        
    Returns:
        Complete HTML string.
        
    Example:
        >>> data = [{'time': i, 'value': i**2} for i in range(10)]
        >>> html = generate_d3_line_chart_html(data, 'time', ['value'])
        >>> Path('chart.html').write_text(html)
    """
    colours = PALETTES['colorblind'][:len(y_keys)]
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 20px;
        }}
        .line {{ fill: none; stroke-width: 2; }}
        .axis-label {{ font-size: 12px; }}
        .title {{ font-size: 16px; font-weight: bold; }}
        .legend {{ font-size: 11px; }}
        .grid line {{ stroke: #e0e0e0; stroke-opacity: 0.7; shape-rendering: crispEdges; }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <script>
        const data = {json.dumps(data)};
        const margin = {{top: 40, right: 120, bottom: 50, left: 60}};
        const width = {width} - margin.left - margin.right;
        const height = {height} - margin.top - margin.bottom;
        
        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
        
        // Scales
        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d["{x_key}"]))
            .range([0, width]);
        
        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => Math.max({", ".join(f'd["{k}"]' for k in y_keys)}))])
            .nice()
            .range([height, 0]);
        
        // Grid
        svg.append("g")
            .attr("class", "grid")
            .call(d3.axisLeft(y).tickSize(-width).tickFormat(""));
        
        // Axes
        svg.append("g")
            .attr("transform", `translate(0,${{height}})`)
            .call(d3.axisBottom(x));
        
        svg.append("g")
            .call(d3.axisLeft(y));
        
        // Title
        svg.append("text")
            .attr("class", "title")
            .attr("x", width / 2)
            .attr("y", -15)
            .attr("text-anchor", "middle")
            .text("{title}");
        
        // Lines and legend
        const colours = {json.dumps(colours)};
        const yKeys = {json.dumps(y_keys)};
        
        yKeys.forEach((key, i) => {{
            const line = d3.line()
                .x(d => x(d["{x_key}"]))
                .y(d => y(d[key]));
            
            svg.append("path")
                .datum(data)
                .attr("class", "line")
                .attr("stroke", colours[i])
                .attr("d", line);
            
            // Legend
            svg.append("circle")
                .attr("cx", width + 15)
                .attr("cy", 10 + i * 25)
                .attr("r", 5)
                .style("fill", colours[i]);
            
            svg.append("text")
                .attr("class", "legend")
                .attr("x", width + 25)
                .attr("y", 10 + i * 25)
                .attr("dy", "0.35em")
                .text(key);
        }});
    </script>
</body>
</html>'''
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: DEMONSTRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_journal_styles() -> None:
    """Demonstrate different journal style configurations."""
    logger.info("Running journal styles demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: Journal Style Configurations")
    print("=" * 70)
    print()
    
    for journal in ['nature', 'science', 'ieee', 'thesis', 'presentation']:
        style = PlotStyle.for_journal(journal)
        print(f"  {journal.upper():15} │ figsize: {str(style.figsize):12} │ "
              f"font: {style.font_size:4}pt {style.font_family:10} │ "
              f"DPI: {style.dpi}")
    
    print()
    print("Usage: style = PlotStyle.for_journal('nature')")
    print("       style.apply()")
    print()


def demo_colour_palettes() -> None:
    """Demonstrate available colour palettes."""
    logger.info("Running colour palettes demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: Colour Palettes (Accessibility-Focused)")
    print("=" * 70)
    print()
    
    for name, colours in PALETTES.items():
        colour_preview = ' '.join(f'[{c}]' for c in colours[:6])
        print(f"  {name:12} │ {colour_preview}")
    
    print()
    print("Recommended for accessibility: 'colorblind' (Wong palette)")
    print()


def demo_d3_export() -> None:
    """Demonstrate D3.js HTML export."""
    logger.info("Running D3.js export demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: D3.js Export")
    print("=" * 70)
    print()
    
    # Generate sample data
    data = [
        {
            'time': i,
            'model_a': math.sin(i / 5) * 30 + 50,
            'model_b': math.cos(i / 5) * 25 + 50
        }
        for i in range(50)
    ]
    
    html = generate_d3_line_chart_html(
        data,
        x_key='time',
        y_keys=['model_a', 'model_b'],
        title='Model Comparison'
    )
    
    print(f"  Generated HTML: {len(html):,} characters")
    print(f"  Data points: {len(data)}")
    print()
    print("  To view: Save HTML to file and open in browser.")
    print("  Example: Path('demo_chart.html').write_text(html)")
    print()


def demo_publication_figure() -> None:
    """Demonstrate creating a publication-ready figure."""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("  [Skipped: matplotlib and numpy required]")
        return
    
    logger.info("Running publication figure demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: Publication-Ready Figure")
    print("=" * 70)
    print()
    
    # Apply Nature style
    style = PlotStyle.for_journal('nature')
    style.apply()
    
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 50)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 50)
    
    # Create figure
    fig, ax = plt.subplots()
    
    colours = style.get_colours(2)
    ax.plot(x, y1, color=colours[0], label='Model A')
    ax.plot(x, y2, color=colours[1], label='Model B')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (a.u.)')
    ax.set_title('Comparison of Model Outputs')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#CCCCCC')
    
    print("  Figure created with Nature journal style")
    print("  - Font: 7pt sans-serif")
    print("  - Size: 3.5\" × 2.5\" (single column)")
    print("  - DPI: 300")
    print()
    print("  To save: save_publication_figure(fig, 'output/figure_1')")
    print()
    
    plt.close(fig)


def run_all_demos() -> None:
    """Execute all demonstration functions."""
    demo_journal_styles()
    demo_colour_palettes()
    demo_d3_export()
    demo_publication_figure()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Week 6 Lab 1: Static Visualisation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo              Run all demonstrations
  %(prog)s --list-palettes     Show available colour palettes
  %(prog)s --list-journals     Show available journal styles
        """
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run all demonstrations'
    )
    parser.add_argument(
        '--list-palettes',
        action='store_true',
        help='List available colour palettes'
    )
    parser.add_argument(
        '--list-journals',
        action='store_true',
        help='List available journal styles'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print()
    print("═" * 70)
    print("  WEEK 6 LAB 1: STATIC VISUALISATION TOOLKIT")
    print("═" * 70)
    print()
    
    if args.list_palettes:
        demo_colour_palettes()
    elif args.list_journals:
        demo_journal_styles()
    elif args.demo:
        run_all_demos()
    else:
        print("  Use --demo to run demonstrations")
        print("  Use --help for all options")
        print()
        print("Quick start:")
        print("  from lab_6_01_static_plots import PlotStyle, create_figure")
        print("  style = PlotStyle.for_journal('nature')")
        print("  fig, ax = create_figure(style=style)")
        print("  ax.plot([1, 2, 3], [1, 4, 9])")
    
    print()
    print("═" * 70)
    print("  © 2025 Antonio Clim. All rights reserved.")
    print("═" * 70)


if __name__ == "__main__":
    main()
