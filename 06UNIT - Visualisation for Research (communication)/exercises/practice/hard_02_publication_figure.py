#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Hard 02 — Publication Figure Automation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create a complete figure generation pipeline for a multi-panel publication figure.

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 60 minutes
BLOOM LEVEL: Create/Evaluate

TASK
────
Complete the `PublicationFigure` class that:
1. Configures matplotlib for a specific journal (Nature, Science, IEEE)
2. Creates a multi-panel figure with consistent styling
3. Adds panel labels (A, B, C, D)
4. Exports to multiple formats (PDF, PNG, SVG)
5. Generates figure captions

HINTS
─────
- Use gridspec for complex layouts
- Panel labels typically go in the top-left corner
- Different journals have different font and size requirements
- Vector formats (PDF) are preferred for line art

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


@dataclass
class JournalStyle:
    """Configuration for journal-specific figure styling."""
    name: str
    column_width_mm: float  # Single column width
    double_column_mm: float  # Double column width
    font_family: str
    font_size: int
    label_size: int
    dpi: int
    
    @property
    def column_width_inches(self) -> float:
        return self.column_width_mm / 25.4
    
    @property
    def double_column_inches(self) -> float:
        return self.double_column_mm / 25.4


# Pre-defined journal styles
JOURNAL_STYLES = {
    'nature': JournalStyle(
        name='Nature',
        column_width_mm=89,
        double_column_mm=183,
        font_family='Arial',
        font_size=7,
        label_size=8,
        dpi=300
    ),
    'science': JournalStyle(
        name='Science',
        column_width_mm=85,
        double_column_mm=174,
        font_family='Helvetica',
        font_size=7,
        label_size=8,
        dpi=300
    ),
    'ieee': JournalStyle(
        name='IEEE',
        column_width_mm=88,
        double_column_mm=181,
        font_family='Times New Roman',
        font_size=8,
        label_size=9,
        dpi=600
    ),
}


@dataclass
class PanelConfig:
    """Configuration for a single panel in a multi-panel figure."""
    label: str  # e.g., 'A', 'B', 'C'
    title: str | None = None
    plot_func: Callable | None = None  # Function that takes (ax, data) and plots


class PublicationFigure:
    """Builder for publication-ready multi-panel figures."""
    
    def __init__(
        self,
        journal: str = 'nature',
        double_column: bool = False
    ):
        """Initialise the figure builder.
        
        Args:
            journal: Journal name ('nature', 'science', 'ieee')
            double_column: Whether to use double column width
        """
        self.style = JOURNAL_STYLES.get(journal.lower())
        if self.style is None:
            raise ValueError(f"Unknown journal: {journal}")
        
        self.double_column = double_column
        self.panels: list[PanelConfig] = []
        self.fig = None
        self.axes = None
        
        # Apply global style
        self._apply_style()
    
    def _apply_style(self) -> None:
        """Apply journal-specific matplotlib settings."""
        # TODO: Configure matplotlib rcParams for the journal
        # Set: font.family, font.size, axes.labelsize, axes.titlesize,
        #      legend.fontsize, xtick.labelsize, ytick.labelsize,
        #      figure.dpi, savefig.dpi, axes.linewidth, lines.linewidth,
        #      axes.spines.top, axes.spines.right
        
        # YOUR CODE HERE
        pass
    
    def add_panel(
        self,
        label: str,
        title: str | None = None,
        plot_func: Callable | None = None
    ) -> 'PublicationFigure':
        """Add a panel to the figure.
        
        Args:
            label: Panel label (e.g., 'A')
            title: Optional panel title
            plot_func: Function to create the plot
            
        Returns:
            Self for method chaining
        """
        self.panels.append(PanelConfig(label=label, title=title, plot_func=plot_func))
        return self
    
    def create_figure(
        self,
        nrows: int = 2,
        ncols: int = 2,
        height_ratios: list[float] | None = None,
        width_ratios: list[float] | None = None
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create the figure with specified layout.
        
        Args:
            nrows: Number of rows
            ncols: Number of columns
            height_ratios: Relative heights of rows
            width_ratios: Relative widths of columns
            
        Returns:
            Tuple of (figure, axes array)
        """
        # TODO: Create figure with appropriate dimensions and layout
        # 1. Calculate figure width based on journal and column type
        # 2. Calculate figure height (typically golden ratio or specified)
        # 3. Create figure and GridSpec
        # 4. Create axes for each panel
        # 5. Add panel labels in top-left corner of each panel
        
        # YOUR CODE HERE
        
        return self.fig, self.axes
    
    def add_panel_labels(
        self,
        fontweight: str = 'bold',
        fontsize: int | None = None,
        loc: tuple[float, float] = (-0.1, 1.1)
    ) -> None:
        """Add labels (A, B, C, ...) to each panel.
        
        Args:
            fontweight: Font weight for labels
            fontsize: Font size (defaults to style.label_size)
            loc: Location as (x, y) in axes coordinates
        """
        # TODO: Add labels to each panel
        # Use ax.text() with transform=ax.transAxes
        
        # YOUR CODE HERE
        pass
    
    def save(
        self,
        path: Path | str,
        formats: list[str] = ['pdf', 'png'],
        transparent: bool = False
    ) -> dict[str, Path]:
        """Save figure in multiple formats.
        
        Args:
            path: Base path (without extension)
            formats: List of formats to save
            transparent: Whether to use transparent background
            
        Returns:
            Dictionary mapping format to saved path
        """
        # TODO: Save figure in each format
        # Handle different DPI for raster vs vector formats
        
        # YOUR CODE HERE
        paths = {}
        
        return paths
    
    def generate_caption(
        self,
        main_description: str,
        panel_descriptions: dict[str, str] | None = None
    ) -> str:
        """Generate a structured figure caption.
        
        Args:
            main_description: Overall figure description
            panel_descriptions: Dict mapping panel labels to descriptions
            
        Returns:
            Formatted caption string
        """
        # TODO: Generate caption text
        # Format: "Figure X. Main description. (A) Panel A description. (B) ..."
        
        # YOUR CODE HERE
        caption = ""
        
        return caption


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_convergence(ax: plt.Axes, data: dict) -> None:
    """Plot Monte Carlo convergence."""
    x = data.get('iterations', np.arange(100))
    y = data.get('estimates', np.cumsum(np.random.randn(100)) / np.arange(1, 101) + 3.14)
    
    ax.plot(x, y, linewidth=1, color='#0072B2')
    ax.axhline(np.pi, color='#E69F00', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Estimate')


def plot_histogram(ax: plt.Axes, data: dict) -> None:
    """Plot distribution histogram."""
    values = data.get('values', np.random.normal(0, 1, 1000))
    
    ax.hist(values, bins=30, color='#009E73', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')


def plot_scatter(ax: plt.Axes, data: dict) -> None:
    """Plot scatter with regression."""
    x = data.get('x', np.random.randn(50))
    y = data.get('y', x * 0.8 + np.random.randn(50) * 0.3)
    
    ax.scatter(x, y, s=20, color='#CC79A7', alpha=0.7)
    ax.set_xlabel('X variable')
    ax.set_ylabel('Y variable')


def plot_bars(ax: plt.Axes, data: dict) -> None:
    """Plot bar chart."""
    categories = data.get('categories', ['A', 'B', 'C', 'D'])
    values = data.get('values', [25, 40, 30, 35])
    
    ax.bar(categories, values, color='#56B4E9', edgecolor='white')
    ax.set_xlabel('Category')
    ax.set_ylabel('Value')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Test journal style loading
    try:
        fig_builder = PublicationFigure(journal='nature')
        print("✓ PublicationFigure initialised")
    except Exception as e:
        print(f"❌ Failed to initialise: {e}")
        return False
    
    # Test adding panels
    fig_builder.add_panel('A', plot_func=plot_convergence)
    fig_builder.add_panel('B', plot_func=plot_histogram)
    fig_builder.add_panel('C', plot_func=plot_scatter)
    fig_builder.add_panel('D', plot_func=plot_bars)
    
    if len(fig_builder.panels) != 4:
        print("❌ Failed to add panels")
        return False
    
    print("✓ Panels added")
    
    # Test figure creation
    try:
        fig, axes = fig_builder.create_figure(nrows=2, ncols=2)
        
        if fig is None:
            print("❌ create_figure() returned None")
            return False
        
        print("✓ Figure created")
    except Exception as e:
        print(f"❌ Figure creation failed: {e}")
        return False
    
    # Test caption generation
    caption = fig_builder.generate_caption(
        "Simulation results from the Monte Carlo study.",
        {
            'A': "Convergence of the estimator over iterations.",
            'B': "Distribution of final estimates.",
            'C': "Correlation between parameters.",
            'D': "Summary statistics by category."
        }
    )
    
    if caption:
        print("✓ Caption generated")
    else:
        print("⚠ Caption generation not implemented")
    
    print("✓ All tests passed!")
    plt.show()
    
    return True


if __name__ == "__main__":
    test_implementation()
