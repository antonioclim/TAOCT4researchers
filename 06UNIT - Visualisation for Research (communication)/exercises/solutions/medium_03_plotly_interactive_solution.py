#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Medium Exercise 3 — Interactive Plotly Scatter Plot
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates creating interactive scatter plots with Plotly,
including custom hover tooltips and optional dashboard layout.

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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# Wong colourblind-safe palette for Plotly
WONG_PLOTLY = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#F0E442",  # Yellow
    "#56B4E9",  # Sky blue
    "#D55E00",  # Vermillion
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ResearchDataset:
    """Container for research dataset with metadata."""

    df: pd.DataFrame
    x_col: str
    y_col: str
    colour_col: str
    size_col: str
    hover_cols: list[str]


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_research_data(
    n_samples: int = 150,
    seed: int = 42,
) -> ResearchDataset:
    """
    Generate synthetic research dataset.

    Simulates a dataset from a multi-site clinical study with
    continuous and categorical variables.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    ResearchDataset
        Container with DataFrame and column specifications.
    """
    rng = np.random.default_rng(seed)

    # Study sites
    sites = ["London", "Edinburgh", "Cardiff", "Belfast"]
    site_effects = {"London": 0, "Edinburgh": 5, "Cardiff": -3, "Belfast": 2}

    # Generate data
    data = {
        "participant_id": [f"P{i:04d}" for i in range(n_samples)],
        "site": rng.choice(sites, n_samples),
        "age": rng.integers(25, 75, n_samples),
        "baseline_score": rng.uniform(40, 90, n_samples),
    }

    # Outcome depends on baseline + site effect + noise
    data["outcome"] = (
        0.6 * data["baseline_score"]
        + np.array([site_effects[s] for s in data["site"]])
        + rng.normal(0, 8, n_samples)
    )

    # Effect size (for bubble size)
    data["effect_size"] = np.abs(data["outcome"] - data["baseline_score"])

    df = pd.DataFrame(data)

    return ResearchDataset(
        df=df,
        x_col="baseline_score",
        y_col="outcome",
        colour_col="site",
        size_col="effect_size",
        hover_cols=["participant_id", "age", "site"],
    )


def create_interactive_scatter(
    dataset: ResearchDataset,
    output_path: Path | None = None,
) -> go.Figure:
    """
    Create interactive scatter plot with custom hover tooltips.

    Features:
    - Colour-coded by categorical variable
    - Size mapped to continuous variable
    - Custom hover template with formatted data
    - Colourblind-friendly palette

    Parameters
    ----------
    dataset : ResearchDataset
        Container with data and column specifications.
    output_path : Path or None
        If provided, save the figure as HTML to this path.

    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    df = dataset.df

    # Create figure using express for convenience
    fig = px.scatter(
        df,
        x=dataset.x_col,
        y=dataset.y_col,
        color=dataset.colour_col,
        size=dataset.size_col,
        hover_data=dataset.hover_cols,
        color_discrete_sequence=WONG_PLOTLY,
        title="Clinical Study: Baseline Score vs Outcome by Site",
    )

    # Customise hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Age: %{customdata[1]}<br>"
            "Site: %{customdata[2]}<br>"
            "Baseline: %{x:.1f}<br>"
            "Outcome: %{y:.1f}<br>"
            "<extra></extra>"
        ),
    )

    # Update layout for publication quality
    fig.update_layout(
        title=dict(
            text="Clinical Study: Baseline Score vs Outcome by Site",
            font=dict(size=16),
            x=0.5,
        ),
        xaxis_title="Baseline Score",
        yaxis_title="Outcome Score",
        legend_title="Study Site",
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial",
        ),
    )

    # Add reference line (y = x)
    min_val = min(df[dataset.x_col].min(), df[dataset.y_col].min())
    max_val = max(df[dataset.x_col].max(), df[dataset.y_col].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="No change (y=x)",
            line=dict(dash="dash", color="gray", width=1),
            showlegend=True,
        )
    )

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        logger.info("Figure saved to %s", output_path)

    return fig


def create_dashboard_layout(
    dataset: ResearchDataset,
    output_path: Path | None = None,
) -> go.Figure:
    """
    Create multi-panel dashboard with scatter, histogram and box plot.

    Parameters
    ----------
    dataset : ResearchDataset
        Container with data and column specifications.
    output_path : Path or None
        If provided, save the figure as HTML to this path.

    Returns
    -------
    go.Figure
        Plotly figure with subplots.
    """
    df = dataset.df

    # Create subplots: 2x2 grid
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Baseline vs Outcome",
            "Outcome Distribution by Site",
            "Age Distribution",
            "Effect Size by Site",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "box"}],
            [{"type": "histogram"}, {"type": "bar"}],
        ],
    )

    # Panel 1: Scatter plot
    for i, site in enumerate(df[dataset.colour_col].unique()):
        mask = df[dataset.colour_col] == site
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, dataset.x_col],
                y=df.loc[mask, dataset.y_col],
                mode="markers",
                name=site,
                marker=dict(color=WONG_PLOTLY[i], size=8),
                legendgroup=site,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Box plot by site
    for i, site in enumerate(df[dataset.colour_col].unique()):
        mask = df[dataset.colour_col] == site
        fig.add_trace(
            go.Box(
                y=df.loc[mask, dataset.y_col],
                name=site,
                marker_color=WONG_PLOTLY[i],
                legendgroup=site,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Panel 3: Histogram of age
    fig.add_trace(
        go.Histogram(
            x=df["age"],
            nbinsx=15,
            marker_color=WONG_PLOTLY[0],
            name="Age",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Panel 4: Mean effect size by site
    effect_by_site = df.groupby(dataset.colour_col)[dataset.size_col].mean()
    fig.add_trace(
        go.Bar(
            x=effect_by_site.index,
            y=effect_by_site.values,
            marker_color=WONG_PLOTLY[:len(effect_by_site)],
            name="Mean Effect",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Clinical Study Dashboard",
            font=dict(size=18),
            x=0.5,
        ),
        height=700,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    # Update axis labels
    fig.update_xaxes(title_text="Baseline Score", row=1, col=1)
    fig.update_yaxes(title_text="Outcome Score", row=1, col=1)
    fig.update_xaxes(title_text="Site", row=1, col=2)
    fig.update_yaxes(title_text="Outcome Score", row=1, col=2)
    fig.update_xaxes(title_text="Age (years)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Site", row=2, col=2)
    fig.update_yaxes(title_text="Mean Effect Size", row=2, col=2)

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_path)
        logger.info("Dashboard saved to %s", output_path)

    return fig


def test_solution() -> None:
    """Test the solution implementation."""
    logger.info("Testing Plotly interactive scatter...")

    # Test 1: Data generation
    dataset = generate_research_data(n_samples=100, seed=42)
    assert len(dataset.df) == 100
    assert all(col in dataset.df.columns for col in [
        dataset.x_col, dataset.y_col, dataset.colour_col, dataset.size_col
    ])
    logger.info("✓ Data generation test passed")

    # Test 2: Scatter plot creation
    fig = create_interactive_scatter(dataset)
    assert fig is not None
    assert len(fig.data) > 0
    logger.info("✓ Scatter plot creation test passed")

    # Test 3: Dashboard creation
    fig = create_dashboard_layout(dataset)
    assert fig is not None
    # Should have multiple traces (scatter + box + histogram + bar)
    assert len(fig.data) >= 4
    logger.info("✓ Dashboard creation test passed")

    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point demonstrating the complete solution."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive Plotly scatter plot")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path")
    parser.add_argument("--dashboard", action="store_true", help="Create dashboard")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        test_solution()
        return

    # Generate data
    dataset = generate_research_data(n_samples=150, seed=42)
    logger.info("Generated dataset with %d samples", len(dataset.df))

    # Create visualisation
    if args.dashboard:
        fig = create_dashboard_layout(dataset, output_path=args.output)
    else:
        fig = create_interactive_scatter(dataset, output_path=args.output)

    if args.output is None:
        fig.show()


if __name__ == "__main__":
    main()
