#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Easy Exercise 2 — Scatter Plot with Linear Regression
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates scatter plot creation with linear regression line,
R² annotation and colourblind-friendly styling.

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
from scipy import stats

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RegressionResult:
    """Container for linear regression results."""

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_err: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict y values for given x."""
        return self.slope * x + self.intercept

    def equation_string(self) -> str:
        """Return formatted equation string."""
        sign = "+" if self.intercept >= 0 else "-"
        return f"y = {self.slope:.3f}x {sign} {abs(self.intercept):.3f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_sample_data(
    n_points: int = 50,
    noise_level: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate sample data with linear relationship plus noise.

    Parameters
    ----------
    n_points : int
        Number of data points to generate.
    noise_level : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (x, y) arrays of data points.
    """
    rng = np.random.default_rng(seed)

    # Generate x values
    x = np.linspace(0, 10, n_points)

    # True relationship: y = 2.5x + 3 + noise
    y = 2.5 * x + 3 + rng.normal(0, noise_level, n_points)

    return x, y


def perform_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
) -> RegressionResult:
    """
    Perform linear regression using scipy.stats.

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Dependent variable values.

    Returns
    -------
    RegressionResult
        Container with regression statistics.
    """
    result = stats.linregress(x, y)

    return RegressionResult(
        slope=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue**2,
        p_value=result.pvalue,
        std_err=result.stderr,
    )


def create_scatter_with_regression(
    x: np.ndarray,
    y: np.ndarray,
    regression: RegressionResult,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    Create scatter plot with regression line and R² annotation.

    This function demonstrates:
    - Scatter plot with appropriate marker size and alpha
    - Overlaid regression line
    - Text annotation with statistical results
    - Colourblind-friendly palette (Wong)

    Parameters
    ----------
    x : np.ndarray
        Independent variable values.
    y : np.ndarray
        Dependent variable values.
    regression : RegressionResult
        Pre-computed regression results.
    output_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # Wong colourblind-safe palette
    colour_scatter = "#0072B2"  # Blue
    colour_line = "#D55E00"  # Vermillion

    # Plot scatter points
    ax.scatter(
        x,
        y,
        c=colour_scatter,
        alpha=0.6,
        s=50,
        edgecolors="white",
        linewidths=0.5,
        label="Data points",
    )

    # Plot regression line
    x_line = np.array([x.min(), x.max()])
    y_line = regression.predict(x_line)
    ax.plot(
        x_line,
        y_line,
        color=colour_line,
        linewidth=2,
        label=f"Regression: {regression.equation_string()}",
    )

    # Add R² annotation
    annotation_text = (
        f"R² = {regression.r_squared:.4f}\n"
        f"p < {regression.p_value:.2e}"
    )
    ax.annotate(
        annotation_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # Configure axes
    ax.set_xlabel("Independent Variable (x)", fontsize=11)
    ax.set_ylabel("Dependent Variable (y)", fontsize=11)
    ax.set_title(
        "Scatter Plot with Linear Regression",
        fontsize=12,
        fontweight="bold",
    )

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Configure legend
    ax.legend(loc="lower right", framealpha=0.9, fontsize=10)

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
    logger.info("Testing scatter plot with regression...")

    # Test 1: Data generation
    x, y = generate_sample_data(n_points=50, seed=42)
    assert len(x) == len(y) == 50
    logger.info("✓ Data generation test passed")

    # Test 2: Regression calculation
    regression = perform_linear_regression(x, y)
    assert 0 < regression.r_squared <= 1
    assert regression.slope > 0  # Positive relationship expected
    logger.info("✓ Regression calculation test passed (R² = %.4f)", regression.r_squared)

    # Test 3: Plot creation
    fig, ax = create_scatter_with_regression(x, y, regression)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
    logger.info("✓ Plot creation test passed")

    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point demonstrating the complete solution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scatter plot with linear regression"
    )
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
    x, y = generate_sample_data(n_points=50, seed=42)

    # Perform regression
    regression = perform_linear_regression(x, y)
    logger.info("Regression: %s", regression.equation_string())
    logger.info("R² = %.4f, p = %.2e", regression.r_squared, regression.p_value)

    # Create plot
    fig, ax = create_scatter_with_regression(x, y, regression, output_path=args.output)

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
