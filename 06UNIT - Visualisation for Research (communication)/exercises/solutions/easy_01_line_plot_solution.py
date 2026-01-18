#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Easy Exercise 1 — Monte Carlo π Estimation Line Plot
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates the creation of a convergence plot showing how
Monte Carlo estimation of π improves with increasing sample sizes.

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

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def estimate_pi_monte_carlo(n_samples: int, seed: int | None = None) -> float:
    """
    Estimate π using Monte Carlo method.

    The method works by generating random points in a unit square [0,1]×[0,1]
    and counting how many fall inside the quarter circle of radius 1.
    The ratio approximates π/4.

    Parameters
    ----------
    n_samples : int
        Number of random points to generate.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    float
        Estimated value of π.

    Examples
    --------
    >>> estimate = estimate_pi_monte_carlo(100000, seed=42)
    >>> 3.1 < estimate < 3.2
    True
    """
    rng = np.random.default_rng(seed)

    # Generate random points in unit square
    x = rng.random(n_samples)
    y = rng.random(n_samples)

    # Count points inside quarter circle (x² + y² ≤ 1)
    inside_circle = np.sum(x**2 + y**2 <= 1)

    # π/4 ≈ inside_circle / n_samples
    return 4 * inside_circle / n_samples


def generate_convergence_data(
    sample_sizes: list[int],
    seed: int = 42,
) -> tuple[list[int], list[float]]:
    """
    Generate convergence data for multiple sample sizes.

    Parameters
    ----------
    sample_sizes : list of int
        List of sample sizes to evaluate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (sample_sizes, estimates) where estimates are π approximations.
    """
    estimates = []
    for n in sample_sizes:
        estimate = estimate_pi_monte_carlo(n, seed=seed)
        estimates.append(estimate)
        logger.debug("n=%d, π estimate=%.6f", n, estimate)

    return sample_sizes, estimates


def create_convergence_plot(
    sample_sizes: list[int],
    estimates: list[float],
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    Create a publication-quality convergence plot.

    This function demonstrates:
    - Object-oriented matplotlib interface
    - Proper axis labels with units
    - Reference line for true value
    - Logarithmic scale for sample sizes
    - Colourblind-friendly colour palette

    Parameters
    ----------
    sample_sizes : list of int
        Sample sizes used for estimation.
    estimates : list of float
        Corresponding π estimates.
    output_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """
    # Create figure with explicit size for publication
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Wong colourblind-safe palette
    colour_blue = "#0072B2"
    colour_orange = "#E69F00"

    # Plot estimates with markers
    ax.plot(
        sample_sizes,
        estimates,
        marker="o",
        markersize=6,
        linewidth=1.5,
        color=colour_blue,
        label="Monte Carlo estimate",
    )

    # Add reference line for true π
    ax.axhline(
        y=np.pi,
        color=colour_orange,
        linestyle="--",
        linewidth=2,
        label=f"True π = {np.pi:.6f}",
    )

    # Configure axes
    ax.set_xscale("log")
    ax.set_xlabel("Number of samples (log scale)", fontsize=11)
    ax.set_ylabel("Estimated value of π", fontsize=11)
    ax.set_title(
        "Monte Carlo Estimation of π: Convergence with Sample Size",
        fontsize=12,
        fontweight="bold",
    )

    # Add grid for readability
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)

    # Configure legend
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)

    # Set y-axis limits to show convergence clearly
    ax.set_ylim(2.8, 3.5)

    # Tight layout for proper spacing
    fig.tight_layout()

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Figure saved to %s", output_path)

    return fig, ax


def test_solution() -> None:
    """
    Test the solution implementation.

    This function validates:
    1. π estimation is within reasonable bounds
    2. Convergence data generation works
    3. Plot creation succeeds without errors
    """
    logger.info("Testing Monte Carlo π estimation...")

    # Test 1: Basic estimation
    estimate = estimate_pi_monte_carlo(100000, seed=42)
    assert 3.0 < estimate < 3.3, f"Estimate {estimate} not in valid range"
    logger.info("✓ Basic estimation test passed (π ≈ %.4f)", estimate)

    # Test 2: Convergence data generation
    sample_sizes = [100, 1000, 10000, 100000]
    sizes, estimates = generate_convergence_data(sample_sizes, seed=42)
    assert len(sizes) == len(estimates) == 4
    logger.info("✓ Convergence data generation test passed")

    # Test 3: Plot creation
    fig, ax = create_convergence_plot(sizes, estimates)
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
        description="Monte Carlo π estimation convergence plot"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the figure",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        test_solution()
        return

    # Generate convergence data
    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    sizes, estimates = generate_convergence_data(sample_sizes, seed=42)

    # Create and display the plot
    fig, ax = create_convergence_plot(sizes, estimates, output_path=args.output)

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
