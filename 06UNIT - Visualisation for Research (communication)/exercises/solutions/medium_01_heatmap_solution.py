#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Medium Exercise 1 — Correlation Matrix Heatmap
═══════════════════════════════════════════════════════════════════════════════

This solution demonstrates creating a correlation heatmap with annotations,
upper triangle masking and diverging colourmap for publication-quality output.

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
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_correlated_data(
    n_samples: int = 200,
    n_features: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate multivariate data with varying correlation structure.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_features : int
        Number of features/variables.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (data array, feature names).
    """
    rng = np.random.default_rng(seed)

    # Create covariance matrix with structure
    cov = np.eye(n_features)

    # Add some correlated pairs
    # Variables 0-1 strongly correlated
    cov[0, 1] = cov[1, 0] = 0.85
    # Variables 2-3 moderately correlated
    cov[2, 3] = cov[3, 2] = 0.6
    # Variables 4-5 negatively correlated
    cov[4, 5] = cov[5, 4] = -0.7
    # Variables 6-7 weakly correlated
    cov[6, 7] = cov[7, 6] = 0.3
    # Some cross-correlations
    cov[0, 2] = cov[2, 0] = 0.4
    cov[1, 3] = cov[3, 1] = 0.35

    # Generate multivariate normal data
    mean = np.zeros(n_features)
    data = rng.multivariate_normal(mean, cov, n_samples)

    # Feature names
    feature_names = [f"Var_{i+1}" for i in range(n_features)]

    return data, feature_names


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation matrix.

    Parameters
    ----------
    data : np.ndarray
        Data array with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Correlation matrix with shape (n_features, n_features).
    """
    return np.corrcoef(data.T)


def create_upper_triangle_mask(n: int) -> np.ndarray:
    """
    Create mask for upper triangle (excluding diagonal).

    Parameters
    ----------
    n : int
        Size of the square matrix.

    Returns
    -------
    np.ndarray
        Boolean mask with True for upper triangle.
    """
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return mask


def create_correlation_heatmap(
    corr_matrix: np.ndarray,
    feature_names: list[str],
    mask_upper: bool = True,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """
    Create publication-quality correlation heatmap.

    Features:
    - Diverging colourmap (blue-white-red) centred at zero
    - Optional upper triangle masking
    - Cell annotations with correlation values
    - Proper axis labels

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix.
    feature_names : list of str
        Names for each feature.
    mask_upper : bool
        Whether to mask the upper triangle.
    output_path : Path or None
        If provided, save the figure to this path.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects.
    """
    n = len(feature_names)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Create mask if requested
    mask = create_upper_triangle_mask(n) if mask_upper else None

    # Create heatmap with seaborn
    # Use diverging colourmap centred at zero
    heatmap = sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",  # Diverging: red (negative) - white - blue (positive)
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 9},
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={
            "shrink": 0.8,
            "label": "Pearson Correlation Coefficient",
        },
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=ax,
    )

    # Configure labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Correlation Matrix Heatmap",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    # Rotate x labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

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
    logger.info("Testing correlation heatmap...")

    # Test 1: Data generation
    data, names = generate_correlated_data(n_samples=100, n_features=5, seed=42)
    assert data.shape == (100, 5)
    assert len(names) == 5
    logger.info("✓ Data generation test passed")

    # Test 2: Correlation computation
    corr = compute_correlation_matrix(data)
    assert corr.shape == (5, 5)
    assert np.allclose(np.diag(corr), 1.0)  # Diagonal should be 1
    assert np.allclose(corr, corr.T)  # Should be symmetric
    logger.info("✓ Correlation computation test passed")

    # Test 3: Mask creation
    mask = create_upper_triangle_mask(5)
    assert mask.shape == (5, 5)
    assert mask[0, 0] == False  # Diagonal not masked
    assert mask[0, 4] == True  # Upper triangle masked
    assert mask[4, 0] == False  # Lower triangle not masked
    logger.info("✓ Mask creation test passed")

    # Test 4: Heatmap creation
    fig, ax = create_correlation_heatmap(corr, names, mask_upper=True)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
    logger.info("✓ Heatmap creation test passed")

    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point demonstrating the complete solution."""
    import argparse

    parser = argparse.ArgumentParser(description="Correlation matrix heatmap")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--no-mask", action="store_true", help="Show full matrix")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.test:
        test_solution()
        return

    # Generate data
    data, feature_names = generate_correlated_data(
        n_samples=200,
        n_features=8,
        seed=42,
    )

    # Compute correlation
    corr_matrix = compute_correlation_matrix(data)
    logger.info("Correlation matrix computed")

    # Create heatmap
    fig, ax = create_correlation_heatmap(
        corr_matrix,
        feature_names,
        mask_upper=not args.no_mask,
        output_path=args.output,
    )

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
