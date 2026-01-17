#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6: Visualisation for Research
Pytest Configuration and Shared Fixtures
═══════════════════════════════════════════════════════════════════════════════

This module provides shared pytest fixtures for testing the Week 6 laboratory
modules on static and interactive visualisation.

FIXTURES PROVIDED
─────────────────
- sample_dataframe: A pandas DataFrame with numerical data
- sample_timeseries: Time series data for line plots
- sample_categories: Categorical data for bar charts
- correlation_matrix: Correlation matrix for heatmaps
- wong_palette: The Wong colourblind-friendly palette
- temp_output_dir: Temporary directory for test outputs
- matplotlib_cleanup: Auto-cleanup matplotlib figures

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

# Use non-interactive backend for testing
matplotlib.use("Agg")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing visualisations.

    Returns:
        DataFrame with columns: x, y, error, category, group
    """
    np.random.seed(42)
    n_points = 50

    return pd.DataFrame({
        "x": np.linspace(0, 10, n_points),
        "y": np.sin(np.linspace(0, 2 * np.pi, n_points)) + np.random.normal(0, 0.1, n_points),
        "error": np.random.uniform(0.05, 0.2, n_points),
        "category": np.random.choice(["A", "B", "C"], n_points),
        "group": np.random.choice(["Control", "Treatment"], n_points),
    })


@pytest.fixture
def sample_timeseries() -> pd.DataFrame:
    """Create sample time series data for line plots.

    Returns:
        DataFrame with datetime index and value columns
    """
    np.random.seed(42)
    n_days = 100
    start_date = datetime(2024, 1, 1)

    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    # Simulate research metrics over time
    base_trend = np.linspace(10, 50, n_days)
    seasonal = 10 * np.sin(np.linspace(0, 4 * np.pi, n_days))
    noise = np.random.normal(0, 3, n_days)

    return pd.DataFrame({
        "date": dates,
        "metric_a": base_trend + seasonal + noise,
        "metric_b": base_trend * 0.8 - seasonal * 0.5 + noise * 1.5,
        "metric_c": np.cumsum(np.random.normal(0.5, 1, n_days)),
    })


@pytest.fixture
def sample_categories() -> pd.DataFrame:
    """Create categorical data for bar charts.

    Returns:
        DataFrame with category, value, error and group columns
    """
    return pd.DataFrame({
        "category": ["Physics", "Chemistry", "Biology", "Medicine", "Engineering"],
        "value": [85, 62, 78, 91, 54],
        "error": [5.2, 4.1, 6.3, 3.8, 7.1],
        "group": ["Science", "Science", "Science", "Health", "Engineering"],
    })


@pytest.fixture
def correlation_matrix() -> pd.DataFrame:
    """Create a correlation matrix for heatmap testing.

    Returns:
        DataFrame representing a correlation matrix
    """
    np.random.seed(42)
    n_vars = 6
    var_names = [f"Var_{chr(65 + i)}" for i in range(n_vars)]

    # Generate a valid correlation matrix
    random_matrix = np.random.randn(100, n_vars)
    corr = np.corrcoef(random_matrix.T)

    return pd.DataFrame(corr, index=var_names, columns=var_names)


@pytest.fixture
def scatter_data() -> pd.DataFrame:
    """Create scatter plot data with multiple groups.

    Returns:
        DataFrame suitable for scatter plots
    """
    np.random.seed(42)

    # Create clustered data
    data_list = []
    colours = ["#0072B2", "#E69F00", "#009E73"]
    groups = ["Control", "Treatment A", "Treatment B"]

    for i, (colour, group) in enumerate(zip(colours, groups)):
        n = 30
        x = np.random.normal(i * 2, 0.5, n)
        y = 2 * x + np.random.normal(i, 0.8, n)
        data_list.append(pd.DataFrame({
            "x": x,
            "y": y,
            "group": group,
            "size": np.random.uniform(20, 100, n),
        }))

    return pd.concat(data_list, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COLOUR PALETTE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def wong_palette() -> list[str]:
    """Return the Wong colourblind-friendly palette.

    Returns:
        List of 8 hex colour codes
    """
    return [
        "#000000",  # Black
        "#E69F00",  # Orange
        "#56B4E9",  # Sky blue
        "#009E73",  # Bluish green
        "#F0E442",  # Yellow
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#CC79A7",  # Reddish purple
    ]


@pytest.fixture
def journal_styles() -> dict[str, dict]:
    """Return journal style specifications.

    Returns:
        Dictionary mapping journal names to their style requirements
    """
    return {
        "nature": {
            "width_mm": 89,
            "font_size": 7,
            "font_family": "Arial",
            "dpi": 300,
        },
        "science": {
            "width_mm": 85,
            "font_size": 7,
            "font_family": "Helvetica",
            "dpi": 300,
        },
        "ieee": {
            "width_mm": 88,
            "font_size": 8,
            "font_family": "Times New Roman",
            "dpi": 600,
        },
        "plos": {
            "width_mm": 140,
            "font_size": 10,
            "font_family": "Arial",
            "dpi": 300,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FILE SYSTEM FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs.

    Yields:
        Path to temporary directory (cleaned up after test)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_output_path(temp_output_dir: Path) -> Path:
    """Create a sample output file path.

    Args:
        temp_output_dir: Temporary directory fixture

    Returns:
        Path for test output file
    """
    return temp_output_dir / "test_figure.png"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MATPLOTLIB FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def matplotlib_cleanup() -> Generator[None, None, None]:
    """Automatically close all matplotlib figures after each test.

    This prevents memory leaks and figure state bleeding between tests.
    """
    yield
    plt.close("all")


@pytest.fixture
def figure_and_axes() -> tuple[plt.Figure, plt.Axes]:
    """Create a basic figure and axes for testing.

    Returns:
        Tuple of (Figure, Axes)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    return fig, ax


@pytest.fixture
def multi_panel_figure() -> tuple[plt.Figure, np.ndarray]:
    """Create a 2x2 multi-panel figure for testing.

    Returns:
        Tuple of (Figure, array of Axes)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    return fig, axes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VALIDATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def valid_hex_colours() -> list[str]:
    """Return a list of valid hex colour codes for testing.

    Returns:
        List of valid hex colours
    """
    return ["#FF0000", "#00FF00", "#0000FF", "#FFFFFF", "#000000", "#123ABC"]


@pytest.fixture
def invalid_hex_colours() -> list[str]:
    """Return a list of invalid hex colour codes for testing.

    Returns:
        List of invalid hex colours
    """
    return ["FF0000", "#GGG", "#12345", "red", "rgb(255,0,0)", "#1234567"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PARAMETRISED TEST DATA
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Generate parametrised test cases.

    This hook provides automatic parametrisation for common test patterns.
    """
    # Parametrise figure format tests
    if "output_format" in metafunc.fixturenames:
        metafunc.parametrize("output_format", ["png", "pdf", "svg", "eps"])

    # Parametrise colourmap tests
    if "colormap_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "colormap_name",
            ["viridis", "plasma", "RdBu", "coolwarm", "Greys"],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def assert_figure_properties(
    fig: plt.Figure,
    expected_width: float | None = None,
    expected_height: float | None = None,
    expected_dpi: int | None = None,
) -> None:
    """Assert that a figure has expected properties.

    Args:
        fig: Matplotlib figure to check
        expected_width: Expected width in inches (optional)
        expected_height: Expected height in inches (optional)
        expected_dpi: Expected DPI (optional)

    Raises:
        AssertionError: If any property doesn't match
    """
    if expected_width is not None:
        actual_width = fig.get_figwidth()
        assert abs(actual_width - expected_width) < 0.01, (
            f"Width mismatch: {actual_width} != {expected_width}"
        )

    if expected_height is not None:
        actual_height = fig.get_figheight()
        assert abs(actual_height - expected_height) < 0.01, (
            f"Height mismatch: {actual_height} != {expected_height}"
        )

    if expected_dpi is not None:
        actual_dpi = fig.get_dpi()
        assert actual_dpi == expected_dpi, f"DPI mismatch: {actual_dpi} != {expected_dpi}"


def assert_file_exists_and_valid(path: Path, min_size_bytes: int = 100) -> None:
    """Assert that a file exists and has a minimum size.

    Args:
        path: Path to file
        min_size_bytes: Minimum expected file size

    Raises:
        AssertionError: If file doesn't exist or is too small
    """
    assert path.exists(), f"File does not exist: {path}"
    assert path.stat().st_size >= min_size_bytes, (
        f"File too small: {path.stat().st_size} < {min_size_bytes}"
    )
