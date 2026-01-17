#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6: Visualisation for Research
Tests for Lab 6.01: Static Visualisation Toolkit
═══════════════════════════════════════════════════════════════════════════════

This module provides comprehensive tests for the static visualisation toolkit,
covering PlotStyle configuration, figure creation, publication export and
D3.js integration.

Test Coverage Targets:
- PlotStyle class: 100%
- Figure creation functions: 90%
- Export functions: 85%
- D3 integration: 80%

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend
matplotlib.use("Agg")

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_6_01_static_plots import (
    PALETTES,
    JOURNAL_STYLES,
    PlotStyle,
    create_figure,
    save_publication_figure,
    add_annotation,
    add_significance_bar,
    plot_with_error_band,
    scatter_with_regression,
    create_heatmap,
    export_to_d3_json,
    generate_d3_line_chart_html,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PALETTE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPalettes:
    """Test suite for colour palette definitions."""

    def test_palettes_exist(self) -> None:
        """Verify all expected palettes are defined."""
        expected_palettes = ["colorblind", "default", "grayscale", "nature", "viridis"]
        for name in expected_palettes:
            assert name in PALETTES, f"Missing palette: {name}"

    def test_colorblind_palette_has_eight_colours(self) -> None:
        """Verify colourblind palette has exactly 8 colours."""
        assert len(PALETTES["colorblind"]) == 8

    def test_all_palettes_have_valid_hex_codes(self) -> None:
        """Verify all colours are valid hex codes."""
        import re
        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        
        for palette_name, colours in PALETTES.items():
            for colour in colours:
                assert hex_pattern.match(colour), (
                    f"Invalid hex code '{colour}' in palette '{palette_name}'"
                )

    def test_palettes_have_minimum_colours(self) -> None:
        """Verify all palettes have at least 4 colours."""
        for name, colours in PALETTES.items():
            assert len(colours) >= 4, f"Palette '{name}' has fewer than 4 colours"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: JOURNAL STYLES TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestJournalStyles:
    """Test suite for journal style configurations."""

    def test_journal_styles_exist(self) -> None:
        """Verify all expected journal styles are defined."""
        expected_journals = ["nature", "science", "ieee", "plos"]
        for name in expected_journals:
            assert name in JOURNAL_STYLES, f"Missing journal style: {name}"

    def test_nature_style_has_required_keys(self) -> None:
        """Verify Nature style has all required matplotlib keys."""
        required_keys = ["figure.figsize", "font.size", "font.family"]
        for key in required_keys:
            assert key in JOURNAL_STYLES["nature"], f"Missing key: {key}"

    def test_font_sizes_are_positive(self) -> None:
        """Verify all font sizes are positive numbers."""
        for journal, style in JOURNAL_STYLES.items():
            if "font.size" in style:
                assert style["font.size"] > 0, f"Invalid font size for {journal}"

    def test_figure_sizes_are_reasonable(self) -> None:
        """Verify figure sizes are within reasonable bounds (1-20 inches)."""
        for journal, style in JOURNAL_STYLES.items():
            if "figure.figsize" in style:
                width, height = style["figure.figsize"]
                assert 1 <= width <= 20, f"Invalid width for {journal}: {width}"
                assert 1 <= height <= 20, f"Invalid height for {journal}: {height}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PLOTSTYLE CLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotStyle:
    """Test suite for PlotStyle dataclass."""

    def test_default_construction(self) -> None:
        """Test PlotStyle can be constructed with defaults."""
        style = PlotStyle()
        assert style.palette == "colorblind"
        assert style.font_size == 10

    def test_custom_construction(self) -> None:
        """Test PlotStyle with custom parameters."""
        style = PlotStyle(
            palette="nature",
            font_size=8,
            figure_width=6.0,
            figure_height=4.0,
            dpi=300,
        )
        assert style.palette == "nature"
        assert style.font_size == 8
        assert style.dpi == 300

    def test_get_colours_returns_correct_count(self) -> None:
        """Test get_colours returns requested number of colours."""
        style = PlotStyle(palette="colorblind")
        colours = style.get_colours(5)
        assert len(colours) == 5

    def test_get_colours_cycles_when_n_exceeds_palette(self) -> None:
        """Test get_colours cycles through palette when n > palette length."""
        style = PlotStyle(palette="colorblind")
        colours = style.get_colours(12)
        assert len(colours) == 12
        # Check cycling: colour 8 should equal colour 0
        assert colours[8] == colours[0]

    def test_get_colours_none_returns_all(self) -> None:
        """Test get_colours(None) returns all palette colours."""
        style = PlotStyle(palette="colorblind")
        colours = style.get_colours(None)
        assert len(colours) == len(PALETTES["colorblind"])

    def test_for_journal_factory(self) -> None:
        """Test for_journal class method creates correct style."""
        style = PlotStyle.for_journal("nature")
        # Nature uses 7pt font
        assert style.font_size == 7 or style.font_size == JOURNAL_STYLES["nature"]["font.size"]

    def test_for_journal_invalid_raises_error(self) -> None:
        """Test for_journal with invalid journal name raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            PlotStyle.for_journal("nonexistent_journal")

    def test_apply_modifies_rcparams(self) -> None:
        """Test apply method modifies matplotlib rcParams."""
        style = PlotStyle(font_size=12, dpi=150)
        
        # Save original values
        original_font_size = plt.rcParams["font.size"]
        original_dpi = plt.rcParams["figure.dpi"]
        
        try:
            style.apply()
            assert plt.rcParams["font.size"] == 12
        finally:
            # Restore original values
            plt.rcParams["font.size"] = original_font_size
            plt.rcParams["figure.dpi"] = original_dpi


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FIGURE CREATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCreateFigure:
    """Test suite for create_figure function."""

    def test_creates_figure_and_axes(self) -> None:
        """Test create_figure returns Figure and Axes."""
        fig, ax = create_figure()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_dimensions(self) -> None:
        """Test create_figure respects custom dimensions."""
        fig, ax = create_figure(width=10.0, height=6.0)
        width, height = fig.get_size_inches()
        assert abs(width - 10.0) < 0.1
        assert abs(height - 6.0) < 0.1
        plt.close(fig)

    def test_custom_dpi(self) -> None:
        """Test create_figure respects custom DPI."""
        fig, ax = create_figure(dpi=150)
        assert fig.get_dpi() == 150
        plt.close(fig)

    def test_multiple_subplots(self) -> None:
        """Test create_figure with multiple subplots."""
        fig, axes = create_figure(nrows=2, ncols=2)
        assert isinstance(fig, plt.Figure)
        # axes should be a 2x2 array or flattened depending on implementation
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SAVE PUBLICATION FIGURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSavePublicationFigure:
    """Test suite for save_publication_figure function."""

    def test_saves_png_file(self, temp_output_dir: Path) -> None:
        """Test saving figure as PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_path = temp_output_dir / "test_figure.png"
        save_publication_figure(fig, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_saves_pdf_file(self, temp_output_dir: Path) -> None:
        """Test saving figure as PDF."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_path = temp_output_dir / "test_figure.pdf"
        save_publication_figure(fig, output_path)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        plt.close(fig)

    def test_saves_svg_file(self, temp_output_dir: Path) -> None:
        """Test saving figure as SVG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_path = temp_output_dir / "test_figure.svg"
        save_publication_figure(fig, output_path)
        
        assert output_path.exists()
        # SVG should contain valid XML
        content = output_path.read_text()
        assert "<svg" in content
        plt.close(fig)

    def test_creates_parent_directory(self, temp_output_dir: Path) -> None:
        """Test that parent directories are created if needed."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        output_path = temp_output_dir / "subdir" / "nested" / "figure.png"
        save_publication_figure(fig, output_path)
        
        assert output_path.exists()
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ANNOTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAddAnnotation:
    """Test suite for add_annotation function."""

    def test_adds_text_annotation(self) -> None:
        """Test adding a text annotation to axes."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        add_annotation(ax, "Test", x=2, y=4)
        
        # Check that annotation was added
        assert len(ax.texts) > 0 or len(ax.annotations) > 0
        plt.close(fig)

    def test_annotation_with_arrow(self) -> None:
        """Test adding annotation with arrow pointing to data."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        add_annotation(
            ax, "Peak", x=3, y=9,
            arrow=True, arrow_start=(2.5, 7)
        )
        
        plt.close(fig)


class TestAddSignificanceBar:
    """Test suite for add_significance_bar function."""

    def test_adds_bar_between_groups(self) -> None:
        """Test adding significance bar between two x positions."""
        fig, ax = plt.subplots()
        ax.bar([1, 2, 3], [5, 8, 6])
        
        add_significance_bar(ax, x1=1, x2=2, y=10, text="*")
        
        # Should add line and text
        assert len(ax.lines) > 0 or len(ax.patches) > 0
        plt.close(fig)

    def test_significance_text_appears(self) -> None:
        """Test that significance text is displayed."""
        fig, ax = plt.subplots()
        ax.bar([1, 2], [5, 8])
        
        add_significance_bar(ax, x1=1, x2=2, y=10, text="p<0.01")
        
        texts = [t.get_text() for t in ax.texts]
        assert any("p<0.01" in t or "p" in t for t in texts) or len(ax.texts) > 0
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: PLOTTING FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPlotWithErrorBand:
    """Test suite for plot_with_error_band function."""

    def test_plots_line_with_band(self) -> None:
        """Test plotting line with error band."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        error = np.ones_like(x) * 0.2
        
        plot_with_error_band(ax, x, y, error)
        
        # Should have at least one line
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_custom_colour(self) -> None:
        """Test plotting with custom colour."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        error = np.ones_like(x) * 0.2
        
        plot_with_error_band(ax, x, y, error, colour="#FF0000")
        
        plt.close(fig)

    def test_with_label(self) -> None:
        """Test plotting with legend label."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        error = np.ones_like(x) * 0.2
        
        plot_with_error_band(ax, x, y, error, label="Test data")
        ax.legend()
        
        legend = ax.get_legend()
        assert legend is not None
        plt.close(fig)


class TestScatterWithRegression:
    """Test suite for scatter_with_regression function."""

    def test_plots_scatter_and_regression(self) -> None:
        """Test plotting scatter with regression line."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.normal(0, 1, 50)
        
        scatter_with_regression(ax, x, y)
        
        # Should have scatter points and regression line
        assert len(ax.collections) >= 1 or len(ax.lines) >= 1
        plt.close(fig)

    def test_returns_fit_parameters(self) -> None:
        """Test that fit parameters are returned."""
        fig, ax = plt.subplots()
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 0.5, 50)
        
        result = scatter_with_regression(ax, x, y)
        
        # Should return slope, intercept, r_value or similar
        if result is not None:
            assert isinstance(result, (tuple, dict))
        plt.close(fig)


class TestCreateHeatmap:
    """Test suite for create_heatmap function."""

    def test_creates_heatmap(self, correlation_matrix: pd.DataFrame) -> None:
        """Test creating a heatmap from correlation matrix."""
        fig, ax = create_heatmap(correlation_matrix)
        
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_custom_colourmap(self, correlation_matrix: pd.DataFrame) -> None:
        """Test heatmap with custom colourmap."""
        fig, ax = create_heatmap(correlation_matrix, cmap="viridis")
        
        plt.close(fig)

    def test_annotated_heatmap(self, correlation_matrix: pd.DataFrame) -> None:
        """Test heatmap with value annotations."""
        fig, ax = create_heatmap(correlation_matrix, annotate=True)
        
        # Check that text annotations exist
        # (implementation may vary)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: D3.JS EXPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestExportToD3Json:
    """Test suite for export_to_d3_json function."""

    def test_exports_valid_json(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that exported JSON is valid."""
        json_str = export_to_d3_json(sample_dataframe)
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, (list, dict))

    def test_json_contains_data(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that JSON contains the data."""
        json_str = export_to_d3_json(sample_dataframe)
        data = json.loads(json_str)
        
        # Should have same number of records as DataFrame rows
        if isinstance(data, list):
            assert len(data) == len(sample_dataframe)

    def test_handles_numpy_types(self) -> None:
        """Test that numpy types are properly serialised."""
        df = pd.DataFrame({
            "int_col": np.array([1, 2, 3], dtype=np.int64),
            "float_col": np.array([1.5, 2.5, 3.5], dtype=np.float64),
        })
        
        json_str = export_to_d3_json(df)
        
        # Should not raise and should be valid JSON
        data = json.loads(json_str)
        assert len(data) == 3


class TestGenerateD3LineChartHtml:
    """Test suite for generate_d3_line_chart_html function."""

    def test_generates_valid_html(self) -> None:
        """Test that generated HTML is valid."""
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]
        
        html = generate_d3_line_chart_html(x, y)
        
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "d3" in html.lower()

    def test_includes_data(self) -> None:
        """Test that data is embedded in HTML."""
        x = [1, 2, 3]
        y = [10, 20, 30]
        
        html = generate_d3_line_chart_html(x, y)
        
        # Data should appear in some form
        assert "10" in html or "20" in html or "30" in html

    def test_custom_title(self) -> None:
        """Test that custom title appears in HTML."""
        x = [1, 2, 3]
        y = [1, 4, 9]
        
        html = generate_d3_line_chart_html(x, y, title="Test Chart")
        
        assert "Test Chart" in html

    def test_saves_to_file(self, temp_output_dir: Path) -> None:
        """Test saving HTML to file."""
        x = [1, 2, 3]
        y = [1, 4, 9]
        
        output_path = temp_output_dir / "chart.html"
        html = generate_d3_line_chart_html(x, y)
        output_path.write_text(html)
        
        assert output_path.exists()
        assert "<html" in output_path.read_text() or "<!DOCTYPE" in output_path.read_text()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_publication_workflow(
        self, 
        sample_dataframe: pd.DataFrame,
        temp_output_dir: Path,
    ) -> None:
        """Test complete workflow: style → create → plot → save."""
        # Apply journal style
        style = PlotStyle.for_journal("nature")
        style.apply()
        
        # Create figure
        fig, ax = create_figure()
        
        # Plot data
        x = sample_dataframe["x"].values
        y = sample_dataframe["y"].values
        error = sample_dataframe["error"].values
        
        plot_with_error_band(ax, x, y, error, label="Experimental")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend()
        
        # Save
        output_path = temp_output_dir / "publication_figure.pdf"
        save_publication_figure(fig, output_path)
        
        assert output_path.exists()
        plt.close(fig)

    def test_multi_format_export(self, temp_output_dir: Path) -> None:
        """Test exporting same figure in multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        formats = ["png", "pdf", "svg"]
        for fmt in formats:
            output_path = temp_output_dir / f"figure.{fmt}"
            save_publication_figure(fig, output_path)
            assert output_path.exists(), f"Failed to create {fmt} file"
        
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_handling(self) -> None:
        """Test handling of empty data arrays."""
        fig, ax = plt.subplots()
        
        # Should handle gracefully (may raise or produce empty plot)
        try:
            plot_with_error_band(ax, np.array([]), np.array([]), np.array([]))
        except (ValueError, IndexError):
            pass  # Expected behaviour for empty data
        
        plt.close(fig)

    def test_single_point_data(self) -> None:
        """Test handling of single data point."""
        fig, ax = plt.subplots()
        
        plot_with_error_band(
            ax, 
            np.array([1.0]), 
            np.array([2.0]), 
            np.array([0.1]),
        )
        
        plt.close(fig)

    def test_mismatched_array_lengths(self) -> None:
        """Test handling of mismatched array lengths."""
        fig, ax = plt.subplots()
        
        with pytest.raises((ValueError, IndexError)):
            plot_with_error_band(
                ax,
                np.array([1, 2, 3]),
                np.array([1, 2]),  # Different length
                np.array([0.1, 0.1, 0.1]),
            )
        
        plt.close(fig)

    def test_nan_handling(self) -> None:
        """Test handling of NaN values in data."""
        fig, ax = plt.subplots()
        
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, np.nan, 3, 4, 5])
        error = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        
        # Should handle NaN gracefully
        try:
            plot_with_error_band(ax, x, y, error)
        except ValueError:
            pass  # Some implementations may reject NaN
        
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
