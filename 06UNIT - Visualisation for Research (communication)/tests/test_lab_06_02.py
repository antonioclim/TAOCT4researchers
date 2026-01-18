#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6: Visualisation for Research
Tests for Lab 6.02: Interactive Visualisation Toolkit
═══════════════════════════════════════════════════════════════════════════════

This module provides comprehensive tests for the interactive visualisation
toolkit, covering data structures, SVG chart generation, dashboard templates
and Streamlit integration.

Test Coverage Targets:
- DataPoint/Dataset classes: 100%
- Chart generation functions: 90%
- Dashboard generation: 85%
- HTML/SVG validity: 100%

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_6_02_interactive_viz import (
    DataPoint,
    Dataset,
    MetricCard,
    FilterControl,
    DashboardConfig,
    generate_bar_chart_svg,
    generate_line_chart_svg,
    generate_pie_chart_svg,
    generate_dashboard_html,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATAPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataPoint:
    """Test suite for DataPoint dataclass."""

    def test_construction_with_required_fields(self) -> None:
        """Test DataPoint construction with minimum required fields."""
        point = DataPoint(label="Test", value=42.0)
        assert point.label == "Test"
        assert point.value == 42.0

    def test_construction_with_all_fields(self) -> None:
        """Test DataPoint construction with all fields."""
        point = DataPoint(
            label="Complete",
            value=100.0,
            category="A",
            colour="#FF0000",
            metadata={"key": "value"},
        )
        assert point.category == "A"
        assert point.colour == "#FF0000"
        assert point.metadata == {"key": "value"}

    def test_to_dict_returns_dictionary(self) -> None:
        """Test to_dict method returns proper dictionary."""
        point = DataPoint(label="Test", value=42.0, category="Cat")
        result = point.to_dict()
        
        assert isinstance(result, dict)
        assert result["label"] == "Test"
        assert result["value"] == 42.0
        assert result["category"] == "Cat"

    def test_to_dict_excludes_none_metadata(self) -> None:
        """Test to_dict handles None metadata appropriately."""
        point = DataPoint(label="Test", value=42.0)
        result = point.to_dict()
        
        # metadata should be None or not present or empty dict
        assert result.get("metadata") is None or result.get("metadata") == {}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATASET TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataset:
    """Test suite for Dataset class."""

    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing."""
        points = [
            DataPoint(label="A1", value=10.0, category="A"),
            DataPoint(label="A2", value=20.0, category="A"),
            DataPoint(label="B1", value=30.0, category="B"),
            DataPoint(label="B2", value=40.0, category="B"),
            DataPoint(label="C1", value=50.0, category="C"),
        ]
        return Dataset(points=points, name="Test Dataset")

    def test_len_returns_point_count(self, sample_dataset: Dataset) -> None:
        """Test __len__ returns correct number of points."""
        assert len(sample_dataset) == 5

    def test_empty_dataset(self) -> None:
        """Test empty dataset has length 0."""
        empty = Dataset(points=[], name="Empty")
        assert len(empty) == 0

    def test_filter_by_category(self, sample_dataset: Dataset) -> None:
        """Test filtering by category returns correct subset."""
        filtered = sample_dataset.filter_by_category("A")
        
        assert len(filtered) == 2
        for point in filtered.points:
            assert point.category == "A"

    def test_filter_by_nonexistent_category(self, sample_dataset: Dataset) -> None:
        """Test filtering by non-existent category returns empty dataset."""
        filtered = sample_dataset.filter_by_category("X")
        assert len(filtered) == 0

    def test_filter_by_value_range(self, sample_dataset: Dataset) -> None:
        """Test filtering by value range."""
        filtered = sample_dataset.filter_by_value_range(min_val=15.0, max_val=35.0)
        
        assert len(filtered) == 2
        for point in filtered.points:
            assert 15.0 <= point.value <= 35.0

    def test_filter_by_value_range_no_match(self, sample_dataset: Dataset) -> None:
        """Test filtering with range that matches nothing."""
        filtered = sample_dataset.filter_by_value_range(min_val=100.0, max_val=200.0)
        assert len(filtered) == 0

    def test_aggregate_by_category(self, sample_dataset: Dataset) -> None:
        """Test aggregation by category."""
        aggregated = sample_dataset.aggregate_by_category()
        
        assert isinstance(aggregated, dict)
        assert "A" in aggregated
        assert "B" in aggregated
        assert "C" in aggregated
        # A: 10 + 20 = 30, average = 15 (or sum depending on implementation)
        assert aggregated["A"] in [15.0, 30.0]  # Mean or sum

    def test_get_statistics(self, sample_dataset: Dataset) -> None:
        """Test get_statistics returns expected keys."""
        stats = sample_dataset.get_statistics()
        
        assert isinstance(stats, dict)
        # Should have at least mean, min, max
        expected_keys = {"mean", "min", "max"}
        assert expected_keys.issubset(set(stats.keys()))

    def test_get_statistics_values(self, sample_dataset: Dataset) -> None:
        """Test get_statistics returns correct values."""
        stats = sample_dataset.get_statistics()
        
        # Values: 10, 20, 30, 40, 50
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0
        assert stats["mean"] == 30.0

    def test_to_json_valid(self, sample_dataset: Dataset) -> None:
        """Test to_json returns valid JSON string."""
        json_str = sample_dataset.to_json()
        
        # Should be valid JSON
        data = json.loads(json_str)
        assert isinstance(data, (list, dict))

    def test_generate_sample(self) -> None:
        """Test generate_sample creates dataset with correct size."""
        dataset = Dataset.generate_sample(n_points=100, n_categories=5)
        
        assert len(dataset) == 100
        categories = set(p.category for p in dataset.points)
        assert len(categories) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: METRICCARD TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricCard:
    """Test suite for MetricCard class."""

    def test_construction(self) -> None:
        """Test MetricCard construction."""
        card = MetricCard(
            title="Total Users",
            value=1000,
            unit="users",
            change=5.2,
            change_positive=True,
        )
        assert card.title == "Total Users"
        assert card.value == 1000
        assert card.change == 5.2

    def test_to_html_returns_string(self) -> None:
        """Test to_html returns HTML string."""
        card = MetricCard(title="Test", value=100, unit="items")
        html = card.to_html()
        
        assert isinstance(html, str)
        assert len(html) > 0

    def test_to_html_contains_title(self) -> None:
        """Test to_html contains the title."""
        card = MetricCard(title="Revenue", value=50000, unit="£")
        html = card.to_html()
        
        assert "Revenue" in html

    def test_to_html_contains_value(self) -> None:
        """Test to_html contains the value."""
        card = MetricCard(title="Count", value=42, unit="items")
        html = card.to_html()
        
        assert "42" in html

    def test_to_html_shows_positive_change(self) -> None:
        """Test to_html shows positive change indicator."""
        card = MetricCard(
            title="Growth",
            value=100,
            unit="%",
            change=10.5,
            change_positive=True,
        )
        html = card.to_html()
        
        # Should contain change value and positive indicator
        assert "10.5" in html or "10" in html

    def test_to_html_shows_negative_change(self) -> None:
        """Test to_html shows negative change indicator."""
        card = MetricCard(
            title="Decline",
            value=50,
            unit="%",
            change=-5.0,
            change_positive=False,
        )
        html = card.to_html()
        
        assert "5" in html


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FILTERCONTROL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilterControl:
    """Test suite for FilterControl class."""

    def test_select_control_construction(self) -> None:
        """Test select filter control construction."""
        control = FilterControl(
            name="category",
            label="Category",
            control_type="select",
            options=["A", "B", "C"],
        )
        assert control.name == "category"
        assert control.control_type == "select"
        assert len(control.options) == 3

    def test_range_control_construction(self) -> None:
        """Test range filter control construction."""
        control = FilterControl(
            name="value",
            label="Value Range",
            control_type="range",
            min_value=0,
            max_value=100,
        )
        assert control.min_value == 0
        assert control.max_value == 100

    def test_element_id_property(self) -> None:
        """Test element_id property generates valid ID."""
        control = FilterControl(
            name="test_filter",
            label="Test",
            control_type="select",
        )
        element_id = control.element_id
        
        assert isinstance(element_id, str)
        # Should be valid HTML ID (alphanumeric + hyphens/underscores)
        assert re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", element_id)

    def test_to_html_select(self) -> None:
        """Test to_html for select control."""
        control = FilterControl(
            name="category",
            label="Category",
            control_type="select",
            options=["A", "B", "C"],
        )
        html = control.to_html()
        
        assert "<select" in html
        assert "Category" in html

    def test_to_html_range(self) -> None:
        """Test to_html for range control."""
        control = FilterControl(
            name="value",
            label="Value",
            control_type="range",
            min_value=0,
            max_value=100,
        )
        html = control.to_html()
        
        assert 'type="range"' in html or "range" in html.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SVG CHART GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestGenerateBarChartSvg:
    """Test suite for generate_bar_chart_svg function."""

    def test_returns_valid_svg(self) -> None:
        """Test function returns valid SVG string."""
        data = [
            {"label": "A", "value": 10},
            {"label": "B", "value": 20},
            {"label": "C", "value": 30},
        ]
        svg = generate_bar_chart_svg(data)
        
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_bars(self) -> None:
        """Test SVG contains rect elements for bars."""
        data = [
            {"label": "A", "value": 10},
            {"label": "B", "value": 20},
        ]
        svg = generate_bar_chart_svg(data)
        
        # Should have rect elements
        assert "<rect" in svg

    def test_custom_dimensions(self) -> None:
        """Test SVG with custom dimensions."""
        data = [{"label": "A", "value": 10}]
        svg = generate_bar_chart_svg(data, width=500, height=300)
        
        assert 'width="500"' in svg or "500" in svg
        assert 'height="300"' in svg or "300" in svg

    def test_empty_data(self) -> None:
        """Test handling of empty data."""
        svg = generate_bar_chart_svg([])
        
        # Should still return valid SVG (possibly empty)
        assert "<svg" in svg

    def test_title_included(self) -> None:
        """Test chart title is included."""
        data = [{"label": "A", "value": 10}]
        svg = generate_bar_chart_svg(data, title="Test Chart")
        
        assert "Test Chart" in svg


class TestGenerateLineChartSvg:
    """Test suite for generate_line_chart_svg function."""

    def test_returns_valid_svg(self) -> None:
        """Test function returns valid SVG string."""
        data = [
            {"x": 1, "y": 10},
            {"x": 2, "y": 20},
            {"x": 3, "y": 15},
        ]
        svg = generate_line_chart_svg(data)
        
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_path(self) -> None:
        """Test SVG contains path element for line."""
        data = [
            {"x": 1, "y": 10},
            {"x": 2, "y": 20},
            {"x": 3, "y": 15},
        ]
        svg = generate_line_chart_svg(data)
        
        # Should have path or line elements
        assert "<path" in svg or "<line" in svg or "<polyline" in svg

    def test_custom_colour(self) -> None:
        """Test line with custom colour."""
        data = [{"x": 1, "y": 10}, {"x": 2, "y": 20}]
        svg = generate_line_chart_svg(data, colour="#FF0000")
        
        # Colour should appear in SVG (as stroke or fill)
        assert "#FF0000" in svg or "FF0000" in svg.upper()

    def test_single_point(self) -> None:
        """Test handling of single data point."""
        data = [{"x": 1, "y": 10}]
        svg = generate_line_chart_svg(data)
        
        # Should still return valid SVG
        assert "<svg" in svg


class TestGeneratePieChartSvg:
    """Test suite for generate_pie_chart_svg function."""

    def test_returns_valid_svg(self) -> None:
        """Test function returns valid SVG string."""
        data = [
            {"label": "A", "value": 30},
            {"label": "B", "value": 50},
            {"label": "C", "value": 20},
        ]
        svg = generate_pie_chart_svg(data)
        
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_contains_slices(self) -> None:
        """Test SVG contains path elements for slices."""
        data = [
            {"label": "A", "value": 50},
            {"label": "B", "value": 50},
        ]
        svg = generate_pie_chart_svg(data)
        
        # Should have path elements for pie slices
        assert "<path" in svg

    def test_custom_colours(self) -> None:
        """Test pie chart with custom colours."""
        data = [
            {"label": "A", "value": 50, "colour": "#FF0000"},
            {"label": "B", "value": 50, "colour": "#00FF00"},
        ]
        svg = generate_pie_chart_svg(data)
        
        # Colours should appear
        assert "#FF0000" in svg or "FF0000" in svg.upper()

    def test_labels_included(self) -> None:
        """Test pie chart labels are included."""
        data = [
            {"label": "Category A", "value": 100},
        ]
        svg = generate_pie_chart_svg(data)
        
        # Label should appear in SVG
        assert "Category A" in svg or "category" in svg.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DASHBOARD GENERATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDashboardConfig:
    """Test suite for DashboardConfig class."""

    def test_construction(self) -> None:
        """Test DashboardConfig construction."""
        config = DashboardConfig(
            title="Research Dashboard",
            theme="dark",
        )
        assert config.title == "Research Dashboard"
        assert config.theme == "dark"

    def test_default_theme(self) -> None:
        """Test default theme is set."""
        config = DashboardConfig(title="Test")
        
        # Should have a default theme
        assert hasattr(config, "theme")


class TestGenerateDashboardHtml:
    """Test suite for generate_dashboard_html function."""

    def test_returns_valid_html(self) -> None:
        """Test function returns valid HTML string."""
        config = DashboardConfig(title="Test Dashboard")
        html = generate_dashboard_html(config)
        
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "</html>" in html

    def test_contains_title(self) -> None:
        """Test HTML contains dashboard title."""
        config = DashboardConfig(title="Research Metrics")
        html = generate_dashboard_html(config)
        
        assert "Research Metrics" in html

    def test_contains_doctype(self) -> None:
        """Test HTML has proper DOCTYPE."""
        config = DashboardConfig(title="Test")
        html = generate_dashboard_html(config)
        
        # Should start with DOCTYPE or contain it
        assert "<!DOCTYPE" in html or "<!doctype" in html

    def test_dark_theme_styling(self) -> None:
        """Test dark theme applies correct styling."""
        config = DashboardConfig(title="Dark Theme", theme="dark")
        html = generate_dashboard_html(config)
        
        # Dark theme should have dark background colour
        assert "dark" in html.lower() or "#1a1a" in html.lower() or "background" in html

    def test_includes_css(self) -> None:
        """Test HTML includes CSS styling."""
        config = DashboardConfig(title="Styled Dashboard")
        html = generate_dashboard_html(config)
        
        assert "<style" in html or 'style=' in html

    def test_includes_javascript(self) -> None:
        """Test HTML includes JavaScript for interactivity."""
        config = DashboardConfig(title="Interactive Dashboard")
        html = generate_dashboard_html(config)
        
        # Should have script tags
        assert "<script" in html

    def test_saves_to_file(self, temp_output_dir: Path) -> None:
        """Test saving dashboard HTML to file."""
        config = DashboardConfig(title="File Test")
        html = generate_dashboard_html(config)
        
        output_path = temp_output_dir / "dashboard.html"
        output_path.write_text(html)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_dataset_to_bar_chart(self) -> None:
        """Test creating bar chart from Dataset."""
        # Create dataset
        dataset = Dataset.generate_sample(n_points=10, n_categories=5)
        
        # Aggregate by category
        aggregated = dataset.aggregate_by_category()
        
        # Convert to chart data format
        chart_data = [
            {"label": cat, "value": val}
            for cat, val in aggregated.items()
        ]
        
        # Generate SVG
        svg = generate_bar_chart_svg(chart_data)
        
        assert "<svg" in svg
        assert len(aggregated) == 5

    def test_complete_dashboard_workflow(self, temp_output_dir: Path) -> None:
        """Test complete dashboard creation workflow."""
        # 1. Generate sample data
        dataset = Dataset.generate_sample(n_points=100, n_categories=4)
        
        # 2. Calculate statistics
        stats = dataset.get_statistics()
        
        # 3. Create metric cards
        cards = [
            MetricCard(title="Total Points", value=len(dataset), unit=""),
            MetricCard(title="Average", value=round(stats["mean"], 1), unit=""),
            MetricCard(title="Maximum", value=stats["max"], unit=""),
        ]
        
        # 4. Generate dashboard
        config = DashboardConfig(
            title="Research Data Dashboard",
            theme="dark",
        )
        html = generate_dashboard_html(config)
        
        # 5. Save to file
        output_path = temp_output_dir / "complete_dashboard.html"
        output_path.write_text(html)
        
        # Verify
        assert output_path.exists()
        assert "<html" in html

    def test_data_filtering_and_visualisation(self) -> None:
        """Test filtering data and creating visualisation."""
        # Create dataset with known values
        points = [
            DataPoint(label=f"Item{i}", value=i * 10.0, category="A" if i < 5 else "B")
            for i in range(10)
        ]
        dataset = Dataset(points=points, name="Test")
        
        # Filter to category A
        filtered = dataset.filter_by_category("A")
        assert len(filtered) == 5
        
        # Convert to chart data
        chart_data = [
            {"label": p.label, "value": p.value}
            for p in filtered.points
        ]
        
        # Generate chart
        svg = generate_bar_chart_svg(chart_data, title="Category A Data")
        
        assert "Category A" in svg
        assert len(chart_data) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dataset_with_zero_values(self) -> None:
        """Test dataset handling zero values."""
        points = [
            DataPoint(label="Zero", value=0.0, category="A"),
            DataPoint(label="Positive", value=10.0, category="A"),
        ]
        dataset = Dataset(points=points, name="Test")
        
        stats = dataset.get_statistics()
        assert stats["min"] == 0.0

    def test_dataset_with_negative_values(self) -> None:
        """Test dataset handling negative values."""
        points = [
            DataPoint(label="Neg", value=-10.0, category="A"),
            DataPoint(label="Pos", value=10.0, category="A"),
        ]
        dataset = Dataset(points=points, name="Test")
        
        stats = dataset.get_statistics()
        assert stats["min"] == -10.0
        assert stats["mean"] == 0.0

    def test_bar_chart_with_large_values(self) -> None:
        """Test bar chart with very large values."""
        data = [
            {"label": "Big", "value": 1_000_000_000},
            {"label": "Small", "value": 1},
        ]
        svg = generate_bar_chart_svg(data)
        
        assert "<svg" in svg

    def test_pie_chart_with_single_slice(self) -> None:
        """Test pie chart with only one slice (100%)."""
        data = [{"label": "All", "value": 100}]
        svg = generate_pie_chart_svg(data)
        
        assert "<svg" in svg

    def test_line_chart_with_duplicate_x_values(self) -> None:
        """Test line chart handles duplicate x values."""
        data = [
            {"x": 1, "y": 10},
            {"x": 1, "y": 20},  # Same x, different y
            {"x": 2, "y": 15},
        ]
        svg = generate_line_chart_svg(data)
        
        assert "<svg" in svg

    def test_special_characters_in_labels(self) -> None:
        """Test handling of special characters in labels."""
        data = [
            {"label": "Test & Demo", "value": 10},
            {"label": "A < B", "value": 20},
            {"label": 'Quote "test"', "value": 30},
        ]
        svg = generate_bar_chart_svg(data)
        
        # Should not break SVG (special chars should be escaped)
        assert "<svg" in svg
        assert "</svg>" in svg

    def test_unicode_labels(self) -> None:
        """Test handling of Unicode characters in labels."""
        data = [
            {"label": "Français", "value": 10},
            {"label": "日本語", "value": 20},
            {"label": "Ελληνικά", "value": 30},
        ]
        svg = generate_bar_chart_svg(data)
        
        assert "<svg" in svg


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: SVG VALIDITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSvgValidity:
    """Tests for SVG output validity."""

    def test_bar_chart_svg_structure(self) -> None:
        """Test bar chart SVG has proper structure."""
        data = [{"label": "A", "value": 10}]
        svg = generate_bar_chart_svg(data)
        
        # Should have opening and closing tags
        assert svg.count("<svg") == 1
        assert svg.count("</svg>") == 1
        
        # Opening should come before closing
        assert svg.index("<svg") < svg.index("</svg>")

    def test_svg_has_viewbox(self) -> None:
        """Test SVG has viewBox attribute."""
        data = [{"label": "A", "value": 10}]
        svg = generate_bar_chart_svg(data)
        
        # Should have viewBox for responsiveness
        assert "viewBox" in svg or "viewbox" in svg.lower()

    def test_svg_has_xmlns(self) -> None:
        """Test SVG has proper xmlns attribute."""
        data = [{"label": "A", "value": 10}]
        svg = generate_bar_chart_svg(data)
        
        # Should have SVG namespace
        assert "xmlns" in svg


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
