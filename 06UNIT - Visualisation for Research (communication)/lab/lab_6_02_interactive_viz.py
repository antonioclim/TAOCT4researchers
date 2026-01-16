#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Lab 2: Interactive Dashboards
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Interactive dashboards enable dynamic data exploration that static figures
cannot support. This laboratory demonstrates dashboard creation using:

- Pure Python/HTML generation (no external servers required)
- Streamlit (rapid prototyping with Python)
- D3.js (custom interactive visualisations)

We focus on the core patterns: filtering, brushing, linked views, and
details-on-demand that make exploratory data analysis effective.

PREREQUISITES
─────────────
- Week 6 Lab 1: Static Visualisation Toolkit
- Python: Intermediate level with data structures
- Basic HTML/CSS/JavaScript knowledge

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Design dashboard layouts with effective information hierarchy
2. Implement interactive filtering and linked views
3. Generate standalone HTML dashboards
4. Create Streamlit applications for data exploration

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 120 minutes
- Total: 2.5 hours

DEPENDENCIES
────────────
- No required dependencies (core functionality)
- Optional: streamlit, plotly, pandas for enhanced features

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataPoint:
    """
    A single data point for dashboard visualisation.
    
    Attributes:
        timestamp: ISO format date string.
        category: Categorical grouping variable.
        value: Numeric value.
        metadata: Additional key-value pairs.
    """
    timestamp: str
    category: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            'timestamp': self.timestamp,
            'category': self.category,
            'value': self.value,
            **self.metadata
        }


@dataclass
class Dataset:
    """
    Container for dashboard data with filtering capabilities.
    
    Attributes:
        name: Descriptive name for the dataset.
        points: List of DataPoint instances.
    """
    name: str
    points: list[DataPoint] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.points)
    
    def filter_by_category(self, category: str) -> 'Dataset':
        """
        Filter dataset to include only points matching the given category.
        
        Args:
            category: Category value to filter by.
            
        Returns:
            New Dataset containing only matching points.
        """
        filtered = [p for p in self.points if p.category == category]
        return Dataset(f"{self.name}_filtered", filtered)
    
    def filter_by_value_range(
        self, 
        min_val: float, 
        max_val: float
    ) -> 'Dataset':
        """
        Filter dataset to include only points within value range.
        
        Args:
            min_val: Minimum value (inclusive).
            max_val: Maximum value (inclusive).
            
        Returns:
            New Dataset containing only points within range.
        """
        filtered = [p for p in self.points if min_val <= p.value <= max_val]
        return Dataset(f"{self.name}_range", filtered)
    
    def aggregate_by_category(self) -> dict[str, float]:
        """
        Calculate sum of values grouped by category.
        
        Returns:
            Dictionary mapping category names to total values.
        """
        aggregation: dict[str, float] = {}
        for point in self.points:
            aggregation[point.category] = (
                aggregation.get(point.category, 0) + point.value
            )
        return aggregation
    
    def get_statistics(self) -> dict[str, float]:
        """
        Calculate basic descriptive statistics.
        
        Returns:
            Dictionary with count, mean, min, max, std.
        """
        if not self.points:
            return {'count': 0, 'mean': 0, 'min': 0, 'max': 0, 'std': 0}
        
        values = [p.value for p in self.points]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        
        return {
            'count': n,
            'mean': mean,
            'min': min(values),
            'max': max(values),
            'std': math.sqrt(variance),
        }
    
    def to_json(self) -> str:
        """Export dataset as JSON string."""
        return json.dumps([p.to_dict() for p in self.points], indent=2)
    
    @classmethod
    def generate_sample(
        cls, 
        n_points: int = 100, 
        seed: int = 42
    ) -> 'Dataset':
        """
        Generate a sample dataset for testing and demonstration.
        
        Args:
            n_points: Number of data points to generate.
            seed: Random seed for reproducibility.
            
        Returns:
            Dataset with randomly generated points.
        """
        random.seed(seed)
        categories = ['Category A', 'Category B', 'Category C', 'Category D']
        points = []
        
        for i in range(n_points):
            point = DataPoint(
                timestamp=f"2025-01-{(i % 28) + 1:02d}",
                category=random.choice(categories),
                value=random.gauss(50, 15),
                metadata={
                    'index': i,
                    'quality': random.choice(['High', 'Medium', 'Low'])
                }
            )
            points.append(point)
        
        logger.info(f"Generated sample dataset with {n_points} points")
        return cls("sample_data", points)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DASHBOARD COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricCard:
    """
    A card component displaying a key metric with optional delta.
    
    Attributes:
        title: Card title/label.
        value: Primary metric value (can be number or formatted string).
        delta: Optional change indicator (percentage).
        delta_colour: Colour for positive delta ('green' or custom hex).
    """
    title: str
    value: float | int | str
    delta: float | None = None
    delta_colour: str = "#3fb950"  # Green
    
    def to_html(self) -> str:
        """Generate HTML representation of the metric card."""
        delta_html = ""
        if self.delta is not None:
            sign = "+" if self.delta >= 0 else ""
            colour = self.delta_colour if self.delta >= 0 else "#f85149"
            delta_html = f'''
                <span style="color: {colour}; font-size: 0.8em; margin-left: 8px;">
                    {sign}{self.delta:.1f}%
                </span>
            '''
        
        return f'''
        <div style="
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            min-width: 150px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        ">
            <div style="color: #6c757d; font-size: 0.85em; margin-bottom: 8px;">
                {self.title}
            </div>
            <div style="font-size: 1.8em; font-weight: bold; color: #212529;">
                {self.value}{delta_html}
            </div>
        </div>
        '''


@dataclass
class FilterControl:
    """
    An interactive filter control for dashboard.
    
    Attributes:
        name: Display name for the control.
        control_type: Type of control ('select', 'range', 'checkbox').
        options: Available options/values.
        default_value: Initial selected value.
    """
    name: str
    control_type: str  # 'select', 'range', 'checkbox'
    options: list[Any] = field(default_factory=list)
    default_value: Any = None
    
    @property
    def element_id(self) -> str:
        """Generate HTML element ID from name."""
        return f"filter_{self.name.lower().replace(' ', '_')}"
    
    def to_html(self) -> str:
        """Generate HTML representation of the filter control."""
        if self.control_type == 'select':
            options_html = '\n'.join(
                f'<option value="{opt}" {"selected" if opt == self.default_value else ""}>{opt}</option>'
                for opt in self.options
            )
            return f'''
            <div style="margin: 12px 0;">
                <label style="
                    display: block;
                    margin-bottom: 6px;
                    font-weight: 600;
                    font-size: 0.9em;
                    color: #374151;
                ">{self.name}</label>
                <select 
                    id="{self.element_id}"
                    onchange="applyFilters()"
                    style="
                        width: 100%;
                        padding: 8px 12px;
                        border-radius: 6px;
                        border: 1px solid #d1d5db;
                        font-size: 0.9em;
                        background: white;
                    "
                >
                    <option value="">All</option>
                    {options_html}
                </select>
            </div>
            '''
        
        elif self.control_type == 'range':
            min_val = self.options[0] if self.options else 0
            max_val = self.options[-1] if self.options else 100
            current = self.default_value if self.default_value is not None else max_val
            
            return f'''
            <div style="margin: 12px 0;">
                <label style="
                    display: block;
                    margin-bottom: 6px;
                    font-weight: 600;
                    font-size: 0.9em;
                    color: #374151;
                ">{self.name}: <span id="value_{self.element_id}">{current}</span></label>
                <input 
                    type="range"
                    id="{self.element_id}"
                    min="{min_val}"
                    max="{max_val}"
                    value="{current}"
                    onchange="updateRangeValue(this); applyFilters()"
                    style="width: 100%;"
                >
            </div>
            '''
        
        elif self.control_type == 'checkbox':
            checkboxes = '\n'.join(
                f'''
                <label style="display: block; margin: 4px 0; font-size: 0.9em;">
                    <input type="checkbox" value="{opt}" 
                           {"checked" if opt in (self.default_value or []) else ""}
                           onchange="applyFilters()">
                    {opt}
                </label>
                '''
                for opt in self.options
            )
            return f'''
            <div style="margin: 12px 0;">
                <label style="
                    display: block;
                    margin-bottom: 6px;
                    font-weight: 600;
                    font-size: 0.9em;
                    color: #374151;
                ">{self.name}</label>
                <div id="{self.element_id}">
                    {checkboxes}
                </div>
            </div>
            '''
        
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SVG CHART GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

# Colourblind-friendly palette
CHART_COLOURS = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7',
    '#F0E442', '#56B4E9', '#D55E00', '#000000',
]


def generate_bar_chart_svg(
    data: dict[str, float],
    width: int = 400,
    height: int = 300,
    title: str = ""
) -> str:
    """
    Generate an SVG bar chart from categorical data.
    
    Args:
        data: Dictionary mapping category names to values.
        width: Chart width in pixels.
        height: Chart height in pixels.
        title: Optional chart title.
        
    Returns:
        SVG markup as string.
    """
    if not data:
        return '<svg width="{width}" height="{height}"></svg>'
    
    margin = {'top': 40, 'right': 20, 'bottom': 60, 'left': 60}
    chart_width = width - margin['left'] - margin['right']
    chart_height = height - margin['top'] - margin['bottom']
    
    categories = list(data.keys())
    values = list(data.values())
    max_value = max(values) if values else 1
    
    bar_width = chart_width / len(categories) * 0.8
    bar_spacing = chart_width / len(categories)
    
    # Build SVG elements
    bars_svg = ""
    labels_svg = ""
    
    for i, (cat, val) in enumerate(data.items()):
        bar_height = (val / max_value) * chart_height
        x = margin['left'] + i * bar_spacing + (bar_spacing - bar_width) / 2
        y = margin['top'] + chart_height - bar_height
        colour = CHART_COLOURS[i % len(CHART_COLOURS)]
        
        bars_svg += f'''
        <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}"
              fill="{colour}" rx="4" class="bar">
            <title>{cat}: {val:.1f}</title>
        </rect>
        '''
        
        # X-axis labels
        labels_svg += f'''
        <text x="{x + bar_width/2}" y="{margin['top'] + chart_height + 20}"
              text-anchor="middle" font-size="11" fill="#374151">{cat}</text>
        '''
    
    # Y-axis
    y_axis_svg = f'''
    <line x1="{margin['left']}" y1="{margin['top']}"
          x2="{margin['left']}" y2="{margin['top'] + chart_height}"
          stroke="#9ca3af" stroke-width="1"/>
    '''
    
    # Y-axis tick marks and labels
    for i in range(5):
        y = margin['top'] + chart_height - (i / 4) * chart_height
        val = (i / 4) * max_value
        y_axis_svg += f'''
        <text x="{margin['left'] - 10}" y="{y + 4}"
              text-anchor="end" font-size="10" fill="#6b7280">{val:.0f}</text>
        <line x1="{margin['left'] - 5}" y1="{y}" x2="{margin['left']}" y2="{y}"
              stroke="#9ca3af" stroke-width="1"/>
        '''
    
    # Title
    title_svg = f'''
    <text x="{width / 2}" y="24" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#111827">{title}</text>
    ''' if title else ""
    
    return f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .bar {{ transition: opacity 0.2s; cursor: pointer; }}
            .bar:hover {{ opacity: 0.8; }}
        </style>
        <rect width="100%" height="100%" fill="white"/>
        {title_svg}
        {y_axis_svg}
        {bars_svg}
        {labels_svg}
    </svg>
    '''


def generate_line_chart_svg(
    data: list[tuple[float, float]],
    width: int = 400,
    height: int = 300,
    title: str = "",
    colour: str = "#0072B2"
) -> str:
    """
    Generate an SVG line chart from x-y data pairs.
    
    Args:
        data: List of (x, y) tuples.
        width: Chart width in pixels.
        height: Chart height in pixels.
        title: Optional chart title.
        colour: Line colour (hex).
        
    Returns:
        SVG markup as string.
    """
    if not data:
        return f'<svg width="{width}" height="{height}"></svg>'
    
    margin = {'top': 40, 'right': 20, 'bottom': 40, 'left': 60}
    chart_width = width - margin['left'] - margin['right']
    chart_height = height - margin['top'] - margin['bottom']
    
    x_values = [d[0] for d in data]
    y_values = [d[1] for d in data]
    
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # Prevent division by zero
    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1
    
    # Scale functions
    def scale_x(val: float) -> float:
        return margin['left'] + ((val - x_min) / x_range) * chart_width
    
    def scale_y(val: float) -> float:
        return margin['top'] + chart_height - ((val - y_min) / y_range) * chart_height
    
    # Build path
    path_points = ' '.join(f"{scale_x(x)},{scale_y(y)}" for x, y in data)
    
    # Axes
    axes_svg = f'''
    <line x1="{margin['left']}" y1="{margin['top'] + chart_height}"
          x2="{margin['left'] + chart_width}" y2="{margin['top'] + chart_height}"
          stroke="#9ca3af" stroke-width="1"/>
    <line x1="{margin['left']}" y1="{margin['top']}"
          x2="{margin['left']}" y2="{margin['top'] + chart_height}"
          stroke="#9ca3af" stroke-width="1"/>
    '''
    
    # Title
    title_svg = f'''
    <text x="{width / 2}" y="24" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#111827">{title}</text>
    ''' if title else ""
    
    return f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="white"/>
        {title_svg}
        {axes_svg}
        <polyline points="{path_points}"
                  fill="none" stroke="{colour}" stroke-width="2"
                  stroke-linejoin="round" stroke-linecap="round"/>
    </svg>
    '''


def generate_pie_chart_svg(
    data: dict[str, float],
    width: int = 300,
    height: int = 300,
    title: str = ""
) -> str:
    """
    Generate an SVG pie chart from categorical data.
    
    Args:
        data: Dictionary mapping category names to values.
        width: Chart width in pixels.
        height: Chart height in pixels.
        title: Optional chart title.
        
    Returns:
        SVG markup as string.
    """
    if not data:
        return f'<svg width="{width}" height="{height}"></svg>'
    
    cx, cy = width / 2, height / 2
    radius = min(width, height) / 2 - 40
    
    total = sum(data.values())
    if total == 0:
        return f'<svg width="{width}" height="{height}"></svg>'
    
    slices_svg = ""
    legend_svg = ""
    start_angle = -math.pi / 2  # Start from top
    
    for i, (category, value) in enumerate(data.items()):
        angle = (value / total) * 2 * math.pi
        end_angle = start_angle + angle
        
        # Calculate arc path
        large_arc = 1 if angle > math.pi else 0
        x1 = cx + radius * math.cos(start_angle)
        y1 = cy + radius * math.sin(start_angle)
        x2 = cx + radius * math.cos(end_angle)
        y2 = cy + radius * math.sin(end_angle)
        
        colour = CHART_COLOURS[i % len(CHART_COLOURS)]
        percentage = (value / total) * 100
        
        slices_svg += f'''
        <path d="M {cx} {cy} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
              fill="{colour}" stroke="white" stroke-width="2" class="slice">
            <title>{category}: {value:.1f} ({percentage:.1f}%)</title>
        </path>
        '''
        
        # Legend
        legend_svg += f'''
        <rect x="{width - 100}" y="{40 + i * 22}" width="12" height="12"
              fill="{colour}" rx="2"/>
        <text x="{width - 82}" y="{51 + i * 22}"
              font-size="11" fill="#374151">{category}</text>
        '''
        
        start_angle = end_angle
    
    # Title
    title_svg = f'''
    <text x="{width / 2}" y="20" text-anchor="middle"
          font-size="14" font-weight="bold" fill="#111827">{title}</text>
    ''' if title else ""
    
    return f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .slice {{ transition: transform 0.2s; cursor: pointer; }}
            .slice:hover {{ transform: scale(1.02); transform-origin: {cx}px {cy}px; }}
        </style>
        <rect width="100%" height="100%" fill="white"/>
        {title_svg}
        {slices_svg}
        {legend_svg}
    </svg>
    '''


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DASHBOARD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    """
    Configuration for dashboard generation.
    
    Attributes:
        title: Dashboard title.
        description: Brief description shown below title.
        theme: Colour theme ('light' or 'dark').
        refresh_interval: Auto-refresh interval in seconds (0 = disabled).
    """
    title: str = "Research Dashboard"
    description: str = "Interactive data exploration"
    theme: str = "light"
    refresh_interval: int = 0


def generate_dashboard_html(
    dataset: Dataset,
    config: DashboardConfig | None = None
) -> str:
    """
    Generate a complete standalone HTML dashboard.
    
    This function creates an interactive dashboard with:
    - Summary metric cards
    - Filter controls
    - Multiple chart types
    - Data table
    
    Args:
        dataset: Dataset to visualise.
        config: Dashboard configuration options.
        
    Returns:
        Complete HTML document as string.
    """
    if config is None:
        config = DashboardConfig()
    
    # Calculate statistics
    stats = dataset.get_statistics()
    aggregation = dataset.aggregate_by_category()
    categories = list(set(p.category for p in dataset.points))
    
    # Create metric cards
    cards = [
        MetricCard("Total Records", int(stats['count'])),
        MetricCard("Average Value", f"{stats['mean']:.2f}"),
        MetricCard("Maximum", f"{stats['max']:.2f}"),
        MetricCard("Categories", len(categories)),
    ]
    
    # Create filter controls
    filters = [
        FilterControl("Category", "select", categories),
        FilterControl(
            "Value Range", 
            "range", 
            [0, 25, 50, 75, 100], 
            100
        ),
    ]
    
    # Generate charts
    bar_chart = generate_bar_chart_svg(
        aggregation,
        width=450,
        height=300,
        title="Total by Category"
    )
    
    pie_chart = generate_pie_chart_svg(
        aggregation,
        width=350,
        height=300,
        title="Distribution"
    )
    
    # Time series data
    time_data = [(i, p.value) for i, p in enumerate(dataset.points[:50])]
    line_chart = generate_line_chart_svg(
        time_data,
        width=800,
        height=250,
        title="Value Trend",
        colour="#0072B2"
    )
    
    # Generate HTML
    cards_html = '\n'.join(card.to_html() for card in cards)
    filters_html = '\n'.join(f.to_html() for f in filters)
    
    # Data for JavaScript
    data_json = dataset.to_json()
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
            background: #f3f4f6;
            color: #111827;
            line-height: 1.5;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .header h1 {{
            font-size: 1.875rem;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        
        .header p {{
            color: #6b7280;
            font-size: 1rem;
        }}
        
        .layout {{
            display: grid;
            grid-template-columns: 280px 1fr;
            gap: 24px;
        }}
        
        .sidebar {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            height: fit-content;
        }}
        
        .sidebar h2 {{
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 16px;
            color: #374151;
        }}
        
        .main {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
        }}
        
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 24px;
        }}
        
        .chart-card {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .full-width {{
            grid-column: 1 / -1;
        }}
        
        .data-table {{
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        
        .data-table h3 {{
            margin-bottom: 16px;
            font-size: 1rem;
            font-weight: 600;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}
        
        tr:hover td {{
            background: #f9fafb;
        }}
        
        .timestamp {{
            font-family: monospace;
            font-size: 0.8em;
            color: #6b7280;
        }}
        
        @media (max-width: 900px) {{
            .layout {{
                grid-template-columns: 1fr;
            }}
            .sidebar {{
                order: 2;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{config.title}</h1>
            <p>{config.description}</p>
        </header>
        
        <div class="layout">
            <aside class="sidebar">
                <h2>Filters</h2>
                {filters_html}
                <button onclick="resetFilters()" style="
                    width: 100%;
                    margin-top: 20px;
                    padding: 10px;
                    background: #f3f4f6;
                    border: 1px solid #d1d5db;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 0.9em;
                ">Reset Filters</button>
            </aside>
            
            <main class="main">
                <section class="metrics">
                    {cards_html}
                </section>
                
                <section class="charts">
                    <div class="chart-card">
                        {bar_chart}
                    </div>
                    <div class="chart-card">
                        {pie_chart}
                    </div>
                    <div class="chart-card full-width">
                        {line_chart}
                    </div>
                </section>
                
                <section class="data-table">
                    <h3>Data Table (First 20 Records)</h3>
                    <table id="data-table">
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>Timestamp</th>
                                <th>Category</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(f"""
                            <tr>
                                <td>{i}</td>
                                <td class="timestamp">{p.timestamp}</td>
                                <td>{p.category}</td>
                                <td>{p.value:.2f}</td>
                            </tr>
                            """ for i, p in enumerate(dataset.points[:20]))}
                        </tbody>
                    </table>
                </section>
            </main>
        </div>
    </div>
    
    <script>
        // Store original data
        const originalData = {data_json};
        
        function applyFilters() {{
            console.log('Filters applied');
            // In a real implementation, this would filter and redraw charts
        }}
        
        function resetFilters() {{
            document.querySelectorAll('select').forEach(s => s.selectedIndex = 0);
            document.querySelectorAll('input[type="range"]').forEach(r => {{
                r.value = r.max;
                const label = document.getElementById('value_' + r.id);
                if (label) label.textContent = r.value;
            }});
            applyFilters();
        }}
        
        function updateRangeValue(input) {{
            const label = document.getElementById('value_' + input.id);
            if (label) label.textContent = input.value;
        }}
    </script>
</body>
</html>'''
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: STREAMLIT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

STREAMLIT_TEMPLATE = '''#!/usr/bin/env python3
"""
Streamlit Dashboard Template
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 Research Data Explorer")
st.markdown("---")

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    st.markdown("---")
    st.subheader("Filters")
    
    # Date range filter
    date_range = st.date_input(
        "Date Range",
        value=(datetime(2025, 1, 1), datetime(2025, 12, 31)),
        key="date_filter"
    )
    
    # Category filter
    categories = st.multiselect(
        "Categories",
        options=['Category A', 'Category B', 'Category C', 'Category D'],
        default=['Category A', 'Category B', 'Category C', 'Category D']
    )
    
    # Value range slider
    value_range = st.slider(
        "Value Range",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    st.markdown("---")
    st.subheader("Export")
    if st.button("📥 Download Data"):
        st.info("Export functionality placeholder")


# Data loading
@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration."""
    import numpy as np
    np.random.seed(42)
    
    n = 200
    data = {
        'date': pd.date_range('2025-01-01', periods=n, freq='D'),
        'category': np.random.choice(
            ['Category A', 'Category B', 'Category C', 'Category D'], n
        ),
        'value': np.random.normal(50, 15, n),
        'metric_1': np.random.uniform(0, 100, n),
        'metric_2': np.random.uniform(0, 100, n),
    }
    return pd.DataFrame(data)


# Load data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()

# Apply filters
df_filtered = df[df['category'].isin(categories)]
df_filtered = df_filtered[
    (df_filtered['value'] >= value_range[0]) &
    (df_filtered['value'] <= value_range[1])
]

# Metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Records",
        value=len(df_filtered),
        delta=f"{len(df_filtered) - len(df)} filtered"
    )

with col2:
    st.metric(
        label="Average Value",
        value=f"{df_filtered['value'].mean():.2f}",
        delta=f"{df_filtered['value'].mean() - df['value'].mean():.2f}"
    )

with col3:
    st.metric(
        label="Maximum Value",
        value=f"{df_filtered['value'].max():.2f}"
    )

with col4:
    st.metric(
        label="Categories",
        value=len(df_filtered['category'].unique())
    )

st.markdown("---")

# Charts
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Distribution by Category")
    fig_bar = px.bar(
        df_filtered.groupby('category')['value'].sum().reset_index(),
        x='category',
        y='value',
        color='category',
        title="Total Value by Category"
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("🥧 Category Distribution")
    fig_pie = px.pie(
        df_filtered.groupby('category')['value'].sum().reset_index(),
        names='category',
        values='value',
        title="Value Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# Time series
st.subheader("📈 Trend Over Time")
df_time = df_filtered.groupby('date')['value'].mean().reset_index()
fig_line = px.line(
    df_time,
    x='date',
    y='value',
    title="Average Value Over Time"
)
fig_line.update_traces(line_color='#0072B2')
st.plotly_chart(fig_line, use_container_width=True)

# Additional charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Correlation Analysis")
    fig_scatter = px.scatter(
        df_filtered,
        x='metric_1',
        y='metric_2',
        color='category',
        title="Metric 1 vs Metric 2"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("📦 Box Plot")
    fig_box = px.box(
        df_filtered,
        x='category',
        y='value',
        color='category',
        title="Value Distribution by Category"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Data table
st.subheader("📋 Data Table")
st.dataframe(
    df_filtered.head(50),
    use_container_width=True,
    hide_index=True
)

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; color: #888;">
        Dashboard created with Streamlit | 
        Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
    </div>
    """,
    unsafe_allow_html=True
)
'''


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_dashboard_generation() -> None:
    """Demonstrate dashboard HTML generation."""
    logger.info("Running dashboard generation demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: Dashboard Generation")
    print("=" * 70)
    print()
    
    # Generate sample data
    dataset = Dataset.generate_sample(100)
    
    print(f"  Dataset name: {dataset.name}")
    print(f"  Total points: {len(dataset.points)}")
    print(f"  Categories: {set(p.category for p in dataset.points)}")
    print()
    
    # Generate dashboard
    config = DashboardConfig(
        title="Research Dashboard Demo",
        description="Sample data visualisation for demonstration"
    )
    
    html = generate_dashboard_html(dataset, config)
    
    print(f"  Generated HTML: {len(html):,} characters")
    print()
    print("  To view the dashboard:")
    print("    1. Save to file: Path('dashboard.html').write_text(html)")
    print("    2. Open in browser")
    print()


def demo_chart_generation() -> None:
    """Demonstrate SVG chart generation."""
    logger.info("Running chart generation demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: SVG Chart Generation")
    print("=" * 70)
    print()
    
    # Bar chart
    bar_data = {
        'Category A': 45,
        'Category B': 32,
        'Category C': 58,
        'Category D': 27
    }
    bar_svg = generate_bar_chart_svg(bar_data, title="Sample Bar Chart")
    print(f"  Bar chart: {len(bar_svg):,} characters")
    
    # Pie chart
    pie_svg = generate_pie_chart_svg(bar_data, title="Sample Pie Chart")
    print(f"  Pie chart: {len(pie_svg):,} characters")
    
    # Line chart
    line_data = [(i, math.sin(i / 5) * 30 + 50) for i in range(40)]
    line_svg = generate_line_chart_svg(line_data, title="Sample Line Chart")
    print(f"  Line chart: {len(line_svg):,} characters")
    
    print()
    print("  Charts are SVG markup that can be:")
    print("    - Embedded directly in HTML")
    print("    - Saved as .svg files")
    print("    - Converted to other formats")
    print()


def demo_streamlit_template() -> None:
    """Show Streamlit template information."""
    logger.info("Running Streamlit template demonstration")
    print()
    print("=" * 70)
    print("DEMONSTRATION: Streamlit Template")
    print("=" * 70)
    print()
    
    print(f"  Template size: {len(STREAMLIT_TEMPLATE):,} characters")
    print()
    print("  To use the Streamlit template:")
    print("    1. Save: Path('dashboard.py').write_text(STREAMLIT_TEMPLATE)")
    print("    2. Install: pip install streamlit plotly pandas")
    print("    3. Run: streamlit run dashboard.py")
    print()
    print("  Features included:")
    print("    - File upload")
    print("    - Interactive filters (date, category, value range)")
    print("    - Multiple chart types (bar, pie, line, scatter, box)")
    print("    - Data table with filtering")
    print("    - Responsive layout")
    print()


def run_all_demos() -> None:
    """Execute all demonstration functions."""
    demo_dashboard_generation()
    demo_chart_generation()
    demo_streamlit_template()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Week 6 Lab 2: Interactive Dashboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                Run all demonstrations
  %(prog)s --generate-html       Generate sample dashboard HTML
  %(prog)s --export-streamlit    Export Streamlit template
        """
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run all demonstrations'
    )
    parser.add_argument(
        '--generate-html',
        type=str,
        metavar='PATH',
        help='Generate dashboard HTML to specified path'
    )
    parser.add_argument(
        '--export-streamlit',
        type=str,
        metavar='PATH',
        help='Export Streamlit template to specified path'
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
    print("  WEEK 6 LAB 2: INTERACTIVE DASHBOARDS")
    print("═" * 70)
    print()
    
    if args.generate_html:
        dataset = Dataset.generate_sample(100)
        html = generate_dashboard_html(dataset)
        Path(args.generate_html).write_text(html)
        print(f"  Dashboard saved to: {args.generate_html}")
    
    elif args.export_streamlit:
        Path(args.export_streamlit).write_text(STREAMLIT_TEMPLATE)
        print(f"  Streamlit template saved to: {args.export_streamlit}")
    
    elif args.demo:
        run_all_demos()
    
    else:
        print("  Use --demo to run demonstrations")
        print("  Use --help for all options")
        print()
        print("  Quick start:")
        print("    from lab_6_02_interactive_viz import Dataset, generate_dashboard_html")
        print("    dataset = Dataset.generate_sample(100)")
        print("    html = generate_dashboard_html(dataset)")
        print("    Path('dashboard.html').write_text(html)")
    
    print()
    print("═" * 70)
    print("  © 2025 Antonio Clim. All rights reserved.")
    print("═" * 70)


if __name__ == "__main__":
    main()
