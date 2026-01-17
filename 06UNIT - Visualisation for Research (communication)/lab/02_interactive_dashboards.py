#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Lab 2: Interactive Dashboards
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Dashboard-urile interactive permit explorarea datelor în mod dinamic.
Acest laborator demonstrează crearea de dashboards cu:
- Streamlit (Python-first, rapid prototyping)
- HTML + JavaScript (deployment static)

OBIECTIVE
─────────
1. Înțelegerea componentelor unui dashboard
2. Crearea de vizualizări interactive
3. Implementarea filtrelor și controalelor
4. Export pentru deployment

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataPoint:
    """Un punct de date pentru dashboard."""
    timestamp: str
    category: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass 
class Dataset:
    """Container pentru date dashboard."""
    name: str
    points: list[DataPoint] = field(default_factory=list)
    
    def filter_by_category(self, category: str) -> 'Dataset':
        """Filtrează după categorie."""
        filtered = [p for p in self.points if p.category == category]
        return Dataset(f"{self.name}_filtered", filtered)
    
    def filter_by_value_range(self, min_val: float, max_val: float) -> 'Dataset':
        """Filtrează după interval valori."""
        filtered = [p for p in self.points if min_val <= p.value <= max_val]
        return Dataset(f"{self.name}_range", filtered)
    
    def aggregate_by_category(self) -> dict[str, float]:
        """Agregare sumă pe categorii."""
        agg: dict[str, float] = {}
        for p in self.points:
            agg[p.category] = agg.get(p.category, 0) + p.value
        return agg
    
    def to_json(self) -> str:
        """Export ca JSON."""
        data = [
            {
                'timestamp': p.timestamp,
                'category': p.category,
                'value': p.value,
                **p.metadata
            }
            for p in self.points
        ]
        return json.dumps(data, indent=2)
    
    @classmethod
    def generate_sample(cls, n_points: int = 100, seed: int = 42) -> 'Dataset':
        """Generează date de test."""
        random.seed(seed)
        categories = ['A', 'B', 'C', 'D']
        points = []
        
        for i in range(n_points):
            point = DataPoint(
                timestamp=f"2025-01-{(i % 28) + 1:02d}",
                category=random.choice(categories),
                value=random.gauss(50, 15),
                metadata={'index': i}
            )
            points.append(point)
        
        return cls("sample_data", points)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: DASHBOARD COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricCard:
    """Card pentru afișare metrici."""
    title: str
    value: float | int | str
    delta: float | None = None
    delta_color: str = "green"
    
    def to_html(self) -> str:
        """Generează HTML pentru card."""
        delta_html = ""
        if self.delta is not None:
            sign = "+" if self.delta >= 0 else ""
            color = self.delta_color if self.delta >= 0 else "red"
            delta_html = f'<span style="color: {color}; font-size: 0.8em;">{sign}{self.delta:.1f}%</span>'
        
        return f'''
        <div style="background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; min-width: 150px;">
            <div style="color: #6c757d; font-size: 0.9em; margin-bottom: 8px;">{self.title}</div>
            <div style="font-size: 2em; font-weight: bold; color: #212529;">{self.value}</div>
            {delta_html}
        </div>
        '''


@dataclass
class FilterControl:
    """Control pentru filtrare interactivă."""
    name: str
    control_type: str  # 'select', 'range', 'checkbox'
    options: list[Any] = field(default_factory=list)
    default_value: Any = None
    
    def to_html(self) -> str:
        """Generează HTML pentru control."""
        if self.control_type == 'select':
            options_html = '\n'.join(
                f'<option value="{opt}" {"selected" if opt == self.default_value else ""}>{opt}</option>'
                for opt in self.options
            )
            return f'''
            <div style="margin: 10px 0;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">{self.name}</label>
                <select id="filter_{self.name.lower().replace(' ', '_')}" 
                        onchange="applyFilters()" 
                        style="width: 100%; padding: 8px; border-radius: 4px; border: 1px solid #ced4da;">
                    <option value="">All</option>
                    {options_html}
                </select>
            </div>
            '''
        elif self.control_type == 'range':
            min_val, max_val = self.options[0], self.options[-1]
            return f'''
            <div style="margin: 10px 0;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">{self.name}</label>
                <input type="range" id="filter_{self.name.lower().replace(' ', '_')}"
                       min="{min_val}" max="{max_val}" value="{self.default_value or max_val}"
                       onchange="applyFilters()"
                       style="width: 100%;">
                <span id="range_value_{self.name.lower().replace(' ', '_')}">{self.default_value or max_val}</span>
            </div>
            '''
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: CHART GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_bar_chart_svg(
    data: dict[str, float],
    width: int = 400,
    height: int = 300,
    title: str = ""
) -> str:
    """Generează SVG pentru bar chart."""
    if not data:
        return "<svg></svg>"
    
    margin = {'top': 40, 'right': 20, 'bottom': 60, 'left': 60}
    chart_width = width - margin['left'] - margin['right']
    chart_height = height - margin['top'] - margin['bottom']
    
    categories = list(data.keys())
    values = list(data.values())
    max_value = max(values) if values else 1
    
    bar_width = chart_width / len(categories) * 0.8
    bar_spacing = chart_width / len(categories)
    
    bars_svg = ""
    labels_svg = ""
    
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    
    for i, (cat, val) in enumerate(data.items()):
        bar_height = (val / max_value) * chart_height
        x = margin['left'] + i * bar_spacing + (bar_spacing - bar_width) / 2
        y = margin['top'] + chart_height - bar_height
        color = colors[i % len(colors)]
        
        bars_svg += f'''
        <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" 
              fill="{color}" rx="4">
            <title>{cat}: {val:.1f}</title>
        </rect>
        '''
        
        labels_svg += f'''
        <text x="{x + bar_width/2}" y="{margin['top'] + chart_height + 20}" 
              text-anchor="middle" font-size="12">{cat}</text>
        '''
    
    # Y axis
    y_axis_svg = f'''
    <line x1="{margin['left']}" y1="{margin['top']}" 
          x2="{margin['left']}" y2="{margin['top'] + chart_height}"
          stroke="#333" stroke-width="1"/>
    '''
    
    # Y axis labels
    for i in range(5):
        y = margin['top'] + chart_height - (i/4) * chart_height
        val = (i/4) * max_value
        y_axis_svg += f'''
        <text x="{margin['left'] - 10}" y="{y + 4}" 
              text-anchor="end" font-size="10">{val:.0f}</text>
        <line x1="{margin['left'] - 5}" y1="{y}" x2="{margin['left']}" y2="{y}"
              stroke="#333" stroke-width="1"/>
        '''
    
    title_svg = f'''
    <text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    ''' if title else ""
    
    return f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            rect {{ transition: opacity 0.2s; }}
            rect:hover {{ opacity: 0.8; }}
        </style>
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
    color: str = "#4C72B0"
) -> str:
    """Generează SVG pentru line chart."""
    if not data:
        return "<svg></svg>"
    
    margin = {'top': 40, 'right': 20, 'bottom': 40, 'left': 60}
    chart_width = width - margin['left'] - margin['right']
    chart_height = height - margin['top'] - margin['bottom']
    
    x_vals = [d[0] for d in data]
    y_vals = [d[1] for d in data]
    
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    
    # Normalize to chart coordinates
    def scale_x(x: float) -> float:
        if x_max == x_min:
            return margin['left'] + chart_width / 2
        return margin['left'] + ((x - x_min) / (x_max - x_min)) * chart_width
    
    def scale_y(y: float) -> float:
        if y_max == y_min:
            return margin['top'] + chart_height / 2
        return margin['top'] + chart_height - ((y - y_min) / (y_max - y_min)) * chart_height
    
    # Build path
    path_points = [f"{scale_x(x)},{scale_y(y)}" for x, y in data]
    path_d = f"M {' L '.join(path_points)}"
    
    # Build area (for fill under line)
    area_path = f"M {scale_x(x_vals[0])},{margin['top'] + chart_height} L {path_d[2:]} L {scale_x(x_vals[-1])},{margin['top'] + chart_height} Z"
    
    title_svg = f'''
    <text x="{width/2}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    ''' if title else ""
    
    return f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        {title_svg}
        <path d="{area_path}" fill="{color}" fill-opacity="0.1"/>
        <path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>
        <!-- Dots -->
        {''.join(f'<circle cx="{scale_x(x)}" cy="{scale_y(y)}" r="4" fill="{color}"><title>{x:.1f}, {y:.1f}</title></circle>' for x, y in data)}
    </svg>
    '''


def generate_pie_chart_svg(
    data: dict[str, float],
    width: int = 300,
    height: int = 300,
    title: str = ""
) -> str:
    """Generează SVG pentru pie chart."""
    if not data:
        return "<svg></svg>"
    
    cx, cy = width / 2, height / 2 + 15
    radius = min(width, height) / 2 - 40
    
    total = sum(data.values())
    if total == 0:
        return "<svg></svg>"
    
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
    
    slices_svg = ""
    legend_svg = ""
    start_angle = 0
    
    for i, (cat, val) in enumerate(data.items()):
        percentage = val / total
        end_angle = start_angle + percentage * 2 * math.pi
        
        # Calculate arc points
        x1 = cx + radius * math.cos(start_angle)
        y1 = cy + radius * math.sin(start_angle)
        x2 = cx + radius * math.cos(end_angle)
        y2 = cy + radius * math.sin(end_angle)
        
        large_arc = 1 if percentage > 0.5 else 0
        color = colors[i % len(colors)]
        
        path = f"M {cx} {cy} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
        
        slices_svg += f'''
        <path d="{path}" fill="{color}" stroke="white" stroke-width="2">
            <title>{cat}: {val:.1f} ({percentage*100:.1f}%)</title>
        </path>
        '''
        
        # Legend item
        legend_y = 20 + i * 20
        legend_svg += f'''
        <rect x="{width - 100}" y="{legend_y}" width="12" height="12" fill="{color}"/>
        <text x="{width - 82}" y="{legend_y + 10}" font-size="11">{cat}</text>
        '''
        
        start_angle = end_angle
    
    title_svg = f'''
    <text x="{width/2}" y="15" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>
    ''' if title else ""
    
    return f'''
    <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            path {{ transition: transform 0.2s; transform-origin: {cx}px {cy}px; }}
            path:hover {{ transform: scale(1.05); }}
        </style>
        {title_svg}
        {slices_svg}
        {legend_svg}
    </svg>
    '''


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: DASHBOARD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    """Configurație pentru dashboard."""
    title: str = "Research Dashboard"
    description: str = ""
    theme: str = "light"  # 'light' or 'dark'
    refresh_interval: int | None = None  # seconds


def generate_dashboard_html(
    dataset: Dataset,
    config: DashboardConfig = DashboardConfig()
) -> str:
    """
    Generează un dashboard HTML complet și interactiv.
    
    Args:
        dataset: Datele pentru vizualizare
        config: Configurația dashboard-ului
    
    Returns:
        String HTML complet
    """
    # Calculate metrics
    total_value = sum(p.value for p in dataset.points)
    avg_value = total_value / len(dataset.points) if dataset.points else 0
    categories = list(set(p.category for p in dataset.points))
    
    # Create metric cards
    metrics = [
        MetricCard("Total Points", len(dataset.points)),
        MetricCard("Total Value", f"{total_value:.1f}", delta=5.2),
        MetricCard("Average", f"{avg_value:.1f}", delta=-2.1, delta_color="red"),
        MetricCard("Categories", len(categories)),
    ]
    
    metrics_html = '<div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 30px;">'
    metrics_html += ''.join(m.to_html() for m in metrics)
    metrics_html += '</div>'
    
    # Create filters
    filters = [
        FilterControl("Category", "select", categories),
        FilterControl("Min Value", "range", [0, 100], 0),
    ]
    
    filters_html = '<div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">'
    filters_html += '<h3 style="margin-top: 0;">Filters</h3>'
    filters_html += ''.join(f.to_html() for f in filters)
    filters_html += '</div>'
    
    # Create charts
    aggregated = dataset.aggregate_by_category()
    bar_chart = generate_bar_chart_svg(aggregated, 500, 350, "Values by Category")
    pie_chart = generate_pie_chart_svg(aggregated, 350, 350, "Distribution")
    
    # Time series (simulated)
    time_data = [(i, p.value) for i, p in enumerate(dataset.points[:50])]
    line_chart = generate_line_chart_svg(time_data, 500, 300, "Values Over Time")
    
    # Theme
    bg_color = "#ffffff" if config.theme == "light" else "#1e1e1e"
    text_color = "#212529" if config.theme == "light" else "#f8f9fa"
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{config.title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: {bg_color};
            color: {text_color};
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .header {{
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        h1 {{ margin: 0 0 10px 0; }}
        .description {{ color: #6c757d; }}
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
        }}
        .chart-card {{
            background: {"#f8f9fa" if config.theme == "light" else "#2d2d2d"};
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-card h3 {{ margin-top: 0; }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .data-table th, .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        .data-table th {{ background: #f8f9fa; font-weight: 600; }}
        .data-table tr:hover {{ background: #f1f3f4; }}
        .sidebar {{
            position: fixed;
            right: 20px;
            top: 20px;
            width: 250px;
        }}
        @media (max-width: 1200px) {{
            .sidebar {{ position: static; width: 100%; margin-bottom: 20px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>{config.title}</h1>
            <p class="description">{config.description or f"Dashboard generated from {dataset.name}"}</p>
            <p style="font-size: 0.9em; color: #6c757d;">Last updated: <span id="update-time"></span></p>
        </header>
        
        {metrics_html}
        
        <div style="display: flex; gap: 30px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px;">
                {filters_html}
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h3>📊 Category Breakdown</h3>
                <div id="bar-chart">{bar_chart}</div>
            </div>
            
            <div class="chart-card">
                <h3>🥧 Distribution</h3>
                <div id="pie-chart">{pie_chart}</div>
            </div>
            
            <div class="chart-card">
                <h3>📈 Trend</h3>
                <div id="line-chart">{line_chart}</div>
            </div>
            
            <div class="chart-card">
                <h3>📋 Data Table</h3>
                <div style="max-height: 300px; overflow-y: auto;">
                    <table class="data-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Category</th>
                                <th>Value</th>
                            </tr>
                        </thead>
                        <tbody id="data-table-body">
                            {''.join(f"<tr><td>{p.timestamp}</td><td>{p.category}</td><td>{p.value:.2f}</td></tr>" for p in dataset.points[:20])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; font-size: 0.9em;">
            <p>Generated with Computational Thinking Dashboard Toolkit</p>
        </footer>
    </div>
    
    <script>
        // Data
        const rawData = {dataset.to_json()};
        
        // Update time
        document.getElementById('update-time').textContent = new Date().toLocaleString();
        
        // Filter function
        function applyFilters() {{
            const categoryFilter = document.getElementById('filter_category')?.value || '';
            const minValueFilter = parseFloat(document.getElementById('filter_min_value')?.value || 0);
            
            // Update range display
            const rangeDisplay = document.getElementById('range_value_min_value');
            if (rangeDisplay) rangeDisplay.textContent = minValueFilter;
            
            // Filter data
            let filtered = rawData;
            if (categoryFilter) {{
                filtered = filtered.filter(d => d.category === categoryFilter);
            }}
            filtered = filtered.filter(d => d.value >= minValueFilter);
            
            // Update table
            const tbody = document.getElementById('data-table-body');
            tbody.innerHTML = filtered.slice(0, 20).map(d => 
                `<tr><td>${{d.timestamp}}</td><td>${{d.category}}</td><td>${{d.value.toFixed(2)}}</td></tr>`
            ).join('');
            
            console.log(`Filtered: ${{filtered.length}} points`);
        }}
        
        // Export function
        function exportData(format) {{
            if (format === 'csv') {{
                const csv = 'timestamp,category,value\\n' + 
                    rawData.map(d => `${{d.timestamp}},${{d.category}},${{d.value}}`).join('\\n');
                const blob = new Blob([csv], {{ type: 'text/csv' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'data.csv';
                a.click();
            }}
        }}
    </script>
</body>
</html>'''
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: STREAMLIT TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

STREAMLIT_TEMPLATE = '''
"""
Streamlit Dashboard Template
============================

Run with: streamlit run dashboard.py

Requirements:
    pip install streamlit pandas plotly
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Research Dashboard</p>', unsafe_allow_html=True)
st.markdown("Interactive visualization of research data")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    
    st.markdown("---")
    
    # Filters
    st.subheader("Filters")
    
    # Date range (example)
    date_range = st.date_input(
        "Date Range",
        value=(datetime(2025, 1, 1), datetime(2025, 12, 31)),
        key="date_filter"
    )
    
    # Category filter (populated from data)
    categories = st.multiselect(
        "Categories",
        options=['A', 'B', 'C', 'D'],
        default=['A', 'B', 'C', 'D']
    )
    
    # Value range
    value_range = st.slider(
        "Value Range",
        min_value=0,
        max_value=100,
        value=(0, 100)
    )
    
    st.markdown("---")
    
    # Export
    st.subheader("Export")
    if st.button("📥 Download Data"):
        st.info("Export functionality here")

# Load data
@st.cache_data
def load_sample_data():
    """Generate sample data."""
    import numpy as np
    np.random.seed(42)
    
    n = 200
    data = {
        'date': pd.date_range('2025-01-01', periods=n, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'value': np.random.normal(50, 15, n),
        'metric_1': np.random.uniform(0, 100, n),
        'metric_2': np.random.uniform(0, 100, n),
    }
    return pd.DataFrame(data)

# Get data
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
        label="Max Value",
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
        df_filtered,
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
fig_line.update_traces(line_color='#4C72B0')
st.plotly_chart(fig_line, use_container_width=True)

# Scatter plot
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
    st.subheader("📊 Box Plot")
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
    """
    <div style="text-align: center; color: #888;">
        Dashboard created with Streamlit | Last updated: {}
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M")),
    unsafe_allow_html=True
)
'''


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA VI: DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_dashboard() -> None:
    """Demonstrație: generare dashboard."""
    print("=" * 60)
    print("DEMO: Dashboard Generation")
    print("=" * 60)
    print()
    
    # Generate sample data
    dataset = Dataset.generate_sample(100)
    
    print(f"Dataset: {dataset.name}")
    print(f"Points: {len(dataset.points)}")
    print(f"Categories: {set(p.category for p in dataset.points)}")
    print()
    
    # Generate dashboard
    config = DashboardConfig(
        title="Research Dashboard Demo",
        description="Sample data visualization"
    )
    
    html = generate_dashboard_html(dataset, config)
    
    print(f"Generated HTML: {len(html)} characters")
    print("Save to .html file and open in browser to view.")
    print()
    
    # Show aggregation
    agg = dataset.aggregate_by_category()
    print("Aggregation by category:")
    for cat, val in agg.items():
        print(f"  {cat}: {val:.1f}")
    print()


def demo_charts() -> None:
    """Demonstrație: generare chart-uri SVG."""
    print("=" * 60)
    print("DEMO: SVG Chart Generation")
    print("=" * 60)
    print()
    
    # Bar chart
    bar_data = {'A': 30, 'B': 45, 'C': 25, 'D': 50}
    bar_svg = generate_bar_chart_svg(bar_data, title="Sample Bar Chart")
    print(f"Bar chart SVG: {len(bar_svg)} characters")
    
    # Pie chart
    pie_svg = generate_pie_chart_svg(bar_data, title="Sample Pie Chart")
    print(f"Pie chart SVG: {len(pie_svg)} characters")
    
    # Line chart
    line_data = [(i, math.sin(i/5) * 50 + 50) for i in range(30)]
    line_svg = generate_line_chart_svg(line_data, title="Sample Line Chart")
    print(f"Line chart SVG: {len(line_svg)} characters")
    
    print()
    print("Charts can be embedded in HTML or saved as .svg files.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 6 LAB 2: INTERACTIVE DASHBOARDS")
    print("═" * 60 + "\n")
    
    demo_dashboard()
    demo_charts()
    
    print("=" * 60)
    print("Streamlit template available in STREAMLIT_TEMPLATE variable.")
    print("Run: streamlit run dashboard.py")
    print("=" * 60)
