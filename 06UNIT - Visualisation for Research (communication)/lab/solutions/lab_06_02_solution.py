#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Lab 2: Interactive Visualisation — SOLUTION
═══════════════════════════════════════════════════════════════════════════════

This solution file demonstrates complete implementations of interactive
visualisations using Plotly, D3.js generation and Streamlit dashboards.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    logger.warning("plotly not available. Install with: pip install plotly")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataPoint:
    """Single data point with metadata.
    
    Attributes:
        x: X coordinate
        y: Y coordinate
        label: Point label
        category: Category for grouping
        size: Optional size value
        colour: Optional colour value
    """
    x: float
    y: float
    label: str = ""
    category: str = "default"
    size: float = 1.0
    colour: float = 0.0


@dataclass
class Dataset:
    """Collection of data points.
    
    Attributes:
        name: Dataset name
        points: List of DataPoint instances
        metadata: Additional metadata dictionary
    """
    name: str
    points: list[DataPoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_arrays(self) -> dict[str, np.ndarray]:
        """Convert to numpy arrays for plotting."""
        return {
            'x': np.array([p.x for p in self.points]),
            'y': np.array([p.y for p in self.points]),
            'size': np.array([p.size for p in self.points]),
            'colour': np.array([p.colour for p in self.points]),
            'labels': [p.label for p in self.points],
            'categories': [p.category for p in self.points]
        }
    
    @classmethod
    def from_arrays(
        cls,
        name: str,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> 'Dataset':
        """Create dataset from numpy arrays."""
        n = len(x)
        labels = kwargs.get('labels', [''] * n)
        categories = kwargs.get('categories', ['default'] * n)
        sizes = kwargs.get('sizes', np.ones(n))
        colours = kwargs.get('colours', np.zeros(n))
        
        points = [
            DataPoint(
                x=float(x[i]),
                y=float(y[i]),
                label=labels[i],
                category=categories[i],
                size=float(sizes[i]),
                colour=float(colours[i])
            )
            for i in range(n)
        ]
        
        return cls(name=name, points=points, metadata=kwargs.get('metadata', {}))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PLOTLY INTERACTIVE VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_interactive_scatter(
    dataset: Dataset,
    title: str = "Interactive Scatter Plot",
    x_label: str = "X",
    y_label: str = "Y",
    colour_by: str | None = None,
    size_by: str | None = None
) -> 'go.Figure':
    """Create interactive scatter plot with Plotly.
    
    Args:
        dataset: Dataset to visualise
        title: Plot title
        x_label: X axis label
        y_label: Y axis label
        colour_by: Column to colour by ('category' or 'colour')
        size_by: Column to size by ('size')
        
    Returns:
        Plotly Figure object
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required for interactive visualisations")
    
    arrays = dataset.to_arrays()
    
    fig = go.Figure()
    
    # Determine colour and size
    if colour_by == 'category':
        # Group by category
        categories = list(set(arrays['categories']))
        for cat in categories:
            mask = [c == cat for c in arrays['categories']]
            x_cat = arrays['x'][mask]
            y_cat = arrays['y'][mask]
            labels_cat = [l for l, m in zip(arrays['labels'], mask) if m]
            
            fig.add_trace(go.Scatter(
                x=x_cat,
                y=y_cat,
                mode='markers',
                name=cat,
                text=labels_cat,
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
    else:
        # Single trace
        marker_kwargs = {
            'size': arrays['size'] * 10 if size_by == 'size' else 8,
            'color': arrays['colour'] if colour_by == 'colour' else '#0072B2',
            'colorscale': 'Viridis' if colour_by == 'colour' else None,
            'showscale': colour_by == 'colour'
        }
        
        fig.add_trace(go.Scatter(
            x=arrays['x'],
            y=arrays['y'],
            mode='markers',
            name=dataset.name,
            text=arrays['labels'],
            marker=marker_kwargs,
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig


def create_interactive_line(
    x: np.ndarray,
    y_dict: dict[str, np.ndarray],
    title: str = "Interactive Line Plot",
    x_label: str = "X",
    y_label: str = "Y"
) -> 'go.Figure':
    """Create interactive line plot with multiple series.
    
    Args:
        x: X values (shared across series)
        y_dict: Dictionary mapping series names to Y values
        title: Plot title
        x_label: X axis label
        y_label: Y axis label
        
    Returns:
        Plotly Figure object
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required for interactive visualisations")
    
    fig = go.Figure()
    
    colours = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442']
    
    for i, (name, y) in enumerate(y_dict.items()):
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=name,
            line={'color': colours[i % len(colours)], 'width': 2}
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='x unified',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02}
    )
    
    return fig


def create_interactive_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str = "Interactive Heatmap"
) -> 'go.Figure':
    """Create interactive heatmap.
    
    Args:
        data: 2D array of values
        row_labels: Row labels
        col_labels: Column labels
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly required for interactive visualisations")
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=col_labels,
        y=row_labels,
        colorscale='RdBu_r',
        hovertemplate='%{y} × %{x}<br>Value: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        template='plotly_white'
    )
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: D3.JS CODE GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

class D3Generator:
    """Generate D3.js visualisation code.
    
    This class produces standalone HTML files with embedded D3.js
    visualisations following the enter-update-exit pattern.
    """
    
    D3_CDN = "https://d3js.org/d3.v7.min.js"
    
    @staticmethod
    def generate_bar_chart(
        data: list[dict[str, Any]],
        width: int = 600,
        height: int = 400,
        x_key: str = 'label',
        y_key: str = 'value',
        title: str = 'Bar Chart'
    ) -> str:
        """Generate D3.js bar chart HTML.
        
        Args:
            data: List of dictionaries with x_key and y_key
            width: Chart width in pixels
            height: Chart height in pixels
            x_key: Key for x-axis values
            y_key: Key for y-axis values
            title: Chart title
            
        Returns:
            Complete HTML string
        """
        margin = {'top': 40, 'right': 30, 'bottom': 60, 'left': 60}
        inner_width = width - margin['left'] - margin['right']
        inner_height = height - margin['top'] - margin['bottom']
        
        data_json = json.dumps(data)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{D3Generator.D3_CDN}"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #c9d1d9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{
            background: #0d1117;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }}
        .bar {{
            fill: #0072B2;
            transition: fill 0.2s;
        }}
        .bar:hover {{
            fill: #58a6ff;
        }}
        .axis-label {{
            fill: #8b949e;
            font-size: 12px;
        }}
        .title {{
            fill: #c9d1d9;
            font-size: 16px;
            font-weight: 600;
        }}
        .tooltip {{
            position: absolute;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }}
    </style>
</head>
<body>
    <div class="container">
        <svg id="chart"></svg>
    </div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const data = {data_json};
        
        const margin = {json.dumps(margin)};
        const width = {width};
        const height = {height};
        const innerWidth = {inner_width};
        const innerHeight = {inner_height};
        
        const svg = d3.select("#chart")
            .attr("width", width)
            .attr("height", height);
        
        const g = svg.append("g")
            .attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);
        
        // Title
        svg.append("text")
            .attr("class", "title")
            .attr("x", width / 2)
            .attr("y", 25)
            .attr("text-anchor", "middle")
            .text("{title}");
        
        // Scales
        const xScale = d3.scaleBand()
            .domain(data.map(d => d.{x_key}))
            .range([0, innerWidth])
            .padding(0.2);
        
        const yScale = d3.scaleLinear()
            .domain([0, d3.max(data, d => d.{y_key}) * 1.1])
            .range([innerHeight, 0]);
        
        // Axes
        g.append("g")
            .attr("transform", `translate(0, ${{innerHeight}})`)
            .call(d3.axisBottom(xScale))
            .selectAll("text")
            .attr("class", "axis-label")
            .attr("transform", "rotate(-45)")
            .attr("text-anchor", "end");
        
        g.append("g")
            .call(d3.axisLeft(yScale).ticks(5))
            .selectAll("text")
            .attr("class", "axis-label");
        
        // Tooltip
        const tooltip = d3.select("#tooltip");
        
        // Bars with enter-update-exit pattern
        const bars = g.selectAll(".bar")
            .data(data)
            .join(
                enter => enter.append("rect")
                    .attr("class", "bar")
                    .attr("x", d => xScale(d.{x_key}))
                    .attr("y", innerHeight)
                    .attr("width", xScale.bandwidth())
                    .attr("height", 0)
                    .call(enter => enter.transition()
                        .duration(800)
                        .delay((d, i) => i * 50)
                        .attr("y", d => yScale(d.{y_key}))
                        .attr("height", d => innerHeight - yScale(d.{y_key}))),
                update => update,
                exit => exit.remove()
            );
        
        // Hover interactions
        bars.on("mouseover", function(event, d) {{
                tooltip.style("opacity", 1)
                    .html(`<strong>${{d.{x_key}}}</strong><br>Value: ${{d.{y_key}.toFixed(2)}}`);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 20) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.style("opacity", 0);
            }});
    </script>
</body>
</html>'''
        
        return html
    
    @staticmethod
    def generate_scatter_plot(
        data: list[dict[str, Any]],
        width: int = 600,
        height: int = 400,
        x_key: str = 'x',
        y_key: str = 'y',
        title: str = 'Scatter Plot'
    ) -> str:
        """Generate D3.js scatter plot HTML.
        
        Args:
            data: List of dictionaries with x and y values
            width: Chart width in pixels
            height: Chart height in pixels
            x_key: Key for x-axis values
            y_key: Key for y-axis values
            title: Chart title
            
        Returns:
            Complete HTML string
        """
        margin = {'top': 40, 'right': 30, 'bottom': 50, 'left': 60}
        inner_width = width - margin['left'] - margin['right']
        inner_height = height - margin['top'] - margin['bottom']
        
        data_json = json.dumps(data)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="{D3Generator.D3_CDN}"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: #c9d1d9;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{
            background: #0d1117;
            border-radius: 8px;
            padding: 20px;
        }}
        .point {{
            fill: #0072B2;
            stroke: #fff;
            stroke-width: 1;
            opacity: 0.7;
            transition: all 0.2s;
        }}
        .point:hover {{
            opacity: 1;
            r: 8;
        }}
        .axis-label {{ fill: #8b949e; font-size: 12px; }}
        .title {{ fill: #c9d1d9; font-size: 16px; font-weight: 600; }}
        .tooltip {{
            position: absolute;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
        }}
    </style>
</head>
<body>
    <div class="container"><svg id="chart"></svg></div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const data = {data_json};
        const margin = {json.dumps(margin)};
        const width = {width}, height = {height};
        const innerWidth = {inner_width}, innerHeight = {inner_height};
        
        const svg = d3.select("#chart")
            .attr("width", width).attr("height", height);
        
        const g = svg.append("g")
            .attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);
        
        svg.append("text").attr("class", "title")
            .attr("x", width / 2).attr("y", 25)
            .attr("text-anchor", "middle").text("{title}");
        
        const xScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.{x_key})).nice()
            .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
            .domain(d3.extent(data, d => d.{y_key})).nice()
            .range([innerHeight, 0]);
        
        g.append("g").attr("transform", `translate(0, ${{innerHeight}})`)
            .call(d3.axisBottom(xScale)).selectAll("text").attr("class", "axis-label");
        
        g.append("g").call(d3.axisLeft(yScale))
            .selectAll("text").attr("class", "axis-label");
        
        const tooltip = d3.select("#tooltip");
        
        g.selectAll(".point").data(data)
            .join("circle")
            .attr("class", "point")
            .attr("cx", d => xScale(d.{x_key}))
            .attr("cy", d => yScale(d.{y_key}))
            .attr("r", 0)
            .transition().duration(500).delay((d, i) => i * 10)
            .attr("r", 5);
        
        g.selectAll(".point")
            .on("mouseover", function(event, d) {{
                tooltip.style("opacity", 1)
                    .html(`X: ${{d.{x_key}.toFixed(2)}}<br>Y: ${{d.{y_key}.toFixed(2)}}`);
            }})
            .on("mousemove", function(event) {{
                tooltip.style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 20) + "px");
            }})
            .on("mouseout", () => tooltip.style("opacity", 0));
    </script>
</body>
</html>'''
        
        return html


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: STREAMLIT DASHBOARD TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

STREAMLIT_TEMPLATE = '''#!/usr/bin/env python3
"""
Interactive Dashboard for Research Data
Generated by lab_6_02_interactive_viz.py

Run with: streamlit run dashboard.py
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Page configuration
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
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Research Data Dashboard")
st.markdown("Interactive visualisation for exploratory data analysis")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

n_points = st.sidebar.slider("Number of data points", 50, 500, 100)
noise_level = st.sidebar.slider("Noise level", 0.0, 2.0, 0.5)
seed = st.sidebar.number_input("Random seed", value=42)

# Generate sample data
np.random.seed(seed)
x = np.linspace(0, 10, n_points)
y1 = np.sin(x) + np.random.normal(0, noise_level, n_points)
y2 = np.cos(x) + np.random.normal(0, noise_level, n_points)

df = pd.DataFrame({
    'x': x,
    'sin': y1,
    'cos': y2,
    'category': np.random.choice(['A', 'B', 'C'], n_points)
})

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Data Points", n_points)

with col2:
    st.metric("Mean (sin)", f"{y1.mean():.3f}")

with col3:
    st.metric("Mean (cos)", f"{y2.mean():.3f}")

# Tabs for different visualisations
tab1, tab2, tab3 = st.tabs(["📈 Line Plot", "🔵 Scatter", "📊 Distribution"])

with tab1:
    fig = px.line(df, x='x', y=['sin', 'cos'], title='Trigonometric Functions')
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(df, x='sin', y='cos', color='category',
                     title='Sin vs Cos Relationship')
    fig.update_layout(template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=y1, name='sin', opacity=0.7))
    fig.add_trace(go.Histogram(x=y2, name='cos', opacity=0.7))
    fig.update_layout(barmode='overlay', template='plotly_dark',
                      title='Value Distributions')
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.subheader("📋 Raw Data")
st.dataframe(df.head(20), use_container_width=True)

# Download button
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "data.csv", "text/csv")
'''


def generate_streamlit_dashboard(output_path: Path | str) -> None:
    """Generate Streamlit dashboard file.
    
    Args:
        output_path: Path to save the dashboard script
    """
    output_path = Path(output_path)
    output_path.write_text(STREAMLIT_TEMPLATE)
    logger.info(f"Generated Streamlit dashboard: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_plotly_interactive() -> None:
    """Demonstrate Plotly interactive visualisations."""
    if not HAS_PLOTLY:
        logger.warning("Plotly not available. Skipping demo.")
        return
    
    logger.info("Creating Plotly demonstrations...")
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    # Create dataset
    points = [
        DataPoint(
            x=np.random.normal(0, 1),
            y=np.random.normal(0, 1),
            label=f"Point {i}",
            category=np.random.choice(['Group A', 'Group B', 'Group C']),
            size=np.random.uniform(0.5, 2.0),
            colour=np.random.uniform(0, 1)
        )
        for i in range(n)
    ]
    
    dataset = Dataset(name="Sample Data", points=points)
    
    # Create scatter plot
    fig = create_interactive_scatter(
        dataset,
        title="Interactive Scatter with Categories",
        colour_by='category'
    )
    fig.write_html("demo_scatter.html")
    logger.info("Saved: demo_scatter.html")
    
    # Create line plot
    x = np.linspace(0, 10, 100)
    y_dict = {
        'sin(x)': np.sin(x),
        'cos(x)': np.cos(x),
        'sin(2x)/2': np.sin(2 * x) / 2
    }
    
    fig = create_interactive_line(x, y_dict, title="Trigonometric Functions")
    fig.write_html("demo_line.html")
    logger.info("Saved: demo_line.html")


def demo_d3_generation() -> None:
    """Demonstrate D3.js code generation."""
    logger.info("Creating D3.js demonstrations...")
    
    # Bar chart data
    bar_data = [
        {'label': 'Category A', 'value': 42.5},
        {'label': 'Category B', 'value': 67.2},
        {'label': 'Category C', 'value': 51.8},
        {'label': 'Category D', 'value': 89.1},
        {'label': 'Category E', 'value': 34.6}
    ]
    
    html = D3Generator.generate_bar_chart(
        bar_data,
        title="Sample Bar Chart",
        width=700,
        height=450
    )
    
    Path("demo_d3_bar.html").write_text(html)
    logger.info("Saved: demo_d3_bar.html")
    
    # Scatter plot data
    np.random.seed(42)
    scatter_data = [
        {'x': float(x), 'y': float(2 * x + np.random.normal(0, 3))}
        for x in np.linspace(0, 20, 50)
    ]
    
    html = D3Generator.generate_scatter_plot(
        scatter_data,
        title="Sample Scatter Plot",
        width=700,
        height=450
    )
    
    Path("demo_d3_scatter.html").write_text(html)
    logger.info("Saved: demo_d3_scatter.html")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_plotly_interactive()
    demo_d3_generation()
    generate_streamlit_dashboard("demo_dashboard.py")
    logger.info("All demonstrations completed!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Interactive visualisation solution demonstrations"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        run_all_demos()
    else:
        print("Run with --demo to generate example visualisations")
        print("See source code for complete implementations")


if __name__ == "__main__":
    main()
