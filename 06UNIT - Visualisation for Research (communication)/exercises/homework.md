# Week 6 Homework: Visualisation for Research

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | Friday 23:59 GMT |
| **Total Points** | 100 |
| **Estimated Time** | 4-5 hours |
| **Difficulty** | ⭐⭐⭐ (3/5) |

## 🔗 Prerequisites

Before starting this homework, ensure you have:

- [ ] Completed Lab 6.1: Static Visualisation Toolkit
- [ ] Completed Lab 6.2: Interactive Dashboards
- [ ] Read the lecture notes on Tufte's principles
- [ ] Installed required libraries (matplotlib, seaborn, plotly)

## 🎯 Objectives Assessed

1. Create publication-quality static figures following journal standards
2. Build interactive dashboards for data exploration
3. Apply Tufte's principles to evaluate and improve visualisations
4. Implement D3.js visualisations for web deployment

---

## ⚠️ Capstone Project Preparation

**Important**: Begin planning your capstone project! You will present your proposal next week.

Prepare a one-page document including:
- Project title and brief description (2-3 sentences)
- Research/engineering problem addressed
- Data sources (existing or to be generated)
- Computational methods from this course
- Expected outputs (code, visualisations, report)
- Simplified timeline

---

## Part 1: Publication-Ready Figures (30 points)

### Context

Scientific journals have strict requirements for figures:
- **Resolution**: Minimum 300 DPI for raster formats
- **Format**: PDF/EPS for vector, TIFF/PNG for raster
- **Fonts**: Embedded, no external dependencies
- **Dimensions**: Specified in millimetres or inches
- **Colour**: Accessible, colourblind-friendly palettes

### Requirements

Create **three publication-quality figures** for a hypothetical research paper, using data from your Week 5 simulations (Monte Carlo, ODE solvers, or agent-based models) or newly generated data.

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

def setup_nature_style():
    """Configure matplotlib for Nature journal figures."""
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.figsize': (3.5, 2.5),  # Single column
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
```

### Figure Specifications

#### Figure 1: Line Plot with Multiple Series (10 points)

| Requirement | Points | Description |
|-------------|--------|-------------|
| 1.1 | 3 | Multiple data series with distinct, accessible colours |
| 1.2 | 2 | Clear legend (positioned without obscuring data) |
| 1.3 | 2 | Axis labels with units (e.g., "Time (s)", "Temperature (°C)") |
| 1.4 | 2 | Appropriate tick marks and grid (subtle) |
| 1.5 | 1 | Correct figure dimensions and DPI |

**Suggested content**: SIR model curves, convergence comparison, time series

#### Figure 2: Scatter Plot with Regression (10 points)

| Requirement | Points | Description |
|-------------|--------|-------------|
| 2.1 | 3 | Data points with error bars (if applicable) |
| 2.2 | 3 | Regression line overlaid |
| 2.3 | 2 | R² value and equation displayed |
| 2.4 | 2 | Appropriate axis scaling and labels |

**Suggested content**: Correlation analysis, model validation, experimental vs predicted

#### Figure 3: Complex Visualisation (10 points)

| Requirement | Points | Description |
|-------------|--------|-------------|
| 3.1 | 3 | Heatmap, contour plot, or multi-panel layout |
| 3.2 | 3 | Colour bar with descriptive label |
| 3.3 | 2 | Annotations where relevant |
| 3.4 | 2 | Consistent styling with other figures |

**Suggested content**: Correlation matrix, parameter sweep, small multiples comparison

### Deliverables

For each figure, submit:
- `figure_1.pdf` (vector format)
- `figure_1.png` (300 DPI raster)
- `figure_1_code.py` (reproducible generation script)
- `figure_1_caption.txt` (50-100 words describing the figure)

---

## Part 2: Interactive Dashboard (35 points)

### Context

Interactive dashboards enable exploratory data analysis that static figures cannot support. They allow users to filter, zoom, brush, and explore data dynamically.

### Requirements

Create an interactive dashboard for exploring a research dataset.

### Option A: Streamlit (Recommended for Python Users)

```python
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Research Data Explorer")

# Sidebar for filters
st.sidebar.header("Filters")
year_range = st.sidebar.slider("Year", 2010, 2024, (2015, 2020))

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()
filtered = data[(data['year'] >= year_range[0]) & 
                (data['year'] <= year_range[1])]

# Visualisations
col1, col2 = st.columns(2)
with col1:
    fig1 = px.histogram(filtered, x='category')
    st.plotly_chart(fig1)
with col2:
    fig2 = px.scatter(filtered, x='x', y='y', color='category')
    st.plotly_chart(fig2)

# Interactive data table
st.dataframe(filtered)
```

### Option B: HTML + D3.js (For Maximum Control)

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <div id="controls">
        <select id="category-select"></select>
        <input type="range" id="value-slider">
    </div>
    <div id="chart1"></div>
    <div id="chart2"></div>
    
    <script>
        // Load data and create interactive visualisations
    </script>
</body>
</html>
```

### Feature Requirements

| Feature | Points | Description |
|---------|--------|-------------|
| Data loading | 5 | Load and display dataset correctly |
| Visualisation variety | 10 | Minimum 2 different chart types |
| Interactive filtering | 10 | Dropdown, slider, or date range filters |
| Linked views | 5 | Selection in one chart updates others |
| Deployment | 5 | Hosted on Streamlit Cloud or GitHub Pages |

### Deliverables

- Dashboard code (Streamlit `.py` or HTML/JS files)
- Deployment URL
- `dashboard_readme.md` explaining design decisions
- Screenshot for offline review

---

## Part 3: Custom D3.js Visualisation (20 points)

### Context

D3.js provides complete control over interactive visualisations. This exercise develops skills for creating bespoke visualisations beyond library defaults.

### Requirements

Create a custom D3.js visualisation appropriate for your research domain.

### Suggested Visualisation Types

1. **Force-Directed Graph** — co-author networks, citation networks
2. **Treemap** — hierarchical category breakdown
3. **Sankey Diagram** — flow between states/stages
4. **Choropleth Map** — geographical data
5. **Animated Timeline** — temporal evolution

### Technical Requirements

```javascript
// Minimum structure
const margin = {top: 20, right: 20, bottom: 30, left: 40};
const width = 600 - margin.left - margin.right;
const height = 400 - margin.top - margin.bottom;

const svg = d3.select("#chart")
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// Scales
const xScale = d3.scaleLinear()
    .domain([/* data extent */])
    .range([0, width]);

// Axes
svg.append("g")
    .attr("transform", `translate(0,${height})`)
    .call(d3.axisBottom(xScale));

// Data binding (enter-update-exit pattern)
// Transitions
// Interactivity (hover, click)
```

| Requirement | Points | Description |
|-------------|--------|-------------|
| Data binding | 5 | Correct use of enter-update-exit |
| Scales and axes | 5 | Appropriate scales for data type |
| Transitions | 5 | Smooth animations between states |
| Interactivity | 5 | Tooltips, hover effects, or click handlers |

### Deliverables

- `index.html` — complete page
- `style.css` — styling (optional, can be inline)
- `visualisation.js` — D3 code
- `data.json` or `data.csv` — dataset
- Screenshot for README

---

## Part 4: Animated Visualisation (15 points)

### Context

Animated visualisations can communicate temporal processes or algorithmic behaviour more effectively than static images.

### Requirements

Create an animation illustrating a concept from this course.

### Option A: Matplotlib Animation

```python
from matplotlib.animation import FuncAnimation
import numpy as np

fig, ax = plt.subplots()
line, = ax.plot([], [])

def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 1)
    return line,

def update(frame):
    x = np.linspace(0, 10, 100)
    y = np.sin(x + frame / 10)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, update, frames=100, init_func=init, blit=True)
ani.save('animation.gif', writer='pillow', fps=30)
```

### Option B: D3.js Animation

### Option C: Manim (for Mathematical Animations)

### Suggested Topics

- Algorithm visualisation (sorting, searching, graph traversal)
- Monte Carlo convergence over iterations
- Agent-based model dynamics
- ODE solution evolution
- BFS vs DFS traversal comparison

### Assessment Criteria

| Criterion | Points | Description |
|-----------|--------|-------------|
| Clarity | 5 | Message communicated effectively |
| Technical quality | 5 | Smooth animation, correct behaviour |
| Aesthetics | 5 | Visually appealing, consistent styling |

### Deliverables

- Animation file (`.gif`, `.mp4`, or `.html`)
- Generation script
- Brief description of what is being visualised

---

## ✅ Submission Checklist

### Part 1: Publication Figures
- [ ] `figure_1.pdf`, `figure_1.png`, `figure_1_code.py`, `figure_1_caption.txt`
- [ ] `figure_2.pdf`, `figure_2.png`, `figure_2_code.py`, `figure_2_caption.txt`
- [ ] `figure_3.pdf`, `figure_3.png`, `figure_3_code.py`, `figure_3_caption.txt`
- [ ] All figures use colourblind-friendly palette
- [ ] All figures at ≥300 DPI

### Part 2: Interactive Dashboard
- [ ] Dashboard code (`.py` or HTML/JS)
- [ ] Deployment URL (Streamlit Cloud or GitHub Pages)
- [ ] `dashboard_readme.md`
- [ ] Screenshot

### Part 3: D3.js Visualisation
- [ ] `index.html`
- [ ] `visualisation.js` (or inline)
- [ ] Data file
- [ ] Screenshot

### Part 4: Animation
- [ ] Animation file
- [ ] Generation script
- [ ] Description

### Code Quality
- [ ] All scripts run without errors
- [ ] Type hints present
- [ ] Docstrings for functions
- [ ] Formatted with ruff or black
- [ ] No hardcoded absolute paths

---

## 📝 Grading Rubric Summary

| Component | Points |
|-----------|--------|
| Part 1: Publication Figures | 30 |
| Part 2: Interactive Dashboard | 35 |
| Part 3: D3.js Visualisation | 20 |
| Part 4: Animation | 15 |
| **Total** | **100** |

### Grade Boundaries

| Grade | Points | Description |
|-------|--------|-------------|
| A | 90-100 | Exceptional work, exceeds all requirements |
| B | 75-89 | Strong work, meets all requirements |
| C | 60-74 | Adequate work, meets most requirements |
| D | 50-59 | Partial completion, significant gaps |
| F | <50 | Incomplete or insufficient |

---

## 💡 Hints

<details>
<summary>Hint 1: Colourblind-Friendly Palettes</summary>

Use the Wong palette or ColorBrewer qualitative schemes:
```python
COLORBLIND_PALETTE = [
    '#0072B2', '#E69F00', '#009E73', '#CC79A7',
    '#F0E442', '#56B4E9', '#D55E00', '#000000'
]
```
</details>

<details>
<summary>Hint 2: Streamlit Deployment</summary>

1. Create a GitHub repository with your `app.py`
2. Add a `requirements.txt` with dependencies
3. Go to share.streamlit.io and connect your repository
4. Your app will be deployed automatically
</details>

<details>
<summary>Hint 3: D3.js Scales</summary>

Choose scales based on data type:
- Continuous → `d3.scaleLinear()`
- Categorical → `d3.scaleOrdinal()`, `d3.scaleBand()`
- Time → `d3.scaleTime()`
- Log-transformed → `d3.scaleLog()`
</details>

<details>
<summary>Hint 4: Animation Performance</summary>

For matplotlib animations:
- Use `blit=True` for better performance
- Reduce frame count if file size is too large
- Consider `interval` parameter for timing
</details>

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 6 — Visualisation for Research*

© 2025 Antonio Clim. All rights reserved.
