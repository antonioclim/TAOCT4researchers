# Week 6 Cheatsheet: Visualisation for Research

## 🎨 Colourblind-Friendly Palettes

### Wong Palette (8 colours)
```python
WONG = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', 
        '#F0E442', '#56B4E9', '#D55E00', '#000000']
```

### Matplotlib Colormaps
| Type | Good | Avoid |
|------|------|-------|
| Sequential | `viridis`, `plasma`, `cividis` | `jet`, `rainbow` |
| Diverging | `RdBu_r`, `coolwarm` | `spectral` |
| Qualitative | `Set2`, `Dark2` | `Paired` |

---

## 📐 Journal Specifications

| Journal | Width (mm) | Font | Size | DPI |
|---------|------------|------|------|-----|
| Nature | 89 / 183 | Arial | 7pt | 300 |
| Science | 85 / 178 | Helvetica | 7pt | 300 |
| IEEE | 88 / 181 | Times | 8pt | 600 |
| PLOS | 140 / 190 | Arial | 8pt | 300 |

```python
# Convert mm to inches
def mm_to_inches(mm): return mm / 25.4
```

---

## 📊 Matplotlib Quick Reference

### Publication Figure Setup
```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Journal-ready rcParams
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Create figure (Nature single column)
fig = plt.figure(figsize=(3.5, 2.6))  # 89mm × 66mm
gs = GridSpec(1, 2, figure=fig)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
```

### Panel Labels
```python
for i, ax in enumerate(axes):
    ax.text(-0.15, 1.05, chr(65+i),  # A, B, C...
            transform=ax.transAxes,
            fontsize=9, fontweight='bold')
```

### Tufte Style (Remove Chartjunk)
```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='out', length=3)
```

### Export Multiple Formats
```python
for fmt in ['pdf', 'png', 'svg']:
    fig.savefig(f'figure.{fmt}', dpi=300, 
                bbox_inches='tight', pad_inches=0.02)
```

---

## 🔄 Plotly Quick Reference

### Interactive Scatter
```python
import plotly.express as px

fig = px.scatter(df, x='x', y='y', color='category',
                 hover_data=['label'], 
                 template='plotly_white')
fig.write_html('plot.html')
```

### Line Plot with Multiple Series
```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y1, name='Series 1'))
fig.add_trace(go.Scatter(x=x, y=y2, name='Series 2'))
fig.update_layout(hovermode='x unified')
```

### Custom Hover Template
```python
hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
```

---

## 🌐 D3.js Quick Reference

### Basic Structure
```javascript
const svg = d3.select("#chart")
    .attr("width", width)
    .attr("height", height);

const g = svg.append("g")
    .attr("transform", `translate(${margin.left}, ${margin.top})`);
```

### Scales
```javascript
// Linear scale (continuous → continuous)
const xScale = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)])
    .range([0, innerWidth]);

// Band scale (categorical → continuous)
const xScale = d3.scaleBand()
    .domain(data.map(d => d.label))
    .range([0, innerWidth])
    .padding(0.2);

// Time scale
const xScale = d3.scaleTime()
    .domain(d3.extent(data, d => d.date))
    .range([0, innerWidth]);
```

### Enter-Update-Exit Pattern
```javascript
g.selectAll(".bar")
    .data(data)
    .join(
        enter => enter.append("rect")
            .attr("class", "bar")
            .attr("x", d => xScale(d.label))
            .attr("y", innerHeight)
            .attr("height", 0)
            .call(enter => enter.transition()
                .duration(800)
                .attr("y", d => yScale(d.value))
                .attr("height", d => innerHeight - yScale(d.value))),
        update => update,
        exit => exit.remove()
    );
```

### Transitions
```javascript
selection.transition()
    .duration(500)
    .ease(d3.easeCubicOut)
    .attr("y", d => yScale(d.value));
```

---

## 🎬 Animation Quick Reference

### Matplotlib FuncAnimation
```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
scatter = ax.scatter([], [])

def init():
    scatter.set_offsets([])
    return (scatter,)

def update(frame):
    positions = simulate_step(frame)
    scatter.set_offsets(positions)
    return (scatter,)

anim = FuncAnimation(fig, update, init_func=init,
                     frames=200, interval=33, blit=True)

# Save as GIF
anim.save('animation.gif', writer='pillow', fps=30)

# Save as MP4
anim.save('animation.mp4', writer='ffmpeg', fps=30)
```

---

## ⚠️ Common Mistakes

| Mistake | Fix |
|---------|-----|
| Using `jet` colourmap | Use `viridis` or Wong palette |
| Colour-only encoding | Add shape, pattern or label |
| Missing random seed | `np.random.seed(42)` at start |
| Hardcoded paths | Use `pathlib.Path` |
| `plt.show()` in scripts | Use `plt.savefig()` instead |
| Not closing figures | `plt.close(fig)` after saving |
| Legend covers data | `bbox_to_anchor=(1.05, 1)` |
| Wrong axis limits | Use `.set_xlim()` explicitly |

---

## 🔗 Connections

| Week 5 Concept | Week 6 Application |
|----------------|-------------------|
| Monte Carlo simulation | Convergence plots |
| ODE solver output | Time series visualisation |
| Agent-based models | Animation of emergent behaviour |
| Numerical data | Publication figures |

| Week 6 Concept | Week 7 Application |
|----------------|-------------------|
| Script-based generation | CI/CD pipelines |
| Multi-format export | Documentation builds |
| Reproducible figures | Testing & validation |

---

## 🎯 Figure Export Guidelines

### Resolution Requirements

| Output | DPI | Format | Notes |
|--------|-----|--------|-------|
| Print publication | 300-600 | PDF, EPS | Vector preferred |
| Web display | 72-150 | PNG, SVG | Optimise file size |
| Poster | 150-300 | PDF | Large canvas |
| Presentation | 150 | PNG | Balance quality/size |

### Matplotlib Backend Selection

```python
# Non-interactive (for scripts)
import matplotlib
matplotlib.use('Agg')  # Before importing pyplot

# For Jupyter notebooks
%matplotlib inline      # Static figures
%matplotlib widget      # Interactive figures
```

### Font Embedding for PDF

```python
# Ensure fonts embed correctly in PDFs
plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts
plt.rcParams['ps.fonttype'] = 42   # PostScript compatibility
```

---

## 📏 Tufte's Principles

1. **Data-ink ratio** = Data ink / Total ink used
   - Maximise: remove gridlines, borders, backgrounds

2. **Lie factor** = Size of effect in graphic / Size of effect in data
   - Should equal 1.0 (no distortion)

3. **Chartjunk** = Non-data visual elements
   - Remove: 3D effects, gradients, decorations

4. **Small multiples** = Same chart repeated for subsets
   - Use: FacetGrid, subplots with shared axes

---

© 2025 Antonio Clim. All rights reserved.
