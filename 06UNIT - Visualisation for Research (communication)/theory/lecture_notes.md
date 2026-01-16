# Week 6: Lecture Notes — Visualisation for Research

## Introduction

Data visualisation is the art and science of representing information graphically. For researchers, it serves two critical purposes: exploration (discovering patterns in data) and communication (conveying findings to others). A well-designed visualisation can reveal insights that tables of numbers obscure, whilst a poorly designed one can mislead or confuse.

This week, we examine the theoretical foundations of effective visualisation, drawing primarily from Edward Tufte's seminal work, before moving to practical implementation with Python libraries and D3.js. The goal is to equip you with both the critical eye to evaluate visualisations and the technical skills to create them.

---

## Part I: Theoretical Foundations

### 1.1 Tufte's Principles of Data Visualisation

Edward Tufte, often called the "Galileo of graphics," established principles that remain foundational to modern visualisation practice. His 1983 book, *The Visual Display of Quantitative Information*, introduced concepts that guide practitioners to this day.

#### The Data-Ink Ratio

Tufte defines "data-ink" as the non-erasable core of a graphic—the ink that represents actual data values. The data-ink ratio is calculated as:

```
Data-Ink Ratio = Data-Ink / Total Ink Used in Graphic
```

A ratio approaching 1.0 indicates a graphic where nearly every mark represents data. Tufte advocates maximising this ratio by eliminating "chartjunk"—decorative elements that add visual interest but no information.

Consider a bar chart with heavy gridlines, three-dimensional bevels, gradient fills, and drop shadows. Each of these elements consumes ink (or pixels) without encoding data. Removing them produces a cleaner graphic that directs attention to what matters: the data itself.

**Practical application**: Before finalising any figure, ask yourself: "If I remove this element, do I lose information?" If the answer is no, remove it.

#### The Lie Factor

The lie factor measures how honestly a graphic represents data proportions:

```
Lie Factor = (Size of effect shown in graphic) / (Size of effect in data)
```

A lie factor of 1.0 indicates perfect accuracy. Factors significantly above or below 1.0 indicate distortion. Common sources of distortion include:

- **Truncated axes**: Starting a y-axis at a value other than zero makes small differences appear dramatic.
- **Area versus length**: Doubling a value but doubling both the width and height of a symbol creates a 4× visual effect.
- **Three-dimensional effects**: Perspective distorts relative sizes.

**Example**: If sales increased from £98 million to £102 million (a 4% increase), and a graphic shows a bar that is twice as tall for the second value, the lie factor is approximately 24 (100% visual increase / 4% data increase).

#### Small Multiples

Small multiples are a series of similar graphs or charts, using the same scale and axes, shown together to facilitate comparison. Rather than overlaying many lines on a single plot (which can become cluttered), small multiples place each series in its own panel.

This technique excels when:
- Comparing trends across categories
- Showing geographic variation
- Displaying change over multiple time periods

The key is consistency: all panels must share identical scales to enable direct comparison.

#### Micro/Macro Readings

Effective visualisations support reading at multiple levels. At the macro level, viewers should immediately grasp the overall pattern or message. At the micro level, they should be able to extract specific data values. Both levels must be served by a single graphic.

### 1.2 The Grammar of Graphics

Leland Wilkinson's *Grammar of Graphics* (1999) provides a formal framework for describing visualisations. This grammar decomposes any statistical graphic into seven components:

1. **Data**: The dataset to be visualised, typically in tabular form.
2. **Aesthetics (aes)**: Mappings from data variables to visual properties (position, colour, size, shape).
3. **Geometries (geom)**: The visual elements representing data (points, lines, bars, areas).
4. **Facets**: Subdivision into panels for comparison (small multiples).
5. **Statistics (stat)**: Transformations applied to data before rendering (binning, smoothing, aggregation).
6. **Coordinates (coord)**: The coordinate system (Cartesian, polar, geographic).
7. **Themes**: Non-data elements controlling appearance (fonts, colours, gridlines).

**Implementation**: Libraries like ggplot2 (R), Altair (Python) and Vega-Lite (JavaScript) implement this grammar directly. Even when using Matplotlib, thinking in terms of these components clarifies design decisions.

### 1.3 Perception and Cognition

Human visual perception has characteristics that influence how we interpret graphics:

#### Pre-attentive Processing

Certain visual features are processed pre-attentively—without conscious effort, in under 250 milliseconds. These include:
- Position
- Length
- Colour hue
- Orientation

Using these features to encode important distinctions ensures viewers perceive them immediately. Conversely, features like area and angle require more cognitive effort to compare accurately.

#### Cleveland and McGill's Hierarchy

Cleveland and McGill (1984) ranked visual encodings by accuracy of human perception:

1. Position along a common scale (most accurate)
2. Position on identical but unaligned scales
3. Length
4. Angle/Slope
5. Area
6. Volume, density, colour saturation (least accurate)

**Practical implication**: Encode your most important variable using position. Reserve colour for categorical distinctions, not quantitative comparison.

#### Colour Considerations

Colour choices affect both aesthetics and accessibility:

- **Colourblind-friendly palettes**: Approximately 8% of men have red-green colour blindness. Use palettes designed for accessibility (ColorBrewer, Viridis).
- **Sequential vs. diverging**: Sequential palettes (light to dark) suit continuous data. Diverging palettes (two hues meeting at a neutral centre) suit data with a meaningful midpoint.
- **Print compatibility**: Ensure visualisations remain interpretable when printed in greyscale.

---

## Part II: Static Visualisation with Python

### 2.1 Matplotlib Architecture

Matplotlib follows an object-oriented architecture with three layers:

1. **Backend layer**: Handles rendering to different outputs (screen, files).
2. **Artist layer**: Primitives that draw on the canvas (Line2D, Patch, Text).
3. **Scripting layer (pyplot)**: MATLAB-style interface for quick plotting.

For publication figures, use the object-oriented interface:

```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.plot(x, y)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (mV)")
fig.savefig("figure.pdf", dpi=300, bbox_inches='tight')
```

### 2.2 Publication Standards

Different journals have specific requirements:

| Journal | Column Width | Font | DPI |
|---------|--------------|------|-----|
| Nature | 89mm (single), 183mm (double) | Sans-serif, 7-8pt | 300 (min) |
| Science | 85mm (single), 180mm (double) | Sans-serif, 7pt | 300 (raster), vector preferred |
| IEEE | 88mm (single), 181mm (double) | Serif, 8pt | 600 |

All journals require:
- Embedded fonts (no external dependencies)
- Legible text at final print size
- Accessible colour choices
- High resolution for raster images

### 2.3 Seaborn for Statistical Graphics

Seaborn provides high-level functions for common statistical visualisations:

- `sns.histplot()`: Histograms with optional KDE overlay
- `sns.boxplot()`, `sns.violinplot()`: Distribution comparison
- `sns.scatterplot()`: Scatter with automatic grouping
- `sns.heatmap()`: Correlation matrices and 2D data
- `sns.pairplot()`: Pairwise relationships in a dataset

Seaborn integrates with Pandas DataFrames and produces aesthetically pleasing defaults that can be customised for publication.

---

## Part III: Interactive Visualisation

### 3.1 The Case for Interactivity

Interactive visualisations enable exploration that static figures cannot support:

- **Filtering**: Show subsets of data matching user criteria.
- **Details on demand**: Display additional information on hover or click.
- **Brushing and linking**: Selections in one view highlight corresponding data in others.
- **Zooming and panning**: Navigate large or high-resolution datasets.
- **Animation**: Show change over time or parameter space.

For research, interactive visualisations serve during the exploration phase and for communicating complex findings (supplementary materials, online publications).

### 3.2 Plotly for Python

Plotly Express provides a concise API for interactive figures:

```python
import plotly.express as px

fig = px.scatter(df, x="x", y="y", color="category",
                 hover_data=["name"], title="Dataset Overview")
fig.show()
```

Figures support zooming, panning, tooltips, and can be exported to standalone HTML files.

### 3.3 D3.js Fundamentals

D3.js (Data-Driven Documents) provides complete control over interactive visualisations in the browser. Unlike chart libraries, D3 does not provide pre-built chart types; instead, it provides primitives for binding data to DOM elements and manipulating them.

#### The Enter-Update-Exit Pattern

D3's core pattern manages the relationship between data and visual elements:

```javascript
const circles = svg.selectAll("circle")
    .data(dataset, d => d.id);

// ENTER: Create elements for new data
circles.enter()
    .append("circle")
    .attr("r", 5)
    .attr("cx", d => xScale(d.x))
    .attr("cy", d => yScale(d.y));

// UPDATE: Modify existing elements
circles.attr("cx", d => xScale(d.x))
       .attr("cy", d => yScale(d.y));

// EXIT: Remove elements for departed data
circles.exit().remove();
```

#### Scales and Axes

D3 scales map from data domain to visual range:

```javascript
const xScale = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)])
    .range([0, width]);
```

Axes are generated from scales:

```javascript
const xAxis = d3.axisBottom(xScale);
svg.append("g")
   .attr("transform", `translate(0,${height})`)
   .call(xAxis);
```

#### Transitions

D3 transitions animate changes:

```javascript
circles.transition()
    .duration(750)
    .ease(d3.easeCubicOut)
    .attr("cx", d => xScale(d.x));
```

---

## Part IV: Dashboards

### 4.1 Dashboard Design Principles

Effective dashboards follow interface design principles:

1. **Progressive disclosure**: Show overview first, details on demand.
2. **Consistency**: Use the same colours, fonts, and conventions throughout.
3. **Feedback**: Respond immediately to user actions.
4. **Undo/Reset**: Allow users to return to initial state.
5. **Guidance**: Provide labels, titles, and instructions.

### 4.2 Streamlit for Rapid Prototyping

Streamlit converts Python scripts to web applications:

```python
import streamlit as st

st.title("Research Dashboard")
category = st.selectbox("Category", options=["A", "B", "C"])
filtered = df[df["category"] == category]
st.plotly_chart(px.scatter(filtered, x="x", y="y"))
```

Benefits: Fast iteration, Python-only, automatic deployment to Streamlit Cloud.

Limitations: Limited customisation, not suitable for complex interactions.

---

## Part V: Reproducibility

### 5.1 Scripted Figure Generation

For reproducibility, figures should be generated by scripts, not interactive sessions:

```bash
python generate_figures.py --output figures/
```

The script should:
- Load data from versioned source
- Set random seeds if randomness is involved
- Save to multiple formats (PDF for vector, PNG for raster)
- Include metadata (date, script version) in file names

### 5.2 Version Control for Figures

Track figure-generating scripts in version control. Consider tracking output files if they are small enough, or use DVC (Data Version Control) for large assets.

---

## Summary

This week covered:

1. **Tufte's principles**: Data-ink ratio, lie factor, small multiples
2. **Grammar of Graphics**: Data, aesthetics, geometries, facets, statistics, coordinates, themes
3. **Static visualisation**: Matplotlib, Seaborn, publication standards
4. **Interactive visualisation**: Plotly, D3.js enter-update-exit, scales, transitions
5. **Dashboards**: Design principles, Streamlit
6. **Reproducibility**: Scripted generation, version control

Next week, we integrate these skills into the capstone project, focusing on reproducibility, testing and documentation.

---

## References

Cleveland, W. S., & McGill, R. (1984). Graphical perception: Theory, experimentation, and application to the development of graphical methods. *Journal of the American Statistical Association*, 79(387), 531-554.

Tufte, E. R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.

Wilkinson, L. (2005). *The Grammar of Graphics* (2nd ed.). Springer.

Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press.

---

*© 2025 Antonio Clim. All rights reserved. See README.md for licence terms.*

**Word count**: ~2,100 words
