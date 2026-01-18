# 06UNIT: Lecture Notes — Visualisation for Research

## Introduction

Data visualisation constitutes the disciplined practice of representing quantitative and categorical information through graphical means. For researchers, it serves two interconnected purposes: exploration—the process of discovering patterns, anomalies and relationships within data—and communication—the conveyance of findings to audiences ranging from specialist peers to the general public. A visualisation of excellence renders insights accessible that tabular representations obscure; conversely, a flawed visualisation may mislead, confuse or actively distort understanding.

This unit examines the theoretical foundations of effective visualisation, drawing principally from Edward Tufte's corpus on graphical excellence and Leland Wilkinson's formal grammar of graphics. We then transition to practical implementation through Python's matplotlib and seaborn libraries for static figures, followed by Plotly and D3.js for interactive web-based representations. The pedagogical aim is twofold: to cultivate the critical faculty necessary for evaluating existing visualisations, and to develop the technical competencies required for producing publication-ready figures.

---

## Part I: Theoretical Foundations

### 1.1 Historical Development of Statistical Graphics

The systematic representation of quantitative data emerged in the late eighteenth century with William Playfair's pioneering innovations. His 1786 *Commercial and Political Atlas* introduced the line chart for displaying time-series data—specifically, England's trade balance over time. The 1801 *Statistical Breviary* followed with the first bar chart and pie chart, establishing fundamental chart types that persist to the present day.

The nineteenth century witnessed remarkable developments. Charles Joseph Minard's 1869 visualisation of Napoleon's Russian campaign remains widely regarded as perhaps the finest statistical graphic ever produced. This single image encodes six variables—army size, geographic position (latitude and longitude), direction of movement, temperature and date—demonstrating the potential for complex multivariate representation. Florence Nightingale employed polar area diagrams (often miscalled "coxcomb charts") to advocate for sanitary reform in military hospitals, illustrating how visualisation can drive policy change.

The twentieth century saw formalisation of graphical theory. Jacques Bertin's 1967 *Sémiologie Graphique* established a systematic vocabulary for visual variables (position, size, shape, value, colour, orientation and texture). Edward Tufte's 1983 *The Visual Display of Quantitative Information* synthesised principles of graphical excellence that remain foundational. Leland Wilkinson's 1999 *The Grammar of Graphics* provided a formal specification language subsequently implemented in ggplot2, Altair and Vega-Lite.

### 1.2 Tufte's Principles of Graphical Excellence

Edward Tufte, often termed the "Galileo of graphics," articulated principles guiding practitioners toward integrity and clarity. His work identifies both positive attributes to cultivate and negative tendencies to avoid.

#### The Data-Ink Ratio

Tufte defines "data-ink" as the non-erasable, non-redundant portion of a graphic—the ink that directly represents data values. All other ink constitutes potential waste. The data-ink ratio formalises this notion:

$$\text{Data-Ink Ratio} = \frac{\text{Data-Ink}}{\text{Total Ink Used in Graphic}} = 1 - \text{Proportion of Erasable Ink}$$

A ratio approaching unity indicates a graphic where nearly every visual element encodes information. The principle advocates systematically removing elements whose absence would not diminish information content.

Consider a typical spreadsheet-generated bar chart: heavy gridlines extending across the entire plot area, three-dimensional bevels creating false depth, gradient fills suggesting lighting effects and drop shadows implying elevation. Each element consumes rendering resources—whether ink or pixels—without encoding data. Stripping these decorative flourishes produces a cleaner graphic directing attention solely to the comparative heights of the bars, which is the actual information content.

**Practical application**: Before finalising any figure, interrogate each visual element with the question: "If I remove this, do I lose information?" Negative answers indicate candidates for elimination.

#### Chartjunk

Tufte coined "chartjunk" to describe decorative elements adding visual interest without information value. Categories include:

1. **Grid backgrounds**: Heavily shaded or patterned backgrounds that compete with data.
2. **Pseudo-3D effects**: Perspective distortions that make accurate comparison impossible.
3. **Pictorial embellishments**: Clip art, icons or decorative imagery surrounding data.
4. **Redundant encoding**: The same value shown through both bar height and numerical label.

The term carries deliberate pejorative force. Chartjunk is not merely superfluous; it actively harms comprehension by splitting attention and introducing perceptual confounds.

#### The Lie Factor

The lie factor quantifies distortion in graphical representation—the degree to which visual impression departs from underlying data:

$$\text{Lie Factor} = \frac{\text{Size of Effect Shown in Graphic}}{\text{Size of Effect in Data}}$$

A lie factor of exactly 1.0 indicates perfect representational fidelity. Values substantially deviating from unity—conventionally, outside the range [0.95, 1.05]—indicate misleading visual rhetoric, whether intentional or inadvertent.

**Example**: Suppose corporate revenue increased from £98 million to £102 million, representing a 4.08% increase. If the accompanying graphic depicts the second bar as twice the height of the first (a 100% visual increase), the lie factor computes to approximately 24.5. Such distortion grossly exaggerates the underlying change, potentially misleading stakeholders about performance magnitude.

Common sources of distortion include truncated axes (where the y-axis begins substantially above zero, making small differences appear dramatic), area encoding (where doubling both width and height of a symbol creates a quadrupled visual effect for a doubled value) and perspective projections (where three-dimensional rendering distorts relative sizes based on apparent depth).

#### Small Multiples

Small multiples constitute a series of similar graphics, employing identical scales and axes, displayed together to enable direct comparison. Rather than overlaying numerous data series on a single plot—risking visual clutter and perceptual confusion—small multiples allocate each series its own panel within a consistent grid.

This technique excels in several contexts: comparing trends across categorical groupings, displaying geographic variation across regions or nations, showing evolution over discrete time periods, and revealing differences between experimental conditions. The critical constraint is consistency: all panels must share precisely identical scales to permit valid comparison. Inconsistent scales defeat the purpose of juxtaposition.

Tufte attributes the power of small multiples to enabling "within-eyespan comparisons"—viewers need not rely on memory to compare values across panels but can instead perceive differences through direct visual juxtaposition.

#### Micro/Macro Readings

Effective visualisations support interpretation at multiple levels simultaneously. At the macro level, viewers should immediately apprehend the overall pattern, trend or message—the "headline" of the graphic. At the micro level, they should be able to extract specific data values for precise comparison or quotation. Both requirements must be satisfied by a single graphic without sacrificing either capability.

This principle has implications for labelling density, legend placement and data point marking. A scatter plot might communicate the macro pattern (positive correlation, clustering) whilst also permitting identification of specific outlying observations at the micro level.

### 1.3 The Grammar of Graphics

Leland Wilkinson's *Grammar of Graphics* provides a formal framework for specifying statistical visualisations. This grammar decomposes any graphic into component layers, each performing a distinct transformation. The key insight is that visualisations are not monolithic artefacts but structured compositions whose components can be independently specified, modified and recombined.

The canonical components are:

1. **DATA**: The dataset to be visualised, typically in tabular (data frame) form with observations as rows and variables as columns.

2. **AESTHETICS (aes)**: Mappings from data variables to visual properties. These specify that, for example, the x-position encodes time, the y-position encodes temperature, colour encodes measurement station, and size encodes sample size.

3. **GEOMETRIES (geom)**: The visual elements representing data points—whether points (scatterplot), lines (time series), bars (barcharts), areas (stacked areas), or more exotic geometries like density contours or violin shapes.

4. **FACETS**: Subdivision of the visualisation into panels for comparison—a formalisation of small multiples. Faceting variables determine how the data is partitioned across panels.

5. **STATISTICS (stat)**: Transformations applied to data before rendering. These include binning (for histograms), smoothing (for trend lines), aggregation (for summary statistics), and density estimation.

6. **COORDINATES (coord)**: The coordinate system governing spatial position. Options include Cartesian (standard x-y), polar (for pie charts and radar plots), and geographic (for maps).

7. **THEMES**: Non-data elements controlling general appearance—fonts, background colours, gridline styles, legend positioning. These affect aesthetics without altering data representation.

**Implementation**: Libraries implementing this grammar include ggplot2 (R), Altair (Python), Vega-Lite (JavaScript) and plotnine (Python). Even when using matplotlib, which does not implement the grammar directly, thinking in terms of these components clarifies design decisions and supports clear communication about visualisation choices.

### 1.4 Perception and Cognition

Human visual perception exhibits characteristics that constrain and inform effective visualisation design. Understanding these characteristics enables practitioners to make principled choices about encoding data in visual form.

#### Pre-attentive Processing

Certain visual features are processed pre-attentively—without conscious effort, in under 250 milliseconds. Research in perceptual psychology has identified these features as including position, length, width, colour hue, colour intensity, orientation, shape, curvature, enclosure, and motion. Pre-attentively processed features "pop out" from their surroundings; a red circle among blue circles is immediately perceived without serial search.

Using pre-attentive features to encode important distinctions ensures that viewers perceive them immediately and effortlessly. Conversely, features requiring serial search or conscious cognitive effort (such as comparing areas of irregular shapes) should be reserved for less critical distinctions.

#### Cleveland and McGill's Hierarchy

William Cleveland and Robert McGill's 1984 empirical research established a rank ordering of visual encodings by accuracy of human perception. Their experiments asked participants to estimate ratios of graphically encoded values, measuring error rates across encoding methods:

1. **Position along a common scale**: Most accurate—comparing points along a shared axis.
2. **Position on identical but unaligned scales**: Slightly less accurate—comparing across separate but identical axes.
3. **Length**: Comparing lengths of bars or segments.
4. **Angle/Slope**: Comparing angular orientations.
5. **Area**: Comparing sizes of two-dimensional shapes.
6. **Volume, density, colour saturation**: Least accurate.

**Practical implication**: Encode your most important variable using position. Reserve area encoding for approximate categorical distinction; reserve colour saturation for continuous variation where precision is unimportant.

#### Colour Perception and Accessibility

Colour choices affect both aesthetics and accessibility. Approximately 8% of men and 0.5% of women experience some form of colour vision deficiency, most commonly red-green confusion (deuteranopia and protanopia).

**Colourblind-friendly palettes**: Palettes designed for accessibility avoid relying on red-green distinctions. The ColorBrewer palettes (developed by Cynthia Brewer), Viridis (developed for matplotlib), and the Wong palette (proposed in Nature Methods) all satisfy accessibility requirements.

**Sequential versus diverging**: Sequential palettes (progressing from light to dark or through a single hue gradient) suit continuous data without a meaningful midpoint. Diverging palettes (with two distinct hues meeting at a neutral centre) suit data with a meaningful midpoint, such as temperature anomalies from a baseline or political polling margins.

**Print compatibility**: Visualisations destined for print publication must remain interpretable when photocopied or printed in greyscale. This requirement motivates using shape or pattern distinctions alongside colour.

---

## Part II: Static Visualisation with Python

### 2.1 Matplotlib Architecture

Matplotlib, Python's foundational plotting library, follows an object-oriented architecture comprising three layers:

1. **Backend layer**: Handles rendering to different output targets—interactive window displays, raster image files (PNG, JPEG), vector image files (PDF, SVG, EPS), and embedded applications.

2. **Artist layer**: Contains the visual primitives that draw on the canvas—Line2D objects for line segments, Patch objects for filled shapes, Text objects for labels and annotations. Every visible element in a matplotlib figure is an Artist.

3. **Scripting layer (pyplot)**: A stateful MATLAB-style interface for rapid prototyping. The pyplot module maintains implicit "current figure" and "current axes" objects, allowing abbreviated commands like `plt.plot()` without explicit object references.

For publication figures, the object-oriented interface offers superior control:

```python
fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Create figure and axes
ax.plot(x, y, color='#0072B2', linewidth=1.5)  # Plot on specific axes
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (mV)")
ax.spines['top'].set_visible(False)  # Remove chartjunk
ax.spines['right'].set_visible(False)
fig.savefig("figure.pdf", dpi=300, bbox_inches='tight')
```

### 2.2 Publication Standards

Academic journals impose specific requirements for submitted figures. Failure to comply results in revision requests or rejection.

| Journal | Single Column | Double Column | Font | Minimum DPI |
|---------|--------------|---------------|------|-------------|
| Nature | 89mm | 183mm | Sans-serif, 7-8pt | 300 |
| Science | 85mm | 180mm | Sans-serif, 7pt | 300 |
| IEEE | 88mm | 181mm | Serif, 8pt | 600 |
| Cell | 85mm | 174mm | Arial, 6-8pt | 300 |
| PNAS | 87mm | 178mm | Sans-serif, 6-8pt | 300 |

Universal requirements include embedded fonts (ensuring rendering fidelity across systems), legible text at final print size (often requiring font sizes of 7-8pt), accessible colour choices (avoiding pure red-green distinctions), and sufficient resolution for raster components (300 DPI minimum, 600 DPI preferred).

### 2.3 Seaborn for Statistical Graphics

Seaborn provides high-level functions for statistical visualisations, built atop matplotlib and integrated with pandas DataFrames:

- `sns.histplot()`: Histograms with optional kernel density estimation overlay
- `sns.boxplot()`, `sns.violinplot()`: Distribution comparison across categories
- `sns.scatterplot()`: Scatter plots with automatic grouping by categorical variables
- `sns.heatmap()`: Correlation matrices and two-dimensional data grids
- `sns.pairplot()`: Pairwise relationships across all numeric variables

Seaborn's defaults emphasise statistical clarity and aesthetic refinement, though customisation for publication often requires reverting to matplotlib's lower-level controls.

---

## Part III: Interactive Visualisation

### 3.1 The Case for Interactivity

Interactive visualisations enable exploratory operations impossible in static media:

- **Filtering**: Display subsets matching user-specified criteria dynamically.
- **Details on demand**: Reveal additional information upon hover or click without cluttering the base view.
- **Brushing and linking**: Selections in one view highlight corresponding data points in coordinated views.
- **Zooming and panning**: Navigate datasets too large or detailed for single-viewport display.
- **Animation**: Reveal temporal evolution or parameter sensitivity through motion.

For research purposes, interactive visualisations serve during exploration (hypothesis generation, anomaly detection) and for communicating complex findings (supplementary materials, online publications, stakeholder presentations).

### 3.2 D3.js Fundamentals

D3.js (Data-Driven Documents) provides complete control over interactive web visualisations through declarative binding of data to DOM elements. Unlike chart libraries offering pre-built types, D3 furnishes primitives from which custom visualisations are constructed.

#### The Enter-Update-Exit Pattern

D3's core pattern manages the correspondence between data elements and visual elements:

```javascript
const circles = svg.selectAll("circle")
    .data(dataset, d => d.id);  // Bind data with key function

// ENTER: Create elements for new data points
circles.enter()
    .append("circle")
    .attr("r", 5)
    .attr("cx", d => xScale(d.x))
    .attr("cy", d => yScale(d.y));

// UPDATE: Modify existing elements for changed data
circles.attr("cx", d => xScale(d.x))
       .attr("cy", d => yScale(d.y));

// EXIT: Remove elements for departed data points
circles.exit().remove();
```

This pattern enables smooth transitions as data changes, supporting animated updates that help viewers track identity through transformation.

---

## Part IV: Reproducibility and Established Methods

### 4.1 Scripted Figure Generation

For reproducibility, figures should be generated through scripts executed in controlled environments rather than interactive sessions:

```bash
python generate_figures.py --output figures/ --config publication.yaml
```

Scripts should load data from versioned sources, set random seeds where randomness is involved, export to multiple formats (PDF for vector, PNG for raster), and include metadata (date, script version, commit hash) enabling traceability.

### 4.2 Version Control Integration

Figure-generating scripts belong in version control alongside analysis code. Large output files may warrant external management through DVC (Data Version Control) or similar tools, maintaining linkage to generating code without bloating repositories.

---

## Summary

This unit established theoretical and practical foundations for research visualisation:

1. **Tufte's principles**: Data-ink ratio, lie factor, chartjunk elimination, small multiples
2. **Grammar of Graphics**: Data, aesthetics, geometries, facets, statistics, coordinates, themes
3. **Perceptual science**: Pre-attentive processing, Cleveland-McGill hierarchy, colour accessibility
4. **Static visualisation**: Matplotlib architecture, publication standards, seaborn statistical graphics
5. **Interactive visualisation**: Plotly, D3.js enter-update-exit pattern, scales and transitions
6. **Reproducibility**: Scripted generation, version control, traceability

The subsequent unit (07UNIT) integrates these capabilities into capstone projects, addressing testing, documentation and deployment.

---

## References

Cleveland, W.S. & McGill, R. (1984). Graphical perception: Theory, experimentation, and application to the development of graphical methods. *Journal of the American Statistical Association*, 79(387), 531-554.

Tufte, E.R. (2001). *The Visual Display of Quantitative Information* (2nd ed.). Graphics Press.

Wilkinson, L. (2005). *The Grammar of Graphics* (2nd ed.). Springer.

Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press.

Healy, K. (2018). *Data Visualization: A Practical Introduction*. Princeton University Press.

---

*© 2025 Antonio Clim. All rights reserved. See README.md for licence terms.*
