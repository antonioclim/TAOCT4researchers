# Week 6 Glossary: Visualisation for Research

## A

### Aesthetic Mapping
The process of linking data variables to visual properties such as position, colour, size or shape. In the Grammar of Graphics, aesthetics define how data values are encoded visually. Example: mapping a continuous variable to the y-axis position.

### Alpha (Transparency)
A value between 0 (fully transparent) and 1 (fully opaque) controlling the visibility of graphical elements. Useful for showing overlapping data points.

### Animation
A sequence of frames displayed in rapid succession to create the illusion of motion. In matplotlib, created using `FuncAnimation`. Used to show temporal evolution of data.

### Annotation
Text or graphical elements added to a plot to highlight specific features. In matplotlib: `ax.annotate()`. Should be used sparingly following Tufte's principles.

### Axes
The matplotlib object representing a single plot area within a figure. Contains the plotting methods and manages scales, labels and legends. Not to be confused with 'axis' (singular).

---

## B

### Blit (Blitting)
An optimisation technique for animations where only changed elements are redrawn. Set `blit=True` in `FuncAnimation` for improved performance.

### Bbox (Bounding Box)
The rectangular region enclosing a graphical element. `bbox_inches='tight'` in `savefig()` crops the figure to content.

---

## C

### Chartjunk
A term coined by Tufte for visual elements that do not convey data information. Includes unnecessary 3D effects, decorative patterns and excessive gridlines. Should be minimised.

### Cleveland-McGill Hierarchy
Ranking of visual encodings by perceptual accuracy: position > length > angle > area > colour > volume. Guides selection of appropriate visualisation types.

### Colourbar
A legend showing the mapping between numerical values and colours in a visualisation. Added with `plt.colorbar()`.

### Colourblind-Friendly Palette
A set of colours distinguishable by people with colour vision deficiencies. The Wong palette is the standard reference in scientific visualisation.

### Colourmap (Colormap)
A function mapping numerical values to colours. Types: sequential (single hue gradient), diverging (two-hue with neutral centre), qualitative (distinct categorical colours).

### Constrained Layout
Matplotlib's automatic layout engine that prevents overlapping elements. Enabled with `constrained_layout=True` or `plt.tight_layout()`.

---

## D

### D3.js
Data-Driven Documents: a JavaScript library for creating interactive web visualisations. Uses a declarative approach to bind data to DOM elements.

### Data-Ink Ratio
Tufte's measure of graphical efficiency: proportion of ink used to display actual data versus total ink. Higher is better.

### Declarative
A programming approach where you specify *what* you want rather than *how* to achieve it. D3.js and ggplot2 use declarative styles.

### Deuteranopia
Colour blindness affecting green cone perception. The most common form, affecting ~6% of males. Red and green appear similar.

### DPI (Dots Per Inch)
Resolution measure for raster images. Journal requirements typically range from 300 DPI (Nature) to 600 DPI (IEEE).

---

## E

### Enter-Update-Exit Pattern
D3.js data binding pattern handling:
- **Enter**: New data points (create elements)
- **Update**: Existing data points (modify elements)
- **Exit**: Removed data points (delete elements)

### EPS (Encapsulated PostScript)
A vector graphics format commonly required by journals. Preserves quality at any scale.

---

## F

### Facet (Faceting)
Splitting data into multiple panels based on a categorical variable. Implements Tufte's "small multiples" concept. In Seaborn: `FacetGrid`.

### Figure
The top-level matplotlib container holding one or more axes. Created with `plt.figure()` or `plt.subplots()`.

### FuncAnimation
Matplotlib class for creating animations by repeatedly calling an update function. Key parameters: `frames`, `interval`, `blit`.

---

## G

### Grammar of Graphics
A theoretical framework describing graphics as combinations of data, aesthetic mappings, geometric objects, scales, coordinates and facets. Foundation of ggplot2 and modern visualisation libraries.

### GridSpec
Matplotlib class for creating complex subplot layouts with varying sizes. More flexible than basic `subplots()`.

---

## H

### Heatmap
A visualisation showing values in a matrix using colour intensity. Useful for correlation matrices and 2D distributions.

### Hover Template
In Plotly, a format string controlling what appears when hovering over data points. Uses placeholders like `%{x}` and `%{text}`.

---

## I

### Interactive Visualisation
Graphics allowing user interaction such as zooming, panning, filtering or hovering for details. Created with Plotly, D3.js or Bokeh.

---

## J

### Join (D3.js)
Modern D3.js method combining enter, update and exit into a single operation. Simplifies the data binding pattern.

---

## L

### Legend
A key explaining the meaning of visual encodings (colours, shapes, sizes). Position with `loc` parameter or `bbox_to_anchor`.

### Lie Factor
Tufte's measure of graphical integrity: (size of effect shown in graphic) / (size of effect in data). Should equal 1.0.

---

## M

### Micro/Macro Readings
Tufte's concept that good visualisations support both detailed examination (micro) and overall pattern recognition (macro).

### Minard Chart
Charles Joseph Minard's 1869 visualisation of Napoleon's Russian campaign, often cited as one of the best statistical graphics ever made.

---

## O

### Object-Oriented Interface
Matplotlib's recommended API where you explicitly create Figure and Axes objects. Provides more control than the pyplot (plt) interface.

---

## P

### Panel Labels
Letters (A, B, C...) identifying subplots in multi-panel figures. Placed in upper-left corner of each panel.

### Perceptual Uniformity
Property of colourmaps where equal numerical differences produce equal perceived colour differences. Viridis is perceptually uniform; rainbow is not.

### Plotly
A Python library for interactive visualisations. Express API for quick plots; Graph Objects for full control.

### Protanopia
Colour blindness affecting red cone perception. Red appears darker and greenish.

### Publication-Ready Figure
A visualisation meeting journal requirements for dimensions, fonts, resolution and file format.

---

## R

### rcParams
Matplotlib's runtime configuration parameters. Set globally with `plt.rcParams.update({})` to ensure consistent styling.

### Reproducibility
The ability to regenerate identical figures from the same data and code. Requires: random seeds, scripted generation, documented dependencies.

---

## S

### Scale (D3.js)
A function mapping data domain to visual range. Types: `scaleLinear` (continuous), `scaleBand` (categorical), `scaleTime` (temporal).

### Seaborn
A statistical visualisation library built on matplotlib. Provides higher-level functions like `FacetGrid` and built-in themes.

### Small Multiples
A series of similar charts showing different subsets of data. Enables comparison across conditions. Coined by Tufte.

### Spine
The lines forming the boundary of an axes area. Tufte style removes top and right spines for cleaner appearance.

### Streamlit
A Python framework for creating interactive web dashboards. Enables rapid prototyping of data applications.

### SVG (Scalable Vector Graphics)
A vector image format based on XML. Maintains quality at any scale; ideal for web and presentations.

---

## T

### Tick (Tick Mark)
Small marks on axes indicating values. Configure with `ax.tick_params()`. Outward ticks are less intrusive.

### Transition (D3.js)
Animated change between states. Created with `.transition().duration(ms)`. Supports easing functions.

### Tritanopia
Rare colour blindness affecting blue cone perception. Blue and yellow are confused.

### Tufte, Edward
Information design pioneer. Author of "The Visual Display of Quantitative Information". Introduced data-ink ratio, chartjunk, small multiples.

### Type Hints
Python annotations indicating expected types. Example: `def plot(x: np.ndarray) -> Figure`. Required for lab code.

---

## V

### Vector Graphics
Image format storing shapes as mathematical descriptions rather than pixels. Includes PDF, SVG, EPS. Infinitely scalable.

### ViewBox (SVG)
Attribute defining the coordinate system for SVG graphics. Enables responsive scaling.

### Viridis
A perceptually uniform, colourblind-friendly colourmap. The default in matplotlib since version 2.0.

---

## W

### Wong Palette
An 8-colour palette designed by Bang Wong for colourblind accessibility. Colours: blue (#0072B2), orange (#E69F00), green (#009E73), pink (#CC79A7), yellow (#F0E442), light blue (#56B4E9), vermillion (#D55E00), black (#000000).

---

## Related Terms from Previous Weeks

| Week | Term | Connection to Week 6 |
|------|------|---------------------|
| 5 | Monte Carlo | Convergence plots, error visualisation |
| 5 | Agent-based model | Animation of emergent behaviour |
| 5 | ODE solver | Time series plots comparing methods |
| 4 | Graph | Network visualisation with D3.js |
| 3 | Complexity | Algorithm comparison charts |

---

© 2025 Antonio Clim. All rights reserved.
