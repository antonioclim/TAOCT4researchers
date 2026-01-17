# Week 6: Learning Objectives

## Overview

This document outlines the measurable learning objectives for Week 6: Visualisation for Research. Each objective is aligned with Bloom's Taxonomy and includes specific success criteria for assessment.

---

## Primary Objectives

### Objective 1: Create Publication-Quality Static Figures

**Bloom's Level**: Apply

**Statement**: Apply Matplotlib and Seaborn to create publication-quality static figures that conform to journal standards (Nature, Science, IEEE).

**Success Criteria**:
- [ ] Figures exported at ≥300 DPI for raster formats
- [ ] Vector formats (PDF, SVG, EPS) produced without rasterisation
- [ ] Fonts embedded correctly without external dependencies
- [ ] Figure dimensions match target journal specifications (single column: 3.5", double column: 7")
- [ ] Colour palette is accessible (colourblind-friendly)
- [ ] Data-ink ratio maximised (minimal chartjunk)
- [ ] Axes labelled with units and appropriate precision
- [ ] Legend positioned without obscuring data
- [ ] Code is reproducible (same input → identical output)

**Assessment Methods**:
- Lab 1 completion checklist
- Homework Exercise 1 (30 points)
- Peer review of submitted figures

---

### Objective 2: Build Interactive Dashboards

**Bloom's Level**: Create

**Statement**: Create interactive dashboards for data exploration using Streamlit, Plotly or HTML/JavaScript that enable filtering, brushing and linked views.

**Success Criteria**:
- [ ] Dashboard loads and displays data correctly
- [ ] Minimum 2 different visualisation types present
- [ ] Interactive filtering implemented (dropdown, slider, date range)
- [ ] Linked views: selection in one chart updates others
- [ ] Responsive design for different screen sizes
- [ ] Data updates reflected immediately in visualisations
- [ ] Deployed to accessible URL (Streamlit Cloud or GitHub Pages)
- [ ] Loading states and error handling implemented

**Assessment Methods**:
- Lab 2 completion checklist
- Homework Exercise 2 (35 points)
- Live demonstration during session

---

### Objective 3: Select Appropriate Visualisation Types

**Bloom's Level**: Evaluate

**Statement**: Evaluate data characteristics and research questions to select the most appropriate visualisation type from a catalogue of options.

**Success Criteria**:
- [ ] Correctly identify data types (categorical, continuous, temporal, hierarchical, network)
- [ ] Match visualisation type to data structure:
  - Trends over time → Line chart
  - Distribution → Histogram, box plot, violin plot
  - Comparison → Bar chart, grouped bar, small multiples
  - Correlation → Scatter plot, heatmap
  - Part-to-whole → Pie chart (sparingly), treemap, stacked bar
  - Hierarchical → Treemap, sunburst, dendrogram
  - Network → Force-directed graph, adjacency matrix
  - Geographic → Choropleth, bubble map
- [ ] Justify visualisation choice in writing
- [ ] Identify when a visualisation type is inappropriate
- [ ] Suggest alternatives when initial choice has limitations

**Assessment Methods**:
- Quiz questions 7-10
- Homework Exercise 3 design rationale
- Self-check questionnaire

---

### Objective 4: Apply Tufte's Principles

**Bloom's Level**: Analyse

**Statement**: Analyse existing visualisations using Tufte's principles to identify improvements and critique design decisions.

**Success Criteria**:
- [ ] Calculate data-ink ratio for a given visualisation
- [ ] Identify and remove chartjunk elements
- [ ] Detect lie factor issues (truncated axes, area/length distortion)
- [ ] Apply small multiples principle for comparison
- [ ] Recognise and correct common visualisation errors:
  - 3D effects without purpose
  - Excessive gridlines
  - Redundant encoding
  - Poor colour choices
  - Missing context
- [ ] Propose concrete improvements with justification

**Assessment Methods**:
- Quiz questions 1-6
- Lab 1 before/after comparison
- Peer review feedback

---

## Secondary Objectives

### Objective 5: Understand the Grammar of Graphics

**Bloom's Level**: Understand

**Statement**: Explain the seven components of the Grammar of Graphics and how they combine to create visualisations.

**Success Criteria**:
- [ ] Define each component: data, aesthetics, geometries, facets, statistics, coordinates, themes
- [ ] Map these components to code (ggplot2, Altair, Plotly)
- [ ] Explain how changing one component affects the visualisation
- [ ] Identify grammar components in existing visualisations

---

### Objective 6: Implement D3.js Fundamentals

**Bloom's Level**: Apply

**Statement**: Apply D3.js selections, scales and transitions to create custom interactive visualisations.

**Success Criteria**:
- [ ] Select and manipulate DOM elements using D3 selections
- [ ] Bind data to elements using the enter-update-exit pattern
- [ ] Create and use linear, ordinal, time and logarithmic scales
- [ ] Generate and customise axes
- [ ] Implement smooth transitions between states
- [ ] Handle user interaction events (hover, click)

**Assessment Methods**:
- Homework Exercise 3 (20 points)
- Optional advanced exercises

---

## Bloom's Taxonomy Mapping

| Level | Verb | Objectives Covered |
|-------|------|-------------------|
| Remember | Define, List | Secondary Objective 5 (partial) |
| Understand | Explain, Describe | Secondary Objective 5 |
| Apply | Implement, Create, Use | Objectives 1, 6 |
| Analyse | Critique, Identify | Objective 4 |
| Evaluate | Select, Justify, Judge | Objective 3 |
| Create | Design, Build, Produce | Objective 2 |

---

## Prerequisite Knowledge Check

Before starting Week 6, ensure you can:

- [ ] Load and manipulate data with Pandas
- [ ] Create basic plots with Matplotlib (`plt.plot`, `plt.scatter`, `plt.bar`)
- [ ] Write functions with type hints
- [ ] Understand basic HTML structure (`<div>`, `<script>`, `<style>`)
- [ ] Run Python scripts from the command line

---

## Connection to Course Outcomes

| Week 6 Objective | Course Learning Outcome |
|------------------|------------------------|
| Publication figures | Communicate research findings effectively |
| Interactive dashboards | Build tools for data exploration |
| Visualisation selection | Apply critical thinking to design decisions |
| Tufte's principles | Evaluate and improve technical artefacts |

---

## Resources for Each Objective

### Objective 1: Static Figures
- Matplotlib documentation: https://matplotlib.org/
- Seaborn gallery: https://seaborn.pydata.org/examples/
- Nature figure guidelines: https://www.nature.com/nature/for-authors/final-submission

### Objective 2: Interactive Dashboards
- Streamlit documentation: https://docs.streamlit.io/
- Plotly Express: https://plotly.com/python/plotly-express/
- Dash by Plotly: https://dash.plotly.com/

### Objective 3: Visualisation Selection
- Data Viz Catalogue: https://datavizcatalogue.com/
- From Data to Viz: https://www.data-to-viz.com/

### Objective 4: Tufte's Principles
- Tufte, E. (2001). The Visual Display of Quantitative Information
- Few, S. (2012). Show Me the Numbers

---

*© 2025 Antonio Clim. All rights reserved. See README.md for licence terms.*
