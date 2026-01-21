# Week 6 Quiz: Visualisation for Research

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Total Questions** | 10 |
| **Multiple Choice** | 6 |
| **Short Answer** | 4 |
| **Time Limit** | 20 minutes |
| **Passing Score** | 70% |

## 🎯 Learning Objectives Assessed

1. Apply publication-quality figure standards
2. Select appropriate visualisation types
3. Implement colourblind-friendly palettes
4. Understand D3.js data binding concepts

---

## Multiple Choice Questions (6 × 10 points = 60 points)

### Question 1

What is the recommended single-column figure width for Nature publications?

- A) 85 mm
- B) 89 mm
- C) 140 mm
- D) 183 mm

---

### Question 2

According to Tufte's principles, the "data-ink ratio" refers to:

- A) The proportion of ink used for axes versus data points
- B) The proportion of ink devoted to displaying data versus non-data elements
- C) The number of colours used divided by the number of data series
- D) The ratio of figure width to height

---

### Question 3

Which of the following colour palettes is specifically designed for colourblind accessibility?

- A) Rainbow (jet)
- B) Spectral
- C) Wong palette
- D) Hot

---

### Question 4

In D3.js, the "enter-update-exit" pattern is used for:

- A) Handling page navigation
- B) Data binding and DOM element management
- C) Animation timing sequences
- D) Server-side rendering

---

### Question 5

Which matplotlib interface is recommended for publication-quality figures?

- A) pyplot (plt) interface only
- B) Object-oriented (Figure, Axes) interface
- C) pylab interface
- D) Interactive MATLAB mode

---

### Question 6

What is the "lie factor" in data visualisation?

- A) The percentage of missing data points
- B) The ratio of data values to their visual representation size
- C) The number of misleading annotations
- D) The statistical p-value threshold

---

## Short Answer Questions (4 × 10 points = 40 points)

### Question 7

Explain the concept of "small multiples" and provide one example of when this technique would be appropriate in research visualisation.

*Your answer (3-4 sentences):*

---

### Question 8

What are the three rules (separation, alignment and cohesion) in the Boids flocking algorithm and how do they relate to visualising emergent behaviour?

*Your answer (3-4 sentences):*

---

### Question 9

Describe two key differences between static visualisations (matplotlib) and interactive visualisations (Plotly/D3.js) in terms of their use cases in research.

*Your answer (3-4 sentences):*

---

### Question 10

Why is reproducibility important in figure generation and what programming practices help achieve it?

*Your answer (3-4 sentences):*

---

## ✅ Submission Instructions

1. Complete all questions within the time limit
2. For multiple choice: select one answer only
3. For short answer: provide concise, well-structured responses
4. Review your answers before submission

---

<details>
<summary>📝 Answer Key (Instructor Use Only)</summary>

### Multiple Choice Answers

| Question | Answer | Explanation |
|----------|--------|-------------|
| 1 | **B** | Nature requires 89 mm for single column, 183 mm for double column |
| 2 | **B** | Data-ink ratio measures how much ink displays actual data versus chartjunk |
| 3 | **C** | Wong palette uses 8 colours optimised for all forms of colour vision deficiency |
| 4 | **B** | Enter-update-exit manages DOM elements based on data array changes |
| 5 | **B** | OO interface provides finer control needed for publication quality |
| 6 | **B** | Lie factor = (size of effect in graphic) / (size of effect in data) |

### Short Answer Rubric

**Question 7 — Small Multiples (10 points)**

Key elements:
- Definition: Series of similar charts showing different subsets/conditions (3 pts)
- Same scales/axes for comparison (2 pts)
- Example: Time series across multiple variables, faceted by category (3 pts)
- Reference to Tufte or Cleveland's work (2 pts)

Sample answer: "Small multiples are a series of similar charts arranged in a grid, each showing a different subset of the data but using identical scales and axes. This technique enables viewers to compare patterns across conditions efficiently. An appropriate use case would be showing climate data across different regions, where each panel displays temperature trends for one country, allowing researchers to identify regional patterns at a glance."

**Question 8 — Boids Algorithm (10 points)**

Key elements:
- Separation: avoid crowding neighbours (3 pts)
- Alignment: steer towards average heading (3 pts)
- Cohesion: steer towards centre of mass (3 pts)
- Connection to emergence/collective behaviour (1 pt)

Sample answer: "The Boids algorithm simulates flocking through three rules: separation (agents avoid crowding nearby neighbours), alignment (agents steer towards the average heading of nearby agents), and cohesion (agents move towards the centre of mass of nearby agents). These simple local rules produce emergent collective behaviour that resembles real flocking, demonstrating how complex patterns arise from simple interactions—a key insight for visualising agent-based models."

**Question 9 — Static vs Interactive (10 points)**

Key elements:
- Static: publication, archival, print media (2 pts)
- Interactive: exploration, dashboards, web (2 pts)
- File format differences (2 pts)
- User engagement differences (2 pts)
- Appropriate context for each (2 pts)

Sample answer: "Static visualisations (matplotlib) are ideal for publications and archival purposes as they produce vector formats (PDF, SVG) that maintain quality at any scale and ensure reproducibility. Interactive visualisations (Plotly/D3.js) excel for exploratory data analysis and dashboards, allowing users to zoom, filter and hover for details. In research, static figures are used for journal submissions where interactivity is impossible, while interactive versions can supplement online materials for deeper exploration."

**Question 10 — Reproducibility (10 points)**

Key elements:
- Importance: verification, collaboration, credibility (3 pts)
- Random seed setting (2 pts)
- Scripted generation (not manual) (2 pts)
- Version control of code/data (2 pts)
- Documentation of parameters (1 pt)

Sample answer: "Reproducibility ensures that figures can be regenerated exactly, enabling verification of results and facilitating collaboration. Key practices include: setting random seeds for any stochastic elements, generating figures programmatically through scripts rather than manual tools, tracking code and data in version control systems, and documenting all parameters including library versions. These practices support the scientific principle that results should be independently verifiable."

</details>

---

© 2025 Antonio Clim. All rights reserved.
