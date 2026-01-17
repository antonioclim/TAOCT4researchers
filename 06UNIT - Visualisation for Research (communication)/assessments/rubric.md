# Week 6 Grading Rubric: Visualisation for Research

## 📋 Overview

This rubric provides standardised criteria for evaluating visualisation assignments.
All work is assessed across five dimensions with consistent expectations.

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Technical Correctness | 25% | Code functionality and accuracy |
| Visual Design | 25% | Adherence to design principles |
| Accessibility | 15% | Colourblind-friendly, readable |
| Documentation | 20% | Comments, docstrings, explanations |
| Reproducibility | 15% | Scripts, seeds, version tracking |

---

## 📊 Detailed Criteria

### 1. Technical Correctness (25 points)

| Grade | Points | Criteria |
|-------|--------|----------|
| **Excellent** | 23-25 | Code runs without errors. All functionality implemented correctly. Edge cases handled. Efficient implementations. |
| **Good** | 18-22 | Code runs with minor issues. Most functionality correct. Some edge cases missed. Reasonable efficiency. |
| **Satisfactory** | 13-17 | Code runs with modifications. Core functionality works. Multiple edge cases missed. |
| **Needs Improvement** | 8-12 | Code has significant errors. Some functionality missing. Limited testing evident. |
| **Unsatisfactory** | 0-7 | Code does not run. Major functionality missing. No error handling. |

**Specific Checks:**
- [ ] All imports resolve correctly
- [ ] Functions return expected types
- [ ] No runtime errors with sample data
- [ ] Handles empty/missing data gracefully
- [ ] Correct mathematical calculations

---

### 2. Visual Design (25 points)

| Grade | Points | Criteria |
|-------|--------|----------|
| **Excellent** | 23-25 | Publication-ready figures. Excellent use of Tufte principles. Appropriate chart types. Clear visual hierarchy. |
| **Good** | 18-22 | High-quality figures. Good application of design principles. Minor improvements possible. |
| **Satisfactory** | 13-17 | Acceptable figures. Basic design principles followed. Some chartjunk or unclear elements. |
| **Needs Improvement** | 8-12 | Below standard figures. Poor chart type choices. Unclear labels or legends. |
| **Unsatisfactory** | 0-7 | Unreadable or misleading figures. Design principles ignored. Inappropriate visualisation choices. |

**Tufte Principles Checklist:**
- [ ] High data-ink ratio (minimal chartjunk)
- [ ] Lie factor close to 1.0
- [ ] Clear labelling without redundancy
- [ ] Appropriate use of small multiples
- [ ] Micro/macro readings enabled

**Journal Compliance Checklist:**
- [ ] Correct dimensions for target journal
- [ ] Appropriate font family and size
- [ ] Sufficient resolution (DPI)
- [ ] Vector format where required

---

### 3. Accessibility (15 points)

| Grade | Points | Criteria |
|-------|--------|----------|
| **Excellent** | 14-15 | Full colourblind accessibility. Wong palette or equivalent. Multiple encoding channels. Alt text provided. |
| **Good** | 11-13 | Good accessibility. Colourblind-safe palette. Most elements distinguishable. |
| **Satisfactory** | 8-10 | Basic accessibility. Some attention to colour choices. Minor issues remain. |
| **Needs Improvement** | 5-7 | Limited accessibility. Problematic colour combinations. Single encoding channel. |
| **Unsatisfactory** | 0-4 | No accessibility consideration. Red-green combinations. Indistinguishable elements. |

**Accessibility Checklist:**
- [ ] Colourblind-safe palette used (Wong, Okabe-Ito, viridis)
- [ ] No reliance on colour alone (use shape, pattern, label)
- [ ] Sufficient contrast ratios
- [ ] Readable font sizes (≥7pt for print)
- [ ] Clear legends with distinct markers

---

### 4. Documentation (20 points)

| Grade | Points | Criteria |
|-------|--------|----------|
| **Excellent** | 18-20 | Comprehensive docstrings. Clear inline comments. README explains usage. Examples provided. |
| **Good** | 14-17 | Good docstrings. Helpful comments. README present. Most elements documented. |
| **Satisfactory** | 10-13 | Basic docstrings. Some comments. Minimal README. Key elements documented. |
| **Needs Improvement** | 6-9 | Incomplete docstrings. Few comments. Missing README. Documentation gaps. |
| **Unsatisfactory** | 0-5 | No docstrings. No comments. No README. Undocumented code. |

**Documentation Requirements:**
- [ ] Google-style docstrings for all functions
- [ ] Type hints on function signatures
- [ ] Module-level docstring with purpose
- [ ] Inline comments for complex logic
- [ ] README with setup and usage instructions

---

### 5. Reproducibility (15 points)

| Grade | Points | Criteria |
|-------|--------|----------|
| **Excellent** | 14-15 | Fully reproducible. Seeds set. requirements.txt provided. Script-based generation. Versioned outputs. |
| **Good** | 11-13 | Mostly reproducible. Seeds for random elements. Dependencies listed. Minor gaps. |
| **Satisfactory** | 8-10 | Partially reproducible. Some seeds set. Basic dependency info. Manual steps needed. |
| **Needs Improvement** | 5-7 | Limited reproducibility. Missing seeds. Unclear dependencies. Significant manual work. |
| **Unsatisfactory** | 0-4 | Not reproducible. No seeds. No dependency info. Interactive/manual generation only. |

**Reproducibility Checklist:**
- [ ] Random seeds set and documented
- [ ] requirements.txt or pyproject.toml provided
- [ ] All figures generated by scripts (not interactive)
- [ ] Clear instructions for regeneration
- [ ] Data sources documented or included

---

## 🎯 Assignment-Specific Criteria

### Homework Part 1: Publication Figures (40 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Journal compliance | 10 | Correct dimensions, fonts, DPI |
| Multi-panel layout | 10 | GridSpec usage, panel labels |
| Statistical accuracy | 10 | Correct error bars, annotations |
| Export formats | 10 | PDF, PNG, SVG all generated |

### Homework Part 2: Interactive Dashboard (35 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Plotly implementation | 10 | Correct use of graph_objects/express |
| Interactivity | 10 | Hover, zoom, filter working |
| Dashboard layout | 10 | Streamlit/Panel structure |
| Responsiveness | 5 | Works at different screen sizes |

### Homework Part 3: D3.js Visualisation (20 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| Data binding | 5 | Correct enter-update-exit |
| Scales and axes | 5 | Proper D3 scale usage |
| Transitions | 5 | Smooth animations |
| Styling | 5 | CSS integration, dark theme |

### Bonus: Animation (15 points)

| Criterion | Points | Description |
|-----------|--------|-------------|
| FuncAnimation | 5 | Correct implementation |
| Performance | 5 | blit=True, efficient updates |
| Export | 5 | GIF/MP4 generation |

---

## 📝 Feedback Template

```markdown
## Week 6 Submission Feedback

**Student:** [Name]
**Submission Date:** [Date]
**Total Score:** [X]/100

### Dimension Scores

| Dimension | Score | Max |
|-----------|-------|-----|
| Technical Correctness | | 25 |
| Visual Design | | 25 |
| Accessibility | | 15 |
| Documentation | | 20 |
| Reproducibility | | 15 |

### Strengths
-

### Areas for Improvement
-

### Specific Comments

#### Technical:

#### Design:

#### Accessibility:

#### Documentation:

#### Reproducibility:

### Grade: [A/B/C/D/F]
```

---

## 📈 Grade Boundaries

| Grade | Percentage | Points |
|-------|------------|--------|
| A (Excellent) | 90-100% | 90-100 |
| B (Good) | 75-89% | 75-89 |
| C (Satisfactory) | 60-74% | 60-74 |
| D (Needs Improvement) | 50-59% | 50-59 |
| F (Unsatisfactory) | 0-49% | 0-49 |

---

© 2025 Antonio Clim. All rights reserved.
