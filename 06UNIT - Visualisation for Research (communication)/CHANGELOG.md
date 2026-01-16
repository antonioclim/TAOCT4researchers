# CHANGELOG — Week 6: Visualisation for Research

## Version 1.0.0 — January 2025

### 📦 Complete Materials Delivered

**Total Files Created: 39**

---

## 📚 Theory Materials

| File | Description | Size |
|------|-------------|------|
| `theory/learning_objectives.md` | 4 measurable objectives using Bloom's taxonomy | 7 KB |
| `theory/lecture_notes.md` | 2,100+ words covering all key concepts | 13 KB |
| `theory/slides.html` | 40+ slides, reveal.js 5.0, dark theme | 76 KB |

### Key Topics Covered
- Grammar of graphics (Wilkinson's 7 layers)
- Publication standards (Nature, Science, IEEE, PLOS)
- Tufte's principles (data-ink ratio, lie factor, chartjunk)
- Colour theory and accessibility (Wong palette, CVD-safe)
- Interactive visualisation with D3.js and Plotly
- Dashboard design patterns

---

## 🔬 Laboratory Files

| File | Description | Lines |
|------|-------------|-------|
| `lab/__init__.py` | Package initialisation with exports | 60 |
| `lab/lab_6_01_static_plots.py` | Static visualisation toolkit | 700+ |
| `lab/lab_6_02_interactive_viz.py` | Interactive dashboard toolkit | 900+ |
| `lab/solutions/lab_6_01_solution.py` | Lab 1 complete solutions | 450 |
| `lab/solutions/lab_6_02_solution.py` | Lab 2 complete solutions | 550 |

### Lab 1 Features (Static Plots)
- `PALETTES` dictionary with Wong, Tol, IBM colourblind-safe schemes
- `JOURNAL_STYLES` for Nature, Science, IEEE, PLOS specifications
- `PlotStyle` dataclass with factory methods
- Publication-quality figure export (PNG, PDF, SVG, EPS)
- Statistical visualisations with error bands
- Scatter plots with regression lines
- Annotated heatmaps with dendrograms
- D3.js JSON export for web visualisations

### Lab 2 Features (Interactive Viz)
- `DataPoint` and `Dataset` classes for data management
- `MetricCard` for KPI displays
- `FilterControl` for interactive filtering
- SVG generators (bar, line, pie charts)
- `DashboardConfig` for layout management
- HTML dashboard generation with dark theme
- Streamlit template for rapid prototyping

---

## ✏️ Exercises

### Homework (`exercises/homework.md`)
| Part | Points | Topic |
|------|--------|-------|
| Part 1 | 30 | Publication-quality figures |
| Part 2 | 35 | Interactive dashboard |
| Part 3 | 20 | D3.js visualisation |
| Bonus | 15 | Animated transitions |
| **Total** | **100** | |

### Practice Exercises (9 files)

| Difficulty | File | Topic |
|------------|------|-------|
| Easy | `easy_01_line_plot.py` | Basic line plot with error bands |
| Easy | `easy_02_scatter_regression.py` | Scatter with linear regression |
| Easy | `easy_03_bar_chart.py` | Grouped bar chart |
| Medium | `medium_01_heatmap.py` | Correlation heatmap |
| Medium | `medium_02_small_multiples.py` | Faceted plots |
| Medium | `medium_03_plotly_interactive.py` | Interactive Plotly chart |
| Hard | `hard_01_d3_bar_chart.py` | D3.js bar chart export |
| Hard | `hard_02_publication_figure.py` | Multi-panel Nature figure |
| Hard | `hard_03_animation.py` | Animated transitions |

### Solutions (9 files)
Complete solutions with:
- 100% type hint coverage
- Google-style docstrings
- Full implementation of all TODO sections
- British English comments

---

## 📋 Assessments

| File | Description |
|------|-------------|
| `assessments/quiz.md` | 10 questions (6 MC, 4 short answer) |
| `assessments/rubric.md` | 5-dimension grading rubric |
| `assessments/self_check.md` | Self-assessment with confidence ratings |

### Quiz Topics
1. Grammar of graphics layers
2. Wong palette colours
3. Figure dimensions for journals
4. D3.js enter-update-exit pattern
5. Data-ink ratio definition
6. Accessibility guidelines
7. Export formats for print
8. Dashboard design principles
9. Lie factor calculation
10. Interactive vs static visualisation

### Rubric Dimensions
- Technical Implementation (25%)
- Visual Design Quality (25%)
- Accessibility Compliance (15%)
- Documentation (20%)
- Reproducibility (15%)

---

## 📚 Resources

| File | Description |
|------|-------------|
| `resources/cheatsheet.md` | 1-2 A4 quick reference |
| `resources/further_reading.md` | 23 annotated resources |
| `resources/glossary.md` | 50+ terms A-Z |

### Datasets

| File | Description |
|------|-------------|
| `sample_timeseries.csv` | 25 rows: sine wave with noise |
| `sample_research_data.json` | Full metadata, timeseries, scatter, heatmap, network |
| `correlation_matrix.csv` | 6×6 correlation matrix |

---

## 🎨 Assets

### SVG Diagrams (3 files)

| File | Dimensions | Content |
|------|------------|---------|
| `tufte_principles.svg` | 800×500 | Data-ink ratio, lie factor, chartjunk, small multiples |
| `grammar_of_graphics.svg` | 900×600 | 7-layer architecture with implementations |
| `d3_enter_update_exit.svg` | 900×600 | Data binding pattern with code examples |

### Interactive Animations (1 file)

| File | Description |
|------|-------------|
| `chart_animation.html` | D3.js v7 bar chart with enter-update-exit demo |

Animation Features:
- Add/Remove/Update/Shuffle/Reset controls
- 750ms transitions with easeCubicInOut
- Wong palette colours
- Tooltip on hover
- Live data array display
- Dark theme (#1a1a2e)

---

## 🧪 Tests

| File | Tests | Coverage Target |
|------|-------|-----------------|
| `tests/__init__.py` | Package init | — |
| `tests/conftest.py` | 25+ fixtures | — |
| `tests/test_lab_6_01.py` | 50+ tests | 85%+ |
| `tests/test_lab_6_02.py` | 60+ tests | 85%+ |

### Fixture Categories
- Data fixtures (dataframes, timeseries, categories)
- Colour fixtures (palettes, journal styles)
- File system fixtures (temp directories)
- Matplotlib fixtures (figure cleanup)
- Parametrised tests (formats, colourmaps)

### Test Coverage Areas
- All classes and dataclasses
- Factory methods
- Export functions
- Edge cases (empty data, NaN, Unicode)
- Integration workflows

---

## ⚙️ Build System

### Makefile Targets (30+)

| Category | Targets |
|----------|---------|
| Running | `run-labs`, `run-lab-01`, `run-lab-02` |
| Testing | `test`, `test-cov`, `test-verbose`, `test-lab-*` |
| Quality | `lint`, `format`, `typecheck`, `check` |
| Exercises | `run-exercises`, `check-solutions` |
| Validation | `validate-structure`, `validate-readme` |
| Installation | `install`, `install-dev` |
| Cleaning | `clean-cache`, `clean-output`, `clean-all` |
| Documentation | `docs`, `serve-slides` |
| Development | `watch`, `dev` |

---

## 🔧 Technical Specifications

### Dependencies
```
Python ≥3.12
NumPy ≥1.24
Pandas ≥2.0
Matplotlib ≥3.7
Seaborn ≥0.12
SciPy ≥1.11
pytest ≥7.0
ruff ≥0.1
mypy ≥1.0
```

### Standards Applied
- British English throughout (colour, visualisation, analyse)
- No Oxford comma
- 100% type hint coverage
- Google-style docstrings
- Licence v2.0.2 in README

### Accessibility
- Wong palette default (colourblind-safe)
- WCAG 2.1 AA contrast ratios
- Alt text guidelines
- Pattern + colour encoding

---

## 📜 Licence

All materials © 2025 Antonio Clim. Restrictive Licence v2.0.2.
See README.md for full terms.

---

## ✅ Validation Checklist

- [x] 39/39 files created
- [x] British English throughout
- [x] No Oxford comma
- [x] Licence in README.md
- [x] Tech stack specified
- [x] Type hints 100%
- [x] Bloom's taxonomy objectives
- [x] Week 5 recap in slides
- [x] Week 7 preview in slides
- [x] Colourblind accessibility
- [x] All tests structured
- [x] Makefile complete
