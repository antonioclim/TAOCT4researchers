# Week 6: Visualisation for Research

## 📊 Overview

This week focuses on creating publication-ready figures and interactive visualisations for research. You will learn the theoretical foundations of data visualisation (Tufte's principles, the Grammar of Graphics) and practical skills in both static (Matplotlib, Seaborn) and interactive (Plotly, D3.js) visualisation tools. By the end of this week, you will be able to produce figures that meet journal standards and dashboards that enable data exploration.

## 🎯 Learning Objectives

After completing this week, you will be able to:

1. **[Apply]** Create publication-quality static figures following journal standards (Nature, Science, IEEE)
2. **[Create]** Build interactive dashboards for data exploration using Streamlit or HTML/JavaScript
3. **[Evaluate]** Select appropriate visualisation types for different data characteristics and research questions
4. **[Analyse]** Apply Tufte's principles to critique and improve existing visualisations

## 📋 Prerequisites

Before starting this week, ensure you have completed:

- **Week 5**: Scientific Computing (simulation output data, numerical methods)
- **Python proficiency**: Intermediate level with NumPy and data manipulation
- **Basic HTML/CSS**: For interactive visualisation components

## ⏱️ Estimated Time

| Component | Duration |
|-----------|----------|
| Lecture and slides | 2 hours |
| Lab 1: Static plots | 2 hours |
| Lab 2: Interactive dashboards | 2.5 hours |
| Exercises | 3-4 hours |
| Homework | 4-5 hours |
| **Total** | **13-15 hours** |

## 📁 Contents

### Theory

| File | Description |
|------|-------------|
| [slides.html](theory/slides.html) | reveal.js presentation (40+ slides) |
| [lecture_notes.md](theory/lecture_notes.md) | Detailed lecture notes (2000+ words) |
| [learning_objectives.md](theory/learning_objectives.md) | Measurable objectives with rubrics |

### Laboratory

| File | Description |
|------|-------------|
| [lab_6_01_static_plots.py](lab/lab_6_01_static_plots.py) | Publication-ready figure creation |
| [lab_6_02_interactive_viz.py](lab/lab_6_02_interactive_viz.py) | Interactive dashboard development |
| [solutions/](lab/solutions/) | Complete solutions with annotations |

### Exercises

| File | Description |
|------|-------------|
| [homework.md](exercises/homework.md) | Main homework assignment with rubric |
| [practice/](exercises/practice/) | 9 graded exercises (easy, medium, hard) |
| [solutions/](exercises/solutions/) | Exercise solutions |

### Assessments

| File | Description |
|------|-------------|
| [quiz.md](assessments/quiz.md) | 10 assessment questions |
| [rubric.md](assessments/rubric.md) | Detailed grading rubric |
| [self_check.md](assessments/self_check.md) | Self-assessment checklist |

### Resources

| File | Description |
|------|-------------|
| [cheatsheet.md](resources/cheatsheet.md) | One-page A4 reference |
| [further_reading.md](resources/further_reading.md) | 10+ curated resources |
| [glossary.md](resources/glossary.md) | Week terminology |
| [datasets/](resources/datasets/) | Sample data files |

### Assets

| Directory | Contents |
|-----------|----------|
| [diagrams/](assets/diagrams/) | SVG diagrams (3+) |
| [animations/](assets/animations/) | HTML animations |
| [images/](assets/images/) | Supporting images |

## 🚀 Quick Start

```bash
# 1. Navigate to week directory
cd week6

# 2. Install dependencies
pip install matplotlib seaborn plotly numpy pandas scipy

# 3. Run the first lab
python -m lab.lab_6_01_static_plots --demo

# 4. Run the second lab
python -m lab.lab_6_02_interactive_viz --demo

# 5. Run tests
pytest tests/ -v

# 6. Open the slides
open theory/slides.html  # macOS
xdg-open theory/slides.html  # Linux
```

## 📊 Key Concepts

This week covers:

- **Tufte's Principles**: Data-ink ratio, lie factor, chartjunk, small multiples
- **Grammar of Graphics**: Data, aesthetics, geometries, facets, statistics, coordinates, themes
- **Publication Standards**: DPI requirements, colour accessibility, font embedding, vector formats
- **Interactive Visualisation**: Brushing, linking, filtering, zooming, tooltips
- **D3.js Fundamentals**: Selections, data binding, scales, axes, transitions

## 🔗 Connections

| Previous Week | Current Week | Next Week |
|---------------|--------------|-----------|
| Week 5: Scientific Computing | **Week 6: Visualisation** | Week 7: Reproducibility |
| Simulation output data | → Visualising results | → Documenting figures |
| Monte Carlo methods | → Convergence plots | → Reproducible scripts |
| Agent-based models | → Animation and dashboards | → Packaging and CI/CD |

## 📝 Assessment Weighting

| Component | Weight | Due |
|-----------|--------|-----|
| Lab completion | 20% | End of session |
| Practice exercises | 20% | Before next week |
| Homework | 40% | Friday 23:59 GMT |
| Quiz | 20% | During session |

---

## 📜 Licence and Terms of Use

╔═══════════════════════════════════════════════════════════════════════════════╗
║                           RESTRICTIVE LICENCE                                  ║
║                              Version 2.0.2                                     ║
║                             January 2025                                       ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   © 2025 Antonio Clim. All rights reserved.                                   ║
║                                                                               ║
║   PERMITTED:                                                                  ║
║   ✓ Personal use for self-study                                               ║
║   ✓ Viewing and running code for personal educational purposes                ║
║   ✓ Local modifications for personal experimentation                          ║
║                                                                               ║
║   PROHIBITED (without prior written consent):                                 ║
║   ✗ Publishing materials (online or offline)                                  ║
║   ✗ Use in formal teaching activities                                         ║
║   ✗ Teaching or presenting materials to third parties                         ║
║   ✗ Redistribution in any form                                                ║
║   ✗ Creating derivative works for public use                                  ║
║   ✗ Commercial use of any kind                                                ║
║                                                                               ║
║   For requests regarding educational use or publication,                      ║
║   please contact the author to obtain written consent.                        ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

### Terms and Conditions

1. **Intellectual Property**: All materials, including but not limited to code,
   documentation, presentations and exercises, are the intellectual property of
   Antonio Clim.

2. **No Warranty**: Materials are provided "as is" without warranty of any kind,
   express or implied.

3. **Limitation of Liability**: The author shall not be liable for any damages
   arising from the use of these materials.

4. **Governing Law**: These terms are governed by the laws of Romania.

5. **Contact**: For permissions and enquiries, contact the author through
   official academic channels.

### Technology Stack

This project uses the following technologies:

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Primary programming language |
| NumPy | ≥1.24 | Numerical computing |
| Pandas | ≥2.0 | Data manipulation |
| Matplotlib | ≥3.7 | Static visualisation |
| Seaborn | ≥0.12 | Statistical visualisation |
| Plotly | ≥5.0 | Interactive visualisation |
| SciPy | ≥1.11 | Scientific computing |
| pytest | ≥7.0 | Testing framework |
| Docker | 24+ | Containerisation |
| reveal.js | 5.0 | Presentation framework |
| D3.js | 7.0 | Data-driven documents |

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 6 — Visualisation for Research*
