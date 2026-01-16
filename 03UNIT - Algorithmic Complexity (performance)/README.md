# Week 3: Algorithmic Complexity

> Understanding and measuring computational performance

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## 📋 Overview

This week explores the fundamental principles of algorithmic complexity analysis—the theoretical foundation that enables researchers to predict and optimise computational performance. We progress from formal asymptotic notation through practical benchmarking frameworks, culminating in empirical complexity estimation techniques applicable across programming languages.

**Duration:** 4 hours (theory + laboratory)

**Prerequisites:**
- Week 2: Abstraction and Encapsulation (abstract data types, encapsulation principles)
- Familiarity with basic Python programming
- Understanding of mathematical functions and logarithms

**Prepares for:** Week 4 (Efficient data structure implementation)

---

## 🎯 Learning Objectives

After completing this week, you will be able to:

1. **[Understand]** Explain Big-O notation and classify algorithms into complexity classes
2. **[Apply]** Implement a comprehensive benchmarking framework with statistical analysis
3. **[Analyse]** Estimate the complexity of algorithms both empirically and theoretically

---

## 📁 Directory Structure

```
week3/
├── README.md                           ← You are here
├── theory/
│   ├── slides.html                     # reveal.js presentation (40+ slides)
│   ├── lecture_notes.md                # Detailed notes (2000+ words)
│   └── learning_objectives.md          # Measurable objectives
├── lab/
│   ├── __init__.py
│   ├── lab_3_01_benchmark_suite.py     # Primary lab: Benchmarking framework
│   ├── lab_3_02_complexity_analyser.py # Secondary lab: Big-O estimation
│   └── solutions/
│       ├── lab_3_01_solution.py
│       └── lab_3_02_solution.py
├── exercises/
│   ├── homework.md                     # Main homework with rubric
│   ├── practice/
│   │   ├── easy_01_timing.py
│   │   ├── easy_02_list_operations.py
│   │   ├── easy_03_loop_analysis.py
│   │   ├── medium_01_sorting_benchmark.py
│   │   ├── medium_02_recursion_analysis.py
│   │   ├── medium_03_space_complexity.py
│   │   ├── hard_01_amortised_analysis.py
│   │   ├── hard_02_cache_effects.py
│   │   └── hard_03_complexity_proof.py
│   └── solutions/
├── assessments/
│   ├── quiz.md                         # 10 questions
│   ├── rubric.md                       # Grading rubric
│   └── self_check.md                   # Self-assessment
├── resources/
│   ├── cheatsheet.md                   # One-pager A4
│   ├── further_reading.md              # 10+ resources
│   ├── glossary.md                     # Week terminology
│   └── datasets/
│       └── benchmark_data.csv
├── assets/
│   ├── diagrams/
│   │   ├── complexity_classes.svg
│   │   ├── benchmark_architecture.svg
│   │   └── memory_hierarchy.svg
│   ├── animations/
│   │   └── sorting_visualiser.html
│   └── images/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_lab_3_01.py
│   └── test_lab_3_02.py
└── Makefile
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib scipy pytest pytest-cov ruff mypy

# Optional: Install Numba for JIT compilation
pip install numba
```

### 2. Run the Laboratory

```bash
# Run primary lab demonstration
python lab/lab_3_01_benchmark_suite.py --demo

# Run secondary lab
python lab/lab_3_02_complexity_analyser.py --demo

# Run all tests
make test
```

### 3. View Presentation

Open `theory/slides.html` in a web browser, or serve locally:

```bash
python -m http.server 8000
# Then navigate to http://localhost:8000/theory/slides.html
```

---

## 📚 Contents

### Theory (50 minutes)

| Topic | Duration | Description |
|-------|----------|-------------|
| Asymptotic Notation | 15 min | Big-O, Big-Ω, Big-Θ definitions and properties |
| Complexity Classes | 15 min | P, NP, common growth rates |
| Analysis Techniques | 10 min | Recurrence relations, amortised analysis |
| Practical Benchmarking | 10 min | Statistical rigour, avoiding pitfalls |

### Laboratory (60 minutes)

| Lab | Topic | Estimated Time |
|-----|-------|----------------|
| Lab 3.1 | Benchmark Suite Implementation | 40 minutes |
| Lab 3.2 | Automatic Complexity Estimation | 20 minutes |

### Key Concepts

- **Big-O Notation**: Upper bound on growth rate
- **Time Complexity**: Operations as a function of input size
- **Space Complexity**: Memory usage as a function of input size
- **Amortised Analysis**: Average cost over a sequence of operations
- **Benchmarking**: Empirical measurement with statistical rigour
- **Profiling**: Identifying performance bottlenecks

---

## 🔬 Research Examples

This week's concepts directly apply to research domains:

| Domain | Application | Relevance |
|--------|-------------|-----------|
| Bioinformatics | Sequence alignment algorithm comparison | Choosing O(n²) vs O(n log n) algorithms for genome analysis |
| Data Science | Sorting algorithm selection for large datasets | Understanding when to use different sorting strategies |
| Network Analysis | Graph algorithm scalability | Predicting runtime for social network analysis |
| Machine Learning | Training time estimation | Complexity analysis of optimisation algorithms |

---

## ✅ Assessment

| Component | Weight | Description |
|-----------|--------|-------------|
| Homework | 40% | Algorithm analysis and optimisation tasks |
| Quiz | 20% | Conceptual understanding verification |
| Lab Completion | 30% | Working benchmark suite and complexity analyser |
| Participation | 10% | Discussion and code review engagement |

---

## 🔗 Week Connections

```
Week 2: Abstraction & Encapsulation    Week 4: Advanced Data Structures
                    ↓                              ↑
        Abstract data types           Efficient implementations
        Encapsulation principles      Performance-driven design
                    ↓                              ↑
              ┌─────────────────────────────────────┐
              │     Week 3: Algorithmic Complexity  │
              │                                     │
              │  • Big-O notation                   │
              │  • Benchmarking frameworks          │
              │  • Empirical analysis               │
              │  • Profiling techniques             │
              └─────────────────────────────────────┘
```

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
| SciPy | ≥1.11 | Scientific computing |
| pytest | ≥7.0 | Testing framework |
| Docker | 24+ | Containerisation |
| reveal.js | 5.0 | Presentation framework |

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Week 3 — Algorithmic Complexity*
