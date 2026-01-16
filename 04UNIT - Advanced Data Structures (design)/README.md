# ═══════════════════════════════════════════════════════════════════════════════
# Week 4: Advanced Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

> **Course**: The Art of Computational Thinking for Researchers  
> **Week**: 4 of 7  
> **Theme**: Graphs, Trees and Probabilistic Data Structures for Research  
> **Bloom Level**: Analyse/Evaluate

---

## 📋 Overview

This week explores advanced data structures that form the backbone of modern computational research. We progress from fundamental graph representations and algorithms to sophisticated probabilistic structures like Bloom filters and Count-Min sketches. You will learn to select appropriate data structures based on specific use cases and evaluate the trade-offs between deterministic and probabilistic approaches.

**Estimated Time**: 8-10 hours (theory + lab + exercises)

---

## 🎯 Learning Objectives

After completing this week, you will be able to:

1. **[Apply]** Implement graph data structures with common algorithms including BFS, DFS, Dijkstra and A*
2. **[Analyse]** Select appropriate data structures based on computational requirements and use case constraints
3. **[Evaluate]** Compare probabilistic versus deterministic structure trade-offs in terms of space, time and accuracy

---

## 📚 Prerequisites

- **Week 3**: Complexity analysis, algorithm efficiency, Big-O notation
- **Python**: Intermediate proficiency (classes, generics, type hints)
- **Mathematics**: Basic probability theory, logarithms

---

## 📁 Contents

### Theory (`theory/`)

| File | Description |
|------|-------------|
| [slides.html](theory/slides.html) | Interactive reveal.js presentation (40+ slides) |
| [lecture_notes.md](theory/lecture_notes.md) | Detailed lecture notes (2000+ words) |
| [learning_objectives.md](theory/learning_objectives.md) | Measurable learning outcomes |

### Laboratory (`lab/`)

| File | Lines | Description |
|------|-------|-------------|
| [lab_4_01_graph_library.py](lab/lab_4_01_graph_library.py) | 500+ | Graph implementation and algorithms |
| [lab_4_02_probabilistic_ds.py](lab/lab_4_02_probabilistic_ds.py) | 300+ | Bloom filters, Count-Min sketch |

### Exercises (`exercises/`)

| File | Description |
|------|-------------|
| [homework.md](exercises/homework.md) | Main homework assignment with rubric |
| `practice/easy_*.py` | Beginner exercises (3 files) |
| `practice/medium_*.py` | Intermediate exercises (3 files) |
| `practice/hard_*.py` | Advanced exercises (3 files) |

### Assessments (`assessments/`)

| File | Description |
|------|-------------|
| [quiz.md](assessments/quiz.md) | 10 assessment questions |
| [rubric.md](assessments/rubric.md) | Grading criteria |
| [self_check.md](assessments/self_check.md) | Self-assessment checklist |

### Resources (`resources/`)

| File | Description |
|------|-------------|
| [cheatsheet.md](resources/cheatsheet.md) | One-page A4 reference |
| [further_reading.md](resources/further_reading.md) | 10+ academic resources |
| [glossary.md](resources/glossary.md) | Week terminology |
| `datasets/` | Sample data files |

### Assets (`assets/`)

| Directory | Contents |
|-----------|----------|
| `diagrams/` | SVG visualisations (3+ files) |
| `animations/` | Interactive HTML demos (1+ files) |
| `images/` | Static images |

### Tests (`tests/`)

| File | Description |
|------|-------------|
| [conftest.py](tests/conftest.py) | pytest fixtures |
| [test_lab_4_01.py](tests/test_lab_4_01.py) | Graph library tests |
| [test_lab_4_02.py](tests/test_lab_4_02.py) | Probabilistic structures tests |

---

## 🚀 Quick Start

```bash
# Navigate to week directory
cd week4

# Install dependencies
pip install networkx matplotlib numpy mmh3 --break-system-packages

# Run lab demonstrations
python lab/lab_4_01_graph_library.py --demo
python lab/lab_4_02_probabilistic_ds.py --demo

# Run tests
make test

# Validate structure
make validate
```

---

## 🔗 Week Connections

### Prerequisites (Week 3)
- Complexity analysis enables understanding of algorithm trade-offs
- Benchmarking skills applied to compare data structure performance
- Big-O reasoning for selecting appropriate structures

### Prepares for (Week 5)
- Graph structures used in agent-based modelling
- Efficient data structures enable large-scale simulations
- Probabilistic methods connect to Monte Carlo techniques

---

## 📊 Research Applications

| Domain | Application |
|--------|-------------|
| **Social Networks** | Community detection algorithms |
| **Bioinformatics** | Protein interaction networks |
| **Big Data** | Approximate membership testing |
| **Network Analysis** | Shortest path routing |
| **Databases** | Query optimisation with Bloom filters |

---

## ⏱️ Time Allocation

| Activity | Duration |
|----------|----------|
| Lecture (slides) | 90 minutes |
| Reading (notes) | 60 minutes |
| Lab 1 (graphs) | 120 minutes |
| Lab 2 (probabilistic) | 90 minutes |
| Homework | 180 minutes |
| Self-assessment | 30 minutes |
| **Total** | ~10 hours |

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
| NetworkX | ≥3.0 | Graph algorithms |
| pytest | ≥7.0 | Testing framework |
| Docker | 24+ | Containerisation |
| reveal.js | 5.0 | Presentation framework |

---
