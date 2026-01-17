# Week 7: Reproducibility and Capstone

> **The Art of Computational Thinking for Researchers**
> Final Week — Building Reproducible Research Software and Integrating All Concepts

---

## 📋 Overview

This capstone week brings together all concepts from the previous six weeks into a cohesive project framework. You will learn to build reproducible research software with comprehensive testing, continuous integration and proper documentation—skills essential for any computational researcher.

**Key Themes**: Reproducibility crisis, Testing (unit and integration), CI/CD (GitHub Actions), Documentation (Sphinx, MkDocs), Version control best practices, Code review

---

## 🎯 Learning Objectives

After completing this week, you will be able to:

1. **[Apply]** Implement comprehensive testing with pytest and configure CI/CD pipelines
2. **[Create]** Build reproducible project structures with proper documentation
3. **[Evaluate]** Conduct peer review using established criteria and best practices

---

## 📚 Prerequisites

Before starting this week, you should have completed:

- [x] Week 1: The Epistemology of Computation
- [x] Week 2: Abstraction and Encapsulation
- [x] Week 3: Algorithmic Complexity
- [x] Week 4: Advanced Data Structures
- [x] Week 5: Scientific Computing
- [x] Week 6: Visualisation for Research

---

## ⏱️ Estimated Time

| Activity | Duration |
|----------|----------|
| Lecture & Slides | 90 minutes |
| Lab 7.1: Reproducibility Toolkit | 120 minutes |
| Lab 7.2: Testing and CI/CD | 120 minutes |
| Practice Exercises | 90 minutes |
| Capstone Project Work | 240+ minutes |
| **Total** | **~11 hours** |

---

## 📁 Contents

### Theory

| File | Description |
|------|-------------|
| [slides.html](theory/slides.html) | Interactive reveal.js presentation (40+ slides) |
| [lecture_notes.md](theory/lecture_notes.md) | Detailed lecture notes (2000+ words) |
| [learning_objectives.md](theory/learning_objectives.md) | Measurable objectives with Bloom's taxonomy |

### Laboratory

| File | Description |
|------|-------------|
| [lab_7_01_reproducibility.py](lab/lab_7_01_reproducibility.py) | Seed management and data manifests |
| [lab_7_02_testing_cicd.py](lab/lab_7_02_testing_cicd.py) | pytest, mocking and GitHub Actions |
| [lab_7_03_project_scaffolder.py](lab/lab_7_03_project_scaffolder.py) | Automatic project structure generator |

### Exercises

| File | Description |
|------|-------------|
| [homework.md](exercises/homework.md) | Capstone project requirements and rubric |
| [practice/](exercises/practice/) | 9 graded exercises (easy, medium, hard) |

### Assessments

| File | Description |
|------|-------------|
| [quiz.md](assessments/quiz.md) | 10-question assessment |
| [rubric.md](assessments/rubric.md) | Detailed grading rubric |
| [self_check.md](assessments/self_check.md) | Self-assessment checklist |

### Resources

| File | Description |
|------|-------------|
| [cheatsheet.md](resources/cheatsheet.md) | One-page A4 reference |
| [further_reading.md](resources/further_reading.md) | 10+ curated resources |
| [glossary.md](resources/glossary.md) | Week terminology definitions |

### Assets

| Directory | Description |
|-----------|-------------|
| [diagrams/](assets/diagrams/) | SVG diagrams (CI/CD pipeline, project structure, testing pyramid) |
| [animations/](assets/animations/) | Interactive HTML demonstrations |

---

## 🚀 Quick Start

```bash
# Navigate to week directory
cd week7

# Run all labs with demos
make run-labs

# Execute tests
make test

# Validate structure
make validate
```

---

## 🔗 Connections to Course

### Building Upon (Weeks 1-6)

| Week | Concept | Application in Week 7 |
|------|---------|----------------------|
| 1 | Computation fundamentals | Testing interpreters and state machines |
| 2 | Design patterns | Testable architecture with dependency injection |
| 3 | Complexity analysis | Benchmarking in CI pipelines |
| 4 | Data structures | Testing graph algorithms and probabilistic structures |
| 5 | Scientific computing | Reproducible simulations with seed management |
| 6 | Visualisation | Automated figure generation in CI |

### Capstone Integration

Your capstone project should demonstrate mastery of concepts from all seven weeks, including proper testing, documentation and reproducibility practices taught in this final week.

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
| pytest-cov | ≥4.0 | Coverage reporting |
| ruff | ≥0.1 | Linting and formatting |
| mypy | ≥1.0 | Type checking |
| Docker | 24+ | Containerisation |
| reveal.js | 5.0 | Presentation framework |

---
