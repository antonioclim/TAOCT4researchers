# Week 2: Abstraction and Encapsulation

## 🎯 Overview

This week explores **design patterns and object-oriented thinking** for research software. Building upon the state concepts and AST hierarchies from Week 1, we examine how abstraction and encapsulation enable the construction of modular, extensible and testable simulation frameworks.

**Estimated Time:** 4 hours (lecture + laboratory)

## 📚 Learning Objectives

After completing this week, you will be able to:

1. **[Understand]** Explain core OOP principles (SOLID) and their applications in research software
2. **[Apply]** Implement Strategy, Observer and Factory patterns in scientific simulations
3. **[Analyse]** Refactor procedural code into well-structured object-oriented designs

## 📋 Prerequisites

- Completion of Week 1 (state concept from Turing machines, AST hierarchies)
- Python proficiency with type hints and dataclasses
- Basic understanding of inheritance and polymorphism

## 📁 Contents

```
week2/
├── README.md                          # This file
├── theory/
│   ├── slides.html                    # reveal.js presentation (40+ slides)
│   ├── lecture_notes.md               # Detailed lecture notes
│   └── learning_objectives.md         # Measurable learning outcomes
├── lab/
│   ├── __init__.py                    # Package initialisation
│   ├── lab_2_01_simulation_framework.py  # OOP simulation framework
│   ├── lab_2_02_design_patterns.py       # Design patterns catalogue
│   └── solutions/                     # Reference solutions
├── exercises/
│   ├── homework.md                    # Main homework assignment
│   ├── practice/                      # Graded practice problems
│   └── solutions/                     # Exercise solutions
├── assessments/
│   ├── quiz.md                        # Self-assessment quiz
│   ├── rubric.md                      # Grading criteria
│   └── self_check.md                  # Learning checkpoint
├── resources/
│   ├── cheatsheet.md                  # Quick reference (A4)
│   ├── further_reading.md             # Additional resources
│   ├── glossary.md                    # Key terminology
│   └── datasets/                      # Sample data files
├── assets/
│   ├── diagrams/                      # SVG diagrams
│   ├── animations/                    # Interactive demos
│   └── images/                        # Static images
├── tests/                             # pytest test suite
└── Makefile                           # Build automation
```

## 🚀 Quick Start

```bash
# Navigate to week 2
cd week2

# Install dependencies
pip install -r ../docker/requirements.txt --break-system-packages

# Run the primary lab
python -m lab.lab_2_01_simulation_framework --demo

# Run tests
make test

# View presentation
open theory/slides.html
```

## 🔗 Connections

| Previous | Current | Next |
|----------|---------|------|
| Week 1: Epistemology of Computation | **Week 2: Abstraction & Encapsulation** | Week 3: Algorithmic Complexity |
| State machines, AST interpreters | Design patterns, OOP principles | Big-O notation, benchmarking |

## 🔬 Research Applications

- **Epidemiology:** SIR model with Strategy pattern for disease transmission
- **Physics:** N-body simulation with Observer pattern for visualisation
- **Economics:** Market simulation with Factory pattern for agent creation

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
