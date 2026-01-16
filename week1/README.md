# Week 1: The Epistemology of Computation

> *What does it mean to compute? From Turing machines to language interpreters.*

---

## 📋 Overview

This week establishes the theoretical foundations of computational thinking by exploring the very nature of computation itself. We begin with Alan Turing's revolutionary 1936 paper and trace the conceptual lineage through lambda calculus to modern programming language interpreters.

**Course:** The Art of Computational Thinking for Researchers  
**Week:** 1 of 7  
**Bloom Level:** Remember / Understand  

---

## 🎯 Learning Objectives

After completing this week, you will be able to:

1. **[Remember]** Define computability and enumerate the components of a Turing machine
2. **[Understand]** Explain the relationship between Turing machines, lambda calculus and modern programming languages
3. **[Apply]** Implement a Turing machine simulator and minimal AST interpreter in Python

---

## 📚 Prerequisites

- Basic Python knowledge (variables, functions, classes)
- Familiarity with command-line interfaces
- No prior knowledge of formal computation theory required

---

## ⏱️ Estimated Time

| Activity | Duration |
|----------|----------|
| Lecture slides | 90 minutes |
| Reading materials | 60 minutes |
| Lab 1: Turing Machine | 120 minutes |
| Lab 2: Lambda Calculus | 90 minutes |
| Lab 3: AST Interpreter | 120 minutes |
| Homework | 180 minutes |
| **Total** | **~11 hours** |

---

## 📁 Contents

### Theory

| File | Description |
|------|-------------|
| [`theory/slides.html`](theory/slides.html) | reveal.js presentation (40+ slides) |
| [`theory/lecture_notes.md`](theory/lecture_notes.md) | Detailed lecture notes (2000+ words) |
| [`theory/learning_objectives.md`](theory/learning_objectives.md) | Measurable objectives with assessment criteria |

### Laboratory

| File | Description |
|------|-------------|
| [`lab/lab_1_01_turing_machine.py`](lab/lab_1_01_turing_machine.py) | Turing machine simulator |
| [`lab/lab_1_02_lambda_calculus.py`](lab/lab_1_02_lambda_calculus.py) | Lambda calculus basics |
| [`lab/lab_1_03_ast_interpreter.py`](lab/lab_1_03_ast_interpreter.py) | Expression interpreter |
| [`lab/solutions/`](lab/solutions/) | Complete solutions for all labs |

### Exercises

| File | Description |
|------|-------------|
| [`exercises/homework.md`](exercises/homework.md) | Main homework assignment with rubric |
| [`exercises/practice/`](exercises/practice/) | 9 practice exercises (3 easy, 3 medium, 3 hard) |
| [`exercises/solutions/`](exercises/solutions/) | Solutions for practice exercises |

### Assessments

| File | Description |
|------|-------------|
| [`assessments/quiz.md`](assessments/quiz.md) | 10-question knowledge check |
| [`assessments/rubric.md`](assessments/rubric.md) | Grading criteria |
| [`assessments/self_check.md`](assessments/self_check.md) | Self-assessment checklist |

### Resources

| File | Description |
|------|-------------|
| [`resources/cheatsheet.md`](resources/cheatsheet.md) | One-page reference (A4) |
| [`resources/further_reading.md`](resources/further_reading.md) | 10+ annotated resources |
| [`resources/glossary.md`](resources/glossary.md) | Week terminology |

### Assets

| Directory | Description |
|-----------|-------------|
| [`assets/diagrams/`](assets/diagrams/) | SVG diagrams (min. 3) |
| [`assets/animations/`](assets/animations/) | Interactive HTML visualisations |
| [`assets/images/`](assets/images/) | Supporting images |

---

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r ../../docker/requirements.txt
```

### 2. Run Labs

```bash
# Run Turing machine simulator demo
python lab/lab_1_01_turing_machine.py --demo

# Run lambda calculus demo
python lab/lab_1_02_lambda_calculus.py --demo

# Run AST interpreter demo
python lab/lab_1_03_ast_interpreter.py --demo
```

### 3. Run Tests

```bash
# From week1 directory
pytest tests/ -v --cov=lab
```

### 4. View Presentation

```bash
# Open in browser
open theory/slides.html
# Or serve locally
python -m http.server 8000
# Then visit http://localhost:8000/theory/slides.html
```

---

## 🔗 Connections

### Builds Upon
- This is the first week; no prerequisites from previous weeks.

### Prepares For
- **Week 2:** State concept → State pattern; AST hierarchies → Design patterns
- **Week 3:** Algorithm complexity analysis of Turing machine operations

---

## 📖 Research Applications

| Domain | Application |
|--------|-------------|
| **Bioinformatics** | Finite state machines for DNA pattern matching |
| **Computational Linguistics** | Parsers for natural language processing |
| **Physics** | Cellular automata for physical simulations |
| **Neuroscience** | Neural computation models |

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

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
