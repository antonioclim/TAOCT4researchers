# Week 5: Scientific Computing

## 🎯 Overview

This week explores **numerical methods and simulation techniques** fundamental to computational research. Researchers across disciplines—from physics to economics—rely on Monte Carlo integration, differential equation solvers and agent-based models to understand complex systems that defy analytical solutions. You will implement these methods from scratch, gaining deep intuition for their convergence properties, error characteristics and computational trade-offs.

**Theme**: Numerical methods and simulation for research

**Bloom Level**: Apply/Create

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Duration** | 3 hours (lecture) + 4 hours (lab) |
| **Prerequisites** | Week 4: Advanced Data Structures |
| **Prepares for** | Week 6: Visualisation for Research |
| **Difficulty** | ⭐⭐⭐⭐ (4/5) |

## 🎯 Learning Objectives

By the end of this week, you will be able to:

1. **[Apply]** Implement Monte Carlo methods for numerical integration with variance reduction techniques
2. **[Apply]** Solve ordinary differential equations using Euler, RK4 and adaptive Runge-Kutta-Fehlberg methods
3. **[Create]** Design agent-based models that exhibit emergent behaviour from simple local rules

## 📚 Prerequisites

Before starting this week, ensure you have completed:

- [ ] Week 4: Graphs, trees and probabilistic data structures
- [ ] Understanding of algorithmic complexity (Big-O notation)
- [ ] Familiarity with Python classes and NumPy arrays
- [ ] Basic calculus (derivatives, integrals)

## 📂 Contents

### Theory Materials
| File | Description |
|------|-------------|
| [theory/slides.html](theory/slides.html) | reveal.js presentation (40+ slides) |
| [theory/lecture_notes.md](theory/lecture_notes.md) | Detailed lecture notes (2500+ words) |
| [theory/learning_objectives.md](theory/learning_objectives.md) | Measurable learning outcomes |

### Laboratory Exercises
| File | Description | Lines |
|------|-------------|-------|
| [lab/lab_5_01_monte_carlo.py](lab/lab_5_01_monte_carlo.py) | Monte Carlo integration and estimation | 500+ |
| [lab/lab_5_02_ode_solvers.py](lab/lab_5_02_ode_solvers.py) | Differential equation solvers | 500+ |
| [lab/lab_5_03_agent_based_modelling.py](lab/lab_5_03_agent_based_modelling.py) | ABM framework (Schelling, Boids) | 600+ |

### Practice Exercises
| Difficulty | Files |
|------------|-------|
| Easy (⭐) | `easy_01_pi_estimation.py`, `easy_02_euler_method.py`, `easy_03_random_walk.py` |
| Medium (⭐⭐) | `medium_01_importance_sampling.py`, `medium_02_rk4_solver.py`, `medium_03_schelling_model.py` |
| Hard (⭐⭐⭐) | `hard_01_adaptive_rkf45.py`, `hard_02_variance_reduction.py`, `hard_03_boids_optimisation.py` |

### Assessments
| File | Description |
|------|-------------|
| [assessments/quiz.md](assessments/quiz.md) | 10-question knowledge check |
| [assessments/rubric.md](assessments/rubric.md) | Grading criteria |
| [assessments/self_check.md](assessments/self_check.md) | Self-assessment checklist |

### Resources
| File | Description |
|------|-------------|
| [resources/cheatsheet.md](resources/cheatsheet.md) | One-page reference (A4) |
| [resources/further_reading.md](resources/further_reading.md) | 15+ curated resources |
| [resources/glossary.md](resources/glossary.md) | Week terminology |

### Assets
| Directory | Contents |
|-----------|----------|
| [assets/diagrams/](assets/diagrams/) | SVG visualisations (3+) |
| [assets/animations/](assets/animations/) | Interactive HTML demos (1+) |

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy scipy matplotlib pytest pytest-cov ruff mypy
```

### 2. Run Laboratory Demos
```bash
# Monte Carlo integration
python lab/lab_5_01_monte_carlo.py --demo

# ODE solvers
python lab/lab_5_02_ode_solvers.py --demo

# Agent-based models
python lab/lab_5_03_agent_based_modelling.py --demo
```

### 3. Execute Tests
```bash
# Run all tests with coverage
pytest tests/ -v --cov=lab --cov-report=term-missing

# Run specific test file
pytest tests/test_lab_5_01.py -v
```

### 4. Quality Checks
```bash
# Linting
ruff check lab/ tests/

# Type checking
mypy lab/ --strict

# Or use Makefile
make check
```

## 🔬 Research Applications

### Physics
- **Particle systems**: N-body gravitational simulations
- **Orbital mechanics**: Trajectory prediction with RK4
- **Statistical mechanics**: Monte Carlo sampling of Boltzmann distributions

### Epidemiology
- **Disease spread**: SIR/SEIR models with ODEs
- **Agent-based transmission**: Individual-level contact networks
- **Intervention analysis**: Parameter sensitivity studies

### Economics
- **Market dynamics**: Agent-based trading models
- **Segregation**: Schelling model for neighbourhood formation
- **Option pricing**: Monte Carlo methods for Black-Scholes

### Ecology
- **Predator-prey**: Lotka-Volterra differential equations
- **Flocking behaviour**: Boids emergence from local rules
- **Population dynamics**: Stochastic growth models

## 📊 Key Concepts

### Monte Carlo Methods
- **Law of Large Numbers**: Convergence at O(1/√n)
- **Variance reduction**: Antithetic variates, stratified sampling
- **Importance sampling**: Bias-variance trade-off

### ODE Solvers
- **Euler method**: First-order, O(h) global error
- **Runge-Kutta 4**: Fourth-order, O(h⁴) global error
- **Adaptive methods**: Error control with step size adjustment

### Agent-Based Modelling
- **Emergence**: Global patterns from local rules
- **Heterogeneity**: Individual agent differences
- **Spatial dynamics**: Neighbourhood interactions

## 🔗 Week Navigation

| Previous | Current | Next |
|----------|---------|------|
| [Week 4: Advanced Data Structures](../week4/) | **Week 5: Scientific Computing** | [Week 6: Visualisation](../week6/) |

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

*Last updated: January 2025*
