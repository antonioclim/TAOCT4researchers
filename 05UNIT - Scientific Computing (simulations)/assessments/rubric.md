# Week 5 Assessment Rubric: Scientific Computing

## 📋 Overview

This rubric provides detailed criteria for assessing Week 5 homework submissions. The assessment is divided into three parts totalling 100 points, plus a 20-point bonus section.

---

## Part 1: Monte Carlo Methods (40 points)

### 1.1 Basic Integration (15 points)

| Criterion | Excellent (15) | Good (12) | Satisfactory (9) | Needs Work (5) | Insufficient (0) |
|-----------|---------------|-----------|------------------|----------------|------------------|
| **Implementation** | Correct MC estimator with proper variance calculation | Minor issues in variance calculation | Works but missing error estimation | Significant bugs | Not attempted |
| **Vectorisation** | Fully vectorised NumPy operations | Mostly vectorised | Partial vectorisation | Loop-based implementation | N/A |
| **Documentation** | Complete docstrings with examples | Good docstrings | Basic documentation | Minimal comments | No documentation |

### 1.2 Variance Reduction (15 points)

| Criterion | Excellent (15) | Good (12) | Satisfactory (9) | Needs Work (5) | Insufficient (0) |
|-----------|---------------|-----------|------------------|----------------|------------------|
| **Antithetic variates** | Correct implementation with demonstrated variance reduction | Works but variance reduction not verified | Implemented but minor errors | Attempted but incorrect | Not attempted |
| **Stratified sampling** | Correct strata handling with proper weighting | Works with minor issues | Basic implementation | Incomplete | Not attempted |

### 1.3 Convergence Analysis (10 points)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (3) | Insufficient (0) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Empirical verification** | Demonstrates O(1/√n) with multiple sample sizes | Shows convergence trend | Basic convergence check | Incomplete analysis | Not attempted |
| **Visualisation** | Clear log-log plot with fitted slope | Readable plot | Basic plot | Unclear presentation | No visualisation |

---

## Part 2: ODE Solvers (40 points)

### 2.1 Euler Method (10 points)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (3) | Insufficient (0) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Correctness** | Correct single-step formula | Works with minor issues | Basic implementation | Significant errors | Not attempted |
| **Order verification** | Empirically shows O(h) | Demonstrates convergence | Basic testing | Incomplete | Not verified |

### 2.2 RK4 Method (15 points)

| Criterion | Excellent (15) | Good (12) | Satisfactory (9) | Needs Work (5) | Insufficient (0) |
|-----------|---------------|-----------|------------------|----------------|------------------|
| **Implementation** | Correct k₁-k₄ calculations with proper weights | Minor coefficient errors | Works on simple cases | Significant bugs | Not attempted |
| **Order verification** | Demonstrates O(h⁴) empirically | Shows improved accuracy over Euler | Basic comparison | Incomplete | Not verified |
| **Systems support** | Handles vector-valued ODEs | Works for 2D systems | Only scalar ODEs | Limited | Not supported |

### 2.3 Problem Solving (15 points)

| Criterion | Excellent (15) | Good (12) | Satisfactory (9) | Needs Work (5) | Insufficient (0) |
|-----------|---------------|-----------|------------------|----------------|------------------|
| **Lotka-Volterra** | Correct implementation with oscillatory behaviour | Works with minor issues | Basic solution | Incomplete | Not attempted |
| **Energy conservation** | Demonstrates RK4 advantage for oscillators | Shows comparison | Basic awareness | Not addressed | Not attempted |

---

## Part 3: Agent-Based Modelling (20 points)

### 3.1 Schelling Model Extension (10 points)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (3) | Insufficient (0) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Heterogeneous thresholds** | Agents have varied thresholds correctly implemented | Works with minor issues | Basic extension | Incomplete | Not attempted |
| **Analysis** | Compares emergence with homogeneous case | Shows results | Basic metrics | Incomplete | Not analysed |

### 3.2 Metrics and Analysis (10 points)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (3) | Insufficient (0) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Segregation tracking** | Correct index computed over time | Works with minor issues | Basic metric | Errors | Not computed |
| **Visualisation** | Clear time series plot with annotations | Readable plot | Basic plot | Unclear | Not visualised |

---

## Bonus: Adaptive ODE Solver (20 points)

| Criterion | Excellent (20) | Good (15) | Satisfactory (10) | Partial (5) | Not Attempted (0) |
|-----------|---------------|-----------|-------------------|-------------|-------------------|
| **RKF45 implementation** | Correct embedded pair with error estimation | Works with minor issues | Basic adaptive stepping | Incomplete | Not attempted |
| **Step size control** | Proper acceptance/rejection with adjustment | Works mostly | Basic control | Unstable | Not implemented |
| **Tolerance handling** | Meets specified rtol/atol | Approximate | Basic | Unreliable | Not handled |

---

## Code Quality (Applied Across All Parts)

### Deductions (up to -20 points total)

| Issue | Deduction |
|-------|-----------|
| No type hints | -5 |
| Missing docstrings | -5 |
| Print statements instead of logging | -3 |
| Hardcoded values without constants | -2 |
| Poor variable naming | -2 |
| No error handling | -3 |
| American English spelling | -2 |

### Bonuses (up to +5 points)

| Enhancement | Bonus |
|-------------|-------|
| Comprehensive unit tests | +3 |
| Performance optimisation | +2 |
| Exceptional documentation | +2 |

---

## Submission Requirements

### Required Files
- [ ] `monte_carlo.py` — MC implementation
- [ ] `ode_solvers.py` — ODE solver implementations
- [ ] `abm_extension.py` — Schelling extension
- [ ] `report.md` — Analysis and results

### Automatic Checks
| Check | Requirement |
|-------|-------------|
| `ruff check` | No errors |
| `mypy --strict` | No errors |
| `pytest` | All tests pass |
| Coverage | ≥80% |

---

## Grade Boundaries

| Grade | Points | Percentage |
|-------|--------|------------|
| A | 90-100+ | 90%+ |
| B | 75-89 | 75-89% |
| C | 60-74 | 60-74% |
| D | 50-59 | 50-59% |
| F | <50 | <50% |

---

## Feedback Template

```markdown
## Week 5 Homework Feedback

**Student**: [Name]
**Submission Date**: [Date]
**Total Score**: [X]/100

### Part 1: Monte Carlo Methods ([X]/40)
- Basic integration: [X]/15
- Variance reduction: [X]/15
- Convergence analysis: [X]/10

**Comments**: [Specific feedback]

### Part 2: ODE Solvers ([X]/40)
- Euler method: [X]/10
- RK4 method: [X]/15
- Problem solving: [X]/15

**Comments**: [Specific feedback]

### Part 3: Agent-Based Modelling ([X]/20)
- Schelling extension: [X]/10
- Metrics and analysis: [X]/10

**Comments**: [Specific feedback]

### Bonus ([X]/20)
**Comments**: [Specific feedback]

### Code Quality
**Deductions**: [X]
**Bonuses**: [X]

### Overall Comments
[Summary feedback and suggestions for improvement]

### Grade: [Letter]
```

---

*© 2025 Antonio Clim. All rights reserved.*
