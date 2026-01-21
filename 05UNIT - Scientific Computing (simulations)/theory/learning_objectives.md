# Week 5: Scientific Computing — Learning Objectives

## 🎯 Overview

This document outlines the measurable learning objectives for Week 5, organised according to Bloom's Taxonomy cognitive levels. Each objective specifies the expected outcome, assessment criteria and connection to subsequent weeks.

---

## Primary Learning Objectives

### Objective 1: Monte Carlo Methods
**[Apply]** Implement Monte Carlo methods for numerical integration with variance reduction techniques

#### Success Criteria
By the end of this module, you will be able to:

| Criterion | Evidence |
|-----------|----------|
| Implement basic MC integration | Working `monte_carlo_integrate()` function that converges to known analytical results |
| Estimate π using hit-or-miss | `estimate_pi(n)` achieves |π̂ - π| < 0.01 with n = 100,000 |
| Apply antithetic variates | Demonstrate variance reduction ≥20% on monotonic functions |
| Apply stratified sampling | Correctly partition domain and combine sub-integrals |
| Analyse convergence | Plot error vs n showing O(1/√n) decay |

#### Assessment Methods
- **Lab 5.1**: Implement `MonteCarloIntegrator` class with variance reduction
- **Quiz**: Questions 1-3 test conceptual understanding
- **Homework Part 1**: Apply MC to multidimensional integration

---

### Objective 2: ODE Solvers
**[Apply]** Solve ordinary differential equations using Euler, RK4 and adaptive Runge-Kutta-Fehlberg methods

#### Success Criteria
By the end of this module, you will be able to:

| Criterion | Evidence |
|-----------|----------|
| Implement Euler method | Correct single-step update: y_{n+1} = y_n + h·f(t_n, y_n) |
| Implement RK4 | Four-stage algorithm with correct coefficients |
| Verify convergence order | Empirically demonstrate O(h) for Euler, O(h⁴) for RK4 |
| Implement adaptive stepping | RKF45 with local error estimation and step adjustment |
| Solve systems of ODEs | Handle vector-valued y for coupled equations |

#### Assessment Methods
- **Lab 5.2**: Implement `ODESolver` hierarchy with multiple methods
- **Quiz**: Questions 4-6 test understanding of error and stability
- **Homework Part 2**: Model predator-prey dynamics with Lotka-Volterra

---

### Objective 3: Agent-Based Modelling
**[Create]** Design agent-based models that exhibit emergent behaviour from simple local rules

#### Success Criteria
By the end of this module, you will be able to:

| Criterion | Evidence |
|-----------|----------|
| Design agent architecture | `Agent` class with state, behaviour and update methods |
| Implement spatial interactions | Grid-based or continuous-space neighbour finding |
| Create Schelling model | Segregation emerges from 30% same-type preference |
| Create Boids model | Flocking emerges from separation, alignment, cohesion |
| Measure emergence metrics | Compute segregation index, polarisation, clustering |

#### Assessment Methods
- **Lab 5.3**: Build ABM framework with Schelling and Boids examples
- **Quiz**: Questions 7-10 test understanding of emergence
- **Homework Part 3**: Extend Schelling with heterogeneous thresholds

---

## Supporting Objectives

### Computational Foundations

#### Numerical Precision
- **[Understand]** Explain floating-point representation and its implications for numerical algorithms
- **[Apply]** Use appropriate tolerances for convergence checks (absolute vs relative)

#### Randomness and Reproducibility
- **[Apply]** Manage random seeds for reproducible stochastic simulations
- **[Analyse]** Distinguish between statistical and systematic errors

#### Performance Considerations
- **[Analyse]** Profile Monte Carlo and ODE code to identify bottlenecks
- **[Apply]** Use NumPy vectorisation to accelerate inner loops

---

## Bloom's Taxonomy Mapping

| Level | Verb | Application in Week 5 |
|-------|------|----------------------|
| **Remember** | Define, List | Recall MC convergence rate, ODE solver orders |
| **Understand** | Explain, Compare | Describe variance reduction mechanisms |
| **Apply** | Implement, Use | Build working MC integrator, ODE solver |
| **Analyse** | Verify, Profile | Empirically measure convergence orders |
| **Evaluate** | Select, Justify | Choose appropriate solver for given problem |
| **Create** | Design, Extend | Build novel ABM for custom research question |

---

## Prerequisites Verification

Before beginning Week 5, verify you can:

| Prerequisite | Self-Check |
|--------------|------------|
| Implement graph algorithms | BFS/DFS from Week 4 |
| Analyse algorithm complexity | Big-O estimation from Week 3 |
| Use OOP patterns | Strategy, Observer from Week 2 |
| Write unit tests | pytest basics from Week 4 |
| Use NumPy arrays | Broadcasting, vectorisation |

---

## Connections to Adjacent Weeks

### From Week 4: Advanced Data Structures
- **Graphs** → Agent interaction networks
- **Probabilistic structures** → Monte Carlo sampling foundations
- **Complexity analysis** → Understanding solver efficiency

### To Week 6: Visualisation for Research
- **Simulation output** → Data for visualisation
- **Convergence plots** → Publication-quality figures
- **ABM animation** → Interactive dashboards

---

## Assessment Alignment Matrix

| Objective | Lab | Quiz | Homework | Self-Check |
|-----------|-----|------|----------|------------|
| MC methods | 5.1 | Q1-3 | Part 1 | §1 |
| ODE solvers | 5.2 | Q4-6 | Part 2 | §2 |
| Agent-based models | 5.3 | Q7-10 | Part 3 | §3 |

---

## Recommended Time Allocation

| Activity | Duration | Notes |
|----------|----------|-------|
| **Lecture** | 3 hours | Includes interactive demos |
| **Lab 5.1** | 1.5 hours | Monte Carlo |
| **Lab 5.2** | 1.5 hours | ODE solvers |
| **Lab 5.3** | 2 hours | Agent-based modelling |
| **Practice exercises** | 2-3 hours | Self-paced |
| **Homework** | 4-5 hours | Due end of week |

**Total**: 14-16 hours

---

## Success Indicators

By the end of Week 5, you should be able to confidently:

✅ Explain why Monte Carlo converges at O(1/√n) regardless of dimension  
✅ Choose between Euler and RK4 based on accuracy requirements  
✅ Implement adaptive step-size control for stiff problems  
✅ Design agent rules that produce desired emergent behaviour  
✅ Debug numerical simulations using convergence plots  
✅ Apply these methods to your own research problems  

---

*© 2025 Antonio Clim. All rights reserved.*
