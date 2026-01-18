# Week 5 Self-Assessment: Scientific Computing

## 📋 Instructions

Use this checklist to evaluate your understanding of Week 5 concepts. For each item, honestly assess your current level:

- ✅ **Confident**: I can explain this to others and apply it independently
- 🔶 **Developing**: I understand the basics but need more practice
- ❌ **Needs work**: I need to review this material

Complete this assessment before and after the lab sessions to track your progress.

---

## Section 1: Monte Carlo Methods

### Conceptual Understanding

| # | Concept | Pre-Lab | Post-Lab |
|---|---------|---------|----------|
| 1.1 | I can explain why Monte Carlo integration uses random sampling | ☐ | ☐ |
| 1.2 | I understand the Law of Large Numbers and its role in MC convergence | ☐ | ☐ |
| 1.3 | I can derive the O(1/√n) convergence rate from the Central Limit Theorem | ☐ | ☐ |
| 1.4 | I understand why MC works regardless of dimension (no curse of dimensionality) | ☐ | ☐ |
| 1.5 | I can explain the difference between bias and variance in estimators | ☐ | ☐ |

### Practical Skills

| # | Skill | Pre-Lab | Post-Lab |
|---|-------|---------|----------|
| 1.6 | I can implement basic MC integration with `monte_carlo_integrate()` | ☐ | ☐ |
| 1.7 | I can estimate π using the hit-or-miss method | ☐ | ☐ |
| 1.8 | I can compute and interpret confidence intervals | ☐ | ☐ |
| 1.9 | I can implement antithetic variates for variance reduction | ☐ | ☐ |
| 1.10 | I can implement stratified sampling with proper strata weighting | ☐ | ☐ |

### Application

| # | Application | Pre-Lab | Post-Lab |
|---|-------------|---------|----------|
| 1.11 | I can empirically verify convergence order using log-log plots | ☐ | ☐ |
| 1.12 | I can choose between standard MC and variance reduction techniques | ☐ | ☐ |
| 1.13 | I can apply MC to multidimensional integrals | ☐ | ☐ |

---

## Section 2: ODE Solvers

### Conceptual Understanding

| # | Concept | Pre-Lab | Post-Lab |
|---|---------|---------|----------|
| 2.1 | I understand the initial value problem (IVP) formulation | ☐ | ☐ |
| 2.2 | I can explain how Euler's method approximates solutions | ☐ | ☐ |
| 2.3 | I understand local vs global truncation error | ☐ | ☐ |
| 2.4 | I can explain why RK4 is more accurate than Euler | ☐ | ☐ |
| 2.5 | I understand stability constraints (e.g., h < 2/|λ| for Euler) | ☐ | ☐ |

### Practical Skills

| # | Skill | Pre-Lab | Post-Lab |
|---|-------|---------|----------|
| 2.6 | I can implement Euler's method step | ☐ | ☐ |
| 2.7 | I can implement the RK4 algorithm with k₁, k₂, k₃, k₄ | ☐ | ☐ |
| 2.8 | I can convert higher-order ODEs to first-order systems | ☐ | ☐ |
| 2.9 | I can implement adaptive step-size control (RKF45) | ☐ | ☐ |
| 2.10 | I can solve systems of coupled ODEs | ☐ | ☐ |

### Application

| # | Application | Pre-Lab | Post-Lab |
|---|-------------|---------|----------|
| 2.11 | I can solve the harmonic oscillator and verify energy conservation | ☐ | ☐ |
| 2.12 | I can simulate Lotka-Volterra predator-prey dynamics | ☐ | ☐ |
| 2.13 | I can empirically determine convergence order of ODE methods | ☐ | ☐ |
| 2.14 | I can select appropriate solver based on problem characteristics | ☐ | ☐ |

---

## Section 3: Agent-Based Modelling

### Conceptual Understanding

| # | Concept | Pre-Lab | Post-Lab |
|---|---------|---------|----------|
| 3.1 | I can define "emergence" and give examples | ☐ | ☐ |
| 3.2 | I understand the Schelling model rules and their implications | ☐ | ☐ |
| 3.3 | I understand the three rules of Boids (separation, alignment, cohesion) | ☐ | ☐ |
| 3.4 | I can explain why ABMs are useful when ODEs are not | ☐ | ☐ |
| 3.5 | I understand the role of heterogeneity in ABMs | ☐ | ☐ |

### Practical Skills

| # | Skill | Pre-Lab | Post-Lab |
|---|-------|---------|----------|
| 3.6 | I can implement agent classes with state and behaviour | ☐ | ☐ |
| 3.7 | I can implement neighbourhood finding (Moore, von Neumann) | ☐ | ☐ |
| 3.8 | I can implement the Schelling segregation model | ☐ | ☐ |
| 3.9 | I can implement Boids flocking with steering vectors | ☐ | ☐ |
| 3.10 | I can compute emergence metrics (segregation index, polarisation) | ☐ | ☐ |

### Application

| # | Application | Pre-Lab | Post-Lab |
|---|-------------|---------|----------|
| 3.11 | I can analyse how threshold affects segregation | ☐ | ☐ |
| 3.12 | I can tune Boids parameters for different flocking behaviours | ☐ | ☐ |
| 3.13 | I can design new ABM rules for custom phenomena | ☐ | ☐ |

---

## Progress Tracking

### Pre-Lab Totals (out of 40)

| Section | Confident | Developing | Needs Work |
|---------|-----------|------------|------------|
| Monte Carlo | /13 | /13 | /13 |
| ODE Solvers | /14 | /14 | /14 |
| Agent-Based | /13 | /13 | /13 |
| **Total** | **/40** | **/40** | **/40** |

### Post-Lab Totals (out of 40)

| Section | Confident | Developing | Needs Work |
|---------|-----------|------------|------------|
| Monte Carlo | /13 | /13 | /13 |
| ODE Solvers | /14 | /14 | /14 |
| Agent-Based | /13 | /13 | /13 |
| **Total** | **/40** | **/40** | **/40** |

---

## Reflection Questions

Answer these questions after completing the labs:

### 1. Most valuable insight
*What was the most important concept you learnt this week?*

```
[Your answer]
```

### 2. Remaining challenges
*Which topics do you still find difficult? What specific aspects are unclear?*

```
[Your answer]
```

### 3. Research applications
*How might you apply these techniques to your own research?*

```
[Your answer]
```

### 4. Connections
*How do this week's topics connect to previous weeks (data structures, complexity)?*

```
[Your answer]
```

---

## Action Items

Based on your self-assessment, identify specific actions:

### Topics to Review
1. [ ] _____________________________
2. [ ] _____________________________
3. [ ] _____________________________

### Practice Exercises to Complete
1. [ ] _____________________________
2. [ ] _____________________________
3. [ ] _____________________________

### Questions for Instructor
1. _____________________________
2. _____________________________

---

## Readiness Checklist

Before moving to Week 6, ensure you can:

- [ ] Implement Monte Carlo integration and explain O(1/√n) convergence
- [ ] Implement and compare Euler and RK4 methods
- [ ] Solve systems of ODEs (e.g., Lotka-Volterra)
- [ ] Implement a basic ABM with emergence metrics
- [ ] Explain emergence with concrete examples
- [ ] Choose appropriate numerical methods for different problems

### Minimum Competency
You should have at least 30/40 items marked as "Confident" or "Developing" before proceeding.

---

## Resources for Further Study

If you marked items as "Needs Work", these resources may help:

### Monte Carlo
- Press et al., *Numerical Recipes*, Chapter 7
- Gentle, *Random Number Generation and Monte Carlo Methods*
- [MIT OpenCourseWare: Monte Carlo Methods](https://ocw.mit.edu)

### ODE Solvers
- Butcher, *Numerical Methods for ODEs*
- [Strogatz lectures on dynamical systems](https://www.youtube.com/playlist?list=PLbN57C5Zdl6j_qJA-pARJnKsmROzPnO9V)
- Hairer et al., *Solving ODEs I*

### Agent-Based Modelling
- Schelling, T.C. (1971). "Dynamic Models of Segregation"
- Reynolds, C.W. (1987). "Flocks, Herds and Schools"
- Wilensky & Rand, *Introduction to ABMs*
- [NetLogo online tutorials](https://ccl.northwestern.edu/netlogo/)

---

*© 2025 Antonio Clim. All rights reserved.*
