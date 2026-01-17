# Week 5 Quiz: Scientific Computing

## 📋 Instructions

- **Time limit**: 20 minutes
- **Questions**: 10 (6 multiple choice + 4 short answer)
- **Total marks**: 20
- **Passing score**: 14/20 (70%)

Answer all questions. Multiple choice questions have exactly one correct answer.

---

## Multiple Choice Questions (2 marks each)

### Question 1
What is the convergence rate of standard Monte Carlo integration?

A) O(n)  
B) O(1/n)  
C) O(1/√n)  
D) O(log n)

---

### Question 2
Which ODE solver has O(h⁴) global error?

A) Euler's method  
B) Midpoint method (RK2)  
C) Fourth-order Runge-Kutta (RK4)  
D) Backward Euler

---

### Question 3
In the Schelling segregation model, what causes agents to move?

A) Random displacement at each time step  
B) Attraction to agents of the opposite type  
C) Having fewer same-type neighbours than their threshold  
D) Global optimisation of segregation index

---

### Question 4
What is the primary advantage of antithetic variates in Monte Carlo?

A) Faster computation per sample  
B) Reduced variance for monotonic integrands  
C) Elimination of bias  
D) Better handling of discontinuous functions

---

### Question 5
In the Boids flocking model, which rule causes agents to avoid collisions?

A) Separation  
B) Alignment  
C) Cohesion  
D) Migration

---

### Question 6
What does the Runge-Kutta-Fehlberg (RKF45) method provide that fixed-step RK4 does not?

A) Higher accuracy for all problems  
B) Automatic step size adjustment based on error estimates  
C) Better stability for stiff equations  
D) Implicit time stepping

---

## Short Answer Questions (2 marks each)

### Question 7
Explain why Monte Carlo integration maintains O(1/√n) convergence regardless of the dimension of the integral, while deterministic quadrature rules suffer from the "curse of dimensionality".

*Write 2-3 sentences.*

---

### Question 8
In the harmonic oscillator system (y'' + y = 0), Euler's method exhibits energy drift while RK4 approximately conserves energy. Explain why this occurs.

*Write 2-3 sentences.*

---

### Question 9
Define "emergence" in the context of agent-based modelling and give one example from the Schelling or Boids model.

*Write 2-3 sentences.*

---

### Question 10
Describe the trade-off between step size (h) and accuracy in ODE solvers. Why might you choose an adaptive method over a fixed-step method?

*Write 2-3 sentences.*

---

## Answer Key

<details>
<summary>Click to reveal answers</summary>

### Multiple Choice Answers

**Q1: C) O(1/√n)**

The standard error of Monte Carlo estimation is SE = σ/√n, giving O(1/√n) convergence. Quadrupling the number of samples halves the error.

**Q2: C) Fourth-order Runge-Kutta (RK4)**

RK4 has local truncation error O(h⁵) and global error O(h⁴). Euler is O(h), midpoint is O(h²).

**Q3: C) Having fewer same-type neighbours than their threshold**

Agents move when their "happiness" condition is not met—specifically, when the fraction of same-type neighbours falls below their preference threshold.

**Q4: B) Reduced variance for monotonic integrands**

Antithetic variates pair each sample X with its "mirror" (1-X for [0,1]). For monotonic f, f(X) and f(1-X) are negatively correlated, reducing the variance of their average.

**Q5: A) Separation**

Separation causes boids to steer away from nearby neighbours to avoid crowding. Alignment matches velocity and cohesion steers toward the centre of mass.

**Q6: B) Automatic step size adjustment based on error estimates**

RKF45 computes both 4th and 5th order estimates, using their difference to estimate local error and adjust step size accordingly.

### Short Answer Marking Guide

**Q7 (2 marks)**
- 1 mark: Monte Carlo samples points randomly without regard to dimension
- 1 mark: Deterministic methods require O((1/ε)^d) points to achieve error ε in d dimensions, while MC needs O(1/ε²) regardless of d

*Sample answer*: Monte Carlo integration samples random points and estimates the integral as a sample mean, with variance that depends only on the integrand's variance—not on dimension. Deterministic quadrature rules (like Simpson's or Gaussian) require a grid of points, and the number of grid points grows exponentially with dimension (curse of dimensionality), while MC maintains O(1/√n) convergence in any dimension.

**Q8 (2 marks)**
- 1 mark: Euler uses only the derivative at the start of each step (tangent line)
- 1 mark: RK4 samples multiple points within each step, better approximating the curved trajectory

*Sample answer*: Euler's method approximates the solution using only the tangent at each step's start, causing systematic drift in oscillatory systems—typically gaining energy (spiralling outward). RK4 uses four derivative evaluations across each step, effectively averaging the curvature and better preserving the closed orbits that characterise energy conservation in conservative systems.

**Q9 (2 marks)**
- 1 mark: Define emergence as macroscopic patterns arising from local rules/interactions
- 1 mark: Give a valid example (Schelling: global segregation from local preferences; Boids: flocking from separation/alignment/cohesion)

*Sample answer*: Emergence refers to complex, system-level patterns or behaviours that arise spontaneously from simple local interactions between agents, without being explicitly programmed at the global level. In the Schelling model, emergence is demonstrated when agents with only 30% same-type preference (a mild individual preference) collectively produce 80-90% segregation (a strong global pattern).

**Q10 (2 marks)**
- 1 mark: Smaller h gives higher accuracy but more computational cost
- 1 mark: Adaptive methods balance accuracy and efficiency by using small steps where needed and large steps where the solution is smooth

*Sample answer*: Smaller step sizes reduce truncation error but increase computational cost (more steps needed). Fixed-step methods waste computation in smooth regions and may lose accuracy in rapidly-changing regions. Adaptive methods estimate local error at each step and adjust h accordingly—taking small steps through challenging regions and large steps through smooth regions—achieving specified accuracy with minimal total function evaluations.

</details>

---

*© 2025 Antonio Clim. All rights reserved.*
