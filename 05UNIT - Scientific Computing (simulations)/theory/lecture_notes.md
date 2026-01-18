# Week 5: Scientific Computing — Lecture Notes

## Introduction

Scientific computing lies at the heart of modern computational research. When analytical solutions become intractable—as they invariably do for complex systems—numerical methods provide the essential bridge between mathematical models and quantitative predictions. This week explores three foundational techniques: Monte Carlo methods for stochastic computation, ordinary differential equation (ODE) solvers for deterministic dynamics and agent-based models for emergent phenomena.

These methods share a common philosophy: decompose complex problems into manageable computational steps, iterate until convergence and carefully analyse the resulting errors. Mastering them equips you to tackle problems across physics, biology, economics and beyond—anywhere mathematical models meet computational constraints.

---

## 1. Monte Carlo Methods

### 1.1 The Fundamental Idea

Monte Carlo methods exploit randomness to solve deterministic problems. Consider estimating the integral I = ∫ₐᵇ f(x) dx. Rather than analytically finding an antiderivative (often impossible) or using deterministic quadrature (which suffers from the "curse of dimensionality"), we draw random samples {x₁, x₂, ..., xₙ} uniformly from [a, b] and compute:

**Î = (b - a) · (1/n) · Σᵢ f(xᵢ)**

By the Law of Large Numbers, Î → I as n → ∞. The Central Limit Theorem tells us the error decreases as O(1/√n), meaning quadrupling samples halves the error. This rate holds regardless of dimension—a property of fundamental importance, making Monte Carlo uniquely suited to high-dimensional integrals where deterministic methods succumb to the curse of dimensionality.

### 1.2 Estimating π: A Canonical Example

The "hit-or-miss" method elegantly illustrates Monte Carlo integration. Inscribe a unit circle within a 2×2 square centred at the origin. The circle's area is π and the square's area is 4. Sample points (x, y) uniformly from [-1, 1]². The fraction landing inside the circle (where x² + y² ≤ 1) approximates π/4.

```
π̂ = 4 · (points inside circle) / (total points)
```

With 10⁶ samples, expect |π̂ - π| ≈ 0.002, demonstrating the O(1/√n) convergence.

### 1.3 Variance Reduction Techniques

The O(1/√n) rate, while dimension-independent, can be painfully slow. Variance reduction techniques improve convergence without changing its order:

**Antithetic Variates**: For monotonic integrands, pair each sample X with its "mirror" X' = 1 - X (for [0,1] sampling). Since f(X) and f(1-X) are negatively correlated for monotonic f, their average has lower variance than independent samples. Implementation requires only computing f at both points and averaging:

```
Î_antithetic = (1/n) · Σᵢ [f(Xᵢ) + f(1 - Xᵢ)] / 2
```

**Stratified Sampling**: Divide the domain into K strata and sample proportionally from each. This ensures coverage across the domain, reducing variance when f varies significantly. For K equal strata with n/K samples each:

```
Î_stratified = (1/K) · Σₖ [(1/(n/K)) · Σᵢ f(Xₖᵢ)]
```

**Importance Sampling**: Sample from a distribution g(x) that concentrates probability where |f(x)| is large. The estimator becomes:

```
Î = (1/n) · Σᵢ [f(Xᵢ) / g(Xᵢ)]
```

where Xᵢ ~ g. Choosing g ∝ |f| minimises variance but requires knowing f—a classic bias-variance trade-off.

### 1.4 Statistical Error Estimation

Unlike deterministic algorithms, Monte Carlo provides built-in error estimates. The sample variance:

```
s² = (1/(n-1)) · Σᵢ (f(Xᵢ) - f̄)²
```

yields the standard error SE = s/√n, giving a 95% confidence interval of approximately Î ± 2·SE. This self-assessment capability is invaluable in research, where knowing uncertainty is as important as knowing the estimate.

The confidence interval construction relies on the Central Limit Theorem, which guarantees that the sampling distribution of the mean approaches normality for sufficiently large n, regardless of the underlying distribution of f(X). In practice, n ≥ 30 usually suffices for reasonable integrands, though heavy-tailed distributions may require more samples.

**Practical Error Budgeting**: When planning a Monte Carlo computation, researchers often work backwards from a target precision. If the goal is to achieve standard error SE_target, and a pilot study with n_pilot samples yields variance estimate s²_pilot, the required sample size is approximately:

```
n_required ≈ s²_pilot / SE²_target
```

This formula enables efficient resource allocation—particularly valuable when function evaluations are computationally expensive.

**Batch Means Estimation**: For correlated samples (common in Markov Chain Monte Carlo), the standard error formula requires modification. The batch means approach divides the sample into k batches, computes the mean of each batch, then estimates variance from these batch means. This accounts for within-batch correlation while the between-batch variance captures true sampling uncertainty.

---

## 2. Ordinary Differential Equation Solvers

### 2.1 The Initial Value Problem

Many physical systems evolve according to differential equations. The initial value problem (IVP) seeks y(t) satisfying:

```
dy/dt = f(t, y),    y(t₀) = y₀
```

Analytical solutions exist only for special cases (linear equations, separable forms). Numerical methods approximate y at discrete times t₀, t₁, t₂, ... by stepping forward iteratively.

### 2.2 Euler's Method

The simplest approach uses the derivative at the current point to extrapolate:

```
yₙ₊₁ = yₙ + h · f(tₙ, yₙ)
```

where h = tₙ₊₁ - tₙ is the step size. Geometrically, this follows the tangent line. The local truncation error (error per step) is O(h²), but errors accumulate, yielding O(h) global error. Euler's method is first-order accurate.

**Stability**: For the test equation dy/dt = λy (where λ < 0 for decay), Euler requires h < 2/|λ| to prevent divergence. This stability constraint can force impractically small steps for "stiff" problems with disparate timescales.

### 2.3 Runge-Kutta Methods

Higher-order methods sample the derivative at intermediate points within each step. The classic fourth-order Runge-Kutta (RK4) uses four function evaluations:

```
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
k₄ = f(tₙ + h, yₙ + h·k₃)
yₙ₊₁ = yₙ + h · (k₁ + 2k₂ + 2k₃ + k₄) / 6
```

This achieves O(h⁴) global error—halving h reduces error by 16×. RK4 offers an excellent accuracy-to-cost ratio and remains the workhorse for non-stiff problems.

### 2.4 Adaptive Step-Size Control

Fixed step sizes waste computation in smooth regions and risk inaccuracy in rapidly-changing regions. Adaptive methods estimate local error and adjust h accordingly.

The Runge-Kutta-Fehlberg (RKF45) method computes two estimates—fourth and fifth order—using shared function evaluations. Their difference estimates the local error:

```
error ≈ |y₅ₜₕ - y₄ₜₕ|
```

If error exceeds tolerance, reject the step and retry with smaller h. If error is much smaller than tolerance, increase h for efficiency. This "embedded pair" approach achieves specified accuracy with minimal function evaluations.

**Step Size Adjustment Formula**: The optimal step size for the next iteration follows from error scaling arguments. If the current step with size h produces error estimate err, the optimal new step size is:

```
h_new = h · (tol / err)^(1/p) · safety_factor
```

where p is the order of the lower-order method (4 for RKF45) and safety_factor ≈ 0.9 prevents overly aggressive increases. The formula derives from the assumption that local truncation error scales as O(h^(p+1)).

**Stiff Equations**: Some systems contain multiple timescales—fast transients that decay rapidly alongside slow dynamics of interest. Explicit methods like RK4 require h proportional to the fastest timescale for stability, even when the fast dynamics have equilibrated. Implicit methods (backward Euler, BDF) sacrifice some accuracy for unconditional stability, permitting step sizes governed by accuracy rather than stability constraints. The trade-off: implicit methods require solving nonlinear equations at each step, typically via Newton iteration.

**Error Control Strategies**: Two common approaches govern step adjustment. Per-step control maintains local error below tolerance at every step but may accumulate error over long integrations. Per-unit-step control bounds error relative to step size, providing tighter global error control. For research-grade computations, per-unit-step control with conservative tolerances (10⁻⁸ to 10⁻¹⁰) is standard practice.

### 2.5 Systems of ODEs

Vector-valued problems dy/dt = f(t, y) where y ∈ ℝⁿ use the same methods component-wise. Higher-order scalar ODEs (e.g., d²y/dt² = F) convert to first-order systems by introducing auxiliary variables:

```
y₁ = y,  y₂ = dy/dt
dy₁/dt = y₂,  dy₂/dt = F(t, y₁, y₂)
```

This reduction enables solving second-order dynamics like Newton's equations of motion.

---

## 3. Agent-Based Modelling

### 3.1 From Equations to Agents

Differential equations describe aggregate behaviour, but many systems consist of discrete, heterogeneous individuals whose interactions produce collective phenomena. Agent-based models (ABMs) simulate individual "agents"—each with state, behaviour rules and local interactions—allowing macroscopic patterns to emerge organically.

ABMs excel when:
- Individual heterogeneity matters
- Spatial structure influences dynamics
- Feedback and adaptation occur
- Analytical models become intractable

### 3.2 The Schelling Segregation Model

Thomas Schelling's (1971) segregation model demonstrates emergence powerfully. Consider a grid populated by two types of agents. Each agent has a simple preference: if fewer than a threshold fraction (e.g., 30%) of neighbours are similar, the agent moves to a vacant cell.

Remarkably, even this mild individual preference (30% tolerance) produces near-complete segregation (80-90% same-type clusters). The lesson: macroscopic patterns need not reflect individual preferences. This insight revolutionised understanding of social dynamics.

**Implementation considerations**:
- **Grid representation**: 2D array with 0 (empty), 1 (type A), 2 (type B)
- **Neighbourhood**: Moore (8 cells) or von Neumann (4 cells)
- **Update scheme**: Random sequential (one agent per step) or synchronous (all agents update simultaneously)
- **Metrics**: Segregation index (fraction of same-type neighbours averaged over agents)

### 3.3 The Boids Flocking Model

Craig Reynolds' (1987) Boids model generates realistic flocking from three simple rules:

1. **Separation**: Steer away from nearby neighbours to avoid crowding
2. **Alignment**: Match velocity with nearby neighbours
3. **Cohesion**: Steer toward the centre of mass of nearby neighbours

Each rule produces a steering vector; the weighted sum determines the agent's acceleration. "Nearby" typically means within a visual range radius.

```
acceleration = w_sep · separation + w_align · alignment + w_coh · cohesion
```

The resulting flock exhibits lifelike behaviour: cohesive movement, smooth turns, splitting and merging around obstacles. No global coordination exists—the "flock" is purely emergent.

**Implementation considerations**:
- **Spatial indexing**: Grid-based binning or k-d trees for efficient neighbour queries
- **Velocity limits**: Cap speed to prevent runaway acceleration
- **Boundary handling**: Wrap (toroidal), bounce, or steer away from edges

### 3.4 Emergence and Measurement

The central insight of ABMs is emergence: system-level patterns that cannot be predicted from individual rules alone. Measuring emergence requires metrics that capture collective structure:

**Segregation index**: Average fraction of same-type neighbours across all agents. Ranges from 0 (perfect integration) to 1 (complete segregation). The Schelling model typically converges to segregation indices of 0.8-0.95 even with tolerance thresholds as low as 0.3, demonstrating the nonlinear amplification of mild individual preferences into stark collective outcomes.

**Polarisation**: For flocking, the magnitude of the average normalised velocity: |⟨v/|v|⟩|. Equals 1 when all agents move identically, 0 for random directions. The transition from disorder to collective motion often exhibits critical behaviour—small parameter changes near the transition produce large shifts in polarisation.

**Clustering coefficient**: Fraction of agent triads that form closed triangles in the interaction network. High clustering indicates local cohesion; many social networks exhibit clustering coefficients of 0.1-0.5, far above random graph expectations.

**Sensitivity Analysis**: Because emergence arises from nonlinear interactions, ABMs require careful sensitivity analysis. Parameter sweeps—systematically varying one parameter while holding others fixed—reveal how collective outcomes depend on individual rules. Latin Hypercube Sampling enables efficient exploration of high-dimensional parameter spaces. For research applications, reporting sensitivity analyses alongside baseline results is standard practice; conclusions that hinge on finely-tuned parameters warrant scepticism.

**Initialisation Effects**: Many ABMs exhibit path dependence—the final state depends on initial conditions, not merely parameter values. Multiple runs with varied initial configurations test whether observed patterns are consistent or artifacts of particular starting states. Statistical summaries across replications (mean, variance, confidence intervals) provide publishable characterisations of model behaviour.

---

## 4. Convergence and Error Analysis

### 4.1 Empirical Convergence Testing

Theoretical convergence rates (O(1/√n) for MC, O(hᵖ) for p-th order ODE methods) must be verified empirically. The standard approach:

1. Run algorithm with n or h
2. Run with 2n or h/2
3. Compare errors; ratio should be 1/√2 or 2ᵖ

For Monte Carlo, run multiple independent trials to estimate both bias and variance. For ODEs, compare against analytical solutions (when available) or high-precision reference solutions.

### 4.2 Sources of Error

**Truncation error**: Inherent to approximation (MC sampling, ODE discretisation)
**Round-off error**: Finite-precision arithmetic, accumulates over many operations
**Statistical error**: Variance in Monte Carlo estimates

Balancing these sources is an art. Increasing n reduces statistical error but increases round-off accumulation. Decreasing h reduces truncation error but multiplies step count (and round-off).

---

## 5. Practical Considerations

### 5.1 Reproducibility

Stochastic methods require careful seed management:
- Always set explicit seeds for reproducibility
- Document seeds in publications
- Use separate RNG instances for different components

For Monte Carlo studies, the seed should appear in both the code and any published results. When reporting results, specify the seed value, the random number generator used (NumPy's default is PCG64), and the library version. This level of detail permits exact replication years later, even as software evolves.

For ABMs, reproducibility extends beyond random seeds to include initialisation procedures, update orderings (random sequential vs. synchronous), and boundary handling. A complete specification enables other researchers to verify findings and build upon the work.

### 5.2 Performance

Monte Carlo and ABMs often require millions of iterations:
- Vectorise with NumPy where possible
- Profile to identify bottlenecks
- Consider Numba/Cython for critical loops
- Parallelise independent trials

The embarrassingly parallel nature of Monte Carlo makes it ideal for high-performance computing. Each sample is independent, permitting distribution across arbitrary numbers of processors with near-linear speedup. ABMs with local interactions also parallelise well, though synchronisation requirements may limit scaling efficiency.

### 5.3 Validation

Numerical methods require validation against:
- Analytical solutions (when available)
- Limiting cases (zero parameters, long time)
- Conservation laws (energy, mass, probability)
- Previous implementations

For ODEs, conservation properties provide powerful sanity checks. The harmonic oscillator conserves total energy E = ½(v² + x²); a solver that violates this conservation is untrustworthy for long integrations. Symplectic integrators preserve such quantities by construction—a consideration for molecular dynamics and celestial mechanics applications.

---

## Summary

This week introduced three pillars of scientific computing:

1. **Monte Carlo methods** turn random sampling into deterministic estimates, with dimension-independent convergence and built-in error quantification.

2. **ODE solvers** discretise continuous dynamics, trading step size against accuracy with well-understood error bounds.

3. **Agent-based models** simulate individual behaviour to discover emergent collective phenomena invisible to aggregate models.

Together, these techniques enable computational exploration of systems too complex for analytical solution—the essence of modern research computing.

---

## Key Formulae

| Method | Formula | Error |
|--------|---------|-------|
| MC Integration | Î = (b-a)·mean(f(Xᵢ)) | O(1/√n) |
| MC Standard Error | SE = s/√n | — |
| Euler | yₙ₊₁ = yₙ + h·f(tₙ,yₙ) | O(h) |
| RK4 | yₙ₊₁ = yₙ + h(k₁+2k₂+2k₃+k₄)/6 | O(h⁴) |
| Separation | **F** = -Σⱼ (rⱼ - rᵢ)/|rⱼ - rᵢ| | — |
| Alignment | **F** = ⟨vⱼ⟩ - vᵢ | — |
| Cohesion | **F** = ⟨rⱼ⟩ - rᵢ | — |

---

## References

1. Metropolis, N. & Ulam, S. (1949). The Monte Carlo method. *JASA*, 44(247), 335-341.
2. Press, W.H. et al. (2007). *Numerical Recipes*. Cambridge University Press.
3. Schelling, T.C. (1971). Dynamic models of segregation. *J. Math. Sociology*, 1(2), 143-186.
4. Reynolds, C.W. (1987). Flocks, herds and schools. *SIGGRAPH '87*, 25-34.
5. Butcher, J.C. (2016). *Numerical Methods for ODEs*. Wiley.

---

*© 2025 Antonio Clim. All rights reserved.*
