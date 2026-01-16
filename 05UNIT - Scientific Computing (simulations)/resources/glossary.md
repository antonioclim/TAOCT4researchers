# Week 5 Glossary: Scientific Computing

## Monte Carlo Methods

### Antithetic Variates
A variance reduction technique that pairs each random sample X with a negatively correlated sample (e.g., 1-X for uniform [0,1]). For monotonic integrands, the negative correlation between f(X) and f(1-X) reduces the variance of their average.

### Central Limit Theorem (CLT)
A theorem stating that the sum (or average) of many independent random variables approaches a normal distribution, regardless of the original distribution. In Monte Carlo, the CLT explains why estimates become approximately Gaussian with increasing sample size.

### Confidence Interval
A range of values that contains the true parameter with a specified probability (typically 95%). For Monte Carlo, the 95% CI is approximately estimate ± 1.96 × standard error.

### Hit-or-Miss Method
A Monte Carlo technique that estimates areas or probabilities by sampling points uniformly and counting the fraction that satisfy a condition (e.g., points inside a circle for estimating π).

### Importance Sampling
A variance reduction technique that samples from a proposal distribution g(x) instead of uniformly, then reweights by f(x)/g(x). Most effective when g(x) concentrates probability where |f(x)| is large.

### Law of Large Numbers
A theorem stating that the sample mean converges to the expected value as sample size increases. This is the fundamental justification for Monte Carlo estimation.

### Monte Carlo Integration
A method for estimating definite integrals using random sampling. The integral ∫f(x)dx is estimated as the average of f evaluated at random points, scaled by the domain volume.

### Standard Error (SE)
The standard deviation of an estimator's sampling distribution. For Monte Carlo, SE = σ/√n, where σ is the integrand's standard deviation and n is the sample size.

### Stratified Sampling
A variance reduction technique that divides the integration domain into strata (sub-regions) and samples from each stratum separately. This ensures coverage across the domain and reduces variance when the integrand varies.

### Variance Reduction
Techniques that decrease the variance of Monte Carlo estimates without increasing sample size. Common methods include antithetic variates, stratified sampling and importance sampling.

---

## ODE Solvers

### Adaptive Step Size
A technique where the step size h is automatically adjusted during integration based on local error estimates. Larger steps are used in smooth regions; smaller steps where the solution changes rapidly.

### Butcher Tableau
A tabular representation of Runge-Kutta method coefficients, specifying where to evaluate the derivative function and how to weight those evaluations.

### Convergence Order
The rate at which numerical error decreases as step size h decreases. An order-p method has global error O(h^p), meaning halving h reduces error by a factor of 2^p.

### Euler's Method
The simplest ODE solver: y_{n+1} = y_n + h·f(t_n, y_n). Uses the derivative at the current point to extrapolate. First-order accurate (global error O(h)).

### Global Error
The accumulated error at the end of integration, resulting from all local truncation errors. For an order-p method, global error is O(h^p).

### Initial Value Problem (IVP)
An ODE with a specified initial condition: dy/dt = f(t, y), y(t_0) = y_0. The goal is to find y(t) for t > t_0.

### Local Truncation Error
The error introduced in a single step, assuming the current value is exact. For Euler, local error is O(h²); for RK4, local error is O(h⁵).

### Lotka-Volterra Equations
A pair of coupled ODEs modelling predator-prey dynamics: dx/dt = αx - βxy (prey), dy/dt = δxy - γy (predator). The system exhibits oscillatory solutions.

### Midpoint Method (RK2)
A second-order Runge-Kutta method that evaluates the derivative at the midpoint of each step, achieving O(h²) global error.

### Runge-Kutta Methods
A family of ODE solvers that achieve high accuracy by evaluating the derivative at multiple points within each step and combining them with carefully chosen weights.

### RK4 (Fourth-order Runge-Kutta)
The "classical" Runge-Kutta method using four function evaluations per step (k_1, k_2, k_3, k_4) to achieve O(h⁴) global error. The workhorse for non-stiff problems.

### RKF45 (Runge-Kutta-Fehlberg)
An embedded pair method computing both 4th and 5th order estimates using 6 function evaluations. The difference provides an error estimate for adaptive stepping.

### Stability
A property of numerical methods describing whether errors grow or decay during integration. Euler's method requires h < 2/|λ| for the test equation y' = λy (λ < 0).

### Stiff Equation
An ODE with vastly different timescales, requiring implicit methods or very small steps for explicit methods. Stiff problems arise in chemical kinetics and electronic circuits.

### Step Size
The time increment h between successive points in numerical integration. Smaller h gives higher accuracy but more computational cost.

---

## Agent-Based Modelling

### Agent
An autonomous entity in a simulation with its own state (attributes) and behaviour rules. Agents interact with their environment and other agents.

### Alignment (Boids)
A flocking rule where each boid adjusts its velocity to match the average velocity of nearby neighbours. Creates coordinated movement.

### Boids
A computational model of flocking behaviour (birds, fish schools) based on three simple rules: separation, alignment and cohesion. Introduced by Craig Reynolds in 1987.

### Cohesion (Boids)
A flocking rule where each boid steers toward the centre of mass of nearby neighbours. Keeps the flock together.

### Emergence
The phenomenon where complex, system-level patterns or behaviours arise from simple local interactions between agents, without being explicitly programmed at the global level.

### Equilibrium
A state where the system no longer changes significantly—in Schelling, when all agents are happy; in Boids, when the flock has stabilised.

### Happy Fraction
In Schelling's model, the proportion of agents whose neighbourhood composition meets their preference threshold.

### Heterogeneity
Variation among agents in their attributes or behaviour rules. Heterogeneity often produces richer emergent dynamics than homogeneous populations.

### Moore Neighbourhood
The eight cells surrounding a central cell in a 2D grid (including diagonals). Contrast with von Neumann neighbourhood (4 cells, cardinal directions only).

### Polarisation
A measure of alignment in flocking models: the magnitude of the average normalised velocity. Equals 1 when all agents move identically, 0 for random directions.

### Schelling Model
Thomas Schelling's (1971) model of residential segregation. Agents of two types move if their neighbourhood has too few same-type neighbours, demonstrating how mild individual preferences produce strong collective segregation.

### Segregation Index
A metric quantifying the degree of spatial clustering by type. Often computed as the average fraction of same-type neighbours across all agents.

### Separation (Boids)
A flocking rule where each boid steers away from neighbours that are too close. Prevents collisions and crowding.

### Threshold (Schelling)
The minimum fraction of same-type neighbours required for an agent to be "happy" and remain in place. Typical values are 0.3-0.5.

### Visual Range
In Boids, the radius within which a boid can perceive other boids. Only neighbours within visual range influence steering.

---

## General Numerical Computing

### Bias
Systematic error in an estimator—the difference between the expected value of the estimate and the true value. An unbiased estimator has zero bias.

### Consistency
A property of estimators: as sample size increases, the estimate converges to the true value. Monte Carlo estimators are consistent.

### Curse of Dimensionality
The exponential growth of computational cost with dimension for deterministic methods (e.g., grid-based quadrature). Monte Carlo avoids this curse.

### Floating-Point Error
Error arising from finite precision representation of real numbers. Accumulates over many operations and can affect numerical stability.

### Numerical Stability
A property of algorithms: stable algorithms do not amplify errors excessively during computation.

### Reproducibility
The ability to obtain identical results from repeated runs. Requires explicit random seed management for stochastic methods.

### Seed
An initial value for a random number generator that determines the entire sequence of "random" numbers. Setting the same seed produces identical sequences.

### Truncation Error
Error introduced by approximating an infinite process (e.g., Taylor series) with a finite computation.

### Unbiasedness
A property where the expected value of an estimator equals the true value being estimated. Monte Carlo integration produces unbiased estimates.

### Vectorisation
Writing code that operates on entire arrays simultaneously rather than looping through elements. Essential for efficient numerical computing in Python/NumPy.

---

*© 2025 Antonio Clim. All rights reserved.*
