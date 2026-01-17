# 05UNIT Homework: Scientific Computing Methods

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | See course schedule |
| **Total Points** | 100 |
| **Estimated Time** | 4–5 hours |
| **Difficulty** | ⭐⭐⭐⭐ (4/5) |

## 🔗 Prerequisites

- [x] Completed Lab 05.1: Monte Carlo Methods
- [x] Completed Lab 05.2: ODE Solvers
- [x] Completed Lab 05.3: Agent-Based Modelling
- [x] Read lecture notes on numerical methods

## 🎯 Objectives Assessed

1. **[Apply]** Implement Monte Carlo integration with variance reduction
2. **[Apply]** Solve differential equations with appropriate numerical methods
3. **[Create]** Design agent-based models exhibiting emergent behaviour

---

## Part 1: Monte Carlo Methods (35 points)

### Context

Researchers frequently encounter integrals lacking closed-form solutions. This section assesses your capacity to implement Monte Carlo estimators and evaluate their statistical properties.

### Task 1.1: Multi-dimensional Integration (15 points)

Implement Monte Carlo integration for the following integral over the unit hypercube:

$$I = \int_0^1 \int_0^1 \int_0^1 \exp(-(x^2 + y^2 + z^2)) \, dx \, dy \, dz$$

**Requirements:**

| Req | Points | Description |
|-----|--------|-------------|
| 1.1.1 | 5 | Implement `monte_carlo_3d(f, n_samples, seed)` returning estimate and standard error |
| 1.1.2 | 5 | Compute estimates for $n \in \{10^3, 10^4, 10^5, 10^6\}$ |
| 1.1.3 | 5 | Plot convergence: estimate vs $n$ with 95% confidence bands |

**Function Signature:**

```python
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray

@dataclass
class MCResult:
    estimate: float
    standard_error: float
    n_samples: int

def monte_carlo_3d(
    f: Callable[[NDArray, NDArray, NDArray], NDArray],
    n_samples: int,
    seed: int | None = None,
) -> MCResult:
    """
    Monte Carlo integration over the unit cube [0,1]³.
    
    Args:
        f: Integrand function f(x, y, z) accepting arrays.
        n_samples: Number of random samples.
        seed: Random state for reproducibility.
    
    Returns:
        MCResult with estimate, standard error, and sample count.
    """
    # YOUR IMPLEMENTATION HERE
    ...
```

**Test Cases:**

```python
# Define integrand
def integrand(x, y, z):
    return np.exp(-(x**2 + y**2 + z**2))

# Basic functionality
result = monte_carlo_3d(integrand, n_samples=100000, seed=42)
assert 0.4 < result.estimate < 0.6  # True value ≈ 0.4964
assert result.standard_error > 0
assert result.n_samples == 100000

# Reproducibility
r1 = monte_carlo_3d(integrand, 10000, seed=123)
r2 = monte_carlo_3d(integrand, 10000, seed=123)
assert r1.estimate == r2.estimate
```

### Task 1.2: Variance Reduction via Antithetic Variates (20 points)

Implement antithetic variate sampling for 1D integration and demonstrate variance reduction.

**Problem:** Estimate $I = \int_0^1 e^x \, dx$ (exact value: $e - 1 \approx 1.7183$)

| Req | Points | Description |
|-----|--------|-------------|
| 1.2.1 | 8 | Implement `monte_carlo_antithetic(f, n_samples, seed)` |
| 1.2.2 | 6 | Compare variance: naive MC vs antithetic for $n = 10^5$ |
| 1.2.3 | 6 | Compute empirical variance reduction factor $\text{VRF} = \sigma^2_{\text{naive}} / \sigma^2_{\text{antithetic}}$ |

**Antithetic Method:**

For samples $U_i \sim \text{Uniform}(0,1)$, the antithetic estimator uses both $U_i$ and $1 - U_i$:

$$\hat{I}_n^{\text{anti}} = \frac{1}{n} \sum_{i=1}^{n} \frac{f(U_i) + f(1 - U_i)}{2}$$

<details>
<summary>💡 Hint 1: Variance formula</summary>

The variance of the antithetic estimator is:
$$\text{Var}[\hat{I}^{\text{anti}}] = \frac{1}{4n}\left(\text{Var}[f(U)] + \text{Var}[f(1-U)] + 2\text{Cov}[f(U), f(1-U)]\right)$$

For monotonic $f$, the covariance is negative, reducing variance.
</details>

<details>
<summary>💡 Hint 2: Empirical variance</summary>

Compute empirical variance from multiple independent runs (e.g., 100 repetitions), not from within a single run.
</details>

---

## Part 2: ODE Solvers (35 points)

### Context

Differential equations model continuous dynamical systems. This section evaluates your implementation of numerical integration methods and their error characteristics.

### Task 2.1: Lotka-Volterra Predator-Prey System (20 points)

Implement and solve the Lotka-Volterra equations:

$$\frac{dx}{dt} = \alpha x - \beta xy$$
$$\frac{dy}{dt} = \delta xy - \gamma y$$

where $x$ is prey population and $y$ is predator population.

| Req | Points | Description |
|-----|--------|-------------|
| 2.1.1 | 8 | Implement `rk4_system(f, y0, t_span, h)` for systems of ODEs |
| 2.1.2 | 6 | Solve with parameters $\alpha=1.1, \beta=0.4, \gamma=0.4, \delta=0.1$ |
| 2.1.3 | 6 | Plot phase portrait ($x$ vs $y$) and time series ($x(t)$, $y(t)$) |

**Initial conditions:** $x_0 = 10$, $y_0 = 5$

**Integration interval:** $t \in [0, 100]$ with step size $h = 0.01$

**Function Signature:**

```python
def rk4_system(
    f: Callable[[float, NDArray], NDArray],
    y0: NDArray,
    t_span: tuple[float, float],
    h: float,
) -> tuple[NDArray, NDArray]:
    """
    Solve ODE system using RK4.
    
    Args:
        f: System function f(t, y) returning dy/dt.
        y0: Initial state vector.
        t_span: Integration interval (t0, tf).
        h: Step size.
    
    Returns:
        Tuple of (t_array, y_array) where y_array[i] is state at t_array[i].
    """
    # YOUR IMPLEMENTATION HERE
    ...
```

**Test Cases:**

```python
# Simple exponential decay: dy/dt = -y, y(0) = 1
def decay(t, y):
    return -y

t, y = rk4_system(decay, np.array([1.0]), (0, 5), h=0.1)
# y(5) should be close to exp(-5) ≈ 0.00674
assert abs(y[-1, 0] - np.exp(-5)) < 1e-6
```

### Task 2.2: Error Analysis (15 points)

Compare Euler and RK4 methods on the simple harmonic oscillator:

$$\frac{d^2\theta}{dt^2} = -\omega^2 \theta$$

Rewritten as a first-order system with $y_1 = \theta$, $y_2 = \dot{\theta}$:

$$\frac{dy_1}{dt} = y_2, \quad \frac{dy_2}{dt} = -\omega^2 y_1$$

| Req | Points | Description |
|-----|--------|-------------|
| 2.2.1 | 5 | Implement `euler_system(f, y0, t_span, h)` |
| 2.2.2 | 5 | Compute global error at $t = 2\pi$ for $h \in \{0.1, 0.05, 0.01, 0.005\}$ |
| 2.2.3 | 5 | Plot log-log error vs step size; verify $O(h)$ for Euler, $O(h^4)$ for RK4 |

**Parameters:** $\omega = 1$, $\theta(0) = 1$, $\dot{\theta}(0) = 0$

**Exact solution:** $\theta(t) = \cos(t)$

<details>
<summary>💡 Hint: Log-log slope</summary>

On a log-log plot, $\log(\text{error}) = p \cdot \log(h) + C$ where $p$ is the order.
The slope of the line reveals the convergence order.
</details>

---

## Part 3: Agent-Based Modelling (30 points)

### Context

Agent-based models capture emergent phenomena arising from local interactions. This section assesses your capacity to design, implement and analyse such systems.

### Task 3.1: Extended Schelling Model (15 points)

Extend the basic Schelling segregation model with heterogeneous tolerance thresholds.

| Req | Points | Description |
|-----|--------|-------------|
| 3.1.1 | 5 | Implement agents with individual tolerance values $\tau_i \sim \text{Uniform}(0.2, 0.8)$ |
| 3.1.2 | 5 | Run simulation for 1000 steps on 50×50 grid with 70% occupancy |
| 3.1.3 | 5 | Compute and plot segregation index over time |

**Segregation Index:**

$$S = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{same-type neighbours of } i}{\text{total neighbours of } i}$$

Values range from 0.5 (random mixing) to 1.0 (complete segregation).

**Function Signature:**

```python
@dataclass
class SchellingAgent:
    x: int
    y: int
    agent_type: int  # 0 or 1
    tolerance: float  # Individual threshold

class SchellingModel:
    def __init__(
        self,
        grid_size: int,
        occupancy: float,
        tolerance_range: tuple[float, float] = (0.2, 0.8),
        seed: int | None = None,
    ) -> None:
        """Initialise Schelling model with heterogeneous agents."""
        ...
    
    def step(self) -> None:
        """Execute one simulation step."""
        ...
    
    def segregation_index(self) -> float:
        """Compute current segregation index."""
        ...
    
    def run(self, steps: int) -> list[float]:
        """Run simulation, returning segregation index history."""
        ...
```

### Task 3.2: Boids Analysis (15 points)

Implement parameter sensitivity analysis for the Boids flocking model.

| Req | Points | Description |
|-----|--------|-------------|
| 3.2.1 | 5 | Implement `compute_order_parameter(boids)` measuring alignment |
| 3.2.2 | 5 | Vary separation weight $w_s \in [0.5, 2.0]$ and measure steady-state order |
| 3.2.3 | 5 | Plot order parameter vs separation weight; identify phase transition |

**Order Parameter:**

$$\phi = \frac{1}{N} \left| \sum_{i=1}^{N} \frac{\mathbf{v}_i}{|\mathbf{v}_i|} \right|$$

Values range from 0 (disordered) to 1 (perfectly aligned).

<details>
<summary>💡 Hint: Steady state</summary>

Run the simulation for 500 steps before measuring, to allow transients to decay.
Average the order parameter over the final 100 steps.
</details>

---

## ✅ Submission Checklist

### Code Quality

- [ ] All functions include type hints
- [ ] Google-style docstrings present
- [ ] No `print()` statements (use logging)
- [ ] Code passes `ruff check`
- [ ] Code passes `mypy --strict`

### Testing

- [ ] All provided test cases pass
- [ ] Additional edge cases tested
- [ ] Results reproducible with fixed seeds

### Documentation

- [ ] Convergence plots included (PNG or PDF)
- [ ] Results summarised in brief report (max 500 words)
- [ ] Code comments explain non-obvious logic

### Submission Format

```
homework_05/
├── part1_monte_carlo.py
├── part2_ode_solvers.py
├── part3_agent_based.py
├── plots/
│   ├── mc_convergence.png
│   ├── lotka_volterra_phase.png
│   ├── euler_vs_rk4_error.png
│   ├── schelling_segregation.png
│   └── boids_order_parameter.png
└── report.md
```

---

## 🎯 Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part 1.1 | 15 | Correct MC estimate, proper error computation |
| Part 1.2 | 20 | Antithetic implementation, variance comparison |
| Part 2.1 | 20 | RK4 system solver, Lotka-Volterra solution |
| Part 2.2 | 15 | Error analysis, convergence order verification |
| Part 3.1 | 15 | Extended Schelling, segregation metric |
| Part 3.2 | 15 | Boids analysis, order parameter |
| **Total** | **100** | |

### Bonus Points (up to 10)

| Bonus | Points | Description |
|-------|--------|-------------|
| B1 | +3 | Implement importance sampling for Part 1 |
| B2 | +3 | Implement adaptive RK45 for Part 2 |
| B3 | +4 | Add predator satiation to Lotka-Volterra (Type II functional response) |

---

## 📚 References

1. Press, W.H. et al. (2007). *Numerical Recipes*. Cambridge University Press.
2. Epstein, J.M. & Axtell, R. (1996). *Growing Artificial Societies*. MIT Press.
3. Reynolds, C.W. (1987). Flocks, herds and schools: A distributed behavioral model. *SIGGRAPH*.
4. Schelling, T.C. (1971). Dynamic models of segregation. *Journal of Mathematical Sociology*.

---

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
