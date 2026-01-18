# Week 5 Cheatsheet: Scientific Computing

## Monte Carlo Integration

### Basic Estimator
```python
# Estimate ∫ₐᵇ f(x) dx
X = np.random.uniform(a, b, n)
estimate = (b - a) * np.mean(f(X))
SE = (b - a) * np.std(f(X)) / np.sqrt(n)
```

### Key Formulae

| Formula | Description |
|---------|-------------|
| Î = (b-a) · mean(f(Xᵢ)) | MC estimator |
| SE = σ/√n | Standard error |
| 95% CI = Î ± 1.96·SE | Confidence interval |
| Error ∝ 1/√n | Convergence rate |

### Variance Reduction

**Antithetic Variates** (monotonic f):
```python
X = np.random.uniform(a, b, n//2)
X_anti = a + b - X  # Mirror samples
estimate = (b-a) * np.mean((f(X) + f(X_anti)) / 2)
```

**Stratified Sampling**:
```python
for k in range(K):  # K strata
    low, high = a + k*w, a + (k+1)*w  # w = (b-a)/K
    X_k = np.random.uniform(low, high, n//K)
    stratum_means[k] = np.mean(f(X_k))
estimate = (b-a) * np.mean(stratum_means)
```

### π Estimation (hit-or-miss)
```python
X, Y = np.random.uniform(0, 1, (2, n))
inside = (X**2 + Y**2) <= 1
pi_estimate = 4 * np.sum(inside) / n
```

---

## ODE Solvers

### Euler's Method
```python
# y' = f(t, y),  y(t₀) = y₀
y_new = y + h * f(t, y)  # O(h) global error
```

### RK4 (Fourth-order Runge-Kutta)
```python
k1 = f(t, y)
k2 = f(t + h/2, y + h*k1/2)
k3 = f(t + h/2, y + h*k2/2)
k4 = f(t + h, y + h*k3)
y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6  # O(h⁴)
```

### Convergence Orders

| Method | Local Error | Global Error | Stages |
|--------|-------------|--------------|--------|
| Euler | O(h²) | O(h) | 1 |
| Midpoint | O(h³) | O(h²) | 2 |
| RK4 | O(h⁵) | O(h⁴) | 4 |
| RKF45 | O(h⁶) | O(h⁵) | 6 |

### Common Systems

**Harmonic oscillator** (y'' + y = 0):
```python
def f(t, y): return np.array([y[1], -y[0]])
# y = [position, velocity]
```

**Lotka-Volterra** (predator-prey):
```python
def f(t, y):
    x, p = y  # prey, predator
    return np.array([
        alpha*x - beta*x*p,    # prey growth - predation
        delta*x*p - gamma*p    # predator growth - death
    ])
```

---

## Agent-Based Modelling

### Schelling Segregation

**Agent rule**: Move if same-type neighbours < threshold
```python
neighbours = get_moore_neighbours(row, col)
same_type = sum(n == agent_type for n in neighbours)
happy = (same_type / len(neighbours)) >= threshold
```

**Segregation index**: Mean same-type neighbour fraction

### Boids Flocking

| Rule | Formula | Effect |
|------|---------|--------|
| Separation | Σ(rᵢ - rⱼ)/\|rᵢ - rⱼ\| | Avoid crowding |
| Alignment | ⟨vⱼ⟩ - vᵢ | Match velocity |
| Cohesion | ⟨rⱼ⟩ - rᵢ | Steer to centre |

```python
v_new = v + w_sep*sep + w_ali*ali + w_coh*coh
speed = np.linalg.norm(v_new)
if speed > max_speed:
    v_new = v_new / speed * max_speed
```

**Polarisation**: |⟨v̂⟩| (0 = random, 1 = aligned)

---

## Common Mistakes

| ❌ Mistake | ✅ Correction |
|-----------|---------------|
| Using `random` instead of `np.random.Generator` | Use `rng = np.random.default_rng(seed)` |
| Forgetting to scale by (b-a) in MC | `estimate = (b-a) * mean(f(X))` |
| RK4 coefficient errors | k₂, k₃ use h/2; weights are 1,2,2,1 |
| Euler on oscillators | Use RK4 for energy conservation |
| Schelling: counting empties as neighbours | Filter out `agent_type == 0` |
| Boids: not limiting speed | Always check `speed <= max_speed` |
| Ignoring seeds | Document and log all random seeds |
| No error estimation in MC | Always compute and report standard error |

---

## Performance Tips

| Technique | When to Use | Speedup Factor |
|-----------|-------------|----------------|
| NumPy vectorisation | Array operations | 10-100× |
| Numba JIT | Tight loops | 10-50× |
| Multiprocessing | Independent trials | ~N cores |
| Spatial hashing (ABM) | Neighbour queries | O(n²) → O(n) |

---

## Code Patterns

### Reproducibility
```python
rng = np.random.default_rng(42)  # Always seed!
X = rng.uniform(a, b, n)
```

### Convergence Study
```python
errors = []
for h in [0.1, 0.05, 0.025, 0.0125]:
    sol = solve_ode(f, (0, T), y0, h=h)
    errors.append(abs(sol.y[-1] - exact(T)))
order = -np.polyfit(np.log(hs), np.log(errors), 1)[0]
```

### ABM Simulation Loop
```python
history = []
for step in range(max_steps):
    metrics = model.step()
    history.append(metrics)
    if equilibrium_reached(metrics):
        break
```

---

## Week Connections

| Week 4 → Week 5 | Week 5 → Week 6 |
|-----------------|-----------------|
| Graphs → Agent networks | Simulation data → Visualisation |
| Bloom filters → Randomness | Convergence plots → Publication figures |
| Complexity → Solver efficiency | ABM patterns → Interactive dashboards |

---

## Quick Reference: Convergence Rates

| Method | Convergence | Doubling Cost Effect |
|--------|-------------|---------------------|
| Monte Carlo | O(1/√n) | Halves error |
| Euler | O(h) | Halves error |
| RK4 | O(h⁴) | Error ÷ 16 |
| RKF45 | O(h⁵) | Error ÷ 32 |

---

*© 2025 Antonio Clim. All rights reserved.*
