# Week 3: Algorithmic Complexity Cheatsheet

> **Quick Reference** — Maximum 2 A4 pages  
> © 2025 Antonio Clim. All rights reserved.

---

## Big-O Notation Quick Reference

| Notation | Name | Example | Operations (n=1000) |
|----------|------|---------|---------------------|
| O(1) | Constant | Array access | 1 |
| O(log n) | Logarithmic | Binary search | 10 |
| O(n) | Linear | Linear search | 1,000 |
| O(n log n) | Linearithmic | Merge sort | 10,000 |
| O(n²) | Quadratic | Bubble sort | 1,000,000 |
| O(n³) | Cubic | Matrix multiplication | 1,000,000,000 |
| O(2ⁿ) | Exponential | Subset enumeration | 10³⁰⁰ |
| O(n!) | Factorial | Permutation enumeration | 10²⁵⁶⁷ |

---

## Formal Definitions

**Big-O (Upper Bound):**
f(n) ∈ O(g(n)) ⟺ ∃c > 0, n₀ ≥ 1 : ∀n ≥ n₀, f(n) ≤ c·g(n)

**Big-Ω (Lower Bound):**
f(n) ∈ Ω(g(n)) ⟺ ∃c > 0, n₀ ≥ 1 : ∀n ≥ n₀, f(n) ≥ c·g(n)

**Big-Θ (Tight Bound):**
f(n) ∈ Θ(g(n)) ⟺ f(n) ∈ O(g(n)) ∧ f(n) ∈ Ω(g(n))

---

## Common Algorithm Complexities

| Algorithm | Best | Average | Worst | Space |
|-----------|------|---------|-------|-------|
| Linear search | O(1) | O(n) | O(n) | O(1) |
| Binary search | O(1) | O(log n) | O(log n) | O(1) |
| Bubble sort | O(n) | O(n²) | O(n²) | O(1) |
| Insertion sort | O(n) | O(n²) | O(n²) | O(1) |
| Merge sort | O(n log n) | O(n log n) | O(n log n) | O(n) |
| Quicksort | O(n log n) | O(n log n) | O(n²) | O(log n) |
| Heapsort | O(n log n) | O(n log n) | O(n log n) | O(1) |
| Hash table lookup | O(1) | O(1) | O(n) | O(n) |

---

## Master Theorem

For recurrences of the form: **T(n) = aT(n/b) + f(n)**

Let k = log_b(a)

| Case | Condition | Solution |
|------|-----------|----------|
| 1 | f(n) = O(n^(k-ε)) for ε > 0 | T(n) = Θ(n^k) |
| 2 | f(n) = Θ(n^k) | T(n) = Θ(n^k log n) |
| 3 | f(n) = Ω(n^(k+ε)) and af(n/b) ≤ cf(n) | T(n) = Θ(f(n)) |

**Examples:**
- T(n) = 2T(n/2) + n → Case 2 → Θ(n log n) [Merge sort]
- T(n) = T(n/2) + 1 → Case 2 → Θ(log n) [Binary search]
- T(n) = 4T(n/2) + n → Case 1 → Θ(n²)

---

## Code Patterns for Analysis

### Pattern 1: Single Loop — O(n)
```python
for i in range(n):
    # O(1) work
```

### Pattern 2: Nested Loops — O(n²)
```python
for i in range(n):
    for j in range(n):
        # O(1) work
```

### Pattern 3: Dependent Loops — O(n²)
```python
for i in range(n):
    for j in range(i):  # O(n(n-1)/2) = O(n²)
        # O(1) work
```

### Pattern 4: Logarithmic Loop — O(log n)
```python
i = n
while i > 0:
    # O(1) work
    i //= 2
```

### Pattern 5: Logarithmic Nested — O(n log n)
```python
for i in range(n):
    j = n
    while j > 0:
        # O(1) work
        j //= 2
```

---

## Benchmarking Code Pattern

```python
import time
import statistics

def benchmark(func, *args, runs=10, warmup=2):
    """Reliable benchmarking with warmup."""
    # Warmup
    for _ in range(warmup):
        func(*args)
    
    # Measure
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return {
        'mean': statistics.mean(times),
        'stdev': statistics.stdev(times) if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }
```

---

## Empirical Complexity Estimation

```python
import numpy as np
from scipy.optimize import curve_fit

def estimate_complexity(sizes, times):
    """Estimate complexity exponent from measurements."""
    log_n = np.log(sizes)
    log_t = np.log(times)
    
    # Linear regression: log(t) = a*log(n) + b
    # Therefore: t ≈ n^a (exponent is slope)
    coeffs = np.polyfit(log_n, log_t, 1)
    exponent = coeffs[0]
    
    return exponent  # ~1 for O(n), ~2 for O(n²), etc.
```

---

## Common Mistakes to Avoid

| ❌ Mistake | ✓ Correction |
|------------|--------------|
| Ignoring constants | Constants matter for small n; Big-O is for large n |
| Confusing O and Θ | O is upper bound; Θ is tight bound |
| Forgetting space | Recursion uses O(depth) stack space |
| Timing once | Always take multiple measurements |
| No warmup | JIT and caches need warmup iterations |
| Wrong input size | Benchmark with realistic data sizes |
| Measuring setup | Exclude data generation from timing |

---

## Amortised Analysis Summary

**Aggregate Method:** Total cost / Number of operations

**Accounting Method:** Assign amortised costs; maintain non-negative credit

**Potential Method:** Φ(state) function; amortised = actual + ΔΦ

**Classic Example — Dynamic Array:**
- Append: O(1) amortised (despite O(n) worst-case resize)
- Proof: Total copies ≤ 2n over n operations

---

## Memory Hierarchy and Cache Effects

| Level | Size | Latency | Bandwidth |
|-------|------|---------|-----------|
| L1 Cache | 32-64 KB | ~1 ns | ~100 GB/s |
| L2 Cache | 256-512 KB | ~4 ns | ~50 GB/s |
| L3 Cache | 4-32 MB | ~15 ns | ~30 GB/s |
| Main Memory | 8-128 GB | ~100 ns | ~20 GB/s |
| SSD | 256 GB-4 TB | ~100 µs | ~3 GB/s |

**Cache Line:** Typically 64 bytes — access sequentially for best performance.

---

## Key Formulas

| Sum | Closed Form | Complexity |
|-----|-------------|------------|
| Σᵢ₌₁ⁿ 1 | n | O(n) |
| Σᵢ₌₁ⁿ i | n(n+1)/2 | O(n²) |
| Σᵢ₌₁ⁿ i² | n(n+1)(2n+1)/6 | O(n³) |
| Σᵢ₌₀ⁿ rⁱ (r≠1) | (rⁿ⁺¹-1)/(r-1) | O(rⁿ) |
| Σᵢ₌₁ⁿ 1/i | ln(n) + γ | O(log n) |
| log(n!) | n log n - n + O(log n) | O(n log n) |

---

## Connections to Other Weeks

| Week | Topic | Connection |
|------|-------|------------|
| Week 2 | OOP Patterns | Complexity affects pattern choice |
| Week 4 | Data Structures | Choose structures by complexity needs |
| Week 5 | Simulations | Scalability determines feasible sizes |
| Week 6 | Visualisation | Plot complexity curves |
| Week 7 | Testing | Benchmark in CI/CD pipelines |

---

## Python Profiling Quick Reference

```bash
# Time profiling
python -m cProfile -s cumulative script.py

# Memory profiling (requires memory_profiler)
python -m memory_profiler script.py

# Line-by-line profiling (requires line_profiler)
kernprof -l -v script.py
```

---

*See `further_reading.md` for thorough resources.*

© 2025 Antonio Clim. All rights reserved.
