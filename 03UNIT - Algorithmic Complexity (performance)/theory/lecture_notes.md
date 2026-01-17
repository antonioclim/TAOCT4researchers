# 03UNIT: Algorithmic Complexity — Lecture Notes

> Understanding and measuring computational performance

---

## 1. Introduction

The study of algorithmic complexity provides researchers with a rigorous framework for predicting and comparing the performance of computational methods. This unit transitions from understanding *what* computation is (01UNIT) and *how* to structure it (02UNIT) to understanding *how well* our algorithms perform as problem sizes grow.

Consider a bioinformatician analysing protein interaction networks. With 10,000 proteins, a poorly chosen algorithm could require weeks of computation where an appropriate one would complete in minutes. The difference between O(n²) and O(n log n) is not merely theoretical—it determines whether research is feasible.

### Historical Context

The formal study of computational complexity began with the work of Hartmanis and Stearns in 1965, who introduced time complexity classes. Their work built upon Alan Turing's foundational concepts from the 1930s and established the mathematical framework we use today.

Perhaps the most famous open problem in computer science is the P vs NP question, posed formally by Stephen Cook in 1971. This problem asks whether every problem whose solution can be verified quickly can also be solved quickly. The Clay Mathematics Institute offers $1,000,000 for its resolution, reflecting its profound importance to cryptography, optimisation and artificial intelligence.

---

## 2. Asymptotic Notation

### 2.1 Big-O Notation

Big-O notation provides an upper bound on the growth rate of a function. Formally:

**Definition:** A function f(n) is O(g(n)) if there exist positive constants c and n₀ such that:

```
f(n) ≤ c · g(n)  for all n ≥ n₀
```

The constants allow us to ignore both constant factors and behaviour for small inputs. We care about *asymptotic* behaviour—how the function grows as n approaches infinity.

**Example:** Consider f(n) = 3n² + 2n + 1. We claim f(n) = O(n²).

*Proof:* For n ≥ 1, we have 3n² + 2n + 1 ≤ 3n² + 2n² + n² = 6n². Thus, with c = 6 and n₀ = 1, the definition is satisfied.

### 2.2 The Asymptotic Family

Three related notations provide different perspectives on growth rates:

1. **Big-O (Upper Bound):** f(n) = O(g(n)) means f grows *no faster* than g.
2. **Big-Ω (Lower Bound):** f(n) = Ω(g(n)) means f grows *at least as fast* as g.
3. **Big-Θ (Tight Bound):** f(n) = Θ(g(n)) means f grows *exactly as fast* as g.

For precise characterisation, Big-Θ is ideal. Merge sort is Θ(n log n)—both its upper and lower bounds are n log n. However, in practice, Big-O is most commonly used because upper bounds are often easier to establish and sufficient for algorithm selection.

### 2.3 Common Complexity Classes

The following table illustrates how different complexity classes scale:

| Class | n=10 | n=1,000 | n=1,000,000 | Example Algorithm |
|-------|------|---------|-------------|-------------------|
| O(1) | 1 | 1 | 1 | Hash table lookup |
| O(log n) | 3 | 10 | 20 | Binary search |
| O(n) | 10 | 1,000 | 1,000,000 | Linear scan |
| O(n log n) | 33 | 10,000 | 20,000,000 | Merge sort |
| O(n²) | 100 | 1,000,000 | 10¹² | Bubble sort |
| O(2ⁿ) | 1,024 | 10³⁰¹ | ∞ | Subset sum |

The exponential class becomes impractical almost immediately. This explains why problems like the travelling salesman (in its exact form) remain computationally intractable for large instances.

---

## 3. Analysing Algorithm Complexity

### 3.1 Basic Rules

Two fundamental rules simplify complexity analysis:

**Sum Rule:** When operations execute sequentially, we take the maximum:
```
O(f(n)) + O(g(n)) = O(max(f(n), g(n)))
```

**Product Rule:** When operations are nested, we multiply:
```
O(f(n)) × O(g(n)) = O(f(n) × g(n))
```

**Example:** Consider this code structure:

```python
for i in range(n):           # O(n)
    simple_operation()       # O(1)
                             # → O(n)
for i in range(n):           # O(n)
    for j in range(n):       # O(n) per outer iteration
        another_operation()  # O(1)
                             # → O(n²)
# Total: O(n) + O(n²) = O(n²) by sum rule
```

### 3.2 Recognising Common Patterns

Certain code patterns correspond to specific complexity classes:

**O(1) — Constant Time:**
- Array access by index
- Hash table lookup (amortised)
- Basic arithmetic operations

**O(log n) — Logarithmic Time:**
- Binary search
- Balanced BST operations
- Halving the problem space each iteration

**O(n) — Linear Time:**
- Single loop over n elements
- Linear search
- One pass through input

**O(n log n) — Linearithmic Time:**
- Efficient sorting (merge sort, quicksort average)
- Divide-and-conquer with linear merge/combine

**O(n²) — Quadratic Time:**
- Nested loops over n elements
- Comparing all pairs
- Simple sorting algorithms (bubble, insertion)

### 3.3 Recurrence Relations

Recursive algorithms are often described by recurrence relations. For example, merge sort satisfies:

```
T(n) = 2T(n/2) + O(n)
```

This expresses that merge sort divides the problem into two half-sized subproblems and spends linear time merging the results.

The **Master Theorem** provides a general solution for recurrences of the form T(n) = aT(n/b) + O(n^d):

- If d > log_b(a): T(n) = O(n^d)
- If d = log_b(a): T(n) = O(n^d log n)
- If d < log_b(a): T(n) = O(n^(log_b(a)))

For merge sort: a = 2, b = 2, d = 1. Since log₂(2) = 1 = d, we have T(n) = O(n log n).

### 3.4 Amortised Analysis

Some data structures have operations that are expensive occasionally but cheap on average. Dynamic arrays exemplify this:

- Most append operations: O(1) — simply write to the next position
- Occasional appends: O(n) — resize and copy when capacity exhausted
- Amortised cost: O(1) per operation

The key insight is that if we double capacity each time, the total cost for n appends is O(n), giving O(1) amortised per operation. This explains why Python lists have efficient appends despite occasional resizing.

---

## 4. Practical Profiling

### 4.1 Why Theory Isn't Enough

Big-O notation deliberately ignores several factors that matter in practice:

1. **Constant factors:** An O(n) algorithm with constant factor 1000 may be slower than O(n²) for small n.
2. **Cache effects:** Memory access patterns dramatically affect performance.
3. **Branch prediction:** Modern CPUs guess which way branches will go.
4. **Implementation language:** Python is typically 10-100× slower than C for equivalent algorithms.

### 4.2 The Memory Hierarchy

Modern computers have a hierarchical memory system:

| Level | Latency | Size | Notes |
|-------|---------|------|-------|
| Registers | ~1 cycle | ~1 KB | Fastest |
| L1 Cache | ~4 cycles | ~64 KB | Per core |
| L2 Cache | ~12 cycles | ~256 KB | Per core |
| L3 Cache | ~40 cycles | ~8-32 MB | Shared |
| RAM | ~200 cycles | ~16-64 GB | Main memory |
| SSD | ~50,000 cycles | ~1 TB | Storage |

A cache miss incurs a 50-200× penalty. This explains why contiguous data structures (arrays) often outperform pointer-based structures (linked lists) despite equivalent theoretical complexity.

### 4.3 Timing in Python

Python provides several timing mechanisms:

```python
import time

# time.time() - wall clock (affected by system load)
# time.perf_counter() - high resolution, recommended for benchmarks
# timeit module - handles warmup and multiple runs automatically
```

For rigorous benchmarking, use `time.perf_counter()` or the `timeit` module.

### 4.4 Profiling Tools

**cProfile** provides function-level profiling:
- Identifies which functions consume the most time
- Low overhead for production code
- Use `pstats` to analyse results

**line_profiler** provides line-level detail:
- Shows exactly which lines are slow
- Higher overhead; use for targeted investigation
- Requires decorator on functions of interest

### 4.5 Optimisation Strategies

Once bottlenecks are identified, consider these strategies:

1. **Algorithmic improvements:** Often provide the largest gains
2. **NumPy vectorisation:** Eliminates Python interpreter overhead
3. **Numba JIT compilation:** Compiles Python to machine code
4. **Cython:** Hybrid Python/C for performance-critical sections
5. **Parallelisation:** Utilise multiple cores with multiprocessing

---

## 5. Empirical Complexity Analysis

### 5.1 The Log-Log Technique

If T(n) = c · n^k, then taking logarithms yields:

```
log T(n) = log c + k · log n
```

On a log-log plot, this is a straight line with slope k. We can therefore:

1. Measure T(n) for several values of n
2. Plot log(T) versus log(n)
3. Fit a line using least squares regression
4. The slope estimates the complexity exponent k

### 5.2 Handling O(n log n)

The simple power-law model doesn't capture O(n log n). For algorithms suspected to be linearithmic:

1. Fit multiple models: O(n), O(n log n), O(n²)
2. Compare sum of squared residuals
3. Select the model with the best fit

```python
from scipy.optimize import curve_fit

def n_log_n_model(n, c):
    return c * n * np.log(n)

params, _ = curve_fit(n_log_n_model, sizes, times)
residuals = np.sum((times - n_log_n_model(sizes, *params)) ** 2)
```

### 5.3 Statistical Considerations

Reliable empirical analysis requires:

- **Multiple runs:** At least 10-30 measurements per data point
- **Warmup:** Discard initial runs (JIT compilation, cache warming)
- **GC control:** Disable garbage collection during measurements
- **Robust statistics:** Report median and interquartile range, not just mean

---

## 6. Research Applications

### 6.1 Bioinformatics

Sequence alignment algorithms illustrate complexity trade-offs:
- **Smith-Waterman:** O(mn) — optimal but slow
- **BLAST:** O(n) heuristic — fast approximation

For genome-scale analysis, the theoretical optimum is often impractical.

### 6.2 Data Science

Sorting algorithm selection depends on data characteristics:
- **Timsort:** O(n) best case for nearly-sorted data
- **Quicksort:** O(n log n) average, but O(n²) worst case
- **Radix sort:** O(nk) for fixed-length integers

Understanding your data distribution enables better algorithm selection.

### 6.3 Network Analysis

Graph algorithms span a wide complexity range:
- **BFS/DFS:** O(V + E)
- **Dijkstra:** O((V + E) log V) with binary heap
- **Floyd-Warshall:** O(V³)

For social networks with billions of nodes, even O(V²) is infeasible.

---

## 7. Key Takeaways

1. **Big-O notation** provides an upper bound on growth rate, enabling algorithm comparison independent of hardware.

2. **The asymptotic family** (O, Ω, Θ) offers different perspectives; Big-O is most commonly used in practice.

3. **Analysis techniques** include sum/product rules, recurrence relations and amortised analysis.

4. **Practical performance** depends on factors Big-O ignores: cache effects, branch prediction and constant factors.

5. **Profiling tools** (cProfile, line_profiler) reveal actual bottlenecks.

6. **Empirical analysis** using log-log regression can estimate complexity from measurements.

7. **Algorithm selection** should consider both theoretical complexity and practical constraints.

---

## 8. Preparation for Week 4

Next week, we examine **Advanced Data Structures**. The complexity analysis skills from this week will help you:

- Evaluate different graph representations
- Compare tree variants (AVL, Red-Black, B-trees)
- Understand probabilistic structures (Bloom filters, Count-Min sketch)

Review the complexity of basic operations (insert, delete, search) on arrays and linked lists before the next session.

---

## References

1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.

2. Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley.

3. Gorelick, M., & Ozsvald, I. (2020). *High Performance Python* (2nd ed.). O'Reilly Media.

4. Hartmanis, J., & Stearns, R. E. (1965). On the computational complexity of algorithms. *Transactions of the American Mathematical Society*, 117, 285-306.

5. Cook, S. A. (1971). The complexity of theorem-proving procedures. *Proceedings of the Third Annual ACM Symposium on Theory of Computing*, 151-158.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Week 3 — Algorithmic Complexity*
*© 2025 Antonio Clim. All rights reserved.*
