# Week 3 Homework: Algorithmic Complexity and Optimisation

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | Friday 23:59 GMT |
| **Total Points** | 100 |
| **Estimated Time** | 6-8 hours |
| **Difficulty** | ⭐⭐⭐ (3/5) |

## 🔗 Prerequisites

- [ ] Completed Lab 3.1 (Benchmark Suite)
- [ ] Completed Lab 3.2 (Complexity Analyser)
- [ ] Read lecture notes on asymptotic notation
- [ ] Reviewed profiling tools (cProfile, timeit)

## 🎯 Objectives Assessed

1. [Apply] Implement benchmarking framework with statistical analysis
2. [Analyse] Estimate complexity of algorithms empirically and theoretically
3. [Evaluate] Compare algorithm performance across different data distributions

---

## Part 1: Sorting Algorithm Analysis (25 points)

### Context

Sorting algorithms exhibit vastly different performance characteristics depending on input data distribution. Understanding these nuances is crucial for selecting appropriate algorithms in research contexts.

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 1.1 | 10 | Benchmark algorithms on four data distributions |
| 1.2 | 10 | Find and explain crossover points |
| 1.3 | 5 | Implement and optimise hybrid sort |

### Task 1.1: Behaviour on Different Distributions (10 points)

Extend the benchmark suite from the laboratory to analyse these distributions:

- **Random uniform:** Standard random data
- **Nearly sorted:** 10 swaps per n elements
- **Reverse sorted:** Descending order
- **Many duplicates:** Only k unique values (k = √n)

Create comparison tables and plots for bubble sort, insertion sort, merge sort, quicksort and Python's built-in `sorted()`.

### Task 1.2: Crossover Points (10 points)

Determine empirically:

1. At what input size does quicksort become faster than insertion sort?
2. At what input size does Timsort outperform quicksort on nearly-sorted data?
3. At what input size does the overhead of merge sort's recursion become negligible?

Provide both experimental results and theoretical justification.

### Task 1.3: Hybrid Sort (5 points)

Implement a hybrid sorting algorithm that:

- Uses insertion sort for subarrays below a threshold
- Uses merge sort otherwise

Find the optimal threshold experimentally and explain why this hybrid approach works.

### Test Cases

```python
# Your benchmark should handle these cases
sizes = [100, 500, 1000, 2000, 5000, 10000]
distributions = ['random', 'nearly_sorted', 'reverse', 'duplicates']

# Expected: Tables and plots showing relative performance
assert len(results) == len(sizes) * len(distributions) * num_algorithms
```

### Deliverables

- `benchmark_analysis.py` — Extended benchmark code
- `analysis_report.md` — Report with figures and conclusions
- `hybrid_sort.py` — Implementation with tests

<details>
<summary>💡 Hint 1</summary>

For nearly-sorted data generation:
```python
def generate_nearly_sorted(n: int, swaps: int = 10) -> list[int]:
    data = list(range(n))
    for _ in range(swaps):
        i, j = random.randrange(n), random.randrange(n)
        data[i], data[j] = data[j], data[i]
    return data
```
</details>

<details>
<summary>💡 Hint 2</summary>

For crossover point detection, use binary search over input sizes and compare median times.
</details>

---

## Part 2: Complexity Analysis on Real Code (25 points)

### Context

Theoretical complexity analysis requires careful examination of code structure. This exercise develops your ability to analyse arbitrary code and verify your analysis empirically.

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 2.1 | 15 | Analyse complexity of five mystery functions |
| 2.2 | 10 | Verify analyses with empirical measurements |

### Task 2.1: Theoretical Analysis (15 points)

Analyse the time and space complexity of each function. Provide step-by-step reasoning.

```python
def mystery_1(n: int) -> int:
    """Mystery function 1."""
    count = 0
    i = n
    while i > 0:
        for j in range(i):
            count += 1
        i = i // 2
    return count


def mystery_2(data: list[int]) -> list[int]:
    """Mystery function 2."""
    result = []
    for i in range(len(data)):
        for j in range(i, len(data)):
            if data[i] > data[j]:
                data[i], data[j] = data[j], data[i]
        result.append(data[i])
    return result


def mystery_3(n: int) -> int:
    """Mystery function 3."""
    if n <= 1:
        return n
    return mystery_3(n - 1) + mystery_3(n - 2)


def mystery_4(matrix: list[list[int]]) -> int:
    """Mystery function 4 (assume square matrix n×n)."""
    n = len(matrix)
    total = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                total += matrix[i][k] * matrix[k][j]
    return total


def mystery_5(data: list[int], target: int) -> int:
    """Mystery function 5 (assume sorted input)."""
    lo, hi = 0, len(data) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

### Task 2.2: Empirical Verification (10 points)

Use the ComplexityAnalyser from Lab 3.2 to:

1. Measure execution time for each function across input sizes
2. Estimate complexity using log-log regression
3. Compare empirical estimates with theoretical analysis
4. Explain any discrepancies

### Test Cases

```python
# Your analysis should produce results like:
# mystery_1: Theoretical O(?), Empirical O(?), R² = ?
# mystery_2: Theoretical O(?), Empirical O(?), R² = ?
# ...
```

### Deliverables

- `complexity_analysis.py` — Code for empirical verification
- `analysis.md` — Detailed theoretical analysis with derivations

<details>
<summary>💡 Hint</summary>

For mystery_1, trace through what happens when n = 16:
- First iteration: i = 16, inner loop runs 16 times
- Second iteration: i = 8, inner loop runs 8 times
- Third iteration: i = 4, inner loop runs 4 times
- ...

What's the total? How does this generalise?
</details>

---

## Part 3: Python Optimisation Challenge (30 points)

### Context

A common research task is computing pairwise distances between points. The naive implementation is O(n²), but significant constant-factor improvements are possible.

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 3.1 | 10 | Implement three versions (naive, NumPy, Numba) |
| 3.2 | 10 | Benchmark and analyse speedup |
| 3.3 | 10 | Profile and explain performance differences |

### Task 3.1: Three Implementations (10 points)

Implement pairwise Euclidean distance calculation:

```python
def distance_matrix_naive(points: list[list[float]]) -> list[list[float]]:
    """Pure Python implementation."""
    ...

def distance_matrix_numpy(points: np.ndarray) -> np.ndarray:
    """NumPy vectorised implementation."""
    ...

def distance_matrix_numba(points: np.ndarray) -> np.ndarray:
    """Numba JIT-compiled implementation."""
    ...
```

All implementations must produce identical results (within floating-point tolerance).

### Task 3.2: Benchmark Analysis (10 points)

1. Benchmark all three implementations for n = 100, 500, 1000, 2000, 5000
2. Calculate speedup of NumPy and Numba relative to naive
3. Plot speedup vs input size
4. Estimate when Numba's JIT compilation overhead is amortised

### Task 3.3: Profiling Deep Dive (10 points)

Use cProfile and line_profiler to:

1. Identify the exact bottleneck in the naive implementation
2. Explain why NumPy is faster (what overhead does it eliminate?)
3. Explain why Numba is faster still (what does JIT compilation enable?)

### Test Cases

```python
import numpy as np

points = np.random.rand(100, 2)
naive_result = distance_matrix_naive(points.tolist())
numpy_result = distance_matrix_numpy(points)
numba_result = distance_matrix_numba(points)

# All results should match within tolerance
assert np.allclose(naive_result, numpy_result, rtol=1e-10)
assert np.allclose(numpy_result, numba_result, rtol=1e-10)
```

### Deliverables

- `distance_implementations.py` — Three implementations
- `benchmark_results.csv` — Raw benchmark data
- `optimisation_report.md` — Analysis and profiling results

<details>
<summary>💡 Hint 1: NumPy Broadcasting</summary>

```python
def distance_matrix_numpy(points: np.ndarray) -> np.ndarray:
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))
```
</details>

<details>
<summary>💡 Hint 2: Numba Parallel</summary>

```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def distance_matrix_numba(points: np.ndarray) -> np.ndarray:
    n = len(points)
    result = np.zeros((n, n))
    for i in prange(n):
        for j in range(n):
            diff = points[i] - points[j]
            result[i, j] = np.sqrt(np.sum(diff ** 2))
    return result
```
</details>

---

## Part 4: Practical Problem — Article Search System (20 points)

### Context

You are designing a search system for a research article database with 10 million records. Each article has a title, abstract and list of keywords.

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 4.1 | 8 | Design data structures with complexity analysis |
| 4.2 | 7 | Implement prototype with benchmarks |
| 4.3 | 5 | Document scaling considerations |

### Task 4.1: System Design (8 points)

Design data structures and algorithms for:

1. **Keyword search:** Find all articles containing a specific keyword
2. **Prefix search:** Find all articles with keywords starting with a prefix
3. **Similarity search:** Find articles most similar to a query article

For each operation, specify:
- Data structure used
- Expected time complexity
- Expected space complexity
- Trade-offs made

### Task 4.2: Prototype Implementation (7 points)

Implement a working prototype that:

1. Supports all three search types
2. Uses appropriate data structures (hash tables, tries, etc.)
3. Includes benchmarks demonstrating scalability

Test with a synthetic dataset of 100,000 articles.

### Task 4.3: Scaling Analysis (5 points)

Document:

1. How would your design scale to 10 million articles?
2. What would be the memory requirements?
3. What optimisations would be needed for production?
4. How would distributed computing change your approach?

### Test Cases

```python
# Example usage
search_system = ArticleSearchSystem()
search_system.add_article("Article 1", "Abstract...", ["machine learning", "neural networks"])
search_system.add_article("Article 2", "Abstract...", ["deep learning", "computer vision"])

# Keyword search
results = search_system.search_keyword("machine learning")
assert "Article 1" in results

# Prefix search
results = search_system.search_prefix("mach")
assert "Article 1" in results

# Similarity search
similar = search_system.find_similar("Article 1", k=5)
```

### Deliverables

- `search_system.py` — Implementation
- `design_document.md` — Design rationale and complexity analysis
- `scaling_analysis.md` — Scaling considerations

<details>
<summary>💡 Hint</summary>

Consider using:
- Inverted index (hash table) for keyword search: O(1) lookup
- Trie for prefix search: O(m) where m is prefix length
- TF-IDF with cosine similarity for similarity search
</details>

---

## ✅ Submission Checklist

### Code Quality
- [ ] All tests pass (`pytest`)
- [ ] Code formatted with ruff (`ruff format`)
- [ ] Type hints complete (`mypy --strict`)
- [ ] Docstrings present (Google style)
- [ ] No print statements (use logging)

### Documentation
- [ ] All Markdown reports complete
- [ ] Figures included and referenced
- [ ] Analysis clearly explained
- [ ] British English used throughout

### Repository Structure
```
week3_homework/
├── part1/
│   ├── benchmark_analysis.py
│   ├── analysis_report.md
│   └── hybrid_sort.py
├── part2/
│   ├── complexity_analysis.py
│   └── analysis.md
├── part3/
│   ├── distance_implementations.py
│   ├── benchmark_results.csv
│   └── optimisation_report.md
├── part4/
│   ├── search_system.py
│   ├── design_document.md
│   └── scaling_analysis.md
├── tests/
│   └── test_all.py
└── README.md
```

---

## Grading Rubric

| Component | Points | Criteria |
|-----------|--------|----------|
| Part 1 | 25 | Correct benchmarks, valid crossover analysis, working hybrid sort |
| Part 2 | 25 | Accurate theoretical analysis, matching empirical results |
| Part 3 | 30 | All implementations correct, meaningful speedup analysis, insightful profiling |
| Part 4 | 20 | Sound design, working prototype, realistic scaling analysis |
| **Total** | **100** | |

### Deductions
- Late submission: -10% per day (maximum 3 days)
- Missing type hints: -5%
- Missing docstrings: -5%
- American English spelling: -2%
- Print statements instead of logging: -2%

---

© 2025 Antonio Clim. All rights reserved.
