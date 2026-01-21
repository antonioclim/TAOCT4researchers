# Week 3 Self-Assessment: Algorithmic Complexity

> **Course:** The Art of Computational Thinking for Researchers  
> **Week:** 3 of 7

---

## Instructions

Complete this self-assessment after finishing all Week 3 materials. Rate your confidence on each item using the scale:

- **1** = Cannot do / No understanding
- **2** = Need significant help / Basic understanding
- **3** = Can do with reference / Good understanding
- **4** = Can do independently / Strong understanding
- **5** = Can teach others / Expert understanding

---

## Learning Objective 1: Big-O Notation and Complexity Classes

### Conceptual Understanding

| Skill | Rating (1-5) | Notes |
|-------|--------------|-------|
| I can state the formal definition of Big-O notation | ☐ | |
| I can explain the difference between O, Ω and Θ notation | ☐ | |
| I can list complexity classes in order of growth rate | ☐ | |
| I can explain why constant factors are ignored in Big-O | ☐ | |
| I can describe what amortised analysis means | ☐ | |

### Practical Application

| Skill | Rating (1-5) | Notes |
|-------|--------------|-------|
| I can determine the complexity of a simple loop | ☐ | |
| I can determine the complexity of nested loops | ☐ | |
| I can apply the sum rule for sequential operations | ☐ | |
| I can apply the product rule for nested operations | ☐ | |
| I can recognise O(log n) patterns (halving) | ☐ | |

### Self-Check Questions

1. What is the complexity of this code?
   ```python
   for i in range(n):
       for j in range(i, n):
           process(i, j)
   ```
   <details>
   <summary>Check Answer</summary>
   O(n²) — The inner loop runs n + (n-1) + (n-2) + ... + 1 = n(n+1)/2 times total.
   </details>

2. Is 3n² + 100n + 5000 = O(n²)? Why?
   <details>
   <summary>Check Answer</summary>
   Yes. We can find c and n₀ such that 3n² + 100n + 5000 ≤ cn² for all n ≥ n₀. For example, c = 6 and n₀ = 100 works.
   </details>

---

## Learning Objective 2: Benchmarking with Statistical Analysis

### Conceptual Understanding

| Skill | Rating (1-5) | Notes |
|-------|--------------|-------|
| I can explain why multiple measurements are needed | ☐ | |
| I can describe the purpose of warmup runs | ☐ | |
| I can explain when to use median vs mean | ☐ | |
| I can interpret coefficient of variation (CV) | ☐ | |
| I can explain how garbage collection affects timing | ☐ | |

### Practical Application

| Skill | Rating (1-5) | Notes |
|-------|--------------|-------|
| I can use time.perf_counter() correctly | ☐ | |
| I can implement a benchmark function with warmup | ☐ | |
| I can calculate and report outlier-resistant statistics | ☐ | |
| I can use cProfile to identify bottlenecks | ☐ | |
| I can generate meaningful benchmark comparisons | ☐ | |

### Self-Check Questions

1. Why do we disable garbage collection during benchmarks?
   <details>
   <summary>Check Answer</summary>
   Garbage collection can pause execution unpredictably, introducing measurement variability. Disabling it during measurements ensures consistent timing, though we must re-enable it afterwards.
   </details>

2. When is median preferable to mean for reporting benchmark results?
   <details>
   <summary>Check Answer</summary>
   Median is preferable when outliers are present (e.g., occasional GC pauses, OS interrupts). It gives a more representative "typical" value. Use mean when the distribution is symmetric and outlier-free.
   </details>

---

## Learning Objective 3: Empirical Complexity Estimation

### Conceptual Understanding

| Skill | Rating (1-5) | Notes |
|-------|--------------|-------|
| I can explain the log-log technique for complexity estimation | ☐ | |
| I can interpret R² (coefficient of determination) | ☐ | |
| I can explain why theory and practice may differ | ☐ | |
| I can describe the memory hierarchy's effect on performance | ☐ | |
| I can explain how to distinguish O(n) from O(n log n) | ☐ | |

### Practical Application

| Skill | Rating (1-5) | Notes |
|-------|--------------|-------|
| I can perform log-log regression to estimate exponent | ☐ | |
| I can fit multiple complexity models and compare them | ☐ | |
| I can generate diagnostic plots for complexity analysis | ☐ | |
| I can identify discrepancies between theory and measurement | ☐ | |
| I can use curve fitting for non-power-law models | ☐ | |

### Self-Check Questions

1. You measure T(1000) = 10ms and T(10000) = 1000ms. What is the approximate complexity?
   <details>
   <summary>Check Answer</summary>
   Size increased by 10×, time increased by 100×. In log-log: slope = log(100)/log(10) = 2. This suggests O(n²).
   </details>

2. Your log-log regression gives exponent 1.35 with R² = 0.98. What complexity class?
   <details>
   <summary>Check Answer</summary>
   Exponent between 1.0 and 1.5 suggests O(n log n). The high R² indicates good fit to the power-law model.
   </details>

---

## Practical Skills Checklist

### Lab 3.1: Benchmark Suite

- [ ] I implemented BenchmarkResult with all required properties
- [ ] I created a working timer() context manager
- [ ] My benchmark() function includes warmup and GC control
- [ ] I implemented BenchmarkSuite with run() and summary()
- [ ] I can export results to CSV format
- [ ] All my code uses logging instead of print statements
- [ ] All functions have type hints and docstrings

### Lab 3.2: Complexity Analyser

- [ ] I implemented estimate_exponent() with log-log regression
- [ ] I can fit multiple complexity models
- [ ] I implemented best_fit() selection logic
- [ ] I can generate complexity analysis summaries
- [ ] I can create diagnostic visualisations (if matplotlib available)
- [ ] All my code uses proper type hints

---

## Reflection Questions

### What went well this week?
```
[Write your reflection here]



```

### What was challenging?
```
[Write your reflection here]



```

### What questions do I still have?
```
[Write your reflection here]



```

### How will I apply this knowledge in my research?
```
[Write your reflection here]



```

---

## Preparation for Week 4

Before starting Week 4 (Advanced Data Structures), ensure you can:

- [ ] Analyse complexity of basic operations on arrays and linked lists
- [ ] Explain time/space trade-offs in algorithm design
- [ ] Implement and benchmark simple data structures
- [ ] Use profiling to identify performance bottlenecks

### Preview Questions

1. What is the complexity of inserting at the beginning of an array vs. a linked list?
2. Why might a theoretically slower algorithm be faster in practice?
3. What is a hash table and why is its lookup O(1) on average?

---

## Score Summary

| Section | Max Points | Your Score |
|---------|------------|------------|
| LO1: Big-O (10 items × 5) | 50 | |
| LO2: Benchmarking (10 items × 5) | 50 | |
| LO3: Empirical Analysis (9 items × 5) | 45 | |
| Lab 3.1 Checklist (7 items × 5) | 35 | |
| Lab 3.2 Checklist (6 items × 5) | 30 | |
| **Total** | **210** | |

### Interpretation

- **180-210**: Excellent mastery — ready for advanced topics
- **150-179**: Good understanding — review weak areas
- **120-149**: Satisfactory — additional practice recommended
- **<120**: Needs improvement — seek assistance before continuing

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 3 — Algorithmic Complexity*  
*© 2025 Antonio Clim. All rights reserved.*
