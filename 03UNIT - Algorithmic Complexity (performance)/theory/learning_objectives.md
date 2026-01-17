# Week 3: Algorithmic Complexity — Learning Objectives

> **Course:** The Art of Computational Thinking for Researchers  
> **Week:** 3 of 7  
> **Bloom Level:** Apply/Analyse

---

## Primary Learning Objectives

By the end of this week, you will be able to:

### 1. [Understand] Explain Big-O Notation and Complexity Classes

**Description:** Define asymptotic notation (Big-O, Big-Ω, Big-Θ) and explain what each represents mathematically. Describe the common complexity classes and their relative growth rates.

**Evidence of Achievement:**
- Correctly define Big-O notation using the formal mathematical definition
- Explain the difference between O, Ω and Θ notation with examples
- Rank complexity classes from O(1) to O(n!) in order of growth rate
- Identify which complexity class describes a given algorithm

**Assessment Methods:**
- Quiz questions 1-4
- Homework Part 2 (complexity identification)
- Lab 3.1 complexity annotations

---

### 2. [Apply] Implement Benchmarking Framework with Statistical Analysis

**Description:** Design and implement a benchmarking framework that measures algorithm performance with proper statistical rigour, including multiple runs, warmup periods and outlier-resistant statistics.

**Evidence of Achievement:**
- Create timing functions using `time.perf_counter()` with proper measurement methodology
- Implement warmup runs to account for JIT compilation and cache effects
- Calculate appropriate statistical measures (median, IQR, coefficient of variation)
- Generate comparison plots with error bars showing measurement uncertainty
- Export benchmark results in machine-readable formats

**Assessment Methods:**
- Lab 3.1 completion (benchmark suite implementation)
- Homework Part 1 (sorting algorithm comparison)
- Practice exercises (medium difficulty)

---

### 3. [Analyse] Estimate Complexity of Algorithms Empirically and Theoretically

**Description:** Apply both theoretical analysis techniques (counting operations, recurrence relations) and empirical methods (log-log regression, curve fitting) to determine the complexity of algorithms.

**Evidence of Achievement:**
- Analyse code to determine theoretical complexity using sum and product rules
- Solve simple recurrence relations using the Master Theorem
- Use log-log plotting to estimate empirical complexity exponents
- Compare multiple complexity models to identify the best fit
- Explain discrepancies between theoretical and empirical results

**Assessment Methods:**
- Lab 3.2 completion (complexity analyser)
- Homework Part 2 (mystery function analysis)
- Homework Part 3 (optimisation challenge)
- Quiz questions 5-10

---

## Supporting Objectives

### 4. [Remember] Recall Standard Algorithm Complexities

- List the time complexity of common sorting algorithms
- State the complexity of basic data structure operations
- Recall the Master Theorem cases

### 5. [Understand] Explain Amortised Analysis Concepts

- Describe why dynamic array append is O(1) amortised
- Explain the accounting method for amortised analysis
- Identify operations suitable for amortised analysis

### 6. [Apply] Use Profiling Tools Effectively

- Profile Python code using cProfile
- Interpret profiler output to identify bottlenecks
- Apply line_profiler for detailed analysis

---

## Prerequisite Knowledge

Before beginning this week, ensure you can:

- [ ] **From Week 2:** Implement classes with proper encapsulation
- [ ] **From Week 2:** Apply design patterns (Strategy, Observer) appropriately
- [ ] **Python:** Write functions with type hints and docstrings
- [ ] **Mathematics:** Understand logarithms and basic algebra

---

## Connection to Course Progression

### Building on Previous Weeks

| Week | Concept | Application in Week 3 |
|------|---------|----------------------|
| 1 | Turing machine states | State space defines problem size |
| 1 | Interpreter implementation | Runtime analysis of evaluation |
| 2 | Abstract data types | Complexity of ADT operations |
| 2 | Strategy pattern | Swappable sorting algorithms |

### Preparing for Future Weeks

| Week | Topic | Prerequisite from Week 3 |
|------|-------|-------------------------|
| 4 | Graph algorithms | Complexity comparison for algorithm selection |
| 4 | Probabilistic structures | Understanding space/time trade-offs |
| 5 | Monte Carlo methods | Convergence rate analysis |
| 5 | ODE solvers | Comparing solver efficiency |

---

## Self-Assessment Checklist

Rate your confidence (1-5) on each objective after completing the week:

| Objective | Before | After |
|-----------|--------|-------|
| Define Big-O notation formally | ☐ | ☐ |
| List complexity classes in order | ☐ | ☐ |
| Implement proper benchmarking | ☐ | ☐ |
| Calculate outlier-resistant statistics | ☐ | ☐ |
| Use log-log regression | ☐ | ☐ |
| Apply Master Theorem | ☐ | ☐ |
| Profile Python code | ☐ | ☐ |

**Target:** Achieve confidence level 4+ on all primary objectives.

---

## Resources for Each Objective

### Objective 1 (Big-O Notation)
- **Slides:** Sections 4-13
- **Lecture Notes:** Section 2
- **Cheatsheet:** Complexity table
- **Further Reading:** Cormen et al., Chapter 3

### Objective 2 (Benchmarking)
- **Slides:** Sections 19-24
- **Lab 3.1:** Complete implementation
- **Lecture Notes:** Section 4
- **Further Reading:** Gorelick & Ozsvald, Chapters 2-3

### Objective 3 (Empirical Analysis)
- **Slides:** Sections 25-28
- **Lab 3.2:** Complete implementation
- **Lecture Notes:** Section 5
- **Practice Exercises:** Hard difficulty

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 3 — Algorithmic Complexity*  
*© 2025 Antonio Clim. All rights reserved.*
