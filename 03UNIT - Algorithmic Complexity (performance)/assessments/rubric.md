# Week 3 Assessment Rubric: Algorithmic Complexity

> **Course:** The Art of Computational Thinking for Researchers  
> **Week:** 3 of 7  
> **Total Points:** 100

---

## Overview

This rubric provides detailed grading criteria for all Week 3 assessments:
- Homework (40 points)
- Quiz (20 points)
- Laboratory Work (30 points)
- Participation (10 points)

---

## Homework Assessment (40 points)

### Part 1: Sorting Algorithm Analysis (10 points)

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|--------------------| -----------------|
| **Experimental Design** | Thorough size range, multiple distributions tested, proper warmup | Good size range, random data tested, warmup included | Limited sizes, single distribution, basic setup | Insufficient testing, no warmup |
| **Statistical Rigour** | Multiple runs, outlier-resistant statistics (median/IQR), error bars | Multiple runs, mean/std reported | Limited runs, basic statistics | Single measurements, no statistics |
| **Analysis Quality** | Crossover points identified, hybrid sort justified, theoretical comparison | Crossover identified, reasonable hybrid design | Basic comparison, simple hybrid | Missing comparison or hybrid |

### Part 2: Complexity Analysis (10 points)

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|--------------------| -----------------|
| **Theoretical Analysis** | Correct complexity for all 5 functions with clear reasoning | 4/5 correct with reasoning | 3/5 correct | Fewer than 3 correct |
| **Empirical Validation** | Log-log regression, R² > 0.95, clear methodology | Log-log used, R² > 0.9 | Basic empirical analysis | No empirical validation |
| **Comparison** | Insightful discussion of theory vs. practice discrepancies | Notes discrepancies present | Basic comparison | No comparison |

### Part 3: Optimisation Challenge (15 points)

| Criterion | Excellent (13-15) | Good (10-12) | Satisfactory (7-9) | Needs Work (0-6) |
|-----------|-------------------|--------------|--------------------| -----------------|
| **Algorithmic Version** | Correct O(n log n) or better approach, well-documented | Correct complexity, documented | Improvement achieved, basic docs | No improvement or broken |
| **NumPy Version** | Efficient vectorisation, 10x+ speedup | Good vectorisation, 5x+ speedup | Some vectorisation, 2x+ speedup | No vectorisation |
| **Numba Version** | Proper JIT decorators, near-C performance | JIT works, significant speedup | JIT attempted, some speedup | Numba not working |
| **Documentation** | Clear explanations, complexity annotations, benchmarks | Good explanations, benchmarks | Basic documentation | Missing documentation |

### Part 4: Practical Problem (5 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (0-2) |
|-----------|---------------|----------|------------------| -----------------|
| **Design Quality** | Thorough design with multiple indexing strategies, scalability analysis | Good design with primary index, scaling considered | Basic design, limited scaling | Incomplete design |

---

## Quiz Assessment (20 points)

### Multiple Choice (12 points)
- 2 points per correct answer
- No partial credit

### Short Answer (8 points)

| Criterion | Full Credit | Partial Credit | No Credit |
|-----------|-------------|----------------|-----------|
| **Complexity Analysis** (Q7) | Correct final answer with clear sum/product rule application | Correct approach, minor errors | Incorrect approach |
| **Master Theorem** (Q8) | Correct case identification, parameter values, final answer | Correct case, minor calculation error | Wrong case or major errors |
| **Benchmarking Concepts** (Q9) | Three valid sources, clear explanations | Two valid sources | Fewer than two sources |
| **Empirical Analysis** (Q10) | Correct exponent, class, prediction with working | Two of three parts correct | One or none correct |

---

## Laboratory Work (30 points)

### Lab 3.1: Benchmark Suite (15 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **BenchmarkResult** | 3 | All properties implemented, correct statistics |
| **timer() context manager** | 2 | Uses perf_counter, correct elapsed calculation |
| **benchmark() function** | 4 | Warmup, GC control, multiple runs, statistics |
| **BenchmarkSuite class** | 4 | All methods working, CSV export, summary generation |
| **Code Quality** | 2 | Type hints, docstrings, logging (no print) |

### Lab 3.2: Complexity Analyser (15 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| **estimate_exponent()** | 4 | Log-log regression, R² calculation, correct exponent |
| **fit_models()** | 4 | Multiple models fitted, correct residual calculation |
| **best_fit()** | 3 | Correct model selection based on R² |
| **ComplexityAnalyser class** | 2 | Integration of all components, summary generation |
| **Code Quality** | 2 | Type hints, docstrings, error handling |

### Code Quality Rubric (applies to both labs)

| Aspect | Excellent | Good | Satisfactory | Needs Work |
|--------|-----------|------|--------------|------------|
| **Type Hints** | 100% coverage | >90% coverage | >75% coverage | <75% coverage |
| **Docstrings** | Google style, all public functions | Most functions documented | Some documentation | Missing documentation |
| **Logging** | Appropriate use, no print statements | Mostly logging, few prints | Some logging | Extensive print statements |
| **Error Handling** | Thorough validation | Good error handling | Basic error handling | No error handling |

---

## Participation (10 points)

| Criterion | Excellent (9-10) | Good (7-8) | Satisfactory (5-6) | Needs Work (0-4) |
|-----------|------------------|------------|--------------------| -----------------|
| **Engagement** | Active discussion, asks insightful questions | Regular participation, good questions | Occasional participation | Minimal engagement |
| **Collaboration** | Helps peers, shares resources | Some peer assistance | Works independently | Disruptive or absent |
| **Preparation** | Completes pre-reading, comes prepared | Usually prepared | Sometimes prepared | Rarely prepared |

---

## Learning Objectives Assessment Matrix

| Objective | Quiz Questions | Homework Parts | Lab Components |
|-----------|----------------|----------------|----------------|
| LO1: Big-O Notation | Q1-3 | Part 2 | Lab 3.2: models |
| LO2: Benchmarking | Q9 | Parts 1, 3 | Lab 3.1 |
| LO3: Empirical Analysis | Q4, Q10 | Parts 1, 2 | Lab 3.2 |

---

## Grade Boundaries

| Grade | Percentage | Points |
|-------|------------|--------|
| A (Excellent) | 90-100% | 90-100 |
| B (Good) | 80-89% | 80-89 |
| C (Satisfactory) | 70-79% | 70-79 |
| D (Pass) | 60-69% | 60-69 |
| F (Fail) | <60% | <60 |

---

## Common Deductions

| Issue | Deduction |
|-------|-----------|
| Late submission (per day) | -10% |
| Missing type hints | -1 per function |
| Print statements instead of logging | -2 |
| Missing docstrings | -1 per function |
| Plagiarism | -100% + disciplinary action |
| Code does not run | -50% |
| Missing test cases | -5 |

---

## Academic Integrity

All submitted work must be original. Collaboration is encouraged for learning, but final submissions must be individual work. Use of AI assistants must be disclosed and code must be understood and explained upon request.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 3 — Algorithmic Complexity*  
*© 2025 Antonio Clim. All rights reserved.*
