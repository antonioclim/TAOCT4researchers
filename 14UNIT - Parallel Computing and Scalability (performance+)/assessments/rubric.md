# 14UNIT: Assessment Rubric

## Parallel Computing and Scalability

---

## Overview

This rubric provides grading criteria for all assessable components of Unit 14. Each criterion specifies expectations for different performance levels.

---

## Laboratory Assessments

### Lab 01: Multiprocessing and Threading (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Process Creation** | Correctly creates processes with proper argument passing; handles lifecycle (start, join) correctly | Minor issues with argument passing or cleanup | Basic process creation works but with some errors | Processes created but with significant issues | Does not demonstrate process creation |
| **Pool Patterns** | Uses Pool.map, starmap, apply_async appropriately; correct worker count selection | Most pool operations correct; minor inefficiencies | Basic pool usage works | Pool usage has significant issues | Cannot use Pool effectively |
| **Monte Carlo** | Parallel implementation correct; proper seed handling; achieves expected speedup | Implementation correct; minor seed or speedup issues | Basic parallelisation works but suboptimal | Significant issues with parallelisation | Monte Carlo not parallelised |
| **Synchronisation** | Correct Lock usage; prevents race conditions; demonstrates understanding | Lock usage mostly correct; minor issues | Basic synchronisation works | Race conditions still present | No synchronisation implemented |
| **Threading I/O** | Correctly implements threaded I/O; demonstrates GIL understanding | Threading works; minor understanding gaps | Basic threading works | Threading implemented but with issues | Threading not implemented |

### Lab 02: Dask and Profiling (20 points)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) | Incomplete (0) |
|-----------|---------------|----------|------------------|----------------|----------------|
| **Dask Delayed** | Correctly builds task graphs; uses delayed appropriately | Task graphs work; minor issues | Basic delayed usage | Significant issues | Not implemented |
| **Dask Arrays/DFs** | Appropriate chunking; correct operations | Most operations correct | Basic usage works | Operations have issues | Not implemented |
| **Out-of-Core** | Handles larger-than-memory data correctly | Works with minor issues | Basic functionality | Significant issues | Not implemented |
| **Profiling** | Uses cProfile effectively; identifies bottlenecks; generates clear reports | Profiling works; minor gaps | Basic profiling | Profiling incomplete | Not implemented |
| **Optimisation** | Applies profiling insights; achieves measurable improvement | Some improvement achieved | Attempts optimisation | Little improvement | No optimisation attempted |

---

## Exercise Assessment

### Difficulty-Weighted Scoring

| Difficulty | Points per Exercise | Total Exercises | Maximum Points |
|------------|---------------------|-----------------|----------------|
| Easy | 5 | 3 | 15 |
| Medium | 10 | 3 | 30 |
| Hard | 15 | 3 | 45 |
| **Total** | — | **9** | **90** |

### Exercise Grading Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Correctness | 40% | Code produces correct results for all test cases |
| Code Quality | 20% | Type hints, docstrings, PEP 8 compliance |
| Efficiency | 20% | Appropriate algorithm choice; achieves expected speedup |
| Understanding | 20% | Comments demonstrate understanding; handles edge cases |

---

## Homework Assessment (100 points)

### Part 1: Theoretical Analysis (30 points)

| Criterion | Points | Expectations |
|-----------|--------|--------------|
| Amdahl's Law application | 10 | Correctly calculates speedup limits for given scenarios |
| Gustafson's Law comparison | 10 | Accurately contrasts scaled speedup perspective |
| Practical implications | 10 | Discusses real-world factors affecting theoretical predictions |

### Part 2: Implementation (50 points)

| Criterion | Points | Expectations |
|-----------|--------|--------------|
| Correctness | 20 | Pipeline produces correct results |
| Parallelisation | 15 | Appropriate use of multiprocessing/Dask |
| Error handling | 10 | Graceful handling of failures |
| Code quality | 5 | Clean, documented, type-hinted code |

### Part 3: Profiling Report (20 points)

| Criterion | Points | Expectations |
|-----------|--------|--------------|
| Profiling methodology | 5 | Clear description of profiling approach |
| Bottleneck identification | 5 | Correctly identifies performance issues |
| Optimisation evidence | 5 | Before/after measurements provided |
| Written analysis | 5 | Clear explanation of findings |

### Bonus: Distributed Computing (+10 points)

| Criterion | Points | Expectations |
|-----------|--------|--------------|
| Dask distributed setup | 5 | Correctly configures distributed scheduler |
| Scaling demonstration | 5 | Shows computation scaling across workers |

---

## Quiz Assessment (20 points)

See quiz.md for detailed question-level rubric.

| Section | Points | Criteria |
|---------|--------|----------|
| Multiple Choice (Q1-Q6) | 6 | 1 point each; correct answer required |
| Short Answer (Q7-Q10) | 14 | Partial credit available; see answer key |

---

## Code Quality Standards

All submitted code must meet these minimum standards:

### Required

- [ ] Runs without errors on Python 3.12+
- [ ] Includes type hints for all function signatures
- [ ] Includes docstrings for all public functions
- [ ] Passes provided test cases

### Expected

- [ ] Follows PEP 8 style guidelines
- [ ] Uses meaningful variable and function names
- [ ] Handles edge cases appropriately
- [ ] Includes inline comments for complex logic

### Exemplary

- [ ] Includes comprehensive error handling
- [ ] Provides performance measurements
- [ ] Demonstrates optimisation reasoning
- [ ] Exceeds minimum requirements creatively

---

## Grade Boundaries

| Grade | Percentage | Description |
|-------|------------|-------------|
| A | 90-100% | Exceptional understanding; exceeds expectations |
| B | 80-89% | Strong understanding; meets all requirements |
| C | 70-79% | Adequate understanding; meets most requirements |
| D | 60-69% | Basic understanding; meets minimum requirements |
| F | <60% | Insufficient understanding; does not meet requirements |

---

## Late Submission Policy

- 10% penalty per day for first 3 days
- 50% maximum penalty
- No submissions accepted after 7 days without prior arrangement

---

## Academic Integrity

All submissions must be original work. Code sharing, plagiarism, and use of unauthorised AI assistance will result in zero marks for the affected assessment and potential disciplinary action.

---

*14UNIT — Parallel Computing and Scalability*
*Assessment Rubric v4.0.0*
