# ═══════════════════════════════════════════════════════════════════════════════
# 04UNIT Grading Rubric: Advanced Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

> **Total Points**: 100  
> **Passing Grade**: 60%

---

## Overview

This rubric provides detailed grading criteria for all 04UNIT assessments including homework, laboratory work and quizzes.

---

## Homework Assessment (100 points total)

### Part 1: Bloom Filter Implementation (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|--------------|----------|------------------|----------------|------------------|
| **Parameter Calculation** | Correctly implements m and k formulas with proper rounding | Minor calculation errors | Hardcoded values that work | Incorrect formulas | Not implemented |
| **Hash Function** | Double hashing with proper collision handling | Single hash function | Basic modulo hashing | Incorrect hashing | Not implemented |
| **Add/Contains** | O(k) operations, correct behaviour | Minor inefficiencies | Works but inefficient | Partially works | Not implemented |
| **False Positive Analysis** | Empirical verification matches theory | Basic testing | Limited testing | No testing | Not attempted |
| **Spell Checker App** | Fully functional with dictionary | Works with limitations | Basic functionality | Mostly broken | Not attempted |

### Part 2: LRU Cache (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|--------------|----------|------------------|----------------|------------------|
| **Data Structure Choice** | Dict + doubly-linked list | Dict + deque | Dict only | List only | Not implemented |
| **Get Operation** | O(1) time, updates recency | O(1) time, minor issues | O(n) time | Incorrect behaviour | Not implemented |
| **Put Operation** | O(1) time, correct eviction | O(1) time, minor issues | O(n) time | Incorrect eviction | Not implemented |
| **Edge Cases** | All cases handled | Most cases handled | Some cases handled | Few cases handled | Not handled |
| **Code Quality** | Clean, documented, typed | Good quality | Acceptable | Poor quality | Unreadable |

### Part 3: Indexed Priority Queue (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|--------------|----------|------------------|----------------|------------------|
| **Heap Implementation** | Correct min-heap with index tracking | Minor heap issues | Works but inefficient | Partially correct | Not implemented |
| **Insert** | O(log n) with index update | O(log n), minor issues | O(n log n) | Incorrect | Not implemented |
| **Extract Min** | O(log n) with proper reheaping | O(log n), minor issues | O(n) | Incorrect | Not implemented |
| **Decrease Key** | O(log n) with bubble-up | O(n) | O(n log n) | Incorrect | Not implemented |
| **Dijkstra Integration** | Works correctly, no duplicates | Works with minor issues | Works but inefficient | Mostly broken | Not attempted |

### Part 4: City Routing Application (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|--------------|----------|------------------|----------------|------------------|
| **Dataset** | 30+ cities, GPS coords, 100+ edges | 20-30 cities | 10-20 cities | <10 cities | No dataset |
| **Dijkstra Implementation** | Correct, efficient, path tracking | Correct, minor issues | Works for simple cases | Partially works | Not implemented |
| **A* Implementation** | Correct heuristic, fewer nodes visited | Works, not optimised | Basic implementation | Mostly broken | Not implemented |
| **Performance Comparison** | Detailed analysis with statistics | Basic comparison | Some comparison | No comparison | Not attempted |
| **Visualisation** | Interactive map with routes | Static map | Basic plot | No visualisation | Not attempted |

---

## Laboratory Assessment

### Lab 4.1: Graph Library (50 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Graph Interface | 10 | Abstract base class with all required methods |
| Adjacency List | 15 | Correct implementation with proper complexity |
| Traversal Algorithms | 15 | BFS, DFS with correct behaviour |
| Path Algorithms | 10 | Dijkstra or A* working correctly |

### Lab 4.2: Probabilistic Structures (50 points)

| Component | Points | Criteria |
|-----------|--------|----------|
| Bloom Filter | 25 | Correct implementation with configurable parameters |
| Count-Min Sketch | 20 | Correct frequency estimation |
| Documentation | 5 | Clear docstrings and comments |

---

## Code Quality Standards (Applied to All Submissions)

### Type Hints (5% of component grade)

| Level | Description |
|-------|-------------|
| **Excellent** | 100% coverage, proper use of generics |
| **Good** | 90%+ coverage, minor omissions |
| **Satisfactory** | 75%+ coverage |
| **Needs Work** | <75% coverage |

### Documentation (5% of component grade)

| Level | Description |
|-------|-------------|
| **Excellent** | Google-style docstrings, clear examples |
| **Good** | Docstrings present, some examples |
| **Satisfactory** | Basic docstrings |
| **Needs Work** | Missing or unclear documentation |

### Testing (5% of component grade)

| Level | Description |
|-------|-------------|
| **Excellent** | ≥80% coverage, edge cases, parametrised tests |
| **Good** | 60-80% coverage, basic edge cases |
| **Satisfactory** | 40-60% coverage |
| **Needs Work** | <40% coverage or no tests |

### Style (5% of component grade)

| Level | Description |
|-------|-------------|
| **Excellent** | Passes ruff, mypy --strict, consistent naming |
| **Good** | Minor linting issues, consistent style |
| **Satisfactory** | Some style issues |
| **Needs Work** | Significant style problems |

---

## Common Deductions

| Issue | Deduction |
|-------|-----------|
| Print statements instead of logging | -2 points |
| Hardcoded paths | -3 points |
| Missing `if __name__ == "__main__"` guard | -2 points |
| No CLI interface | -5 points |
| American spelling (color, analyze, etc.) | -1 point per instance |
| Oxford comma usage | -1 point per instance |
| Missing licence header | -5 points |

---

## Submission Checklist

Before submitting, verify:

- [ ] All tests pass (`pytest --cov`)
- [ ] Code formatted (`ruff format .`)
- [ ] No linting errors (`ruff check .`)
- [ ] Type hints complete (`mypy --strict`)
- [ ] British English throughout
- [ ] Licence headers present
- [ ] README updated if needed

---

## Appeal Process

If you believe your grade does not accurately reflect your work:

1. Review this rubric carefully
2. Identify specific criteria you believe were misapplied
3. Submit a written appeal within 7 days of grade release
4. Include references to specific code sections

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*04UNIT — Advanced Data Structures*

© 2025 Antonio Clim. All rights reserved.
