# Week 2 Grading Rubric: Abstraction and Encapsulation

## 📋 Overview

This rubric provides detailed criteria for evaluating Week 2 homework submissions. Each criterion uses a 4-level scale aligned with Bloom's taxonomy expectations.

---

## Homework Grading Criteria

### Part 1: Units Library (40 points)

| Criterion | Excellent (100%) | Good (75%) | Satisfactory (50%) | Needs Improvement (25%) |
|-----------|------------------|------------|--------------------|-----------------------|
| **Type Safety (15 pts)** | Prevents all unit mismatches at type-check time; mypy --strict passes with no errors | Most unit mismatches caught; minor type issues | Basic type hints present; some errors not caught | Missing type hints or significant type errors |
| **Correctness (10 pts)** | All conversions mathematically correct; handles edge cases (zero, negative) | Conversions correct for common cases; minor edge case issues | Basic conversions work; some numerical errors | Significant calculation errors |
| **API Design (10 pts)** | Intuitive interface; follows Python conventions; excellent docstrings | Good interface; minor inconsistencies; adequate documentation | Functional but awkward interface; sparse documentation | Confusing interface; missing documentation |
| **Testing (5 pts)** | Comprehensive tests; edge cases covered; parametrised tests | Good test coverage; some edge cases | Basic tests present | Minimal or no tests |

### Part 2: Result Monad (25 points)

| Criterion | Excellent (100%) | Good (75%) | Satisfactory (50%) | Needs Improvement (25%) |
|-----------|------------------|------------|--------------------|-----------------------|
| **Monad Laws (10 pts)** | and_then/or_else follow monad laws; compose correctly | Mostly correct behaviour; minor law violations | Basic chaining works; inconsistent behaviour | Does not follow monadic patterns |
| **Error Handling (8 pts)** | Graceful error propagation; no exceptions leak; informative error messages | Good error handling; minor issues with edge cases | Basic error handling; some exception leakage | Poor error handling; crashes on errors |
| **Generic Types (7 pts)** | Fully generic Result[T, E]; type inference works correctly | Generic but with some type constraints | Partially generic; hardcoded types in places | Not generic; fixed types only |

### Part 3: Type-Safe Builder (20 points)

| Criterion | Excellent (100%) | Good (75%) | Satisfactory (50%) | Needs Improvement (25%) |
|-----------|------------------|------------|--------------------|-----------------------|
| **Compile-Time Safety (10 pts)** | Phantom types prevent calling build() before required fields set | Most build errors caught at type-check | Runtime validation of required fields | No validation of required fields |
| **Fluent Interface (5 pts)** | Clean method chaining; returns correct type at each step | Chaining works; minor type issues | Basic chaining; inconsistent returns | No chaining or broken interface |
| **Completeness (5 pts)** | All HTTP request fields supported; extensible design | Most fields supported; reasonably extensible | Basic fields only; hard to extend | Missing critical fields |

### Part 4: Mini Simulation Framework (15 points)

| Criterion | Excellent (100%) | Good (75%) | Satisfactory (50%) | Needs Improvement (25%) |
|-----------|------------------|------------|--------------------|-----------------------|
| **Protocol Compliance (8 pts)** | Correctly implements Simulable protocol; generic runner works | Protocol implemented; minor compliance issues | Basic implementation; some methods missing | Does not follow protocol |
| **Domain Modelling (7 pts)** | Realistic simulation; appropriate state representation; physical accuracy | Good model; minor accuracy issues | Basic model; oversimplified physics | Unrealistic or incorrect model |

---

## Code Quality Criteria (Applied to All Parts)

| Criterion | Excellent | Good | Satisfactory | Needs Improvement |
|-----------|-----------|------|--------------|-------------------|
| **Type Hints** | 100% coverage; complex types correctly annotated | >90% coverage; minor annotation issues | >70% coverage; some missing hints | <70% coverage |
| **Docstrings** | Google-style; all public APIs documented; examples included | Most APIs documented; consistent style | Some documentation; inconsistent style | Missing or poor documentation |
| **Code Style** | ruff passes; consistent naming; logical organisation | Minor style issues; mostly consistent | Some style violations; organisation could improve | Significant style issues |
| **Testing** | pytest coverage ≥80%; parametrised tests; edge cases | Coverage ≥60%; good test variety | Coverage ≥40%; basic tests | Coverage <40% or no tests |

---

## Submission Checklist

Before submitting, verify:

- [ ] All tests pass (`pytest`)
- [ ] Code formatted (`ruff format .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy --strict .`)
- [ ] All docstrings present
- [ ] README updated if applicable

---

## Late Submission Policy

| Days Late | Penalty |
|-----------|---------|
| 1 day | -10% |
| 2 days | -20% |
| 3 days | -30% |
| >3 days | Not accepted without prior arrangement |

---

## Academic Integrity

All submitted work must be your own. You may:
- ✓ Discuss concepts with peers
- ✓ Use documentation and official tutorials
- ✓ Reference Stack Overflow with attribution

You may not:
- ✗ Copy code from other students
- ✗ Use AI to generate solutions without understanding
- ✗ Submit work from previous course offerings

Violations will be reported according to university policy.

---

## Feedback Timeline

- **Initial feedback**: Within 7 days of deadline
- **Detailed rubric scores**: Via learning management system
- **Office hours**: Available for discussion of grades

---

© 2025 Antonio Clim. All rights reserved.
