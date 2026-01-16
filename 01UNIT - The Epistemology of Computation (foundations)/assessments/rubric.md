# Week 1: Grading Rubric

## The Epistemology of Computation

This rubric provides detailed criteria for assessing Week 1 submissions.

---

## Homework Rubric (100 points total)

### Exercise 1: Binary Increment Turing Machine (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Correctness** (×3) | All test cases pass | 6-7 test cases pass | 4-5 test cases pass | 2-3 test cases pass | 0-1 test cases pass |
| **State Design** | Minimal states, clear naming | Good design, minor redundancy | Functional but verbose | Confusing state names | Missing or broken |
| **Code Quality** | Type hints, docstrings, clean | Minor issues | Some documentation | Minimal documentation | No documentation |

**Total: 15 + 5 + 5 = 25 points**

---

### Exercise 2: Balanced Parentheses Checker (25 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Correctness** (×3) | All test cases pass | 9-11 test cases pass | 6-8 test cases pass | 3-5 test cases pass | 0-2 test cases pass |
| **Nested Handling** | Handles all nesting levels | Minor edge case issues | Works for simple nesting | Fails on deep nesting | Does not handle nesting |
| **Efficiency** | Minimal states, clear logic | Reasonable complexity | Somewhat verbose | Very complex | Broken logic |

**Total: 15 + 5 + 5 = 25 points**

---

### Exercise 3: Lambda Calculus Reduction (20 points)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|---------------|----------|------------------|----------------|------------------|
| **Part A** (5) | Correct, all steps shown | Correct, minor gaps | Correct answer, incomplete steps | Partially correct | Missing or wrong |
| **Part B** (8) | Full reduction, clear | Minor errors | Mostly correct | Major errors | Missing or wrong |
| **Part C** (7) | Full reduction, clear | Minor errors | Mostly correct | Major errors | Missing or wrong |

**Total: 5 + 8 + 7 = 20 points**

---

### Exercise 4: Research Connection Essay (30 points)

| Criterion | Excellent (6-5) | Good (4) | Satisfactory (3) | Needs Work (2) | Incomplete (0-1) |
|-----------|-----------------|----------|------------------|----------------|------------------|
| **Domain Clarity** (5) | Clear, specific domain | Well-defined | Adequate | Vague | Missing |
| **Connection Depth** (10) | Deep, insightful links to theory | Good connections | Adequate links | Superficial | No connection |
| **Specificity** (8) | Concrete examples, applications | Good examples | Some examples | Generic | No examples |
| **Writing Quality** (5) | Clear, well-organised | Minor issues | Adequate | Poor organisation | Unreadable |
| **Citations** (2) | Proper citations where needed | Minor issues | Incomplete | Missing needed citations | N/A |

**Total: 5 + 10 + 8 + 5 + 2 = 30 points**

---

## Lab Completion Rubric

### Lab 1: Turing Machine Simulator

| Criterion | Complete | Partial | Incomplete |
|-----------|----------|---------|------------|
| Ran demonstrations | ✓ | - | - |
| Traced execution | ✓ | ✓ | - |
| Modified code | ✓ | - | - |

### Lab 2: Lambda Calculus Basics

| Criterion | Complete | Partial | Incomplete |
|-----------|----------|---------|------------|
| Ran demonstrations | ✓ | - | - |
| Performed reductions | ✓ | ✓ | - |
| Implemented Church encodings | ✓ | - | - |

### Lab 3: AST Interpreter

| Criterion | Complete | Partial | Incomplete |
|-----------|----------|---------|------------|
| Ran demonstrations | ✓ | - | - |
| Used REPL | ✓ | ✓ | - |
| Understood AST structure | ✓ | ✓ | - |

---

## Quiz Rubric (50 points)

### Multiple Choice (30 points)

- 5 points per correct answer
- No partial credit

### Short Answer (20 points)

| Criterion | Full (5) | Partial (3) | Minimal (1) | Missing (0) |
|-----------|----------|-------------|-------------|-------------|
| Correct content | Accurate, complete | Mostly accurate | Partially correct | Incorrect/missing |
| Clarity | Clear explanation | Understandable | Confusing | N/A |
| Examples (if asked) | Appropriate | Adequate | Weak | Missing |

---

## Code Quality Standards

All submitted code should meet these standards:

### Type Hints (Required)

```python
# Good
def evaluate(expr: Expr, env: Environment | None = None) -> Value:
    ...

# Bad
def evaluate(expr, env=None):
    ...
```

### Docstrings (Required)

```python
# Good
def create_machine() -> TuringMachine:
    """Create a Turing machine for binary increment.
    
    Returns:
        A TuringMachine that increments binary numbers.
    """

# Bad
def create_machine():
    # creates machine
```

### Naming Conventions

- States: `q_` prefix (e.g., `q_scan`, `q_return`)
- Variables: snake_case
- Classes: PascalCase
- Constants: UPPER_SNAKE_CASE

---

## Common Deductions

| Issue | Deduction |
|-------|-----------|
| Missing type hints | -2 per function |
| Missing docstrings | -2 per function |
| Print statements instead of logging | -1 per occurrence |
| American spelling instead of British | -1 per file |
| Code does not run | -50% of exercise |
| Plagiarism detected | -100% + academic integrity review |

---

## Late Submission Policy

| Days Late | Penalty |
|-----------|---------|
| 1 day | -10% |
| 2 days | -20% |
| 3 days | -30% |
| >3 days | Not accepted |

Extensions must be requested before the deadline.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
