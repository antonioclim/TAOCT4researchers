# Week 1: Homework Assignment

## The Epistemology of Computation

**Course:** The Art of Computational Thinking for Researchers  
**Week:** 1 of 7  
**Total Points:** 100  
**Estimated Time:** 3 hours  

---

## Overview

This homework consolidates your understanding of foundational computation theory through practical implementation and reflective analysis. You will implement Turing machine extensions, trace lambda calculus reductions and connect these theoretical concepts to your research domain.

---

## Submission Requirements

- Submit all Python files with `.py` extension
- Include your written responses in a single Markdown file (`responses.md`)
- Ensure all code passes the provided test cases
- Follow Google-style docstrings and type hints throughout
- Use British English spelling in all documentation

---

## Exercise 1: Binary Increment Turing Machine (25 points)

### Task

Complete the `create_binary_increment_machine()` function in `lab_1_01_turing_machine.py` to implement a Turing machine that increments a binary number.

### Requirements

| Criterion | Points |
|-----------|--------|
| Correct output for all test cases | 15 |
| Proper state naming conventions | 3 |
| Clean transition logic | 4 |
| Code documentation | 3 |

### Test Cases

```python
test_cases = [
    ("0", "1"),       # 0 → 1
    ("1", "10"),      # 1 → 2
    ("10", "11"),     # 2 → 3
    ("11", "100"),    # 3 → 4
    ("111", "1000"),  # 7 → 8
    ("1011", "1100"), # 11 → 12
    ("1111", "10000"), # 15 → 16
]
```

<details>
<summary>💡 Hint 1: Algorithm Overview</summary>

The algorithm works from right to left:
1. Move to the rightmost digit
2. If the digit is '0', change it to '1' and halt
3. If the digit is '1', change it to '0' and continue left (carry)
4. If you reach a blank with a carry, write '1'

</details>

<details>
<summary>💡 Hint 2: State Design</summary>

You need approximately three states:
- `q_seek_end`: Move right to find the end of the number
- `q_increment`: Process digits from right to left
- `accept`: Final accepting state

</details>

---

## Exercise 2: Balanced Parentheses Checker (25 points)

### Task

Complete the `create_balanced_parentheses_machine()` function to verify whether a string of parentheses is balanced.

### Requirements

| Criterion | Points |
|-----------|--------|
| Correct accept/reject for all test cases | 15 |
| Handles nested parentheses | 5 |
| Efficient state design | 3 |
| Code documentation | 2 |

### Test Cases

```python
test_cases = [
    ("", True),           # Empty string is balanced
    ("()", True),         # Single pair
    ("(())", True),       # Nested
    ("()()", True),       # Sequential
    ("((()))", True),     # Deeply nested
    ("(()())", True),     # Mixed
    ("(", False),         # Unmatched open
    (")", False),         # Unmatched close
    ("(()", False),       # Missing close
    ("())", False),       # Extra close
    (")(", False),        # Wrong order
    ("(()))(", False),    # Complex invalid
]
```

<details>
<summary>💡 Hint 1: Matching Strategy</summary>

One approach:
1. Find the rightmost '('
2. Find the first ')' to its right
3. Mark both as matched (e.g., replace with 'X')
4. Repeat until no more pairs or mismatch found
5. Accept if all characters are 'X'

</details>

<details>
<summary>💡 Hint 2: State Machine Design</summary>

Consider these states:
- `q_scan`: Scan for unmatched parentheses
- `q_find_close`: Found '(', looking for matching ')'
- `q_mark_open`: Going back to mark the '('
- `q_check_done`: Verify all are matched

</details>

---

## Exercise 3: Lambda Calculus Reduction (20 points)

### Task

Manually trace the beta reduction of the following lambda expressions. Show each step clearly.

### Part A: Simple Application (5 points)

Reduce: `(λx.λy.x y) a b`

Show each step of the reduction.

### Part B: Church Arithmetic (8 points)

Using Church numerals where:
- `2 = λf.λx.f (f x)`
- `3 = λf.λx.f (f (f x))`
- `SUCC = λn.λf.λx.f (n f x)`

Reduce: `SUCC 2`

Show that the result equals `3`.

### Part C: Boolean Logic (7 points)

Using Church booleans where:
- `TRUE = λx.λy.x`
- `FALSE = λx.λy.y`
- `NOT = λp.p FALSE TRUE`

Reduce: `NOT TRUE`

Show that the result equals `FALSE`.

---

## Exercise 4: Research Connection Essay (30 points)

### Task

Write a 500-800 word essay connecting computation theory to your research domain.

### Requirements

| Criterion | Points |
|-----------|--------|
| Clear identification of research domain | 5 |
| Meaningful connection to Turing machines or lambda calculus | 10 |
| Specific example or application | 8 |
| Quality of writing and argumentation | 5 |
| Proper citations (if applicable) | 2 |

### Guiding Questions

Address at least two of the following:

1. **Finite State Machines in Your Field**: Are there processes in your research that can be modelled as state machines? Consider data processing pipelines, experimental protocols or decision procedures.

2. **Computability Limits**: Are there problems in your domain that might be undecidable? How do researchers work around computational limits?

3. **Domain-Specific Languages**: Could a custom language or notation improve how you express computations in your field? What operations would it need?

4. **Lambda Calculus and Functional Approaches**: Are there transformations in your research that could be expressed as pure functions? How might functional programming principles apply?

### Format

- Use Markdown formatting
- Include section headings
- Cite any referenced papers or resources
- Save as `essay.md`

---

## Grading Rubric Summary

| Exercise | Points | Key Criteria |
|----------|--------|--------------|
| Binary Increment TM | 25 | Correctness, state design, documentation |
| Balanced Parentheses TM | 25 | Correctness, nested handling, efficiency |
| Lambda Reduction | 20 | Step accuracy, completeness, clarity |
| Research Essay | 30 | Connection depth, specificity, writing quality |
| **Total** | **100** | |

---

## Submission Checklist

- [ ] `lab_1_01_turing_machine.py` with completed exercises
- [ ] `responses.md` with lambda calculus reductions
- [ ] `essay.md` with research connection essay
- [ ] All code passes provided test cases
- [ ] All files use British English spelling
- [ ] Type hints on all functions
- [ ] Google-style docstrings on all functions

---

## Academic Integrity

This is an individual assignment. You may:
- Discuss concepts with classmates
- Use course materials and documentation
- Consult textbooks and academic papers

You may not:
- Share code with other students
- Copy solutions from the internet
- Use AI assistants to generate solutions

All submitted work must be your own.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
