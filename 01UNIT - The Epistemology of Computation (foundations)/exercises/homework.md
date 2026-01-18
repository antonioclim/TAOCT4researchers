# Week 1: Homework Assignment

## The Epistemology of Computation

**Course:** The Art of Computational Thinking for Researchers  
**Week:** 1 of 7  
**Total Points:** 100  
**Estimated Time:** 3 hours  

---

## Overview

This homework consolidates your understanding of foundational computation theory through practical implementation and reflective analysis. You will implement Turing machine extensions, trace lambda calculus reductions and connect these theoretical concepts to your research domain.

### Conceptual Foundation

The exercises in this assignment traverse three interconnected pillars of computability theory. The first pillar—Turing machine implementation—requires you to translate abstract state-transition specifications into executable code, thereby experiencing firsthand the precision demanded by formal computational models. The second pillar—lambda calculus reduction—develops your capacity for systematic symbol manipulation according to rigorous rewrite rules, a skill transferable to any formal system encountered in research. The third pillar—domain connection—demands synthesis: identifying correspondences between these theoretical constructs and computational phenomena in your field of study.

Each exercise targets specific learning outcomes mapped to the cognitive taxonomy employed throughout this course. Exercises 1 and 2 emphasise application: taking theoretical definitions and instantiating them in working code. Exercise 3 requires analysis: tracing reduction sequences and verifying equivalences. Exercise 4 calls for synthesis and evaluation: constructing arguments that bridge theory and practice whilst assessing the relevance of formal models to contemporary research.

### Assessment Philosophy

The grading rubric rewards both correctness and clarity. Correct implementations that lack documentation receive partial credit; conversely, well-documented attempts that fail edge cases demonstrate valuable learning even when incomplete. The research connection essay receives substantial weight because it measures your ability to transfer knowledge—the ultimate goal of any educational endeavour.

---

## Submission Requirements

- Submit all Python files with `.py` extension
- Include your written responses in a single Markdown file (`responses.md`)
- Ensure all code passes the provided test cases
- Follow Google-style docstrings and type hints throughout
- Use British English spelling in all documentation

---

## Preparation Checklist

Before commencing this assignment, verify that you have completed the following preparatory steps. Each item corresponds to prerequisite knowledge necessary for successful completion.

**Theoretical Prerequisites:**
- [ ] Reviewed the formal definition of a Turing machine (M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject))
- [ ] Understood the distinction between input alphabet Σ and tape alphabet Γ
- [ ] Traced at least three Turing machine executions manually
- [ ] Studied the syntax of lambda calculus expressions
- [ ] Practised beta reduction on simple applications
- [ ] Examined Church encodings for booleans and natural numbers

**Technical Prerequisites:**
- [ ] Installed Python 3.12 or later
- [ ] Verified pytest is available (`pytest --version`)
- [ ] Cloned or downloaded the laboratory code repository
- [ ] Successfully executed `make test` in the lab directory
- [ ] Reviewed the TuringMachine class interface in `lab_01_01_turing_machine.py`

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

## Common Pitfalls to Avoid

Experience from previous cohorts reveals several recurring difficulties. First, students often confuse the tape alphabet with the input alphabet; remember that Γ must include the blank symbol and any auxiliary markers your machine requires. Second, state naming conventions matter for debugging—use descriptive names like `q_carry` rather than `q3`. Third, when tracing lambda reductions, apply substitutions meticulously; a single misplaced parenthesis invalidates the entire derivation. Fourth, the research connection essay should identify specific computational phenomena, not vague analogies; stating that "my field uses algorithms" earns no credit without concrete elaboration of which algorithms and how they relate to the theoretical models studied. Fifth, ensure your test cases cover boundary conditions: empty input, single-character input, and maximum-length input where applicable.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
