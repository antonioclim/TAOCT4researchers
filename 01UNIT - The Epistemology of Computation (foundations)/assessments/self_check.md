# Week 1: Self-Assessment Checklist

## The Epistemology of Computation

Use this checklist to verify your understanding before moving to Week 2.

---

## Core Concepts

### Turing Machines

- [ ] I can list all seven components of a Turing machine from memory
- [ ] I can explain what each component represents
- [ ] I can trace the execution of a simple Turing machine by hand
- [ ] I can design a Turing machine for a basic string manipulation task
- [ ] I understand why the tape must be infinite in one or both directions

### Lambda Calculus

- [ ] I can identify variables, abstractions and applications in lambda expressions
- [ ] I can perform beta reduction step by step
- [ ] I understand what "free" and "bound" variables mean
- [ ] I can write Church encodings for small numbers (0, 1, 2, 3)
- [ ] I can explain the Church-Turing thesis in my own words

### Computability Theory

- [ ] I can state the Halting Problem informally
- [ ] I understand why the Halting Problem is undecidable
- [ ] I can give an example of an undecidable problem
- [ ] I understand the practical implications of uncomputability

### Interpreters and ASTs

- [ ] I can explain the role of a lexer (tokeniser)
- [ ] I can explain the role of a parser
- [ ] I can draw an AST for a simple arithmetic expression
- [ ] I understand how operator precedence affects AST structure
- [ ] I can trace the evaluation of an expression through its AST

---

## Practical Skills

### Programming

- [ ] I can run the lab files and understand their output
- [ ] I can modify the Turing machine simulator to add new machines
- [ ] I can implement Church encodings in Python
- [ ] I can use the mini-interpreter's REPL
- [ ] I can write type hints for my Python functions

### Problem Solving

- [ ] I can design a state diagram for a Turing machine
- [ ] I can identify the base case and recursive case in lambda expressions
- [ ] I can debug a Turing machine by tracing its execution
- [ ] I can convert between mathematical notation and code

---

## Connections to Research

- [ ] I can identify at least one finite state machine in my research domain
- [ ] I can explain how parsing relates to data processing
- [ ] I understand how computational limits affect my field
- [ ] I have thought about whether a DSL could help my research

---

## Self-Assessment Questions

Answer these questions to verify your understanding:

### Question 1
What would happen if we tried to build a Turing machine with only a finite tape?

<details>
<summary>Check your answer</summary>

With only finite tape, the machine would be limited to recognising only regular languages (equivalent to finite automata). It could not solve problems requiring unbounded memory, such as checking whether a string has an equal number of 0s and 1s. The infinite tape is essential for Turing completeness.

</details>

### Question 2
Why does `(λx.x x) (λx.x x)` not have a normal form?

<details>
<summary>Check your answer</summary>

This is the omega combinator Ω. When we beta-reduce:
- (λx.x x) (λx.x x) → (λx.x x) (λx.x x)

We get back the same expression, so reduction loops forever. This demonstrates that not all lambda expressions terminate.

</details>

### Question 3
What is the difference between a syntax error and a runtime error in the context of our interpreter?

<details>
<summary>Check your answer</summary>

A syntax error occurs during parsing when the input does not follow the grammar rules (e.g., "let x =" is incomplete). A runtime error occurs during evaluation when the AST is valid but evaluation fails (e.g., dividing by zero, referencing an undefined variable). The lexer/parser catches syntax errors; the evaluator catches runtime errors.

</details>

### Question 4
Why do we need closures when implementing functions?

<details>
<summary>Check your answer</summary>

Closures capture the environment (variable bindings) at the point where a function is defined. Without closures, a function like `let x = 5 in (fun y -> x + y)` would fail because when we later call the function, `x` would no longer be in scope. The closure remembers that `x = 5` from definition time.

</details>

---

## Readiness Check

Count your checked boxes:

| Checked | Readiness |
|---------|-----------|
| 25-28 | Excellent — Ready for Week 2 |
| 20-24 | Good — Review a few concepts |
| 15-19 | Fair — Spend more time on labs |
| <15 | Needs work — Review lecture notes |

---

## Areas for Review

If you struggled with any section, revisit these resources:

| Topic | Resource |
|-------|----------|
| Turing machines | `theory/lecture_notes.md`, Lab 1 |
| Lambda calculus | Lab 2, `resources/further_reading.md` |
| AST and interpreters | Lab 3, REPL practice |
| Computability | `theory/lecture_notes.md` Section 5 |

---

## Preparing for Week 2

Week 2 covers Design Patterns. The concepts from Week 1 that will be most relevant:

1. **State concept** → Used in the State design pattern
2. **AST hierarchies** → Example of the Composite pattern
3. **Transition functions** → Related to the Strategy pattern
4. **Evaluation with environments** → Context patterns

Make sure you are comfortable with:
- How state machines transition between states
- How tree structures (like ASTs) are traversed recursively
- How different behaviours can be encapsulated and swapped

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
