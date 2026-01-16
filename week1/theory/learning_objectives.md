# Week 1: Learning Objectives

> **The Epistemology of Computation**  
> *Foundations of computation — What does it mean to compute?*

---

## Overview

This document specifies the measurable learning objectives for Week 1, aligned with Bloom's Taxonomy. Each objective includes assessment criteria and evidence of achievement.

---

## Primary Objectives

### Objective 1: Define Computability and Turing Machine Components

**Bloom Level:** Remember  
**Verb:** Define, Enumerate, List

**Statement:**  
By the end of this week, learners will be able to define computability and enumerate the seven components of a Turing machine from memory.

**Assessment Criteria:**
- Correctly state the informal definition of computability
- List all seven components: Q (states), Σ (input alphabet), Γ (tape alphabet), δ (transition function), q₀ (initial state), q_accept (accept state), q_reject (reject state)
- Explain the role of each component in one sentence

**Evidence of Achievement:**
- Quiz questions 1-3
- Lab 1 completion (Turing machine implementation)

---

### Objective 2: Explain Computational Equivalences

**Bloom Level:** Understand  
**Verb:** Explain, Compare, Contrast

**Statement:**  
By the end of this week, learners will be able to explain the relationship between Turing machines, lambda calculus and modern programming languages, including the Church-Turing thesis.

**Assessment Criteria:**
- Articulate the Church-Turing thesis in own words
- Compare Turing machines (imperative) with lambda calculus (functional)
- Explain how modern interpreters relate to these foundational models
- Identify at least two implications for research computing

**Evidence of Achievement:**
- Quiz questions 4-6
- Homework Essay component
- Discussion participation

---

### Objective 3: Implement Computational Models

**Bloom Level:** Apply  
**Verb:** Implement, Construct, Execute

**Statement:**  
By the end of this week, learners will be able to implement a working Turing machine simulator and a minimal abstract syntax tree (AST) interpreter in Python.

**Assessment Criteria:**
- Turing machine simulator accepts valid machine definitions
- Simulator correctly executes unary addition
- AST interpreter parses and evaluates arithmetic expressions
- Code follows type hints and Google-style docstrings
- All provided tests pass

**Evidence of Achievement:**
- Lab 1: Turing machine simulator (500+ lines)
- Lab 3: AST interpreter (300+ lines)
- Homework Exercise 1-2

---

## Secondary Objectives

### Objective 4: Trace Execution Steps

**Bloom Level:** Apply  
**Verb:** Trace, Demonstrate, Show

**Statement:**  
Learners will be able to manually trace the execution of a Turing machine on a given input, recording each configuration.

**Assessment Criteria:**
- Correctly record tape contents at each step
- Track head position accurately
- Identify halting conditions
- Determine accept/reject outcome

**Evidence of Achievement:**
- Practice exercises (Easy 1-3)
- Quiz question 7

---

### Objective 5: Recognise Computational Limits

**Bloom Level:** Understand  
**Verb:** Recognise, Identify, Describe

**Statement:**  
Learners will be able to recognise problems that are undecidable and describe why the Halting Problem cannot be solved algorithmically.

**Assessment Criteria:**
- State the Halting Problem precisely
- Explain the diagonalisation argument informally
- Identify at least one real-world implication

**Evidence of Achievement:**
- Quiz questions 8-9
- Lecture participation

---

### Objective 6: Connect Theory to Practice

**Bloom Level:** Understand  
**Verb:** Connect, Relate, Map

**Statement:**  
Learners will be able to connect foundational computation theory to their own research domain, identifying at least one application of formal language theory or state machines.

**Assessment Criteria:**
- Identify a relevant application in learner's research field
- Explain how the theoretical concept applies
- Propose a potential computational approach

**Evidence of Achievement:**
- Homework reflective essay
- Self-assessment checklist

---

## Objective Mapping

| Objective | Bloom Level | Primary Assessment | Weight |
|-----------|-------------|-------------------|--------|
| 1. Define computability | Remember | Quiz, Lab 1 | 15% |
| 2. Explain equivalences | Understand | Quiz, Essay | 20% |
| 3. Implement models | Apply | Labs 1 & 3 | 35% |
| 4. Trace execution | Apply | Practice exercises | 10% |
| 5. Recognise limits | Understand | Quiz | 10% |
| 6. Connect to research | Understand | Essay | 10% |

---

## Prerequisites Verification

Before starting Week 1, verify learners can:

- [ ] Write Python functions with parameters and return values
- [ ] Use Python dictionaries and lists
- [ ] Define simple classes with `__init__` methods
- [ ] Run Python scripts from the command line
- [ ] Navigate directories using terminal commands

---

## Preparation for Week 2

Week 1 objectives prepare learners for Week 2 by establishing:

1. **State concept** → Used in State design pattern
2. **AST hierarchies** → Foundation for understanding OOP hierarchies
3. **Transition functions** → Leads to Strategy pattern
4. **Formal definitions** → Enables rigorous analysis of abstractions

---

## Self-Assessment Questions

Use these questions to verify objective achievement:

1. Can I write the formal definition of a Turing machine without reference materials?
2. Can I explain to a colleague why Turing machines and lambda calculus compute the same things?
3. Could I implement a Turing machine for a new problem (e.g., palindrome detection)?
4. Can I trace through a program's AST and predict its output?
5. Can I explain why some problems have no algorithmic solution?
6. Have I identified one way computation theory applies to my research?

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
