# 02UNIT: Learning Objectives

## Abstraction and Encapsulation

### Overview

This document specifies the measurable learning objectives for 02UNIT of "The Art of Computational Thinking for Researchers". These objectives are aligned with Bloom's Taxonomy and build upon the foundational concepts from 01UNIT.

---

## Primary Learning Objectives

### 1. [Understand] Explain OOP Principles and Research Applications

**By the end of this week, learners will be able to:**

- Articulate the four pillars of object-oriented programming (encapsulation, abstraction, inheritance and polymorphism)
- Explain each SOLID principle with concrete examples from scientific computing
- Compare nominal typing (ABC) with structural typing (Protocol) and justify when to use each
- Identify opportunities to apply OOP principles in existing research code

**Assessment Criteria:**
- Correctly explain at least 3 of 5 SOLID principles in written form
- Provide at least 2 research-relevant examples for each principle
- Score ≥70% on the conceptual quiz questions

---

### 2. [Apply] Implement Design Patterns in Simulations

**By the end of this week, learners will be able to:**

- Implement the Strategy pattern for interchangeable algorithms (e.g. numerical integration methods)
- Implement the Observer pattern for event-driven simulation monitoring
- Implement the Factory pattern for flexible object creation in agent-based models
- Use Python Protocols to define type-safe interfaces

**Assessment Criteria:**
- Successfully implement a simulation framework using the Simulable Protocol
- Create at least one working Strategy pattern implementation
- Connect Observer pattern to visualisation or logging components
- All implementations pass mypy --strict type checking

---

### 3. [Analyse] Refactor Procedural Code into OOP Design

**By the end of this week, learners will be able to:**

- Identify code smells that suggest need for refactoring (e.g. long functions, repeated conditionals)
- Determine appropriate patterns for given problem contexts
- Evaluate trade-offs between different design approaches
- Apply systematic refactoring techniques whilst preserving behaviour

**Assessment Criteria:**
- Correctly identify ≥80% of code smells in provided examples
- Successfully refactor procedural code to use at least one design pattern
- Justify design decisions with reference to SOLID principles
- Maintain test coverage during refactoring

---

## Supporting Objectives

### Knowledge Prerequisites (from 01UNIT)

Learners should be able to:
- Define state and state transitions (Turing machine concept)
- Implement basic data structures with type hints
- Write functions with proper documentation

### Skills Developed in This Unit

- Type-safe interface design with Python Protocols
- Generic programming with TypeVar
- Dataclass design for immutable state
- Composition over inheritance patterns

### Preparation for 03UNIT

This week prepares learners for:
- Algorithm analysis and complexity measurement
- Benchmarking framework design
- Performance profiling techniques

---

## Bloom's Taxonomy Alignment

| Level | Verb | Objective |
|-------|------|-----------|
| **Understand** | Explain | OOP principles and their research applications |
| **Apply** | Implement | Strategy, Observer and Factory patterns |
| **Analyse** | Refactor | Procedural code into well-structured OOP |

---

## Assessment Methods

### Formative Assessment
- In-class quizzes during presentation (3 questions)
- Lab completion checkpoints
- Self-assessment checklist

### Summative Assessment
- Homework exercises (4 parts, 100 points total)
- Code review of implementation quality
- Written justification of design decisions

---

## Time Allocation

| Activity | Duration |
|----------|----------|
| Lecture presentation | 90 minutes |
| Laboratory exercises | 120 minutes |
| Homework | 4-6 hours |
| Self-study | 2-3 hours |

---

© 2025 Antonio Clim. All rights reserved.
