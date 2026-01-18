# 02UNIT Self-Assessment: Abstraction and Encapsulation

## 📋 Instructions

Complete this self-assessment **before** submitting your homework. Rate your confidence honestly — this helps identify areas for review and does not affect your grade.

Use the following scale:
- **5** = I can explain this to others and apply it in novel situations
- **4** = I understand this well and can apply it independently
- **3** = I understand the basics but need practice
- **2** = I have a vague understanding; need to review
- **1** = I don't understand this; need help

---

## Learning Objective 1: OOP Principles

*"Explain core OOP principles and their research applications"*

### Concepts

| Concept | Self-Rating (1-5) | Notes |
|---------|------------------|-------|
| Abstraction (hiding complexity) | | |
| Encapsulation (bundling data + behaviour) | | |
| Single Responsibility Principle | | |
| Open/Closed Principle | | |
| Liskov Substitution Principle | | |
| Interface Segregation Principle | | |
| Dependency Inversion Principle | | |

### Self-Check Questions

1. Can you explain why abstraction is important for managing complexity in large codebases?
   - [ ] Yes, confidently
   - [ ] Somewhat
   - [ ] Need to review

2. Can you identify when encapsulation has been violated in existing code?
   - [ ] Yes, confidently
   - [ ] Somewhat
   - [ ] Need to review

3. Can you give a research example where SOLID principles improve code maintainability?
   - [ ] Yes, confidently
   - [ ] Somewhat
   - [ ] Need to review

---

## Learning Objective 2: Design Patterns

*"Implement Strategy, Observer and Factory patterns in simulations"*

### Patterns

| Pattern | Can Identify Use Case (1-5) | Can Implement (1-5) | Notes |
|---------|---------------------------|-------------------|-------|
| Strategy | | | |
| Observer | | | |
| Factory | | | |
| Decorator | | | |
| Command | | | |

### Self-Check Questions

1. Given a requirement, can you choose the appropriate design pattern?
   - [ ] Yes, confidently
   - [ ] Somewhat — I recognise patterns but sometimes choose incorrectly
   - [ ] Need more practice

2. Can you implement the Strategy pattern from scratch?
   - [ ] Yes, without reference
   - [ ] Yes, with documentation
   - [ ] Need to review

3. Can you explain when Observer is preferable to direct method calls?
   - [ ] Yes, confidently
   - [ ] Somewhat
   - [ ] Need to review

---

## Learning Objective 3: Refactoring

*"Refactor procedural code into well-structured OOP design"*

### Skills

| Skill | Self-Rating (1-5) | Notes |
|-------|------------------|-------|
| Identifying code smells | | |
| Extracting classes from functions | | |
| Introducing interfaces/protocols | | |
| Replacing conditionals with polymorphism | | |
| Composition over inheritance decisions | | |

### Self-Check Questions

1. Can you recognise when a function is doing too much and should be split?
   - [ ] Yes, confidently
   - [ ] Sometimes
   - [ ] Rarely

2. Can you refactor a large `if-elif-else` chain into a Strategy pattern?
   - [ ] Yes, confidently
   - [ ] With guidance
   - [ ] Need to learn

---

## Python-Specific Skills

| Skill | Self-Rating (1-5) | Notes |
|-------|------------------|-------|
| Using `Protocol` for structural typing | | |
| Using `ABC` for nominal typing | | |
| Creating generic classes with `TypeVar` | | |
| Writing comprehensive type hints | | |
| Using `@dataclass` effectively | | |
| Understanding `frozen=True` immutability | | |

---

## Practical Exercises Completed

Check off exercises you've completed and understood:

### Easy
- [ ] easy_01_protocol.py — Incrementable Protocol
- [ ] easy_02_dataclass.py — Point2D with distance
- [ ] easy_03_generics.py — Generic Box container

### Medium
- [ ] medium_01_strategy.py — Sorting strategies
- [ ] medium_02_observer.py — Temperature sensor
- [ ] medium_03_factory.py — Shape factory

### Hard
- [ ] hard_01_di_container.py — Dependency injection
- [ ] hard_02_state_machine.py — Type-safe FSM
- [ ] hard_03_event_sourcing.py — Event sourcing

---

## Reflection Questions

Answer these briefly (2-3 sentences each):

### 1. What was the most challenging concept this week?

*Your answer:*

---

### 2. What would help you understand that concept better?

*Your answer:*

---

### 3. How might you apply this week's concepts in your research?

*Your answer:*

---

### 4. What questions do you still have?

*Your answer:*

---

## Action Items

Based on your self-assessment, list 3 concrete actions to improve:

1. _________________________________________________

2. _________________________________________________

3. _________________________________________________

---

## Recommended Review

If your ratings were below 3 in any area, consider reviewing:

| Topic | Recommended Resource |
|-------|---------------------|
| SOLID Principles | Lecture notes Section 2; Martin, R. "Clean Code" Ch. 10 |
| Design Patterns | Lab 2.02; Gamma et al. "Design Patterns" (selected chapters) |
| Python Protocols | Python docs: typing.Protocol; Lab 2.01 |
| Generics | Python docs: typing module; PEP 484 |
| Refactoring | Fowler, M. "Refactoring" Ch. 1-3 |

---

## Submission Readiness

Before submitting homework, confirm:

- [ ] Completed all required exercises
- [ ] All tests pass locally
- [ ] Code passes `ruff check` and `mypy --strict`
- [ ] Self-assessment completed
- [ ] Questions noted for office hours

---

*This self-assessment is for your own learning. Honest reflection helps identify gaps before they become problems.*

---

© 2025 Antonio Clim. All rights reserved.
