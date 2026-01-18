# 09UNIT: Learning Objectives

## Exception Handling and Defensive Programming

---

## Overview

This document specifies the learning objectives for Unit 09, organised by cognitive level and mapped to assessment instruments. Upon successful completion, students will demonstrate competence in Python's exception handling mechanisms, context manager protocols, defensive programming practices and resilience patterns for scientific computing.

---

## Cognitive Objectives

### LO1: Exception Handling Fundamentals

**Level**: Apply

**Objective Statement**: Implement exception handling using try/except/finally/else blocks with appropriate exception types, demonstrating understanding of exception propagation and handler selection.

**Performance Criteria**:
- Correctly structure try/except/finally/else blocks for given error scenarios
- Select appropriate built-in exception types matching semantic error categories
- Implement exception chaining using `raise ... from` syntax
- Apply bare `raise` for exception re-throwing with preserved tracebacks
- Avoid common anti-patterns (bare except, catching BaseException, silent swallowing)

**Evidence of Achievement**:
- Lab 01 §1: Exception mechanism fundamentals exercises
- Lab 01 §4: Exception patterns implementation
- Quiz Q1-3: Multiple choice and short answer on exception syntax
- Homework P1: Exception handling in data processing pipeline

**Key Concepts**:
- Exception hierarchy (BaseException → Exception → specific types)
- Handler matching by type and supertype
- Control flow through try/except/else/finally
- Exception propagation and stack unwinding
- Exception chaining (__cause__ vs __context__)

---

### LO2: Custom Exception Hierarchies

**Level**: Apply

**Objective Statement**: Create custom exception hierarchies that communicate domain-specific error conditions, incorporating meaningful attributes and maintaining appropriate inheritance relationships.

**Performance Criteria**:
- Design exception class hierarchies reflecting domain error categories
- Implement exception classes with informative attributes and messages
- Maintain single inheritance from appropriate base exceptions
- Document exception semantics in docstrings and type hints
- Balance granularity: specific enough for differential handling, not excessively fragmented

**Evidence of Achievement**:
- Lab 01 §2: Custom exception hierarchy implementation
- Quiz Q4-5: Exception hierarchy design questions
- Homework P1: Domain-specific exceptions for research application

**Key Concepts**:
- Inheritance from Exception (not BaseException)
- Attribute design for error context
- Exception naming conventions (*Error suffix)
- Hierarchy depth and breadth trade-offs
- Integration with existing Python exceptions

---

### LO3: Context Managers

**Level**: Apply

**Objective Statement**: Design and implement context managers for resource management using both class-based (__enter__/__exit__) and decorator (@contextmanager) approaches.

**Performance Criteria**:
- Implement class-based context managers with proper protocol methods
- Use @contextmanager decorator for generator-based context managers
- Handle exception information in __exit__ methods appropriately
- Apply ExitStack for dynamic context management
- Ensure deterministic cleanup regardless of exit path

**Evidence of Achievement**:
- Lab 01 §3: Context manager implementation exercises
- Quiz Q6: Context manager protocol questions
- Homework P2: Context manager for research resource management

**Key Concepts**:
- __enter__ and __exit__ protocol methods
- Exception handling in __exit__ (parameters and return value)
- contextlib.contextmanager decorator
- contextlib.ExitStack for multiple contexts
- RAII (Resource Acquisition Is Initialisation) pattern

---

### LO4: Defensive Programming Patterns

**Level**: Evaluate

**Objective Statement**: Evaluate error handling strategies and select appropriate patterns for given computational scenarios, applying design-by-contract principles and resilience patterns.

**Performance Criteria**:
- Apply precondition, postcondition and invariant checking systematically
- Implement input validation with informative error reporting
- Select appropriate resilience patterns (retry, circuit breaker, checkpoint)
- Evaluate trade-offs between fail-fast and graceful degradation
- Design validation frameworks for scientific data pipelines

**Evidence of Achievement**:
- Lab 02 §1-4: Defensive programming implementation
- Quiz Q7-10: Pattern selection and evaluation questions
- Homework P2-3: Resilience pattern application
- Self-check: Reflective evaluation of design decisions

**Key Concepts**:
- Design by contract (preconditions, postconditions, invariants)
- Fail-fast principle and early validation
- Retry with exponential backoff
- Circuit breaker pattern
- Checkpoint-based recovery
- Numerical resilience techniques

---

## Skill Objectives

### Technical Skills

| Skill | Description | Proficiency Level | Evidence |
|-------|-------------|-------------------|----------|
| Exception handling | Use try/except/finally/else blocks effectively | Competent | Lab 01 §1, §4 |
| Custom exceptions | Design domain-specific exception hierarchies | Proficient | Lab 01 §2 |
| Context managers | Implement both protocol methods and @contextmanager | Competent | Lab 01 §3 |
| Input validation | Create comprehensive validation with clear feedback | Proficient | Lab 02 §2 |
| Resilience patterns | Apply retry, circuit breaker and checkpoint patterns | Competent | Lab 02 §4 |
| Numerical resilience | Handle floating-point comparison and accumulation | Competent | Lab 02 §3 |

### Transferable Skills

| Skill | Description | Development Activity |
|-------|-------------|---------------------|
| Analytical reasoning | Decompose failure modes into handling strategies | Lab 02 design exercises |
| Risk assessment | Identify potential failure points in workflows | Homework P3 analysis |
| Technical communication | Document error conditions precisely | All code documentation |
| Critical evaluation | Assess pattern suitability for scenarios | Quiz evaluation questions |

---

## Affective Objectives

### Appreciation for resilience

Students will develop appreciation for the scientific imperative of reliable, failure-resistant computational tools. Through laboratory exercises demonstrating silent failures and cascading errors, students recognise that error handling is not optional overhead but essential infrastructure for trustworthy research software.

### Value of Defensive Programming

Students will value defensive programming as integral to research quality. By experiencing the debugging challenges of poorly-handled errors versus the clarity of fail-fast validation, students internalise that investment in error handling yields dividends in reduced debugging time and increased confidence in results.

### Commitment to Graceful Degradation

Students will develop intellectual commitment to graceful degradation over catastrophic failure. Understanding that research computations often involve long-running processes and irreplaceable data, students appreciate the importance of partial recovery, checkpointing and informative failure modes.

---

## Objectives-Assessment Alignment Matrix

| Learning Objective | Quiz | Lab 01 | Lab 02 | Homework | Self-check |
|--------------------|------|--------|--------|----------|------------|
| **LO1**: Exception handling | Q1-3 | §1, §4 | — | P1 | ✓ |
| **LO2**: Custom exceptions | Q4-5 | §2 | — | P1 | ✓ |
| **LO3**: Context managers | Q6 | §3 | — | P2 | ✓ |
| **LO4**: Defensive patterns | Q7-10 | §5 | §1-4 | P2-3 | ✓ |

---

## Prerequisites and Preparation

### Required Prior Knowledge

| Prerequisite | Source | Specific Concepts |
|--------------|--------|-------------------|
| Function definition | 01-02UNIT | Parameters, return values, scope |
| Class definition | 02UNIT | __init__, methods, inheritance |
| Control flow | 01UNIT | if/elif/else, loops, comprehensions |
| Data structures | 04UNIT | Lists, dicts, dataclasses |
| Type hints | 08UNIT | Basic annotations, Optional, Union |

### Recommended Preparation

1. Review Python class definition and inheritance from 02UNIT
2. Ensure familiarity with the `with` statement for file operations
3. Practise reading stack traces and understanding call chains
4. Review logging module basics

---

## Assessment Weighting

| Component | Weight | Passing Threshold |
|-----------|--------|-------------------|
| Quiz | 30% | 70% (21/30 points) |
| Laboratory Completion | 40% | All sections complete |
| Homework | 20% | 70% on rubric |
| Self-check | 10% | Meaningful reflection |

---

## Success Criteria

Students demonstrate mastery of Unit 09 objectives when they can:

1. **Implement** exception handling that catches appropriate types, preserves context and ensures cleanup
2. **Design** custom exception hierarchies that communicate domain semantics effectively
3. **Create** context managers using both class and decorator approaches with proper resource management
4. **Evaluate** error handling strategies and select appropriate patterns for research computing scenarios
5. **Apply** defensive programming principles including validation, contracts and resilience patterns
6. **Articulate** the rationale for exception handling decisions in code reviews and documentation

---

## Progression to Subsequent Units

Mastery of Unit 09 objectives prepares students for:

| Unit | Dependency | Application |
|------|------------|-------------|
| 10UNIT | LO1, LO3 | File operation error handling, context managers for file I/O |
| 11UNIT | LO1, LO4 | Database connection management, transaction error handling |
| 12UNIT | LO1, LO4 | Network error handling, retry patterns for API calls |
| 13UNIT | LO4 | Parallel computation fault tolerance, distributed error handling |
