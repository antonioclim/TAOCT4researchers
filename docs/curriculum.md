# Curriculum Overview

## The Art of Computational Thinking for Researchers

**Extended Edition — Fourteen Instructional Units**

---

## Introduction

This document provides a structured overview of the complete curriculum, detailing the scope, sequence and interrelationships amongst the fourteen instructional units. The curriculum has been designed to guide researchers from foundational computational concepts through advanced techniques in machine learning and parallel computing.

---

## Curriculum Architecture

### Structural Layers

The fourteen units are organised into five thematic strata:

| Layer | Units | Focus |
|-------|-------|-------|
| **Foundation** | 01 | Theoretical underpinnings of computation |
| **Core Skills** | 02–04 | Essential programming approaches and structures |
| **Application** | 05–06 | Domain-specific computational methods |
| **Integration** | 07 | Synthesis and reproducibility |
| **Intermediate** | 08–09 | Advanced algorithmic techniques |
| **Extended** | 10–12 | Specialised data handling and web integration |
| **Advanced** | 13–14 | Machine learning and high-performance computing |

### Visual Overview

```
                    ┌─────────────────────────────────────┐
                    │          ADVANCED TOPICS            │
                    │   13: Machine Learning              │
                    │   14: Parallel Computing            │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │          EXTENDED SKILLS            │
                    │   10: Data Persistence              │
                    │   11: Text Processing / NLP         │
                    │   12: Web APIs                      │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │        INTERMEDIATE SKILLS          │
                    │   08: Recursion / Dynamic Prog.     │
                    │   09: Exception Handling            │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │           INTEGRATION               │
                    │   07: Reproducibility & Capstone    │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │           APPLICATION               │
                    │   05: Scientific Computing          │
                    │   06: Visualisation                 │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │           CORE SKILLS               │
                    │   02: Abstraction & Encapsulation   │
                    │   03: Algorithmic Complexity        │
                    │   04: Advanced Data Structures      │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │           FOUNDATION                │
                    │   01: Epistemology of Computation   │
                    └─────────────────────────────────────┘
```

---

## Unit Summaries

### 01UNIT: The Epistemology of Computation

**Duration:** 13 hours | **Difficulty:** ★★☆☆☆

Establishes theoretical foundations through examination of Turing machines, lambda calculus and abstract syntax trees. Participants develop understanding of computability, decidability and the Church-Turing thesis.

**Key Topics:**
- Turing machine architecture and state transitions
- Lambda calculus: abstraction, application, beta reduction
- Abstract syntax trees and interpretation
- Decidability and the halting problem

---

### 02UNIT: Abstraction and Encapsulation

**Duration:** 8 hours | **Difficulty:** ★★★☆☆

Addresses software design principles enabling construction of maintainable, extensible research software. Covers SOLID principles and common design patterns.

**Key Topics:**
- SOLID principles applied to research software
- Behavioural patterns: Strategy, Observer, Factory
- Python protocols and structural subtyping
- Plugin architectures for extensibility

---

### 03UNIT: Algorithmic Complexity

**Duration:** 4 hours | **Difficulty:** ★★★☆☆

Introduces formal analysis of algorithm efficiency through Big-O notation and empirical benchmarking techniques.

**Key Topics:**
- Asymptotic notation: O, Ω, Θ
- Common complexity classes
- Empirical timing and profiling
- Space-time trade-offs

---

### 04UNIT: Advanced Data Structures

**Duration:** 10 hours | **Difficulty:** ★★★★☆

Examines graph-theoretic and probabilistic data structures addressing real-world scale requirements.

**Key Topics:**
- Graph representations and traversal algorithms
- Shortest path algorithms: BFS, Dijkstra, A*
- Probabilistic structures: Bloom filters, Count-Min sketch
- Hash tables and collision resolution

---

### 05UNIT: Scientific Computing

**Duration:** 7 hours | **Difficulty:** ★★★★☆

Covers numerical methods fundamental to computational research including Monte Carlo simulation and differential equation solvers.

**Key Topics:**
- Monte Carlo integration and variance reduction
- Ordinary differential equation solvers
- Agent-based modelling fundamentals
- Stochastic simulation techniques

---

### 06UNIT: Visualisation for Research

**Duration:** 14 hours | **Difficulty:** ★★★★☆

Develops proficiency in creating publication-quality figures and interactive visualisations for research communication.

**Key Topics:**
- Grammar of graphics and Tufte's principles
- Matplotlib and Seaborn for static figures
- Plotly for interactive visualisation
- D3.js integration for web-based graphics

---

### 07UNIT: Reproducibility and Capstone

**Duration:** 11 hours | **Difficulty:** ★★★★★

Synthesises prior learning through examination of reproducibility practices and completion of an integrative capstone project.

**Key Topics:**
- Reproducibility spectrum and established methods
- Testing frameworks: pytest, fixtures, mocking
- Continuous integration and deployment
- Research project scaffolding

---

### 08UNIT: Recursion and Dynamic Programming

**Duration:** 10 hours | **Difficulty:** ★★★★☆

Addresses recursive problem decomposition and dynamic programming optimisation techniques for efficient algorithm design.

**Key Topics:**
- Recursive decomposition and base cases
- Memoisation and caching strategies
- Bottom-up tabulation
- Classical problems: knapsack, LCS, edit distance

---

### 09UNIT: Exception Handling and Defensive Code

**Duration:** 8 hours | **Difficulty:** ★★★☆☆

Covers error management and defensive programming practices for building reliable research software.

**Key Topics:**
- Python exception hierarchy
- Custom exception classes
- Context managers for resource management
- Resilience patterns: retry, circuit breaker

---

### 10UNIT: Data Persistence and Serialisation

**Duration:** 10 hours | **Difficulty:** ★★★★☆

Examines mechanisms for durable data storage from file-based formats to relational databases.

**Key Topics:**
- JSON, CSV and binary serialisation
- SQLite and relational database design
- ACID properties and transactions
- Schema versioning and migration

---

### 11UNIT: Text Processing and NLP Fundamentals

**Duration:** 10 hours | **Difficulty:** ★★★★☆

Introduces computational text analysis from string manipulation through natural language processing techniques.

**Key Topics:**
- Regular expressions and pattern matching
- Tokenisation and text normalisation
- Stemming, lemmatisation, stopword removal
- TF-IDF and document vectorisation

---

### 12UNIT: Web APIs and Data Acquisition

**Duration:** 12 hours | **Difficulty:** ★★★★☆

Covers programmatic interaction with web services and ethical data collection practices.

**Key Topics:**
- HTTP protocol and REST architecture
- API authentication and rate limiting
- Ethical web scraping with BeautifulSoup
- Flask for API development

---

### 13UNIT: Machine Learning for Researchers

**Duration:** 14 hours | **Difficulty:** ★★★★★

Introduces machine learning methodology with emphasis on research applications and rigorous evaluation.

**Key Topics:**
- Supervised vs unsupervised learning
- Model evaluation: cross-validation, metrics
- scikit-learn pipelines and preprocessing
- Clustering and dimensionality reduction

---

### 14UNIT: Parallel Computing and Scalability

**Duration:** 12 hours | **Difficulty:** ★★★★★

Addresses computational acceleration through concurrent and parallel execution strategies.

**Key Topics:**
- Parallelism vs concurrency
- Python's GIL and multiprocessing
- Dask for distributed computing
- Performance profiling and optimisation

---

## Prerequisite Dependencies

| Unit | Hard Prerequisites | Soft Prerequisites |
|------|-------------------|-------------------|
| 01 | — | Basic Python |
| 02 | 01 | — |
| 03 | 01, 02 | — |
| 04 | 01, 02, 03 | — |
| 05 | 01–04 | — |
| 06 | 01–05 | — |
| 07 | 01–06 | — |
| 08 | 01–04, 07 | — |
| 09 | 01–02 | — |
| 10 | 01–03, 09 | — |
| 11 | 01–04, 10 | — |
| 12 | 01–04, 09–10 | — |
| 13 | 01–06, 08 | — |
| 14 | 01–03, 08 | — |

---

## Learning Pathways

### Standard Pathway (Full Curriculum)

Complete all fourteen units in sequence:

**01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14**

Estimated duration: 143 hours

### Data Science Focus

For researchers primarily interested in data analysis:

**01 → 02 → 03 → 04 → 05 → 06 → 10 → 11 → 13**

Estimated duration: 96 hours

### Software Engineering Focus

For researchers developing research software:

**01 → 02 → 03 → 04 → 07 → 08 → 09 → 12**

Estimated duration: 74 hours

### High-Performance Computing Focus

For researchers with intensive computational needs:

**01 → 02 → 03 → 04 → 05 → 08 → 14**

Estimated duration: 64 hours

---

## Assessment Strategy

Each unit employs multiple assessment modalities:

| Modality | Purpose | Presence |
|----------|---------|----------|
| Quiz | Knowledge verification | All units |
| Laboratory | Skill demonstration | All units |
| Exercises | Deliberate practice | All units |
| Self-assessment | Metacognitive reflection | All units |
| Rubric | Evaluation criteria | All units |

---

## Contact

For curriculum enquiries, contact the author through official academic channels or via the repository issue tracker.

---

*Last updated: January 2026*
