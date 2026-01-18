# 14UNIT: Learning Objectives

## Parallel Computing and Scalability

---

## Overview

This document specifies the learning objectives for Unit 14, mapping each objective to assessment components and providing measurable criteria for evaluation.

---

## Cognitive Learning Objectives

### LO1: Understand Parallelism Fundamentals

**Level**: Understand (Bloom's Taxonomy Level 2)

**Objective Statement**: Explain the distinction between parallelism and concurrency, processes and threads, and the implications of Python's Global Interpreter Lock for computational performance.

**Measurable Criteria**:
- Define parallelism and concurrency with correct technical precision
- Distinguish process-based from thread-based execution models
- Explain how the GIL constrains Python threading for CPU-bound work
- Predict when threading versus multiprocessing provides performance benefit

**Assessment Alignment**:
| Assessment | Items | Weight |
|------------|-------|--------|
| Quiz | Q1, Q2, Q3 | 15% |
| Homework | Part 1 | 10% |
| Self-Check | Items 1-4 | — |

---

### LO2: Apply Multiprocessing Techniques

**Level**: Apply (Bloom's Taxonomy Level 3)

**Objective Statement**: Implement multiprocessing solutions for CPU-bound tasks using Process, Pool and shared memory primitives with appropriate error handling.

**Measurable Criteria**:
- Create Process instances with correct argument passing
- Use Pool.map, Pool.starmap and Pool.apply_async appropriately
- Implement inter-process communication via Queue
- Handle process lifecycle (start, join, terminate) correctly

**Assessment Alignment**:
| Assessment | Items | Weight |
|------------|-------|--------|
| Lab 01 | Sections 1-4 | 25% |
| Exercises | easy_01, easy_02 | 10% |
| Homework | Part 2 | 15% |

---

### LO3: Apply Threading for I/O

**Level**: Apply (Bloom's Taxonomy Level 3)

**Objective Statement**: Implement threading and asyncio patterns for I/O-bound workloads with appropriate synchronisation primitives to prevent race conditions.

**Measurable Criteria**:
- Identify I/O-bound versus CPU-bound workloads
- Implement thread pools using ThreadPoolExecutor
- Apply Lock, Event and other synchronisation primitives
- Recognise and prevent common concurrency bugs (deadlock, race conditions)

**Assessment Alignment**:
| Assessment | Items | Weight |
|------------|-------|--------|
| Lab 01 | Section 5 | 10% |
| Quiz | Q4, Q5, Q6 | 15% |
| Exercise | easy_03, medium_02 | 10% |

---

### LO4: Apply concurrent.futures Patterns

**Level**: Apply (Bloom's Taxonomy Level 3)

**Objective Statement**: Use concurrent.futures for high-level parallel execution patterns with proper Future handling and error management.

**Measurable Criteria**:
- Create and configure ProcessPoolExecutor and ThreadPoolExecutor
- Submit tasks and handle Future objects
- Use as_completed for progressive result processing
- Implement timeout handling and cancellation

**Assessment Alignment**:
| Assessment | Items | Weight |
|------------|-------|--------|
| Lab 01 | Section 4 | 10% |
| Exercise | medium_03 | 5% |
| Homework | Part 2 | 10% |

---

### LO5: Scale with Dask

**Level**: Apply (Bloom's Taxonomy Level 3)

**Objective Statement**: Scale computations using Dask for datasets exceeding available memory, applying appropriate chunking strategies and scheduler selection.

**Measurable Criteria**:
- Construct Dask delayed computations and visualise task graphs
- Create and manipulate Dask arrays with appropriate chunking
- Process Dask DataFrames with groupby and aggregation operations
- Select appropriate schedulers for different workload types

**Assessment Alignment**:
| Assessment | Items | Weight |
|------------|-------|--------|
| Lab 02 | Sections 1-3 | 20% |
| Exercise | medium_03, hard_01 | 10% |
| Homework | Part 2, Bonus | 15% |

---

### LO6: Analyse with Profiling

**Level**: Analyse (Bloom's Taxonomy Level 4)

**Objective Statement**: Profile and optimise code using cProfile, line_profiler and memory_profiler to identify performance bottlenecks and validate optimisation efforts.

**Measurable Criteria**:
- Execute profiling tools and interpret output
- Identify hot spots consuming disproportionate resources
- Measure memory usage patterns
- Compare before/after performance with statistical rigour

**Assessment Alignment**:
| Assessment | Items | Weight |
|------------|-------|--------|
| Lab 02 | Section 4 | 10% |
| Quiz | Q7, Q8, Q9, Q10 | 20% |
| Exercise | hard_02 | 5% |

---

## Skill Objectives

### Technical Skills

1. **Process Management**: Spawn, monitor and terminate processes programmatically
2. **Parallel Mapping**: Distribute work across multiple workers efficiently
3. **Synchronisation**: Protect shared state with appropriate primitives
4. **Memory Profiling**: Measure and optimise memory consumption
5. **Out-of-Core Processing**: Handle datasets larger than available RAM

### Transferable Skills

1. **Performance Analysis**: Systematic approach to identifying bottlenecks
2. **Trade-off Evaluation**: Balance parallelisation benefits against overhead
3. **Scalability Planning**: Design systems that scale with available resources
4. **Debugging Concurrent Code**: Diagnose non-deterministic bugs

---

## Affective Objectives

Participants will develop:

1. **Appreciation** for the complexity hidden beneath simple parallel abstractions
2. **Scepticism** toward claims of linear speedup without empirical validation
3. **Patience** with debugging challenges inherent in concurrent code
4. **Professional Judgement** in selecting appropriate parallelisation strategies

---

## Objectives-Assessment Matrix

```
┌─────────┬────────────────────────────────────────────────────────────────────┐
│         │                        ASSESSMENTS                                 │
│ LO      ├──────┬──────┬──────┬──────────┬───────────┬──────────┬───────────┤
│         │ Q1-3 │ Q4-6 │ Q7-10│ Lab 01   │ Lab 02    │ Homework │ Exercises │
├─────────┼──────┼──────┼──────┼──────────┼───────────┼──────────┼───────────┤
│ LO1     │  ●   │      │      │    ○     │           │    ●     │           │
│ LO2     │      │      │      │    ●     │           │    ●     │     ●     │
│ LO3     │      │  ●   │      │    ●     │           │          │     ●     │
│ LO4     │      │      │      │    ●     │           │    ●     │     ●     │
│ LO5     │      │      │      │          │     ●     │    ●     │     ●     │
│ LO6     │      │      │  ●   │          │     ●     │          │     ●     │
└─────────┴──────┴──────┴──────┴──────────┴───────────┴──────────┴───────────┘

● = Primary assessment    ○ = Secondary assessment
```

---

## Mastery Levels

### Level 1: Novice (50-59%)
- Can explain basic concepts with some errors
- Implements simple parallel patterns with guidance
- Requires significant debugging support

### Level 2: Competent (60-69%)
- Explains concepts accurately
- Implements standard patterns independently
- Debugs common issues with some effort

### Level 3: Proficient (70-79%)
- Explains concepts with nuance and context
- Selects appropriate patterns for given problems
- Debugs efficiently using profiling tools

### Level 4: Expert (80-100%)
- Explains concepts with theoretical depth
- Designs novel solutions for complex problems
- Optimises systematically with measured results

---

## Self-Assessment Checklist

Before proceeding to the next unit, verify:

□ I can explain why threading does not improve CPU-bound Python performance
□ I can create a multiprocessing Pool and use map/starmap correctly
□ I can implement inter-process communication with Queue
□ I can identify race conditions and apply appropriate synchronisation
□ I can construct Dask delayed computations
□ I can process larger-than-memory datasets with Dask
□ I can profile code and identify performance bottlenecks
□ I can calculate speedup and compare to Amdahl's Law predictions

---

*14UNIT — Parallel Computing and Scalability*
*Learning Objectives Document v4.0.0*
