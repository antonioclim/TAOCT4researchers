# 14UNIT: Self-Assessment Checklist

## Parallel Computing and Scalability

---

## Instructions

Use this checklist to assess your understanding before proceeding to the next unit. For each item, honestly evaluate your confidence level:

- ✅ **Confident**: I can explain and apply this concept independently
- ⚠️ **Developing**: I understand the basics but need more practice
- ❌ **Review Needed**: I need to revisit this material

---

## Conceptual Understanding

### Parallelism Fundamentals [LO1]

| Item | Status | Notes |
|------|--------|-------|
| I can explain the difference between parallelism and concurrency | ⬜ | |
| I can distinguish processes from threads and their trade-offs | ⬜ | |
| I understand why Python's GIL limits threading for CPU-bound work | ⬜ | |
| I can explain Amdahl's Law and calculate maximum speedup | ⬜ | |
| I know when to use multiprocessing vs threading | ⬜ | |

### Theoretical Foundations

| Item | Status | Notes |
|------|--------|-------|
| I can calculate speedup and efficiency from timing data | ⬜ | |
| I understand Gustafson's Law and scaled speedup | ⬜ | |
| I can identify embarrassingly parallel problems | ⬜ | |
| I understand the work-span model for parallel analysis | ⬜ | |

---

## Practical Skills

### Multiprocessing [LO2]

| Item | Status | Notes |
|------|--------|-------|
| I can create and manage Process objects | ⬜ | |
| I can use Pool.map and Pool.starmap correctly | ⬜ | |
| I can pass data to processes using Queue | ⬜ | |
| I can use shared memory with Value and Array | ⬜ | |
| I can implement proper process cleanup (join, terminate) | ⬜ | |

### Threading [LO3]

| Item | Status | Notes |
|------|--------|-------|
| I can create thread pools for I/O-bound tasks | ⬜ | |
| I understand when threading provides genuine benefit | ⬜ | |
| I can use Lock to prevent race conditions | ⬜ | |
| I can recognise potential deadlock situations | ⬜ | |

### concurrent.futures [LO4]

| Item | Status | Notes |
|------|--------|-------|
| I can use ProcessPoolExecutor and ThreadPoolExecutor | ⬜ | |
| I can handle Future objects (result, exception, cancel) | ⬜ | |
| I can use as_completed for progressive result handling | ⬜ | |
| I can implement timeout handling | ⬜ | |

### Dask [LO5]

| Item | Status | Notes |
|------|--------|-------|
| I can construct delayed computations | ⬜ | |
| I can create and manipulate Dask arrays | ⬜ | |
| I can process Dask DataFrames with groupby | ⬜ | |
| I understand chunking and can choose appropriate sizes | ⬜ | |
| I can select the right scheduler for my workload | ⬜ | |

### Profiling [LO6]

| Item | Status | Notes |
|------|--------|-------|
| I can use cProfile to profile functions | ⬜ | |
| I can interpret profiling output to find bottlenecks | ⬜ | |
| I can measure memory usage with tracemalloc | ⬜ | |
| I can generate before/after optimisation reports | ⬜ | |

---

## Application Scenarios

Rate your confidence in applying unit concepts to these scenarios:

| Scenario | Confidence |
|----------|------------|
| Parallelising a Monte Carlo simulation | ⬜ |
| Implementing parallel cross-validation for ML | ⬜ |
| Processing a larger-than-memory CSV file | ⬜ |
| Downloading many files from the internet concurrently | ⬜ |
| Identifying why parallel code is slower than expected | ⬜ |
| Choosing between threading and multiprocessing | ⬜ |

---

## Reflection Questions

Answer these questions to consolidate your learning:

### 1. Key Insight

What was the most important concept you learned in this unit?

_______________________________________________________________

_______________________________________________________________

### 2. Challenge

What did you find most challenging? How did you overcome it?

_______________________________________________________________

_______________________________________________________________

### 3. Application

How might you apply parallel computing in your own research?

_______________________________________________________________

_______________________________________________________________

### 4. Questions Remaining

What questions do you still have about parallel computing?

_______________________________________________________________

_______________________________________________________________

---

## Readiness Check

### Minimum Requirements for Proceeding

Before considering this unit complete, ensure you can:

- [ ] Explain the GIL and its implications
- [ ] Use multiprocessing.Pool for parallel computation
- [ ] Implement thread pools for I/O tasks
- [ ] Create basic Dask workflows
- [ ] Profile code to identify bottlenecks
- [ ] Calculate and interpret speedup metrics

### Recommended Review

If you marked ❌ or ⚠️ for any items, consider:

1. Re-reading the relevant lecture notes section
2. Reviewing the corresponding lab exercises
3. Attempting additional practice exercises
4. Consulting the further reading resources

---

## Progress Summary

| Category | Confident | Developing | Review Needed |
|----------|-----------|------------|---------------|
| Conceptual (5 items) | ___ | ___ | ___ |
| Multiprocessing (5 items) | ___ | ___ | ___ |
| Threading (4 items) | ___ | ___ | ___ |
| concurrent.futures (4 items) | ___ | ___ | ___ |
| Dask (5 items) | ___ | ___ | ___ |
| Profiling (4 items) | ___ | ___ | ___ |
| **Total (27 items)** | ___ | ___ | ___ |

**Recommendation**:
- 24+ Confident: Ready to proceed
- 18-23 Confident: Review weak areas, then proceed
- <18 Confident: Significant review recommended before proceeding

---

## Instructor Feedback

*(To be completed by instructor if submitted)*

| Aspect | Rating | Comments |
|--------|--------|----------|
| Self-awareness | ⬜ | |
| Reflection depth | ⬜ | |
| Identified gaps | ⬜ | |

---

*14UNIT — Parallel Computing and Scalability*
*Self-Assessment Checklist v4.0.0*
