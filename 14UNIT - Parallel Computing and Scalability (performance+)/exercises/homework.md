# 14UNIT: Homework Assignment

## Parallel Computing and Scalability

**Total Points**: 100 + 10 Bonus  
**Due Date**: See course schedule  
**Submission**: ZIP archive via course platform

---

## Overview

This homework assesses your ability to analyse parallel computing theoretically, implement parallel solutions practically, and profile code systematically. The assignment is divided into three parts plus an optional bonus.

**Learning Objectives Assessed**:
- Understand the theoretical foundations of parallel speedup (Amdahl's Law, Gustafson's Law)
- Apply multiprocessing techniques to CPU-bound workloads
- Evaluate performance through systematic profiling and measurement
- Create scalable data processing pipelines

**Prerequisites**: Ensure you have completed the laboratory exercises and reviewed the lecture notes on parallelism fundamentals, the GIL, and profiling techniques before attempting this assignment.

---

## General Guidelines

Before beginning the implementation tasks, consider the following guidance:

**Design Principles**:
- Start with a sequential implementation that produces correct results
- Profile the sequential version to identify parallelisation opportunities
- Apply parallelism incrementally, measuring speedup at each step
- Document your design decisions and any trade-offs made

**Code Quality Expectations**:
- Include comprehensive docstrings for all functions
- Use type hints throughout your implementation
- Follow PEP 8 style guidelines (enforced via ruff)
- Handle errors gracefully with informative messages

**Testing Requirements**:
- Write unit tests for individual functions
- Include integration tests for the complete pipeline
- Verify results match between sequential and parallel implementations

---

## Part 1: Theoretical Analysis (30 points)

### Problem 1.1: Amdahl's Law (15 points)

A computational biology pipeline processes genomic data in four stages:

| Stage | Duration (sequential) | Parallelisable? |
|-------|----------------------|-----------------|
| Data loading | 30 seconds | No |
| Sequence alignment | 240 seconds | Yes |
| Variant calling | 180 seconds | Yes |
| Report generation | 50 seconds | No |

**Tasks**:

a) **(5 points)** Calculate the parallelisable fraction *P* of this pipeline.

b) **(5 points)** Using Amdahl's Law, calculate the maximum theoretical speedup achievable with:
   - 4 processors
   - 16 processors
   - Unlimited processors

c) **(5 points)** The research team has budget for either:
   - Option A: Upgrade from 4 to 16 processors
   - Option B: Optimise the data loading stage to take 10 seconds instead of 30
   
   Which option provides greater speedup improvement? Show your calculations.

### Problem 1.2: Gustafson's Perspective (10 points)

The same genomic pipeline is used to process increasingly large datasets as more processors become available. With 8 processors, the team processes datasets 8× larger than what they could process with 1 processor.

**Tasks**:

a) **(5 points)** Explain how Gustafson's Law provides a different perspective than Amdahl's Law for this scenario.

b) **(5 points)** If the serial fraction α = 0.16 (data loading + report generation), calculate the scaled speedup with 8 processors using Gustafson's Law.

### Problem 1.3: Short Essay (5 points)

In 150-200 words, explain why Python's Global Interpreter Lock exists and discuss the trade-offs it creates for parallel programming. Include at least one specific scenario where the GIL is problematic and one where it is not.

---

## Part 2: Implementation (50 points)

### Problem 2.1: Parallel Data Pipeline (35 points)

Implement a parallel data processing pipeline that:

1. Generates synthetic sensor data (temperature, humidity, pressure)
2. Applies transformations (unit conversion, anomaly detection)
3. Aggregates statistics by sensor location
4. Measures and reports speedup compared to sequential execution

**Requirements**:

```python
# File: homework_pipeline.py

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    aggregated_data: dict[str, Any]
    sequential_time: float
    parallel_time: float
    speedup: float
    n_workers: int
    n_records: int


def generate_sensor_data(
    n_records: int,
    n_sensors: int = 100,
    seed: int = 42
) -> list[dict[str, Any]]:
    """
    Generate synthetic sensor readings.
    
    Each record should contain:
    - sensor_id: str (e.g., "sensor_001")
    - location: str (one of "north", "south", "east", "west")
    - temperature_f: float (Fahrenheit, 32-120)
    - humidity_pct: float (0-100)
    - pressure_hpa: float (950-1050)
    - timestamp: float (Unix timestamp)
    
    Returns:
        List of sensor reading dictionaries.
    """
    # YOUR IMPLEMENTATION HERE
    pass


def transform_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Apply transformations to a single record.
    
    Transformations:
    - Convert temperature from Fahrenheit to Celsius
    - Flag anomalies: temp > 40°C or humidity > 95%
    - Calculate heat index (simplified formula)
    
    Returns:
        Transformed record with additional fields.
    """
    # YOUR IMPLEMENTATION HERE
    pass


def aggregate_by_location(
    records: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """
    Aggregate statistics by sensor location.
    
    For each location, calculate:
    - mean_temp_c: average temperature (Celsius)
    - max_temp_c: maximum temperature
    - mean_humidity: average humidity
    - anomaly_rate: fraction of anomalous readings
    - record_count: number of records
    
    Returns:
        Dictionary mapping location to statistics.
    """
    # YOUR IMPLEMENTATION HERE
    pass


def run_pipeline_sequential(
    n_records: int = 1_000_000
) -> tuple[dict[str, Any], float]:
    """
    Run pipeline sequentially.
    
    Returns:
        Tuple of (aggregated_results, execution_time).
    """
    # YOUR IMPLEMENTATION HERE
    pass


def run_pipeline_parallel(
    n_records: int = 1_000_000,
    n_workers: int = 4
) -> tuple[dict[str, Any], float]:
    """
    Run pipeline in parallel using multiprocessing.
    
    Strategy:
    - Generate data (can be parallelised)
    - Transform records (embarrassingly parallel)
    - Aggregate (parallel partial aggregation + merge)
    
    Returns:
        Tuple of (aggregated_results, execution_time).
    """
    # YOUR IMPLEMENTATION HERE
    pass


def run_experiment(
    n_records: int = 500_000,
    n_workers: int = 4
) -> PipelineResult:
    """
    Run complete experiment comparing sequential vs parallel.
    
    Returns:
        PipelineResult with all metrics.
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Grading Criteria**:

| Criterion | Points |
|-----------|--------|
| Data generation correct | 5 |
| Transform function correct | 5 |
| Aggregation correct | 5 |
| Sequential pipeline works | 5 |
| Parallel pipeline works | 10 |
| Achieves meaningful speedup (>2× on 4 cores) | 5 |

### Problem 2.2: Error Handling (15 points)

Extend your pipeline to handle errors gracefully:

```python
def run_pipeline_robust(
    n_records: int = 500_000,
    n_workers: int = 4,
    failure_rate: float = 0.001  # 0.1% of records fail
) -> PipelineResult:
    """
    Run pipeline with error handling.
    
    Requirements:
    - Randomly fail processing for failure_rate fraction of records
    - Log failures but continue processing
    - Report failure statistics in result
    - Use timeouts to prevent hanging workers
    
    Returns:
        PipelineResult with error statistics.
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Grading Criteria**:

| Criterion | Points |
|-----------|--------|
| Simulates failures correctly | 3 |
| Catches and logs exceptions | 4 |
| Continues after failures | 4 |
| Reports failure statistics | 4 |

---

## Part 3: Profiling Report (20 points)

### Problem 3.1: Profile Your Pipeline

Profile your parallel pipeline implementation and write a report (300-500 words) that includes:

a) **(5 points)** **Methodology**: Describe your profiling approach (tools used, number of runs, warmup, etc.)

b) **(5 points)** **Findings**: Present profiling results identifying the top 3 time-consuming operations

c) **(5 points)** **Analysis**: Explain why these operations consume the most time

d) **(5 points)** **Recommendations**: Suggest specific optimisations that could improve performance

**Include**:
- Profiling output (cProfile stats or line_profiler output)
- Before/after timing if you implement any optimisations
- A table comparing sequential vs parallel performance across different input sizes

---

## Bonus: Distributed Computing (+10 points)

Extend your pipeline to use Dask for distributed execution:

```python
def run_pipeline_dask(
    n_records: int = 5_000_000,
    n_workers: int = 4
) -> PipelineResult:
    """
    Run pipeline using Dask.
    
    Requirements:
    - Use Dask DataFrame for data representation
    - Demonstrate out-of-core capability
    - Compare performance with multiprocessing version
    
    Returns:
        PipelineResult with Dask-specific metrics.
    """
    # YOUR IMPLEMENTATION HERE
    pass
```

**Bonus Grading**:

| Criterion | Points |
|-----------|--------|
| Dask implementation works | 4 |
| Handles larger-than-memory data | 3 |
| Performance comparison included | 3 |

---

## Verification and Testing Guidance

Before submission, ensure your implementation passes these verification steps:

**Correctness Verification**:
- Run your sequential and parallel implementations on identical input data
- Verify that aggregated results match exactly (allowing for floating-point tolerance)
- Test with small inputs first where manual verification is feasible
- Use deterministic random seeds for reproducible testing

**Performance Verification**:
- Measure execution time across multiple runs (minimum 3) to account for variance
- Verify speedup increases with worker count up to available cores
- Check that memory usage remains bounded as input size grows
- Profile to confirm overhead is not dominating execution time

**Error Handling Verification**:
- Test with deliberately malformed input data
- Verify timeout mechanisms function correctly
- Confirm failure statistics are accurate
- Ensure partial results are collected even when some workers fail

---

## Submission Requirements

Your submission should be a ZIP archive containing:

```
homework_14UNIT_[YourName]/
├── README.md                      # Brief description of your solution
├── part1_theory.md                # Answers to Part 1
├── homework_pipeline.py           # Part 2 implementation
├── test_pipeline.py               # Your test cases
├── profiling_report.md            # Part 3 report
├── profiling_output/              # Raw profiling data
│   ├── cprofile_output.txt
│   └── timing_results.csv
└── bonus_dask.py                  # Bonus (if attempted)
```

---

## Evaluation Criteria

| Component | Points | Percentage |
|-----------|--------|------------|
| Part 1: Theory | 30 | 30% |
| Part 2: Implementation | 50 | 50% |
| Part 3: Profiling | 20 | 20% |
| **Total** | **100** | **100%** |
| Bonus | +10 | +10% |

---

## Academic Integrity

This is an individual assignment. You may:
- Consult course materials and approved references
- Discuss concepts with classmates at a high level
- Use standard library documentation

You may NOT:
- Share code with other students
- Copy code from external sources without attribution
- Use AI assistants to generate solutions

All submissions will be checked for plagiarism.

---

*14UNIT — Parallel Computing and Scalability*
*Homework Assignment v4.0.0*
