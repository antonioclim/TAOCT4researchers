# 03UNIT Homework: Algorithmic Complexity

## 📋 Metadata

| Property | Value |
|---|---|
| **Deadline** | 31 January 2026, 23:59 GMT |
| **Total Points** | 100 |
| **Estimated Time** | 6–8 hours |
| **Difficulty** | ⭐⭐⭐ (3/5) |
| **Permitted Languages** | Python 3.12+ |
| **Permitted Libraries** | NumPy (≥1.24), Pandas (≥2.0), Matplotlib (≥3.7), SciPy (≥1.11) |

## 🔗 Prerequisites

- [ ] Completed **Lab 03.1** (benchmark suite)
- [ ] Completed **Lab 03.2** (complexity analyser)
- [ ] Read the lecture notes, especially the sections on asymptotic notation, recurrence relations and empirical measurement

## 🎯 Objectives Assessed

This homework assesses the following learning outcomes of 03UNIT:

1. **[Understand]** Explain Big-O notation and complexity classes
2. **[Apply]** Implement benchmarking infrastructure with principled statistical summaries
3. **[Analyse]** Estimate algorithmic complexity empirically and reconcile empirical findings with theoretical analysis

### Reporting Expectations

Assessment in this UNIT is not limited to obtaining numerically plausible timings. The substantive criterion is whether the measurement protocol and the inferential step would survive scrutiny in a research methods section. Consequently, each part of this homework expects you to articulate, in prose, the assumptions under which your measurements are meaningful.

Include a short technical note (approximately 600–900 words) either as a Markdown section at the end of your submission or in a separate `report.md` stored alongside your solution. The note should: (i) describe the execution environment (CPU architecture, operating system, Python version and library versions), (ii) state the timing protocol (warm-up policy, number of repetitions, randomisation of run order and any isolation measures) and (iii) report results as both a table of summary statistics and a log–log plot with the fitted model superimposed. Where you reject a candidate model, record the evidence (residual pattern, cross-validation error or an information criterion) rather than an intuition.

When discussing limitations, concentrate on concrete threats to validity such as cache effects, dynamic frequency scaling, interpreter dispatch overhead and input-instance heterogeneity. A concise limitations paragraph that names mechanisms is preferable to generic cautionary remarks.

---

## Context and General Requirements

Algorithm selection in research software frequently turns upon performance claims whose evidential basis is, at best, uneven. The conceptual apparatus of asymptotic analysis is indispensable, yet it rarely suffices in isolation: constant factors, implementation details and hardware effects may dominate at realistic input sizes. Conversely, empirical timing without a defensible protocol invites spurious conclusions, particularly under measurement noise, caching behaviour and runtime warm-up.

This homework requires the construction of a small, testable benchmarking module and an accompanying complexity estimator. The tasks are intentionally constrained in scope, but they must exhibit the engineering discipline expected in publishable research code: explicit assumptions, clear error semantics, deterministic behaviour under fixed seeds and meaningful statistical summaries.


### Submission Constraints

- **Type hints**: 100% coverage.
- **Docstrings**: Google style for all public functions and classes.
- **Logging**: Use the `logging` module; do not use `print`.
- **Testing**: Provide unit tests for each required function (you may add a new test module under `tests/`).
- **Formatting**: `ruff format` and `ruff check` must pass.
- **Determinism**: Any randomness must be controlled via an explicit seed.

### Naming Convention

Implement your solution in a new file:

- `exercises/solutions/homework_03_solution.py`

Your tests should be placed in:

- `tests/test_homework_03.py`

You may reuse non-trivial components from your labs, provided you cite the relevant file and commit hash in a comment.

---

## Part 1: Microbenchmark Driver with Statistical Summaries (40 points)

### Aim

Implement a minimal microbenchmark driver that times callables under a controlled protocol and returns a structured result with well-defined summary statistics.

### Formal Specification

Let $f$ denote a Python callable and let $t_i$ be the measured wall-clock duration (in seconds) of the $i$-th timed repetition, measured by a high-resolution monotonic clock.

Define the observed sample as $\{t_1, \dots, t_n\}$. Your driver must compute:

- **Median**: $\tilde{t} = \mathrm{median}(t_1, \dots, t_n)$
- **Median absolute deviation (MAD)**:

$$
\mathrm{MAD} = \mathrm{median}(|t_i - \tilde{t}|), \quad i = 1, \dots, n \tag{1}
$$

- **Trimmed mean** at trimming fraction $\alpha \in [0, 0.45]$:

$$
\bar{t}_{\alpha} = \frac{1}{n - 2k} \sum_{i=k+1}^{n-k} t_{(i)}, \quad k = \lfloor \alpha n \rfloor \tag{2}
$$

where $t_{(i)}$ are the order statistics.

These summaries are required because timing distributions are frequently skewed. The median and MAD are less sensitive to outliers than the arithmetic mean.

### Requirements

Implement the following public API.

1. `BenchmarkConfig` (10 points)
   - A `@dataclass(frozen=True)` containing:
     - `warmup: int` (default 3, must be ≥ 0)
     - `repetitions: int` (default 25, must be ≥ 5)
     - `trim_fraction: float` (default 0.1, must satisfy $0 \leq \alpha \leq 0.45$)
     - `seed: int | None` (default `0`, `None` indicates no shuffling)

2. `BenchmarkResult` (10 points)
   - A `@dataclass(frozen=True)` containing:
     - `name: str`
     - `times_s: list[float]` (raw repetition times)
     - `median_s: float`
     - `mad_s: float`
     - `trimmed_mean_s: float`
     - `n: int`

3. `run_benchmark` (20 points)
   - Signature:

```python
from collections.abc import Callable

def run_benchmark(
    name: str,
    fn: Callable[[], object],
    *,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    ...
```

   - Behaviour:
     - Executes `warmup` calls to `fn` which are not timed.
     - Runs `repetitions` timed calls.
     - Uses `time.perf_counter()` (or `perf_counter_ns()` with scaling) as the timing source.
     - If `seed` is not `None`, shuffles the order of repetitions in a way that is deterministic under the seed. (A simple method is to time a list of identical callables in shuffled order.)
     - Raises `ValueError` on invalid configuration.

### Test Cases

Your tests must include at least the following assertions (you may extend them).

```python
from exercises.solutions.homework_03_solution import (
    BenchmarkConfig,
    run_benchmark,
)


def test_benchmark_repetitions_count() -> None:
    cfg = BenchmarkConfig(warmup=0, repetitions=7, trim_fraction=0.0, seed=0)
    res = run_benchmark("noop", lambda: None, config=cfg)
    assert res.n == 7
    assert len(res.times_s) == 7


def test_benchmark_statistics_non_negative() -> None:
    cfg = BenchmarkConfig(warmup=1, repetitions=10, trim_fraction=0.1, seed=0)
    res = run_benchmark("noop", lambda: None, config=cfg)
    assert res.median_s >= 0.0
    assert res.mad_s >= 0.0
    assert res.trimmed_mean_s >= 0.0


def test_trim_fraction_validation() -> None:
    try:
        BenchmarkConfig(warmup=0, repetitions=10, trim_fraction=0.9, seed=0)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_repetitions_validation() -> None:
    try:
        BenchmarkConfig(warmup=0, repetitions=3, trim_fraction=0.0, seed=0)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_deterministic_seed_produces_same_result_shape() -> None:
    cfg = BenchmarkConfig(warmup=0, repetitions=8, trim_fraction=0.1, seed=123)
    r1 = run_benchmark("noop", lambda: None, config=cfg)
    r2 = run_benchmark("noop", lambda: None, config=cfg)
    assert r1.n == r2.n
    assert len(r1.times_s) == len(r2.times_s)
```

### Hints

<details>
<summary>💡 Hint 1</summary>
To compute the MAD, sort is unnecessary: compute the median, then compute the absolute deviations and take their median.
</details>

<details>
<summary>💡 Hint 2</summary>
For a small homework-scale driver, timing overhead can dominate. Keep the body of the timing loop minimal and isolate the statistics calculation outside the loop.
</details>

<details>
<summary>💡 Hint 3</summary>
If you shuffle repetitions, do not shuffle the measured times after the fact. Shuffle the execution order and record times as observed.
</details>

---

## Part 2: Empirical Complexity Estimation via Model Fitting (40 points)

### Aim

Implement a function that estimates the most plausible asymptotic family from empirical $(n, t)$ observations, where $n$ denotes input size and $t$ denotes the median (or trimmed mean) time for that size.

### Formal Specification

Given observations $(n_i, t_i)$ for $i = 1, \dots, m$ with strictly increasing $n_i$ and $t_i > 0$, we consider candidate models:

- $t(n) = a$ (constant)
- $t(n) = a \log n$
- $t(n) = a n$
- $t(n) = a n \log n$
- $t(n) = a n^2$
- $t(n) = a n^3$

with $a > 0$.

For each candidate $g(n)$, estimate $a$ by least squares on $t_i \approx a g(n_i)$. Let $\hat{t}_i = \hat{a} g(n_i)$ and define the residual sum of squares:

$$
\mathrm{RSS} = \sum_{i=1}^{m} (t_i - \hat{t}_i)^2 \tag{3}
$$

Select the model minimising RSS, subject to a tie-breaker preferring the lower-complexity model when the RSS difference is within 1%.

### Requirements

Implement the following public API.

1. `ComplexityClass` (10 points)
   - An `Enum` with members:
     - `O1`, `OLOGN`, `ON`, `ONLOGN`, `ON2`, `ON3`

2. `fit_complexity` (30 points)

```python
from collections.abc import Sequence
from typing import NamedTuple

class ComplexityFit(NamedTuple):
    """Result of complexity fitting."""

    cls: ComplexityClass
    a: float
    rss: float


def fit_complexity(n: Sequence[int], t: Sequence[float]) -> ComplexityFit:
    ...
```

   - Behaviour:
     - Validates inputs: lengths match, length ≥ 4, all $n_i > 0$, all $t_i > 0$.
     - Uses `math.log` for the $\log n$ term.
     - Computes $\hat{a}$ analytically for each model by minimising RSS for a one-parameter linear model.
     - Applies the tie-break rule stated above.
     - Raises `ValueError` on invalid input.

### Test Cases

Your tests must include at least the following assertions.

```python
import math

from exercises.solutions.homework_03_solution import (
    ComplexityClass,
    fit_complexity,
)


def test_fit_prefers_quadratic_when_time_scales_as_n2() -> None:
    n = [100, 200, 400, 800]
    t = [1e-6 * (x ** 2) for x in n]
    fit = fit_complexity(n, t)
    assert fit.cls == ComplexityClass.ON2


def test_fit_prefers_linear_for_linear_data() -> None:
    n = [100, 200, 400, 800]
    t = [2e-6 * x for x in n]
    fit = fit_complexity(n, t)
    assert fit.cls == ComplexityClass.ON


def test_fit_handles_nlogn_data() -> None:
    n = [128, 256, 512, 1024]
    t = [1e-6 * x * math.log(x) for x in n]
    fit = fit_complexity(n, t)
    assert fit.cls in {ComplexityClass.ONLOGN, ComplexityClass.ON}


def test_fit_rejects_non_positive_times() -> None:
    try:
        fit_complexity([1, 2, 3, 4], [0.1, 0.0, 0.3, 0.4])
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_fit_requires_minimum_points() -> None:
    try:
        fit_complexity([1, 2, 3], [0.1, 0.2, 0.3])
        assert False, "Expected ValueError"
    except ValueError:
        assert True
```

### Hints

<details>
<summary>💡 Hint 1</summary>
For a model $t \approx a g$, the least-squares minimiser is $\hat{a} = \frac{\langle t, g \rangle}{\langle g, g \rangle}$ provided $\langle g, g \rangle > 0$.
</details>

<details>
<summary>💡 Hint 2</summary>
The tie-breaker is a deliberate reminder that empirical curves are often ambiguous at small scales. Encode the complexity ordering explicitly and apply it deterministically.
</details>

---

## Part 3: Bonus — Theory Meets Measurement (20 points)

### Aim

Select **two** algorithms from the list below, derive their theoretical time complexity and then verify the prediction empirically using your Part 1 driver.

Choose two of:

- Naïve matrix multiplication (triple nested loops)
- Polynomial evaluation by Horner's method
- Computing all pairwise Euclidean distances for $n$ points in $\mathbb{R}^d$ (with fixed $d$)
- Sorting a list of $n$ numbers and then performing binary search for $k$ queries (with $k$ fixed)

### Requirements

1. **Theoretical analysis** (10 points)
   - Provide a short derivation in Markdown (place it at the end of `exercises/solutions/homework_03_solution.py` as a module-level string or in a separate `exercises/solutions/homework_03_bonus.md`).
   - The derivation must state the cost model and justify each step (loop bounds, recurrence, dominating term).

2. **Empirical confirmation** (10 points)
   - Measure runtimes for at least four input sizes per algorithm.
   - Use the median or trimmed mean from Part 1.
   - Run `fit_complexity` from Part 2 on the resulting series and report the selected class.

### Evaluation Criteria

Credit is awarded for coherence between the theoretical argument, the experimental protocol and the interpretation of results. Discrepancies are not penalised if they are explained plausibly (for example, cache effects or constant factors dominating the measured range).

---

## ✅ Submission Checklist

- [ ] `exercises/solutions/homework_03_solution.py` added
- [ ] `tests/test_homework_03.py` added
- [ ] All tests pass: `make test`
- [ ] Lint and format clean: `make lint` and `make format`
- [ ] Type checking passes: `make type`
- [ ] No `print` usage
- [ ] Deterministic behaviour under fixed seeds

---

© 2025 Antonio Clim. All rights reserved.
