# 03UNIT Quiz: Algorithmic Complexity

## 📋 Metadata

| Property | Value |
|---|---|
| **Time limit** | 30–40 minutes |
| **Total questions** | 10 |
| **Passing threshold** | 70% |
| **Permitted materials** | Personal notes from 03UNIT, Python documentation |

## Instructions

Answer all questions. Multiple-choice questions have exactly one correct option. For short-answer questions, state assumptions explicitly and justify each non-trivial step. Unless stated otherwise, assume the standard RAM cost model where elementary arithmetic and indexing into an array are treated as constant-time operations.

---

## Multiple-Choice Questions (6)

### Q1. Formal Big-O definition

Let $T(n)$ and $f(n)$ be non-negative functions for all sufficiently large $n$. Which statement is logically equivalent to $T(n) = O(f(n))$?

A. $\forall c > 0, \exists n_0: \forall n \geq n_0,\; T(n) \leq c\, f(n)$

B. $\exists c > 0, \exists n_0: \forall n \geq n_0,\; T(n) \leq c\, f(n)$

C. $\exists c > 0: \forall n \geq 1,\; T(n) \leq c\, f(n)$

D. $\forall n \geq 1,\; T(n) \leq f(n)$

---

### Q2. Relationship between $O$ and $\Theta$

Which implication holds for all functions $T(n)$ and $f(n)$ defined on the positive integers?

A. $T(n) = O(f(n)) \Rightarrow T(n) = \Theta(f(n))$

B. $T(n) = \Theta(f(n)) \Rightarrow T(n) = O(f(n))$

C. $T(n) = O(f(n)) \Rightarrow T(n) = \Omega(f(n))$

D. $T(n) = o(f(n)) \Rightarrow T(n) = \Omega(f(n))$

---

### Q3. Complexity of nested iteration

Consider the following function, where `n` is a positive integer:

```python
def f(n: int) -> int:
    s = 0
    for i in range(n):
        j = 1
        while j < n:
            s += i + j
            j *= 2
    return s
```

Assuming multiplication by 2 and integer comparison are constant-time, the worst-case time complexity is:

A. $O(n)$

B. $O(n \log n)$

C. $O(n^2)$

D. $O(\log n)$

---

### Q4. Master theorem classification

Let $T(n) = 2T(n/2) + n$ with $T(1) = 1$. Using the master theorem, which asymptotic bound is correct?

A. $T(n) = \Theta(n)$

B. $T(n) = \Theta(n \log n)$

C. $T(n) = \Theta(n^2)$

D. $T(n) = \Theta(\log n)$

---

### Q5. Measurement protocol

Two alternative implementations of the same algorithm are benchmarked. Which protocol is the most defensible if the aim is to reduce bias introduced by input order and transient system state?

A. Time each implementation once on a single randomly chosen input and report the faster one.

B. Fix a single input, execute both implementations in a fixed alternating order and report the mean time.

C. Execute a warm-up phase, randomise the order of implementations per trial and report the median with an uncertainty estimate.

D. Execute both implementations in parallel threads to ensure identical system load.

---

### Q6. Linearising a power-law model

Suppose empirical timings suggest $T(n) \approx \alpha n^k$ for constants $\alpha > 0$ and $k > 0$. Which transformation yields a linear relationship suitable for ordinary least squares regression?

A. Regress $T(n)$ on $\log n$.

B. Regress $\log T(n)$ on $n$.

C. Regress $\log T(n)$ on $\log n$.

D. Regress $T(n)$ on $n^k$ with $k$ fixed to 1.

---

## Short-Answer Questions (4)

### Q7. Recurrence solving

Solve the recurrence below and state the bound using $\Theta(\cdot)$ notation. A brief justification is required.

$$
T(n) = 3T\left(\frac{n}{2}\right) + n\log n, \quad T(1) = 1
$$

---

### Q8. Amortised versus worst-case complexity

Define amortised time complexity and contrast it with worst-case time complexity. Provide one concrete example from dynamic arrays or self-adjusting trees where amortised analysis changes the interpretation of algorithmic cost.

---

### Q9. Inferring complexity from scaling

An experiment measures running time for increasing input sizes with the same implementation and the same input-generation distribution:

| $n$ | 1,000 | 2,000 | 4,000 |
|---:|---:|---:|---:|
| $T(n)$ (s) | 0.010 | 0.040 | 0.160 |

State the most plausible asymptotic class from the set $\{O(n), O(n\log n), O(n^2)\}$ and justify your selection using scaling arguments.

---

### Q10. Minimal fair comparison protocol

Describe a minimal protocol for comparing two algorithms empirically. The answer should address at least: input selection, replication strategy, summary statistics and one threat to validity specific to high-level interpreted languages.

---

## Answers and Explanations

<details>
<summary>Show answers</summary>

### Q1
**Correct answer: B.**

The definition of $T(n) = O(f(n))$ requires the existence of a constant $c > 0$ and a threshold $n_0$ such that $T(n) \leq c f(n)$ for all $n \geq n_0$. Option A reverses the quantifiers on $c$, which is a strictly stronger statement and is generally false.

### Q2
**Correct answer: B.**

$\Theta$ denotes a tight bound: $T(n) = \Theta(f(n))$ is equivalent to $T(n) = O(f(n))$ and $T(n) = \Omega(f(n))$ simultaneously. Hence $\Theta$ implies $O$. The reverse does not hold because $O$ is an upper bound only.

### Q3
**Correct answer: B.**

The outer loop runs $n$ iterations. The inner loop doubles $j$ each time, so it executes $\lceil \log_2 n \rceil$ iterations. Multiplying yields $n \cdot \log n$ operations in the worst case.

### Q4
**Correct answer: B.**

Here $a = 2$, $b = 2$ and $f(n) = n$. Since $n^{\log_b a} = n^{\log_2 2} = n$, this is the boundary case, yielding $T(n) = \Theta(n \log n)$.

### Q5
**Correct answer: C.**

Warm-up reduces artefacts from caching, dynamic compilation and allocator state. Randomising execution order reduces confounding by time-varying system state. The median is less sensitive to outliers than the mean, and an uncertainty estimate communicates measurement variability.

### Q6
**Correct answer: C.**

Taking logarithms gives $\log T(n) = \log \alpha + k \log n$, which is linear in $\log n$. Ordinary least squares on $(\log n, \log T(n))$ estimates both intercept and slope.

### Q7
A suitable bound is **$T(n) = \Theta\bigl(n^{\log_2 3}\bigr)$**.

A convenient approach is to compare $f(n) = n\log n$ with $n^{\log_b a} = n^{\log_2 3}$. Since $\log_2 3 \approx 1.585$, one has $n\log n = O\bigl(n^{\log_2 3 - \varepsilon}\bigr)$ for some $\varepsilon > 0$. This corresponds to the first master-theorem case, hence $T(n)$ grows as $n^{\log_2 3}$.

### Q8
Amortised complexity assigns an average cost per operation over a sequence, even when individual operations occasionally incur a higher cost. Worst-case complexity bounds the cost of a single operation in isolation.

A standard example is appending to a dynamic array that doubles its capacity when full. Most appends take $\Theta(1)$ time, but a resize requires copying $\Theta(n)$ elements. Over $m$ appends starting from an empty array, the total number of element moves is $O(m)$, giving $\Theta(1)$ amortised time per append, despite $\Theta(n)$ worst-case time for a resize step.

### Q9
The scaling is most consistent with **$O(n^2)$**.

Doubling $n$ from 1,000 to 2,000 multiplies time by $4$ (0.010 to 0.040). Doubling again multiplies time by $4$ (0.040 to 0.160). Quadratic growth satisfies $T(2n) \approx 4T(n)$, whereas linear growth yields $T(2n) \approx 2T(n)$ and $n\log n$ yields slightly more than a factor of 2.

### Q10
A defensible minimal protocol specifies: (i) an input-generation distribution and a fixed random seed policy, (ii) replicated measurements across multiple independently generated instances per $n$, (iii) a warm-up phase and randomised run order to reduce temporal confounding and (iv) summary statistics that report a location estimate (often median) and an uncertainty estimate (for example, percentile interval obtained by bootstrap resampling).

In high-level interpreted languages, a salient threat to validity is variation induced by memory allocation and garbage collection. Measurements should therefore record whether garbage collection is enabled or controlled, and the protocol should avoid mixing allocation-heavy and allocation-light workloads in a fixed order.

</details>


## References

| Reference (APA 7th ed) | DOI |
|---|---|
| Knuth, D. E. (1976). Big Omicron and Big Omega and Big Theta. *SIGACT News, 8*(2), 18–24. | https://doi.org/10.1145/1008328.1008329 |
| Akra, M., & Bazzi, L. (1998). On the solution of linear recurrence equations. *Computational Optimization and Applications, 10*(2), 195–210. | https://doi.org/10.1023/A:1018373005182 |
| Dolan, E. D., & Moré, J. J. (2002). Benchmarking optimization software with performance profiles. *Mathematical Programming, 91*(2), 201–213. | https://doi.org/10.1007/s101070100263 |
