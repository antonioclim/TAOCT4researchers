# Week 3 Quiz: Algorithmic Complexity

> **Course:** The Art of Computational Thinking for Researchers  
> **Week:** 3 of 7  
> **Time Limit:** 20 minutes  
> **Total Points:** 100

---

## Instructions

- Answer all 10 questions
- Questions 1-6 are multiple choice (8 points each = 48 points)
- Questions 7-10 are short answer (13 points each = 52 points)
- No external resources permitted
- Show your working for short answer questions

---

## Section A: Multiple Choice (48 points)

**Question 1 (8 points)**

What is the time complexity of the following code?

```python
def mystery(n):
    count = 0
    i = n
    while i > 0:
        count += 1
        i = i // 2
    return count
```

- A) O(1)
- B) O(log n)
- C) O(n)
- D) O(n log n)

---

**Question 2 (8 points)**

Which of the following correctly describes the relationship between Big-O and Big-Θ notation?

- A) Big-O provides a tight bound; Big-Θ provides an upper bound
- B) Big-O provides an upper bound; Big-Θ provides a tight bound
- C) Both provide the same information and are interchangeable
- D) Big-O is for time complexity; Big-Θ is for space complexity

---

**Question 3 (8 points)**

Consider an algorithm with time complexity T(n) = 3n² + 5n + 100. Which statement is TRUE?

- A) T(n) = O(n) because linear terms dominate
- B) T(n) = O(n²) and T(n) = Ω(n²), therefore T(n) = Θ(n²)
- C) T(n) = O(n³) but not O(n²)
- D) The constant 100 makes this O(1) for small n

---

**Question 4 (8 points)**

You measure an algorithm's running time and obtain: T(100) = 10ms, T(1000) = 100ms, T(10000) = 1000ms. What is the likely complexity?

- A) O(1)
- B) O(log n)
- C) O(n)
- D) O(n²)

---

**Question 5 (8 points)**

Which sorting algorithm has O(n log n) complexity in the WORST case?

- A) Quicksort
- B) Bubble sort
- C) Merge sort
- D) Insertion sort

---

**Question 6 (8 points)**

A dynamic array doubles its capacity when full. What is the AMORTISED time complexity of appending n elements?

- A) O(n²) total, O(n) amortised per element
- B) O(n log n) total, O(log n) amortised per element
- C) O(n) total, O(1) amortised per element
- D) O(2^n) total due to repeated doubling

---

## Section B: Short Answer (52 points)

**Question 7 (13 points)**

Analyse the following code and determine its time complexity. Show your reasoning using the sum and product rules.

```python
def process(n):
    # Part A
    for i in range(n):
        for j in range(n):
            print(i, j)
    
    # Part B
    for k in range(n):
        print(k)
    
    # Part C
    m = n
    while m > 1:
        print(m)
        m = m // 2
```

---

**Question 8 (13 points)**

The recurrence relation for merge sort is: T(n) = 2T(n/2) + O(n)

Using the Master Theorem, determine the complexity of merge sort. State which case of the Master Theorem applies and show your working.

*Hint: The Master Theorem states that for T(n) = aT(n/b) + O(n^d):*
- *Case 1: If d < log_b(a), then T(n) = O(n^(log_b(a)))*
- *Case 2: If d = log_b(a), then T(n) = O(n^d log n)*
- *Case 3: If d > log_b(a), then T(n) = O(n^d)*

---

**Question 9 (13 points)**

Explain why measuring algorithm performance requires multiple runs and statistical analysis rather than a single timing measurement. Describe at least three sources of measurement variability that benchmarking must account for.

---

**Question 10 (13 points)**

You have measured the following timing data for an algorithm:

| n | Time (ms) |
|---|-----------|
| 1000 | 5 |
| 2000 | 20 |
| 4000 | 80 |
| 8000 | 320 |

a) Estimate the complexity exponent using the log-log technique. Show your calculation.

b) What complexity class does this suggest?

c) Predict the approximate running time for n = 16000.

---

## Answer Key

<details>
<summary>Click to reveal answers (instructor use only)</summary>

### Section A: Multiple Choice

**Q1: B) O(log n)**
The variable i is halved each iteration (i = i // 2), so the loop runs log₂(n) times.

**Q2: B) Big-O provides an upper bound; Big-Θ provides a tight bound**
Big-O gives an asymptotic upper bound, while Big-Θ (Theta) provides both upper and lower bounds, indicating exact growth rate.

**Q3: B) T(n) = O(n²) and T(n) = Ω(n²), therefore T(n) = Θ(n²)**
The dominant term is 3n², which determines both the upper and lower bounds.

**Q4: C) O(n)**
The time increases proportionally with n (10× size increase → 10× time increase), indicating linear complexity.

**Q5: C) Merge sort**
Merge sort guarantees O(n log n) in all cases. Quicksort is O(n²) worst case; bubble and insertion sort are O(n²).

**Q6: C) O(n) total, O(1) amortised per element**
While individual resizes cost O(n), they occur exponentially less frequently, giving O(1) amortised cost per operation.

### Section B: Short Answer

**Q7: O(n²)**
- Part A: O(n) × O(n) = O(n²) (nested loops, product rule)
- Part B: O(n) (single loop)
- Part C: O(log n) (halving each iteration)
- Total: O(n²) + O(n) + O(log n) = O(n²) (sum rule, dominant term)

**Q8: T(n) = Θ(n log n)**

From the recurrence T(n) = 2T(n/2) + O(n):
- a = 2 (two subproblems)
- b = 2 (each half size)
- d = 1 (linear work to merge)

Calculate log_b(a) = log₂(2) = 1

Since d = 1 = log_b(a), this is Case 2 of the Master Theorem.

Therefore: T(n) = O(n^d log n) = O(n log n)

**Q9: Sources of measurement variability (any three):**

1. **JIT compilation**: First runs may include compilation overhead
2. **Cache effects**: Cold cache vs. warm cache significantly affects timing
3. **Garbage collection**: GC pauses introduce unpredictable delays
4. **OS scheduling**: Other processes compete for CPU time
5. **Branch prediction**: Initial runs train the branch predictor
6. **Memory allocation**: Heap state varies between runs
7. **Thermal throttling**: CPU may slow down under sustained load

Statistical analysis (mean, median, standard deviation) provides robust estimates despite this variability.

**Q10:**

a) Using doubling:
- From n=1000 to n=2000: time goes 5 → 20, ratio = 4 = 2²
- From n=2000 to n=4000: time goes 20 → 80, ratio = 4 = 2²
- From n=4000 to n=8000: time goes 80 → 320, ratio = 4 = 2²

When n doubles, time quadruples → exponent = log₂(4) = 2

Alternative log-log method:
- log(8000/1000) = log(8) ≈ 2.08
- log(320/5) = log(64) ≈ 4.16
- Slope = 4.16 / 2.08 ≈ 2.0

b) **O(n²)** — quadratic complexity

c) For n = 16000 (double of 8000):
- T(16000) = T(8000) × 4 = 320 × 4 = **1280 ms**

</details>

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 3 — Algorithmic Complexity*  
*© 2025 Antonio Clim. All rights reserved.*
