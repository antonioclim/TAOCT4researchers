# Unit 8 Quiz: Recursion and Dynamic Programming

**Duration**: 30 minutes  
**Passing Score**: 70% (7/10)  
**Attempts**: Unlimited

---

## Instructions

Select the best answer for each question. Some questions require analysis of code or complexity expressions. You may use scratch paper for calculations.

---

### Question 1 (LO2)

What is the time complexity of the following recursive function?

```python
def mystery(n):
    if n <= 1:
        return 1
    return mystery(n - 1) + mystery(n - 1)
```

- [ ] A) O(n)
- [ ] B) O(n²)
- [ ] C) O(2ⁿ)
- [ ] D) O(n log n)

---

### Question 2 (LO2)

Given the recurrence relation T(n) = 2T(n/2) + O(n), what is T(n)?

- [ ] A) O(n)
- [ ] B) O(n log n)
- [ ] C) O(n²)
- [ ] D) O(2ⁿ)

---

### Question 3 (LO1)

Which of the following is NOT a valid base case for computing factorial recursively?

- [ ] A) `if n == 0: return 1`
- [ ] B) `if n == 1: return 1`
- [ ] C) `if n <= 1: return 1`
- [ ] D) `if n == 2: return 1`

---

### Question 4 (LO3)

After applying memoisation to the naive Fibonacci implementation, what happens to the time complexity?

- [ ] A) Remains O(φⁿ)
- [ ] B) Becomes O(n²)
- [ ] C) Becomes O(n)
- [ ] D) Becomes O(log n)

---

### Question 5 (LO4)

In the 0-1 Knapsack problem with n items and capacity W, what is the time complexity of the tabulated DP solution?

- [ ] A) O(n)
- [ ] B) O(W)
- [ ] C) O(nW)
- [ ] D) O(2ⁿ)

---

### Question 6 (LO5)

When comparing memoisation (top-down) vs tabulation (bottom-up), which statement is TRUE?

- [ ] A) Memoisation always uses less space than tabulation
- [ ] B) Tabulation may compute subproblems that are never needed
- [ ] C) Memoisation eliminates all recursion overhead
- [ ] D) Tabulation cannot be space-optimised

---

### Question 7 (LO4)

In the Longest Common Subsequence problem, what does `dp[i][j]` represent?

- [ ] A) The LCS of the first i characters of s1 and first j characters of s2
- [ ] B) The number of matching characters at positions i and j
- [ ] C) Whether characters at positions i and j are equal
- [ ] D) The edit distance between prefixes of length i and j

---

### Question 8 (LO6)

A researcher needs to find all subsets of a set that sum to a target value. Which approach is most appropriate?

- [ ] A) Dynamic programming only
- [ ] B) Backtracking with pruning
- [ ] C) Greedy algorithm
- [ ] D) Binary search

---

### Question 9 (LO6)

For the Fibonacci sequence, which implementation provides the best time AND space complexity?

- [ ] A) Naive recursion: O(φⁿ) time, O(n) space
- [ ] B) Memoised recursion: O(n) time, O(n) space
- [ ] C) Tabulated DP: O(n) time, O(n) space
- [ ] D) Space-optimised iterative: O(n) time, O(1) space

---

### Question 10 (LO2)

Analyse the following code. What recurrence relation describes its time complexity?

```python
def process(n):
    if n <= 0:
        return 0
    return process(n - 1) + process(n - 1) + process(n - 1)
```

- [ ] A) T(n) = T(n-1) + O(1)
- [ ] B) T(n) = 2T(n-1) + O(1)
- [ ] C) T(n) = 3T(n-1) + O(1)
- [ ] D) T(n) = T(n/3) + O(1)

---

## Answer Key

*(For instructor use only)*

1. **C** — Each call makes 2 recursive calls, creating a binary tree of depth n.

2. **B** — By the Master Theorem (Case 2): a=2, b=2, f(n)=n, so T(n) = Θ(n log n).

3. **D** — factorial(2) = 2, not 1. This would produce incorrect results.

4. **C** — Memoisation ensures each subproblem is computed exactly once: O(n) unique subproblems.

5. **C** — The DP table has dimensions (n+1) × (W+1), each cell computed in O(1).

6. **B** — Tabulation computes all subproblems systematically; memoisation computes only those needed.

7. **A** — Standard LCS state definition: length of LCS for prefixes of both strings.

8. **B** — Subset sum enumeration requires exploring combinations; backtracking with pruning is efficient.

9. **D** — Space-optimised version achieves optimal O(n) time with O(1) space.

10. **C** — Each call makes exactly 3 recursive calls, giving T(n) = 3T(n-1) + O(1).

---

## Grading Rubric

| Score | Grade | Feedback |
|-------|-------|----------|
| 10/10 | Excellent | Full mastery of recursion and DP concepts |
| 8-9/10 | Good | Strong understanding with minor gaps |
| 7/10 | Satisfactory | Meets learning objectives; review weak areas |
| 5-6/10 | Needs Improvement | Revisit lecture notes and lab exercises |
| <5/10 | Unsatisfactory | Schedule office hours for additional support |

---

*End of Quiz*
