# Unit 8: Recursion and Dynamic Programming Cheatsheet

## Quick Reference Guide

---

## 1. Recursion Template

```python
def recursive_function(problem):
    # Base case(s) — ALWAYS FIRST
    if is_base_case(problem):
        return base_solution
    
    # Recursive case(s)
    subproblems = decompose(problem)
    subsolutions = [recursive_function(sub) for sub in subproblems]
    return combine(subsolutions)
```

---

## 2. Memoisation Template

```python
def memoised_function(n, memo=None):
    if memo is None:
        memo = {}
    
    # Check cache
    if n in memo:
        return memo[n]
    
    # Base case
    if is_base_case(n):
        return base_value
    
    # Compute and cache
    memo[n] = compute_result(n, memo)
    return memo[n]
```

**Using @lru_cache:**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def function(n):
    if n <= base:
        return base_value
    return recursive_computation(n)
```

---

## 3. DP Tabulation Template

```python
def dp_solution(n):
    # Initialise table
    dp = [base_value] * (n + 1)
    
    # Set base cases explicitly
    dp[0] = base_0
    dp[1] = base_1
    
    # Fill table in dependency order
    for i in range(2, n + 1):
        dp[i] = transition(dp[i-1], dp[i-2], ...)
    
    return dp[n]
```

---

## 4. Complexity Quick Reference

| Pattern | Time | Space | Example |
|---------|------|-------|---------|
| Linear Recursion | O(n) | O(n) | Factorial |
| Binary Recursion (naive) | O(2ⁿ) | O(n) | Fibonacci naive |
| Binary Recursion (memo) | O(n) | O(n) | Fibonacci memo |
| Logarithmic | O(log n) | O(log n) | Binary search |
| Divide-and-Conquer | O(n log n) | O(n) | Merge sort |
| DP Tabulation | O(states × transitions) | O(states) | Knapsack |

---

## 5. Master Theorem

For **T(n) = aT(n/b) + f(n)**:

| Case | Condition | Result |
|------|-----------|--------|
| 1 | f(n) = O(n^(log_b(a) - ε)) | T(n) = Θ(n^log_b(a)) |
| 2 | f(n) = Θ(n^log_b(a)) | T(n) = Θ(n^log_b(a) · log n) |
| 3 | f(n) = Ω(n^(log_b(a) + ε)) | T(n) = Θ(f(n)) |

**Common Examples:**
- T(n) = 2T(n/2) + O(n) → O(n log n) [Merge sort]
- T(n) = 2T(n/2) + O(1) → O(n) [Binary tree traversal]
- T(n) = T(n/2) + O(1) → O(log n) [Binary search]

---

## 6. Classical DP Formulas

### Fibonacci
```
F(n) = F(n-1) + F(n-2)
F(0) = 0, F(1) = 1
```

### 0-1 Knapsack
```
dp[i][w] = max(dp[i-1][w], 
               values[i-1] + dp[i-1][w - weights[i-1]])
```
If `weights[i-1] > w`: `dp[i][w] = dp[i-1][w]`

### Longest Common Subsequence
```
dp[i][j] = dp[i-1][j-1] + 1           if s1[i-1] == s2[j-1]
         = max(dp[i-1][j], dp[i][j-1]) otherwise
```

### Edit Distance
```
dp[i][j] = dp[i-1][j-1]               if s1[i-1] == s2[j-1]
         = 1 + min(dp[i-1][j],        (delete)
                   dp[i][j-1],        (insert)
                   dp[i-1][j-1])      (substitute)
```

---

## 7. Backtracking Template

```python
def backtrack(state, solutions):
    if is_solution(state):
        solutions.append(state.copy())
        return
    
    for choice in get_choices(state):
        if is_valid(choice, state):
            state.add(choice)      # Make choice
            backtrack(state, solutions)
            state.remove(choice)   # Undo choice (backtrack)
```

---

## 8. Space Optimisation Patterns

### When previous row suffices:
```python
# Instead of dp[i][j] using dp[i-1][...]
prev = [0] * (n + 1)
curr = [0] * (n + 1)
for i in range(1, m + 1):
    for j in range(1, n + 1):
        curr[j] = compute(prev, curr, j)
    prev, curr = curr, prev
```

### When only two values needed (Fibonacci):
```python
prev2, prev1 = 0, 1
for _ in range(2, n + 1):
    prev2, prev1 = prev1, prev2 + prev1
```

### Reverse iteration for 1D optimisation:
```python
# Knapsack space optimisation
for i in range(n):
    for w in range(capacity, weights[i] - 1, -1):
        dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
```

---

## 9. Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Missing base case | Always handle n=0 and n=1 explicitly |
| Off-by-one in DP table | Allocate size n+1 for 0-indexed problems |
| Wrong iteration order | Trace dependencies; iterate in dependency order |
| Mutable default argument | Use `memo=None` then `if memo is None: memo = {}` |
| Stack overflow | Consider iterative solution or increase limit |
| Forgetting to backtrack | Always undo state changes after recursive call |

---

## 10. Decision Guide

```
Problem requires optimisation?
├─ Yes → Check for optimal substructure
│   ├─ Has it? → Check for overlapping subproblems
│   │   ├─ Yes → Use DP (memoisation or tabulation)
│   │   └─ No → Use Divide-and-Conquer
│   └─ No → Consider greedy or other approaches
└─ No → Problem requires enumeration?
    ├─ Yes → Use Backtracking with pruning
    └─ No → Simple recursion may suffice
```

---

## 11. Python Tips

```python
# Set recursion limit if needed
import sys
sys.setrecursionlimit(10000)

# Clear @lru_cache between tests
function.cache_clear()

# Get cache statistics
info = function.cache_info()
# CacheInfo(hits=X, misses=Y, maxsize=Z, currsize=W)

# Hashable arguments for memoisation
# Lists → Tuples
tuple(my_list)

# Dictionary → Frozen items
frozenset(my_dict.items())
```

---

## 12. Common Pitfalls and Solutions

| Pitfall | Solution |
|---------|----------|
| Missing base case | Always write base cases first |
| Wrong recursion direction | Ensure progress toward base case |
| Mutable default arguments | Use `None` and initialise inside function |
| Stack overflow | Increase limit or convert to iteration |
| Slow memoisation lookup | Use tuples, not lists, as keys |
| Off-by-one in DP tables | Draw the table by hand first |

---

## 13. Verification Checklist

Before submitting any DP solution, verify:
- [ ] Base cases handle all boundary conditions
- [ ] Recurrence relation is mathematically correct
- [ ] Table dimensions match problem constraints
- [ ] Iteration order respects dependencies
- [ ] Space optimisation preserves correctness
- [ ] Solution traces back correctly (if required)

---

*Keep this cheatsheet handy during exercises and exams!*
