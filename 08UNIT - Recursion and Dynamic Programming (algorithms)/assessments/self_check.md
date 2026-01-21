# Unit 8: Self-Assessment Checklist

## Instructions

Use this checklist to evaluate your understanding before attempting the quiz. Mark each item as you gain confidence in that area.

---

## Core Concepts

### Recursion Fundamentals

- [ ] I can identify base cases in a recursive problem
- [ ] I can formulate recursive cases that make progress toward base cases
- [ ] I understand how the call stack works during recursive execution
- [ ] I can trace through a recursive function call by hand
- [ ] I can identify when recursion will cause stack overflow

### Complexity Analysis

- [ ] I can derive recurrence relations from recursive code
- [ ] I can apply the Master Theorem to divide-and-conquer recurrences
- [ ] I understand the difference between O(n), O(n log n), O(2ⁿ)
- [ ] I can analyse both time and space complexity of recursive algorithms
- [ ] I understand why naive Fibonacci is O(φⁿ) and not O(2ⁿ)

### Memoisation

- [ ] I understand what overlapping subproblems means
- [ ] I can implement memoisation using a dictionary
- [ ] I can use @lru_cache decorator effectively
- [ ] I know when memoisation will and won't help
- [ ] I can analyse the complexity improvement from memoisation

### Dynamic Programming

- [ ] I understand optimal substructure
- [ ] I can define DP state variables appropriately
- [ ] I can formulate state transition equations
- [ ] I can initialise DP tables with correct base values
- [ ] I can determine correct iteration order based on dependencies
- [ ] I can implement space optimisation when applicable

---

## Problem-Specific Skills

### Fibonacci Sequence

- [ ] I can implement naive recursive Fibonacci
- [ ] I can implement memoised Fibonacci
- [ ] I can implement tabulated Fibonacci
- [ ] I can implement space-optimised Fibonacci (O(1) space)
- [ ] I understand why each version has its stated complexity

### Divide-and-Conquer

- [ ] I can implement binary search recursively
- [ ] I can implement merge sort with the merge operation
- [ ] I can implement quick sort with partitioning
- [ ] I understand why merge sort is always O(n log n)
- [ ] I understand quick sort's worst-case O(n²) behaviour

### Tree Algorithms

- [ ] I can implement preorder, inorder, postorder traversals
- [ ] I can compute tree height recursively
- [ ] I can count nodes in a tree recursively
- [ ] I understand traversal applications (BST sorting, expression evaluation)

### Backtracking

- [ ] I understand the backtracking template
- [ ] I can generate permutations using backtracking
- [ ] I can generate subsets (power set)
- [ ] I can solve N-Queens with pruning
- [ ] I can implement subset sum with backtracking

### Classical DP Problems

- [ ] I can solve 0-1 Knapsack with tabulation
- [ ] I can reconstruct selected items from Knapsack DP table
- [ ] I can implement Longest Common Subsequence
- [ ] I can implement Edit Distance (Levenshtein)
- [ ] I understand Matrix Chain Multiplication optimisation

---

## Self-Evaluation Questions

Answer these questions to gauge your readiness:

1. **Explain in your own words** why memoisation transforms Fibonacci from exponential to linear time.

2. **Without looking at notes**, write the state transition for the 0-1 Knapsack problem.

3. **Identify the error**: Why doesn't the following base case work for factorial?
   ```python
   if n == 2:
       return 2
   ```

4. **Compare and contrast**: When would you choose memoisation over tabulation?

5. **Apply your knowledge**: Given a problem asking for the minimum cost path through a grid, what DP state would you define?

---

## Confidence Rating

Rate your confidence for each learning objective:

| Objective | Not Confident | Somewhat Confident | Very Confident |
|-----------|---------------|-------------------|----------------|
| LO1: Implement recursive solutions | ⬜ | ⬜ | ⬜ |
| LO2: Derive recurrence relations | ⬜ | ⬜ | ⬜ |
| LO3: Apply memoisation | ⬜ | ⬜ | ⬜ |
| LO4: Construct DP solutions | ⬜ | ⬜ | ⬜ |
| LO5: Compare approaches | ⬜ | ⬜ | ⬜ |
| LO6: Select optimal strategy | ⬜ | ⬜ | ⬜ |

---

## Action Plan

If you marked "Not Confident" for any area:

1. **Revisit** the corresponding section in lecture notes
2. **Re-watch** the relevant portion of the slides
3. **Re-attempt** the lab exercises for that topic
4. **Practice** additional exercises from the practice folder
5. **Seek help** during office hours if still unclear

---

## Ready for Quiz?

You should feel ready for the quiz when:

- ✅ All checklist items are marked
- ✅ All learning objectives rated "Somewhat" or "Very Confident"
- ✅ Labs completed with passing tests
- ✅ Can answer self-evaluation questions without notes

---

*End of Self-Assessment*
