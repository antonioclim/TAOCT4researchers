# Unit 8: Homework Assignment

**Due Date**: See course schedule  
**Total Points**: 20  
**Submission**: Upload Python file(s) to course portal

---

## Overview and Objectives

This assignment assesses your ability to design and implement recursive and dynamic programming solutions for optimisation problems. You will demonstrate understanding of recursive decomposition, memoisation techniques, tabulation strategies and space optimisation. Each problem requires multiple implementations to illustrate the progression from naive to optimised solutions.

The problems are carefully selected to represent common patterns in research computing: combinatorial counting (Problem 1), path optimisation on grids (Problem 2) and string segmentation (Problem 3). These patterns appear frequently in domains ranging from bioinformatics to operations research.

---

## Instructions

Complete all three problems. Submit well-documented Python code with:
- Type hints on all function signatures
- Docstrings explaining approach and complexity
- Test cases demonstrating correctness
- Comments explaining non-obvious algorithmic choices

Your solutions should follow the code quality standards established in the laboratory sessions. Use the provided function signatures exactly as specified to enable automated testing.

---

## Problem 1: Climbing Stairs (6 points)

### Context

A researcher needs to climb a staircase with `n` steps. At each step, they can climb either 1, 2, or 3 steps. This problem models combinatorial counting scenarios that arise frequently in probability calculations and state machine analysis. The structure exhibits optimal substructure: the number of ways to reach step n equals the sum of ways to reach steps n-1, n-2 and n-3 (from which we can take 1, 2 or 3 steps respectively).

### Requirements

```python
def count_climbing_ways(n: int) -> int:
    """
    Count distinct ways to climb n stairs taking 1, 2, or 3 steps at a time.
    
    Args:
        n: Number of stairs (n >= 0)
    
    Returns:
        Number of distinct ways to reach the top
    
    Examples:
        >>> count_climbing_ways(3)
        4  # (1+1+1), (1+2), (2+1), (3)
        >>> count_climbing_ways(4)
        7
    """
    pass
```

### Analytical Guidance

The recurrence relation for this problem is: `C(n) = C(n-1) + C(n-2) + C(n-3)`, with base cases `C(0) = 1` (one way to stay at ground level), `C(1) = 1` and `C(2) = 2`. Note the similarity to the Tribonacci sequence and its connection to the Fibonacci sequence you studied in the laboratory.

### Deliverables

1. **Naive recursive solution** with complexity analysis (2 points)
   - Implement direct recursive solution following the recurrence relation
   - Document time complexity (exponential) and space complexity (stack depth)
   - Include comments explaining why this approach is inefficient

2. **Optimised solution** using memoisation OR tabulation (2 points)
   - Transform your naive solution using caching techniques
   - Document the improved time complexity (linear)
   - Explain your choice between top-down and bottom-up approaches

3. **Space-optimised solution** using O(1) extra space (2 points)
   - Observe that only the three most recent values are needed
   - Implement using rolling variables instead of array storage
   - Verify correctness against previous implementations

### Hints

- Start by computing small cases by hand: C(0)=1, C(1)=1, C(2)=2, C(3)=4
- The naive recursive solution will have exponential time complexity
- For the optimised version, consider whether top-down or bottom-up better suits this problem
- The space-optimised version should maintain only three variables

---

## Problem 2: Minimum Path Sum (8 points)

### Context

Given an `m × n` grid filled with non-negative integers, find a path from top-left to bottom-right that minimises the sum of all numbers along the path. You can only move down or right at any point. This problem represents a fundamental pattern in grid-based optimisation found in robotics (path planning), economics (resource allocation) and image processing (seam carving).

### Requirements

```python
def min_path_sum(grid: list[list[int]]) -> int:
    """
    Find minimum path sum from top-left to bottom-right.
    
    Args:
        grid: m×n grid of non-negative integers
    
    Returns:
        Minimum sum along any valid path
    
    Example:
        >>> grid = [[1, 3, 1],
        ...         [1, 5, 1],
        ...         [4, 2, 1]]
        >>> min_path_sum(grid)
        7  # Path: 1→3→1→1→1
    """
    pass


def min_path_sum_with_path(
    grid: list[list[int]]
) -> tuple[int, list[tuple[int, int]]]:
    """
    Find minimum path sum and return the actual path taken.
    
    Returns:
        Tuple of (minimum_sum, path_coordinates)
        Path coordinates are (row, col) tuples from start to end.
    
    Example:
        >>> grid = [[1, 3, 1],
        ...         [1, 5, 1],
        ...         [4, 2, 1]]
        >>> min_path_sum_with_path(grid)
        (7, [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
    """
    pass
```

### Analytical Guidance

Define the state as `dp[i][j]` representing the minimum sum to reach cell (i, j) from the origin. The state transition equation is: `dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])`, reflecting that we can arrive from above or from the left and we select the cheaper option.

Handle boundary conditions carefully: the first row can only be reached from the left, and the first column can only be reached from above.

### Deliverables

1. **State definition**: Clearly define what `dp[i][j]` represents (1 point)
   - Provide a clear, unambiguous definition
   - Explain why this formulation captures the problem structure

2. **State transition**: Derive the recurrence relation (1 point)
   - Write the mathematical equation
   - Explain each term's meaning

3. **Implementation** with correct initialisation and iteration (3 points)
   - Initialise boundary conditions correctly
   - Iterate in proper order (ensure dependencies are computed first)
   - Handle edge cases (1×1 grid, single row, single column)

4. **Path reconstruction**: Return the actual minimum path (2 points)
   - Store predecessor information or trace back through DP table
   - Return path as list of (row, col) coordinates

5. **Complexity analysis**: Time and space (1 point)
   - State time complexity with justification
   - State space complexity and discuss potential optimisation

### Hints

- Draw the DP table by hand for the example grid before coding
- The iteration order matters: process cells so that dp[i-1][j] and dp[i][j-1] are already computed
- For path reconstruction, work backwards from the destination
- Consider whether space can be optimised to O(n) by processing row by row

---

## Problem 3: Word Break (6 points)

### Context

Given a string `s` and a dictionary of words, determine if `s` can be segmented into a space-separated sequence of dictionary words. This problem arises in natural language processing (tokenisation), compiler design (lexical analysis) and bioinformatics (sequence annotation).

### Requirements

```python
def word_break(s: str, word_dict: list[str]) -> bool:
    """
    Determine if string can be segmented into dictionary words.
    
    Args:
        s: Input string to segment
        word_dict: List of valid dictionary words
    
    Returns:
        True if s can be segmented, False otherwise
    
    Examples:
        >>> word_break("leetcode", ["leet", "code"])
        True  # "leet code"
        >>> word_break("applepenapple", ["apple", "pen"])
        True  # "apple pen apple"
        >>> word_break("catsandog", ["cats", "dog", "sand", "and", "cat"])
        False
    """
    pass


def word_break_all(s: str, word_dict: list[str]) -> list[str]:
    """
    Return all possible segmentations of string into dictionary words.
    
    Returns:
        List of all valid segmentations as space-separated strings
    
    Example:
        >>> word_break_all("catsanddog", ["cat", "cats", "and", "sand", "dog"])
        ["cats and dog", "cat sand dog"]
    """
    pass
```

### Analytical Guidance

For the boolean version, define `dp[i]` as True if `s[0:i]` can be segmented. The recurrence is: `dp[i] = any(dp[j] and s[j:i] in word_dict for j in range(i))`. The base case is `dp[0] = True` (empty string is trivially segmentable).

For finding all segmentations, combine dynamic programming with backtracking: first identify which positions are reachable (can form valid segments from start), then recursively enumerate all valid decompositions.

### Deliverables

1. **Boolean solution** determining if segmentation exists (2 points)
   - Implement the DP solution
   - Convert word_dict to a set for O(1) lookup
   - Handle empty string and single-character cases

2. **All segmentations** using backtracking with memoisation (3 points)
   - Use the boolean DP table to prune the search space
   - Apply backtracking to enumerate all valid segmentations
   - Return results in consistent order (sorted or in order of discovery)

3. **Complexity analysis** for both solutions (1 point)
   - Boolean version: O(n² × m) where m is maximum word length
   - All segmentations: Potentially exponential in output size

### Hints

- Converting word_dict to a set dramatically improves lookup performance
- For word_break_all, first run word_break to check if any solution exists
- Consider using a helper function that builds segmentations recursively
- The number of segmentations can be exponential; test with manageable inputs

---

## Grading Rubric

| Criterion | Points |
|-----------|--------|
| Correctness | 10 |
| Complexity Analysis | 4 |
| Code Quality | 3 |
| Documentation | 3 |
| **Total** | **20** |

### Correctness (10 points)
- All provided test cases pass
- Edge cases handled (empty input, single element, etc.)
- No runtime errors
- Numerical results match expected outputs

### Complexity Analysis (4 points)
- Accurate time complexity with justification
- Accurate space complexity with justification
- Comparison between approaches where applicable
- Clear explanation of recurrence relation solving

### Code Quality (3 points)
- Type hints on all functions
- Consistent naming conventions (snake_case)
- Efficient implementation (avoid unnecessary operations)
- Proper use of Python idioms

### Documentation (3 points)
- Clear docstrings explaining approach
- Comments on non-obvious code sections
- Examples in docstrings
- Explanation of algorithmic choices

---

## Submission Checklist

- [ ] All three problems implemented
- [ ] Type hints on all function signatures
- [ ] Docstrings with complexity analysis
- [ ] Test cases included
- [ ] Code runs without errors
- [ ] File named: `homework_08_<student_id>.py`

---

## Additional Resources

- Refer to `lab_08_01_recursive_patterns.py` for memoisation patterns
- Refer to `lab_08_02_dynamic_programming.py` for tabulation techniques
- The `resources/cheatsheet.md` provides quick reference for complexity analysis
- Test your solutions against the datasets in `resources/datasets/`

---

*Good luck with your implementations!*
