# Unit 8: Glossary of Terms

## A

**Algorithm**
: A finite sequence of well-defined instructions for solving a problem or performing a computation.

## B

**Backtracking**
: An algorithmic technique that incrementally builds candidates for solutions, abandoning candidates ("backtracking") as soon as they cannot possibly lead to a valid solution.

**Base Case**
: The simplest instance of a recursive problem that can be solved directly without further recursion. Provides termination for the recursion.

**Bellman Equation**
: A recursive decomposition of the value function in dynamic programming, expressing the optimal value at one state in terms of optimal values at successor states.

**Binary Recursion**
: A recursive pattern where each function call makes exactly two recursive calls, potentially creating exponential time complexity without optimisation.

**Bottom-Up**
: An approach to dynamic programming that starts with the smallest subproblems and iteratively builds up to the full solution. Also called tabulation.

## C

**Cache**
: A data structure (typically a dictionary or array) used to store previously computed results for reuse. Central to memoisation.

**Call Stack**
: A stack data structure that stores information about active function calls. Each recursive call adds a frame; function returns pop frames.

**Curse of Dimensionality**
: The exponential growth in problem complexity as the number of state variables increases, particularly relevant in dynamic programming.

## D

**Divide-and-Conquer**
: An algorithmic paradigm that solves a problem by dividing it into independent subproblems, solving each recursively, and combining the results.

**Dynamic Programming (DP)**
: An algorithmic technique for solving problems by breaking them into simpler subproblems and storing solutions to avoid redundant computation. Requires optimal substructure and overlapping subproblems.

## E

**Edit Distance**
: The minimum number of single-character edits (insertions, deletions, substitutions) required to transform one string into another. Also called Levenshtein distance.

## F

**Fibonacci Sequence**
: A sequence where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, ...

## G

**Golden Ratio (φ)**
: The irrational number (1 + √5)/2 ≈ 1.618. Appears in the complexity analysis of naive Fibonacci recursion.

## K

**Knapsack Problem**
: An optimisation problem: given items with weights and values, select items to maximise total value without exceeding a weight capacity.

## L

**Linear Recursion**
: A recursive pattern where each function call makes at most one recursive call, forming a single chain of calls.

**Logarithmic Recursion**
: A recursive pattern where each call reduces the problem size by a constant factor (typically half), yielding O(log n) complexity.

**Longest Common Subsequence (LCS)**
: The longest sequence of characters that appears in both of two given strings in the same order, not necessarily contiguously.

**LRU Cache**
: Least Recently Used cache; an eviction policy that removes the least recently accessed items when the cache reaches capacity.

## M

**Master Theorem**
: A formula for solving recurrence relations of the form T(n) = aT(n/b) + f(n), commonly arising in divide-and-conquer algorithms.

**Memoisation**
: An optimisation technique that stores the results of expensive function calls and returns cached results when the same inputs recur.

**Merge Sort**
: A divide-and-conquer sorting algorithm with guaranteed O(n log n) time complexity that splits, recursively sorts, and merges arrays.

## O

**Optimal Substructure**
: A property where the optimal solution to a problem contains optimal solutions to its subproblems. Required for dynamic programming.

**Overlapping Subproblems**
: A property where the same subproblems are encountered multiple times during recursive computation. Required for dynamic programming.

## P

**Principle of Optimality**
: Bellman's theorem stating that optimal solutions contain optimal subsolutions, enabling incremental construction of global optima.

**Pruning**
: Eliminating portions of the search space in backtracking by recognising that certain partial solutions cannot lead to valid complete solutions.

## Q

**Quick Sort**
: A divide-and-conquer sorting algorithm that partitions around a pivot element. O(n log n) average case, O(n²) worst case.

## R

**Recurrence Relation**
: An equation expressing the value of a function at input n in terms of values at smaller inputs. Used to analyse recursive algorithm complexity.

**Recursion**
: A technique where a function calls itself with smaller inputs, eventually reaching base cases that terminate the process.

**Recursive Case**
: The part of a recursive definition that breaks the problem into smaller instances of the same type.

## S

**Space Complexity**
: The amount of memory an algorithm requires as a function of input size, including both auxiliary space and call stack.

**Stack Overflow**
: A runtime error occurring when the call stack exceeds its maximum size, typically due to unbounded or excessively deep recursion.

**State**
: In dynamic programming, the parameters that uniquely identify a subproblem. The DP table is indexed by state variables.

**State Transition**
: The equation defining how to compute the solution for one state from solutions to previously computed states.

## T

**Tabulation**
: A bottom-up dynamic programming approach that fills a table iteratively from base cases to the final solution.

**Tail Recursion**
: A recursive call that is the last operation in a function, enabling potential optimisation to constant space (though not in Python).

**Time Complexity**
: The computational time an algorithm requires as a function of input size, typically expressed in Big-O notation.

**Top-Down**
: An approach to dynamic programming that starts with the original problem and recursively solves subproblems, caching results. Synonymous with memoisation.

**Tree Traversal**
: Systematically visiting all nodes in a tree. Common orders: preorder (root-left-right), inorder (left-root-right), postorder (left-right-root).

---

*End of Glossary*
