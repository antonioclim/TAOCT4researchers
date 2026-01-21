# Lecture Notes: Recursion and Dynamic Programming

## Unit 8 | The Art of Computational Thinking for Researchers

---

## 1. Introduction: The Power of Self-Reference

Recursive thinking embodies a powerful form of reductionism—the discipline of defining something in terms of simpler versions of itself. This intellectual framework, whilst appearing circular on the surface, unlocks remarkably natural approaches to computational problems that would otherwise demand complex iterative logic.

Consider the ancient problem of computing a factorial: the product of all positive integers up to a given number. We might express this concisely as "the factorial of n equals n multiplied by the factorial of n-1," with the understanding that the factorial of 0 is 1. This definition is simultaneously intuitive and self-referential, yet therein lies its power.

The relationship between recursive thinking and algorithmic efficiency reveals a critical tension that defines much of algorithm design. Whilst recursion often yields conceptually clear solutions, naive implementations can lead to catastrophically inefficient algorithms. Understanding this tension—and the techniques for resolving it—forms the core of this unit's intellectual contribution.

---

## 2. Formalising Recursive Structure

### 2.1 The Anatomy of Recursive Solutions

Every properly structured recursive algorithm contains at least two essential components that work in concert to guarantee both correctness and termination.

**Base Case(s)** constitute the simplest instances of the problem that can be solved directly without further recursion. These cases serve as termination conditions, preventing infinite recursion and providing the foundation upon which more complex solutions are constructed. A recursive algorithm must possess at least one base case, though multiple base cases are sometimes necessary for completeness.

**Recursive Case(s)** define how to solve the problem by decomposing it into one or more simpler subproblems of the same type, then combining their solutions. Each recursive call must operate on a problem that is "smaller" in some well-defined sense, ensuring progress toward a base case.

The key insight in designing recursive algorithms lies in identifying how to decompose the original problem into strictly simpler subproblems of the same form. This decomposition must eventually lead to a base case, ensuring the algorithm terminates.

### 2.2 The Recursive Call Tree

To understand the computational complexity of recursive algorithms, we can visualise their execution as a tree structure called the "recursive call tree" or "recursion tree."

In this representation, each node represents a function call, the children of a node represent the recursive calls it makes, leaf nodes represent base cases (calls that make no further recursive calls), and the root represents the initial function call.

The structure of this tree directly reflects the algorithm's time and space complexity:

- **Tree Height**: The maximum depth of the recursion tree corresponds to the maximum recursion depth, which typically determines the space complexity due to the call stack.

- **Branching Factor**: The number of recursive calls made in each function invocation. A branching factor greater than 1 can lead to exponential growth in the number of calls.

- **Total Node Count**: The total number of nodes in the tree represents the total number of function calls, which typically dominates the time complexity.

### 2.3 Recurrence Relations

To formally analyse the computational complexity of recursive algorithms, we employ recurrence relations—equations that express the cost of solving a problem of size n in terms of the cost of solving smaller instances.

For a recursive algorithm, we can construct a recurrence relation T(n) that describes its time complexity by identifying the base case complexity T(c) = d, where c is the base case input size and d is its constant cost, then expressing T(n) in terms of the complexity of the recursive calls plus the additional work performed at the current level.

For the factorial function, the recurrence relation is T(n) = T(n-1) + O(1), which solves to T(n) = O(n).

For the naive Fibonacci implementation, the recurrence becomes T(n) = T(n-1) + T(n-2) + O(1), which solves to T(n) = O(φⁿ) where φ ≈ 1.618 is the golden ratio.

---

## 3. Recursive Patterns and Their Complexity Profiles

Different recursive patterns lead to distinct complexity profiles. Recognising these patterns enables prediction of algorithm performance and selection of appropriate optimisation strategies.

### 3.1 Linear Recursion

Linear recursion occurs when each function call makes at most one recursive call. The recursion forms a single chain, and the depth is typically proportional to the input size.

**Complexity Profile**: Time O(n), Space O(n)

Common examples include factorial calculation, sum of array elements and linked list traversal. This pattern is conceptually simple and often maps directly to the mathematical definition of the problem.

### 3.2 Binary Recursion

In binary recursion, each function call makes up to two recursive calls. This typically creates a binary tree of recursive calls, which can lead to exponential complexity if not optimised.

**Complexity Profile (without optimisation)**: Time O(2ⁿ), Space O(n)

The naive Fibonacci implementation exemplifies this pattern. Binary recursion becomes inefficient when it recalculates the same subproblems multiple times—a phenomenon we shall address through memoisation.

### 3.3 Logarithmic Recursion

In logarithmic recursion, each function call reduces the problem size by a constant factor (typically half), leading to O(log n) time complexity. This pattern is extraordinarily efficient even for enormous datasets.

**Complexity Profile**: Time O(log n), Space O(log n)

Binary search exemplifies this pattern, where each recursive call examines half of the remaining search space.

### 3.4 Divide-and-Conquer Recursion

Divide-and-conquer algorithms partition the problem into multiple independent subproblems, solve them recursively, and combine their results. This pattern often yields efficient algorithms for a wide range of problems.

**Complexity Profile**: Time often O(n log n), Space varies

Merge sort illustrates this pattern, with recurrence T(n) = 2T(n/2) + O(n), which solves to T(n) = O(n log n) by the Master Theorem.

---

## 4. The Fibonacci Case Study

The Fibonacci sequence serves as the canonical example for understanding the transition from naive recursion to optimised dynamic programming solutions.

### 4.1 The Problem of Overlapping Subproblems

A direct recursive implementation—computing F(n) by calculating F(n-1) and F(n-2) and summing them—results in an exponential explosion of redundant calculations. Computing F(50) through this approach would require billions of operations, most of them repeatedly solving the same subproblems.

This inefficiency stems from a fundamental characteristic of naive recursion: overlapping subproblems. When the same subproblems are encountered repeatedly but solved independently each time, the computational cost grows dramatically.

Visualising the call structure as a tree illuminates this phenomenon. For linear recursion (like factorial calculation), the call tree forms a single branch with depth proportional to input size, yielding linear time complexity. For binary recursion (like naive Fibonacci), the tree branches at each level, leading to exponential complexity as subproblems are redundantly solved.

### 4.2 The Critical Insight

Recursion's inefficiency is not inherent to the paradigm itself but rather to specific implementations that fail to address redundant computation. This insight opens the door to memoisation and dynamic programming—techniques that preserve recursion's clarity whilst eliminating its inefficiency.

---

## 5. Memoisation: Trading Space for Time

Memoisation is perhaps the most transformative optimisation technique for recursive algorithms with overlapping subproblems. This approach stores previously computed results in a cache (typically a dictionary or array), checking this cache before performing any calculation. When a subproblem is encountered multiple times, its solution is computed only once.

The impact of memoisation on performance can be dramatic. For computing the 40th Fibonacci number, the naive implementation requires over 300 million function calls, whilst the memoised version requires only 79 unique calculations.

Memoisation is particularly effective when the algorithm has overlapping subproblems (the same subproblems are solved multiple times), the total number of distinct subproblems is relatively small, and the results of subproblems depend only on their inputs, not on global state.

This technique represents a classic space-time tradeoff, using additional memory to avoid redundant computation. For most applications, this tradeoff is well justified, as memory is typically less constraining than computational time.

---

## 6. Dynamic Programming: Bottom-Up Construction

Whilst memoisation offers a top-down approach to optimisation, dynamic programming provides a complementary bottom-up strategy. Rather than starting with the original problem and recursively breaking it down, dynamic programming begins with the smallest subproblems and systematically builds up to the complete solution, storing intermediate results in a table.

### 6.1 The Principle of Optimality

Richard Bellman's Principle of Optimality provides the mathematical foundation for dynamic programming: "An optimal policy has the property that whatever the initial state and initial decision are, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision."

This principle ensures that optimal solutions can be constructed incrementally from optimal subsolutions—the key insight that makes dynamic programming work.

### 6.2 Two Essential Properties

For dynamic programming to be applicable, a problem must exhibit two key properties:

**Optimal Substructure**: The optimal solution to the problem contains optimal solutions to its subproblems. This property ensures the correctness of the decomposition approach.

**Overlapping Subproblems**: The same subproblems are encountered multiple times during recursive solution. This property justifies the computational savings from caching results.

### 6.3 Space Optimisation

Many dynamic programming solutions can be further optimised for space efficiency by recognising that the full history of all subproblem solutions is not always necessary. Often, only the most recent results are needed to compute the next step.

For the Fibonacci sequence, we need only maintain the two most recent numbers, reducing space complexity from O(n) to O(1) whilst preserving O(n) time complexity.

---

## 7. Classical Dynamic Programming Problems

### 7.1 The 0-1 Knapsack Problem

Given n items with weights and values, and a knapsack with capacity W, select items to maximise total value without exceeding capacity. The state transition captures the fundamental choice for each item: include or exclude.

The DP formulation uses dp[i][w] to represent the maximum value achievable using items 1 through i with capacity w. The recurrence relation expresses that we either exclude the current item (taking dp[i-1][w]) or include it (taking the item's value plus the optimal solution for reduced capacity).

### 7.2 Longest Common Subsequence

Finding the longest subsequence common to two strings is fundamental to bioinformatics (sequence alignment) and text comparison (diff algorithms). The DP solution builds a table where dp[i][j] represents the LCS length for the first i characters of one string and the first j characters of the other.

### 7.3 Edit Distance

The Levenshtein distance measures the minimum number of single-character edits (insertions, deletions, substitutions) required to transform one string into another. This metric finds applications in spell checking, DNA analysis and plagiarism detection.

---

## 8. Divide-and-Conquer

The divide-and-conquer paradigm differs from dynamic programming in that subproblems are independent (non-overlapping). The approach consists of three phases: divide the problem into smaller subproblems, conquer by solving subproblems recursively, and combine solutions into the final answer.

Merge sort exemplifies this paradigm, achieving guaranteed O(n log n) time complexity by systematically dividing arrays, recursively sorting halves, and merging results.

### 8.1 The Master Theorem

The Master Theorem provides a systematic method for solving recurrence relations of the form T(n) = aT(n/b) + f(n), where a ≥ 1 and b > 1 are constants, and f(n) is an asymptotically positive function. This theorem covers the majority of divide-and-conquer recurrences encountered in practice.

The theorem distinguishes three cases based on how f(n) compares to n^(log_b a):
- If f(n) grows slower than n^(log_b a), the recursion tree dominates and T(n) = Θ(n^(log_b a))
- If f(n) grows at the same rate, both contribute equally and T(n) = Θ(n^(log_b a) log n)
- If f(n) grows faster and satisfies a regularity condition, the root dominates and T(n) = Θ(f(n))

For merge sort with T(n) = 2T(n/2) + n, we have a=2, b=2, and f(n)=n. Since n = n^(log_2 2), the second case applies, yielding T(n) = Θ(n log n).

### 8.2 When to Apply Each Paradigm

The distinction between divide-and-conquer and dynamic programming centres on subproblem independence. Divide-and-conquer partitions problems into disjoint subproblems that can be solved independently and potentially in parallel. Dynamic programming addresses problems where subproblems overlap and share dependencies, making caching essential for efficiency.

A useful diagnostic is to ask whether solving one subproblem can benefit from results of other subproblems at the same level. If yes, dynamic programming is indicated. If subproblems are truly independent, divide-and-conquer may be more appropriate and admits natural parallelisation.

---

## 9. Backtracking

Backtracking is a systematic approach to exploring solution spaces for constraint satisfaction problems. The technique incrementally builds candidates for solutions, abandoning ("backtracking" from) candidates as soon as they cannot possibly lead to a valid solution.

The N-Queens problem illustrates backtracking beautifully: we place queens row by row, immediately abandoning partial solutions that violate the constraint that no two queens threaten each other. This pruning eliminates vast portions of the search space, making tractable what would otherwise be an exponential enumeration.

---

## 10. Selecting the Right Approach

The choice among naive recursion, memoisation, tabulation and other approaches depends on problem structure and constraints:

- **Naive recursion** suits small inputs and initial prototyping, where clarity outweighs efficiency concerns.

- **Memoisation** transforms exponential algorithms when overlapping subproblems are present, maintaining the recursive structure whilst eliminating redundancy.

- **Tabulation** eliminates recursion overhead entirely and enables space optimisation when all subproblems must be computed.

- **Divide-and-conquer** applies when subproblems are independent, often enabling natural parallelisation.

- **Backtracking** addresses constraint satisfaction problems where early pruning can dramatically reduce the search space.

### 10.1 Practical Decision Framework

When confronted with a new algorithmic problem, a systematic approach aids in selecting the most effective solution strategy. Begin by characterising the problem's structure through these diagnostic questions:

First, does the problem exhibit optimal substructure? If the optimal solution can be expressed in terms of optimal solutions to smaller subproblems, recursive approaches are viable. Second, are subproblems overlapping? A naive recursive solution that recalculates the same subproblems indicates potential for memoisation or tabulation. Third, how many distinct subproblems exist? This determines whether caching is worthwhile and what data structure to employ.

For problems with few distinct subproblems relative to input size, memoisation using hash tables provides flexibility. For problems where subproblems form a dense grid (as in sequence alignment or grid path problems), tabulation with arrays offers superior cache performance. When only the most recent rows of a DP table are needed for computing subsequent rows, space can be reduced from O(n²) to O(n) or even O(1).

### 10.2 Research Applications

These paradigms find widespread application across scientific domains. In bioinformatics, the Needleman-Wunsch and Smith-Waterman algorithms for sequence alignment employ DP with O(nm) complexity, where n and m are sequence lengths. In operations research, resource allocation and scheduling problems frequently admit DP formulations. Natural language processing relies on parsing algorithms like CKY that use DP to efficiently analyse sentence structure.

Understanding these paradigms and their trade-offs enables researchers to select appropriate algorithmic strategies for the computational challenges they encounter across domains.

---

## References

Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.

Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley.

---

*© 2026 Dr. Antonio Clim | Academy of Economic Studies, Bucharest (ASE-CSIE)*
