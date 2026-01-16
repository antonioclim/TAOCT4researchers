# Week 3: Glossary of Terms

> **Algorithmic Complexity Terminology**  
> © 2025 Antonio Clim. All rights reserved.

---

## A

### Aggregate Analysis
A method for amortised analysis that computes the total cost of a sequence of operations and divides by the number of operations to get the average cost per operation.

### Algorithm
A finite sequence of well-defined instructions for solving a class of problems or performing a computation.

### Amortised Analysis
A technique for analysing the average time complexity of operations over a worst-case sequence, even when individual operations may be expensive. The amortised cost is the average cost per operation over a worst-case sequence of operations.

### Asymptotic Analysis
The study of algorithm performance as input size approaches infinity. Focuses on growth rate rather than exact operation counts.

### Asymptotic Notation
Mathematical notation (Big-O, Big-Ω, Big-Θ) used to describe the limiting behaviour of functions as arguments tend to infinity.

---

## B

### Benchmark
A standardised test used to measure and compare the performance of algorithms, systems or hardware under specific conditions.

### Best Case
The scenario that results in the minimum execution time or resource usage for a given algorithm. Example: finding the target at the first position in linear search.

### Big-O Notation (O)
Upper bound notation. f(n) ∈ O(g(n)) means f(n) grows at most as fast as g(n) for sufficiently large n. Formally: ∃c > 0, n₀ ≥ 1 such that f(n) ≤ c·g(n) for all n ≥ n₀.

### Big-Omega Notation (Ω)
Lower bound notation. f(n) ∈ Ω(g(n)) means f(n) grows at least as fast as g(n) for sufficiently large n. Formally: ∃c > 0, n₀ ≥ 1 such that f(n) ≥ c·g(n) for all n ≥ n₀.

### Big-Theta Notation (Θ)
Tight bound notation. f(n) ∈ Θ(g(n)) means f(n) ∈ O(g(n)) and f(n) ∈ Ω(g(n)). The function grows at the same rate as g(n).

---

## C

### Cache
A small, fast memory that stores frequently accessed data to reduce average access time. Modern CPUs have multiple cache levels (L1, L2, L3).

### Cache Hit
When requested data is found in the cache, avoiding slower main memory access.

### Cache Line
The unit of data transfer between cache and main memory. Typically 64 bytes on modern processors.

### Cache Miss
When requested data is not in the cache, requiring access to slower memory levels.

### Cache-Oblivious Algorithm
An algorithm that achieves optimal cache performance without knowing cache parameters (size, line size). Uses recursive divide-and-conquer to automatically adapt to the memory hierarchy.

### Complexity Class
A category of problems grouped by their computational resource requirements. Examples: P, NP, PSPACE.

### Constant Factor
The multiplicative constant hidden by asymptotic notation. While O(n) and O(2n) are equivalent asymptotically, the constant factor affects practical performance.

### Constant Time — O(1)
An operation whose execution time does not depend on input size. Example: array access by index.

---

## D

### Divide and Conquer
An algorithm design paradigm that solves problems by breaking them into smaller subproblems, solving recursively, then combining results. Examples: merge sort, quicksort.

### Dominant Term
The term in a complexity expression that grows fastest and determines asymptotic behaviour. In 3n² + 5n + 7, the dominant term is 3n².

### Dynamic Array
A resizable array that automatically grows when capacity is exceeded. Append operations have O(1) amortised time despite occasional O(n) resize operations.

---

## E

### Empirical Complexity
Complexity estimated experimentally by measuring execution time for various input sizes and fitting to theoretical models.

### Exponential Time — O(2ⁿ)
Time complexity that doubles with each unit increase in input size. Generally considered intractable for large inputs. Example: brute-force subset enumeration.

---

## F

### Factorial Time — O(n!)
Time complexity that grows as the factorial of input size. Example: brute-force travelling salesman problem.

### False Sharing
A performance problem in parallel computing where threads on different cores modify independent variables that share a cache line, causing unnecessary cache invalidation.

---

## G

### Growth Rate
How quickly a function's value increases as its input increases. Central to comparing algorithm efficiency.

---

## H

### Harmonic Series
The sum H_n = 1 + 1/2 + 1/3 + ... + 1/n = Θ(log n). Appears in the analysis of quicksort's average case and hash table operations.

---

## I

### In-Place Algorithm
An algorithm that uses O(1) auxiliary space beyond the input. Example: in-place quicksort.

### Input Size
A measure of the amount of data an algorithm processes, typically denoted n. May represent array length, number of nodes, digits in a number, etc.

---

## L

### Linear Time — O(n)
Time complexity proportional to input size. Example: summing array elements.

### Linearithmic Time — O(n log n)
Time complexity of n multiplied by log n. Optimal for comparison-based sorting. Examples: merge sort, heapsort.

### Little-o Notation (o)
Strict upper bound. f(n) ∈ o(g(n)) means f(n) grows strictly slower than g(n). Formally: lim(n→∞) f(n)/g(n) = 0.

### Little-omega Notation (ω)
Strict lower bound. f(n) ∈ ω(g(n)) means f(n) grows strictly faster than g(n). Formally: lim(n→∞) f(n)/g(n) = ∞.

### Logarithmic Time — O(log n)
Time complexity proportional to the logarithm of input size. Characteristic of algorithms that halve the problem size each step. Example: binary search.

### Log-Log Plot
A graph with logarithmic scales on both axes. Power-law relationships f(n) = anᵇ appear as straight lines with slope b.

---

## M

### Master Theorem
A formula for solving recurrence relations of the form T(n) = aT(n/b) + f(n). Provides the complexity for many divide-and-conquer algorithms.

### Memory Hierarchy
The organisation of computer memory into levels of increasing size and decreasing speed: registers → L1 → L2 → L3 → RAM → SSD → HDD.

---

## N

### NUMA (Non-Uniform Memory Access)
A memory architecture where memory access time depends on memory location relative to the processor. Common in multi-socket systems.

---

## O

### Operation Count
The number of basic operations (comparisons, assignments, arithmetic) performed by an algorithm. Used to derive complexity.

---

## P

### Polynomial Time
Time complexity bounded by some polynomial function of input size. Generally considered tractable. Examples: O(n), O(n²), O(n³).

### Potential Method
An amortised analysis technique that defines a potential function Φ on data structure states. Amortised cost = actual cost + ΔΦ.

### Profiling
The process of measuring program execution to identify performance bottlenecks. Includes time profiling, memory profiling and line-by-line analysis.

---

## Q

### Quadratic Time — O(n²)
Time complexity proportional to the square of input size. Example: bubble sort, comparing all pairs.

---

## R

### Recurrence Relation
An equation defining a function in terms of its values at smaller inputs. Example: T(n) = 2T(n/2) + n for merge sort.

### Recursion Tree
A tree diagram showing the recursive calls of an algorithm. Used to visualise and solve recurrence relations.

---

## S

### Space Complexity
The amount of memory an algorithm requires as a function of input size. Includes both auxiliary space and input space.

### Spatial Locality
The tendency for memory accesses to be near recently accessed locations. Exploited by caches through cache line prefetching.

### Stable Sort
A sorting algorithm that preserves the relative order of equal elements. Example: merge sort is stable; heapsort is not.

---

## T

### Temporal Locality
The tendency for recently accessed memory locations to be accessed again soon. Exploited by keeping recent data in cache.

### Theoretical Complexity
Complexity derived mathematically from algorithm analysis, as opposed to empirical measurement.

### Tight Bound
An asymptotic bound that is both an upper and lower bound. Expressed using Big-Θ notation.

### Time Complexity
The amount of time an algorithm takes as a function of input size. Usually expressed in asymptotic notation.

### Tractable
A problem that can be solved in polynomial time. Informally, a problem with a "reasonable" algorithm.

---

## W

### Warmup
Initial iterations of a benchmark that are discarded to allow JIT compilation, cache population and other startup effects to stabilise.

### Worst Case
The scenario that results in the maximum execution time or resource usage for a given algorithm. Provides a guaranteed upper bound on performance.

---

## Symbols

### n
The conventional variable for input size.

### c
A positive constant used in formal definitions of asymptotic notation.

### n₀
The threshold beyond which an asymptotic bound holds.

### log n
Logarithm base 2 by convention in computer science unless otherwise specified.

### Σ (Sigma)
Summation notation. Σᵢ₌₁ⁿ f(i) = f(1) + f(2) + ... + f(n).

### Π (Pi)
Product notation. Πᵢ₌₁ⁿ f(i) = f(1) × f(2) × ... × f(n).

### ∈ (Element of)
Set membership. f(n) ∈ O(g(n)) means f(n) is a member of the set of functions bounded by g(n).

### ∀ (For all)
Universal quantifier. "For all n ≥ n₀" means the statement holds for every value of n at least n₀.

### ∃ (There exists)
Existential quantifier. "There exists c > 0" means at least one such value exists.

---

## Complexity Classes Summary

| Class | Definition | Example Algorithm |
|-------|------------|-------------------|
| O(1) | Constant | Array access |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Linear search |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Bubble sort |
| O(n³) | Cubic | Matrix multiplication |
| O(2ⁿ) | Exponential | Subset enumeration |
| O(n!) | Factorial | Permutation enumeration |

---

*For detailed explanations, see `lecture_notes.md` and `further_reading.md`.*

© 2025 Antonio Clim. All rights reserved.
