# ═══════════════════════════════════════════════════════════════════════════════
# Week 4: Lecture Notes
# Advanced Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

## Introduction

Data structures are the scaffolding upon which efficient algorithms are built.
This week, we explore advanced structures that enable researchers to tackle
problems at scale: graphs for modelling relationships, hash tables for rapid
lookup and probabilistic structures for approximate computation when exactness
is either impossible or prohibitively expensive.

The fundamental insight driving this week is that the choice of data structure
is often more consequential than the choice of algorithm. A well-chosen
structure can reduce complexity by orders of magnitude, whilst a poor choice
can render even the cleverest algorithm impractical.

---

## Part I: Hash Tables

### The Promise of Constant Time

Hash tables offer something remarkable: O(1) average-case lookup, insertion
and deletion. This enables efficient implementation of dictionaries, sets and
caches that would otherwise require O(log n) operations with tree-based
structures.

The mechanism is conceptually simple. A hash function maps keys to indices in
an array, allowing direct access without searching. However, this simplicity
conceals significant subtlety in handling collisions, managing load and
ensuring good hash distribution.

### Hash Functions

A good hash function must satisfy several properties. It must be deterministic,
producing the same output for the same input. It should distribute keys
uniformly across the output range to minimise collisions. It must be efficient
to compute, typically O(k) where k is the key size. For security-sensitive
applications, it should also be resistant to adversarial inputs.

In Python, the built-in `hash()` function suffices for most purposes, but
understanding its limitations is crucial. For strings, Python uses SipHash,
a cryptographically-inspired function that provides good distribution and
protection against hash-flooding attacks.

For custom objects, you must implement `__hash__()` alongside `__eq__()`. The
invariant is fundamental: objects that compare equal must have equal hashes.
The converse need not hold—collisions are permitted, merely discouraged.

### Collision Resolution

When two keys hash to the same index, we must resolve the collision. Two
primary strategies exist.

**Chaining** stores colliding elements in a linked list at each bucket. This
approach is simple and degrades gracefully under high load, with expected
chain length equal to the load factor. However, it incurs pointer overhead
and suffers from poor cache locality.

**Open addressing** seeks an alternative slot through probing. Linear probing
checks consecutive slots, offering good cache performance but suffering from
clustering. Quadratic probing reduces clustering at the cost of more complex
index calculation. Double hashing uses a second hash function to determine
probe distance, providing the best distribution but requiring two hash
computations.

### Load Factor and Resizing

The load factor α = n/m relates the number of elements n to the table size m.
Performance degrades as α increases. For chaining, expected lookup time is
O(1 + α). For open addressing, the relationship is more severe, with
performance degrading rapidly as α approaches 1.

Practical implementations maintain α below 0.75 (Python) or 0.70 (Java),
triggering a resize when exceeded. Resizing involves allocating a larger
table and rehashing all elements—an O(n) operation amortised over insertions.

---

## Part II: Trees and Priority Queues

### Binary Heaps

A binary heap is a complete binary tree satisfying the heap property: each
node is smaller (min-heap) or larger (max-heap) than its children. This
enables O(1) access to the minimum element whilst maintaining O(log n)
insertion and extraction.

The elegant insight is that completeness permits array representation without
explicit pointers. For a node at index i, its children reside at 2i+1 and
2i+2, its parent at (i-1)//2. This provides excellent cache locality and
eliminates pointer overhead.

Insertion appends to the array and "bubbles up" to restore the heap property.
Extraction removes the root, replaces it with the last element and "bubbles
down". Both operations traverse at most the tree height, O(log n).

### Priority Queues for Graph Algorithms

Dijkstra's algorithm and Prim's minimum spanning tree algorithm both require
a priority queue supporting efficient decrease-key operations. The naive
approach—reinserting with updated priority—creates duplicate entries and
wastes memory.

An indexed priority queue maintains a mapping from keys to heap positions,
enabling O(log n) decrease-key through direct access and bubble-up. This
refinement is essential for achieving the theoretical O((V + E) log V)
complexity of Dijkstra with sparse graphs.

---

## Part III: Graphs

### Representations

Graphs model pairwise relationships and appear throughout research: social
networks, protein interactions, citation graphs, road networks and dependency
structures.

The **adjacency list** representation stores, for each vertex, a list of its
neighbours. Space complexity is O(V + E), optimal for sparse graphs. Neighbour
iteration is O(degree), whilst edge existence checking is also O(degree).

The **adjacency matrix** uses a V×V matrix where entry (i,j) indicates edge
presence or weight. Space is O(V²), making it suitable only for dense graphs.
Edge existence checking is O(1), a significant advantage when this operation
dominates.

The choice between representations depends on graph density and access patterns.
Sparse graphs (E << V²) favour adjacency lists. Dense graphs or algorithms
requiring frequent edge existence queries favour matrices.

### Traversal Algorithms

**Breadth-first search** explores vertices level by level, using a queue to
maintain the frontier. It discovers shortest paths in unweighted graphs and
determines connectivity. The complexity is O(V + E) since each vertex and edge
is examined once.

**Depth-first search** explores as deeply as possible before backtracking,
using a stack (explicit or via recursion). It is fundamental to cycle detection,
topological sorting and finding strongly connected components. Complexity is
again O(V + E).

The traversal order difference has profound algorithmic implications. BFS
finds shortest paths; DFS reveals structural properties like cycles and
components.

### Shortest Path Algorithms

**Dijkstra's algorithm** finds shortest paths from a source to all vertices
in graphs with non-negative weights. It maintains a priority queue of vertices
ordered by tentative distance, repeatedly extracting the minimum and relaxing
outgoing edges.

With a binary heap, complexity is O((V + E) log V). The algorithm fails with
negative weights—it assumes that once a vertex is extracted, its shortest
distance is final.

**A*** extends Dijkstra with a heuristic function h(v) estimating the distance
from v to the goal. The priority becomes f(v) = g(v) + h(v), where g(v) is
the known distance from source to v. When h is admissible (never overestimates)
and consistent, A* is optimal and typically explores fewer vertices than
Dijkstra.

### Topological Sorting

A topological sort orders vertices of a directed acyclic graph (DAG) such that
for every edge (u,v), u precedes v. Applications include task scheduling, build
systems and course prerequisites.

Kahn's algorithm repeatedly removes vertices with zero in-degree, adding them
to the result. If the graph contains cycles, some vertices remain when the
algorithm terminates—a useful cycle detection method.

---

## Part IV: Probabilistic Data Structures

### The Case for Approximation

Exact membership testing—"Is x in set S?"—requires Ω(n) space in the worst
case. When processing billions of elements, exact methods become impractical.
Probabilistic structures trade exactness for dramatic space savings.

The key insight is that many applications tolerate false positives. A cache
can afford to occasionally check for items that aren't present. A spell
checker can accept occasional false suggestions. What matters is controlling
and understanding the error rate.

### Bloom Filters

A Bloom filter is a bit array of m bits, initially all zero. To insert element
x, we compute k hash functions h₁(x), ..., hₖ(x) and set those bit positions.
To query, we check if all k positions are set—if any is zero, x is definitely
absent; if all are set, x is probably present.

The false positive probability depends on the number of elements n, bits m
and hash functions k:

    p ≈ (1 - e^(-kn/m))^k

The optimal number of hash functions is k = (m/n) · ln(2), yielding:

    p_optimal ≈ (1/2)^k = (0.6185)^(m/n)

For 1% false positive rate, we need approximately 10 bits per element.

### Count-Min Sketch

The Count-Min sketch extends Bloom filters to frequency estimation. It
maintains d arrays of w counters each. To increment element x, we add 1 to
position hᵢ(x) in each array i. To query frequency, we return the minimum
across all d arrays.

The estimate is never below the true count (no false negatives for frequency)
but may overestimate due to collisions. With probability at least 1-δ, the
overestimate is at most εn, where ε = e/w and δ = e^(-d).

Applications include network traffic analysis, database query optimisation
and streaming algorithms where storing exact counts is infeasible.

### HyperLogLog

Counting distinct elements exactly requires O(n) space. HyperLogLog achieves
approximate cardinality estimation using O(log log n) space with remarkable
accuracy—typically within 2% using just 1.5 KB.

The insight is probabilistic: if we hash elements uniformly, the maximum
number of leading zeros observed in any hash tells us about the cardinality.
HyperLogLog refines this with multiple estimators and harmonic mean averaging.

---

## Part V: Practical Considerations

### Cache Efficiency

Modern CPUs access memory through a hierarchy of caches. Accessing L1 cache
is approximately 1 ns; main memory is 100 ns. Data structures with good
locality—arrays, heaps, hash tables with open addressing—outperform pointer-
heavy structures like linked lists and trees.

When choosing between theoretically equivalent structures, favour those with
contiguous memory layout. A hash table with chaining may have better worst-
case bounds than open addressing, but the latter often wins in practice due
to cache effects.

### Memory Allocation

Dynamic allocation (malloc, new) is expensive—often 100+ cycles. Structures
requiring frequent small allocations suffer accordingly. Arena allocators,
object pools and pre-allocation can dramatically improve performance when
allocation patterns are predictable.

For research applications processing large datasets, memory layout often
dominates runtime. Profile actual memory access patterns before optimising.

### When to Use Each Structure

| Structure | Use When |
|-----------|----------|
| Hash table | Fast lookup by key, no ordering needed |
| Tree | Ordered iteration, range queries |
| Heap | Priority queue operations dominate |
| Graph (list) | Sparse graph, traversal-heavy |
| Graph (matrix) | Dense graph, frequent edge queries |
| Bloom filter | Membership tests, space critical |
| Count-Min | Frequency estimation, streaming |

---

## Research Applications

### Social Network Analysis

Graph algorithms reveal community structure, influential nodes and information
flow patterns. Clustering coefficients, centrality measures and shortest
paths all rely on efficient graph representations.

### Bioinformatics

Protein interaction networks, metabolic pathways and phylogenetic trees are
all graphs. Sequence databases use Bloom filters for fast pre-filtering before
expensive alignment.

### Big Data Processing

Stream processing systems cannot store all data. Probabilistic structures
enable approximate aggregations—distinct counts, heavy hitters, quantiles—
with bounded memory.

---

## Summary

This week introduced data structures that enable efficient computation at
scale. Hash tables provide constant-time operations through clever collision
handling. Heaps support priority queue operations essential for graph
algorithms. Graphs model relationships central to countless research domains.
Probabilistic structures sacrifice exactness for dramatic space savings when
approximation suffices.

The unifying theme is trade-offs: time versus space, exactness versus
approximation, theory versus practice. Mastering these structures means
understanding not just how they work, but when each is appropriate.

---

## Key Takeaways

1. **Hash tables** achieve O(1) average-case operations through uniform
   distribution and careful load factor management.

2. **Binary heaps** enable efficient priority queues with array-based
   implementation providing excellent cache locality.

3. **Graph representation** choice depends on density and access patterns—
   adjacency lists for sparse, matrices for dense.

4. **Dijkstra and A*** find shortest paths; A* uses heuristics to reduce
   exploration.

5. **Bloom filters** test membership in O(k) time and O(m) space with
   controllable false positive rates.

6. **Probabilistic structures** trade exactness for space when approximation
   is acceptable.

---

## Looking Ahead

Week 5 applies these structures to scientific computing. Efficient data
structures enable large-scale simulations—Monte Carlo methods, ODE solvers
and agent-based models. The graph algorithms from this week support agent
interaction networks, whilst probabilistic counting enables streaming
analytics in simulations producing vast data volumes.

---

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
