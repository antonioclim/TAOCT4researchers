# ═══════════════════════════════════════════════════════════════════════════════
# Week 4 Quiz: Advanced Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

> **Duration**: 20 minutes  
> **Total Points**: 30  
> **Format**: 6 Multiple Choice + 4 Short Answer

---

## Multiple Choice Questions (3 points each)

### Question 1: Hash Table Complexity

What is the **average-case** time complexity for searching in a hash table with a good hash function and load factor α < 0.75?

- A) O(1)
- B) O(log n)
- C) O(n)
- D) O(n log n)

---

### Question 2: Graph Representation Trade-offs

For a **sparse** graph with V vertices and E edges where E << V², which representation is more memory-efficient?

- A) Adjacency matrix
- B) Adjacency list
- C) Both use the same space
- D) Edge list

---

### Question 3: BFS vs DFS

Which traversal algorithm guarantees finding the **shortest path** in an **unweighted** graph?

- A) Depth-First Search (DFS)
- B) Breadth-First Search (BFS)
- C) Both guarantee shortest paths
- D) Neither guarantees shortest paths

---

### Question 4: Bloom Filter Properties

Which statement about Bloom filters is **TRUE**?

- A) Bloom filters can produce false negatives
- B) Bloom filters can produce false positives
- C) Bloom filters support deletion of elements
- D) Bloom filters store the actual elements

---

### Question 5: Dijkstra's Algorithm Limitation

Dijkstra's algorithm does NOT work correctly when the graph contains:

- A) Cycles
- B) Negative edge weights
- C) Disconnected components
- D) Self-loops

---

### Question 6: A* Heuristic Property

For A* to guarantee finding the optimal path, the heuristic function h(n) must be:

- A) Consistent (monotonic)
- B) Admissible (never overestimates)
- C) Exactly equal to the true distance
- D) Greater than zero for all nodes

---

## Short Answer Questions (3 points each)

### Question 7: Load Factor

A hash table has 100 buckets and contains 75 elements.

**Calculate the load factor and explain what happens when it exceeds 1.0.**

*Your answer (2-3 sentences):*

```
[Write your answer here]
```

---

### Question 8: Graph Algorithm Selection

You need to find the shortest path between two cities in a weighted road network with 10,000 cities. 

**Which algorithm would you choose: BFS, DFS, or Dijkstra? Justify your choice.**

*Your answer (2-3 sentences):*

```
[Write your answer here]
```

---

### Question 9: Bloom Filter Parameters

A Bloom filter is configured with m = 1000 bits and k = 7 hash functions.

**What happens to the false positive rate if we double m to 2000 bits while keeping k constant?**

*Your answer (2-3 sentences):*

```
[Write your answer here]
```

---

### Question 10: Topological Sort Application

**Give one real-world example where topological sort is essential, and explain why the ordering matters.**

*Your answer (2-3 sentences):*

```
[Write your answer here]
```

---

## Answer Key

<details>
<summary>Click to reveal answers</summary>

### Multiple Choice Answers

1. **A) O(1)** — With a good hash function and reasonable load factor, hash table operations are constant time on average.

2. **B) Adjacency list** — Adjacency lists use O(V + E) space, while matrices use O(V²). For sparse graphs where E << V², lists are more efficient.

3. **B) Breadth-First Search (BFS)** — BFS explores nodes level by level, guaranteeing the first path found to any node is the shortest (in terms of edge count).

4. **B) Bloom filters can produce false positives** — If all k hash positions are set, an element might falsely appear to be in the set. However, if any position is 0, the element is definitely not present (no false negatives).

5. **B) Negative edge weights** — Dijkstra's greedy approach assumes once a node is processed, we've found the shortest path. Negative edges can invalidate this assumption.

6. **B) Admissible (never overestimates)** — An admissible heuristic ensures A* never prunes the optimal path. Consistency is sufficient but not necessary for optimality.

### Short Answer Rubric

**Question 7** (3 points):
- Load factor = 75/100 = 0.75 (1 point)
- When α > 1.0, there are more elements than buckets (1 point)
- This increases collisions significantly, degrading performance towards O(n) (1 point)

**Question 8** (3 points):
- Dijkstra's algorithm (1 point)
- BFS doesn't handle weighted edges (1 point)
- Dijkstra finds shortest weighted paths in O((V+E) log V) (1 point)

**Question 9** (3 points):
- The false positive rate will decrease (1 point)
- More bits means lower probability that all k positions are set by chance (1 point)
- Approximately halving the rate: p ≈ (1-e^(-kn/m))^k (1 point)

**Question 10** (3 points):
- Valid example: build systems, course prerequisites, task scheduling, package dependencies (1 point)
- Ordering ensures prerequisites are completed before dependents (1 point)
- Violating order causes failures or undefined behaviour (1 point)

</details>

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*  
*Week 4 — Advanced Data Structures*

© 2025 Antonio Clim. All rights reserved.
