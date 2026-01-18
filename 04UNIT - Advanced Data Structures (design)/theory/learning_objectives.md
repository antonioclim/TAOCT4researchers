# ═══════════════════════════════════════════════════════════════════════════════
# 04UNIT: Learning Objectives
# Advanced Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

## Overview

This document defines the measurable learning objectives for 04UNIT, aligned with
Bloom's Taxonomy levels appropriate for postgraduate researchers.

---

## Primary Learning Objectives

### Objective 1: Graph Implementation and Algorithms

**Bloom Level**: Apply

**Statement**: Implement graph data structures with common algorithms including
breadth-first search, depth-first search, Dijkstra's algorithm and A* search.

**Assessment Criteria**:
- [ ] Correctly implement adjacency list and adjacency matrix representations
- [ ] Implement BFS traversal with correct ordering and complexity
- [ ] Implement DFS traversal (iterative and recursive variants)
- [ ] Implement Dijkstra's algorithm with proper priority queue usage
- [ ] Implement A* with admissible heuristic functions
- [ ] Demonstrate understanding of time and space complexity trade-offs

**Evidence of Mastery**:
- Working code that passes all unit tests
- Correct output for provided test cases
- Ability to explain algorithm behaviour step-by-step

---

### Objective 2: Data Structure Selection

**Bloom Level**: Analyse

**Statement**: Select appropriate data structures based on computational
requirements and use case constraints.

**Assessment Criteria**:
- [ ] Analyse problem requirements to identify key operations
- [ ] Compare complexity of operations across different structures
- [ ] Consider memory constraints and cache efficiency
- [ ] Evaluate sparse versus dense representation trade-offs
- [ ] Justify selection decisions with complexity analysis

**Evidence of Mastery**:
- Written justification for structure selection in homework
- Correct identification of optimal structure for given scenarios
- Ability to adapt choice when constraints change

---

### Objective 3: Probabilistic vs Deterministic Trade-offs

**Bloom Level**: Evaluate

**Statement**: Compare probabilistic versus deterministic structure trade-offs
in terms of space, time and accuracy.

**Assessment Criteria**:
- [ ] Explain false positive/negative rates in probabilistic structures
- [ ] Calculate optimal parameters for Bloom filters
- [ ] Evaluate when approximate answers are acceptable
- [ ] Compare memory usage between exact and approximate methods
- [ ] Assess accuracy guarantees and their implications

**Evidence of Mastery**:
- Correct calculation of Bloom filter parameters
- Empirical validation of theoretical false positive rates
- Appropriate recommendation for given use cases

---

## Secondary Learning Objectives

### Objective 4: Hash Table Internals

**Bloom Level**: Understand

**Statement**: Explain hash table implementation including collision resolution
strategies and load factor management.

**Assessment Criteria**:
- [ ] Describe chaining and open addressing strategies
- [ ] Explain load factor and its effect on performance
- [ ] Understand amortised analysis of dynamic resizing
- [ ] Recognise pathological inputs and hash function weaknesses

---

### Objective 5: Tree Structures

**Bloom Level**: Apply

**Statement**: Implement and use tree-based data structures including binary
search trees, heaps and priority queues.

**Assessment Criteria**:
- [ ] Implement binary heap operations (insert, extract-min)
- [ ] Use heaps for priority queue operations
- [ ] Understand heap property and its maintenance
- [ ] Apply decrease-key operation for graph algorithms

---

### Objective 6: Graph Properties

**Bloom Level**: Analyse

**Statement**: Analyse graph properties including connectivity, cycles and
topological ordering.

**Assessment Criteria**:
- [ ] Detect cycles in directed and undirected graphs
- [ ] Find connected components
- [ ] Compute topological sort for DAGs
- [ ] Identify strongly connected components

---

## Bloom's Taxonomy Mapping

| Level | Verb | 04UNIT Coverage |
|-------|------|-----------------|
| Remember | Define, list, identify | Hash function definition, graph terminology |
| Understand | Explain, describe | Collision resolution, traversal patterns |
| **Apply** | Implement, use | Graph algorithms, probabilistic structures |
| **Analyse** | Compare, examine | Structure selection, complexity analysis |
| **Evaluate** | Assess, justify | Probabilistic trade-offs, design decisions |
| Create | Design, construct | (05UNIT preparation) |

---

## Connection to Course Objectives

### Builds Upon (03UNIT)
- Complexity analysis skills applied to data structure operations
- Benchmarking methodology used to validate theoretical complexity
- Algorithm efficiency concepts extended to structure-specific operations

### Prepares For (05UNIT)
- Efficient data structures enable large-scale simulations
- Graph algorithms support agent-based model implementation
- Probabilistic methods connect to Monte Carlo techniques

---

## Assessment Methods

| Objective | Assessment Type | Weight |
|-----------|-----------------|--------|
| Graph Implementation | Lab exercises, homework | 30% |
| Structure Selection | Written justification | 25% |
| Probabilistic Trade-offs | Homework Part 1 | 25% |
| Quiz | Multiple choice + short answer | 20% |

---

## Self-Assessment Questions

Before proceeding to 05UNIT, ensure you can answer:

1. When would you choose adjacency list over adjacency matrix?
2. What is the time complexity of Dijkstra with a binary heap?
3. How do you calculate optimal Bloom filter parameters?
4. When might you accept false positives in exchange for space savings?
5. How does A* improve upon Dijkstra for pathfinding?

---

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
