# 04UNIT Cheatsheet: Advanced Data Structures

> **One-Page Reference** | The Art of Computational Thinking for Researchers

---

## Key Concepts at a Glance

| Structure | Access | Search | Insert | Delete | Space | Use Case |
|-----------|--------|--------|--------|--------|-------|----------|
| Hash Table | O(1)* | O(1)* | O(1)* | O(1)* | O(n) | Fast key-value lookup |
| Binary Heap | O(1) top | O(n) | O(log n) | O(log n) | O(n) | Priority queues |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) | O(n) | Ordered data |
| Graph (adj. list) | O(1) | O(V+E) | O(1) | O(E) | O(V+E) | Network modelling |
| Bloom Filter | — | O(k) | O(k) | — | O(m) | Membership testing |
| Count-Min Sketch | — | O(d) | O(d) | — | O(w×d) | Frequency estimation |

*\*Amortised average case*

---

## Essential Formulas

### Bloom Filter Parameters
```
m = -(n × ln(p)) / (ln(2))²    # bits needed
k = (m/n) × ln(2)              # optimal hash functions
p = (1 - e^(-kn/m))^k          # false positive rate
```

### Count-Min Sketch Parameters
```
w = ⌈e/ε⌉                      # width (e ≈ 2.718)
d = ⌈ln(1/δ)⌉                  # depth (number of rows)
```

### Graph Complexity
```
BFS/DFS: O(V + E) time, O(V) space
Dijkstra: O((V + E) log V) with binary heap
A*: O(E) best case, O(b^d) worst case
Topological Sort: O(V + E)
```

---

## Code Patterns

### 1. Graph Representation (Adjacency List)
```python
from collections import defaultdict
graph: dict[str, list[tuple[str, float]]] = defaultdict(list)
graph[u].append((v, weight))  # Add edge u → v
```

### 2. BFS Template
```python
from collections import deque
def bfs(graph: dict, start: str) -> list[str]:
    visited, queue = {start}, deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour, _ in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order
```

### 3. DFS Template (Iterative)
```python
def dfs(graph: dict, start: str) -> list[str]:
    visited, stack = set(), [start]
    order = []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            order.append(node)
            for neighbour, _ in reversed(graph[node]):
                stack.append(neighbour)
    return order
```

### 4. Dijkstra's Algorithm
```python
import heapq
def dijkstra(graph: dict, start: str) -> dict[str, float]:
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in graph[u]:
            if d + w < dist.get(v, float('inf')):
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

### 5. Bloom Filter Insertion
```python
def add(self, item: str) -> None:
    for i in range(self.k):
        idx = self._hash(item, i) % self.m
        self.bits[idx] = 1
```

---

## Common Mistakes to Avoid

| ❌ Mistake | ✅ Correct Approach |
|-----------|---------------------|
| Using adjacency matrix for sparse graphs | Use adjacency list: O(V+E) vs O(V²) space |
| Forgetting visited set in BFS/DFS | Always track visited nodes to avoid cycles |
| Using `list.pop(0)` for BFS | Use `collections.deque.popleft()` for O(1) |
| Negative weights with Dijkstra | Use Bellman-Ford for negative weights |
| Deleting from Bloom filter | Bloom filters don't support deletion; use counting variant |
| Treating Count-Min as exact | Always treat estimates as upper bounds |
| Rehashing entire hash table | Use incremental resizing or consistent hashing |

---

## Week Connections

| Previous (03UNIT) | Current (04UNIT) | Next (05UNIT) |
|-------------------|------------------|---------------|
| Big-O analysis | Apply to graph algorithms | Monte Carlo on graphs |
| Benchmarking | Measure data structure performance | Simulation frameworks |
| Profiling | Identify bottlenecks | ODE solver efficiency |

---

## Quick Reference: When to Use What

```
Need fast lookup?           → Hash Table
Need ordered traversal?     → BST / TreeMap
Need priority access?       → Binary Heap
Need network analysis?      → Graph (adjacency list)
Need approximate membership?→ Bloom Filter
Need frequency estimates?   → Count-Min Sketch
Space is critical?          → Probabilistic structures
```

---

## Edge Cases and Pitfalls

### Hash Table Edge Cases
```python
# Empty key handling
hash("")  # Valid, but consider if empty strings make sense

# Mutable keys are forbidden
d = {}
d[[1, 2, 3]] = "value"  # TypeError: unhashable type: 'list'

# Hash collisions with __eq__ but different __hash__
# Violates the invariant: equal objects must have equal hashes
```

### Graph Algorithm Edge Cases
```python
# Disconnected graphs
# BFS/DFS from one node won't visit disconnected components
# Solution: iterate over all nodes, start new traversal for unvisited

# Self-loops
# Depends on application: sometimes valid, sometimes error

# Parallel edges (multigraphs)
# Adjacency list handles naturally; matrix needs extension
```

### Bloom Filter Gotchas
```python
# Cannot delete from standard Bloom filter
# Solution: use Counting Bloom Filter (replaces bits with counters)

# False positive rate increases with insertions
# Monitor fill ratio; rebuild if too high

# Serialisation: bit array + parameters (m, k, hash seeds)
```

---

## Performance Tuning Tips

| Situation | Optimisation |
|-----------|--------------|
| Many small allocations | Use object pools or arena allocators |
| Poor cache performance | Switch to array-based structures |
| High hash collision rate | Improve hash function or increase table size |
| Frequent resizing | Pre-allocate with expected size |
| Deep recursion in DFS | Convert to iterative with explicit stack |

---

## Research Application Examples

- **Social Networks**: Graph algorithms for community detection
- **Bioinformatics**: Bloom filters for k-mer counting in DNA sequencing
- **Big Data**: Count-Min sketch for stream processing
- **Route Planning**: A* algorithm for pathfinding
- **Citation Networks**: PageRank via graph traversal

---

*© 2025 Antonio Clim. All rights reserved.*
