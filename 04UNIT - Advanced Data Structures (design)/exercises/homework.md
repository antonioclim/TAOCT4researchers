# 04UNIT Homework: Advanced Data Structures

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | Friday 23:59 GMT |
| **Total Points** | 100 |
| **Estimated Time** | 4-5 hours |
| **Difficulty** | ⭐⭐⭐⭐ (4/5) |

## 🔗 Prerequisites

- [ ] Completed Lab 4.1 (Graph Library)
- [ ] Completed Lab 4.2 (Probabilistic Data Structures)
- [ ] Read lecture notes
- [ ] Reviewed 03UNIT complexity analysis

## 🎯 Objectives Assessed

1. Implement Bloom filters with optimal parameter calculation
2. Build an LRU cache with O(1) operations
3. Create an indexed priority queue supporting decrease-key
4. Apply graph algorithms to real-world routing problems

---

## Part 1: Bloom Filter Implementation (25 points)

### Context

Bloom filters are used in production systems worldwide:
- **Chrome**: Checking URLs against malicious site databases without sending the URL to Google
- **Cassandra/HBase**: Avoiding expensive disk reads for non-existent keys
- **Medium**: Tracking which articles a user has read
- **Spell checkers**: Fast dictionary membership testing

### Requirements

Implement a Bloom filter with the following interface:

```python
class BloomFilter:
    def __init__(self, expected_items: int, false_positive_rate: float):
        """
        Initialise a Bloom filter.
        
        Automatically calculate:
        - m (number of bits) = -n·ln(p) / (ln(2)²)
        - k (number of hash functions) = (m/n) · ln(2)
        """
        ...
    
    def add(self, item: str) -> None:
        """Add an item to the filter."""
        ...
    
    def __contains__(self, item: str) -> bool:
        """Check if an item MIGHT be in the filter."""
        ...
    
    def estimated_false_positive_rate(self) -> float:
        """Estimate current FP rate based on fill ratio."""
        ...
    
    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        """Return the union of two filters (bitwise OR)."""
        ...
```

### Detailed Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 1.1 | 10 | Basic implementation using `hashlib` for hashing. Generate k hashes using double hashing: h(x, i) = h₁(x) + i·h₂(x) |
| 1.2 | 10 | Empirical validation: verify FP rate formula with 10,000 insertions and 100,000 queries |
| 1.3 | 5 | Practical application: spell checker using Bloom filter with dictionary |

### Test Cases

```python
bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)

# Add items
for i in range(1000):
    bf.add(f"item_{i}")

# True positives
assert "item_0" in bf
assert "item_999" in bf

# False negatives must never occur
for i in range(1000):
    assert f"item_{i}" in bf, "False negative detected!"

# FP rate should be close to target
false_positives = sum(1 for i in range(10000) if f"notitem_{i}" in bf)
assert false_positives < 200, f"FP rate too high: {false_positives/10000:.2%}"
```

### Hints

<details>
<summary>💡 Hint 1: Hash Function</summary>

Use SHA-256 to generate two independent hashes:
```python
import hashlib

def get_hashes(item: str) -> tuple[int, int]:
    h = hashlib.sha256(item.encode()).digest()
    h1 = int.from_bytes(h[:8], 'big')
    h2 = int.from_bytes(h[8:16], 'big')
    return h1, h2
```
</details>

<details>
<summary>💡 Hint 2: Optimal Parameters</summary>

For a target false positive rate p with n expected items:
```python
import math

m = int(-n * math.log(p) / (math.log(2) ** 2))
k = max(1, int((m / n) * math.log(2)))
```
</details>

---

## Part 2: LRU Cache with O(1) Operations (25 points)

### Context

LRU (Least Recently Used) Cache evicts the least recently accessed item when
the cache is full. This policy is used in:
- Browser caches
- Database query caches
- CPU cache replacement policies
- CDN content caching

### Requirements

Implement an LRU Cache with ALL operations in O(1):

```python
class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int):
        """Initialise cache with fixed capacity."""
        ...
    
    def get(self, key: K) -> V | None:
        """
        Return value for key or None.
        Marks key as recently used.
        Complexity: O(1)
        """
        ...
    
    def put(self, key: K, value: V) -> None:
        """
        Add or update key-value pair.
        Evicts LRU item if over capacity.
        Complexity: O(1)
        """
        ...
    
    def __len__(self) -> int:
        ...
    
    def __contains__(self, key: K) -> bool:
        """Check existence WITHOUT modifying recency."""
        ...
```

### Data Structure Hint

```
Combine:
- dict[K, Node] for O(1) lookup
- Doubly-linked list for O(1) move-to-front and remove

┌─────────────────────────────────────────────────────┐
│  HEAD ←→ [A] ←→ [B] ←→ [C] ←→ TAIL                 │
│           ↑      ↑      ↑                           │
│  dict: {"A": ●, "B": ●, "C": ●}                     │
└─────────────────────────────────────────────────────┘
```

### Test Cases

```python
cache = LRUCache[str, int](2)
cache.put("a", 1)
cache.put("b", 2)
assert cache.get("a") == 1      # "a" becomes most recent
cache.put("c", 3)               # Evicts "b" (LRU), not "a"
assert cache.get("b") is None   # "b" was evicted
assert cache.get("a") == 1      # "a" still present
assert cache.get("c") == 3      # "c" present
```

| Req | Points | Description |
|-----|--------|-------------|
| 2.1 | 15 | Basic LRU implementation with O(1) operations |
| 2.2 | 5 | Support for `__contains__` without modifying recency |
| 2.3 | 5 | Thread-safe variant using `threading.Lock` |

---

## Part 3: Indexed Priority Queue (25 points)

### Context

Algorithms like Dijkstra and Prim require a priority queue that efficiently
supports the decrease-key operation (updating the priority of an element).

### Requirements

Implement a priority queue based on binary heap with:

```python
class IndexedPriorityQueue(Generic[K]):
    """
    Min-heap with support for:
    - insert: O(log n)
    - extract_min: O(log n)
    - decrease_key: O(log n)
    - contains: O(1)
    - get_priority: O(1)
    """
    
    def __init__(self) -> None:
        self._heap: list[tuple[float, K]] = []
        self._index: dict[K, int] = {}  # key → position in heap
    
    def insert(self, key: K, priority: float) -> None:
        """Add element with priority."""
        ...
    
    def extract_min(self) -> tuple[K, float]:
        """Extract and return element with minimum priority."""
        ...
    
    def decrease_key(self, key: K, new_priority: float) -> None:
        """
        Decrease priority of an element.
        Raises KeyError if key does not exist.
        Raises ValueError if new_priority > current priority.
        """
        ...
    
    def __contains__(self, key: K) -> bool:
        ...
    
    def __len__(self) -> int:
        ...
```

### Application: Optimised Dijkstra

Use IndexedPriorityQueue to implement efficient Dijkstra:

```python
def dijkstra_optimised(graph, start):
    """
    Dijkstra with decrease-key.
    Complexity: O((V + E) log V) without heap duplicates.
    """
    ...
```

| Req | Points | Description |
|-----|--------|-------------|
| 3.1 | 15 | Indexed priority queue with decrease-key support |
| 3.2 | 10 | Dijkstra implementation using the priority queue |

---

## Part 4: City Network Routing (25 points)

### Dataset

Create or find a dataset of UK cities with road distances.
Minimum: 30 cities, 100+ connections.

Possible sources:
- OpenStreetMap via osmnx
- Manual entry of major city distances
- National statistics datasets

### Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 4.1 | 5 | Load data into graph library from Lab 4.1. Add GPS coordinates for A* heuristic |
| 4.2 | 10 | Implement routing algorithms: Dijkstra (shortest path), A* with Euclidean heuristic, comparison of nodes visited |
| 4.3 | 10 | Visualisation: Map showing cities and connections, highlighted route, use matplotlib or folium |

### Example Output

```
Route: London → Edinburgh

Dijkstra:
  Distance: 640 km
  Nodes visited: 28
  Time: 2.3 ms
  Path: London → Leicester → Sheffield → Newcastle → Edinburgh

A*:
  Distance: 640 km
  Nodes visited: 12 (57% fewer!)
  Time: 1.1 ms
  Path: London → Leicester → Sheffield → Newcastle → Edinburgh
```

### Performance Analysis Requirements

For Part 4, provide a written analysis (minimum 200 words) addressing:

1. **Empirical comparison**: Run both Dijkstra and A* on at least 10 different source-destination pairs. Record nodes visited and execution time for each. Present results in a table and discuss patterns observed.

2. **Heuristic quality**: Explain why Euclidean distance is an admissible heuristic for geographic routing. Under what circumstances might it perform poorly? Consider cases where roads do not follow straight-line paths.

3. **Scalability considerations**: If the network grew to 10,000 cities with 50,000 connections, how would performance change? What algorithmic improvements might help?

---

## 💡 Additional Implementation Hints

### Hash Function Design

When implementing hash functions for Bloom filters, ensure independence between hash values. Two approaches work well:

1. **Double hashing**: Use two independent hash functions h₁ and h₂, then generate k hashes as h(x,i) = h₁(x) + i·h₂(x). This requires only two hash computations regardless of k.

2. **Salted hashing**: Hash the concatenation of the item with a unique salt for each hash function: h_i(x) = hash(salt_i || x). This is simpler but requires k hash computations.

### LRU Cache Node Structure

The doubly-linked list node should contain:
- The key (for O(1) removal from dict when evicting)
- The value (for O(1) retrieval)
- Pointers to previous and next nodes

Maintain sentinel head and tail nodes to simplify edge cases—you never insert at a null pointer.

### Priority Queue Index Maintenance

The trickiest part of IndexedPriorityQueue is maintaining the index mapping during heap operations. When swapping elements during bubble-up or bubble-down, update both elements' positions in the index dictionary. Forgetting this update is the most common bug.

---

## ⚠️ Midterm Preparation

### Topics to Review

**01UNIT:**
- Turing machines, computability
- Pattern matching, AST interpretation

**02UNIT:**
- Protocols, Generics
- Design by Contract
- Polymorphism

**03UNIT:**
- Asymptotic notation (O, Ω, Θ)
- Master theorem
- Profiling and optimisation

**04UNIT:**
- Hash tables (collision resolution, load factor)
- Balanced trees (AVL, B-trees)
- Graphs (representations, BFS, DFS, Dijkstra)
- Probabilistic structures (Bloom filters)

### Exam Format

- Duration: 2 hours
- Open-book
- Practical problem (implementation + analysis)

---

## ✅ Submission Checklist

- [ ] All tests pass (`pytest`)
- [ ] Code formatted with ruff (`ruff format`)
- [ ] Type hints complete (`mypy --strict`)
- [ ] Docstrings present (Google style)
- [ ] No print statements (use logging)

---

## 📊 Grading Rubric

| Category | Points | Criteria |
|----------|--------|----------|
| **Correctness** | 40 | All test cases pass |
| **Code Quality** | 25 | Type hints, docstrings, no linting errors |
| **Analysis** | 20 | Complexity analysis, empirical validation |
| **Documentation** | 15 | Clear explanations, comments |

---

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
