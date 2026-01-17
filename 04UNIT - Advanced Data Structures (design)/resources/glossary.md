# 04UNIT Glossary: Advanced Data Structures

> **Terminology Reference** | The Art of Computational Thinking for Researchers

---

## Graph Theory

### Adjacency List
A graph representation where each vertex stores a list of its neighbouring vertices. Space complexity O(V + E), efficient for sparse graphs.

### Adjacency Matrix
A graph representation using a V × V matrix where entry (i, j) indicates an edge between vertices i and j. Space complexity O(V²), efficient for dense graphs.

### Bipartite Graph
A graph whose vertices can be divided into two disjoint sets such that every edge connects a vertex from one set to a vertex in the other. No edges exist within the same set.

### Breadth-First Search (BFS)
A graph traversal algorithm that explores all neighbours at the current depth before moving to vertices at the next depth level. Uses a queue data structure. Time complexity O(V + E).

### Connected Component
A maximal subgraph in which any two vertices are connected by a path. In directed graphs, we distinguish between strongly and weakly connected components.

### Cycle
A path in a graph that starts and ends at the same vertex with no repeated edges. A graph without cycles is called acyclic.

### Degree
The number of edges incident to a vertex. In directed graphs, we distinguish between in-degree (incoming edges) and out-degree (outgoing edges).

### Depth-First Search (DFS)
A graph traversal algorithm that explores as far as possible along each branch before backtracking. Uses a stack (explicit or via recursion). Time complexity O(V + E).

### Directed Acyclic Graph (DAG)
A directed graph with no directed cycles. DAGs have a topological ordering and are used to model dependencies and precedence relationships.

### Directed Graph (Digraph)
A graph where edges have a direction, going from one vertex (source) to another (target). Represented as ordered pairs (u, v).

### Edge
A connection between two vertices in a graph. May be directed (arc) or undirected, and may carry a weight or label.

### Eulerian Path
A path that visits every edge exactly once. A graph has an Eulerian path if and only if it has exactly zero or two vertices of odd degree.

### Graph
A mathematical structure consisting of a set of vertices (nodes) and a set of edges connecting pairs of vertices.

### Hamiltonian Path
A path that visits every vertex exactly once. Determining whether such a path exists is NP-complete.

### Path
A sequence of vertices where each adjacent pair is connected by an edge. The length of a path is the number of edges it contains.

### Shortest Path
A path between two vertices with the minimum sum of edge weights (or minimum number of edges in unweighted graphs).

### Sink
A vertex in a directed graph with no outgoing edges (out-degree = 0).

### Source
A vertex in a directed graph with no incoming edges (in-degree = 0).

### Spanning Tree
A subgraph that includes all vertices and is a tree (connected and acyclic). A minimum spanning tree has the minimum total edge weight.

### Strongly Connected Components (SCC)
Maximal subsets of vertices in a directed graph where every vertex is reachable from every other vertex in the same subset.

### Topological Sort
A linear ordering of vertices in a DAG such that for every directed edge (u, v), vertex u appears before vertex v in the ordering.

### Undirected Graph
A graph where edges have no direction; an edge between u and v can be traversed in either direction.

### Vertex (Node)
A fundamental unit of a graph representing an entity. Vertices may have associated data or labels.

### Weighted Graph
A graph where each edge has an associated numerical value (weight) representing cost, distance, capacity or other quantities.

---

## Graph Algorithms

### A* Algorithm
An informed search algorithm that uses heuristics to find the shortest path. Combines actual distance from start with estimated distance to goal: f(n) = g(n) + h(n).

### Bellman-Ford Algorithm
A single-source shortest path algorithm that handles negative edge weights. Time complexity O(V × E). Can detect negative cycles.

### Dijkstra's Algorithm
A single-source shortest path algorithm for graphs with non-negative edge weights. Uses a priority queue. Time complexity O((V + E) log V) with binary heap.

### Floyd-Warshall Algorithm
An all-pairs shortest path algorithm using dynamic programming. Time complexity O(V³). Works with negative weights but not negative cycles.

### Kahn's Algorithm
A BFS-based algorithm for topological sorting that repeatedly removes vertices with no incoming edges.

### Kruskal's Algorithm
A greedy algorithm for finding minimum spanning trees by selecting edges in order of increasing weight while avoiding cycles.

### Prim's Algorithm
A greedy algorithm for finding minimum spanning trees by growing a tree from an arbitrary starting vertex, always adding the minimum-weight edge to a new vertex.

---

## Hash-Based Structures

### Chaining
A collision resolution strategy where elements that hash to the same index are stored in a linked list at that index.

### Collision
When two different keys hash to the same index in a hash table. Resolved through chaining or open addressing.

### Hash Function
A function that maps data of arbitrary size to fixed-size values. Good hash functions distribute keys uniformly and are fast to compute.

### Hash Table
A data structure that implements an associative array abstract data type, mapping keys to values using a hash function for O(1) average-case operations.

### Load Factor
The ratio of the number of elements to the number of buckets in a hash table (n/m). Higher load factors increase collision probability.

### Open Addressing
A collision resolution strategy where all elements are stored in the hash table itself, probing for empty slots using linear, quadratic or double hashing.

### Perfect Hashing
A hash function that maps n keys to n distinct values with no collisions. Requires knowledge of all keys in advance.

### Rehashing
The process of creating a larger hash table and reinserting all elements when the load factor exceeds a threshold.

### Universal Hashing
A family of hash functions from which one is chosen randomly, providing probabilistic guarantees against adversarial inputs.

---

## Tree Structures

### AVL Tree
A self-balancing binary search tree where the heights of the two child subtrees of any node differ by at most one.

### B-Tree
A self-balancing tree structure optimised for systems that read and write large blocks of data. Used in databases and file systems.

### Binary Heap
A complete binary tree satisfying the heap property: each node is greater than or equal to (max-heap) or less than or equal to (min-heap) its children.

### Binary Search Tree (BST)
A binary tree where each node's left subtree contains only values less than the node's value, and the right subtree contains only greater values.

### Complete Binary Tree
A binary tree where all levels except possibly the last are completely filled, and all nodes are as far left as possible.

### Heap Property
The invariant that in a max-heap, every parent node is greater than or equal to its children (reversed for min-heap).

### Heapify
The operation of converting an array into a heap structure, or restoring the heap property after modification. Can be done in O(n) time.

### Priority Queue
An abstract data type where each element has a priority, and elements are served according to their priority rather than insertion order.

### Red-Black Tree
A self-balancing binary search tree with an extra bit per node for colour (red or black), ensuring the tree remains approximately balanced.

### Trie (Prefix Tree)
A tree structure for storing strings where each edge represents a character and paths from root to leaves represent complete strings.

---

## Probabilistic Data Structures

### Bloom Filter
A space-efficient probabilistic data structure for testing set membership. May return false positives but never false negatives.

### Cardinality Estimation
The problem of estimating the number of distinct elements in a dataset, often solved with probabilistic structures like HyperLogLog.

### Count-Min Sketch
A probabilistic data structure for frequency estimation in data streams. Provides upper-bound estimates with bounded error probability.

### Counting Bloom Filter
A variant of Bloom filter that uses counters instead of bits, supporting deletion operations at the cost of increased space.

### Cuckoo Filter
A probabilistic data structure similar to Bloom filters but supporting deletion and providing better space efficiency for applications with low false positive rates.

### False Negative
An error where a membership test incorrectly reports that an element is not in a set when it actually is. Bloom filters guarantee no false negatives.

### False Positive
An error where a membership test incorrectly reports that an element is in a set when it is not. Bloom filters have a tunable false positive rate.

### False Positive Rate (FPR)
The probability that a membership query returns true for an element not in the set. Denoted as p or ε, typically configured between 0.01 and 0.001.

### HyperLogLog
A probabilistic algorithm for cardinality estimation using much less space than exact counting while providing accurate estimates.

### MinHash
A technique for quickly estimating the similarity between two sets using hash functions to create compact signatures.

### Sketch
A compressed summary of a dataset that supports approximate queries. Count-Min sketch is a type of frequency sketch.

### Streaming Algorithm
An algorithm that processes input in a single pass (or small number of passes) using limited memory, suitable for data streams.

---

## Complexity and Performance

### Amortised Analysis
A method for analysing the average time per operation over a sequence of operations, accounting for occasional expensive operations.

### Expected Time
The average running time of a randomised algorithm, computed over all possible random choices.

### Space-Time Trade-off
The principle that increased space usage can reduce computation time and vice versa. Probabilistic structures exemplify this trade-off.

### Worst-Case Time
The maximum running time of an algorithm over all possible inputs of a given size.

---

## Research Terminology

### Community Detection
The task of identifying groups of densely connected vertices in a network, relevant to social network analysis.

### K-mer
A substring of length k from a biological sequence. Bloom filters are used for efficient k-mer counting in genomics.

### Network Motif
A recurring pattern of connections in a network that appears more frequently than expected by chance.

### PageRank
An algorithm that assigns importance scores to vertices in a directed graph based on the link structure. Originally developed for web search.

### Scale-Free Network
A network whose degree distribution follows a power law, common in real-world networks like the internet and social networks.

### Small-World Network
A network with high clustering and short average path lengths, characteristic of social networks.

---

*© 2025 Antonio Clim. All rights reserved.*
