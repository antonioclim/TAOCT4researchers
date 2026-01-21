#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4, Lab 2: Probabilistic Data Structures - SOLUTION KEY
═══════════════════════════════════════════════════════════════════════════════

INSTRUCTOR NOTES
────────────────
This solution provides reference implementations for Bloom filters and
Count-Min sketches with detailed explanations of the mathematical foundations.

ASSESSMENT FOCUS
────────────────
1. Understanding of probabilistic guarantees (30%)
2. Correct implementation of hash-based operations (30%)
3. Parameter calculation from error bounds (20%)
4. Code quality and documentation (20%)

MATHEMATICAL BACKGROUND
───────────────────────
Bloom Filter:
- m = -(n × ln(p)) / (ln(2))² where n=elements, p=FPR
- k = (m/n) × ln(2) optimal hash functions
- Actual FPR ≈ (1 - e^(-kn/m))^k

Count-Min Sketch:
- w = ⌈e/ε⌉ width for additive error ε
- d = ⌈ln(1/δ)⌉ depth for confidence 1-δ
- Estimate is min over all rows

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import math
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 1: BLOOM FILTER
# ═══════════════════════════════════════════════════════════════════════════════


class BloomFilter:
    """
    Space-efficient probabilistic set membership testing.
    
    PROPERTIES:
    - No false negatives: if element was added, contains() returns True
    - May have false positives: contains() may return True for non-members
    - Cannot delete elements (use Counting Bloom Filter for deletion)
    
    GRADING CRITERIA:
    1. Correct bit array manipulation
    2. Multiple independent hash functions
    3. Proper parameter calculation
    4. Clear documentation of trade-offs
    """
    
    def __init__(self, size: int, num_hash_functions: int) -> None:
        """
        Initialise Bloom filter with explicit parameters.
        
        Args:
            size: Number of bits in the filter (m)
            num_hash_functions: Number of hash functions to use (k)
        
        GRADING NOTE:
        Students should validate parameters (size > 0, k > 0).
        """
        if size <= 0:
            raise ValueError("Size must be positive")
        if num_hash_functions <= 0:
            raise ValueError("Number of hash functions must be positive")
        
        self._size = size
        self._k = num_hash_functions
        self._bits = [False] * size
        self._count = 0
        self._items: set[Any] = set()  # For tracking unique additions
    
    @classmethod
    def from_parameters(
        cls,
        expected_elements: int,
        false_positive_rate: float
    ) -> BloomFilter:
        """
        Create Bloom filter from desired error parameters.
        
        FORMULAS:
        - Optimal size: m = -(n × ln(p)) / (ln(2))²
        - Optimal k: k = (m/n) × ln(2)
        
        DERIVATION (for interested students):
        The false positive rate for a Bloom filter is:
            p = (1 - (1 - 1/m)^(kn))^k ≈ (1 - e^(-kn/m))^k
        
        Taking the derivative with respect to k and setting to 0:
            k_optimal = (m/n) × ln(2)
        
        Substituting back and solving for m:
            m = -(n × ln(p)) / (ln(2))²
        """
        if expected_elements <= 0:
            raise ValueError("Expected elements must be positive")
        if not 0 < false_positive_rate < 1:
            raise ValueError("False positive rate must be between 0 and 1")
        
        n = expected_elements
        p = false_positive_rate
        
        # Calculate optimal size
        # m = -(n * ln(p)) / (ln(2))^2
        m = -int((n * math.log(p)) / (math.log(2) ** 2))
        m = max(m, 1)  # Ensure at least 1 bit
        
        # Calculate optimal number of hash functions
        # k = (m/n) * ln(2)
        k = max(1, int((m / n) * math.log(2)))
        
        return cls(size=m, num_hash_functions=k)
    
    def _hash(self, item: Any, seed: int) -> int:
        """
        Generate hash value for item with given seed.
        
        IMPLEMENTATION CHOICES:
        1. Double hashing: h(i) = h1 + i*h2 (most common)
        2. Independent hash functions (expensive)
        3. Augmented double hashing with salt
        
        GRADING: Accept any valid approach that provides
        sufficiently independent hash values.
        
        This implementation uses improved double hashing:
        h_i(x) = (h1(x) + i * h2(x) + i²) mod m
        
        The i² term improves independence for similar seeds.
        """
        # Primary hash
        data = str(item).encode('utf-8')
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        
        # Secondary hash (different algorithm for independence)
        h2 = int(hashlib.sha256(data).hexdigest(), 16)
        
        # Augmented double hashing
        return (h1 + seed * h2 + seed * seed) % self._size
    
    def add(self, item: Any) -> None:
        """
        Add item to the filter.
        
        OPERATION:
        For each hash function, set the corresponding bit to 1.
        
        TIME COMPLEXITY: O(k) where k is number of hash functions
        """
        is_new = item not in self._items
        
        for i in range(self._k):
            index = self._hash(item, i)
            self._bits[index] = True
        
        if is_new:
            self._items.add(item)
            self._count += 1
    
    def contains(self, item: Any) -> bool:
        """
        Check if item might be in the filter.
        
        OPERATION:
        Check if ALL corresponding bits are set.
        
        RETURNS:
        - True: Item MAY be in the set (could be false positive)
        - False: Item is DEFINITELY NOT in the set
        
        INSTRUCTOR NOTE:
        Emphasise the asymmetry: False is definitive, True is probabilistic.
        """
        for i in range(self._k):
            index = self._hash(item, i)
            if not self._bits[index]:
                return False  # Definitely not in set
        
        return True  # Possibly in set
    
    @property
    def size(self) -> int:
        """Return size of bit array."""
        return self._size
    
    @property
    def num_hash_functions(self) -> int:
        """Return number of hash functions."""
        return self._k
    
    @property
    def count(self) -> int:
        """Return number of items added."""
        return self._count
    
    def estimated_false_positive_rate(self) -> float:
        """
        Estimate current false positive rate.
        
        FORMULA: p ≈ (1 - e^(-kn/m))^k
        
        This is the theoretical FPR given current load.
        """
        if self._count == 0:
            return 0.0
        
        k = self._k
        n = self._count
        m = self._size
        
        # p = (1 - e^(-kn/m))^k
        return (1 - math.exp(-k * n / m)) ** k
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BloomFilter(size={self._size}, k={self._k}, "
            f"count={self._count}, est_fpr={self.estimated_false_positive_rate():.4f})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 2: COUNT-MIN SKETCH
# ═══════════════════════════════════════════════════════════════════════════════


class CountMinSketch:
    """
    Probabilistic frequency estimation for streaming data.
    
    PROPERTIES:
    - Always overestimates (never underestimates)
    - Error bounded by εN with probability 1-δ
    - Supports merging for distributed computation
    
    GRADING CRITERIA:
    1. Correct 2D array structure
    2. Minimum over rows for estimation
    3. Proper parameter calculation
    4. Understanding of error bounds
    """
    
    def __init__(self, width: int, depth: int) -> None:
        """
        Initialise Count-Min sketch with explicit dimensions.
        
        Args:
            width: Number of counters per row (w)
            depth: Number of rows/hash functions (d)
        
        SPACE: O(w × d) = O((e/ε) × ln(1/δ))
        """
        if width <= 0:
            raise ValueError("Width must be positive")
        if depth <= 0:
            raise ValueError("Depth must be positive")
        
        self._width = width
        self._depth = depth
        self._table: list[list[int]] = [[0] * width for _ in range(depth)]
        self._total = 0
    
    @classmethod
    def from_parameters(cls, epsilon: float, delta: float) -> CountMinSketch:
        """
        Create Count-Min sketch from error parameters.
        
        PARAMETERS:
        - epsilon (ε): Additive error factor (estimate ≤ true + εN)
        - delta (δ): Failure probability (bounds hold with prob 1-δ)
        
        FORMULAS:
        - Width: w = ⌈e/ε⌉ where e ≈ 2.718
        - Depth: d = ⌈ln(1/δ)⌉
        
        EXAMPLE:
        ε=0.01, δ=0.01 gives w=272, d=5
        For N=1,000,000 items, error ≤ 10,000 with 99% confidence
        """
        if not 0 < epsilon < 1:
            raise ValueError("Epsilon must be between 0 and 1")
        if not 0 < delta < 1:
            raise ValueError("Delta must be between 0 and 1")
        
        # w = ceil(e / epsilon)
        width = math.ceil(math.e / epsilon)
        
        # d = ceil(ln(1/delta))
        depth = math.ceil(math.log(1 / delta))
        
        return cls(width=width, depth=depth)
    
    def _hash(self, item: Any, row: int) -> int:
        """
        Generate hash value for item in specified row.
        
        Each row needs a different hash function.
        Uses the same improved double hashing technique as Bloom filter.
        """
        data = str(item).encode('utf-8')
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        h2 = int(hashlib.sha256(data).hexdigest(), 16)
        
        return (h1 + row * h2 + row * row) % self._width
    
    def update(self, item: Any, count: int = 1) -> None:
        """
        Update frequency estimate for item.
        
        OPERATION:
        Increment counter at position hash(item) in each row.
        
        TIME COMPLEXITY: O(d)
        
        GRADING NOTE:
        Some implementations support negative counts for deletion.
        Standard Count-Min only supports positive updates.
        """
        if count < 0:
            raise ValueError("Count must be non-negative (standard CMS)")
        
        for row in range(self._depth):
            col = self._hash(item, row)
            self._table[row][col] += count
        
        self._total += count
    
    def estimate(self, item: Any) -> int:
        """
        Estimate frequency of item.
        
        OPERATION:
        Return MINIMUM of counters across all rows.
        
        WHY MINIMUM?
        - Each counter may have collisions (overestimate)
        - Minimum is least affected by collisions
        - Provides tightest upper bound
        
        GUARANTEE:
        true_count ≤ estimate ≤ true_count + εN
        (with probability 1-δ)
        """
        min_count = float('inf')
        
        for row in range(self._depth):
            col = self._hash(item, row)
            min_count = min(min_count, self._table[row][col])
        
        return int(min_count)
    
    def merge(self, other: CountMinSketch) -> CountMinSketch:
        """
        Merge two Count-Min sketches.
        
        OPERATION:
        Point-wise addition of counters.
        
        REQUIREMENT:
        Both sketches must have same dimensions (same hash functions).
        
        USE CASE:
        Distributed stream processing - each node maintains local
        sketch, then merge for global estimate.
        """
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("Cannot merge sketches with different dimensions")
        
        result = CountMinSketch(self._width, self._depth)
        
        for row in range(self._depth):
            for col in range(self._width):
                result._table[row][col] = (
                    self._table[row][col] + other._table[row][col]
                )
        
        result._total = self._total + other._total
        return result
    
    @property
    def width(self) -> int:
        """Return width of sketch."""
        return self._width
    
    @property
    def depth(self) -> int:
        """Return depth of sketch."""
        return self._depth
    
    @property
    def total_count(self) -> int:
        """Return total count of all updates."""
        return self._total
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CountMinSketch(width={self._width}, depth={self._depth}, total={self._total})"


# ═══════════════════════════════════════════════════════════════════════════════
# GRADING RUBRIC SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
"""
BLOOM FILTER (50 points)

Implementation (25 points):
- Correct bit array operations: 10 pts
- Multiple hash functions: 5 pts
- Add operation: 5 pts
- Contains operation: 5 pts

Parameter Calculation (15 points):
- Optimal size formula: 8 pts
- Optimal k formula: 7 pts

Understanding (10 points):
- Documentation of false positives: 5 pts
- No false negatives guarantee: 5 pts


COUNT-MIN SKETCH (50 points)

Implementation (25 points):
- Correct 2D table structure: 8 pts
- Update operation: 7 pts
- Estimate operation (minimum): 10 pts

Parameter Calculation (15 points):
- Width from epsilon: 7 pts
- Depth from delta: 8 pts

Advanced Features (10 points):
- Merge operation: 5 pts
- Error bound documentation: 5 pts


BONUS (10 points)
- Counting Bloom Filter: 5 pts
- Heavy Hitters using CMS: 3 pts
- Space comparison analysis: 2 pts


COMMON DEDUCTIONS:
- -5: Using single hash function for Bloom filter
- -5: Taking maximum instead of minimum in CMS
- -3: Missing error handling for invalid parameters
- -3: Incorrect formula implementation
- -2: Missing type hints
- -2: Missing docstrings
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════


def demonstrate_bloom_filter() -> None:
    """Demonstrate Bloom filter usage."""
    print("=" * 60)
    print("BLOOM FILTER DEMONSTRATION")
    print("=" * 60)
    
    # Create filter for 100 elements with 1% FPR
    bf = BloomFilter.from_parameters(
        expected_elements=100,
        false_positive_rate=0.01
    )
    print(f"Created: {bf}")
    
    # Add elements
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    for word in words:
        bf.add(word)
    
    print(f"\nAfter adding {len(words)} words: {bf}")
    
    # Test membership
    print("\nMembership tests:")
    for word in words:
        print(f"  '{word}' in filter: {bf.contains(word)}")
    
    test_words = ["fig", "grape", "honeydew"]
    for word in test_words:
        print(f"  '{word}' in filter: {bf.contains(word)} (not added)")


def demonstrate_count_min_sketch() -> None:
    """Demonstrate Count-Min sketch usage."""
    print("\n" + "=" * 60)
    print("COUNT-MIN SKETCH DEMONSTRATION")
    print("=" * 60)
    
    # Create sketch with error parameters
    cms = CountMinSketch.from_parameters(epsilon=0.1, delta=0.05)
    print(f"Created: {cms}")
    
    # Simulate stream
    stream = (
        ["apple"] * 100 +
        ["banana"] * 50 +
        ["cherry"] * 25 +
        ["date"] * 10
    )
    
    for item in stream:
        cms.update(item)
    
    print(f"\nAfter processing {len(stream)} items: {cms}")
    
    # Estimate frequencies
    print("\nFrequency estimates (true in parentheses):")
    print(f"  apple: {cms.estimate('apple')} (100)")
    print(f"  banana: {cms.estimate('banana')} (50)")
    print(f"  cherry: {cms.estimate('cherry')} (25)")
    print(f"  date: {cms.estimate('date')} (10)")
    print(f"  elderberry: {cms.estimate('elderberry')} (0)")


if __name__ == "__main__":
    demonstrate_bloom_filter()
    demonstrate_count_min_sketch()
