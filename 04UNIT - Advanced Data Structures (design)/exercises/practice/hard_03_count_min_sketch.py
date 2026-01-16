#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4, Practice Exercise: Count-Min Sketch
Difficulty: HARD
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Count-Min Sketch (CMS) is a probabilistic data structure for approximating
frequency counts in streaming data. Unlike exact counting which requires
O(n) space for n unique items, CMS uses O(1/ε · log(1/δ)) space regardless
of the number of items.

Key properties:
- NEVER underestimates: count ≥ true count
- Overestimates bounded: count ≤ true count + εN with probability 1-δ
- Space-efficient: sublinear in stream size
- Supports point queries, range queries and inner products

Applications:
- Network traffic analysis (heavy hitters detection)
- NLP (word frequency estimation)
- Database query optimisation
- Anomaly detection in streaming data

TASK
────
Implement a Count-Min Sketch with:
1. Configurable error bounds (ε and δ)
2. Automatic parameter calculation
3. Heavy hitters identification
4. Merge operation for distributed counting

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Implement probabilistic counting structures
2. Apply mathematical bounds to configure parameters
3. Handle streaming data efficiently
4. Detect heavy hitters in data streams

ESTIMATED TIME
──────────────
90-120 minutes

DEPENDENCIES
────────────
- Python 3.12+
- hashlib (standard library)
- typing (standard library)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import Iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: COUNT-MIN SKETCH IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class CountMinSketch:
    """
    Count-Min Sketch for approximate frequency counting.
    
    The data structure uses a 2D array of counters with dimensions:
    - width (w) = ⌈e/ε⌉ where e is Euler's number
    - depth (d) = ⌈ln(1/δ)⌉
    
    For each item, we increment d counters (one per row) using d independent
    hash functions. To query, we take the minimum of all d counters.
    
    Error bounds:
    - With probability at least (1 - δ):
      estimated_count ≤ true_count + ε × total_count
    
    Attributes:
        width: Number of counters per row
        depth: Number of rows (hash functions)
        epsilon: Error factor (determines width)
        delta: Failure probability (determines depth)
        
    Example:
        >>> cms = CountMinSketch(epsilon=0.01, delta=0.001)
        >>> cms.add("hello", count=5)
        >>> cms.add("world", count=3)
        >>> cms.query("hello")  # Returns at least 5
        5
    """
    
    def __init__(
        self,
        epsilon: float = 0.01,
        delta: float = 0.001,
        width: int | None = None,
        depth: int | None = None
    ) -> None:
        """
        Initialise Count-Min Sketch.
        
        Either specify (epsilon, delta) for automatic sizing or
        provide (width, depth) directly.
        
        Args:
            epsilon: Error factor (smaller = more accurate, more space)
            delta: Failure probability (smaller = more reliable, more space)
            width: Override automatic width calculation
            depth: Override automatic depth calculation
            
        Raises:
            ValueError: If parameters are out of valid range
        """
        # TODO: Implement initialisation
        #
        # 1. Calculate or validate dimensions:
        #    - width = ceil(e / epsilon)  where e ≈ 2.718
        #    - depth = ceil(ln(1 / delta))
        #
        # 2. Create the counter table:
        #    - 2D list of integers, all initialised to 0
        #    - self._table[row][col] = 0
        #
        # 3. Store parameters for later use
        #
        # 4. Initialise total count to 0
        
        raise NotImplementedError("Implement CountMinSketch.__init__")
    
    def _hash(self, item: str, seed: int) -> int:
        """
        Generate hash value for item with given seed.
        
        Uses double hashing technique: h(x, i) = (h1(x) + i × h2(x)) mod width
        
        Args:
            item: Item to hash
            seed: Seed value (row index)
            
        Returns:
            Hash value in range [0, width)
        """
        # TODO: Implement hashing
        #
        # Option 1: Double hashing with MD5/SHA
        # - h1 = int(md5(item).hexdigest(), 16)
        # - h2 = int(sha1(item).hexdigest(), 16)
        # - return (h1 + seed * h2) % self.width
        #
        # Option 2: Use hashlib with different salts
        # - combined = f"{seed}:{item}".encode()
        # - return int(hashlib.md5(combined).hexdigest(), 16) % self.width
        
        raise NotImplementedError("Implement _hash")
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add item to the sketch.
        
        Increments the counter at each of the d hash positions.
        
        Args:
            item: Item to add
            count: Number of occurrences (default 1)
            
        Example:
            >>> cms = CountMinSketch()
            >>> cms.add("apple")  # Add 1 occurrence
            >>> cms.add("apple", count=5)  # Add 5 more
        """
        # TODO: Implement add operation
        #
        # For each row i in [0, depth):
        #   - Calculate column j = _hash(item, i)
        #   - Increment _table[i][j] by count
        # Update total count
        
        raise NotImplementedError("Implement add")
    
    def query(self, item: str) -> int:
        """
        Estimate the count of an item.
        
        Returns the minimum value across all d counters.
        This is guaranteed to be >= true count.
        
        Args:
            item: Item to query
            
        Returns:
            Estimated count (never underestimates)
            
        Example:
            >>> cms = CountMinSketch()
            >>> cms.add("apple", count=10)
            >>> cms.query("apple")
            10
            >>> cms.query("banana")  # Never seen
            0  # Or small positive due to collisions
        """
        # TODO: Implement query operation
        #
        # min_count = infinity
        # For each row i in [0, depth):
        #   - Calculate column j = _hash(item, i)
        #   - min_count = min(min_count, _table[i][j])
        # Return min_count
        
        raise NotImplementedError("Implement query")
    
    def add_all(self, items: Iterable[str]) -> None:
        """
        Add multiple items to the sketch.
        
        Args:
            items: Iterable of items to add
        """
        for item in items:
            self.add(item)
    
    @property
    def total_count(self) -> int:
        """Return total number of items added."""
        # TODO: Return stored total count
        raise NotImplementedError("Implement total_count property")
    
    def merge(self, other: CountMinSketch) -> CountMinSketch:
        """
        Merge two Count-Min Sketches.
        
        Both sketches must have the same dimensions.
        Returns a new sketch containing the sum of both.
        
        This enables distributed counting: each node maintains its own
        sketch, then merges them for global statistics.
        
        Args:
            other: Another CountMinSketch with same dimensions
            
        Returns:
            New CountMinSketch containing merged counts
            
        Raises:
            ValueError: If dimensions don't match
        """
        # TODO: Implement merge operation
        #
        # 1. Verify dimensions match
        # 2. Create new sketch with same dimensions
        # 3. For each cell, sum the values from both sketches
        # 4. Update total count
        
        raise NotImplementedError("Implement merge")
    
    def __add__(self, other: CountMinSketch) -> CountMinSketch:
        """Enable cms1 + cms2 syntax for merging."""
        return self.merge(other)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HEAVY HITTERS
# ═══════════════════════════════════════════════════════════════════════════════

class HeavyHittersTracker:
    """
    Track frequent items (heavy hitters) in a data stream.
    
    A heavy hitter is an item that appears more than a threshold
    fraction of the total stream.
    
    Uses Count-Min Sketch for frequency estimation plus a candidate
    set of potential heavy hitters.
    
    Example:
        >>> tracker = HeavyHittersTracker(threshold=0.1)
        >>> for word in text.split():
        ...     tracker.add(word)
        >>> tracker.get_heavy_hitters()
        [('the', 1523), ('a', 892), ...]
    """
    
    def __init__(
        self,
        threshold: float = 0.01,
        epsilon: float | None = None,
        delta: float = 0.001
    ) -> None:
        """
        Initialise heavy hitters tracker.
        
        Args:
            threshold: Minimum frequency to be considered heavy (φ)
            epsilon: CMS error factor (default: threshold/2)
            delta: CMS failure probability
        """
        # TODO: Implement initialisation
        #
        # 1. Store threshold
        # 2. Create Count-Min Sketch with appropriate parameters
        #    - epsilon should be at most threshold/2 to distinguish heavy hitters
        # 3. Maintain a set of candidate heavy hitters
        
        raise NotImplementedError("Implement HeavyHittersTracker.__init__")
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add item to the stream.
        
        Updates CMS and candidate set.
        
        Args:
            item: Item to add
            count: Number of occurrences
        """
        # TODO: Implement add
        #
        # 1. Add item to CMS
        # 2. Query estimated count
        # 3. If estimated count exceeds threshold × total_count,
        #    add to candidate set
        
        raise NotImplementedError("Implement add")
    
    def get_heavy_hitters(self) -> list[tuple[str, int]]:
        """
        Return items that are likely heavy hitters.
        
        Returns:
            List of (item, estimated_count) tuples, sorted by count descending
        """
        # TODO: Implement heavy hitters retrieval
        #
        # 1. For each candidate, query its estimated count
        # 2. Filter to those exceeding threshold × total_count
        # 3. Sort by count descending
        # 4. Return list of (item, count) tuples
        
        raise NotImplementedError("Implement get_heavy_hitters")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CONSERVATIVE UPDATE (BONUS)
# ═══════════════════════════════════════════════════════════════════════════════

class ConservativeCountMinSketch(CountMinSketch):
    """
    Count-Min Sketch with conservative update.
    
    Instead of incrementing all counters by the same amount,
    conservative update only increments counters that are at the
    current minimum. This reduces overestimation.
    
    Trade-off: Slightly slower updates, better accuracy.
    """
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add item with conservative update.
        
        Only increments counters that are at or below current minimum + count.
        """
        # TODO: Implement conservative update
        #
        # 1. Find all hash positions
        # 2. Find current minimum value
        # 3. Set target value = min + count
        # 4. For each position, set counter to max(current, target)
        
        raise NotImplementedError("Implement conservative add")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_cms_basic() -> None:
    """Test basic CMS operations."""
    cms = CountMinSketch(epsilon=0.1, delta=0.01)
    
    # Add items
    cms.add("apple", count=10)
    cms.add("banana", count=5)
    cms.add("cherry", count=3)
    
    # Query should never underestimate
    assert cms.query("apple") >= 10
    assert cms.query("banana") >= 5
    assert cms.query("cherry") >= 3
    
    # Items never added should have low counts
    # (might be > 0 due to collisions)
    assert cms.query("durian") >= 0
    
    # Total count
    assert cms.total_count == 18
    
    logger.info("✓ Basic CMS tests passed")


def test_cms_accuracy() -> None:
    """Test CMS accuracy bounds."""
    epsilon = 0.01
    delta = 0.001
    cms = CountMinSketch(epsilon=epsilon, delta=delta)
    
    # Add many items
    items = [f"item_{i}" for i in range(1000)]
    counts = {item: (i % 100) + 1 for i, item in enumerate(items)}
    
    for item, count in counts.items():
        cms.add(item, count)
    
    # Check error bounds
    total = cms.total_count
    errors = 0
    
    for item, true_count in counts.items():
        estimated = cms.query(item)
        
        # Should never underestimate
        assert estimated >= true_count
        
        # Should not overestimate too much (with high probability)
        if estimated > true_count + epsilon * total:
            errors += 1
    
    # Allow delta fraction of errors
    assert errors <= delta * len(items) * 2  # Some slack for testing
    
    logger.info("✓ CMS accuracy tests passed")


def test_cms_merge() -> None:
    """Test CMS merge operation."""
    cms1 = CountMinSketch(epsilon=0.1, delta=0.01)
    cms2 = CountMinSketch(epsilon=0.1, delta=0.01)
    
    cms1.add("apple", count=10)
    cms1.add("banana", count=5)
    
    cms2.add("apple", count=3)
    cms2.add("cherry", count=7)
    
    merged = cms1 + cms2
    
    assert merged.query("apple") >= 13
    assert merged.query("banana") >= 5
    assert merged.query("cherry") >= 7
    assert merged.total_count == 25
    
    logger.info("✓ CMS merge tests passed")


def test_heavy_hitters() -> None:
    """Test heavy hitters detection."""
    tracker = HeavyHittersTracker(threshold=0.05)
    
    # Add items with varying frequencies
    for _ in range(100):
        tracker.add("common")
    for _ in range(50):
        tracker.add("medium")
    for _ in range(10):
        tracker.add("rare")
    
    heavy = tracker.get_heavy_hitters()
    heavy_items = [item for item, _ in heavy]
    
    # "common" should definitely be heavy
    assert "common" in heavy_items
    
    # "rare" should not be heavy (only 6% of stream)
    # Actually 10/160 = 6.25% > 5%, so might be included
    
    logger.info("✓ Heavy hitters tests passed")


def test_cms_dimensions() -> None:
    """Test CMS dimension calculations."""
    e = math.e
    
    # Test standard parameters
    cms = CountMinSketch(epsilon=0.01, delta=0.001)
    
    expected_width = math.ceil(e / 0.01)
    expected_depth = math.ceil(math.log(1 / 0.001))
    
    assert cms.width == expected_width
    assert cms.depth == expected_depth
    
    logger.info("✓ CMS dimension tests passed")


def run_all_tests() -> None:
    """Run all test cases."""
    logger.info("Running Count-Min Sketch tests...")
    
    test_cms_dimensions()
    test_cms_basic()
    test_cms_accuracy()
    test_cms_merge()
    test_heavy_hitters()
    
    logger.info("═" * 50)
    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def demo_word_frequency() -> None:
    """Demonstrate CMS for word frequency counting."""
    logger.info("Word Frequency Demonstration")
    logger.info("=" * 50)
    
    # Sample text
    text = """
    The quick brown fox jumps over the lazy dog.
    The dog was not amused by the fox.
    The fox ran away quickly but the dog gave chase.
    In the end the fox escaped into the forest.
    """
    
    # Create tracker
    tracker = HeavyHittersTracker(threshold=0.05)
    
    # Process words
    words = text.lower().split()
    for word in words:
        # Clean punctuation
        word = "".join(c for c in word if c.isalnum())
        if word:
            tracker.add(word)
    
    # Show heavy hitters
    logger.info(f"Total words: {tracker.cms.total_count}")
    logger.info("Heavy hitters (>5% frequency):")
    
    for item, count in tracker.get_heavy_hitters():
        pct = count / tracker.cms.total_count * 100
        logger.info(f"  '{item}': {count} ({pct:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Count-Min Sketch Exercise"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        run_all_tests()
    elif args.demo:
        demo_word_frequency()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
