#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4, Lab 2: Probabilistic Data Structures
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
When processing billions of elements, exact methods become impractical.
Probabilistic data structures trade exactness for dramatic space savings,
enabling membership testing, frequency estimation and cardinality counting
with bounded memory. This lab explores Bloom filters, Count-Min sketches
and their applications in research computing.

PREREQUISITES
─────────────
- Week 3: Complexity analysis, amortised analysis
- Week 4 Lab 1: Hash table concepts, collision handling
- Mathematics: Basic probability theory, expected value

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement Bloom filters with optimal parameter selection
2. Apply Count-Min sketch for frequency estimation
3. Analyse false positive rates empirically and theoretically
4. Evaluate when probabilistic structures are appropriate

ESTIMATED TIME
──────────────
- Reading: 20 minutes
- Coding: 70 minutes
- Total: 90 minutes

DEPENDENCIES
────────────
- Python 3.12+
- mmh3 >= 4.0 (MurmurHash3)
- numpy >= 1.24

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type variable for hashable items
T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BLOOM FILTER
# ═══════════════════════════════════════════════════════════════════════════════


class BloomFilter:
    """
    A space-efficient probabilistic data structure for membership testing.

    A Bloom filter tests set membership with:
    - No false negatives: If it says "not present", the item is definitely absent
    - Possible false positives: If it says "maybe present", verify if needed

    Trade-offs:
    ───────────
    Space: O(m) bits, where m = -n·ln(p) / (ln(2)²)
    Add: O(k) where k is the number of hash functions
    Query: O(k)
    Delete: NOT SUPPORTED (use Counting Bloom Filter instead)

    Applications:
    ─────────────
    - Chrome: Checking URLs against malicious site database
    - Cassandra/HBase: Avoiding disk reads for non-existent keys
    - Medium: Tracking which articles a user has read
    - Spell checkers: Fast dictionary membership testing

    Mathematical Foundation:
    ────────────────────────
    Optimal number of hash functions: k = (m/n) · ln(2)
    False positive probability: p ≈ (1 - e^(-kn/m))^k
    Bits per element for target FP rate: m/n = -ln(p) / (ln(2)²)

    Example:
        >>> bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        >>> bf.add("hello")
        >>> "hello" in bf
        True
        >>> "world" in bf  # Probably False (may be a false positive)
        False
    """

    def __init__(
        self,
        expected_items: int,
        false_positive_rate: float = 0.01,
    ) -> None:
        """
        Initialise a Bloom filter with optimal parameters.

        Args:
            expected_items: Expected number of items to be inserted
            false_positive_rate: Target false positive rate (0 < p < 1)

        Raises:
            ValueError: If parameters are out of valid range
        """
        if expected_items <= 0:
            raise ValueError("expected_items must be positive")
        if not 0 < false_positive_rate < 1:
            raise ValueError("false_positive_rate must be between 0 and 1")

        self._n = expected_items
        self._p = false_positive_rate

        # Calculate optimal parameters
        # m = -n·ln(p) / (ln(2)²)
        self._m = int(-expected_items * math.log(false_positive_rate) / (math.log(2) ** 2))
        # k = (m/n) · ln(2)
        self._k = max(1, int((self._m / expected_items) * math.log(2)))

        # Bit array (using Python list of bools for clarity)
        self._bits: list[bool] = [False] * self._m
        self._count = 0

        logger.debug(
            "BloomFilter created: m=%d bits, k=%d hashes, expected n=%d, target p=%.4f",
            self._m,
            self._k,
            self._n,
            self._p,
        )

    def _get_hash_indices(self, item: str) -> Iterator[int]:
        """
        Generate k hash indices for an item.

        Uses the double hashing technique:
        h(x, i) = (h1(x) + i·h2(x)) mod m

        This generates k different hash values from two base hashes,
        which is more efficient than computing k independent hashes.
        """
        # Use SHA-256 to get two 64-bit hashes
        h = hashlib.sha256(item.encode()).digest()
        h1 = int.from_bytes(h[:8], "big")
        h2 = int.from_bytes(h[8:16], "big")

        for i in range(self._k):
            yield (h1 + i * h2) % self._m

    def add(self, item: str) -> None:
        """
        Add an item to the filter.

        Args:
            item: The item to add (will be converted to string)
        """
        for idx in self._get_hash_indices(str(item)):
            self._bits[idx] = True
        self._count += 1

    def __contains__(self, item: str) -> bool:
        """
        Check if an item might be in the filter.

        Returns:
            False: Item is definitely NOT in the filter
            True: Item MIGHT be in the filter (could be false positive)
        """
        return all(self._bits[idx] for idx in self._get_hash_indices(str(item)))

    def estimated_false_positive_rate(self) -> float:
        """
        Estimate the current false positive rate based on fill ratio.

        The actual FP rate depends on how many bits are set:
        p ≈ (bits_set / m)^k

        Returns:
            Estimated false positive probability
        """
        bits_set = sum(self._bits)
        if bits_set == 0:
            return 0.0
        return (bits_set / self._m) ** self._k

    def union(self, other: BloomFilter) -> BloomFilter:
        """
        Return the union of two Bloom filters.

        Both filters must have the same parameters (m and k).

        Args:
            other: Another Bloom filter

        Returns:
            New Bloom filter representing the union

        Raises:
            ValueError: If filters have different parameters
        """
        if self._m != other._m or self._k != other._k:
            raise ValueError("Cannot union filters with different parameters")

        result = BloomFilter(self._n, self._p)
        result._bits = [a or b for a, b in zip(self._bits, other._bits)]
        result._count = self._count + other._count
        return result

    @property
    def size_bytes(self) -> int:
        """Return the size of the filter in bytes."""
        return math.ceil(self._m / 8)

    @property
    def fill_ratio(self) -> float:
        """Return the proportion of bits that are set."""
        return sum(self._bits) / self._m

    def __repr__(self) -> str:
        return (
            f"BloomFilter(m={self._m}, k={self._k}, "
            f"items={self._count}, fill={self.fill_ratio:.2%})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COUNT-MIN SKETCH
# ═══════════════════════════════════════════════════════════════════════════════


class CountMinSketch:
    """
    A probabilistic data structure for frequency estimation.

    Count-Min Sketch maintains d arrays of w counters. To increment an
    element, we add 1 to position h_i(x) in each array. To query frequency,
    we return the minimum across all arrays.

    Properties:
    ───────────
    - Never underestimates: Reported count ≥ true count
    - May overestimate: Due to hash collisions
    - Error bound: With probability ≥ 1-δ, error ≤ εN where N is total count

    Parameters:
    ───────────
    - Width w: Controls accuracy, ε = e/w
    - Depth d: Controls confidence, δ = e^(-d)

    Applications:
    ─────────────
    - Network traffic analysis
    - Database query optimisation
    - Finding heavy hitters in data streams
    - Click-stream analysis

    Example:
        >>> cms = CountMinSketch(epsilon=0.001, delta=0.01)
        >>> for word in text.split():
        ...     cms.add(word)
        >>> cms.estimate("the")
        142  # Might be slightly higher than actual count
    """

    def __init__(
        self,
        epsilon: float = 0.001,
        delta: float = 0.01,
    ) -> None:
        """
        Initialise a Count-Min Sketch.

        Args:
            epsilon: Error factor (ε). Error ≤ ε·N with high probability
            delta: Failure probability (δ). Guarantee holds with prob ≥ 1-δ

        The sketch uses:
        - w = ⌈e/ε⌉ counters per row
        - d = ⌈ln(1/δ)⌉ rows (hash functions)
        """
        self._epsilon = epsilon
        self._delta = delta

        # Calculate dimensions
        self._w = int(math.ceil(math.e / epsilon))
        self._d = int(math.ceil(math.log(1.0 / delta)))

        # Counter arrays
        self._counters: list[list[int]] = [[0] * self._w for _ in range(self._d)]
        self._total_count = 0

        logger.debug(
            "CountMinSketch created: width=%d, depth=%d, ε=%.4f, δ=%.4f",
            self._w,
            self._d,
            epsilon,
            delta,
        )

    def _get_hash_indices(self, item: str) -> list[int]:
        """Generate d hash indices for an item, one per row."""
        indices = []
        for i in range(self._d):
            # Use different seed for each row
            h = hashlib.sha256(f"{i}:{item}".encode()).digest()
            idx = int.from_bytes(h[:8], "big") % self._w
            indices.append(idx)
        return indices

    def add(self, item: str, count: int = 1) -> None:
        """
        Add an item to the sketch with optional count.

        Args:
            item: The item to add
            count: Number of occurrences (default: 1)
        """
        indices = self._get_hash_indices(str(item))
        for row, idx in enumerate(indices):
            self._counters[row][idx] += count
        self._total_count += count

    def estimate(self, item: str) -> int:
        """
        Estimate the frequency of an item.

        Returns:
            Estimated count (may overestimate, never underestimates)
        """
        indices = self._get_hash_indices(str(item))
        return min(self._counters[row][idx] for row, idx in enumerate(indices))

    def merge(self, other: CountMinSketch) -> CountMinSketch:
        """
        Merge two Count-Min Sketches.

        Both sketches must have the same dimensions.

        Args:
            other: Another Count-Min Sketch

        Returns:
            New sketch with merged counts
        """
        if self._w != other._w or self._d != other._d:
            raise ValueError("Cannot merge sketches with different dimensions")

        result = CountMinSketch(self._epsilon, self._delta)
        for row in range(self._d):
            for col in range(self._w):
                result._counters[row][col] = (
                    self._counters[row][col] + other._counters[row][col]
                )
        result._total_count = self._total_count + other._total_count
        return result

    @property
    def total_count(self) -> int:
        """Return the total count of all items added."""
        return self._total_count

    @property
    def size_bytes(self) -> int:
        """Return the approximate size in bytes."""
        return self._w * self._d * 8  # Assuming 8 bytes per counter

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(w={self._w}, d={self._d}, "
            f"total={self._total_count}, size={self.size_bytes:,} bytes)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_bloom_filter() -> None:
    """Demonstrate Bloom filter usage and false positive rates."""
    logger.info("=" * 60)
    logger.info("DEMO: Bloom Filter")
    logger.info("=" * 60)

    # Create a Bloom filter for 10,000 items with 1% FP rate
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)
    logger.info("Created: %s", bf)
    logger.info("Size: %d bytes (%.2f KB)", bf.size_bytes, bf.size_bytes / 1024)

    # Add some items
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    for word in words:
        bf.add(word)

    logger.info("Added %d items", len(words))

    # Test membership
    logger.info("Membership tests:")
    for word in ["apple", "fig", "banana", "grape"]:
        result = "MAYBE" if word in bf else "NO"
        logger.info("  '%s' in filter? %s", word, result)

    # Demonstrate false positives
    logger.info("Testing false positive rate...")

    # Add 1000 items
    bf_test = BloomFilter(expected_items=1000, false_positive_rate=0.05)
    for i in range(1000):
        bf_test.add(f"item_{i}")

    # Check items NOT in the filter
    false_positives = 0
    test_count = 10000
    for i in range(test_count):
        if f"notitem_{i}" in bf_test:
            false_positives += 1

    empirical_fp_rate = false_positives / test_count
    logger.info("  Expected FP rate: 5%%")
    logger.info("  Empirical FP rate: %.2f%%", empirical_fp_rate * 100)
    logger.info("  Estimated FP rate: %.2f%%", bf_test.estimated_false_positive_rate() * 100)


def demo_count_min_sketch() -> None:
    """Demonstrate Count-Min Sketch for frequency estimation."""
    logger.info("=" * 60)
    logger.info("DEMO: Count-Min Sketch")
    logger.info("=" * 60)

    # Create a sketch
    cms = CountMinSketch(epsilon=0.01, delta=0.01)
    logger.info("Created: %s", cms)

    # Sample text for frequency counting
    text = """
    the quick brown fox jumps over the lazy dog
    the fox was quick and the dog was lazy
    the brown fox and the brown dog were friends
    """

    # Count actual frequencies
    words = text.lower().split()
    actual_counts: dict[str, int] = {}
    for word in words:
        actual_counts[word] = actual_counts.get(word, 0) + 1
        cms.add(word)

    # Compare estimates with actual counts
    logger.info("Frequency estimates vs actual:")
    for word in sorted(set(words)):
        actual = actual_counts[word]
        estimate = cms.estimate(word)
        error = estimate - actual
        logger.info("  '%s': actual=%d, estimate=%d, error=%+d", word, actual, estimate, error)


def demo_spell_checker() -> None:
    """Demonstrate a simple spell checker using Bloom filter."""
    logger.info("=" * 60)
    logger.info("DEMO: Spell Checker with Bloom Filter")
    logger.info("=" * 60)

    # Build dictionary (small sample for demonstration)
    dictionary = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "computational", "thinking", "research",
    ]

    # Create Bloom filter dictionary
    bf = BloomFilter(expected_items=len(dictionary), false_positive_rate=0.01)
    for word in dictionary:
        bf.add(word.lower())

    logger.info("Dictionary loaded: %d words", len(dictionary))
    logger.info("Bloom filter size: %d bytes", bf.size_bytes)

    # Check some words
    test_words = ["the", "research", "teh", "reserch", "computational", "compuational"]

    logger.info("Spell check results:")
    for word in test_words:
        if word.lower() in bf:
            logger.info("  '%s' - possibly correct", word)
        else:
            logger.info("  '%s' - MISSPELLED", word)


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_bloom_filter()
    demo_count_min_sketch()
    demo_spell_checker()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Week 4 Lab 2: Probabilistic Data Structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_4_02_probabilistic_ds.py --demo
  python lab_4_02_probabilistic_ds.py -v --demo
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration examples",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_all_demos()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
