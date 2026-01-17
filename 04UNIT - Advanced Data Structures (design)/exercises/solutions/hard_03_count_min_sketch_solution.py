#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 4, Practice Exercise Solution: Count-Min Sketch
Difficulty: HARD
═══════════════════════════════════════════════════════════════════════════════

SOLUTION OVERVIEW
─────────────────
This module provides complete implementations of:
1. CountMinSketch — probabilistic frequency estimation structure
2. HeavyHittersTracker — stream heavy hitters detection
3. ConservativeCountMinSketch — reduced overestimation variant
4. Count-Mean-Min Sketch — bias-corrected estimation

ALGORITHM SUMMARY
─────────────────
Count-Min Sketch is a probabilistic data structure that uses:
- A 2D array of counters (width × depth)
- d independent hash functions (one per row)
- Point queries return minimum across all rows

Error bounds with parameters (ε, δ):
- Width w = ⌈e/ε⌉ ensures error bounded by εN
- Depth d = ⌈ln(1/δ)⌉ ensures bounds hold with probability 1-δ

Key properties:
- NEVER underestimates (always ≥ true count)
- Space complexity: O(1/ε · log(1/δ))
- Update complexity: O(d) = O(log(1/δ))
- Query complexity: O(d) = O(log(1/δ))

IMPLEMENTATION NOTES
────────────────────
- Uses double hashing technique for generating d hash values
- Supports arithmetic operations (+) for distributed counting
- Heavy hitters tracked via candidate set with filtering

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
from dataclasses import dataclass
from dataclasses import field
from typing import Iterable
from typing import Iterator

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
        
    Complexity:
        Space: O(w × d) = O((e/ε) × ln(1/δ))
        Add: O(d) = O(ln(1/δ))
        Query: O(d) = O(ln(1/δ))
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
            epsilon: Error factor (smaller = more accurate, more space).
                     Error bounded by ε × total_count.
            delta: Failure probability (smaller = more reliable, more space).
                   Bounds hold with probability 1 - δ.
            width: Override automatic width calculation
            depth: Override automatic depth calculation
            
        Raises:
            ValueError: If parameters are out of valid range
            
        Example:
            >>> # Automatic sizing for 1% error, 0.1% failure
            >>> cms = CountMinSketch(epsilon=0.01, delta=0.001)
            >>> cms.width
            272
            >>> cms.depth
            7
            >>> # Manual sizing
            >>> cms2 = CountMinSketch(width=1000, depth=5)
        """
        # Validate parameters
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(
                f"epsilon must be in (0, 1), got {epsilon}"
            )
        if delta <= 0 or delta >= 1:
            raise ValueError(
                f"delta must be in (0, 1), got {delta}"
            )
        
        # Store error parameters
        self._epsilon = epsilon
        self._delta = delta
        
        # Calculate dimensions from error bounds
        # Width: w = ceil(e / epsilon) ensures additive error <= epsilon * N
        # Depth: d = ceil(ln(1/delta)) ensures bounds hold with prob 1 - delta
        e = math.e
        
        if width is not None:
            if width <= 0:
                raise ValueError(f"width must be positive, got {width}")
            self._width = width
        else:
            self._width = math.ceil(e / epsilon)
        
        if depth is not None:
            if depth <= 0:
                raise ValueError(f"depth must be positive, got {depth}")
            self._depth = depth
        else:
            self._depth = math.ceil(math.log(1 / delta))
        
        # Initialise counter table (2D list of zeros)
        self._table: list[list[int]] = [
            [0] * self._width for _ in range(self._depth)
        ]
        
        # Track total count for error calculations
        self._total_count = 0
        
        logger.debug(
            f"Initialised CMS: width={self._width}, depth={self._depth}, "
            f"epsilon={epsilon}, delta={delta}"
        )
    
    @property
    def width(self) -> int:
        """Return the number of counters per row."""
        return self._width
    
    @property
    def depth(self) -> int:
        """Return the number of rows (hash functions)."""
        return self._depth
    
    @property
    def epsilon(self) -> float:
        """Return the error factor."""
        return self._epsilon
    
    @property
    def delta(self) -> float:
        """Return the failure probability."""
        return self._delta
    
    @property
    def total_count(self) -> int:
        """Return total number of items added."""
        return self._total_count
    
    @property
    def space_bytes(self) -> int:
        """
        Estimate memory usage in bytes.
        
        Returns:
            Approximate memory consumption
        """
        # Each counter is a Python int (typically 28 bytes for small ints)
        # Plus list overhead
        counter_size = 28  # Approximate for small integers
        list_overhead = 56  # Per list object
        
        return (
            self._width * self._depth * counter_size +
            self._depth * list_overhead +
            list_overhead
        )
    
    def _hash(self, item: str, seed: int) -> int:
        """
        Generate hash value for item with given seed.
        
        Uses double hashing technique: h(x, i) = (h1(x) + i × h2(x)) mod width
        
        This generates d pairwise independent hash values from two base hashes.
        Research shows this achieves similar accuracy to truly independent
        hash functions while being more efficient.
        
        Args:
            item: Item to hash
            seed: Seed value (row index, 0 to depth-1)
            
        Returns:
            Hash value in range [0, width)
            
        Example:
            >>> cms = CountMinSketch(width=100, depth=5)
            >>> cms._hash("test", 0)  # First hash
            42
            >>> cms._hash("test", 1)  # Second hash (different)
            87
        """
        # Encode item to bytes
        item_bytes = item.encode('utf-8')
        
        # Generate two independent hash values using MD5 and SHA256
        # MD5 gives us h1, SHA256 gives us h2
        h1_bytes = hashlib.md5(item_bytes).digest()
        h2_bytes = hashlib.sha256(item_bytes).digest()
        
        # Convert first 8 bytes of each hash to integers
        h1 = struct.unpack('<Q', h1_bytes[:8])[0]
        h2 = struct.unpack('<Q', h2_bytes[:8])[0]
        
        # Double hashing: h(x, i) = (h1(x) + i * h2(x)) mod width
        combined = (h1 + seed * h2) % self._width
        
        return combined
    
    def _get_positions(self, item: str) -> list[int]:
        """
        Get all hash positions for an item.
        
        Args:
            item: Item to hash
            
        Returns:
            List of column indices, one per row
            
        Example:
            >>> cms = CountMinSketch(width=100, depth=3)
            >>> cms._get_positions("apple")
            [42, 17, 89]
        """
        return [self._hash(item, i) for i in range(self._depth)]
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add item to the sketch.
        
        Increments the counter at each of the d hash positions.
        
        Args:
            item: Item to add
            count: Number of occurrences (default 1)
            
        Raises:
            ValueError: If count is negative
            
        Example:
            >>> cms = CountMinSketch()
            >>> cms.add("apple")  # Add 1 occurrence
            >>> cms.add("apple", count=5)  # Add 5 more
            >>> cms.query("apple")
            6
            
        Complexity:
            Time: O(d) where d is the depth
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        
        if count == 0:
            return
        
        # Increment counter at each hash position
        for row in range(self._depth):
            col = self._hash(item, row)
            self._table[row][col] += count
        
        # Update total count
        self._total_count += count
        
        logger.debug(f"Added '{item}' with count {count}")
    
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
            
        Complexity:
            Time: O(d) where d is the depth
        """
        min_count = float('inf')
        
        for row in range(self._depth):
            col = self._hash(item, row)
            min_count = min(min_count, self._table[row][col])
        
        return int(min_count)
    
    def add_all(self, items: Iterable[str]) -> None:
        """
        Add multiple items to the sketch.
        
        Args:
            items: Iterable of items to add
            
        Example:
            >>> cms = CountMinSketch()
            >>> cms.add_all(["apple", "banana", "apple", "cherry"])
            >>> cms.query("apple")
            2
        """
        for item in items:
            self.add(item)
    
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
            
        Example:
            >>> cms1 = CountMinSketch(epsilon=0.1, delta=0.01)
            >>> cms2 = CountMinSketch(epsilon=0.1, delta=0.01)
            >>> cms1.add("apple", count=10)
            >>> cms2.add("apple", count=5)
            >>> merged = cms1.merge(cms2)
            >>> merged.query("apple")
            15
            
        Complexity:
            Time: O(w × d)
            Space: O(w × d) for new sketch
        """
        if self._width != other._width or self._depth != other._depth:
            raise ValueError(
                f"Dimension mismatch: ({self._width}, {self._depth}) vs "
                f"({other._width}, {other._depth})"
            )
        
        # Create new sketch with same dimensions
        result = CountMinSketch(
            width=self._width,
            depth=self._depth,
            epsilon=self._epsilon,
            delta=self._delta
        )
        
        # Sum all counters
        for row in range(self._depth):
            for col in range(self._width):
                result._table[row][col] = (
                    self._table[row][col] + other._table[row][col]
                )
        
        # Sum total counts
        result._total_count = self._total_count + other._total_count
        
        return result
    
    def __add__(self, other: CountMinSketch) -> CountMinSketch:
        """
        Support + operator for merging.
        
        Example:
            >>> merged = cms1 + cms2
        """
        return self.merge(other)
    
    def inner_product(self, other: CountMinSketch) -> int:
        """
        Estimate inner product of two sketches.
        
        The inner product Σ(a_i × b_i) estimates the similarity
        between two frequency distributions.
        
        Args:
            other: Another CountMinSketch with same dimensions
            
        Returns:
            Estimated inner product
            
        Raises:
            ValueError: If dimensions don't match
            
        Example:
            >>> # Measure overlap between two word distributions
            >>> cms1.add_all(doc1_words)
            >>> cms2.add_all(doc2_words)
            >>> similarity = cms1.inner_product(cms2)
            
        Complexity:
            Time: O(w × d)
        """
        if self._width != other._width or self._depth != other._depth:
            raise ValueError("Dimension mismatch for inner product")
        
        # Inner product is minimum of row-wise dot products
        min_product = float('inf')
        
        for row in range(self._depth):
            row_product = sum(
                self._table[row][col] * other._table[row][col]
                for col in range(self._width)
            )
            min_product = min(min_product, row_product)
        
        return int(min_product)
    
    def clear(self) -> None:
        """
        Reset all counters to zero.
        
        Example:
            >>> cms.add("test", count=100)
            >>> cms.clear()
            >>> cms.query("test")
            0
        """
        for row in range(self._depth):
            for col in range(self._width):
                self._table[row][col] = 0
        self._total_count = 0
    
    def copy(self) -> CountMinSketch:
        """
        Create a deep copy of this sketch.
        
        Returns:
            New CountMinSketch with identical state
        """
        result = CountMinSketch(
            width=self._width,
            depth=self._depth,
            epsilon=self._epsilon,
            delta=self._delta
        )
        
        for row in range(self._depth):
            for col in range(self._width):
                result._table[row][col] = self._table[row][col]
        
        result._total_count = self._total_count
        
        return result
    
    def visualise(self, max_width: int = 40) -> str:
        """
        Create ASCII visualisation of the counter table.
        
        Args:
            max_width: Maximum display width
            
        Returns:
            ASCII representation of sketch state
            
        Example:
            >>> print(cms.visualise())
            Count-Min Sketch (272 × 7)
            ─────────────────────────
            Row 0: ▁▁▁▂▁▁▁▅▁▁▁▁▁▁▃▁...
            Row 1: ▁▁▁▁▁▂▁▁▁▁▅▁▁▁▁▁...
            ...
        """
        lines = [
            f"Count-Min Sketch ({self._width} × {self._depth})",
            "─" * 40
        ]
        
        # Find max value for scaling
        max_val = max(
            max(row) for row in self._table
        ) or 1
        
        # Unicode block characters for visualisation
        blocks = " ▁▂▃▄▅▆▇█"
        
        for row_idx, row in enumerate(self._table):
            # Sample if row is wider than display
            if self._width > max_width:
                step = self._width // max_width
                samples = [row[i] for i in range(0, self._width, step)]
            else:
                samples = row
            
            # Convert to block characters
            chars = []
            for val in samples[:max_width]:
                level = min(8, int(val / max_val * 8))
                chars.append(blocks[level])
            
            suffix = "..." if self._width > max_width else ""
            lines.append(f"Row {row_idx}: {''.join(chars)}{suffix}")
        
        lines.extend([
            "─" * 40,
            f"Total count: {self._total_count:,}",
            f"Error bound: ±{self._epsilon * self._total_count:.1f}"
        ])
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CountMinSketch(width={self._width}, depth={self._depth}, "
            f"epsilon={self._epsilon}, delta={self._delta}, "
            f"total_count={self._total_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: HEAVY HITTERS TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class HeavyHittersTracker:
    """
    Track frequent items (heavy hitters) in a data stream.
    
    A heavy hitter is an item that appears with frequency above
    a threshold fraction of the total stream.
    
    This implementation combines Count-Min Sketch for frequency
    estimation with a candidate set for potential heavy hitters.
    
    Attributes:
        threshold: Minimum frequency fraction to be a heavy hitter
        cms: Underlying Count-Min Sketch
        
    Example:
        >>> tracker = HeavyHittersTracker(threshold=0.05)
        >>> for word in document.split():
        ...     tracker.add(word)
        >>> for item, count in tracker.get_heavy_hitters():
        ...     print(f"{item}: {count}")
        
    Complexity:
        Add: O(d) amortised
        Get heavy hitters: O(|candidates| × d)
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
            threshold: Minimum frequency fraction to be considered
                       a heavy hitter (e.g. 0.01 = 1%)
            epsilon: CMS error factor (default: threshold/2)
            delta: CMS failure probability
            
        Raises:
            ValueError: If threshold not in (0, 1)
            
        Example:
            >>> # Track items appearing >5% of the time
            >>> tracker = HeavyHittersTracker(threshold=0.05)
        """
        if threshold <= 0 or threshold >= 1:
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold}"
            )
        
        self._threshold = threshold
        
        # Use epsilon = threshold/2 to ensure accurate heavy hitter detection
        if epsilon is None:
            epsilon = threshold / 2
        
        self._cms = CountMinSketch(epsilon=epsilon, delta=delta)
        
        # Candidate set for potential heavy hitters
        self._candidates: set[str] = set()
        
        logger.debug(
            f"Initialised HeavyHittersTracker: threshold={threshold}"
        )
    
    @property
    def threshold(self) -> float:
        """Return the heavy hitter threshold."""
        return self._threshold
    
    @property
    def cms(self) -> CountMinSketch:
        """Return the underlying Count-Min Sketch."""
        return self._cms
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add item to the tracker.
        
        Updates the sketch and checks if item becomes a heavy hitter
        candidate.
        
        Args:
            item: Item to add
            count: Number of occurrences
            
        Example:
            >>> tracker.add("apple")
            >>> tracker.add("apple", count=10)
        """
        # Add to sketch
        self._cms.add(item, count)
        
        # Check if this item might be a heavy hitter
        estimated = self._cms.query(item)
        current_threshold = self._threshold * self._cms.total_count
        
        if estimated >= current_threshold:
            self._candidates.add(item)
    
    def add_all(self, items: Iterable[str]) -> None:
        """
        Add multiple items to the tracker.
        
        Args:
            items: Iterable of items to add
        """
        for item in items:
            self.add(item)
    
    def get_heavy_hitters(self) -> list[tuple[str, int]]:
        """
        Return items that are likely heavy hitters.
        
        Filters candidates based on current threshold and
        returns sorted by estimated count.
        
        Returns:
            List of (item, estimated_count) tuples, sorted by count descending
            
        Example:
            >>> heavy = tracker.get_heavy_hitters()
            >>> for item, count in heavy:
            ...     print(f"{item}: {count}")
            the: 1523
            and: 892
            is: 654
        """
        current_threshold = self._threshold * self._cms.total_count
        
        # Filter and estimate counts for all candidates
        results: list[tuple[str, int]] = []
        
        for item in self._candidates:
            estimated = self._cms.query(item)
            if estimated >= current_threshold:
                results.append((item, estimated))
        
        # Sort by count descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def is_heavy_hitter(self, item: str) -> bool:
        """
        Check if an item is currently a heavy hitter.
        
        Args:
            item: Item to check
            
        Returns:
            True if item exceeds heavy hitter threshold
        """
        estimated = self._cms.query(item)
        current_threshold = self._threshold * self._cms.total_count
        return estimated >= current_threshold
    
    def get_frequency(self, item: str) -> float:
        """
        Get estimated frequency of an item as a fraction.
        
        Args:
            item: Item to query
            
        Returns:
            Estimated frequency in [0, 1]
        """
        if self._cms.total_count == 0:
            return 0.0
        return self._cms.query(item) / self._cms.total_count
    
    def clear(self) -> None:
        """Reset the tracker."""
        self._cms.clear()
        self._candidates.clear()
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HeavyHittersTracker(threshold={self._threshold}, "
            f"candidates={len(self._candidates)}, "
            f"total_count={self._cms.total_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CONSERVATIVE UPDATE VARIANT
# ═══════════════════════════════════════════════════════════════════════════════

class ConservativeCountMinSketch(CountMinSketch):
    """
    Count-Min Sketch with conservative update.
    
    Instead of incrementing all counters by the same amount,
    conservative update only increments counters to reach the
    new minimum estimate. This reduces overestimation while
    maintaining the no-underestimation guarantee.
    
    The idea: if current estimates for positions are [5, 8, 6]
    and we add 1, standard CMS gives [6, 9, 7]. Conservative
    update recognises the true count must be at most 5+1=6,
    so it updates to [6, 8, 6] — only updating counters at or
    below the new target value.
    
    Trade-offs:
    - Slightly slower updates (need to read before write)
    - Better accuracy (reduced overestimation)
    - Same space complexity
    
    Example:
        >>> cms = ConservativeCountMinSketch(epsilon=0.01)
        >>> cms.add("apple", count=10)
        >>> # Less overestimation than standard CMS
    """
    
    def add(self, item: str, count: int = 1) -> None:
        """
        Add item with conservative update.
        
        Only increments counters to reach the new minimum estimate,
        reducing overestimation while preserving accuracy guarantees.
        
        Args:
            item: Item to add
            count: Number of occurrences
            
        Complexity:
            Time: O(d) where d is the depth (2 passes)
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        
        if count == 0:
            return
        
        # Get all hash positions
        positions = self._get_positions(item)
        
        # Find current minimum
        current_min = min(
            self._table[row][col]
            for row, col in enumerate(positions)
        )
        
        # Target value after update
        target = current_min + count
        
        # Conservative update: set each counter to max(current, target)
        for row, col in enumerate(positions):
            current = self._table[row][col]
            self._table[row][col] = max(current, target)
        
        # Update total count
        self._total_count += count
        
        logger.debug(
            f"Conservative add '{item}': count={count}, "
            f"min={current_min} → target={target}"
        )
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"ConservativeCountMinSketch(width={self._width}, "
            f"depth={self._depth}, total_count={self._total_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COUNT-MEAN-MIN SKETCH (BONUS)
# ═══════════════════════════════════════════════════════════════════════════════

class CountMeanMinSketch(CountMinSketch):
    """
    Count-Mean-Min Sketch for bias-corrected estimation.
    
    Standard CMS overestimates due to hash collisions. The Count-Mean-Min
    variant subtracts an estimate of the collision noise to provide
    more accurate (though potentially underestimating) queries.
    
    For each row, it estimates the noise as:
        noise_i = (total_count - counter_i) / (width - 1)
    
    The query returns the median of (counter_i - noise_i) across all rows.
    
    Trade-offs:
    - May underestimate (unlike standard CMS)
    - Better accuracy on average
    - Good for scenarios where underestimation is acceptable
    
    Example:
        >>> cms = CountMeanMinSketch(epsilon=0.01)
        >>> # More accurate for high-collision scenarios
    """
    
    def query(self, item: str) -> int:
        """
        Estimate count using bias-corrected mean-min.
        
        Subtracts estimated collision noise from each counter,
        then returns the median of corrected values.
        
        Args:
            item: Item to query
            
        Returns:
            Bias-corrected estimated count (may be negative for rare items)
            
        Note:
            Unlike standard CMS, this may underestimate or return
            negative values for items with very low true counts.
        """
        corrected_estimates: list[float] = []
        
        for row in range(self._depth):
            col = self._hash(item, row)
            counter = self._table[row][col]
            
            # Estimate noise from collisions
            # noise = (total - counter) / (width - 1)
            if self._width > 1:
                noise = (self._total_count - counter) / (self._width - 1)
            else:
                noise = 0
            
            corrected = counter - noise
            corrected_estimates.append(corrected)
        
        # Return median of corrected estimates
        corrected_estimates.sort()
        mid = len(corrected_estimates) // 2
        
        if len(corrected_estimates) % 2 == 0:
            median = (
                corrected_estimates[mid - 1] + corrected_estimates[mid]
            ) / 2
        else:
            median = corrected_estimates[mid]
        
        # Return max with 0 to avoid negative counts
        return max(0, int(median))
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"CountMeanMinSketch(width={self._width}, "
            f"depth={self._depth}, total_count={self._total_count})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: STREAMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def stream_from_file(
    filepath: str,
    tokenise: bool = True
) -> Iterator[str]:
    """
    Create a token stream from a text file.
    
    Args:
        filepath: Path to text file
        tokenise: If True, split into words; if False, yield lines
        
    Yields:
        Tokens (words) or lines from the file
        
    Example:
        >>> for word in stream_from_file("document.txt"):
        ...     cms.add(word)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if tokenise:
                # Simple tokenisation: lowercase, alphanumeric only
                words = line.lower().split()
                for word in words:
                    cleaned = ''.join(c for c in word if c.isalnum())
                    if cleaned:
                        yield cleaned
            else:
                yield line.strip()


def compare_cms_variants(
    items: list[str],
    true_counts: dict[str, int],
    epsilon: float = 0.01,
    delta: float = 0.001
) -> dict[str, dict[str, float]]:
    """
    Compare accuracy of different CMS variants.
    
    Args:
        items: List of items in stream order
        true_counts: Dictionary of true counts for validation
        epsilon: Error parameter
        delta: Failure probability
        
    Returns:
        Dictionary mapping variant name to error statistics
        
    Example:
        >>> results = compare_cms_variants(words, actual_counts)
        >>> print(results["standard"]["mean_error"])
    """
    # Create each variant
    variants = {
        "standard": CountMinSketch(epsilon=epsilon, delta=delta),
        "conservative": ConservativeCountMinSketch(epsilon=epsilon, delta=delta),
        "count_mean_min": CountMeanMinSketch(epsilon=epsilon, delta=delta)
    }
    
    # Add all items to each variant
    for item in items:
        for cms in variants.values():
            cms.add(item)
    
    # Calculate error statistics
    results: dict[str, dict[str, float]] = {}
    
    for name, cms in variants.items():
        errors: list[float] = []
        
        for item, true_count in true_counts.items():
            estimated = cms.query(item)
            error = abs(estimated - true_count)
            errors.append(error)
        
        results[name] = {
            "mean_error": sum(errors) / len(errors) if errors else 0,
            "max_error": max(errors) if errors else 0,
            "total_count": cms.total_count
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_cms_basic() -> None:
    """Test basic CMS operations."""
    cms = CountMinSketch(epsilon=0.1, delta=0.01)
    
    # Add items
    cms.add("apple", count=10)
    cms.add("banana", count=5)
    cms.add("cherry", count=3)
    
    # Query should never underestimate
    assert cms.query("apple") >= 10, "CMS underestimated apple"
    assert cms.query("banana") >= 5, "CMS underestimated banana"
    assert cms.query("cherry") >= 3, "CMS underestimated cherry"
    
    # Items never added should have low counts
    # (might be > 0 due to collisions)
    assert cms.query("durian") >= 0, "CMS returned negative"
    
    # Total count
    assert cms.total_count == 18, f"Wrong total: {cms.total_count}"
    
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
        assert estimated >= true_count, (
            f"Underestimation: {estimated} < {true_count}"
        )
        
        # Should not overestimate too much (with high probability)
        if estimated > true_count + epsilon * total:
            errors += 1
    
    # Allow delta fraction of errors (with some slack)
    max_allowed_errors = int(delta * len(items) * 2)
    assert errors <= max_allowed_errors, (
        f"Too many errors: {errors} > {max_allowed_errors}"
    )
    
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
    
    assert merged.query("apple") >= 13, "Merge failed for apple"
    assert merged.query("banana") >= 5, "Merge failed for banana"
    assert merged.query("cherry") >= 7, "Merge failed for cherry"
    assert merged.total_count == 25, f"Wrong total: {merged.total_count}"
    
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
    
    # "common" should definitely be heavy (100/160 = 62.5%)
    assert "common" in heavy_items, "common should be heavy hitter"
    
    # "medium" should be heavy (50/160 = 31.25%)
    assert "medium" in heavy_items, "medium should be heavy hitter"
    
    logger.info("✓ Heavy hitters tests passed")


def test_cms_dimensions() -> None:
    """Test CMS dimension calculations."""
    e = math.e
    
    # Test standard parameters
    cms = CountMinSketch(epsilon=0.01, delta=0.001)
    
    expected_width = math.ceil(e / 0.01)
    expected_depth = math.ceil(math.log(1 / 0.001))
    
    assert cms.width == expected_width, (
        f"Wrong width: {cms.width} != {expected_width}"
    )
    assert cms.depth == expected_depth, (
        f"Wrong depth: {cms.depth} != {expected_depth}"
    )
    
    logger.info("✓ CMS dimension tests passed")


def test_conservative_update() -> None:
    """Test conservative update variant."""
    standard = CountMinSketch(epsilon=0.01, delta=0.001)
    conservative = ConservativeCountMinSketch(epsilon=0.01, delta=0.001)
    
    # Add same items to both
    items = ["apple"] * 100 + ["banana"] * 50 + ["cherry"] * 25
    for item in items:
        standard.add(item)
        conservative.add(item)
    
    # Both should not underestimate
    assert standard.query("apple") >= 100
    assert conservative.query("apple") >= 100
    
    # Conservative should have same or less overestimation
    standard_error = standard.query("apple") - 100
    conservative_error = conservative.query("apple") - 100
    
    assert conservative_error <= standard_error + 1, (
        "Conservative should reduce overestimation"
    )
    
    logger.info("✓ Conservative update tests passed")


def test_inner_product() -> None:
    """Test inner product estimation."""
    cms1 = CountMinSketch(epsilon=0.1, delta=0.01)
    cms2 = CountMinSketch(epsilon=0.1, delta=0.01)
    
    # Create similar distributions
    cms1.add("apple", count=10)
    cms1.add("banana", count=5)
    
    cms2.add("apple", count=10)
    cms2.add("banana", count=5)
    
    # Inner product should be approximately 10*10 + 5*5 = 125
    ip = cms1.inner_product(cms2)
    assert ip >= 125, f"Inner product too low: {ip}"
    
    logger.info("✓ Inner product tests passed")


def run_all_tests() -> None:
    """Run all test cases."""
    logger.info("Running Count-Min Sketch tests...")
    logger.info("═" * 50)
    
    test_cms_dimensions()
    test_cms_basic()
    test_cms_accuracy()
    test_cms_merge()
    test_heavy_hitters()
    test_conservative_update()
    test_inner_product()
    
    logger.info("═" * 50)
    logger.info("All tests passed!")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_basic_cms() -> None:
    """Demonstrate basic Count-Min Sketch operations."""
    logger.info("Basic Count-Min Sketch Demonstration")
    logger.info("═" * 50)
    
    # Create sketch with 1% error, 0.1% failure probability
    cms = CountMinSketch(epsilon=0.01, delta=0.001)
    logger.info(f"Created CMS: {cms}")
    logger.info(f"  Width: {cms.width}")
    logger.info(f"  Depth: {cms.depth}")
    logger.info(f"  Space: ~{cms.space_bytes / 1024:.1f} KB")
    
    # Add items
    logger.info("\nAdding items...")
    cms.add("python", count=100)
    cms.add("java", count=75)
    cms.add("javascript", count=50)
    cms.add("rust", count=25)
    cms.add("go", count=10)
    
    # Query items
    logger.info("\nQuerying items:")
    for item in ["python", "java", "javascript", "rust", "go", "cobol"]:
        count = cms.query(item)
        logger.info(f"  {item}: {count}")
    
    logger.info(f"\nTotal count: {cms.total_count}")
    logger.info(f"Error bound: ±{cms.epsilon * cms.total_count:.1f}")


def demonstrate_heavy_hitters() -> None:
    """Demonstrate heavy hitters detection."""
    logger.info("\nHeavy Hitters Detection")
    logger.info("═" * 50)
    
    # Sample text
    text = """
    The quick brown fox jumps over the lazy dog.
    The dog was not amused by the fox.
    The fox ran away quickly but the dog gave chase.
    In the end the fox escaped into the forest.
    The moral of the story is that the fox is clever.
    The dog learned to be more vigilant.
    But the fox would return another day.
    """
    
    # Create tracker with 5% threshold
    tracker = HeavyHittersTracker(threshold=0.05)
    
    # Process words
    words = text.lower().split()
    for word in words:
        cleaned = "".join(c for c in word if c.isalnum())
        if cleaned:
            tracker.add(cleaned)
    
    logger.info(f"Processed {tracker.cms.total_count} words")
    logger.info(f"Tracking heavy hitters (>5% frequency)")
    logger.info("\nHeavy hitters:")
    
    for item, count in tracker.get_heavy_hitters():
        freq = count / tracker.cms.total_count * 100
        logger.info(f"  '{item}': {count} ({freq:.1f}%)")


def demonstrate_distributed_counting() -> None:
    """Demonstrate distributed counting with merge."""
    logger.info("\nDistributed Counting")
    logger.info("═" * 50)
    
    # Simulate 3 nodes counting different data streams
    node1 = CountMinSketch(epsilon=0.1, delta=0.01)
    node2 = CountMinSketch(epsilon=0.1, delta=0.01)
    node3 = CountMinSketch(epsilon=0.1, delta=0.01)
    
    # Node 1: process stream 1
    for _ in range(100):
        node1.add("error_404")
    for _ in range(50):
        node1.add("error_500")
    
    # Node 2: process stream 2
    for _ in range(80):
        node2.add("error_404")
    for _ in range(30):
        node2.add("error_503")
    
    # Node 3: process stream 3
    for _ in range(120):
        node3.add("error_404")
    for _ in range(20):
        node3.add("error_500")
    
    # Merge all nodes
    global_cms = node1 + node2 + node3
    
    logger.info("Individual node counts:")
    logger.info(f"  Node 1: {node1.total_count} events")
    logger.info(f"  Node 2: {node2.total_count} events")
    logger.info(f"  Node 3: {node3.total_count} events")
    
    logger.info(f"\nMerged global view: {global_cms.total_count} events")
    logger.info("\nGlobal error frequencies:")
    
    for error in ["error_404", "error_500", "error_503"]:
        count = global_cms.query(error)
        freq = count / global_cms.total_count * 100
        logger.info(f"  {error}: {count} ({freq:.1f}%)")


def demonstrate_variant_comparison() -> None:
    """Compare different CMS variants."""
    logger.info("\nCMS Variant Comparison")
    logger.info("═" * 50)
    
    # Create test data with known counts
    true_counts = {
        "common": 1000,
        "frequent": 500,
        "moderate": 100,
        "rare": 10,
        "very_rare": 1
    }
    
    # Build stream
    stream: list[str] = []
    for item, count in true_counts.items():
        stream.extend([item] * count)
    
    # Create each variant
    standard = CountMinSketch(epsilon=0.01, delta=0.001)
    conservative = ConservativeCountMinSketch(epsilon=0.01, delta=0.001)
    mean_min = CountMeanMinSketch(epsilon=0.01, delta=0.001)
    
    # Add all items
    for item in stream:
        standard.add(item)
        conservative.add(item)
        mean_min.add(item)
    
    # Compare estimates
    logger.info("Item          | True | Standard | Conservative | Mean-Min")
    logger.info("-" * 65)
    
    for item, true_count in sorted(
        true_counts.items(), key=lambda x: -x[1]
    ):
        std_est = standard.query(item)
        con_est = conservative.query(item)
        mm_est = mean_min.query(item)
        
        logger.info(
            f"{item:13} | {true_count:4} | {std_est:8} | {con_est:12} | {mm_est:8}"
        )
    
    # Calculate total errors
    logger.info("\nTotal absolute error:")
    
    for name, cms in [
        ("Standard", standard),
        ("Conservative", conservative),
        ("Mean-Min", mean_min)
    ]:
        total_error = sum(
            abs(cms.query(item) - count)
            for item, count in true_counts.items()
        )
        logger.info(f"  {name}: {total_error}")


def demonstrate_visualisation() -> None:
    """Demonstrate CMS visualisation."""
    logger.info("\nCMS Visualisation")
    logger.info("═" * 50)
    
    cms = CountMinSketch(width=50, depth=4)
    
    # Add items with varying frequencies
    import random
    random.seed(42)
    
    for _ in range(100):
        cms.add("high_freq")
    for _ in range(50):
        cms.add("medium_freq")
    for _ in range(10):
        cms.add("low_freq")
    
    # Add some noise
    for i in range(100):
        cms.add(f"noise_{random.randint(0, 1000)}")
    
    logger.info("\n" + cms.visualise())


def run_all_demonstrations() -> None:
    """Run all demonstrations."""
    demonstrate_basic_cms()
    demonstrate_heavy_hitters()
    demonstrate_distributed_counting()
    demonstrate_variant_comparison()
    demonstrate_visualisation()
    
    logger.info("\n" + "═" * 50)
    logger.info("All demonstrations complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Count-Min Sketch Solution"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run all tests"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations"
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
        run_all_demonstrations()
    else:
        # Default: run both
        run_all_tests()
        print()
        run_all_demonstrations()


if __name__ == "__main__":
    main()
