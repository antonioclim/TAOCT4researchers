#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
04UNIT, Lab 2: Probabilistic Data Structures
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
- 03UNIT: Complexity analysis, amortised analysis
- 04UNIT Lab 1: Hash table concepts, collision handling
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
    Bloom filter for approximate membership queries.

    Properties:
        - No false negatives: if contains(x) is False, x was not added.
        - False positives are possible: if contains(x) is True, x may be present.

    The standard parameterisation uses:
        m = -n ln(p) / (ln 2)^2
        k = (m/n) ln 2

    Where:
        n is the expected number of inserted elements,
        p is the target false positive rate,
        m is the number of bits,
        k is the number of hash functions.
    """

    def __init__(self, *, size: int, num_hash_functions: int) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if num_hash_functions <= 0:
            raise ValueError("num_hash_functions must be positive")

        self.size: int = int(size)
        self.num_hash_functions: int = int(num_hash_functions)
        self._bits: bytearray = bytearray((0 for _ in range(self.size)))
        self._seen: set[str] = set()
        self.count: int = 0

    @classmethod
    def from_parameters(cls, *, expected_elements: int, false_positive_rate: float) -> "BloomFilter":
        if expected_elements <= 0:
            raise ValueError("expected_elements must be positive")
        if not (0.0 < false_positive_rate < 1.0):
            raise ValueError("false_positive_rate must be in (0, 1)")

        n = float(expected_elements)
        p = float(false_positive_rate)
        ln2 = math.log(2.0)

        m = int(math.ceil((-n * math.log(p)) / (ln2 * ln2)))
        k = int(max(1, round((m / n) * ln2)))
        return cls(size=m, num_hash_functions=k)

    def _digest(self, item: object) -> bytes:
        data = repr(item).encode("utf-8")
        return hashlib.sha256(data).digest()

    def _positions(self, item: object) -> list[int]:
        digest = self._digest(item)
        h1 = int.from_bytes(digest[:8], "big", signed=False)
        h2 = int.from_bytes(digest[8:16], "big", signed=False) or 1
        return [((h1 + i * h2) % self.size) for i in range(self.num_hash_functions)]

    def add(self, item: object) -> None:
        signature = self._digest(item).hex()
        for pos in self._positions(item):
            self._bits[pos] = 1

        if signature not in self._seen:
            self._seen.add(signature)
            self.count += 1

    def contains(self, item: object) -> bool:
        return all(self._bits[pos] == 1 for pos in self._positions(item))




class CountMinSketch:
    """
    Count-Min sketch for frequency estimation in streaming data.

    The sketch stores counts in a (depth × width) array. Each update hashes an
    element into one column per row and increments the corresponding counters.
    The estimate for an element is the minimum counter across rows.

    Parameterisation:
        width  = ceil(e / epsilon)
        depth  = ceil(ln(1 / delta))

    Where:
        epsilon controls the additive error bound,
        delta controls the probability of exceeding the error bound.
    """

    def __init__(self, *, width: int, depth: int) -> None:
        if width <= 0:
            raise ValueError("width must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")

        self.width: int = int(width)
        self.depth: int = int(depth)
        self._table: list[list[int]] = [[0 for _ in range(self.width)] for _ in range(self.depth)]
        self._seeds: list[int] = [0x9E3779B1 * (i + 1) for i in range(self.depth)]
        self.total_count: int = 0

    @classmethod
    def from_parameters(cls, *, epsilon: float, delta: float) -> "CountMinSketch":
        if not (0.0 < epsilon < 1.0):
            raise ValueError("epsilon must be in (0, 1)")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must be in (0, 1)")

        width = int(math.ceil(math.e / epsilon))
        depth = int(math.ceil(math.log(1.0 / delta)))
        return cls(width=width, depth=depth)

    def _hash(self, item: object, seed: int) -> int:
        data = (repr(item) + "|" + str(seed)).encode("utf-8")
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "big", signed=False)

    def update(self, item: object, count: int = 1) -> None:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return
        for row, seed in enumerate(self._seeds):
            col = self._hash(item, seed) % self.width
            self._table[row][col] += count
        self.total_count += count

    def estimate(self, item: object) -> int:
        estimates: list[int] = []
        for row, seed in enumerate(self._seeds):
            col = self._hash(item, seed) % self.width
            estimates.append(self._table[row][col])
        return min(estimates) if estimates else 0

    def merge(self, other: "CountMinSketch") -> "CountMinSketch":
        if self.width != other.width or self.depth != other.depth:
            raise ValueError("CountMinSketch dimensions must match for merge")
        merged = CountMinSketch(width=self.width, depth=self.depth)
        for r in range(self.depth):
            for c in range(self.width):
                merged._table[r][c] = self._table[r][c] + other._table[r][c]
        merged.total_count = self.total_count + other.total_count
        return merged

