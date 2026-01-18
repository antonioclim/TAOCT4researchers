#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
04UNIT Practice: Easy Exercise 2 — Hash Set Implementation
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 15 minutes
TOPICS: Hash tables, collision handling

TASK
────
Implement a simple hash set using separate chaining.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations


class SimpleHashSet:
    """
    A simple hash set implementation using separate chaining.
    
    Example:
        >>> hs = SimpleHashSet(capacity=10)
        >>> hs.add("hello")
        >>> "hello" in hs
        True
        >>> "world" in hs
        False
    """
    
    def __init__(self, capacity: int = 16) -> None:
        """
        Initialise hash set with given capacity.
        
        Args:
            capacity: Number of buckets
        """
        self._capacity = capacity
        self._buckets: list[list[str]] = [[] for _ in range(capacity)]
        self._size = 0
    
    def _hash(self, item: str) -> int:
        """Return the bucket index for an item."""
        return hash(item) % self._capacity
    
    def add(self, item: str) -> None:
        """
        Add an item to the set.
        
        Args:
            item: Item to add
        """
        # TODO: Implement this
        # 1. Calculate the bucket index
        # 2. Check if item already exists in bucket
        # 3. If not, append to bucket and increment size
        pass
    
    def remove(self, item: str) -> bool:
        """
        Remove an item from the set.
        
        Args:
            item: Item to remove
            
        Returns:
            True if item was removed, False if not found
        """
        # TODO: Implement this
        pass
    
    def __contains__(self, item: str) -> bool:
        """Check if item is in the set."""
        # TODO: Implement this
        pass
    
    def __len__(self) -> int:
        """Return the number of items in the set."""
        return self._size
    
    def load_factor(self) -> float:
        """Return the current load factor."""
        return self._size / self._capacity


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_add_and_contains() -> None:
    hs = SimpleHashSet(10)
    hs.add("apple")
    hs.add("banana")
    hs.add("cherry")
    
    assert "apple" in hs
    assert "banana" in hs
    assert "cherry" in hs
    assert "date" not in hs


def test_duplicate_add() -> None:
    hs = SimpleHashSet(10)
    hs.add("apple")
    hs.add("apple")
    
    assert len(hs) == 1


def test_remove() -> None:
    hs = SimpleHashSet(10)
    hs.add("apple")
    
    assert hs.remove("apple") is True
    assert "apple" not in hs
    assert hs.remove("apple") is False


def test_load_factor() -> None:
    hs = SimpleHashSet(10)
    for i in range(5):
        hs.add(f"item_{i}")
    
    assert hs.load_factor() == 0.5


if __name__ == "__main__":
    test_add_and_contains()
    test_duplicate_add()
    test_remove()
    test_load_factor()
    print("All tests passed! ✓")
