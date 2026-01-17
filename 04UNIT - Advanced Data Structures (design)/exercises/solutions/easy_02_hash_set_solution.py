#!/usr/bin/env python3
"""
Solution: Simple Hash Set with Chaining
=======================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements a hash set data structure using separate chaining
for collision resolution. The implementation demonstrates fundamental
hashing concepts including hash functions, load factors and dynamic resizing.

Complexity Analysis:
    - Insert: O(1) average, O(n) worst case
    - Delete: O(1) average, O(n) worst case
    - Contains: O(1) average, O(n) worst case
    - Resize: O(n)

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class HashSet(Generic[T]):
    """
    A hash set implementation using separate chaining.
    
    Separate chaining handles collisions by storing all elements that hash
    to the same bucket in a linked list (here implemented as a Python list).
    The set automatically resizes when the load factor exceeds a threshold.
    
    Attributes:
        initial_capacity: Initial number of buckets.
        load_factor_threshold: Maximum load factor before resizing.
        _buckets: Internal bucket array.
        _size: Number of elements in the set.
    
    Examples:
        >>> s = HashSet[int]()
        >>> s.add(42)
        >>> s.add(17)
        >>> 42 in s
        True
        >>> len(s)
        2
    """
    initial_capacity: int = 16
    load_factor_threshold: float = 0.75
    _buckets: list[list[T]] = field(default_factory=list, repr=False)
    _size: int = field(default=0, repr=False)
    
    def __post_init__(self) -> None:
        """Initialise the bucket array."""
        if not self._buckets:
            self._buckets = [[] for _ in range(self.initial_capacity)]
        logger.debug(f"Initialised HashSet with {self.initial_capacity} buckets")
    
    def _hash(self, element: T) -> int:
        """
        Compute the bucket index for an element.
        
        Uses Python's built-in hash function, then maps to a valid
        bucket index using modular arithmetic.
        
        Args:
            element: The element to hash.
        
        Returns:
            The bucket index (0 to capacity-1).
        """
        return hash(element) % len(self._buckets)
    
    @property
    def capacity(self) -> int:
        """Return the current number of buckets."""
        return len(self._buckets)
    
    @property
    def load_factor(self) -> float:
        """
        Return the current load factor.
        
        The load factor is the ratio of elements to buckets.
        Higher load factors increase collision probability.
        """
        return self._size / self.capacity if self.capacity > 0 else 0.0
    
    def _resize(self, new_capacity: int) -> None:
        """
        Resize the bucket array and rehash all elements.
        
        Args:
            new_capacity: The new number of buckets.
        
        Time Complexity: O(n) where n is the number of elements.
        """
        logger.info(
            f"Resizing from {self.capacity} to {new_capacity} buckets "
            f"(load factor was {self.load_factor:.2f})"
        )
        
        old_buckets = self._buckets
        self._buckets = [[] for _ in range(new_capacity)]
        self._size = 0
        
        # Rehash all existing elements
        for bucket in old_buckets:
            for element in bucket:
                self._add_without_resize(element)
    
    def _add_without_resize(self, element: T) -> bool:
        """
        Add an element without triggering resize.
        
        Internal method used during resize operations.
        
        Args:
            element: The element to add.
        
        Returns:
            True if the element was added, False if it already existed.
        """
        index = self._hash(element)
        bucket = self._buckets[index]
        
        if element not in bucket:
            bucket.append(element)
            self._size += 1
            return True
        return False
    
    def add(self, element: T) -> bool:
        """
        Add an element to the set.
        
        If the element already exists, the set is unchanged.
        May trigger a resize if the load factor exceeds the threshold.
        
        Args:
            element: The element to add.
        
        Returns:
            True if the element was added, False if it already existed.
        
        Time Complexity: O(1) amortised average case.
        
        Examples:
            >>> s = HashSet[str]()
            >>> s.add('hello')
            True
            >>> s.add('hello')
            False
        """
        # Check if resize is needed
        if self.load_factor >= self.load_factor_threshold:
            self._resize(self.capacity * 2)
        
        result = self._add_without_resize(element)
        
        if result:
            logger.debug(f"Added element: {element}")
        
        return result
    
    def remove(self, element: T) -> bool:
        """
        Remove an element from the set.
        
        Args:
            element: The element to remove.
        
        Returns:
            True if the element was removed, False if it was not present.
        
        Time Complexity: O(1) average case, O(n) worst case.
        
        Examples:
            >>> s = HashSet[int]()
            >>> s.add(42)
            True
            >>> s.remove(42)
            True
            >>> s.remove(42)
            False
        """
        index = self._hash(element)
        bucket = self._buckets[index]
        
        try:
            bucket.remove(element)
            self._size -= 1
            logger.debug(f"Removed element: {element}")
            return True
        except ValueError:
            return False
    
    def contains(self, element: T) -> bool:
        """
        Check whether an element is in the set.
        
        Args:
            element: The element to search for.
        
        Returns:
            True if the element is present, False otherwise.
        
        Time Complexity: O(1) average case, O(n) worst case.
        """
        index = self._hash(element)
        return element in self._buckets[index]
    
    def __contains__(self, element: T) -> bool:
        """Support the 'in' operator."""
        return self.contains(element)
    
    def __len__(self) -> int:
        """Return the number of elements in the set."""
        return self._size
    
    def __iter__(self) -> Iterator[T]:
        """Iterate over all elements in the set."""
        for bucket in self._buckets:
            yield from bucket
    
    def clear(self) -> None:
        """Remove all elements from the set."""
        self._buckets = [[] for _ in range(self.initial_capacity)]
        self._size = 0
        logger.debug("Cleared all elements")
    
    def union(self, other: HashSet[T]) -> HashSet[T]:
        """
        Return a new set containing all elements from both sets.
        
        Args:
            other: Another HashSet.
        
        Returns:
            A new HashSet containing the union.
        
        Time Complexity: O(n + m) where n, m are the sizes of the sets.
        """
        result: HashSet[T] = HashSet()
        
        for element in self:
            result.add(element)
        
        for element in other:
            result.add(element)
        
        return result
    
    def intersection(self, other: HashSet[T]) -> HashSet[T]:
        """
        Return a new set containing elements present in both sets.
        
        Args:
            other: Another HashSet.
        
        Returns:
            A new HashSet containing the intersection.
        
        Time Complexity: O(min(n, m)) average case.
        """
        result: HashSet[T] = HashSet()
        
        # Iterate over the smaller set for efficiency
        smaller, larger = (self, other) if len(self) <= len(other) else (other, self)
        
        for element in smaller:
            if element in larger:
                result.add(element)
        
        return result
    
    def difference(self, other: HashSet[T]) -> HashSet[T]:
        """
        Return a new set containing elements in self but not in other.
        
        Args:
            other: Another HashSet.
        
        Returns:
            A new HashSet containing the difference.
        
        Time Complexity: O(n) average case.
        """
        result: HashSet[T] = HashSet()
        
        for element in self:
            if element not in other:
                result.add(element)
        
        return result
    
    def is_subset(self, other: HashSet[T]) -> bool:
        """
        Check if this set is a subset of another.
        
        Args:
            other: Another HashSet.
        
        Returns:
            True if all elements of self are in other.
        """
        return all(element in other for element in self)
    
    def bucket_distribution(self) -> dict[str, float | int]:
        """
        Return statistics about bucket utilisation.
        
        Useful for analysing hash function quality and collision rates.
        
        Returns:
            Dictionary with distribution statistics.
        """
        bucket_sizes = [len(b) for b in self._buckets]
        non_empty = sum(1 for s in bucket_sizes if s > 0)
        
        return {
            'total_buckets': self.capacity,
            'non_empty_buckets': non_empty,
            'empty_buckets': self.capacity - non_empty,
            'utilisation': non_empty / self.capacity if self.capacity > 0 else 0,
            'max_bucket_size': max(bucket_sizes) if bucket_sizes else 0,
            'avg_bucket_size': sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0,
            'load_factor': self.load_factor,
        }
    
    def __repr__(self) -> str:
        """Return a string representation of the set."""
        elements = list(self)[:5]  # Show first 5 elements
        if len(self) > 5:
            return f"HashSet({elements}... and {len(self) - 5} more)"
        return f"HashSet({elements})"


def demonstrate_hash_set() -> None:
    """Demonstrate hash set operations."""
    print("=" * 60)
    print("Hash Set Demonstration")
    print("=" * 60)
    
    # Example 1: Basic operations
    print("\n1. Basic Operations")
    print("-" * 40)
    
    numbers: HashSet[int] = HashSet()
    
    for i in [10, 20, 30, 40, 50]:
        numbers.add(i)
    
    print(f"   Set: {numbers}")
    print(f"   Size: {len(numbers)}")
    print(f"   Contains 30: {30 in numbers}")
    print(f"   Contains 99: {99 in numbers}")
    
    numbers.remove(30)
    print(f"   After removing 30: {list(numbers)}")
    
    # Example 2: Automatic resizing
    print("\n2. Automatic Resizing")
    print("-" * 40)
    
    growing_set: HashSet[int] = HashSet(initial_capacity=4)
    print(f"   Initial capacity: {growing_set.capacity}")
    
    for i in range(20):
        growing_set.add(i)
        if i in [3, 7, 15]:
            print(
                f"   After adding {i + 1} elements: "
                f"capacity={growing_set.capacity}, "
                f"load_factor={growing_set.load_factor:.2f}"
            )
    
    # Example 3: Set operations
    print("\n3. Set Operations")
    print("-" * 40)
    
    set_a: HashSet[str] = HashSet()
    set_b: HashSet[str] = HashSet()
    
    for word in ['apple', 'banana', 'cherry', 'date']:
        set_a.add(word)
    
    for word in ['cherry', 'date', 'elderberry', 'fig']:
        set_b.add(word)
    
    print(f"   Set A: {list(set_a)}")
    print(f"   Set B: {list(set_b)}")
    print(f"   Union: {list(set_a.union(set_b))}")
    print(f"   Intersection: {list(set_a.intersection(set_b))}")
    print(f"   A - B: {list(set_a.difference(set_b))}")
    
    # Example 4: Bucket distribution
    print("\n4. Bucket Distribution Analysis")
    print("-" * 40)
    
    large_set: HashSet[int] = HashSet(initial_capacity=32)
    for i in range(100):
        large_set.add(i * 7)  # Use multiples to see distribution
    
    stats = large_set.bucket_distribution()
    print(f"   Total buckets: {stats['total_buckets']}")
    print(f"   Non-empty buckets: {stats['non_empty_buckets']}")
    print(f"   Utilisation: {stats['utilisation']:.1%}")
    print(f"   Max bucket size: {stats['max_bucket_size']}")
    print(f"   Load factor: {stats['load_factor']:.2f}")
    
    # Example 5: String hashing
    print("\n5. String Elements")
    print("-" * 40)
    
    words: HashSet[str] = HashSet()
    text = "to be or not to be that is the question"
    
    for word in text.split():
        words.add(word)
    
    print(f"   Original words: {len(text.split())}")
    print(f"   Unique words: {len(words)}")
    print(f"   Unique set: {list(words)}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_hash_set()
