#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 4 Practice: Medium Exercise 3 — Binary Heap
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 30 minutes
TOPICS: Heaps, priority queues, array representation

TASK
────
Implement a binary min-heap from scratch.

© 2025 Antonio Clim. All rights reserved.
"""

from __future__ import annotations
from typing import TypeVar, Generic

T = TypeVar("T")


class MinHeap(Generic[T]):
    """
    A binary min-heap implementation.
    
    The heap is stored as an array where:
    - Parent of node at index i: (i - 1) // 2
    - Left child of node at index i: 2 * i + 1
    - Right child of node at index i: 2 * i + 2
    
    Example:
        >>> heap = MinHeap[int]()
        >>> heap.push(3)
        >>> heap.push(1)
        >>> heap.push(2)
        >>> heap.pop()
        1
    """
    
    def __init__(self) -> None:
        self._data: list[T] = []
    
    def _parent(self, i: int) -> int:
        """Return parent index."""
        return (i - 1) // 2
    
    def _left(self, i: int) -> int:
        """Return left child index."""
        return 2 * i + 1
    
    def _right(self, i: int) -> int:
        """Return right child index."""
        return 2 * i + 2
    
    def _swap(self, i: int, j: int) -> None:
        """Swap elements at indices i and j."""
        self._data[i], self._data[j] = self._data[j], self._data[i]
    
    def _bubble_up(self, i: int) -> None:
        """
        Move element at index i up to restore heap property.
        
        While the element is smaller than its parent, swap with parent.
        """
        # TODO: Implement bubble up
        pass
    
    def _bubble_down(self, i: int) -> None:
        """
        Move element at index i down to restore heap property.
        
        While the element is larger than a child, swap with smaller child.
        """
        # TODO: Implement bubble down
        pass
    
    def push(self, item: T) -> None:
        """
        Add an item to the heap.
        
        Complexity: O(log n)
        """
        # TODO: Implement push
        # 1. Append item to end
        # 2. Bubble up from last position
        pass
    
    def pop(self) -> T:
        """
        Remove and return the minimum item.
        
        Complexity: O(log n)
        
        Raises:
            IndexError: If heap is empty
        """
        # TODO: Implement pop
        # 1. Save minimum (root)
        # 2. Move last element to root
        # 3. Bubble down from root
        # 4. Return saved minimum
        pass
    
    def peek(self) -> T:
        """
        Return the minimum item without removing it.
        
        Complexity: O(1)
        """
        if not self._data:
            raise IndexError("Heap is empty")
        return self._data[0]
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __bool__(self) -> bool:
        return len(self._data) > 0


def heapsort(items: list[T]) -> list[T]:
    """
    Sort items using a heap.
    
    Complexity: O(n log n)
    
    Args:
        items: List of comparable items
        
    Returns:
        Sorted list (ascending order)
    """
    # TODO: Implement heapsort using MinHeap
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_push_pop() -> None:
    heap = MinHeap[int]()
    heap.push(5)
    heap.push(3)
    heap.push(7)
    heap.push(1)
    
    assert heap.pop() == 1
    assert heap.pop() == 3
    assert heap.pop() == 5
    assert heap.pop() == 7


def test_peek() -> None:
    heap = MinHeap[int]()
    heap.push(10)
    heap.push(5)
    
    assert heap.peek() == 5
    assert len(heap) == 2  # peek doesn't remove


def test_heapsort() -> None:
    items = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    sorted_items = heapsort(items)
    assert sorted_items == [1, 1, 2, 3, 3, 4, 5, 5, 6, 9]


def test_empty_heap() -> None:
    heap = MinHeap[int]()
    assert len(heap) == 0
    assert not heap
    
    try:
        heap.pop()
        assert False, "Should raise IndexError"
    except IndexError:
        pass


if __name__ == "__main__":
    test_push_pop()
    test_peek()
    test_heapsort()
    test_empty_heap()
    print("All tests passed! ✓")
