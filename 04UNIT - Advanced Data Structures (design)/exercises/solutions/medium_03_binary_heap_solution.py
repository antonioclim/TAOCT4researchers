#!/usr/bin/env python3
"""
Solution: Binary Heap (Min-Heap) Implementation
===============================================

Week 4 - The Art of Computational Thinking for Researchers
Academy of Economic Studies, Bucharest

This solution implements a binary min-heap from scratch, demonstrating
the array-based heap representation, heap operations, and heapification.
Includes both iterative and recursive approaches.

Complexity Analysis:
    - Insert (push): O(log n)
    - Extract-min (pop): O(log n)
    - Peek: O(1)
    - Heapify (build heap): O(n)
    - Decrease-key: O(log n)

The heap property: For a min-heap, every parent node has a value
less than or equal to its children.

Author: Educational Materials
Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class MinHeap(Generic[T]):
    """
    A binary min-heap implementation using an array.
    
    The heap is stored as a complete binary tree in an array:
    - Root is at index 0
    - For node at index i:
      - Parent is at index (i - 1) // 2
      - Left child is at index 2 * i + 1
      - Right child is at index 2 * i + 2
    
    Attributes:
        _data: The underlying array storing heap elements.
        _key: Optional key function for custom comparison.
    
    Examples:
        >>> heap = MinHeap[int]()
        >>> heap.push(5)
        >>> heap.push(3)
        >>> heap.push(7)
        >>> heap.pop()
        3
    """
    _data: list[T] = field(default_factory=list)
    _key: Callable[[T], any] | None = None
    
    def _compare(self, a: T, b: T) -> bool:
        """
        Compare two elements.
        
        Returns True if a should be higher in the heap (smaller for min-heap).
        """
        if self._key:
            return self._key(a) < self._key(b)
        return a < b
    
    def _parent(self, index: int) -> int:
        """Return the parent index."""
        return (index - 1) // 2
    
    def _left_child(self, index: int) -> int:
        """Return the left child index."""
        return 2 * index + 1
    
    def _right_child(self, index: int) -> int:
        """Return the right child index."""
        return 2 * index + 2
    
    def _has_left_child(self, index: int) -> bool:
        """Check if node has a left child."""
        return self._left_child(index) < len(self._data)
    
    def _has_right_child(self, index: int) -> bool:
        """Check if node has a right child."""
        return self._right_child(index) < len(self._data)
    
    def _has_parent(self, index: int) -> bool:
        """Check if node has a parent."""
        return index > 0
    
    def _swap(self, i: int, j: int) -> None:
        """Swap elements at indices i and j."""
        self._data[i], self._data[j] = self._data[j], self._data[i]
    
    def _sift_up(self, index: int) -> None:
        """
        Restore heap property by moving element up.
        
        Called after insertion to restore the heap property.
        Compares the element with its parent and swaps if needed.
        
        Args:
            index: The index of the element to sift up.
        
        Time Complexity: O(log n)
        """
        while self._has_parent(index):
            parent_idx = self._parent(index)
            
            if self._compare(self._data[index], self._data[parent_idx]):
                self._swap(index, parent_idx)
                logger.debug(
                    f"Sift up: swapped indices {index} and {parent_idx}"
                )
                index = parent_idx
            else:
                break
    
    def _sift_down(self, index: int) -> None:
        """
        Restore heap property by moving element down.
        
        Called after extraction to restore the heap property.
        Compares the element with its smallest child and swaps if needed.
        
        Args:
            index: The index of the element to sift down.
        
        Time Complexity: O(log n)
        """
        while self._has_left_child(index):
            # Find the smaller child
            smaller_child_idx = self._left_child(index)
            
            if (
                self._has_right_child(index) and
                self._compare(
                    self._data[self._right_child(index)],
                    self._data[smaller_child_idx]
                )
            ):
                smaller_child_idx = self._right_child(index)
            
            # Check if we need to swap
            if self._compare(self._data[smaller_child_idx], self._data[index]):
                self._swap(index, smaller_child_idx)
                logger.debug(
                    f"Sift down: swapped indices {index} and {smaller_child_idx}"
                )
                index = smaller_child_idx
            else:
                break
    
    def push(self, item: T) -> None:
        """
        Insert an element into the heap.
        
        Adds the element at the end and sifts it up to its correct position.
        
        Args:
            item: The element to insert.
        
        Time Complexity: O(log n)
        
        Examples:
            >>> heap = MinHeap[int]()
            >>> heap.push(10)
            >>> heap.push(5)
            >>> heap.peek()
            5
        """
        self._data.append(item)
        self._sift_up(len(self._data) - 1)
        logger.debug(f"Pushed {item}, heap size: {len(self._data)}")
    
    def pop(self) -> T:
        """
        Remove and return the minimum element.
        
        Replaces the root with the last element and sifts down.
        
        Returns:
            The minimum element in the heap.
        
        Raises:
            IndexError: If the heap is empty.
        
        Time Complexity: O(log n)
        
        Examples:
            >>> heap = MinHeap[int]([3, 1, 4, 1, 5])
            >>> heap.pop()
            1
            >>> heap.pop()
            1
        """
        if not self._data:
            raise IndexError("Cannot pop from empty heap")
        
        # Store the minimum
        min_item = self._data[0]
        
        # Move last element to root and sift down
        last_item = self._data.pop()
        
        if self._data:
            self._data[0] = last_item
            self._sift_down(0)
        
        logger.debug(f"Popped {min_item}, heap size: {len(self._data)}")
        return min_item
    
    def peek(self) -> T:
        """
        Return the minimum element without removing it.
        
        Returns:
            The minimum element in the heap.
        
        Raises:
            IndexError: If the heap is empty.
        
        Time Complexity: O(1)
        """
        if not self._data:
            raise IndexError("Cannot peek empty heap")
        return self._data[0]
    
    def replace(self, item: T) -> T:
        """
        Pop the minimum and push a new item efficiently.
        
        More efficient than separate pop() and push() calls.
        
        Args:
            item: The new element to insert.
        
        Returns:
            The previous minimum element.
        
        Time Complexity: O(log n)
        """
        if not self._data:
            raise IndexError("Cannot replace in empty heap")
        
        min_item = self._data[0]
        self._data[0] = item
        self._sift_down(0)
        
        return min_item
    
    def pushpop(self, item: T) -> T:
        """
        Push a new item and pop the minimum efficiently.
        
        Args:
            item: The element to push.
        
        Returns:
            The minimum element (either the pushed item or previous min).
        
        Time Complexity: O(log n)
        """
        if self._data and self._compare(self._data[0], item):
            # Current root is smaller, so it will be returned
            item, self._data[0] = self._data[0], item
            self._sift_down(0)
        return item
    
    @classmethod
    def heapify(cls, items: list[T], key: Callable[[T], any] | None = None) -> MinHeap[T]:
        """
        Build a heap from an existing list in O(n) time.
        
        Uses the bottom-up heapification algorithm: start from the
        last non-leaf node and sift down each node.
        
        This is more efficient than inserting elements one by one (O(n log n)).
        
        Args:
            items: List of items to heapify.
            key: Optional key function for comparison.
        
        Returns:
            A new MinHeap containing all items.
        
        Time Complexity: O(n)
        
        Examples:
            >>> heap = MinHeap.heapify([3, 1, 4, 1, 5, 9, 2, 6])
            >>> heap.pop()
            1
        """
        heap = cls(_data=list(items), _key=key)
        
        # Start from the last non-leaf node and sift down
        # Last non-leaf is at index (n // 2) - 1
        for i in range(len(heap._data) // 2 - 1, -1, -1):
            heap._sift_down(i)
        
        logger.info(f"Heapified {len(items)} items")
        return heap
    
    def decrease_key(self, index: int, new_value: T) -> None:
        """
        Decrease the value at a given index.
        
        Used in algorithms like Dijkstra's where we need to update
        priorities in the heap.
        
        Args:
            index: The index of the element to update.
            new_value: The new (smaller) value.
        
        Raises:
            ValueError: If new_value is larger than current value.
        
        Time Complexity: O(log n)
        """
        if self._compare(self._data[index], new_value):
            raise ValueError("New value must be smaller than current value")
        
        self._data[index] = new_value
        self._sift_up(index)
    
    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._data)
    
    def __bool__(self) -> bool:
        """Return True if the heap is non-empty."""
        return bool(self._data)
    
    def __iter__(self) -> Iterator[T]:
        """
        Iterate over elements in sorted order.
        
        Note: This creates a copy and pops all elements.
        """
        heap_copy = MinHeap(_data=list(self._data), _key=self._key)
        while heap_copy:
            yield heap_copy.pop()
    
    def __repr__(self) -> str:
        """Return string representation."""
        if len(self._data) <= 10:
            return f"MinHeap({self._data})"
        return f"MinHeap([{self._data[0]}, ...], size={len(self._data)})"
    
    def is_valid(self) -> bool:
        """
        Verify the heap property holds.
        
        Returns:
            True if the heap property is satisfied for all nodes.
        """
        for i in range(len(self._data)):
            left = self._left_child(i)
            right = self._right_child(i)
            
            if left < len(self._data):
                if self._compare(self._data[left], self._data[i]):
                    return False
            
            if right < len(self._data):
                if self._compare(self._data[right], self._data[i]):
                    return False
        
        return True
    
    def visualise(self) -> str:
        """
        Create a text visualisation of the heap as a tree.
        
        Returns:
            A string representation of the heap tree.
        """
        if not self._data:
            return "(empty heap)"
        
        lines = []
        level = 0
        level_start = 0
        
        while level_start < len(self._data):
            level_size = 2 ** level
            level_end = min(level_start + level_size, len(self._data))
            level_items = self._data[level_start:level_end]
            
            spacing = " " * (2 ** (4 - level))
            line = spacing.join(str(item) for item in level_items)
            lines.append(f"Level {level}: {line}")
            
            level_start = level_end
            level += 1
        
        return "\n".join(lines)


def heap_sort(items: list[T]) -> list[T]:
    """
    Sort a list using heap sort.
    
    Args:
        items: The list to sort.
    
    Returns:
        A new sorted list.
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    heap = MinHeap.heapify(items)
    return [heap.pop() for _ in range(len(heap))]


def k_smallest(items: list[T], k: int) -> list[T]:
    """
    Find the k smallest elements using a heap.
    
    Args:
        items: The input list.
        k: Number of smallest elements to find.
    
    Returns:
        The k smallest elements in sorted order.
    
    Time Complexity: O(n log k)
    """
    if k >= len(items):
        return sorted(items)
    
    heap = MinHeap.heapify(items)
    return [heap.pop() for _ in range(k)]


def merge_k_sorted_lists(lists: list[list[T]]) -> list[T]:
    """
    Merge k sorted lists into one sorted list.
    
    Uses a min-heap to efficiently select the next smallest element.
    
    Args:
        lists: List of sorted lists.
    
    Returns:
        A single sorted list containing all elements.
    
    Time Complexity: O(N log k) where N is total elements, k is number of lists.
    """
    # Heap entries: (value, list_index, element_index)
    heap: MinHeap[tuple[T, int, int]] = MinHeap()
    
    # Initialise with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heap.push((lst[0], i, 0))
    
    result: list[T] = []
    
    while heap:
        value, list_idx, elem_idx = heap.pop()
        result.append(value)
        
        # Add next element from the same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heap.push((next_val, list_idx, elem_idx + 1))
    
    return result


def demonstrate_binary_heap() -> None:
    """Demonstrate binary heap operations."""
    print("=" * 60)
    print("Binary Heap (Min-Heap) Demonstration")
    print("=" * 60)
    
    # Example 1: Basic operations
    print("\n1. Basic Heap Operations")
    print("-" * 40)
    
    heap: MinHeap[int] = MinHeap()
    
    values = [7, 3, 9, 1, 5, 2, 8, 4, 6]
    print(f"   Inserting: {values}")
    
    for v in values:
        heap.push(v)
    
    print(f"   Heap after insertions: {heap}")
    print(f"   Minimum (peek): {heap.peek()}")
    print(f"   Heap valid: {heap.is_valid()}")
    
    print("\n   Popping elements in order:")
    while heap:
        print(f"      {heap.pop()}", end="")
    print()
    
    # Example 2: Heapify
    print("\n2. Heapify (O(n) Construction)")
    print("-" * 40)
    
    data = [15, 3, 17, 8, 25, 1, 7, 12, 5, 9]
    print(f"   Input: {data}")
    
    heap = MinHeap.heapify(data)
    print(f"   Heapified: {heap}")
    print(f"   Heap valid: {heap.is_valid()}")
    print(f"\n   Tree visualisation:\n{heap.visualise()}")
    
    # Example 3: Heap sort
    print("\n3. Heap Sort")
    print("-" * 40)
    
    unsorted = [64, 34, 25, 12, 22, 11, 90]
    print(f"   Unsorted: {unsorted}")
    
    sorted_list = heap_sort(unsorted)
    print(f"   Sorted:   {sorted_list}")
    
    # Example 4: K smallest elements
    print("\n4. Finding K Smallest Elements")
    print("-" * 40)
    
    numbers = [10, 4, 3, 8, 6, 2, 9, 1, 7, 5]
    k = 4
    print(f"   Numbers: {numbers}")
    print(f"   {k} smallest: {k_smallest(numbers, k)}")
    
    # Example 5: Merge k sorted lists
    print("\n5. Merging K Sorted Lists")
    print("-" * 40)
    
    sorted_lists = [
        [1, 4, 7, 10],
        [2, 5, 8, 11],
        [3, 6, 9, 12],
    ]
    print("   Lists:")
    for i, lst in enumerate(sorted_lists):
        print(f"      {i + 1}. {lst}")
    
    merged = merge_k_sorted_lists(sorted_lists)
    print(f"   Merged: {merged}")
    
    # Example 6: Custom key function
    print("\n6. Heap with Custom Key")
    print("-" * 40)
    
    tasks = [
        ("Low priority task", 3),
        ("High priority task", 1),
        ("Medium priority task", 2),
        ("Urgent task", 0),
    ]
    print("   Tasks (name, priority):")
    for task in tasks:
        print(f"      {task}")
    
    # Create heap ordered by priority (second element)
    task_heap: MinHeap[tuple[str, int]] = MinHeap(_key=lambda x: x[1])
    for task in tasks:
        task_heap.push(task)
    
    print("\n   Processing by priority:")
    while task_heap:
        name, priority = task_heap.pop()
        print(f"      [{priority}] {name}")
    
    # Example 7: pushpop and replace operations
    print("\n7. Efficient Push-Pop Operations")
    print("-" * 40)
    
    heap = MinHeap.heapify([5, 10, 15, 20])
    print(f"   Initial heap: {list(MinHeap.heapify([5, 10, 15, 20]))}")
    
    heap = MinHeap.heapify([5, 10, 15, 20])
    result = heap.pushpop(3)
    print(f"   pushpop(3): returned {result}, heap min now {heap.peek()}")
    
    heap = MinHeap.heapify([5, 10, 15, 20])
    result = heap.pushpop(7)
    print(f"   pushpop(7): returned {result}, heap min now {heap.peek()}")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_binary_heap()
