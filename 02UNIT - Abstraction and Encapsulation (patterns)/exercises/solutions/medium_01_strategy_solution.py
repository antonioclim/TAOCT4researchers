#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Medium Exercise 1 — Strategy Pattern SOLUTION
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Protocol
from dataclasses import dataclass


class SortStrategy(Protocol):
    """Protocol for sorting strategies."""
    
    def sort(self, data: list[int]) -> list[int]:
        """Sort the data and return a new sorted list."""
        ...
    
    @property
    def name(self) -> str:
        """Return the algorithm name."""
        ...


@dataclass
class BubbleSort:
    """Bubble sort implementation.
    
    Time complexity: O(n²) worst and average case
    Space complexity: O(1) - in-place
    Stable: Yes
    """
    
    @property
    def name(self) -> str:
        return "Bubble Sort"
    
    def sort(self, data: list[int]) -> list[int]:
        """Sort using bubble sort algorithm.
        
        Repeatedly steps through the list, compares adjacent elements
        and swaps them if they are in the wrong order.
        """
        result = data.copy()  # Don't modify original
        n = len(result)
        
        for i in range(n):
            # Flag to detect if any swap happened
            swapped = False
            
            for j in range(0, n - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
                    swapped = True
            
            # If no swapping occurred, array is sorted
            if not swapped:
                break
        
        return result


@dataclass
class QuickSort:
    """Quick sort implementation.
    
    Time complexity: O(n log n) average, O(n²) worst case
    Space complexity: O(log n) for recursion stack
    Stable: No
    """
    
    @property
    def name(self) -> str:
        return "Quick Sort"
    
    def sort(self, data: list[int]) -> list[int]:
        """Sort using quick sort algorithm.
        
        Uses the divide-and-conquer approach with a pivot element.
        """
        if len(data) <= 1:
            return data.copy()
        
        return self._quicksort(data.copy())
    
    def _quicksort(self, arr: list[int]) -> list[int]:
        """Recursive quicksort helper."""
        if len(arr) <= 1:
            return arr
        
        # Choose middle element as pivot (more reliable than first/last)
        pivot = arr[len(arr) // 2]
        
        # Partition into three lists
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return self._quicksort(left) + middle + self._quicksort(right)


@dataclass 
class MergeSort:
    """Merge sort implementation.
    
    Time complexity: O(n log n) in all cases
    Space complexity: O(n) for auxiliary arrays
    Stable: Yes
    """
    
    @property
    def name(self) -> str:
        return "Merge Sort"
    
    def sort(self, data: list[int]) -> list[int]:
        """Sort using merge sort algorithm.
        
        Divides the list in half, recursively sorts each half,
        then merges the sorted halves.
        """
        if len(data) <= 1:
            return data.copy()
        
        return self._mergesort(data.copy())
    
    def _mergesort(self, arr: list[int]) -> list[int]:
        """Recursive mergesort helper."""
        if len(arr) <= 1:
            return arr
        
        # Split in half
        mid = len(arr) // 2
        left = self._mergesort(arr[:mid])
        right = self._mergesort(arr[mid:])
        
        # Merge sorted halves
        return self._merge(left, right)
    
    def _merge(self, left: list[int], right: list[int]) -> list[int]:
        """Merge two sorted lists."""
        result: list[int] = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        # Append remaining elements
        result.extend(left[i:])
        result.extend(right[j:])
        
        return result


class Sorter:
    """Context class that uses a sorting strategy."""
    
    def __init__(self, strategy: SortStrategy) -> None:
        """Initialise with a sorting strategy."""
        self._strategy = strategy
    
    @property
    def strategy(self) -> SortStrategy:
        """Get current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: SortStrategy) -> None:
        """Set new strategy."""
        self._strategy = strategy
    
    def sort(self, data: list[int]) -> list[int]:
        """Sort using current strategy."""
        return self._strategy.sort(data)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_bubble_sort() -> None:
    """Test bubble sort."""
    sorter = Sorter(BubbleSort())
    result = sorter.sort([3, 1, 4, 1, 5, 9, 2, 6])
    assert result == [1, 1, 2, 3, 4, 5, 6, 9]


def test_quick_sort() -> None:
    """Test quick sort."""
    sorter = Sorter(QuickSort())
    result = sorter.sort([3, 1, 4, 1, 5, 9, 2, 6])
    assert result == [1, 1, 2, 3, 4, 5, 6, 9]


def test_merge_sort() -> None:
    """Test merge sort."""
    sorter = Sorter(MergeSort())
    result = sorter.sort([3, 1, 4, 1, 5, 9, 2, 6])
    assert result == [1, 1, 2, 3, 4, 5, 6, 9]


def test_strategy_swap() -> None:
    """Test swapping strategies at runtime."""
    sorter = Sorter(BubbleSort())
    assert sorter.strategy.name == "Bubble Sort"
    
    sorter.strategy = QuickSort()
    assert sorter.strategy.name == "Quick Sort"
    
    result = sorter.sort([5, 3, 8, 1])
    assert result == [1, 3, 5, 8]


def test_empty_list() -> None:
    """Test sorting empty list."""
    for strategy in [BubbleSort(), QuickSort(), MergeSort()]:
        sorter = Sorter(strategy)
        assert sorter.sort([]) == []


def test_single_element() -> None:
    """Test sorting single element list."""
    for strategy in [BubbleSort(), QuickSort(), MergeSort()]:
        sorter = Sorter(strategy)
        assert sorter.sort([42]) == [42]


def test_already_sorted() -> None:
    """Test sorting already sorted list."""
    for strategy in [BubbleSort(), QuickSort(), MergeSort()]:
        sorter = Sorter(strategy)
        assert sorter.sort([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]


def test_reverse_sorted() -> None:
    """Test sorting reverse sorted list."""
    for strategy in [BubbleSort(), QuickSort(), MergeSort()]:
        sorter = Sorter(strategy)
        assert sorter.sort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]


def test_duplicates() -> None:
    """Test sorting with many duplicates."""
    for strategy in [BubbleSort(), QuickSort(), MergeSort()]:
        sorter = Sorter(strategy)
        assert sorter.sort([3, 3, 3, 1, 1, 2, 2]) == [1, 1, 2, 2, 3, 3, 3]


def test_original_unchanged() -> None:
    """Test that original list is not modified."""
    original = [3, 1, 4, 1, 5]
    for strategy in [BubbleSort(), QuickSort(), MergeSort()]:
        sorter = Sorter(strategy)
        sorter.sort(original)
        assert original == [3, 1, 4, 1, 5]


if __name__ == "__main__":
    test_bubble_sort()
    test_quick_sort()
    test_merge_sort()
    test_strategy_swap()
    test_empty_list()
    test_single_element()
    test_already_sorted()
    test_reverse_sorted()
    test_duplicates()
    test_original_unchanged()
    print("All tests passed! ✓")
