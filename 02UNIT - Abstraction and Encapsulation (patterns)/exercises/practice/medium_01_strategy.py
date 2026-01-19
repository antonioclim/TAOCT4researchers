#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Medium Exercise 1 — Strategy Pattern
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐⭐ (Medium)
Estimated Time: 30 minutes

TASK
────
Implement a sorting strategy pattern with multiple algorithms:
- BubbleSort
- QuickSort  
- MergeSort

LEARNING OBJECTIVES
───────────────────
- Implement the Strategy design pattern
- Use Protocols for algorithm interfaces
- Swap algorithms at runtime

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Protocol, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


class SortStrategy(Protocol):
    """Protocol for sorting strategies."""
    
    def sort(self, data: list[int]) -> list[int]:
        """Sort the data and return a new sorted list."""
        ...
    
    @property
    def name(self) -> str:
        """Return the algorithm name."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the sorting strategies below
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BubbleSort:
    """Bubble sort implementation."""
    
    @property
    def name(self) -> str:
        return "Bubble Sort"
    
    def sort(self, data: list[int]) -> list[int]:
        """
        Sort using bubble sort algorithm.
        
        Time complexity: O(n²)
        """
        # TODO: Implement bubble sort
        # Remember to return a NEW list, not modify in place
        pass


@dataclass
class QuickSort:
    """Quick sort implementation."""
    
    @property
    def name(self) -> str:
        return "Quick Sort"
    
    def sort(self, data: list[int]) -> list[int]:
        """
        Sort using quick sort algorithm.
        
        Time complexity: O(n log n) average
        """
        # TODO: Implement quick sort
        pass


@dataclass 
class MergeSort:
    """Merge sort implementation."""
    
    @property
    def name(self) -> str:
        return "Merge Sort"
    
    def sort(self, data: list[int]) -> list[int]:
        """
        Sort using merge sort algorithm.
        
        Time complexity: O(n log n)
        """
        # TODO: Implement merge sort
        pass


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


if __name__ == "__main__":
    test_bubble_sort()
    test_quick_sort()
    test_merge_sort()
    test_strategy_swap()
    test_empty_list()
    print("All tests passed! ✓")
