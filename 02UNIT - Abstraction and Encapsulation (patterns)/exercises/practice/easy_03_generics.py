#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Easy Exercise 3 — Generic Container
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐ (Easy)
Estimated Time: 15 minutes

TASK
────
Implement a generic Box class that can hold any type of value.

LEARNING OBJECTIVES
───────────────────
- Understand TypeVar and Generic
- Create type-safe generic containers
- Use type parameters consistently

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import TypeVar, Generic

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the Box class below
# ═══════════════════════════════════════════════════════════════════════════════

class Box(Generic[T]):
    """
    A generic container that holds a single value.
    
    Type parameter T determines what type the box can hold.
    
    Example:
        >>> int_box: Box[int] = Box(42)
        >>> int_box.get()
        42
        >>> int_box.set(100)
        >>> int_box.get()
        100
    """
    
    def __init__(self, value: T) -> None:
        """Initialise box with a value."""
        # TODO: Store the value
        pass
    
    def get(self) -> T:
        """Return the stored value."""
        # TODO: Return the value
        pass
    
    def set(self, value: T) -> None:
        """Update the stored value."""
        # TODO: Update the value
        pass
    
    def map(self, func: 'Callable[[T], T]') -> 'Box[T]':
        """
        Apply a function to the value and return a new Box.
        
        Args:
            func: Function to apply to the contained value
            
        Returns:
            New Box containing the transformed value
        """
        # TODO: Apply func and return new Box
        pass


from typing import Callable


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_box_int() -> None:
    """Test box with integers."""
    box: Box[int] = Box(42)
    assert box.get() == 42


def test_box_string() -> None:
    """Test box with strings."""
    box: Box[str] = Box("hello")
    assert box.get() == "hello"


def test_box_set() -> None:
    """Test setting new value."""
    box: Box[int] = Box(1)
    box.set(2)
    assert box.get() == 2


def test_box_map() -> None:
    """Test map transformation."""
    box: Box[int] = Box(5)
    new_box = box.map(lambda x: x * 2)
    assert new_box.get() == 10
    assert box.get() == 5  # Original unchanged


if __name__ == "__main__":
    test_box_int()
    test_box_string()
    test_box_set()
    test_box_map()
    print("All tests passed! ✓")
