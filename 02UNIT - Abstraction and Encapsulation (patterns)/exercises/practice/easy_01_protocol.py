#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 2 Practice: Easy Exercise 1 — Protocol Implementation
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐ (Easy)
Estimated Time: 15 minutes

TASK
────
Implement a simple `Counter` class that satisfies the `Incrementable` protocol.

LEARNING OBJECTIVES
───────────────────
- Understand structural typing with Protocols
- Implement a class that matches a Protocol interface

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Protocol


class Incrementable(Protocol):
    """Protocol for objects that can be incremented."""
    
    def increment(self) -> None:
        """Increase the internal value by 1."""
        ...
    
    def value(self) -> int:
        """Return the current value."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the Counter class below
# ═══════════════════════════════════════════════════════════════════════════════

class Counter:
    """
    A simple counter that implements the Incrementable protocol.
    
    Example:
        >>> c = Counter(start=5)
        >>> c.value()
        5
        >>> c.increment()
        >>> c.value()
        6
    """
    
    def __init__(self, start: int = 0) -> None:
        # TODO: Initialise the counter
        pass
    
    def increment(self) -> None:
        # TODO: Increment the internal value
        pass
    
    def value(self) -> int:
        # TODO: Return the current value
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_counter_initial_value() -> None:
    """Test that counter starts at specified value."""
    c = Counter(start=10)
    assert c.value() == 10


def test_counter_increment() -> None:
    """Test that increment increases value by 1."""
    c = Counter(start=0)
    c.increment()
    assert c.value() == 1
    c.increment()
    assert c.value() == 2


def test_counter_satisfies_protocol() -> None:
    """Test that Counter satisfies Incrementable protocol."""
    def use_incrementable(obj: Incrementable) -> int:
        obj.increment()
        return obj.value()
    
    c = Counter(start=5)
    result = use_incrementable(c)
    assert result == 6


if __name__ == "__main__":
    test_counter_initial_value()
    test_counter_increment()
    test_counter_satisfies_protocol()
    print("All tests passed! ✓")
