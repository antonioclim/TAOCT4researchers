#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Easy Exercise 1 — Protocol SOLUTION
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import Protocol
from dataclasses import dataclass


class Incrementable(Protocol):
    """Protocol for types that can be incremented."""
    
    def increment(self) -> "Incrementable":
        """Return a new instance with value incremented by 1."""
        ...
    
    @property
    def value(self) -> int:
        """Get the current value."""
        ...


@dataclass(frozen=True)
class Counter:
    """An immutable counter that can be incremented.
    
    Attributes:
        value: The current count.
    """
    value: int = 0
    
    def increment(self) -> "Counter":
        """Return a new Counter with value + 1.
        
        Returns:
            A new Counter instance with incremented value.
        
        Example:
            >>> c = Counter(5)
            >>> c.increment().value
            6
        """
        return Counter(self.value + 1)


@dataclass(frozen=True)
class WrappingCounter:
    """A counter that wraps around at a maximum value.
    
    When increment() would exceed max_value, it wraps to 0.
    
    Attributes:
        value: The current count.
        max_value: Maximum value before wrapping (default 10).
    """
    value: int = 0
    max_value: int = 10
    
    def increment(self) -> "WrappingCounter":
        """Return a new counter, wrapping to 0 if at max.
        
        Returns:
            A new WrappingCounter instance.
        
        Example:
            >>> c = WrappingCounter(9, max_value=10)
            >>> c.increment().value
            0
        """
        new_value = (self.value + 1) % (self.max_value + 1)
        return WrappingCounter(new_value, self.max_value)


def increment_twice(item: Incrementable) -> Incrementable:
    """Increment any Incrementable twice.
    
    This function demonstrates structural typing - it works with any type
    that has increment() and value, without requiring inheritance.
    
    Args:
        item: Any object conforming to the Incrementable protocol.
    
    Returns:
        The item incremented twice.
    """
    return item.increment().increment()


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_counter_increment() -> None:
    """Test basic counter increment."""
    c = Counter(0)
    assert c.increment().value == 1
    assert c.increment().increment().value == 2


def test_counter_immutability() -> None:
    """Test that Counter is immutable."""
    c1 = Counter(5)
    c2 = c1.increment()
    assert c1.value == 5  # Original unchanged
    assert c2.value == 6


def test_wrapping_counter() -> None:
    """Test wrapping behaviour."""
    c = WrappingCounter(9, max_value=10)
    assert c.increment().value == 10
    assert c.increment().increment().value == 0


def test_increment_twice() -> None:
    """Test the generic increment function."""
    c1 = Counter(0)
    result1 = increment_twice(c1)
    assert result1.value == 2
    
    c2 = WrappingCounter(9, max_value=10)
    result2 = increment_twice(c2)
    assert result2.value == 0  # 9 → 10 → 0


def test_protocol_compliance() -> None:
    """Test that both types satisfy the protocol."""
    def accepts_incrementable(x: Incrementable) -> int:
        return x.value
    
    assert accepts_incrementable(Counter(5)) == 5
    assert accepts_incrementable(WrappingCounter(3)) == 3


if __name__ == "__main__":
    test_counter_increment()
    test_counter_immutability()
    test_wrapping_counter()
    test_increment_twice()
    test_protocol_compliance()
    print("All tests passed! ✓")
