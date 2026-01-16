#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 2 Practice: Easy Exercise 3 — Generics SOLUTION
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class Box(Generic[T]):
    """A generic container that holds a single value of type T.
    
    Type Parameters:
        T: The type of value stored in the box.
    
    Attributes:
        value: The stored value.
    
    Example:
        >>> int_box = Box(42)
        >>> str_box = Box("hello")
    """
    value: T
    
    def map(self, func: "Callable[[T], U]") -> "Box[U]":
        """Transform the contained value using a function.
        
        This is the functor map operation.
        
        Type Parameters:
            U: The output type of the transformation.
        
        Args:
            func: A function from T to U.
        
        Returns:
            A new Box containing the transformed value.
        
        Example:
            >>> Box(5).map(lambda x: x * 2)
            Box(value=10)
        """
        return Box(func(self.value))
    
    def flat_map(self, func: "Callable[[T], Box[U]]") -> "Box[U]":
        """Transform and flatten (monadic bind).
        
        This is the monad bind operation.
        
        Args:
            func: A function from T to Box[U].
        
        Returns:
            The Box returned by the function.
        
        Example:
            >>> Box(5).flat_map(lambda x: Box(x + 1))
            Box(value=6)
        """
        return func(self.value)
    
    def get_or_default(self, default: T) -> T:
        """Get the value or a default if None.
        
        Args:
            default: Value to return if self.value is None.
        
        Returns:
            The stored value or the default.
        """
        if self.value is None:
            return default
        return self.value


from typing import Callable

U = TypeVar('U')


def swap(pair: tuple[T, U]) -> tuple[U, T]:
    """Swap the elements of a pair.
    
    A generic function demonstrating multiple type variables.
    
    Args:
        pair: A tuple of two elements.
    
    Returns:
        A tuple with elements in reverse order.
    
    Example:
        >>> swap((1, "hello"))
        ('hello', 1)
    """
    return (pair[1], pair[0])


def identity(value: T) -> T:
    """Return the value unchanged.
    
    The simplest generic function.
    
    Args:
        value: Any value.
    
    Returns:
        The same value.
    """
    return value


def first_or_default(items: list[T], default: T) -> T:
    """Get the first element or a default if the list is empty.
    
    Args:
        items: A list of items.
        default: Default value if list is empty.
    
    Returns:
        First element or default.
    
    Example:
        >>> first_or_default([1, 2, 3], 0)
        1
        >>> first_or_default([], 0)
        0
    """
    if items:
        return items[0]
    return default


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_box_creation() -> None:
    """Test creating boxes with different types."""
    int_box: Box[int] = Box(42)
    str_box: Box[str] = Box("hello")
    list_box: Box[list[int]] = Box([1, 2, 3])
    
    assert int_box.value == 42
    assert str_box.value == "hello"
    assert list_box.value == [1, 2, 3]


def test_box_map() -> None:
    """Test the map operation."""
    box = Box(5)
    result = box.map(lambda x: x * 2)
    assert result.value == 10
    
    # Type changes through map
    str_result = box.map(str)
    assert str_result.value == "5"


def test_box_flat_map() -> None:
    """Test the flat_map operation."""
    box = Box(5)
    result = box.flat_map(lambda x: Box(x + 1))
    assert result.value == 6


def test_box_chaining() -> None:
    """Test chaining map operations."""
    result = (
        Box(5)
        .map(lambda x: x + 1)   # 6
        .map(lambda x: x * 2)   # 12
        .map(str)               # "12"
    )
    assert result.value == "12"


def test_swap() -> None:
    """Test the swap function."""
    result = swap((1, "hello"))
    assert result == ("hello", 1)
    
    # Works with any types
    result2 = swap(([1, 2], {"a": 1}))
    assert result2 == ({"a": 1}, [1, 2])


def test_identity() -> None:
    """Test the identity function."""
    assert identity(42) == 42
    assert identity("hello") == "hello"
    assert identity([1, 2, 3]) == [1, 2, 3]


def test_first_or_default() -> None:
    """Test first_or_default function."""
    assert first_or_default([1, 2, 3], 0) == 1
    assert first_or_default([], 0) == 0
    assert first_or_default(["a", "b"], "default") == "a"
    assert first_or_default([], "default") == "default"


def test_get_or_default() -> None:
    """Test Box.get_or_default."""
    box = Box(5)
    assert box.get_or_default(0) == 5
    
    none_box: Box[int | None] = Box(None)
    assert none_box.get_or_default(0) == 0


if __name__ == "__main__":
    test_box_creation()
    test_box_map()
    test_box_flat_map()
    test_box_chaining()
    test_swap()
    test_identity()
    test_first_or_default()
    test_get_or_default()
    print("All tests passed! ✓")
