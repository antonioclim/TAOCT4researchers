#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
Week 2 Practice: Easy Exercise 2 — Immutable Dataclasses
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐ (Easy)
Estimated Time: 15 minutes

TASK
────
Create an immutable Point2D dataclass with x and y coordinates.
Implement a method to calculate distance to another point.

LEARNING OBJECTIVES
───────────────────
- Use frozen dataclasses for immutability
- Add methods to dataclasses
- Understand immutable design patterns

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
import math


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the Point2D class below
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Point2D:
    """
    An immutable 2D point.
    
    Attributes:
        x: The x coordinate
        y: The y coordinate
        
    Example:
        >>> p1 = Point2D(0, 0)
        >>> p2 = Point2D(3, 4)
        >>> p1.distance_to(p2)
        5.0
    """
    x: float
    y: float
    
    def distance_to(self, other: 'Point2D') -> float:
        """
        Calculate Euclidean distance to another point.
        
        Args:
            other: The target point
            
        Returns:
            The distance as a float
        """
        # TODO: Implement using Pythagorean theorem
        pass
    
    def translate(self, dx: float, dy: float) -> 'Point2D':
        """
        Return a new point translated by (dx, dy).
        
        Note: Since Point2D is frozen, this returns a NEW point.
        
        Args:
            dx: Translation in x
            dy: Translation in y
            
        Returns:
            New Point2D at translated position
        """
        # TODO: Return new translated point
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_point_creation() -> None:
    """Test point creation."""
    p = Point2D(3.0, 4.0)
    assert p.x == 3.0
    assert p.y == 4.0


def test_point_immutability() -> None:
    """Test that point is immutable."""
    p = Point2D(1.0, 2.0)
    try:
        p.x = 5.0  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass  # Expected


def test_distance_calculation() -> None:
    """Test distance calculation."""
    origin = Point2D(0, 0)
    p = Point2D(3, 4)
    assert origin.distance_to(p) == 5.0


def test_translate() -> None:
    """Test translation returns new point."""
    p1 = Point2D(1, 1)
    p2 = p1.translate(2, 3)
    assert p2.x == 3
    assert p2.y == 4
    assert p1.x == 1  # Original unchanged


if __name__ == "__main__":
    test_point_creation()
    test_point_immutability()
    test_distance_calculation()
    test_translate()
    print("All tests passed! ✓")
