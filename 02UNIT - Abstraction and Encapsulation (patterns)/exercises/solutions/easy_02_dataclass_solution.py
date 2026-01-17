#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Easy Exercise 2 — Dataclass SOLUTION
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class Point2D:
    """An immutable 2D point.
    
    Using frozen=True makes this dataclass immutable (hashable).
    
    Attributes:
        x: The x-coordinate.
        y: The y-coordinate.
    """
    x: float
    y: float
    
    def distance_to(self, other: "Point2D") -> float:
        """Calculate Euclidean distance to another point.
        
        Uses the formula: sqrt((x2-x1)² + (y2-y1)²)
        
        Args:
            other: The other point.
        
        Returns:
            The Euclidean distance.
        
        Example:
            >>> p1 = Point2D(0, 0)
            >>> p2 = Point2D(3, 4)
            >>> p1.distance_to(p2)
            5.0
        """
        dx = other.x - self.x
        dy = other.y - self.y
        return sqrt(dx * dx + dy * dy)
    
    def midpoint(self, other: "Point2D") -> "Point2D":
        """Calculate the midpoint between this point and another.
        
        Args:
            other: The other point.
        
        Returns:
            A new Point2D at the midpoint.
        
        Example:
            >>> p1 = Point2D(0, 0)
            >>> p2 = Point2D(4, 6)
            >>> p1.midpoint(p2)
            Point2D(x=2.0, y=3.0)
        """
        mid_x = (self.x + other.x) / 2
        mid_y = (self.y + other.y) / 2
        return Point2D(mid_x, mid_y)
    
    def translate(self, dx: float, dy: float) -> "Point2D":
        """Return a new point translated by (dx, dy).
        
        Args:
            dx: Translation in x direction.
            dy: Translation in y direction.
        
        Returns:
            A new translated Point2D.
        
        Example:
            >>> p = Point2D(1, 2)
            >>> p.translate(3, -1)
            Point2D(x=4, y=1)
        """
        return Point2D(self.x + dx, self.y + dy)
    
    @classmethod
    def origin(cls) -> "Point2D":
        """Create a point at the origin (0, 0).
        
        Returns:
            A Point2D at (0, 0).
        """
        return cls(0.0, 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_point_creation() -> None:
    """Test basic point creation."""
    p = Point2D(3.0, 4.0)
    assert p.x == 3.0
    assert p.y == 4.0


def test_distance_to() -> None:
    """Test distance calculation (3-4-5 triangle)."""
    p1 = Point2D(0, 0)
    p2 = Point2D(3, 4)
    assert p1.distance_to(p2) == 5.0


def test_distance_symmetric() -> None:
    """Test that distance is symmetric."""
    p1 = Point2D(1, 2)
    p2 = Point2D(4, 6)
    assert abs(p1.distance_to(p2) - p2.distance_to(p1)) < 1e-10


def test_midpoint() -> None:
    """Test midpoint calculation."""
    p1 = Point2D(0, 0)
    p2 = Point2D(4, 6)
    mid = p1.midpoint(p2)
    assert mid.x == 2.0
    assert mid.y == 3.0


def test_translate() -> None:
    """Test translation."""
    p = Point2D(1, 2)
    moved = p.translate(3, -1)
    assert moved.x == 4
    assert moved.y == 1
    # Original unchanged (immutable)
    assert p.x == 1
    assert p.y == 2


def test_origin() -> None:
    """Test origin factory method."""
    o = Point2D.origin()
    assert o.x == 0.0
    assert o.y == 0.0


def test_immutability() -> None:
    """Test that points are immutable."""
    p = Point2D(1, 2)
    try:
        p.x = 5  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass


def test_hashable() -> None:
    """Test that frozen dataclass is hashable (usable in sets)."""
    p1 = Point2D(1, 2)
    p2 = Point2D(1, 2)
    p3 = Point2D(3, 4)
    
    points = {p1, p2, p3}
    assert len(points) == 2  # p1 and p2 are equal


if __name__ == "__main__":
    test_point_creation()
    test_distance_to()
    test_distance_symmetric()
    test_midpoint()
    test_translate()
    test_origin()
    test_immutability()
    test_hashable()
    print("All tests passed! ✓")
