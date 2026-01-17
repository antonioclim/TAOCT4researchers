#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Medium Exercise 3 — Factory Pattern
═══════════════════════════════════════════════════════════════════════════════

Difficulty: ⭐⭐ (Medium)
Estimated Time: 35 minutes

TASK
────
Implement a Shape factory system that creates different geometric shapes
with a unified interface. The factory should support:
- Circle
- Rectangle
- Triangle

LEARNING OBJECTIVES
───────────────────
- Implement the Factory design pattern
- Use ABC for shape interface
- Demonstrate polymorphism through a common interface

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from math import pi, sqrt
from typing import Protocol


class ShapeType(Enum):
    """Enumeration of available shape types."""
    CIRCLE = auto()
    RECTANGLE = auto()
    TRIANGLE = auto()


class Shape(Protocol):
    """Protocol defining the shape interface."""
    
    def area(self) -> float:
        """Calculate and return the shape's area."""
        ...
    
    def perimeter(self) -> float:
        """Calculate and return the shape's perimeter."""
        ...
    
    @property
    def name(self) -> str:
        """Return the shape's name."""
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the shape classes below
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Circle:
    """Circle shape with a radius.
    
    Attributes:
        radius: The circle's radius (must be positive).
    """
    radius: float
    
    def __post_init__(self) -> None:
        """Validate that radius is positive."""
        if self.radius <= 0:
            raise ValueError("Radius must be positive")
    
    @property
    def name(self) -> str:
        return "Circle"
    
    def area(self) -> float:
        """Calculate area: π × r²."""
        # TODO: Implement
        pass
    
    def perimeter(self) -> float:
        """Calculate perimeter (circumference): 2 × π × r."""
        # TODO: Implement
        pass


@dataclass(frozen=True)
class Rectangle:
    """Rectangle shape with width and height.
    
    Attributes:
        width: The rectangle's width (must be positive).
        height: The rectangle's height (must be positive).
    """
    width: float
    height: float
    
    def __post_init__(self) -> None:
        """Validate that dimensions are positive."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Width and height must be positive")
    
    @property
    def name(self) -> str:
        return "Rectangle"
    
    def area(self) -> float:
        """Calculate area: width × height."""
        # TODO: Implement
        pass
    
    def perimeter(self) -> float:
        """Calculate perimeter: 2 × (width + height)."""
        # TODO: Implement
        pass


@dataclass(frozen=True)
class Triangle:
    """Triangle shape defined by three sides.
    
    Uses Heron's formula for area calculation.
    
    Attributes:
        a: First side length (must be positive).
        b: Second side length (must be positive).
        c: Third side length (must be positive).
    """
    a: float
    b: float
    c: float
    
    def __post_init__(self) -> None:
        """Validate triangle inequality."""
        if self.a <= 0 or self.b <= 0 or self.c <= 0:
            raise ValueError("All sides must be positive")
        if not (self.a + self.b > self.c and 
                self.b + self.c > self.a and 
                self.a + self.c > self.b):
            raise ValueError("Triangle inequality violated")
    
    @property
    def name(self) -> str:
        return "Triangle"
    
    def area(self) -> float:
        """Calculate area using Heron's formula.
        
        s = (a + b + c) / 2
        area = sqrt(s × (s-a) × (s-b) × (s-c))
        """
        # TODO: Implement using Heron's formula
        pass
    
    def perimeter(self) -> float:
        """Calculate perimeter: a + b + c."""
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# YOUR TASK: Implement the ShapeFactory class
# ═══════════════════════════════════════════════════════════════════════════════

class ShapeFactory:
    """Factory for creating shapes.
    
    This factory creates shapes based on the ShapeType enum.
    Each shape type requires different parameters.
    
    Example:
        factory = ShapeFactory()
        circle = factory.create(ShapeType.CIRCLE, radius=5.0)
        rect = factory.create(ShapeType.RECTANGLE, width=3.0, height=4.0)
    """
    
    def create(self, shape_type: ShapeType, **kwargs: float) -> Shape:
        """Create a shape of the specified type.
        
        Args:
            shape_type: The type of shape to create.
            **kwargs: Shape-specific parameters:
                - CIRCLE: radius
                - RECTANGLE: width, height
                - TRIANGLE: a, b, c
        
        Returns:
            A shape instance.
        
        Raises:
            ValueError: If shape_type is unknown or parameters are invalid.
        """
        # TODO: Implement factory method
        # Hint: Use match-case or if-elif to handle different shape types
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS: Implement a shape calculator that works with any shape
# ═══════════════════════════════════════════════════════════════════════════════

def print_shape_info(shape: Shape) -> None:
    """Print information about any shape.
    
    Demonstrates polymorphism - works with any shape implementation.
    
    Args:
        shape: Any object conforming to the Shape protocol.
    """
    print(f"Shape: {shape.name}")
    print(f"  Area: {shape.area():.4f}")
    print(f"  Perimeter: {shape.perimeter():.4f}")


def total_area(shapes: list[Shape]) -> float:
    """Calculate total area of multiple shapes.
    
    Args:
        shapes: List of shapes.
    
    Returns:
        Sum of all areas.
    """
    # TODO: Implement
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_circle() -> None:
    """Test Circle implementation."""
    circle = Circle(radius=5.0)
    assert circle.name == "Circle"
    assert abs(circle.area() - 78.5398) < 0.001
    assert abs(circle.perimeter() - 31.4159) < 0.001


def test_rectangle() -> None:
    """Test Rectangle implementation."""
    rect = Rectangle(width=3.0, height=4.0)
    assert rect.name == "Rectangle"
    assert rect.area() == 12.0
    assert rect.perimeter() == 14.0


def test_triangle() -> None:
    """Test Triangle implementation (3-4-5 right triangle)."""
    tri = Triangle(a=3.0, b=4.0, c=5.0)
    assert tri.name == "Triangle"
    assert abs(tri.area() - 6.0) < 0.001
    assert tri.perimeter() == 12.0


def test_factory() -> None:
    """Test ShapeFactory."""
    factory = ShapeFactory()
    
    circle = factory.create(ShapeType.CIRCLE, radius=1.0)
    assert circle.name == "Circle"
    
    rect = factory.create(ShapeType.RECTANGLE, width=2.0, height=3.0)
    assert rect.area() == 6.0
    
    tri = factory.create(ShapeType.TRIANGLE, a=3.0, b=4.0, c=5.0)
    assert tri.perimeter() == 12.0


def test_invalid_shapes() -> None:
    """Test validation of invalid shapes."""
    import pytest
    
    # Negative radius
    try:
        Circle(radius=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Triangle inequality violation
    try:
        Triangle(a=1.0, b=2.0, c=10.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_total_area() -> None:
    """Test total area calculation."""
    factory = ShapeFactory()
    shapes = [
        factory.create(ShapeType.CIRCLE, radius=1.0),
        factory.create(ShapeType.RECTANGLE, width=2.0, height=2.0),
    ]
    # Circle area ≈ 3.14159, Rectangle area = 4
    assert abs(total_area(shapes) - (pi + 4.0)) < 0.001


if __name__ == "__main__":
    test_circle()
    test_rectangle()
    test_triangle()
    test_factory()
    test_invalid_shapes()
    test_total_area()
    print("All tests passed! ✓")
