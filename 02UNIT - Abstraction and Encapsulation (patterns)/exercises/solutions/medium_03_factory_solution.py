#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
02UNIT Practice: Medium Exercise 3 — Factory Pattern SOLUTION
═══════════════════════════════════════════════════════════════════════════════

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

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


@dataclass(frozen=True)
class Circle:
    """Circle shape with a radius."""
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
        return pi * self.radius * self.radius
    
    def perimeter(self) -> float:
        """Calculate perimeter (circumference): 2 × π × r."""
        return 2 * pi * self.radius


@dataclass(frozen=True)
class Rectangle:
    """Rectangle shape with width and height."""
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
        return self.width * self.height
    
    def perimeter(self) -> float:
        """Calculate perimeter: 2 × (width + height)."""
        return 2 * (self.width + self.height)


@dataclass(frozen=True)
class Triangle:
    """Triangle shape defined by three sides using Heron's formula."""
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
        """Calculate area using Heron's formula."""
        s = (self.a + self.b + self.c) / 2
        return sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))
    
    def perimeter(self) -> float:
        """Calculate perimeter: a + b + c."""
        return self.a + self.b + self.c


class ShapeFactory:
    """Factory for creating shapes."""
    
    def create(self, shape_type: ShapeType, **kwargs: float) -> Shape:
        """Create a shape of the specified type."""
        match shape_type:
            case ShapeType.CIRCLE:
                return Circle(radius=kwargs["radius"])
            case ShapeType.RECTANGLE:
                return Rectangle(width=kwargs["width"], height=kwargs["height"])
            case ShapeType.TRIANGLE:
                return Triangle(a=kwargs["a"], b=kwargs["b"], c=kwargs["c"])
            case _:
                raise ValueError(f"Unknown shape type: {shape_type}")


def print_shape_info(shape: Shape) -> None:
    """Print information about any shape."""
    print(f"Shape: {shape.name}")
    print(f"  Area: {shape.area():.4f}")
    print(f"  Perimeter: {shape.perimeter():.4f}")


def total_area(shapes: list[Shape]) -> float:
    """Calculate total area of multiple shapes."""
    return sum(shape.area() for shape in shapes)


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
    try:
        Circle(radius=-1.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    try:
        Triangle(a=1.0, b=2.0, c=10.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_total_area() -> None:
    """Test total area calculation."""
    factory = ShapeFactory()
    shapes: list[Shape] = [
        factory.create(ShapeType.CIRCLE, radius=1.0),
        factory.create(ShapeType.RECTANGLE, width=2.0, height=2.0),
    ]
    assert abs(total_area(shapes) - (pi + 4.0)) < 0.001


if __name__ == "__main__":
    test_circle()
    test_rectangle()
    test_triangle()
    test_factory()
    test_invalid_shapes()
    test_total_area()
    print("All tests passed! ✓")
