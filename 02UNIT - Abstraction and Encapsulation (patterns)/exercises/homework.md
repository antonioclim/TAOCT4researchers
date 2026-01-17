# 02UNIT Homework: Abstraction and Encapsulation

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | Friday 23:59 GMT |
| **Total Points** | 100 |
| **Estimated Time** | 4-6 hours |
| **Difficulty** | ⭐⭐⭐ (3/5) |

## 🔗 Prerequisites

- [x] Completed Lab 2.1: Simulation Framework
- [x] Completed Lab 2.2: Design Patterns
- [x] Read lecture notes on OOP principles

## 🎯 Objectives Assessed

1. Design type systems that prevent errors at compile time
2. Implement functional patterns (Result, Option)
3. Use Protocols for structural typing
4. Create fluent and type-safe APIs

---

## Part 1: Units Library — Type-Safe Dimensional Analysis (40 points)

### Context

In 1999, the Mars Climate Orbiter crashed because a function returned impulse in pound-seconds, but the caller expected newton-seconds. A more expressive type system would have prevented this error at compile time.

### Requirements

Implement a physical units library that makes it impossible at the type level to add incompatible quantities.

```python
# What should work:
distance = Meters(100)
time = Seconds(10)
speed = distance / time  # → MetersPerSecond(10)

# What should NOT work (type error!):
invalid = distance + time  # TypeError at static checking

# Explicit conversions:
feet = Feet(328.084)
total = distance + feet.to_meters()  # OK: both are lengths
```

### Suggested Structure

```python
from __future__ import annotations
from typing import TypeVar, Generic
from dataclasses import dataclass

# Phantom types for dimensions
class Length: pass
class Time: pass
class Mass: pass
class Velocity: pass  # Length / Time

T = TypeVar('T')  # Dimension

@dataclass(frozen=True)
class Quantity(Generic[T]):
    """A quantity with dimension T and value."""
    value: float
    
    def __add__(self, other: 'Quantity[T]') -> 'Quantity[T]':
        # Only quantities of the same dimension!
        return Quantity(self.value + other.value)
    
    def __truediv__(self, other: 'Quantity[Time]') -> 'Quantity[Velocity]':
        # Specialisation for Length / Time
        ...

# Type aliases for clarity
Meters = Quantity[Length]
Seconds = Quantity[Time]
```

### Detailed Requirements

| Req | Points | Description |
|-----|--------|-------------|
| 1.1 | 15 | **Base units**: Length (Meters, Feet, Kilometres, Miles), Time (Seconds, Minutes, Hours), Mass (Kilograms, Pounds). Include conversions between units of the same dimension. |
| 1.2 | 15 | **Dimensional operations**: Length / Time → Velocity, Length / Time / Time → Acceleration, Mass × Acceleration → Force. Type checker must detect errors. |
| 1.3 | 10 | **Tests and documentation**: pytest for correctness, mypy passes without errors, docstrings with examples. |

### Test Cases

```python
# Conversions
assert Feet(3.28084).to_meters().value == pytest.approx(1.0, rel=1e-3)

# Dimensional operations
distance = Meters(100)
time = Seconds(10)
speed = distance / time
assert isinstance(speed, Quantity)  # Should be Velocity type

# Type errors (mypy should catch these)
# invalid = Meters(1) + Seconds(1)  # Should fail mypy
```

### Hints

<details>
<summary>💡 Hint 1: Phantom Types</summary>
Phantom types are type parameters that do not appear in the runtime representation.
They exist solely for type checking.
</details>

<details>
<summary>💡 Hint 2: Overloaded Methods</summary>
Use `@overload` from typing to define multiple type signatures for operators.
</details>

---

## Part 2: Result Monad — Railway Oriented Programming (25 points)

### Context

In functional programming, the Result/Either monad enables composing operations that may fail, without explicit try/except at each step.

### Requirements

Extend the `Result[T]` class from the laboratory with the following methods:

```python
@dataclass
class Result(Generic[T]):
    _value: T | None
    _error: str | None
    
    # Already implemented
    @classmethod
    def ok(cls, value: T) -> 'Result[T]': ...
    
    @classmethod
    def err(cls, message: str) -> 'Result[T]': ...
    
    def unwrap(self) -> T: ...
    
    def map(self, f: Callable[[T], U]) -> 'Result[U]': ...
    
    # TO IMPLEMENT:
    
    def and_then(self, f: Callable[[T], 'Result[U]']) -> 'Result[U]':
        """
        If self is Ok, apply f(value) which returns a new Result.
        If self is Err, propagate the error.
        
        Alias: flatMap, bind, >>=
        """
        ...
    
    def or_else(self, f: Callable[[str], 'Result[T]']) -> 'Result[T]':
        """
        If self is Err, apply f(error) for recovery.
        If self is Ok, return self.
        """
        ...
    
    def unwrap_or(self, default: T) -> T:
        """Return value if Ok, else default."""
        ...
    
    def unwrap_or_else(self, f: Callable[[str], T]) -> T:
        """Return value if Ok, else f(error)."""
        ...
    
    def map_err(self, f: Callable[[str], str]) -> 'Result[T]':
        """Transform the error message whilst preserving the value."""
        ...
    
    @staticmethod
    def collect(results: list['Result[T]']) -> 'Result[list[T]]':
        """
        Transform a list of Results into a Result of list.
        If all are Ok, return Ok with the list of values.
        If any is Err, return the first Err.
        """
        ...
```

### Example Usage: Data Pipeline

```python
def parse_int(s: str) -> Result[int]:
    try:
        return Result.ok(int(s))
    except ValueError:
        return Result.err(f"Cannot parse '{s}' as int")

def validate_positive(n: int) -> Result[int]:
    if n > 0:
        return Result.ok(n)
    return Result.err(f"Expected positive, got {n}")

def divide_100_by(n: int) -> Result[float]:
    if n == 0:
        return Result.err("Division by zero")
    return Result.ok(100 / n)

# Railway oriented programming
result = (
    parse_int("42")
    .and_then(validate_positive)
    .and_then(divide_100_by)
    .map(lambda x: f"Result: {x:.2f}")
    .unwrap_or("Computation failed")
)
```

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| and_then, or_else | 8 |
| unwrap_or, unwrap_or_else | 5 |
| map_err, collect | 7 |
| Comprehensive tests | 5 |

---

## Part 3: Type-Safe Builder Pattern (20 points)

### Context

The Builder pattern enables step-by-step construction of complex objects. With the type state pattern, the compiler verifies that all mandatory fields are set before build.

### Requirements

Implement a builder for an `HttpRequest` object where the compiler (mypy) verifies at type level that:
- `url` and `method` are mandatory
- `build()` can only be called after setting mandatory fields

```python
# Should work:
request = (
    HttpRequestBuilder()
    .url("https://api.example.com/users")
    .method("GET")
    .header("Authorization", "Bearer token")
    .timeout(30)
    .build()  # OK - url and method set
)

# Should give TYPE error (not runtime!):
invalid = (
    HttpRequestBuilder()
    .header("X-Custom", "value")
    .build()  # TYPE ERROR: Missing url and method
)
```

### Hint: Phantom Type State

```python
class NotSet: pass
class Set: pass

@dataclass
class HttpRequestBuilder(Generic[UrlState, MethodState]):
    _url: str | None = None
    _method: str | None = None
    _headers: dict[str, str] = field(default_factory=dict)
    
    def url(self, url: str) -> 'HttpRequestBuilder[Set, MethodState]':
        ...
    
    # build() is defined ONLY for state [Set, Set]
    @overload
    def build(self: 'HttpRequestBuilder[Set, Set]') -> HttpRequest: ...
```

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| Functional builder | 8 |
| Correct type state | 8 |
| mypy verifies at compile time | 4 |

---

## Part 4: Mini Simulation Framework (15 points)

### Requirements

Using the `Simulable` Protocol from the laboratory, implement a new model relevant to your research domain.

### Suggested Options

1. **Biology:** Lotka-Volterra model (predator-prey)
2. **Economics:** Supply-demand market model
3. **Physics:** Simple/double pendulum
4. **Sociology:** Innovation diffusion model
5. **Climatology:** Simple energy balance model
6. **Other:** Propose a model from your field

### Required Structure

```python
@dataclass
class YourModelState:
    """State of your model."""
    ...

class YourSimulation:
    """Implements Simulable[YourModelState]."""
    
    def state(self) -> YourModelState:
        ...
    
    def step(self, dt: float) -> None:
        ...
    
    def is_done(self) -> bool:
        ...
```

### Deliverables

1. Model implementation (compatible with SimulationRunner)
2. README explaining:
   - Mathematical description of the model
   - Differential equations (if applicable)
   - Meaning of parameters
   - Interpretation of results
3. Results visualisation (matplotlib)
4. At least 3 tests

### Evaluation Criteria

| Criterion | Points |
|-----------|--------|
| Mathematically correct model | 6 |
| Simulable implementation | 4 |
| Documentation | 3 |
| Visualisation | 2 |

---

## ✅ Submission Checklist

- [ ] All tests pass (`pytest`)
- [ ] Formatted with ruff (`ruff check .`)
- [ ] Type hints complete
- [ ] mypy passes (`mypy --strict`)
- [ ] Docstrings present
- [ ] README updated

## 📁 Repository Format

```
homework-02UNIT-[name]/
├── README.md
├── ex1_units/
│   ├── units.py
│   ├── tests/
│   └── py.typed          # marker for mypy
├── ex2_result_monad/
│   ├── result.py
│   └── tests/
├── ex3_builder/
│   ├── http_builder.py
│   └── tests/
└── ex4_simulation/
    ├── your_model.py
    ├── README.md
    └── visualisation.py
```

## 🔍 Pre-Submission Verification

```bash
# All must pass:
mypy . --strict
pytest
ruff check .
```

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*02UNIT — Abstraction and Encapsulation*

© 2025 Antonio Clim. All rights reserved.
