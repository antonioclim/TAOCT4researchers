# Week 2 Cheatsheet: Abstraction and Encapsulation

> **A4 Quick Reference** — Print-friendly summary of key concepts

---

## SOLID Principles

| Principle | Summary | Python Example |
|-----------|---------|----------------|
| **S**ingle Responsibility | One class, one reason to change | `DataLoader` vs `DataValidator` |
| **O**pen/Closed | Open for extension, closed for modification | Add new `Strategy`, don't modify `Context` |
| **L**iskov Substitution | Subtypes must be substitutable | `Square` shouldn't break `Rectangle` expectations |
| **I**nterface Segregation | Many specific interfaces > one general | `Readable`, `Writable` vs `File` |
| **D**ependency Inversion | Depend on abstractions, not concretions | Inject `Protocol`, not concrete class |

---

## Python Protocols vs ABCs

```python
# Protocol (structural typing - "duck typing with hints")
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None: ...

# ABC (nominal typing - requires inheritance)
from abc import ABC, abstractmethod

class Drawable(ABC):
    @abstractmethod
    def draw(self) -> None: ...
```

| Feature | Protocol | ABC |
|---------|----------|-----|
| Inheritance required | No | Yes |
| Runtime checkable | With decorator | Yes |
| Multiple inheritance | Implicit | Explicit |
| Use when | Duck typing needed | Shared implementation |

---

## Design Patterns Quick Reference

### Strategy Pattern
```python
class Strategy(Protocol):
    def execute(self, data: T) -> R: ...

class Context:
    def __init__(self, strategy: Strategy) -> None:
        self._strategy = strategy
    
    def do_work(self, data: T) -> R:
        return self._strategy.execute(data)
```

### Observer Pattern
```python
class Observer(Protocol):
    def update(self, value: T) -> None: ...

class Subject:
    _observers: list[Observer]
    
    def notify(self, value: T) -> None:
        for obs in self._observers:
            obs.update(value)
```

### Factory Pattern
```python
class Factory:
    def create(self, type_: str) -> Product:
        match type_:
            case "a": return ProductA()
            case "b": return ProductB()
```

---

## Generics Syntax

```python
from typing import TypeVar, Generic

T = TypeVar('T')           # Any type
T = TypeVar('T', int, str) # Constrained
T = TypeVar('T', bound=Base) # Upper bound

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self._value = value
    
    def get(self) -> T:
        return self._value
```

---

## Dataclass Quick Reference

```python
from dataclasses import dataclass, field

@dataclass
class Point:
    x: float
    y: float
    label: str = "origin"  # Default value

@dataclass(frozen=True)  # Immutable
class ImmutablePoint:
    x: float
    y: float

@dataclass
class Container:
    items: list[str] = field(default_factory=list)
```

| Parameter | Effect |
|-----------|--------|
| `frozen=True` | Immutable, hashable |
| `eq=True` | Generate `__eq__` |
| `order=True` | Generate comparison methods |
| `slots=True` | Use `__slots__` (Python 3.10+) |

---

## Common Code Patterns

### Composition over Inheritance
```python
# ❌ Inheritance
class SQLLogger(Logger, SQLConnection): ...

# ✅ Composition
class SQLLogger:
    def __init__(self, logger: Logger, conn: SQLConnection):
        self._logger = logger
        self._conn = conn
```

### Dependency Injection
```python
# ❌ Hard dependency
class Service:
    def __init__(self):
        self._db = PostgresDB()  # Concrete!

# ✅ Injected dependency
class Service:
    def __init__(self, db: Database):  # Protocol!
        self._db = db
```

---

## Common Mistakes

| Mistake | Problem | Solution |
|---------|---------|----------|
| God class | Too many responsibilities | Split into focused classes |
| Leaky abstraction | Implementation details exposed | Hide behind interface |
| Inheritance for code reuse | Tight coupling | Use composition |
| Mutable default args | Shared state bugs | Use `field(default_factory=...)` |
| Missing type hints | No static checking | Add hints, run mypy |

---

## Type Hint Cheatsheet

```python
# Basic types
x: int = 5
s: str = "hello"

# Collections
items: list[int] = [1, 2, 3]
mapping: dict[str, int] = {"a": 1}
coords: tuple[float, float] = (1.0, 2.0)

# Optional
maybe: int | None = None

# Callable
fn: Callable[[int, str], bool]

# Union
value: int | str

# Literal
mode: Literal["read", "write"]
```

---

## Connections to Other Weeks

| Week | Connection |
|------|------------|
| Week 1 | State → State pattern; AST → Visitor pattern |
| Week 3 | Strategy pattern for algorithm selection in benchmarks |
| Week 4 | Factory for data structure creation |
| Week 5 | Observer for simulation visualisation |
| Week 6 | Decorator for plot styling |
| Week 7 | Dependency injection for testable code |

---

## Quick Commands

```bash
# Type checking
mypy --strict .

# Linting
ruff check .

# Formatting
ruff format .

# Testing
pytest -v

# Coverage
pytest --cov=. --cov-report=html
```

---

© 2025 Antonio Clim. All rights reserved.
