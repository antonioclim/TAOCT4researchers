# Week 2 Glossary: Abstraction and Encapsulation

## Core Concepts

### Abstraction
The process of identifying essential characteristics of an object while ignoring irrelevant details. In programming, abstraction hides implementation complexity behind simple interfaces.

*Example*: A `Simulation` interface with `step()` and `state()` methods abstracts away whether the underlying implementation uses Euler or Runge-Kutta integration.

### Abstract Base Class (ABC)
A class that cannot be instantiated directly and serves as a template for derived classes. In Python, created using the `abc` module.

*Example*: `class Shape(ABC): @abstractmethod def area(self) -> float: ...`

### Composition
A design principle where objects contain other objects to achieve functionality, as opposed to inheriting from them. Expressed as "has-a" rather than "is-a".

*Example*: A `Car` *has-a* `Engine` rather than `Car` *is-an* `Engine`.

### Dependency Injection (DI)
A technique where dependencies are provided to an object rather than the object creating them itself. Promotes loose coupling and testability.

*Example*: `def __init__(self, logger: Logger)` instead of `self._logger = FileLogger()`.

### Design Pattern
A reusable solution to a commonly occurring problem in software design. Patterns provide templates, not specific code.

*See*: Strategy, Observer, Factory, Decorator, Command patterns.

### Duck Typing
A programming style where an object's suitability is determined by the presence of certain methods/properties rather than the object's type itself. "If it walks like a duck and quacks like a duck, it's a duck."

*Example*: Any object with a `read()` method can be used where a file-like object is expected.

### Encapsulation
The bundling of data and the methods that operate on that data within a single unit (class), restricting direct access to some components.

*Example*: Using `_private_attribute` convention and providing getter/setter methods.

---

## Python-Specific Terms

### Dataclass
A decorator (`@dataclass`) that automatically generates special methods like `__init__`, `__repr__` and `__eq__` based on class annotations.

*Example*: `@dataclass class Point: x: float; y: float`

### Frozen Dataclass
A dataclass with `frozen=True` that produces immutable instances. Attempting to modify attributes raises `FrozenInstanceError`.

*Benefit*: Hashable, can be used in sets and as dictionary keys.

### Generic
A class or function parameterised by types, allowing the same code to work with different types whilst maintaining type safety.

*Example*: `class Box(Generic[T]): def get(self) -> T: ...`

### Protocol
A way to define structural subtyping (duck typing) with type hints. A class satisfies a protocol if it has the required methods, without explicit inheritance.

*Example*: `class Drawable(Protocol): def draw(self) -> None: ...`

### TypeVar
A type variable used to define generic types. Can be unconstrained, constrained to specific types, or bounded by a superclass.

*Example*: `T = TypeVar('T')`, `T = TypeVar('T', int, float)`, `T = TypeVar('T', bound=Number)`

---

## Design Patterns

### Command Pattern
Encapsulates a request as an object, allowing parameterisation, queuing, logging and undo operations.

*Components*: Command (interface), ConcreteCommand, Invoker, Receiver.

### Decorator Pattern
Attaches additional responsibilities to an object dynamically. Provides a flexible alternative to subclassing.

*Not to be confused with*: Python decorators (`@decorator`), though they serve a similar purpose.

### Factory Pattern
Creates objects without specifying the exact class to create. The factory method returns a product that conforms to a common interface.

*Variants*: Simple Factory, Factory Method, Abstract Factory.

### Observer Pattern
Defines a one-to-many dependency between objects so that when one object (Subject) changes state, all dependents (Observers) are notified.

*Also known as*: Publish-Subscribe, Event-Listener.

### Strategy Pattern
Defines a family of algorithms, encapsulates each one, and makes them interchangeable. Lets the algorithm vary independently from clients that use it.

*Example*: Different integration methods (Euler, RK4) as interchangeable strategies.

---

## SOLID Principles

### Single Responsibility Principle (SRP)
A class should have only one reason to change, meaning it should have only one job or responsibility.

### Open/Closed Principle (OCP)
Software entities should be open for extension but closed for modification. Add new behaviour without changing existing code.

### Liskov Substitution Principle (LSP)
Objects of a superclass should be replaceable with objects of its subclasses without breaking the application.

### Interface Segregation Principle (ISP)
No client should be forced to depend on methods it does not use. Prefer many specific interfaces over one general-purpose interface.

### Dependency Inversion Principle (DIP)
High-level modules should not depend on low-level modules. Both should depend on abstractions.

---

## Type System Terms

### Nominal Typing
Type compatibility based on explicit declarations (inheritance). A type `B` is a subtype of `A` only if explicitly declared.

*Used by*: ABCs in Python, most statically typed languages.

### Structural Typing
Type compatibility based on structure (methods/attributes present). A type `B` is compatible with `A` if it has the required members.

*Used by*: Protocols in Python, TypeScript, Go.

### Type Erasure
The process by which generic type information is removed at runtime. In Python, `List[int]` and `List[str]` are both just `list` at runtime.

### Type Inference
The automatic detection of types without explicit annotations. Mypy infers types where possible.

*Example*: `x = 5` infers `x: int`.

---

## Object-Oriented Terms

### Cohesion
The degree to which elements within a module belong together. High cohesion means closely related functionality.

### Coupling
The degree of interdependence between modules. Low coupling is desirable for maintainability.

### Information Hiding
Restricting access to implementation details. Clients interact only through defined interfaces.

### Inheritance
The mechanism by which a class (child) derives properties and behaviour from another class (parent).

*Types*: Single, multiple, multilevel.

### Polymorphism
The ability of different classes to be treated as instances of the same class through a common interface.

*Types*: Subtype polymorphism (inheritance), parametric polymorphism (generics).

---

## Research Computing Terms

### Simulation Framework
A software structure that provides common infrastructure for running simulations, including state management, time stepping and output handling.

### State Pattern
Allows an object to alter its behaviour when its internal state changes. The object appears to change its class.

### Time Stepping
Advancing a simulation through discrete time intervals. Common methods: Euler, Runge-Kutta, adaptive methods.

---

## Abbreviations

| Abbreviation | Full Term |
|--------------|-----------|
| ABC | Abstract Base Class |
| API | Application Programming Interface |
| DI | Dependency Injection |
| DIP | Dependency Inversion Principle |
| GoF | Gang of Four (Design Patterns authors) |
| ISP | Interface Segregation Principle |
| LSP | Liskov Substitution Principle |
| OCP | Open/Closed Principle |
| OOP | Object-Oriented Programming |
| SRP | Single Responsibility Principle |

---

© 2025 Antonio Clim. All rights reserved.
