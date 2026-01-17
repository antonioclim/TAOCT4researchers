# 02UNIT: Abstraction and Encapsulation

## Lecture Notes — Theoretical Foundations

### Introduction

The preceding unit (01UNIT) established the foundations of computation through Turing machines and abstract syntax trees, revealing that computation fundamentally involves state transitions governed by deterministic rules. This unit elevates our perspective to examine how computational elements can be organised and structured through abstraction and encapsulation—the cornerstone principles of object-oriented design that enable construction of maintainable research software.

For researchers, principled software design transcends aesthetic preference. Poorly structured code resists modification, defies testing, and harbours subtle defects. Conversely, well-designed systems permit rapid iteration on scientific hypotheses, reliable reproduction of experimental results, and collaborative development across distributed research teams.

---

### 1. The Motivation for Abstraction

Consider a common research scenario: you have developed a simulation of epidemic spread using the SIR model. Your supervisor asks you to extend it to include an exposed compartment (SEIR model). Later, a collaborator wants to use your framework for a completely different predator-prey dynamics model. Without proper abstraction, each modification requires extensive surgery on your codebase.

**Abstraction** is the process of hiding implementation details whilst exposing only the essential interface. In mathematical terms, it is the identification of common structure across different concrete instances.

The key insight is that many simulations share a common structure:
- They maintain some **state**
- They **evolve** that state over time
- They have **termination conditions**

By abstracting this common structure into an interface, we can write generic code that works with any simulation, present or future.

---

### 2. Encapsulation and Information Hiding

**Encapsulation** bundles data with the methods that operate on that data. More importantly, it restricts direct access to some components, preventing external code from depending on internal implementation details.

Consider a simulation state object. If external code can directly modify the internal arrays, it might inadvertently violate invariants (e.g. negative population counts, energy non-conservation). By encapsulating the state and providing controlled access methods, we can enforce these invariants.

In Python, encapsulation is conventionally indicated through naming conventions:
- `_single_underscore`: Internal use, discouraged from external access
- `__double_underscore`: Name mangling, stronger discouragement
- No underscore: Public interface

However, true encapsulation in Python is more about discipline and convention than enforcement. The language philosophy is "we are all consenting adults here."

---

### 3. The SOLID Principles

The SOLID principles, articulated by Robert C. Martin, provide guidelines for creating maintainable object-oriented systems. Let us examine each in the context of research software.

#### Single Responsibility Principle (SRP)

*A class should have only one reason to change.*

In research code, this often means separating:
- **Model logic**: The mathematical equations being simulated
- **Numerical methods**: How equations are solved (Euler, RK4, etc.)
- **Visualisation**: How results are displayed
- **Data I/O**: How results are saved and loaded

When your SIR model class also handles file writing and plotting, a change to your plot style requires modifying the same class that implements epidemic dynamics. This coupling makes the code fragile.

#### Open/Closed Principle (OCP)

*Software entities should be open for extension but closed for modification.*

Your simulation framework should allow adding new models without modifying existing code. This is achieved through polymorphism: define an interface (Protocol or ABC), then implement new models as new classes.

```python
class Simulable(Protocol[StateT]):
    def state(self) -> StateT: ...
    def step(self, dt: float) -> None: ...
    def is_done(self) -> bool: ...
```

The `SimulationRunner` works with any `Simulable` without knowing the concrete type. Adding a new model requires only implementing the interface—no changes to the runner.

#### Liskov Substitution Principle (LSP)

*Subtypes must be substitutable for their base types.*

If `SIRSimulation` implements `Simulable`, then code expecting a `Simulable` must work correctly with an `SIRSimulation`. This requires that subclasses honour the contracts (preconditions, postconditions, invariants) of their parent types.

A violation example: if `Simulable.step()` promises not to raise exceptions under normal operation, but your subclass raises an exception when `dt` is too large, you have violated LSP.

#### Interface Segregation Principle (ISP)

*Clients should not be forced to depend on interfaces they do not use.*

If your `Simulable` protocol required methods for 3D visualisation, models that only need 2D plotting would be burdened with unnecessary implementation. Better to have separate protocols:

```python
class Simulable(Protocol[StateT]):
    # Core simulation interface
    ...

class Visualisable(Protocol):
    # Visualisation interface
    def get_plot_data(self) -> dict[str, Any]: ...
```

Models can implement one or both as appropriate.

#### Dependency Inversion Principle (DIP)

*High-level modules should not depend on low-level modules. Both should depend on abstractions.*

Your `SimulationRunner` (high-level) should not import and use `SIRSimulation` (low-level) directly. Instead, both depend on the `Simulable` protocol (abstraction). This allows the runner to work with any simulation and allows simulations to be developed independently.

---

### 4. Python Protocols vs Abstract Base Classes

Python offers two approaches for defining interfaces:

**Abstract Base Classes (ABCs)** use nominal (name-based) typing:
```python
from abc import ABC, abstractmethod

class Simulable(ABC):
    @abstractmethod
    def step(self, dt: float) -> None:
        pass
```

Classes must explicitly inherit from the ABC to be considered implementations.

**Protocols** use structural (duck) typing:
```python
from typing import Protocol

class Simulable(Protocol):
    def step(self, dt: float) -> None: ...
```

Any class with a matching `step` method satisfies the Protocol, regardless of inheritance.

For research software, Protocols offer significant advantages:
- Integration with existing libraries without modification
- More Pythonic "duck typing" philosophy
- Better compatibility with generic programming
- Runtime checking available via `typing.runtime_checkable`

---

### 5. Design Patterns for Research Software

Design patterns are reusable solutions to common problems. Three patterns are particularly useful in research computing.

#### Strategy Pattern

**Problem**: You need to switch between different algorithms for the same task (e.g. different numerical integrators, different optimisation methods).

**Solution**: Encapsulate each algorithm in a class implementing a common interface. The context class holds a reference to a strategy and delegates to it.

```python
class IntegrationStrategy(Protocol):
    def integrate(self, f: Callable, a: float, b: float) -> float: ...

class NumericalIntegrator:
    def __init__(self, strategy: IntegrationStrategy):
        self._strategy = strategy
    
    def compute(self, f: Callable, a: float, b: float) -> float:
        return self._strategy.integrate(f, a, b)
```

You can swap strategies at runtime, compare results from different methods and add new strategies without modifying existing code.

#### Observer Pattern

**Problem**: When your simulation state changes, multiple components need to respond (update plots, log metrics, check convergence).

**Solution**: Define an observer interface. Subjects maintain a list of observers and notify them of state changes.

```python
class Observable(Generic[T]):
    def subscribe(self, observer: Observer[T]) -> None: ...
    def notify(self, data: T) -> None: ...
```

This decouples the simulation from its observers. The simulation does not need to know whether it is being plotted, logged or both.

#### Factory Pattern

**Problem**: Object creation involves complex logic or configuration. You want to decouple client code from concrete classes.

**Solution**: Create a factory class or method that encapsulates instantiation logic.

In agent-based models, a factory can create heterogeneous agent populations:
```python
class AgentFactory(Protocol):
    def create(self, agent_id: int, position: tuple) -> Agent: ...
```

Different factories create different agent types. The population manager does not need to know which types.

---

### 6. Type Systems and Safety

Python's type system, while optional, provides significant benefits for research software:

1. **Documentation**: Types serve as machine-checked documentation
2. **Early error detection**: Type checkers catch errors before runtime
3. **IDE support**: Better autocompletion and refactoring
4. **Design guidance**: Thinking about types encourages better design

Generic types (`TypeVar`, `Generic`) enable type-safe code reuse:
```python
StateT = TypeVar('StateT')

class SimulationRunner(Generic[StateT]):
    def run(self) -> SimulationResult[StateT]: ...
```

The type parameter flows through, ensuring type consistency.

---

### 7. Composition vs Inheritance

Classical OOP emphasises inheritance as the mechanism for code reuse. Modern practice favours composition. The mantra is "favour composition over inheritance."

**Inheritance problems**:
- Tight coupling between parent and child
- Fragile base class problem (parent changes break children)
- Limited flexibility (single inheritance in Python for most purposes)

**Composition advantages**:
- Loose coupling through interfaces
- Flexible runtime configuration
- Clear dependencies

Consider extending a simulation to add logging:

*Inheritance approach* (problematic):
```python
class LoggingSIRSimulation(SIRSimulation):
    def step(self, dt: float) -> None:
        super().step(dt)
        self._log_state()
```

*Composition approach* (preferred):
```python
class LoggingSimulationRunner:
    def __init__(self, simulation: Simulable, logger: Logger):
        self._simulation = simulation
        self._logger = logger
```

The composition approach works with any simulation and any logger, demonstrating superior flexibility.

---

### 8. Research Application: Epidemic Modelling Framework

Let us trace how these principles apply to building an epidemic modelling framework.

**Requirements**:
- Support multiple compartmental models (SIR, SEIR, SIS)
- Allow different numerical methods
- Enable real-time visualisation
- Record metrics history

**Design**:
1. **Simulable Protocol**: Abstract interface for all models
2. **SIRSimulation, SEIRSimulation**: Concrete implementations
3. **IntegrationStrategy**: Swappable numerical methods
4. **Observable pattern**: Simulation notifies visualisers
5. **SimulationRunner**: Generic orchestration

This design allows:
- Adding new models without modifying the runner
- Comparing models with identical numerical methods
- Swapping visualisation during runtime
- Testing models in isolation

---

### 9. Common Pitfalls and Anti-Patterns

**God Class**: A class that knows too much and does too much. Split responsibilities.

**Feature Envy**: A method that uses more features of another class than its own. Consider moving the method.

**Premature Abstraction**: Creating interfaces for code that will never vary. Wait for the need to emerge.

**Inheritance for Code Reuse**: Using inheritance solely to reuse code rather than to model "is-a" relationships. Use composition instead.

**Leaky Abstraction**: When implementation details leak through the interface. Ensure your abstractions are complete.

---

### 10. Practical Guidelines

1. **Start simple**: Begin with concrete implementations. Extract abstractions when patterns emerge.

2. **Write tests first**: Tests reveal the interface you need. They also ensure refactoring preserves behaviour.

3. **Use type hints**: They document intent and catch errors early.

4. **Favour immutability**: Immutable state objects prevent accidental corruption.

5. **Document invariants**: State clearly what must always be true about your objects.

---

### Summary

Abstraction and encapsulation are fundamental to building maintainable research software. The SOLID principles provide guidance for designing classes and their relationships. Design patterns offer proven solutions to recurring problems. Python's Protocol system enables flexible, type-safe interface definitions.

The goal is not perfect design from the start—that is impossible. The goal is design that accommodates change gracefully. When requirements evolve, well-designed code bends; poorly designed code breaks.

In 03UNIT, we will apply these principles to build a benchmarking framework for algorithmic complexity analysis. The design patterns learned this week will enable us to create a flexible system for comparing algorithm performance.

---

### References

1. Martin, R. C. (2003). *Agile Software Development, Principles, Patterns, and Practices*
2. Gamma, E., et al. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*
3. Van Rossum, G. (2015). PEP 484 — Type Hints
4. Hettinger, R. (2020). PEP 544 — Protocols: Structural subtyping

---

© 2025 Antonio Clim. All rights reserved.
