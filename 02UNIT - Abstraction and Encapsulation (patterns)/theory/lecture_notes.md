# 02UNIT: Abstraction and Encapsulation

## Lecture Notes — Theoretical Foundations and Practical Applications

---

### Introduction and Historical Context

The preceding unit (01UNIT) established the foundations of computation through Turing machines and abstract syntax trees, revealing that computation fundamentally involves state transitions governed by deterministic rules. This unit elevates our perspective to examine how computational elements can be organised and structured through abstraction and encapsulation—the foundational principles of object-oriented design that enable construction of maintainable research software.

The intellectual lineage of abstraction traces to antiquity. Aristotle's distinction between essential and accidental properties foreshadowed modern abstraction mechanisms. In the twentieth century, Dijkstra's work on structured programming (1968) and Parnas's seminal paper on information hiding (1972) established the theoretical groundwork. Parnas argued that modules should be characterised by their interfaces rather than their implementations—a principle that remains central to software engineering practice five decades hence.

For researchers, principled software design transcends aesthetic preference. Poorly structured code resists modification, defies testing, and harbours subtle defects that propagate silently through computational pipelines. Conversely, well-designed systems permit rapid iteration on scientific hypotheses, reliable reproduction of experimental results, and collaborative development across distributed research teams. The reproducibility crisis afflicting multiple scientific domains often traces, in part, to software that cannot be understood, modified, or reliably executed by investigators other than the original authors.

---

### 1. The Motivation for Abstraction

Consider a common research scenario: you have developed a simulation of epidemic spread using the SIR model. Your supervisor asks you to extend it to include an exposed compartment (SEIR model). Later, a collaborator wants to use your framework for a completely different predator-prey dynamics model. Without proper abstraction, each modification requires extensive surgery on your codebase—changes that risk introducing defects into previously functional code.

**Abstraction** is the process of hiding implementation details whilst exposing only the essential interface. In mathematical terms, abstraction identifies common structure across different concrete instances, enabling reasoning at higher levels of generality.

Formally, let $\mathcal{I}$ denote an interface comprising a set of method signatures $\{m_1, m_2, \ldots, m_k\}$. A concrete type $T$ *satisfies* interface $\mathcal{I}$ if $T$ provides implementations for all methods in $\mathcal{I}$ with compatible signatures. The abstraction ratio $\alpha$ can be defined as:

$$
\alpha(T) = \frac{|\mathcal{I}|}{|M_T|}
$$

where $|M_T|$ is the total number of public methods in $T$. Higher $\alpha$ indicates a more focused, cohesive type with minimal surface area beyond the interface contract.

The key insight is that many simulations share a common structure:
- They maintain some **state** $\sigma \in \Sigma$ from a state space
- They **evolve** that state over time via a transition function $\delta: \Sigma \times \mathbb{R}^+ \to \Sigma$
- They have **termination conditions** defined by a predicate $\phi: \Sigma \to \{0, 1\}$

By abstracting this common structure into an interface, we can write generic code that works with any simulation, present or future, without knowledge of the underlying dynamics.

---

### 2. Encapsulation and Information Hiding

**Encapsulation** bundles data with the methods that operate on that data, yet its greater significance lies in restricting direct access to some components, preventing external code from depending on internal implementation details. The distinction between encapsulation (bundling) and information hiding (access restriction) is often elided in practice, though Parnas emphasised the latter as the more consequential concept.

Consider a simulation state object representing population compartments in an epidemic model. If external code can directly modify the internal arrays, it might inadvertently violate invariants—for instance, producing negative population counts or violating conservation laws where total population must remain constant. By encapsulating the state and providing controlled access methods, we can enforce these invariants programmatically.

The **encapsulation invariant** for a compartmental model can be expressed formally as:

$$
\mathcal{E}(\sigma) = \left( \sum_{i=1}^{k} \sigma_i = N \right) \land \left( \forall i: \sigma_i \geq 0 \right)
$$

where $\sigma_i$ represents the population in compartment $i$, $k$ is the number of compartments, and $N$ is the total population. Well-encapsulated code ensures $\mathcal{E}$ holds as a class invariant—true upon construction and preserved by all public methods.

In Python, encapsulation is conventionally indicated through naming conventions:
- `_single_underscore`: Internal use, discouraged from external access
- `__double_underscore`: Name mangling, stronger discouragement (the attribute becomes `_ClassName__attribute`)
- Avoid stylistic emphasis markers in identifiers: prefer descriptive names for the public interface

However, true encapsulation in Python is more about discipline and convention than enforcement. The language philosophy is "we are all consenting adults here." This cultural norm places responsibility upon the developer to respect interface boundaries, even when technical barriers are absent.

---

### 3. The SOLID Principles

The SOLID principles, articulated by Robert C. Martin across multiple publications in the late 1990s and early 2000s, provide guidelines for creating maintainable object-oriented systems. The acronym, coined by Michael Feathers, encapsulates five interdependent principles that, when applied judiciously, yield software resilient to changing requirements.

#### Single Responsibility Principle (SRP)

*A class should have only one reason to change.*

Formally, if we consider the set of change vectors $\mathcal{C} = \{c_1, c_2, \ldots, c_n\}$ representing distinct stakeholder requirements, a class $C$ satisfies SRP when:

$$
|\{c \in \mathcal{C} : c \text{ necessitates change to } C\}| = 1
$$

In research code, this often means separating:
- **Model logic**: The mathematical equations being simulated (domain specialists)
- **Numerical methods**: How equations are solved—Euler, RK4, adaptive methods (numerical analysts)
- **Visualisation**: How results are displayed (presentation requirements)
- **Data I/O**: How results are saved and loaded (data format standards)

When your SIR model class also handles file writing and plotting, a change to your plot style requires modifying the same class that implements epidemic dynamics. This coupling makes the code fragile—changes in unrelated concerns ripple through the codebase.

#### Open/Closed Principle (OCP)

*Software entities should be open for extension but closed for modification.*

This principle, attributed to Bertrand Meyer (1988), appears paradoxical until one recognises the role of abstraction. Your simulation framework should allow adding new models without modifying existing code. This is achieved through polymorphism: define an interface (Protocol or ABC), then implement new models as new classes.

```python
class Simulable(Protocol[StateT]):
    def state(self) -> StateT: ...
    def step(self, dt: float) -> None: ...
    def is_done(self) -> bool: ...
```

The `SimulationRunner` works with any `Simulable` without knowing the concrete type. Adding a new model requires only implementing the interface—the runner remains closed to modification, yet the system is open to extension through the addition of new conforming types.

#### Liskov Substitution Principle (LSP)

*Subtypes must be substitutable for their base types.*

Barbara Liskov formalised this principle in 1987: if $S$ is a declared subtype of $T$, then objects of type $S$ may be substituted for objects of type $T$ without altering any desirable properties of the programme. More precisely:

$$
\forall x: T.\ \phi(x) \implies \forall y: S.\ \phi(y)
$$

where $\phi$ represents any property provable about objects of type $T$.

If `SIRSimulation` implements `Simulable`, then code expecting a `Simulable` must work correctly with an `SIRSimulation` instance. This requires that subclasses honour the contracts of their parent types—preconditions may be weakened, postconditions may be strengthened, but never the reverse.

A violation example: if `Simulable.step()` promises not to raise exceptions under normal operation, but your subclass raises an exception when `dt` exceeds some threshold, you have violated LSP. The client code, expecting the base type's behaviour, will fail unexpectedly.

#### Interface Segregation Principle (ISP)

*Clients should not be forced to depend on interfaces they do not use.*

If your `Simulable` protocol required methods for 3D visualisation, models that only need 2D plotting would be burdened with unnecessary implementation. Better to have separate, focused protocols:

```python
class Simulable(Protocol[StateT]):
    """Core simulation interface."""
    def state(self) -> StateT: ...
    def step(self, dt: float) -> None: ...
    def is_done(self) -> bool: ...

class Visualisable(Protocol):
    """Visualisation capability interface."""
    def get_plot_data(self) -> dict[str, Any]: ...
    def get_dimensions(self) -> int: ...
```

Models can implement one or both as appropriate. This segregation prevents the accumulation of "fat interfaces" that burden implementers with irrelevant obligations.

#### Dependency Inversion Principle (DIP)

*High-level modules should not depend on low-level modules. Both should depend on abstractions.*

Your `SimulationRunner` (high-level policy) should not import and use `SIRSimulation` (low-level detail) directly. Instead, both depend on the `Simulable` protocol (abstraction). This inversion—where dependencies flow towards abstractions rather than concretes—allows the runner to work with any simulation and allows simulations to be developed independently, perhaps by different team members or even different research groups.

---

### 4. Python Protocols versus Abstract Base Classes

Python offers two approaches for defining interfaces, each with distinct characteristics rooted in different typing philosophies.

**Abstract Base Classes (ABCs)** use nominal (name-based) typing:
```python
from abc import ABC, abstractmethod

class Simulable(ABC):
    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance simulation by time increment dt."""
        pass
```

Classes must explicitly inherit from the ABC to be considered implementations. This inheritance relationship is checked nominally—the type's name must appear in the inheritance chain.

**Protocols** use structural (duck) typing:
```python
from typing import Protocol

class Simulable(Protocol):
    def step(self, dt: float) -> None: ...
```

Any class with a matching `step` method satisfies the Protocol, regardless of inheritance. The type checker examines structure—the presence and signatures of methods—rather than names.

For research software, Protocols offer significant advantages:
- Integration with existing libraries without modification (wrap third-party classes)
- More Pythonic "duck typing" philosophy
- Better compatibility with generic programming
- Runtime checking available via `typing.runtime_checkable`
- No import dependencies between the protocol definition and implementations

---

### 5. Design Patterns for Research Software

Design patterns are reusable solutions to common problems, catalogued by Gamma, Helm, Johnson, and Vlissides in their influential 1994 text. Three patterns are particularly useful in research computing.

#### Strategy Pattern

**Problem**: You need to switch between different algorithms for the same task—different numerical integrators, different optimisation methods, different distance metrics.

**Solution**: Encapsulate each algorithm in a class implementing a common interface. The context class holds a reference to a strategy and delegates to it.

```python
class IntegrationStrategy(Protocol):
    """Strategy interface for numerical integration."""
    def integrate(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int
    ) -> float: ...

class TrapezoidalStrategy:
    """Trapezoidal rule implementation."""
    def integrate(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        n: int
    ) -> float:
        h = (b - a) / n
        result = 0.5 * (f(a) + f(b))
        for i in range(1, n):
            result += f(a + i * h)
        return result * h
```

You can swap strategies at runtime, compare results from different methods, and add new strategies without modifying existing code. The context class remains oblivious to which strategy is currently in use.

#### Observer Pattern

**Problem**: When your simulation state changes, multiple components need to respond—update plots, log metrics, check convergence criteria, trigger parameter adaptation.

**Solution**: Define an observer interface. Subjects maintain a list of observers and notify them of state changes.

```python
class SimulationObserver(Protocol[T]):
    """Observer interface for simulation events."""
    def on_step(self, state: T, time: float) -> None: ...
    def on_complete(self, final_state: T) -> None: ...

class Observable(Generic[T]):
    """Mixin providing observation capability."""
    def __init__(self) -> None:
        self._observers: list[SimulationObserver[T]] = []
    
    def subscribe(self, observer: SimulationObserver[T]) -> None:
        self._observers.append(observer)
    
    def notify_step(self, state: T, time: float) -> None:
        for observer in self._observers:
            observer.on_step(state, time)
```

This decouples the simulation from its observers. The simulation does not need to know whether it is being plotted, logged, or both—it merely announces state changes, and interested parties respond accordingly.

#### Factory Pattern

**Problem**: Object creation involves complex logic or configuration. You want to decouple client code from concrete classes.

**Solution**: Create a factory class or method that encapsulates instantiation logic.

In agent-based models, a factory can create heterogeneous agent populations:
```python
class AgentFactory(Protocol):
    """Factory interface for agent creation."""
    def create(self, agent_id: int, position: tuple[float, float]) -> Agent: ...

class HeterogeneousFactory:
    """Creates agents with varied characteristics."""
    def __init__(self, behaviour_distribution: dict[str, float]):
        self._distribution = behaviour_distribution
    
    def create(self, agent_id: int, position: tuple[float, float]) -> Agent:
        behaviour_type = self._sample_behaviour()
        return Agent(agent_id, position, behaviour_type)
```

Different factories create different agent types. The population manager does not need to know which types, enabling clean separation between population structure decisions and agent implementation details.

---

### 6. Type Systems and Safety

Python's type system, while optional and gradual, provides significant benefits for research software:

1. **Documentation**: Types serve as machine-checked documentation that cannot become stale
2. **Early error detection**: Type checkers catch errors before runtime, often before commit
3. **IDE support**: Better autocompletion and refactoring tools
4. **Design guidance**: Thinking about types encourages better interface design

Generic types (`TypeVar`, `Generic`) enable type-safe code reuse whilst preserving type information:
```python
StateT = TypeVar('StateT')

class SimulationRunner(Generic[StateT]):
    """Generic runner preserving state type information."""
    def __init__(self, simulation: Simulable[StateT]):
        self._simulation = simulation
    
    def run(self, max_time: float) -> SimulationResult[StateT]:
        # Type checker knows final_state: StateT
        ...
```

The type parameter flows through, ensuring type consistency from input through output.

---

### 7. Composition versus Inheritance

Classical OOP emphasises inheritance as the mechanism for code reuse. Modern practice favours composition—the construction of complex objects from simpler components. The mantra is "favour composition over inheritance."

**Inheritance problems**:
- Tight coupling between parent and child (changes propagate)
- Fragile base class problem (parent changes break children unexpectedly)
- Limited flexibility (single inheritance in Python for most purposes)
- Violation of encapsulation (children depend on parent internals)

**Composition advantages**:
- Loose coupling through interfaces
- Flexible runtime configuration (swap components)
- Clear, explicit dependencies
- Easier testing (mock individual components)

Consider extending a simulation to add logging:

*Inheritance approach* (problematic):
```python
class LoggingSIRSimulation(SIRSimulation):
    def step(self, dt: float) -> None:
        super().step(dt)
        self._log_state()  # Coupled to SIRSimulation implementation
```

*Composition approach* (preferred):
```python
class LoggingSimulationRunner:
    def __init__(self, simulation: Simulable, logger: Logger):
        self._simulation = simulation
        self._logger = logger
    
    def run_step(self, dt: float) -> None:
        self._simulation.step(dt)
        self._logger.log(self._simulation.state())
```

The composition approach works with any simulation and any logger, demonstrating superior flexibility and testability.

---

### 8. Research Application: Epidemic Modelling Framework

Let us trace how these principles apply to building an epidemic modelling framework.

**Requirements**:
- Support multiple compartmental models (SIR, SEIR, SIS, SIRS)
- Allow different numerical methods (Euler, RK4, adaptive RK45)
- Enable real-time visualisation during long simulations
- Record metrics history for post-hoc analysis

**Design**:
1. **Simulable Protocol**: Abstract interface for all models
2. **SIRSimulation, SEIRSimulation, etc.**: Concrete implementations
3. **IntegrationStrategy Protocol**: Swappable numerical methods
4. **Observable pattern**: Simulation notifies visualisers of state changes
5. **SimulationRunner**: Generic orchestration independent of model details

This design allows:
- Adding new models without modifying the runner (OCP)
- Comparing models with identical numerical methods (Strategy)
- Swapping visualisation during runtime (Observer)
- Testing models in isolation with mock observers
- Researchers unfamiliar with the codebase to add models by implementing a simple interface

---

### 9. Common Pitfalls and Anti-Patterns

**God Class**: A class that knows too much and does too much. When a class exceeds 500 lines or has more than 10 methods, consider splitting responsibilities.

**Feature Envy**: A method that uses more features of another class than its own. Consider moving the method to the class whose data it manipulates.

**Premature Abstraction**: Creating interfaces for code that will never vary. Abstraction has costs—indirection, complexity, cognitive load. Wait for the need to emerge from actual requirements before abstracting.

**Inheritance for Code Reuse**: Using inheritance solely to reuse code rather than to model "is-a" relationships. Use composition instead.

**Leaky Abstraction**: When implementation details leak through the interface, forcing clients to understand internals. Ensure your abstractions are complete and self-contained.

---

### 10. Practical Guidelines

1. **Start simple**: Begin with concrete implementations. Extract abstractions when patterns emerge across multiple concrete cases.

2. **Write tests first**: Tests reveal the interface you need. They also ensure refactoring preserves behaviour.

3. **Use type hints**: They document intent and catch errors early. Configure mypy with strict settings.

4. **Favour immutability**: Immutable state objects prevent accidental corruption and simplify reasoning about programme behaviour.

5. **Document invariants**: State clearly what must always be true about your objects—in docstrings and, where possible, in assertions.

---

### Summary

Abstraction and encapsulation are fundamental to building maintainable research software. The SOLID principles provide guidance for designing classes and their relationships. Design patterns offer proven solutions to recurring problems. Python's Protocol system enables flexible, type-safe interface definitions that align with the language's dynamic nature whilst providing static verification benefits.

The goal is not perfect design from the start—that is impossible. The goal is design that accommodates change gracefully. When requirements evolve, well-designed code bends; poorly designed code breaks. The investment in principled design pays dividends in reduced debugging time, easier collaboration, and more reliable scientific results.

In 03UNIT, we will apply these principles to build a benchmarking framework for algorithmic complexity analysis. The design patterns learned this unit will enable us to create a flexible system for comparing algorithm performance across problem sizes and implementation strategies.

---

### References

1. Martin, R. C. (2003). *Agile Software Development, Principles, Patterns, and Practices*. Pearson.
2. Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design Patterns: Elements of Reusable Object-Oriented Software*. Addison-Wesley.
3. Liskov, B. H., & Wing, J. M. (1994). A behavioral notion of subtyping. *ACM Transactions on Programming Languages and Systems*, 16(6), 1811–1841.
4. Parnas, D. L. (1972). On the criteria to be used in decomposing systems into modules. *Communications of the ACM*, 15(12), 1053–1058.
5. Van Rossum, G., et al. (2014). PEP 484 — Type Hints. Python Enhancement Proposals.
6. Levkivskyi, I., et al. (2017). PEP 544 — Protocols: Structural subtyping. Python Enhancement Proposals.

---

© 2025 Antonio Clim. All rights reserved.
