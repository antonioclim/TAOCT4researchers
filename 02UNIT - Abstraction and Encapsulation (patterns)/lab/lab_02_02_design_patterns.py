#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Lab 2: Design Patterns Catalogue
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Design patterns are reusable solutions to common problems in software design.
In research computing, well-chosen patterns can dramatically improve code
maintainability, testability and extensibility.

This lab presents the most useful patterns for scientific software:
- Strategy: Interchangeable algorithms
- Observer: Event-driven updates
- Factory: Flexible object creation
- Decorator: Dynamic behaviour extension
- Command: Encapsulated operations

PREREQUISITES
─────────────
- 02UNIT, Lab 1: Protocol-based simulation framework
- Python: Intermediate proficiency with type hints
- Libraries: dataclasses, abc, functools

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Recognise when to apply each design pattern
2. Implement patterns using Python protocols and generics
3. Refactor existing code to use appropriate patterns
4. Evaluate trade-offs between different design approaches

ESTIMATED TIME
──────────────
- Reading: 45 minutes
- Coding: 75 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
numpy>=1.24
matplotlib>=3.7

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Generic, Protocol, TypeVar

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: STRATEGY PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ STRATEGY PATTERN                                                              ║
║                                                                               ║
║ Intent: Define a family of algorithms, encapsulate each one, and make them    ║
║         interchangeable. Strategy lets the algorithm vary independently from  ║
║         clients that use it.                                                  ║
║                                                                               ║
║ When to Use:                                                                  ║
║ - Multiple algorithms exist for a task                                        ║
║ - You need to switch algorithms at runtime                                    ║
║ - You want to isolate algorithm implementation from calling code              ║
║                                                                               ║
║ Research Examples:                                                            ║
║ - Numerical integration (Euler, RK4, RK45)                                    ║
║ - Optimisation algorithms (gradient descent, Adam, L-BFGS)                    ║
║ - Distance metrics (Euclidean, Manhattan, cosine)                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


class IntegrationStrategy(Protocol):
    """
    Protocol for numerical integration strategies.
    
    Any class implementing this protocol can be used to integrate
    a function over an interval [a, b].
    """
    
    def integrate(
        self, 
        f: Callable[[float], float], 
        a: float, 
        b: float
    ) -> float:
        """
        Integrate function f over interval [a, b].
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            
        Returns:
            Approximate integral value
        """
        ...


@dataclass
class RectangleRule:
    """
    Rectangle rule (midpoint) integration.
    
    Approximates the integral using rectangles with heights
    determined at interval midpoints:
    
    ∫f(x)dx ≈ Σᵢ f(xᵢ) · Δx
    
    where xᵢ is the midpoint of each subinterval.
    
    Attributes:
        n_intervals: Number of subintervals
    """
    n_intervals: int = 100

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return "Rectangle rule"
    
    def integrate(
        self, 
        f: Callable[[float], float], 
        a: float, 
        b: float
    ) -> float:
        """Integrate using the rectangle rule."""
        dx = (b - a) / self.n_intervals
        total = 0.0
        for i in range(self.n_intervals):
            midpoint = a + (i + 0.5) * dx
            total += f(midpoint) * dx
        return total


@dataclass
class TrapezoidRule:
    """
    Trapezoidal rule integration.
    
    Approximates the integral using trapezoids:
    
    ∫f(x)dx ≈ Δx/2 · [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]
    
    More accurate than rectangle rule for smooth functions.
    
    Attributes:
        n_intervals: Number of subintervals
    """
    n_intervals: int = 100

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return "Trapezoid rule"
    
    def integrate(
        self, 
        f: Callable[[float], float], 
        a: float, 
        b: float
    ) -> float:
        """Integrate using the trapezoidal rule."""
        dx = (b - a) / self.n_intervals
        total = 0.5 * (f(a) + f(b))
        for i in range(1, self.n_intervals):
            x = a + i * dx
            total += f(x)
        return total * dx


@dataclass
class SimpsonRule:
    """
    Simpson's rule integration.
    
    Uses parabolic interpolation between points:
    
    ∫f(x)dx ≈ Δx/3 · [f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(xₙ)]
    
    More accurate than trapezoidal rule; exact for polynomials up to degree 3.
    
    Attributes:
        n_intervals: Number of subintervals (must be even)
    """
    n_intervals: int = 100

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return "Simpson's rule"
    
    def __post_init__(self) -> None:
        """Ensure n_intervals is even."""
        if self.n_intervals % 2 != 0:
            self.n_intervals += 1
    
    def integrate(
        self, 
        f: Callable[[float], float], 
        a: float, 
        b: float
    ) -> float:
        """Integrate using Simpson's rule."""
        dx = (b - a) / self.n_intervals
        total = f(a) + f(b)
        
        for i in range(1, self.n_intervals):
            x = a + i * dx
            coefficient = 4 if i % 2 == 1 else 2
            total += coefficient * f(x)
        
        return total * dx / 3


class NumericalIntegrator:
    """
    Context class that uses an integration strategy.
    
    Demonstrates the Strategy pattern: the integration algorithm
    can be changed at runtime without modifying this class.
    
    Example:
        >>> integrator = NumericalIntegrator(SimpsonRule(n_intervals=1000))
        >>> result = integrator.compute(lambda x: x**2, 0, 1)
        >>> print(f"∫x² dx from 0 to 1 = {result:.6f}")  # Should be ~0.333333
    """
    
    def __init__(self, strategy: IntegrationStrategy) -> None:
        """
        Initialise with an integration strategy.
        
        Args:
            strategy: Any object implementing IntegrationStrategy
        """
        self._strategy = strategy
    
    @property
    def strategy(self) -> IntegrationStrategy:
        """Get the current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: IntegrationStrategy) -> None:
        """Set a new strategy."""
        self._strategy = strategy
    
    def compute(
        self, 
        f: Callable[[float], float], 
        a: float, 
        b: float
    ) -> float:
        """
        Compute the integral using the current strategy.
        
        Args:
            f: Function to integrate
            a: Lower bound
            b: Upper bound
            
        Returns:
            Approximate integral value
        """
        return self._strategy.integrate(f, a, b)

    def integrate(
        self,
        f: Callable[[float], float],
        a: float,
        b: float,
        *,
        n: int = 100,
    ) -> float:
        """Integrate a function using the current strategy.

        The unit tests use a uniform interface ``integrate(..., n=...)``.
        Individual strategies store the discretisation as ``n_intervals``.
        """
        if hasattr(self._strategy, "n_intervals"):
            setattr(self._strategy, "n_intervals", int(n))
            # Simpson requires an even number of intervals.
            if isinstance(self._strategy, SimpsonRule) and self._strategy.n_intervals % 2 != 0:
                self._strategy.n_intervals += 1
        return self.compute(f, float(a), float(b))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: OBSERVER PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ OBSERVER PATTERN                                                              ║
║                                                                               ║
║ Intent: Define a one-to-many dependency between objects so that when one      ║
║         object changes state, all its dependents are notified automatically.  ║
║                                                                               ║
║ When to Use:                                                                  ║
║ - Changes in one object must trigger updates in others                        ║
║ - You want loose coupling between the subject and observers                   ║
║ - The set of observers may change at runtime                                  ║
║                                                                               ║
║ Research Examples:                                                            ║
║ - Real-time visualisation of simulation progress                              ║
║ - Logging and metrics collection                                              ║
║ - Event-driven data pipelines                                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

T = TypeVar('T')

# The unit tests treat observers as callables (typically bound ``update`` methods).


@dataclass
class Observable(Generic[T]):
    """
    Subject that notifies observers of state changes.
    
    Implements the Observable side of the Observer pattern.
    Observers can subscribe and unsubscribe dynamically.
    
    Example:
        >>> subject = Observable[int]()
        >>> subject.subscribe(logger_observer)
        >>> subject.notify(42)  # All observers receive 42
    """
    _observers: list[Callable[[T], None]] = field(default_factory=list)
    
    def subscribe(self, observer: Callable[[T], None]) -> None:
        """
        Add an observer to the subscription list.
        
        Args:
            observer: Observer to add
        """
        if observer not in self._observers:
            self._observers.append(observer)
            logger.debug(f"Observer {observer} subscribed")
    
    def unsubscribe(self, observer: Callable[[T], None]) -> None:
        """
        Remove an observer from the subscription list.
        
        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
            logger.debug(f"Observer {observer} unsubscribed")
    
    def notify(self, data: T) -> None:
        """
        Notify all observers of a state change.
        
        Args:
            data: Data to send to all observers
        """
        for observer in list(self._observers):
            observer(data)


@dataclass
class SimulationState:
    """State data broadcast to observers."""
    step: int
    time: float
    values: dict[str, float]


@dataclass
class ConsoleLogger:
    """Observer that logs simulation state to console."""
    prefix: str = "[LOG]"
    
    def update(self, data: SimulationState) -> None:
        """Log the simulation state."""
        logger.info(
            f"{self.prefix} Step {data.step}: t={data.time:.4f}, "
            f"values={data.values}"
        )


@dataclass
class MetricsCollector:
    """Collect numeric observations and expose summary statistics.

    The unit tests use this observer as a minimal example of state aggregation.
    The class therefore provides `count`, `mean`, `min_value` and `max_value`.
    """

    notifications: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        """Record an observed value."""
        self.notifications.append(float(value))

    @property
    def count(self) -> int:
        return len(self.notifications)

    @property
    def mean(self) -> float:
        if not self.notifications:
            return float("nan")
        return float(sum(self.notifications) / len(self.notifications))

    @property
    def min_value(self) -> float:
        return min(self.notifications) if self.notifications else float("nan")

    @property
    def max_value(self) -> float:
        return max(self.notifications) if self.notifications else float("nan")

    def get_timeseries(self) -> list[float]:
        """Return the collected values as a new list."""
        return list(self.notifications)

@dataclass
class ProgressBar:
    """Observer that displays a progress bar."""
    total_steps: int
    width: int = 40
    _current: int = field(default=0, init=False)
    
    def update(self, data: SimulationState) -> None:
        """Update the progress bar."""
        self._current = data.step
        progress = self._current / self.total_steps
        filled = int(self.width * progress)
        bar = '█' * filled + '░' * (self.width - filled)
        percentage = progress * 100
        print(f"\r[{bar}] {percentage:.1f}%", end='', flush=True)
        if self._current >= self.total_steps:
            print()  # New line when complete


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: FACTORY PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ FACTORY PATTERN                                                               ║
║                                                                               ║
║ Intent: Define an interface for creating objects, but let subclasses or       ║
║         factory methods decide which class to instantiate.                    ║
║                                                                               ║
║ When to Use:                                                                  ║
║ - Object creation involves complex logic                                      ║
║ - You want to decouple client code from concrete classes                      ║
║ - The exact class to instantiate depends on runtime conditions                ║
║                                                                               ║
║ Research Examples:                                                            ║
║ - Creating agents in agent-based models                                       ║
║ - Instantiating experiment configurations                                     ║
║ - Building model components from configuration files                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


@dataclass(frozen=True)
class Agent:
    """Base class for agents in a simulation."""
    agent_id: str
    position: tuple[float, float] = (0.0, 0.0)
    
    def act(self) -> str:
        """Perform an action. Override in subclasses."""
        return f"Agent {self.agent_id} at {self.position}"


@dataclass(frozen=True)
class CooperativeAgent(Agent):
    """Agent that cooperates with neighbours."""
    cooperation_rate: float = 0.8
    
    def act(self) -> str:
        return f"Cooperative Agent {self.agent_id} (rate={self.cooperation_rate})"


@dataclass(frozen=True)
class CompetitiveAgent(Agent):
    """Agent that competes with neighbours."""
    aggression: float = 0.5
    
    def act(self) -> str:
        return f"Competitive Agent {self.agent_id} (aggression={self.aggression})"


@dataclass(frozen=True)
class RandomAgent(Agent):
    """Agent with random behaviour."""
    seed: int = 42
    
    def act(self) -> str:
        return f"Random Agent {self.agent_id} (seed={self.seed})"


class AgentFactory(Protocol):
    """Protocol for agent factories."""
    
    def create(self, agent_id: str, position: tuple[float, float] = (0.0, 0.0)) -> Agent:
        """Create an agent."""
        ...


@dataclass
class CooperativeFactory:
    """Factory that creates cooperative agents."""
    cooperation_rate: float = 0.8
    
    def create(self, agent_id: str, position: tuple[float, float] = (0.0, 0.0)) -> CooperativeAgent:
        """Create a cooperative agent."""
        return CooperativeAgent(
            agent_id=agent_id,
            position=position,
            cooperation_rate=self.cooperation_rate
        )


@dataclass
class CompetitiveFactory:
    """Factory that creates competitive agents."""

    aggression: float = 0.5

    def create(self, agent_id: str, position: tuple[float, float] = (0.0, 0.0)) -> CompetitiveAgent:
        """Create a competitive agent."""
        return CompetitiveAgent(
            agent_id=agent_id,
            position=position,
            aggression=self.aggression,
        )


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------


class CooperativeAgentFactory(CooperativeFactory):
    """Compatibility name expected by the test suite.

    The teaching material originally used the shorter name ``CooperativeFactory``.
    The unit tests, however, import ``CooperativeAgentFactory`` to make the intent
    explicit at the call site. The behaviour is identical.
    """


class CompetitiveAgentFactory(CompetitiveFactory):
    """Compatibility name expected by the test suite.

    The behaviour is identical to :class:`CompetitiveFactory`.
    """
    # Inherits behaviour from CompetitiveFactory.


@dataclass
class MixedFactory:
    """
    Factory that creates agents of different types based on probability.
    
    Demonstrates the Factory pattern with probabilistic selection.
    """
    cooperative_ratio: float = 0.5
    cooperative_rate: float = 0.8
    aggression: float = 0.5
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42)
    )
    
    def create(
        self, 
        agent_id: str, 
        position: tuple[float, float] = (0.0, 0.0)
    ) -> Agent:
        """Create an agent, type chosen by probability."""
        if self._rng.random() < self.cooperative_ratio:
            return CooperativeAgent(
                agent_id=agent_id,
                position=position,
                cooperation_rate=self.cooperative_rate
            )
        else:
            return CompetitiveAgent(
                agent_id=agent_id,
                position=position,
                aggression=self.aggression
            )


class AgentPopulation:
    """
    Manager for a population of agents.
    
    Uses the Factory pattern to create agents. The factory can be
    swapped to change the types of agents created.
    """
    
    def __init__(self, factory: AgentFactory) -> None:
        """
        Initialise with an agent factory.
        
        Args:
            factory: Factory for creating agents
        """
        self._factory = factory
        self._agents: list[Agent] = []
        self._next_id = 0
    
    def populate(
        self, 
        n: int, 
        bounds: tuple[float, float, float, float]
    ) -> None:
        """
        Create n agents at random positions within bounds.
        
        Args:
            n: Number of agents to create
            bounds: (x_min, x_max, y_min, y_max)
        """
        x_min, x_max, y_min, y_max = bounds
        rng = np.random.default_rng()
        
        for _ in range(n):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            agent = self._factory.create(str(self._next_id), (x, y))
            self._agents.append(agent)
            self._next_id += 1
    
    @property
    def agents(self) -> list[Agent]:
        """Get all agents."""
        return self._agents.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DECORATOR PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ DECORATOR PATTERN                                                             ║
║                                                                               ║
║ Intent: Attach additional responsibilities to an object dynamically.          ║
║         Decorators provide a flexible alternative to subclassing.             ║
║                                                                               ║
║ When to Use:                                                                  ║
║ - You need to add behaviour to individual objects, not entire classes         ║
║ - Responsibilities can be combined in multiple ways                           ║
║ - Extension by subclassing is impractical                                     ║
║                                                                               ║
║ Research Examples:                                                            ║
║ - Adding logging to functions                                                 ║
║ - Caching expensive computations                                              ║
║ - Timing function execution                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that times function execution.
    
    Logs the execution time to the module logger.
    
    Example:
        >>> @timed
        ... def slow_function():
        ...     time.sleep(1)
        ...     return "done"
        >>> result = slow_function()  # Logs execution time
    """
    import time
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    
    return wrapper


def cached(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that caches function results.
    
    Uses a simple dictionary cache. For more sophisticated caching,
    consider functools.lru_cache.
    
    Note: Only works with hashable arguments.
    
    Example:
        >>> @cached
        ... def expensive_computation(n: int) -> int:
        ...     return sum(range(n))
        >>> expensive_computation(1000000)  # Computed
        >>> expensive_computation(1000000)  # Cached
    """
    cache: dict[tuple[Any, ...], T] = {}
    
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            logger.debug(f"Cache miss for {func.__name__}{args}")
        else:
            logger.debug(f"Cache hit for {func.__name__}{args}")
        return cache[key]
    
    return wrapper


def validated(func: Callable[..., T]) -> Callable[..., T]:
    """Validate common preconditions for small teaching examples.

    The unit tests expect ``@validated`` to work as a zero-argument decorator.
    The implementation performs two checks that are broadly useful in numerical
    code examples:

    1. If the function has a parameter named ``b`` and it is zero, raise
       ``ZeroDivisionError``.
    2. Reject ``None`` values for any positional arguments.

    The decorator remains intentionally lightweight: it is designed to
    demonstrate the *pattern* rather than to provide a production validation
    framework.
    """
    import inspect

    sig = inspect.signature(func)
    param_names = list(sig.parameters)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        if any(a is None for a in args):
            raise ValueError("Arguments must not be None")

        bound = sig.bind_partial(*args, **kwargs)
        if "b" in param_names and "b" in bound.arguments:
            b_val = bound.arguments["b"]
            if isinstance(b_val, (int, float)) and b_val == 0:
                raise ZeroDivisionError("Division by zero")
        return func(*args, **kwargs)

    return wrapper


def retry(
    max_attempts: int = 3, 
    delay: float = 1.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator factory that retries a function on failure.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Delay between attempts in seconds
        
    Returns:
        Decorator function
        
    Example:
        >>> @retry(max_attempts=3, delay=0.5)
        ... def flaky_network_call():
        ...     # May fail intermittently
        ...     ...
    """
    import time
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise RuntimeError(
                f"{func.__name__} failed after {max_attempts} attempts"
            ) from last_exception
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: COMMAND PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ COMMAND PATTERN                                                               ║
║                                                                               ║
║ Intent: Encapsulate a request as an object, thereby letting you parameterise  ║
║         clients with different requests, queue requests, and support undo.    ║
║                                                                               ║
║ When to Use:                                                                  ║
║ - You need to parameterise objects with operations                            ║
║ - You need to queue, log, or undo operations                                  ║
║ - You need to support transactions                                            ║
║                                                                               ║
║ Research Examples:                                                            ║
║ - Experiment command pipelines                                                ║
║ - Undo/redo in interactive analysis                                           ║
║ - Batch job scheduling                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


class Command(Protocol):
    """Protocol for command objects."""
    
    def execute(self) -> Any:
        """Execute the command."""
        ...
    
    def undo(self) -> None:
        """Undo the command (if supported)."""
        ...


@dataclass
class DataTransformCommand:
    """
    Command that transforms data in a DataFrame-like object.
    
    Supports undo by storing the previous state.
    """
    data: dict[str, list[float]]
    column: str
    transform: Callable[[float], float]
    _previous: list[float] | None = field(default=None, init=False)
    
    def execute(self) -> dict[str, list[float]]:
        """Apply the transformation to the column."""
        if self.column not in self.data:
            raise KeyError(f"Column '{self.column}' not found")
        
        # Store previous state for undo
        self._previous = self.data[self.column].copy()
        
        # Apply transformation
        self.data[self.column] = [
            self.transform(v) for v in self.data[self.column]
        ]
        
        logger.info(f"Transformed column '{self.column}'")
        return self.data
    
    def undo(self) -> None:
        """Restore the previous state."""
        if self._previous is not None:
            self.data[self.column] = self._previous
            self._previous = None
            logger.info(f"Undid transformation on column '{self.column}'")


@dataclass
class CompositeCommand:
    """
    Command that executes multiple commands in sequence.
    
    Supports undo by reversing the sequence.
    """
    commands: list[Command] = field(default_factory=list)
    
    def add(self, command: Command) -> None:
        """Add a command to the sequence."""
        self.commands.append(command)
    
    def execute(self) -> None:
        """Execute all commands in order."""
        for cmd in self.commands:
            cmd.execute()
    
    def undo(self) -> None:
        """Undo all commands in reverse order."""
        for cmd in reversed(self.commands):
            cmd.undo()


class CommandHistory:
    """
    Maintains a history of executed commands for undo/redo.
    
    Example:
        >>> history = CommandHistory()
        >>> history.execute(some_command)
        >>> history.undo()  # Undoes some_command
        >>> history.redo()  # Re-executes some_command
    """
    
    def __init__(self) -> None:
        """Initialise empty history."""
        self._done: list[Command] = []
        self._undone: list[Command] = []
    
    def execute(self, command: Command) -> Any:
        """
        Execute a command and add to history.
        
        Clears the redo stack.
        """
        result = command.execute()
        self._done.append(command)
        self._undone.clear()
        return result
    
    def undo(self) -> None:
        """Undo the most recent command."""
        if self._done:
            command = self._done.pop()
            command.undo()
            self._undone.append(command)
        else:
            logger.warning("Nothing to undo")
    
    def redo(self) -> None:
        """Redo the most recently undone command."""
        if self._undone:
            command = self._undone.pop()
            command.execute()
            self._done.append(command)
        else:
            logger.warning("Nothing to redo")
    
    def can_undo(self) -> bool:
        """Return True when at least one command can be undone."""
        return bool(self._done)

    def can_redo(self) -> bool:
        """Return True when at least one undone command can be re-executed."""
        return bool(self._undone)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_strategy() -> None:
    """Demonstrate the Strategy pattern."""
    logger.info("=" * 60)
    logger.info("DEMO: Strategy Pattern — Numerical Integration")
    logger.info("=" * 60)
    
    # Function to integrate: f(x) = x²
    # Exact integral from 0 to 1 = 1/3
    def f(x: float) -> float:
        return x ** 2
    exact = 1 / 3
    
    # Try different strategies
    strategies: list[tuple[str, IntegrationStrategy]] = [
        ("Rectangle Rule", RectangleRule(n_intervals=100)),
        ("Trapezoid Rule", TrapezoidRule(n_intervals=100)),
        ("Simpson's Rule", SimpsonRule(n_intervals=100)),
    ]
    
    integrator = NumericalIntegrator(strategies[0][1])
    
    for name, strategy in strategies:
        integrator.strategy = strategy
        result = integrator.compute(f, 0, 1)
        error = abs(result - exact)
        logger.info(f"{name}: {result:.8f} (error: {error:.2e})")


def demo_observer() -> None:
    """Demonstrate the Observer pattern."""
    logger.info("=" * 60)
    logger.info("DEMO: Observer Pattern — Simulation Monitoring")
    logger.info("=" * 60)
    
    # Create subject
    subject: Observable[SimulationState] = Observable[SimulationState]()
    
    # Create and subscribe observers
    console = ConsoleLogger(prefix="[SIM]")
    metrics = MetricsCollector()
    
    subject.subscribe(console.update)
    def record_temperature(state: SimulationState) -> None:
        temp = state.values.get('temperature')
        if isinstance(temp, (int, float)):
            metrics.update(float(temp))

    subject.subscribe(record_temperature)
    
    # Simulate some updates
    for step in range(5):
        state = SimulationState(
            step=step,
            time=step * 0.1,
            values={'temperature': 20 + step, 'pressure': 101.3 + step * 0.5}
        )
        subject.notify(state)
    
    # Show collected metrics
    values = metrics.get_timeseries()
    logger.info(f"Collected values: {values}")


def demo_factory() -> None:
    """Demonstrate the Factory pattern."""
    logger.info("=" * 60)
    logger.info("DEMO: Factory Pattern — Agent Creation")
    logger.info("=" * 60)
    
    # Create populations with different factories
    factories: list[tuple[str, AgentFactory]] = [
        ("Cooperative", CooperativeFactory(cooperation_rate=0.9)),
        ("Competitive", CompetitiveFactory(aggression=0.7)),
        ("Mixed (50/50)", MixedFactory(cooperative_ratio=0.5)),
    ]
    
    for name, factory in factories:
        population = AgentPopulation(factory)
        population.populate(5, bounds=(0, 100, 0, 100))
        
        logger.info(f"\n{name} Population:")
        for agent in population.agents:
            logger.info(f"  {agent.act()}")


def demo_decorator() -> None:
    """Demonstrate the Decorator pattern."""
    logger.info("=" * 60)
    logger.info("DEMO: Decorator Pattern — Function Enhancement")
    logger.info("=" * 60)
    
    @timed
    @cached
    def fibonacci(n: int) -> int:
        """Compute the nth Fibonacci number."""
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    # First call computes
    logger.info(f"fib(30) = {fibonacci(30)}")
    
    # Second call uses cache
    logger.info(f"fib(30) again = {fibonacci(30)}")
    
    # Validation example
    @validated
    def safe_sqrt(x: float) -> float:
        if x <= 0:
            raise ValueError("Input must be positive")
        return x ** 0.5
    
    logger.info(f"sqrt(16) = {safe_sqrt(16)}")
    try:
        safe_sqrt(-1)
    except ValueError as e:
        logger.info(f"Expected error: {e}")


def demo_command() -> None:
    """Demonstrate the Command pattern."""
    logger.info("=" * 60)
    logger.info("DEMO: Command Pattern — Data Transformations")
    logger.info("=" * 60)
    
    # Sample data
    data = {
        'values': [1.0, 2.0, 3.0, 4.0, 5.0],
        'squared': [1.0, 4.0, 9.0, 16.0, 25.0]
    }
    
    logger.info(f"Original: {data['values']}")
    
    # Create command history
    history = CommandHistory()
    
    # Execute transformation
    cmd = DataTransformCommand(data, 'values', lambda x: x * 2)
    history.execute(cmd)
    logger.info(f"After *2: {data['values']}")
    
    # Undo
    history.undo()
    logger.info(f"After undo: {data['values']}")
    
    # Redo
    history.redo()
    logger.info(f"After redo: {data['values']}")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_strategy()
    print()
    demo_observer()
    print()
    demo_factory()
    print()
    demo_decorator()
    print()
    demo_command()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="02UNIT Lab 2: Design Patterns Catalogue"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstrations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--pattern", 
        choices=['strategy', 'observer', 'factory', 'decorator', 'command'],
        help="Demo a specific pattern"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.pattern:
        demo_funcs = {
            'strategy': demo_strategy,
            'observer': demo_observer,
            'factory': demo_factory,
            'decorator': demo_decorator,
            'command': demo_command,
        }
        demo_funcs[args.pattern]()
    elif args.demo:
        run_all_demos()
    else:
        print("\n" + "═" * 60)
        print("  WEEK 2 LAB 2: DESIGN PATTERNS CATALOGUE")
        print("═" * 60 + "\n")
        print("Patterns covered:")
        print("  1. Strategy — Interchangeable algorithms")
        print("  2. Observer — Event-driven updates")
        print("  3. Factory — Flexible object creation")
        print("  4. Decorator — Dynamic behaviour extension")
        print("  5. Command — Encapsulated operations with undo")
        print("\nUse --demo to run all demonstrations")
        print("Use --pattern <name> to demo a specific pattern")
        print("Use -v for verbose output")
        print("═" * 60)


if __name__ == "__main__":
    main()
