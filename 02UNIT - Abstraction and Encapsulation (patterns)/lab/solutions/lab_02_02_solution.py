#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT, Lab 02 SOLUTIONS: Design Patterns — Extended Implementations
═══════════════════════════════════════════════════════════════════════════════

This file contains extended reference implementations for the design patterns
covered in lab_02_02_design_patterns.py, with additional research-oriented
applications and advanced techniques.

SOLUTIONS INCLUDED
──────────────────
1. StrategyChain — Composable strategy pipelines
2. AsyncObserver — Observer pattern with async support
3. AbstractFactory — Family of related factories
4. ContextDecorator — Decorators with setup/teardown
5. MacroCommand — Composite commands for complex operations

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Type variables
T = TypeVar('T')
U = TypeVar('U')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 1: COMPOSABLE STRATEGY PIPELINES
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessingStrategy(Protocol[InputT, OutputT]):
    """Protocol for data processing strategies."""
    
    def process(self, data: InputT) -> OutputT:
        """Process input data and return result."""
        ...
    
    @property
    def name(self) -> str:
        """Strategy identifier."""
        ...


@dataclass
class StrategyChain(Generic[T]):
    """
    Composable chain of processing strategies.
    
    Enables building complex data pipelines from simple, reusable
    transformation steps. Each strategy in the chain transforms
    the data, passing its output to the next strategy.
    
    Supports:
    - Sequential composition: chain.add(strategy)
    - Branching: chain.branch(condition, if_strategy, else_strategy)
    - Parallel execution: chain.parallel([strategy1, strategy2])
    
    Example:
        chain = StrategyChain[NDArray]()
        chain.add(NormaliseStrategy())
        chain.add(FilterOutliersStrategy(threshold=3.0))
        chain.add(SmoothingStrategy(window=5))
        result = chain.process(raw_data)
    """
    
    _strategies: list[ProcessingStrategy[Any, Any]] = field(default_factory=list)
    _execution_times: list[tuple[str, float]] = field(default_factory=list)
    
    def add(self, strategy: ProcessingStrategy[Any, Any]) -> 'StrategyChain[T]':
        """Add a strategy to the chain (fluent interface)."""
        self._strategies.append(strategy)
        return self
    
    def process(self, data: T) -> T:
        """
        Execute all strategies in sequence.
        
        Args:
            data: Input data to process.
        
        Returns:
            Transformed data after all strategies applied.
        """
        self._execution_times.clear()
        result: Any = data
        
        for strategy in self._strategies:
            start = time.perf_counter()
            result = strategy.process(result)
            elapsed = time.perf_counter() - start
            self._execution_times.append((strategy.name, elapsed))
            logger.debug(f"{strategy.name}: {elapsed:.4f}s")
        
        return result
    
    @property
    def execution_report(self) -> str:
        """Generate timing report for last execution."""
        if not self._execution_times:
            return "No execution recorded"
        
        total = sum(t for _, t in self._execution_times)
        lines = [f"{'Strategy':<30} {'Time':>10} {'%':>8}"]
        lines.append("-" * 50)
        
        for name, elapsed in self._execution_times:
            pct = 100 * elapsed / total if total > 0 else 0
            lines.append(f"{name:<30} {elapsed:>10.4f} {pct:>7.1f}%")
        
        lines.append("-" * 50)
        lines.append(f"{'TOTAL':<30} {total:>10.4f}")
        
        return "\n".join(lines)


# Concrete strategies for numerical data

@dataclass
class NormaliseStrategy:
    """Z-score normalisation strategy."""
    
    @property
    def name(self) -> str:
        return "Normalise (Z-score)"
    
    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalise data to zero mean and unit variance."""
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            return (data - mean) / std
        return data - mean


@dataclass
class FilterOutliersStrategy:
    """Remove outliers beyond threshold standard deviations."""
    
    threshold: float = 3.0
    
    @property
    def name(self) -> str:
        return f"Filter Outliers (>{self.threshold}σ)"
    
    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Replace outliers with NaN."""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        outlier_mask = np.abs(data - mean) > self.threshold * std
        result = data.copy()
        result[outlier_mask] = np.nan
        return result


@dataclass
class MovingAverageStrategy:
    """Apply moving average smoothing."""
    
    window: int = 5
    
    @property
    def name(self) -> str:
        return f"Moving Average (w={self.window})"
    
    def process(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Smooth data with moving average."""
        kernel = np.ones(self.window) / self.window
        return np.convolve(data, kernel, mode='same')


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 2: ASYNC OBSERVER PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

@runtime_checkable
class AsyncObserver(Protocol[T]):
    """Protocol for asynchronous observers."""
    
    async def on_update(self, data: T) -> None:
        """Handle update notification asynchronously."""
        ...


class AsyncSubject(Generic[T]):
    """
    Subject that notifies observers asynchronously.
    
    Useful for:
    - Non-blocking UI updates during long computations
    - Distributed systems with network observers
    - Concurrent data processing pipelines
    
    Example:
        subject = AsyncSubject[SimulationState]()
        subject.subscribe(AsyncPlotter())
        subject.subscribe(AsyncLogger())
        
        async def simulation():
            for state in simulate():
                await subject.notify(state)
    """
    
    def __init__(self) -> None:
        self._observers: list[AsyncObserver[T]] = []
        self._notification_count = 0
    
    def subscribe(self, observer: AsyncObserver[T]) -> None:
        """Add an observer."""
        self._observers.append(observer)
        logger.debug(f"Observer subscribed, total: {len(self._observers)}")
    
    def unsubscribe(self, observer: AsyncObserver[T]) -> None:
        """Remove an observer."""
        self._observers.remove(observer)
    
    async def notify(self, data: T) -> None:
        """
        Notify all observers concurrently.
        
        Uses asyncio.gather for parallel notification.
        """
        self._notification_count += 1
        tasks = [observer.on_update(data) for observer in self._observers]
        await asyncio.gather(*tasks)
    
    async def notify_sequential(self, data: T) -> None:
        """Notify observers one at a time (for ordered processing)."""
        self._notification_count += 1
        for observer in self._observers:
            await observer.on_update(data)
    
    @property
    def observer_count(self) -> int:
        """Number of subscribed observers."""
        return len(self._observers)


# Concrete async observers

class AsyncLogObserver:
    """Observer that logs updates asynchronously."""
    
    def __init__(self, name: str = "AsyncLogger"):
        self.name = name
        self.update_count = 0
    
    async def on_update(self, data: Any) -> None:
        """Log the update (simulates async I/O)."""
        await asyncio.sleep(0.01)  # Simulate async write
        self.update_count += 1
        logger.info(f"[{self.name}] Update #{self.update_count}: {data}")


class AsyncBufferObserver(Generic[T]):
    """Observer that buffers updates for batch processing."""
    
    def __init__(self, buffer_size: int = 10):
        self.buffer: list[T] = []
        self.buffer_size = buffer_size
        self.flush_count = 0
    
    async def on_update(self, data: T) -> None:
        """Buffer update, flush when full."""
        self.buffer.append(data)
        if len(self.buffer) >= self.buffer_size:
            await self._flush()
    
    async def _flush(self) -> None:
        """Process buffered data."""
        self.flush_count += 1
        logger.info(f"Flushing {len(self.buffer)} items (batch #{self.flush_count})")
        await asyncio.sleep(0.05)  # Simulate batch processing
        self.buffer.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 3: ABSTRACT FACTORY PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentFactory(Protocol):
    """
    Abstract factory for creating experiment components.
    
    An abstract factory produces families of related objects without
    specifying concrete classes. This is useful when an experiment
    requires consistent components that must work together.
    """
    
    def create_model(self) -> 'Model':
        """Create the mathematical model."""
        ...
    
    def create_solver(self) -> 'Solver':
        """Create the numerical solver."""
        ...
    
    def create_visualiser(self) -> 'Visualiser':
        """Create the visualisation component."""
        ...


class Model(Protocol):
    """Protocol for mathematical models."""
    
    def evaluate(self, t: float, state: NDArray) -> NDArray:
        """Evaluate model at time t with given state."""
        ...


class Solver(Protocol):
    """Protocol for numerical solvers."""
    
    def solve(self, model: Model, initial: NDArray, t_span: tuple[float, float]) -> NDArray:
        """Solve the model over time span."""
        ...


class Visualiser(Protocol):
    """Protocol for visualisation."""
    
    def plot(self, data: NDArray) -> None:
        """Create visualisation of data."""
        ...


# Concrete factory: High-fidelity simulation

@dataclass
class HighFidelityModel:
    """Detailed model with many parameters."""
    
    def evaluate(self, t: float, state: NDArray) -> NDArray:
        # Complex dynamics
        return -0.1 * state + 0.01 * np.sin(t) * state
    
    @property
    def name(self) -> str:
        return "HighFidelityModel"


@dataclass
class AdaptiveSolver:
    """Adaptive step solver for high accuracy."""
    
    def solve(self, model: Model, initial: NDArray, t_span: tuple[float, float]) -> NDArray:
        # Simplified adaptive RK45
        t, t_end = t_span
        state = initial.copy()
        trajectory = [state.copy()]
        dt = 0.01
        
        while t < t_end:
            k1 = model.evaluate(t, state)
            k2 = model.evaluate(t + 0.5*dt, state + 0.5*dt*k1)
            k3 = model.evaluate(t + 0.5*dt, state + 0.5*dt*k2)
            k4 = model.evaluate(t + dt, state + dt*k3)
            
            state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            t += dt
            trajectory.append(state.copy())
        
        return np.array(trajectory)


@dataclass
class PublicationVisualiser:
    """Publication-quality visualisation."""
    
    def plot(self, data: NDArray) -> None:
        logger.info(f"Creating publication-quality plot: {data.shape}")
        # Would use matplotlib with publication settings


class HighFidelityFactory:
    """Factory for high-fidelity experiments."""
    
    def create_model(self) -> HighFidelityModel:
        return HighFidelityModel()
    
    def create_solver(self) -> AdaptiveSolver:
        return AdaptiveSolver()
    
    def create_visualiser(self) -> PublicationVisualiser:
        return PublicationVisualiser()


# Concrete factory: Quick prototyping

@dataclass
class SimplifiedModel:
    """Simplified model for quick testing."""
    
    def evaluate(self, t: float, state: NDArray) -> NDArray:
        return -0.1 * state
    
    @property
    def name(self) -> str:
        return "SimplifiedModel"


@dataclass
class EulerSolver:
    """Simple Euler solver for fast iteration."""
    
    def solve(self, model: Model, initial: NDArray, t_span: tuple[float, float]) -> NDArray:
        t, t_end = t_span
        state = initial.copy()
        trajectory = [state.copy()]
        dt = 0.1  # Larger steps
        
        while t < t_end:
            state = state + dt * model.evaluate(t, state)
            t += dt
            trajectory.append(state.copy())
        
        return np.array(trajectory)


@dataclass 
class QuickVisualiser:
    """Fast visualisation for exploration."""
    
    def plot(self, data: NDArray) -> None:
        logger.info(f"Quick plot: {data.shape}")


class PrototypingFactory:
    """Factory for quick prototyping experiments."""
    
    def create_model(self) -> SimplifiedModel:
        return SimplifiedModel()
    
    def create_solver(self) -> EulerSolver:
        return EulerSolver()
    
    def create_visualiser(self) -> QuickVisualiser:
        return QuickVisualiser()


def run_experiment(factory: ExperimentFactory) -> None:
    """Run experiment using provided factory."""
    model = factory.create_model()
    solver = factory.create_solver()
    visualiser = factory.create_visualiser()
    
    initial = np.array([1.0, 0.5, 0.2])
    result = solver.solve(model, initial, (0, 10))
    visualiser.plot(result)
    
    logger.info(f"Experiment complete: {len(result)} steps")


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 4: CONTEXT DECORATOR PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

class ContextDecorator:
    """
    Decorator that manages setup and teardown around function execution.
    
    Combines the decorator pattern with context management for:
    - Resource acquisition and release
    - Performance measurement
    - Exception handling and logging
    - State setup and restoration
    
    Example:
        @ContextDecorator.timing()
        @ContextDecorator.logging(level=logging.DEBUG)
        def expensive_computation(data):
            ...
    """
    
    @classmethod
    def timing(cls, name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that measures and logs execution time."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                label = name or func.__name__
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    logger.info(f"[TIMING] {label}: {elapsed:.4f}s")
            return wrapper
        return decorator
    
    @classmethod
    def retry(
        cls,
        max_attempts: int = 3,
        delay: float = 1.0,
        exceptions: tuple[type[Exception], ...] = (Exception,)
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that retries on failure."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                last_exception: Exception | None = None
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}"
                        )
                        if attempt < max_attempts - 1:
                            time.sleep(delay)
                
                raise last_exception or RuntimeError("All attempts failed")
            return wrapper
        return decorator
    
    @classmethod
    def cache_result(
        cls,
        maxsize: int = 128
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that caches results based on arguments."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            cache: dict[tuple[Any, ...], T] = {}
            hits = 0
            misses = 0
            
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                nonlocal hits, misses
                
                # Create cache key from arguments
                key = (args, tuple(sorted(kwargs.items())))
                
                if key in cache:
                    hits += 1
                    return cache[key]
                
                misses += 1
                result = func(*args, **kwargs)
                
                # Enforce size limit (simple LRU-like)
                if len(cache) >= maxsize:
                    # Remove oldest entry
                    oldest = next(iter(cache))
                    del cache[oldest]
                
                cache[key] = result
                return result
            
            # Attach cache info to wrapper
            wrapper.cache_info = lambda: {'hits': hits, 'misses': misses, 'size': len(cache)}  # type: ignore
            wrapper.cache_clear = lambda: cache.clear()  # type: ignore
            
            return wrapper
        return decorator
    
    @classmethod
    def validate_args(
        cls,
        **validators: Callable[[Any], bool]
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator that validates arguments before execution."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                # Get argument names
                import inspect
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                
                # Validate specified arguments
                for arg_name, validator in validators.items():
                    if arg_name in bound.arguments:
                        value = bound.arguments[arg_name]
                        if not validator(value):
                            raise ValueError(
                                f"Validation failed for '{arg_name}': {value}"
                            )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION 5: MACRO COMMAND PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

class Command(Protocol):
    """Protocol for undoable commands."""
    
    def execute(self) -> None:
        """Execute the command."""
        ...
    
    def undo(self) -> None:
        """Undo the command."""
        ...
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        ...


@dataclass
class MacroCommand:
    """
    Composite command that groups multiple commands.
    
    Executes sub-commands in sequence and undoes them in reverse order.
    Useful for complex operations that must be atomic (all-or-nothing).
    
    Example:
        macro = MacroCommand("Prepare Dataset")
        macro.add(LoadDataCommand(path))
        macro.add(CleanDataCommand())
        macro.add(NormaliseDataCommand())
        macro.add(SplitDataCommand(0.8))
        
        try:
            macro.execute()
        except Exception:
            macro.undo()  # Rolls back all steps
    """
    
    name: str
    _commands: list[Command] = field(default_factory=list)
    _executed: list[Command] = field(default_factory=list)
    
    def add(self, command: Command) -> 'MacroCommand':
        """Add a command to the macro (fluent interface)."""
        self._commands.append(command)
        return self
    
    @property
    def description(self) -> str:
        """Description including sub-commands."""
        sub = ", ".join(c.description for c in self._commands)
        return f"{self.name} [{sub}]"
    
    def execute(self) -> None:
        """Execute all commands in order."""
        self._executed.clear()
        
        for cmd in self._commands:
            try:
                cmd.execute()
                self._executed.append(cmd)
            except Exception as e:
                logger.error(f"Command failed: {cmd.description}")
                # Roll back executed commands
                self.undo()
                raise
    
    def undo(self) -> None:
        """Undo executed commands in reverse order."""
        while self._executed:
            cmd = self._executed.pop()
            try:
                cmd.undo()
            except Exception as e:
                logger.error(f"Undo failed for: {cmd.description}")


# Concrete commands for data processing

@dataclass
class SetValueCommand:
    """Command to set a value in a dictionary."""
    
    data: dict[str, Any]
    key: str
    new_value: Any
    _old_value: Any = field(default=None, init=False)
    
    @property
    def description(self) -> str:
        return f"Set {self.key}"
    
    def execute(self) -> None:
        self._old_value = self.data.get(self.key)
        self.data[self.key] = self.new_value
    
    def undo(self) -> None:
        if self._old_value is None:
            del self.data[self.key]
        else:
            self.data[self.key] = self._old_value


@dataclass
class TransformArrayCommand:
    """Command to transform an array in place."""
    
    data: dict[str, NDArray]
    key: str
    transform: Callable[[NDArray], NDArray]
    _backup: NDArray | None = field(default=None, init=False)
    
    @property
    def description(self) -> str:
        return f"Transform {self.key}"
    
    def execute(self) -> None:
        self._backup = self.data[self.key].copy()
        self.data[self.key] = self.transform(self.data[self.key])
    
    def undo(self) -> None:
        if self._backup is not None:
            self.data[self.key] = self._backup


@dataclass
class AppendToListCommand:
    """Command to append to a list."""
    
    data: dict[str, list]
    key: str
    value: Any
    
    @property
    def description(self) -> str:
        return f"Append to {self.key}"
    
    def execute(self) -> None:
        self.data[self.key].append(self.value)
    
    def undo(self) -> None:
        self.data[self.key].pop()


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_strategy_chain() -> None:
    """Demonstrate composable strategy pipeline."""
    logger.info("=== Strategy Chain Demo ===")
    
    # Create sample data with noise and outliers
    rng = np.random.default_rng(42)
    data = np.sin(np.linspace(0, 4*np.pi, 100)) + rng.normal(0, 0.3, 100)
    data[50] = 10.0  # Add outlier
    
    # Build processing pipeline
    chain = StrategyChain[NDArray]()
    chain.add(FilterOutliersStrategy(threshold=2.5))
    chain.add(NormaliseStrategy())
    chain.add(MovingAverageStrategy(window=5))
    
    # Process
    result = chain.process(data)
    
    logger.info(f"Input range: [{data.min():.2f}, {data.max():.2f}]")
    logger.info(f"Output range: [{np.nanmin(result):.2f}, {np.nanmax(result):.2f}]")
    logger.info("\nExecution Report:")
    logger.info(chain.execution_report)


async def demo_async_observer() -> None:
    """Demonstrate async observer pattern."""
    logger.info("=== Async Observer Demo ===")
    
    subject: AsyncSubject[dict[str, float]] = AsyncSubject()
    
    # Subscribe observers
    log_observer = AsyncLogObserver("SimulationLog")
    buffer_observer: AsyncBufferObserver[dict[str, float]] = AsyncBufferObserver(buffer_size=3)
    
    subject.subscribe(log_observer)
    subject.subscribe(buffer_observer)
    
    # Simulate updates
    for i in range(5):
        state = {"time": i * 0.1, "value": np.sin(i * 0.5)}
        await subject.notify(state)
        await asyncio.sleep(0.1)
    
    logger.info(f"Total updates: log={log_observer.update_count}, flushes={buffer_observer.flush_count}")


def demo_abstract_factory() -> None:
    """Demonstrate abstract factory pattern."""
    logger.info("=== Abstract Factory Demo ===")
    
    logger.info("\nRunning HIGH-FIDELITY experiment:")
    run_experiment(HighFidelityFactory())
    
    logger.info("\nRunning PROTOTYPING experiment:")
    run_experiment(PrototypingFactory())


def demo_context_decorator() -> None:
    """Demonstrate context decorator pattern."""
    logger.info("=== Context Decorator Demo ===")
    
    @ContextDecorator.timing("fibonacci")
    @ContextDecorator.cache_result(maxsize=100)
    def fibonacci(n: int) -> int:
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)
    
    # First call - computes
    result1 = fibonacci(30)
    logger.info(f"fib(30) = {result1}")
    
    # Second call - cached
    result2 = fibonacci(30)
    logger.info(f"fib(30) cached = {result2}")
    
    # Cache stats
    logger.info(f"Cache info: {fibonacci.cache_info()}")  # type: ignore
    
    # Validation decorator
    @ContextDecorator.validate_args(n=lambda x: x > 0, data=lambda x: len(x) > 0)
    def process_data(n: int, data: list) -> int:
        return n * len(data)
    
    try:
        process_data(0, [1, 2, 3])  # Should fail validation
    except ValueError as e:
        logger.info(f"Validation caught: {e}")


def demo_macro_command() -> None:
    """Demonstrate macro command pattern."""
    logger.info("=== Macro Command Demo ===")
    
    # Data to modify
    data: dict[str, Any] = {
        "values": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "metadata": {"processed": False},
        "log": []
    }
    
    logger.info(f"Initial: {data['values']}")
    
    # Create macro
    macro = MacroCommand("Process Dataset")
    macro.add(TransformArrayCommand(data, "values", lambda x: x * 2))
    macro.add(SetValueCommand(data["metadata"], "processed", True))
    macro.add(AppendToListCommand(data, "log", "Processed at step 1"))
    
    # Execute
    macro.execute()
    logger.info(f"After execute: {data['values']}")
    logger.info(f"Metadata: {data['metadata']}")
    logger.info(f"Log: {data['log']}")
    
    # Undo
    macro.undo()
    logger.info(f"After undo: {data['values']}")
    logger.info(f"Metadata: {data['metadata']}")
    logger.info(f"Log: {data['log']}")


def run_all_demos() -> None:
    """Run all demonstration functions."""
    logger.info("\n" + "═" * 60)
    logger.info("  02UNIT LAB 02 SOLUTIONS - DEMONSTRATIONS")
    logger.info("═" * 60 + "\n")
    
    demo_strategy_chain()
    print()
    
    # Run async demo
    asyncio.run(demo_async_observer())
    print()
    
    demo_abstract_factory()
    print()
    
    demo_context_decorator()
    print()
    
    demo_macro_command()
    
    logger.info("\n" + "═" * 60)
    logger.info("  Demonstrations complete")
    logger.info("═" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="02UNIT Lab 02 Solutions: Design Patterns Extensions"
    )
    parser.add_argument("--demo", action="store_true", help="Run demonstrations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        run_all_demos()
    else:
        print("\n" + "═" * 60)
        print("  02UNIT LAB 02 SOLUTIONS")
        print("═" * 60 + "\n")
        print("Extended pattern implementations:")
        print("  1. StrategyChain — Composable pipelines")
        print("  2. AsyncSubject — Async observer pattern")
        print("  3. AbstractFactory — Component families")
        print("  4. ContextDecorator — Setup/teardown decorators")
        print("  5. MacroCommand — Composite commands")
        print("\nUse --demo to run demonstrations")
        print("Use -v for verbose output")
        print("═" * 60)


if __name__ == "__main__":
    main()
