#!/usr/bin/env python3
"""Solution for Exercise 07: Circuit Breaker Pattern.

This module provides reference implementations for the circuit
breaker resilience pattern.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    
    def __init__(self, message: str, time_until_reset: float = 0.0) -> None:
        super().__init__(message)
        self.time_until_reset = time_until_reset


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behaviour."""
    
    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout: float = 30.0
    half_open_max_calls: int = 3


@dataclass
class CircuitStats:
    """Statistics for circuit breaker operation."""
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: list[tuple[float, CircuitState, CircuitState]] = field(
        default_factory=list
    )


class CircuitBreaker:
    """Circuit breaker for fault-tolerant service calls."""
    
    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()
        self._stats = CircuitStats()
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def stats(self) -> CircuitStats:
        """Circuit breaker statistics."""
        with self._lock:
            return CircuitStats(
                total_calls=self._stats.total_calls,
                successful_calls=self._stats.successful_calls,
                failed_calls=self._stats.failed_calls,
                rejected_calls=self._stats.rejected_calls,
                state_transitions=list(self._stats.state_transitions),
            )
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._stats.state_transitions.append(
            (time.time(), old_state, new_state)
        )
        logger.info(
            "Circuit breaker: %s -> %s",
            old_state.name,
            new_state.name,
        )
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        
        self._state = new_state
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self._config.reset_timeout
    
    def _handle_success(self) -> None:
        """Handle successful call."""
        self._success_count += 1
        self._stats.successful_calls += 1
        self._failure_count = 0
        
        if (
            self._state == CircuitState.HALF_OPEN
            and self._success_count >= self._config.success_threshold
        ):
            self._transition_to(CircuitState.CLOSED)
    
    def _handle_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._stats.failed_calls += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN)
        elif (
            self._state == CircuitState.CLOSED
            and self._failure_count >= self._config.failure_threshold
        ):
            self._transition_to(CircuitState.OPEN)
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker."""
        with self._lock:
            self._stats.total_calls += 1
            
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    self._stats.rejected_calls += 1
                    time_remaining = (
                        self._config.reset_timeout
                        - (time.time() - self._last_failure_time)
                    )
                    raise CircuitOpenError(
                        "Circuit is open",
                        time_until_reset=max(0, time_remaining),
                    )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._config.half_open_max_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitOpenError("Circuit in half-open, max calls reached")
                self._half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._handle_success()
            return result
        except Exception:
            with self._lock:
                self._handle_failure()
            raise
    
    def reset(self) -> None:
        """Manually reset circuit to CLOSED state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
    
    def get_time_until_reset(self) -> float:
        """Get time remaining until automatic reset attempt."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                return 0.0
            elapsed = time.time() - self._last_failure_time
            return max(0, self._config.reset_timeout - elapsed)


def circuit_protected(
    breaker: CircuitBreaker,
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that protects a function with a circuit breaker."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return breaker.call(func, *args, **kwargs)
            except CircuitOpenError:
                if fallback is not None:
                    return fallback(*args, **kwargs)
                raise
        
        return wrapper
    return decorator


def main() -> None:
    """Verify solution implementations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing CircuitBreaker...")
    
    # Test initial state
    config = CircuitBreakerConfig(failure_threshold=3, reset_timeout=0.1)
    breaker = CircuitBreaker(config)
    assert breaker.state == CircuitState.CLOSED
    print("  ✓ Initial state is CLOSED")
    
    # Test successful calls
    breaker.call(lambda: "success")
    assert breaker.state == CircuitState.CLOSED
    assert breaker.stats.successful_calls == 1
    print("  ✓ Successful calls tracked")
    
    # Test failure threshold
    def failing_func() -> None:
        raise ConnectionError("service down")
    
    for _ in range(3):
        try:
            breaker.call(failing_func)
        except ConnectionError:
            pass
    
    assert breaker.state == CircuitState.OPEN
    assert breaker.stats.failed_calls == 3
    print("  ✓ Circuit opens after failures")
    
    # Test rejection in OPEN state
    try:
        breaker.call(lambda: "should not execute")
        assert False
    except CircuitOpenError:
        assert breaker.stats.rejected_calls == 1
    print("  ✓ Calls rejected when OPEN")
    
    # Test automatic reset attempt
    time.sleep(0.15)
    
    try:
        breaker.call(failing_func)
    except ConnectionError:
        pass
    
    assert breaker.state == CircuitState.OPEN
    print("  ✓ HALF_OPEN transitions back to OPEN on failure")
    
    # Test successful recovery
    time.sleep(0.15)
    config2 = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        reset_timeout=0.1,
    )
    breaker2 = CircuitBreaker(config2)
    
    for _ in range(3):
        try:
            breaker2.call(failing_func)
        except ConnectionError:
            pass
    assert breaker2.state == CircuitState.OPEN
    
    time.sleep(0.15)
    
    breaker2.call(lambda: "success")
    assert breaker2.state == CircuitState.HALF_OPEN
    
    breaker2.call(lambda: "success")
    assert breaker2.state == CircuitState.CLOSED
    print("  ✓ Circuit closes after successes in HALF_OPEN")
    
    # Test manual reset
    breaker.reset()
    assert breaker.state == CircuitState.CLOSED
    print("  ✓ Manual reset works")
    
    # Test decorator
    print("\nTesting circuit_protected decorator...")
    breaker3 = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
    
    @circuit_protected(breaker3, fallback=lambda: "fallback_value")
    def protected_func() -> str:
        raise ConnectionError("down")
    
    protected_func()  # Failure 1
    protected_func()  # Failure 2, circuit opens
    result = protected_func()  # Should use fallback
    assert result == "fallback_value"
    print("  ✓ Decorator with fallback works")
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
