#!/usr/bin/env python3
"""Exercise 07: Circuit Breaker Pattern (Hard).

This exercise implements the circuit breaker pattern for protecting
against cascading failures when calling external services.

Learning Objectives:
    - Implement state machine for circuit breaker
    - Handle state transitions correctly
    - Integrate with logging and metrics

Estimated Time: 20 minutes
Difficulty: Hard (★★★)

Instructions:
    1. Implement the CircuitBreaker class with all states
    2. Handle state transitions correctly
    3. Add proper thread safety considerations
    4. Run tests to verify your implementation
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    
    CLOSED = auto()     # Normal operation, calls pass through
    OPEN = auto()       # Failing fast, calls rejected
    HALF_OPEN = auto()  # Testing recovery with limited calls


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting calls.
    
    Attributes:
        time_until_reset: Seconds until circuit will attempt reset.
    """
    
    def __init__(self, message: str, time_until_reset: float = 0.0) -> None:
        super().__init__(message)
        self.time_until_reset = time_until_reset


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behaviour.
    
    Attributes:
        failure_threshold: Number of failures before opening.
        success_threshold: Successes in HALF_OPEN to close.
        reset_timeout: Seconds before attempting reset from OPEN.
        half_open_max_calls: Maximum calls allowed in HALF_OPEN.
    """
    
    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout: float = 30.0
    half_open_max_calls: int = 3


@dataclass
class CircuitStats:
    """Statistics for circuit breaker operation.
    
    Attributes:
        total_calls: Total number of call attempts.
        successful_calls: Number of successful calls.
        failed_calls: Number of failed calls.
        rejected_calls: Calls rejected while circuit open.
        state_transitions: List of (timestamp, from_state, to_state).
    """
    
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: list[tuple[float, CircuitState, CircuitState]] = field(
        default_factory=list
    )


class CircuitBreaker:
    """Circuit breaker for fault-tolerant service calls.
    
    Implements the circuit breaker pattern with three states:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Fast failure, all calls rejected
    - HALF_OPEN: Testing recovery with limited calls
    
    Example:
        >>> breaker = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        >>> def unreliable_service():
        ...     raise ConnectionError("service down")
        >>> for _ in range(5):
        ...     try:
        ...         breaker.call(unreliable_service)
        ...     except (ConnectionError, CircuitOpenError):
        ...         pass
        >>> breaker.state
        <CircuitState.OPEN: 2>
    """
    
    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialise circuit breaker.
        
        Args:
            config: Circuit breaker configuration.
        """
        # TODO: Initialise with config (use defaults if None)
        # TODO: Set initial state to CLOSED
        # TODO: Initialise counters: _failure_count, _success_count
        # TODO: Initialise _last_failure_time to 0.0
        # TODO: Initialise _half_open_calls to 0
        # TODO: Create _lock for thread safety (threading.Lock())
        # TODO: Create _stats instance
        pass
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        # TODO: Return current state
        pass
    
    @property
    def stats(self) -> CircuitStats:
        """Circuit breaker statistics."""
        # TODO: Return copy of stats
        pass
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state.
        
        Args:
            new_state: State to transition to.
        """
        # TODO: Record state transition in stats
        # TODO: Log the transition
        # TODO: Reset appropriate counters based on new state
        # TODO: Update _state
        pass
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN state.
        
        Returns:
            True if reset_timeout has elapsed since last failure.
        """
        # TODO: Calculate time since last failure
        # TODO: Return True if >= reset_timeout
        pass
    
    def _handle_success(self) -> None:
        """Handle successful call.
        
        Updates counters and potentially transitions state.
        """
        # TODO: Increment _success_count and stats.successful_calls
        # TODO: Reset _failure_count
        # TODO: If HALF_OPEN and enough successes, transition to CLOSED
        pass
    
    def _handle_failure(self) -> None:
        """Handle failed call.
        
        Updates counters and potentially transitions state.
        """
        # TODO: Increment _failure_count and stats.failed_calls
        # TODO: Update _last_failure_time
        # TODO: If HALF_OPEN, immediately transition to OPEN
        # TODO: If CLOSED and failures >= threshold, transition to OPEN
        pass
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.
            
        Returns:
            Result of function call.
            
        Raises:
            CircuitOpenError: If circuit is open.
            Exception: Any exception from the function.
        """
        # TODO: Use lock for thread safety
        # TODO: Increment total_calls
        
        # TODO: Handle OPEN state
        # - Check if should attempt reset
        # - If yes, transition to HALF_OPEN
        # - If no, increment rejected_calls and raise CircuitOpenError
        
        # TODO: Handle HALF_OPEN state
        # - Check if max calls exceeded
        # - If yes, raise CircuitOpenError
        # - Increment _half_open_calls
        
        # TODO: Try to execute function
        # - On success, call _handle_success
        # - On exception, call _handle_failure and re-raise
        pass
    
    def reset(self) -> None:
        """Manually reset circuit to CLOSED state."""
        # TODO: Acquire lock and transition to CLOSED
        pass
    
    def get_time_until_reset(self) -> float:
        """Get time remaining until automatic reset attempt.
        
        Returns:
            Seconds until reset, or 0 if not in OPEN state.
        """
        # TODO: Return remaining time if OPEN, else 0
        pass


# =============================================================================
# TASK 2: Circuit Breaker Decorator
# =============================================================================

def circuit_protected(
    breaker: CircuitBreaker,
    fallback: Callable[..., T] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that protects a function with a circuit breaker.
    
    Args:
        breaker: Circuit breaker instance to use.
        fallback: Optional fallback function when circuit is open.
        
    Returns:
        Decorator function.
        
    Example:
        >>> breaker = CircuitBreaker()
        >>> @circuit_protected(breaker, fallback=lambda: "fallback")
        ... def external_call():
        ...     raise ConnectionError("down")
    """
    # TODO: Implement decorator
    # - Use breaker.call() to protect the function
    # - If CircuitOpenError and fallback provided, call fallback
    # - Otherwise re-raise
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
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
    def failing_func():
        raise ConnectionError("service down")
    
    for i in range(3):
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
        assert False, "Should have raised CircuitOpenError"
    except CircuitOpenError as e:
        assert breaker.stats.rejected_calls == 1
    print("  ✓ Calls rejected when OPEN")
    
    # Test automatic reset attempt
    time.sleep(0.15)  # Wait for reset_timeout
    
    # Circuit should transition to HALF_OPEN on next call
    try:
        breaker.call(failing_func)
    except ConnectionError:
        pass
    
    # After failure in HALF_OPEN, should be OPEN again
    assert breaker.state == CircuitState.OPEN
    print("  ✓ HALF_OPEN transitions back to OPEN on failure")
    
    # Test successful recovery
    time.sleep(0.15)
    config2 = CircuitBreakerConfig(failure_threshold=3, success_threshold=2, reset_timeout=0.1)
    breaker2 = CircuitBreaker(config2)
    
    # Open the circuit
    for _ in range(3):
        try:
            breaker2.call(failing_func)
        except ConnectionError:
            pass
    assert breaker2.state == CircuitState.OPEN
    
    time.sleep(0.15)
    
    # Successful calls in HALF_OPEN should close circuit
    breaker2.call(lambda: "success")
    assert breaker2.state == CircuitState.HALF_OPEN  # Need more successes
    
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
    def protected_func():
        raise ConnectionError("down")
    
    # Should use fallback after circuit opens
    result = protected_func()  # Failure 1
    # Result depends on implementation - either exception or fallback
    result = protected_func()  # Failure 2, circuit opens
    result = protected_func()  # Should use fallback
    assert result == "fallback_value"
    print("  ✓ Decorator with fallback works")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
