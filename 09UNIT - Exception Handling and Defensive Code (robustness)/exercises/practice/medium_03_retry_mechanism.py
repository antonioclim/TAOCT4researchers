#!/usr/bin/env python3
"""Exercise 06: Retry Mechanism (Medium).

This exercise develops skills in implementing retry patterns
for handling transient failures.

Learning Objectives:
    - Implement retry logic with exponential backoff
    - Create configurable retry decorators
    - Handle different exception types appropriately

Estimated Time: 15 minutes
Difficulty: Medium (★★☆)

Instructions:
    1. Implement retry mechanisms with various configurations
    2. Add proper logging for retry attempts
    3. Run tests to verify your implementation
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


# =============================================================================
# TASK 1: Simple Retry Function
# =============================================================================

def retry_simple(
    func: Callable[[], T],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> T:
    """Execute function with simple retry logic.
    
    Retries the function up to max_attempts times with a fixed
    delay between attempts.
    
    Args:
        func: Function to execute (no arguments).
        max_attempts: Maximum number of attempts.
        delay: Delay in seconds between attempts.
        
    Returns:
        Result of successful function call.
        
    Raises:
        Exception: Last exception if all attempts fail.
        
    Example:
        >>> counter = [0]
        >>> def flaky():
        ...     counter[0] += 1
        ...     if counter[0] < 3:
        ...         raise ValueError("not yet")
        ...     return "success"
        >>> retry_simple(flaky, max_attempts=5, delay=0.01)
        'success'
    """
    # TODO: Implement simple retry logic
    # 1. Loop for max_attempts
    # 2. Try to call func()
    # 3. On exception, log warning and sleep (unless last attempt)
    # 4. Raise last exception if all attempts fail
    pass


# =============================================================================
# TASK 2: Retry with Exponential Backoff
# =============================================================================

def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate delay for exponential backoff.
    
    Args:
        attempt: Current attempt number (1-indexed).
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        exponential_base: Base for exponential calculation.
        jitter: Whether to add random jitter.
        
    Returns:
        Delay in seconds for this attempt.
        
    Example:
        >>> delay = calculate_backoff_delay(1, base_delay=1.0, jitter=False)
        >>> delay
        1.0
        >>> delay = calculate_backoff_delay(3, base_delay=1.0, jitter=False)
        >>> delay
        4.0
    """
    # TODO: Implement exponential backoff calculation
    # 1. Calculate: base_delay * (exponential_base ** (attempt - 1))
    # 2. Cap at max_delay using min()
    # 3. If jitter, multiply by random factor (0.5 to 1.0)
    pass


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute function with exponential backoff retry.
    
    Args:
        func: Function to execute.
        max_attempts: Maximum number of attempts.
        base_delay: Initial delay between retries.
        max_delay: Maximum delay between retries.
        retryable_exceptions: Exception types to retry on.
        
    Returns:
        Result of successful function call.
        
    Raises:
        Exception: Last exception if all attempts fail, or
                   non-retryable exception immediately.
    """
    # TODO: Implement retry with backoff
    # 1. Loop for max_attempts
    # 2. Catch only retryable_exceptions
    # 3. Use calculate_backoff_delay for sleep duration
    # 4. Log attempt number and delay
    pass


# =============================================================================
# TASK 3: Retry Decorator
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behaviour.
    
    Attributes:
        max_attempts: Maximum retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay in seconds.
        exponential_base: Base for exponential backoff.
        retryable_exceptions: Exceptions to retry on.
        on_retry: Optional callback for each retry.
    """
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    on_retry: Callable[[int, Exception], None] | None = None


def retry(config: RetryConfig | None = None) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that adds retry logic to a function.
    
    Args:
        config: Retry configuration (uses defaults if None).
        
    Returns:
        Decorator function.
        
    Example:
        >>> @retry(RetryConfig(max_attempts=3, base_delay=0.01))
        ... def flaky_function():
        ...     if random.random() < 0.5:
        ...         raise ValueError("random failure")
        ...     return "success"
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # TODO: Implement retry logic in wrapper
            # 1. Loop for max_attempts
            # 2. Try calling func(*args, **kwargs)
            # 3. Catch retryable exceptions
            # 4. Call on_retry callback if provided
            # 5. Calculate and apply backoff delay
            # 6. Raise last exception if all attempts fail
            pass
        
        return wrapper
    return decorator


# =============================================================================
# TASK 4: Retry with Result Tracking
# =============================================================================

@dataclass
class RetryResult:
    """Result of a retry operation.
    
    Attributes:
        success: Whether operation succeeded.
        value: Return value if successful.
        attempts: Number of attempts made.
        exceptions: List of exceptions from failed attempts.
        total_delay: Total time spent waiting between retries.
    """
    
    success: bool
    value: Any = None
    attempts: int = 0
    exceptions: list[Exception] = field(default_factory=list)
    total_delay: float = 0.0


def retry_with_result(
    func: Callable[[], T],
    config: RetryConfig | None = None,
) -> RetryResult:
    """Execute function with retry and return detailed result.
    
    Unlike other retry functions, this does not raise exceptions.
    Instead, it returns a RetryResult with all attempt information.
    
    Args:
        func: Function to execute.
        config: Retry configuration.
        
    Returns:
        RetryResult with outcome details.
        
    Example:
        >>> def always_fails():
        ...     raise ValueError("always")
        >>> result = retry_with_result(always_fails, RetryConfig(max_attempts=2, base_delay=0.01))
        >>> result.success
        False
        >>> result.attempts
        2
        >>> len(result.exceptions)
        2
    """
    if config is None:
        config = RetryConfig()
    
    # TODO: Implement retry with result tracking
    # 1. Track all attempts and exceptions
    # 2. Track total delay time
    # 3. Return RetryResult with all information
    # 4. Don't raise exceptions, just record them
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing retry_simple...")
    call_count = [0]
    def succeed_on_third():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError(f"attempt {call_count[0]}")
        return "success"
    
    result = retry_simple(succeed_on_third, max_attempts=5, delay=0.01)
    assert result == "success"
    assert call_count[0] == 3
    print("  ✓ retry_simple passed")
    
    print("Testing calculate_backoff_delay...")
    assert calculate_backoff_delay(1, base_delay=1.0, jitter=False) == 1.0
    assert calculate_backoff_delay(2, base_delay=1.0, jitter=False) == 2.0
    assert calculate_backoff_delay(3, base_delay=1.0, jitter=False) == 4.0
    assert calculate_backoff_delay(10, base_delay=1.0, max_delay=30.0, jitter=False) == 30.0
    print("  ✓ calculate_backoff_delay passed")
    
    print("Testing retry_with_backoff...")
    call_count[0] = 0
    result = retry_with_backoff(
        succeed_on_third,
        max_attempts=5,
        base_delay=0.01,
        retryable_exceptions=(ValueError,),
    )
    assert result == "success"
    print("  ✓ retry_with_backoff passed")
    
    print("Testing retry decorator...")
    
    @retry(RetryConfig(max_attempts=3, base_delay=0.01))
    def decorated_flaky():
        if random.random() < 0.3:
            raise ValueError("random")
        return "decorated success"
    
    # Should eventually succeed (high probability with 3 attempts)
    successes = 0
    for _ in range(10):
        try:
            if decorated_flaky() == "decorated success":
                successes += 1
        except ValueError:
            pass
    assert successes > 0
    print("  ✓ retry decorator passed")
    
    print("Testing retry_with_result...")
    def always_fails():
        raise ValueError("always fails")
    
    result = retry_with_result(always_fails, RetryConfig(max_attempts=2, base_delay=0.01))
    assert not result.success
    assert result.attempts == 2
    assert len(result.exceptions) == 2
    assert result.value is None
    
    def always_succeeds():
        return "success"
    
    result = retry_with_result(always_succeeds, RetryConfig(max_attempts=3))
    assert result.success
    assert result.attempts == 1
    assert result.value == "success"
    assert len(result.exceptions) == 0
    print("  ✓ retry_with_result passed")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
