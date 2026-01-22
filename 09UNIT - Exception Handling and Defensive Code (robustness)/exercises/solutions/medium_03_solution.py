#!/usr/bin/env python3
"""Solution for Exercise 06: Retry Mechanism.

This module provides reference implementations for retry patterns
with exponential backoff.
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


def retry_simple(
    func: Callable[[], T],
    max_attempts: int = 3,
    delay: float = 1.0,
) -> T:
    """Execute function with simple retry logic."""
    last_exception: Exception | None = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs",
                    attempt,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
    
    assert last_exception is not None
    raise last_exception


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
) -> float:
    """Calculate delay for exponential backoff."""
    delay = base_delay * (exponential_base ** (attempt - 1))
    delay = min(delay, max_delay)
    
    if jitter:
        delay *= 0.5 + random.random() * 0.5
    
    return delay


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Execute function with exponential backoff retry."""
    last_exception: Exception | None = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                delay = calculate_backoff_delay(
                    attempt,
                    base_delay=base_delay,
                    max_delay=max_delay,
                )
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.2fs",
                    attempt,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)
    
    assert last_exception is not None
    raise last_exception


@dataclass
class RetryConfig:
    """Configuration for retry behaviour."""
    
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,)
    on_retry: Callable[[int, Exception], None] | None = None


def retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that adds retry logic to a function."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if config.on_retry is not None:
                        config.on_retry(attempt, e)
                    
                    if attempt < config.max_attempts:
                        delay = calculate_backoff_delay(
                            attempt,
                            base_delay=config.base_delay,
                            max_delay=config.max_delay,
                            exponential_base=config.exponential_base,
                        )
                        time.sleep(delay)
            
            assert last_exception is not None
            raise last_exception
        
        return wrapper
    return decorator


@dataclass
class RetryResult:
    """Result of a retry operation."""
    
    success: bool
    value: Any = None
    attempts: int = 0
    exceptions: list[Exception] = field(default_factory=list)
    total_delay: float = 0.0


def retry_with_result(
    func: Callable[[], T],
    config: RetryConfig | None = None,
) -> RetryResult:
    """Execute function with retry and return detailed result."""
    if config is None:
        config = RetryConfig()
    
    result = RetryResult(success=False)
    total_delay = 0.0
    
    for attempt in range(1, config.max_attempts + 1):
        result.attempts = attempt
        
        try:
            result.value = func()
            result.success = True
            result.total_delay = total_delay
            return result
        except config.retryable_exceptions as e:
            result.exceptions.append(e)
            
            if attempt < config.max_attempts:
                delay = calculate_backoff_delay(
                    attempt,
                    base_delay=config.base_delay,
                    max_delay=config.max_delay,
                    exponential_base=config.exponential_base,
                    jitter=False,  # Predictable for testing
                )
                total_delay += delay
                time.sleep(delay)
    
    result.total_delay = total_delay
    return result


def main() -> None:
    """Verify solution implementations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testing retry_simple...")
    call_count = [0]
    
    def succeed_on_third() -> str:
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
    def decorated_flaky() -> str:
        if random.random() < 0.3:
            raise ValueError("random")
        return "decorated success"
    
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
    
    def always_fails() -> None:
        raise ValueError("always fails")
    
    result = retry_with_result(always_fails, RetryConfig(max_attempts=2, base_delay=0.01))
    assert not result.success
    assert result.attempts == 2
    assert len(result.exceptions) == 2
    assert result.value is None
    
    def always_succeeds() -> str:
        return "success"
    
    result = retry_with_result(always_succeeds, RetryConfig(max_attempts=3))
    assert result.success
    assert result.attempts == 1
    assert result.value == "success"
    assert len(result.exceptions) == 0
    print("  ✓ retry_with_result passed")
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
