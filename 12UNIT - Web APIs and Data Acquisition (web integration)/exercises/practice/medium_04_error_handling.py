#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Practice Exercise: medium_02_error_handling.py

Difficulty: ★★★☆☆ (Medium)
Estimated Time: 25 minutes
Prerequisites: HTTP basics, exceptions

Learning Objectives:
- LO4: Handle API errors and implement retry logic

This exercise focuses on resilient error handling for API requests.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: int = 60):
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str):
        super().__init__(message, status_code=401)


# =============================================================================
# EXERCISE 1: Classify HTTP Errors
# =============================================================================

def classify_http_error(status_code: int) -> str:
    """
    Classify an HTTP status code into error category.
    
    Categories:
    - 'success': 200-299
    - 'redirect': 300-399
    - 'client_error': 400-499
    - 'server_error': 500-599
    - 'unknown': anything else
    
    Args:
        status_code: HTTP status code
        
    Returns:
        Error category string
        
    Example:
        >>> classify_http_error(200)
        'success'
        >>> classify_http_error(404)
        'client_error'
        >>> classify_http_error(503)
        'server_error'
    """
    # TODO: Implement this function
    pass


# =============================================================================
# EXERCISE 2: Should Retry
# =============================================================================

def should_retry(status_code: int, attempt: int, max_attempts: int = 3) -> bool:
    """
    Determine if a request should be retried based on status and attempts.
    
    Retry rules:
    - Never retry if attempt >= max_attempts
    - Retry on 429 (rate limited)
    - Retry on 5xx errors (server errors)
    - Never retry on other 4xx errors (client errors)
    - Never retry on success (2xx)
    
    Args:
        status_code: HTTP status code from response
        attempt: Current attempt number (1-indexed)
        max_attempts: Maximum number of attempts allowed
        
    Returns:
        True if should retry, False otherwise
        
    Example:
        >>> should_retry(429, 1, 3)
        True
        >>> should_retry(404, 1, 3)
        False
        >>> should_retry(500, 3, 3)
        False
    """
    # TODO: Implement this function
    pass


# =============================================================================
# EXERCISE 3: Calculate Backoff
# =============================================================================

def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = False
) -> float:
    """
    Calculate exponential backoff delay for retry.
    
    Formula: min(base_delay * 2^(attempt-1), max_delay)
    If jitter is True, add random factor (0.5 to 1.5 multiplier).
    
    Args:
        attempt: Current attempt number (1-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Whether to add randomness
        
    Returns:
        Delay in seconds
        
    Example:
        >>> calculate_backoff(1)  # 1 * 2^0 = 1
        1.0
        >>> calculate_backoff(2)  # 1 * 2^1 = 2
        2.0
        >>> calculate_backoff(3)  # 1 * 2^2 = 4
        4.0
        >>> calculate_backoff(10, max_delay=30)  # Capped at 30
        30.0
    """
    # TODO: Implement this function
    # Note: For jitter, use random.uniform(0.5, 1.5) as multiplier
    pass


# =============================================================================
# EXERCISE 4: Retry Wrapper
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behaviour."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    retriable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)


def retry_with_backoff(
    operation: Callable[[], dict[str, Any]],
    config: RetryConfig | None = None
) -> dict[str, Any]:
    """
    Execute operation with retry and exponential backoff.
    
    The operation callable should return a dict with at least:
    - 'status_code': HTTP status code
    - 'data': Response data (on success)
    
    On failure, it may raise exceptions or return error status codes.
    
    Args:
        operation: Callable that performs the API request
        config: Retry configuration (uses defaults if None)
        
    Returns:
        Result from successful operation
        
    Raises:
        APIError: If all retries exhausted
        
    Example:
        >>> def mock_api():
        ...     return {'status_code': 200, 'data': {'result': 'ok'}}
        >>> result = retry_with_backoff(mock_api)
        >>> result['data']['result']
        'ok'
    """
    # TODO: Implement this function
    # 1. Use default config if None provided
    # 2. Loop up to max_attempts times
    # 3. Call operation and check status_code
    # 4. If success (2xx), return result
    # 5. If retriable status, calculate backoff and sleep
    # 6. If non-retriable, raise APIError
    # 7. After all attempts, raise APIError
    pass


# =============================================================================
# EXERCISE 5: Parse Error Response
# =============================================================================

def parse_error_response(response_data: dict[str, Any]) -> str:
    """
    Extract error message from various API error response formats.
    
    APIs return errors in different formats:
    - {'error': 'message'}
    - {'error': {'message': 'text'}}
    - {'message': 'text'}
    - {'errors': ['msg1', 'msg2']}
    - {'detail': 'message'}
    
    Args:
        response_data: Parsed JSON response body
        
    Returns:
        Extracted error message or 'Unknown error'
        
    Example:
        >>> parse_error_response({'error': 'Not found'})
        'Not found'
        >>> parse_error_response({'errors': ['Bad input', 'Missing field']})
        'Bad input; Missing field'
    """
    # TODO: Implement this function
    # Check each format in order and extract message
    pass


# =============================================================================
# TESTS
# =============================================================================

def test_classify_http_error():
    """Test HTTP error classification."""
    assert classify_http_error(200) == 'success'
    assert classify_http_error(201) == 'success'
    assert classify_http_error(301) == 'redirect'
    assert classify_http_error(400) == 'client_error'
    assert classify_http_error(404) == 'client_error'
    assert classify_http_error(429) == 'client_error'
    assert classify_http_error(500) == 'server_error'
    assert classify_http_error(503) == 'server_error'
    assert classify_http_error(100) == 'unknown'
    
    print('✓ classify_http_error tests passed')


def test_should_retry():
    """Test retry decision logic."""
    # Should retry on 429
    assert should_retry(429, 1, 3) is True
    assert should_retry(429, 2, 3) is True
    
    # Should retry on 5xx
    assert should_retry(500, 1, 3) is True
    assert should_retry(503, 1, 3) is True
    
    # Should not retry on client errors (except 429)
    assert should_retry(400, 1, 3) is False
    assert should_retry(404, 1, 3) is False
    
    # Should not retry on success
    assert should_retry(200, 1, 3) is False
    
    # Should not retry if max attempts reached
    assert should_retry(500, 3, 3) is False
    assert should_retry(429, 4, 3) is False
    
    print('✓ should_retry tests passed')


def test_calculate_backoff():
    """Test backoff calculation."""
    assert calculate_backoff(1) == 1.0
    assert calculate_backoff(2) == 2.0
    assert calculate_backoff(3) == 4.0
    assert calculate_backoff(4) == 8.0
    
    # Max delay cap
    assert calculate_backoff(10, max_delay=30) == 30.0
    
    # Different base
    assert calculate_backoff(1, base_delay=2.0) == 2.0
    assert calculate_backoff(2, base_delay=2.0) == 4.0
    
    print('✓ calculate_backoff tests passed')


def test_parse_error_response():
    """Test error message parsing."""
    assert parse_error_response({'error': 'Not found'}) == 'Not found'
    assert parse_error_response({'message': 'Bad request'}) == 'Bad request'
    assert parse_error_response({'detail': 'Auth failed'}) == 'Auth failed'
    
    # Nested error
    result = parse_error_response({'error': {'message': 'Nested error'}})
    assert result == 'Nested error'
    
    # Error list
    result = parse_error_response({'errors': ['Error 1', 'Error 2']})
    assert result == 'Error 1; Error 2'
    
    # Unknown format
    assert parse_error_response({}) == 'Unknown error'
    
    print('✓ parse_error_response tests passed')


def test_retry_with_backoff():
    """Test retry wrapper."""
    # Successful operation
    call_count = 0
    
    def success_op():
        nonlocal call_count
        call_count += 1
        return {'status_code': 200, 'data': 'success'}
    
    result = retry_with_backoff(success_op)
    assert result['data'] == 'success'
    assert call_count == 1
    
    # Operation that fails then succeeds
    attempt = 0
    
    def flaky_op():
        nonlocal attempt
        attempt += 1
        if attempt < 2:
            return {'status_code': 503, 'data': None}
        return {'status_code': 200, 'data': 'recovered'}
    
    config = RetryConfig(max_attempts=3, base_delay=0.01)
    result = retry_with_backoff(flaky_op, config)
    assert result['data'] == 'recovered'
    assert attempt == 2
    
    print('✓ retry_with_backoff tests passed')


if __name__ == '__main__':
    test_classify_http_error()
    test_should_retry()
    test_calculate_backoff()
    test_parse_error_response()
    test_retry_with_backoff()
    print('\n✓ All tests passed!')
