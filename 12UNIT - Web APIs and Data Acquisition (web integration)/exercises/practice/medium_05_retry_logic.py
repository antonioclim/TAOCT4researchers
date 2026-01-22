#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Medium 03 - Error Handling and Retry Logic

Difficulty: ★★★☆☆
Estimated Time: 25 minutes
Learning Objective: LO4

Task:
Implement robust error handling and retry logic for API clients.
Your implementation must gracefully handle network failures, rate
limits and server errors.

Requirements:
1. Implement exponential backoff retry
2. Handle rate limit responses (429)
3. Distinguish retriable from non-retriable errors
4. Implement comprehensive logging

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import requests
from requests.exceptions import RequestException, HTTPError, Timeout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

T = TypeVar('T')


def is_retriable_error(status_code: int) -> bool:
    """
    Determine if an HTTP status code indicates a retriable error.
    
    Retriable errors are temporary failures that may succeed on retry:
    - 429 Too Many Requests (rate limited)
    - 500 Internal Server Error
    - 502 Bad Gateway
    - 503 Service Unavailable
    - 504 Gateway Timeout
    
    Args:
        status_code: HTTP status code
        
    Returns:
        True if the error is retriable
        
    Example:
        >>> is_retriable_error(429)
        True
        >>> is_retriable_error(404)
        False
        >>> is_retriable_error(503)
        True
    """
    # TODO: Implement retriable error detection
    # HINT: 429 and 5xx errors (except 501) are typically retriable
    
    pass  # Replace with your implementation


def calculate_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate delay for exponential backoff.
    
    Exponential backoff increases delay exponentially with each attempt:
    delay = min(base_delay * 2^attempt, max_delay)
    
    Adding jitter (randomness) prevents thundering herd problems when
    multiple clients retry simultaneously.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
        
    Example:
        >>> delay = calculate_backoff(0, base_delay=1.0, jitter=False)
        >>> delay
        1.0
        >>> delay = calculate_backoff(3, base_delay=1.0, jitter=False)
        >>> delay
        8.0
    """
    # TODO: Implement exponential backoff calculation
    # HINT: Use 2 ** attempt for exponential growth
    # HINT: Add random.uniform(0, delay * 0.1) for jitter
    
    pass  # Replace with your implementation


def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: int = 30
) -> Optional[dict[str, Any]]:
    """
    Fetch URL with automatic retry on retriable errors.
    
    Args:
        url: URL to fetch
        max_retries: Maximum retry attempts
        base_delay: Base delay for exponential backoff
        timeout: Request timeout in seconds
        
    Returns:
        Response JSON or None if all retries failed
        
    Example:
        >>> result = fetch_with_retry('https://httpbin.org/get')
        >>> result is not None
        True
    """
    # TODO: Implement retry logic
    # HINT: Loop up to max_retries
    # HINT: Catch RequestException and HTTPError
    # HINT: Use is_retriable_error() to decide whether to retry
    # HINT: Use calculate_backoff() for delay between retries
    # HINT: Log each attempt and failure
    
    pass  # Replace with your implementation


@dataclass
class RateLimitHandler:
    """
    Handle rate limit responses from APIs.
    
    Extracts rate limit information from response headers and
    calculates appropriate wait times.
    """
    
    default_wait: float = 60.0
    
    def get_retry_after(self, response: requests.Response) -> float:
        """
        Extract retry delay from response.
        
        Checks for Retry-After header (seconds or HTTP date) and
        X-RateLimit-Reset header (Unix timestamp).
        
        Args:
            response: HTTP response object
            
        Returns:
            Seconds to wait before retry
        """
        # TODO: Implement retry-after extraction
        # HINT: Check 'Retry-After' header first
        # HINT: Check 'X-RateLimit-Reset' as fallback
        # HINT: Return self.default_wait if no headers found
        
        pass  # Replace with your implementation
    
    def get_rate_limit_status(
        self,
        response: requests.Response
    ) -> dict[str, Optional[int]]:
        """
        Extract rate limit status from response headers.
        
        Args:
            response: HTTP response object
            
        Returns:
            Dictionary with 'limit', 'remaining', 'reset' keys
        """
        # TODO: Extract rate limit headers
        # Common patterns: X-RateLimit-Limit, X-RateLimit-Remaining
        
        pass  # Replace with your implementation


def retry_with_rate_limit(
    func: Callable[[], requests.Response],
    max_retries: int = 5
) -> Optional[requests.Response]:
    """
    Execute function with retry and rate limit handling.
    
    This is a higher-order function that wraps any function returning
    a Response object with retry and rate limit logic.
    
    Args:
        func: Function that makes HTTP request
        max_retries: Maximum retry attempts
        
    Returns:
        Response object or None if all retries failed
        
    Example:
        >>> def make_request():
        ...     return requests.get('https://httpbin.org/get')
        >>> response = retry_with_rate_limit(make_request)
        >>> response.status_code
        200
    """
    handler = RateLimitHandler()
    
    # TODO: Implement retry with rate limit handling
    # HINT: Call func() and check response status
    # HINT: If 429, use handler.get_retry_after() for delay
    # HINT: If other retriable error, use exponential backoff
    # HINT: Log all retries and waits
    
    pass  # Replace with your implementation


class RobustAPIClient:
    """
    Production-quality API client with comprehensive error handling.
    
    Features:
    - Automatic retry with exponential backoff
    - Rate limit detection and compliance
    - Comprehensive logging
    - Configurable timeouts
    """
    
    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        timeout: int = 30,
        respect_rate_limits: bool = True
    ):
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.timeout = timeout
        self.respect_rate_limits = respect_rate_limits
        self.session = requests.Session()
        self._rate_limit_handler = RateLimitHandler()
    
    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retry and error handling.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            **kwargs: Additional request arguments
            
        Returns:
            Response object or None if failed
        """
        url = f'{self.base_url}/{endpoint.lstrip("/")}'
        kwargs.setdefault('timeout', self.timeout)
        
        # TODO: Implement robust request with retry
        # HINT: Use retry loop with exponential backoff
        # HINT: Handle rate limits specially if respect_rate_limits
        # HINT: Log all attempts, successes and failures
        
        pass  # Replace with your implementation
    
    def get(self, endpoint: str, **kwargs: Any) -> Optional[dict[str, Any]]:
        """Make GET request and return JSON."""
        response = self.request('GET', endpoint, **kwargs)
        if response and response.ok:
            return response.json()
        return None
    
    def post(
        self,
        endpoint: str,
        data: dict[str, Any],
        **kwargs: Any
    ) -> Optional[dict[str, Any]]:
        """Make POST request and return JSON."""
        response = self.request('POST', endpoint, json=data, **kwargs)
        if response and response.ok:
            return response.json()
        return None


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing Error Handling exercises...\n')
    
    # Test is_retriable_error
    print('Test 1: is_retriable_error')
    try:
        assert is_retriable_error(429) is True, '429 should be retriable'
        assert is_retriable_error(500) is True, '500 should be retriable'
        assert is_retriable_error(503) is True, '503 should be retriable'
        assert is_retriable_error(400) is False, '400 should not be retriable'
        assert is_retriable_error(404) is False, '404 should not be retriable'
        assert is_retriable_error(401) is False, '401 should not be retriable'
        print('  PASSED: Retriable detection works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test calculate_backoff
    print('\nTest 2: calculate_backoff')
    try:
        # Without jitter
        assert calculate_backoff(0, 1.0, jitter=False) == 1.0
        assert calculate_backoff(1, 1.0, jitter=False) == 2.0
        assert calculate_backoff(2, 1.0, jitter=False) == 4.0
        assert calculate_backoff(10, 1.0, 60.0, jitter=False) == 60.0  # Capped
        
        # With jitter (should vary)
        delays = [calculate_backoff(2, 1.0, jitter=True) for _ in range(10)]
        assert len(set(delays)) > 1, 'Jitter should add variation'
        
        print('  PASSED: Backoff calculation works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test fetch_with_retry
    print('\nTest 3: fetch_with_retry')
    try:
        result = fetch_with_retry('https://httpbin.org/get', max_retries=2)
        assert result is not None, 'Expected successful fetch'
        print('  PASSED: Retry fetch works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test RobustAPIClient
    print('\nTest 4: RobustAPIClient')
    try:
        client = RobustAPIClient('https://httpbin.org', max_retries=2)
        result = client.get('/get')
        assert result is not None, 'Expected response'
        print('  PASSED: Robust client works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_exercises()
