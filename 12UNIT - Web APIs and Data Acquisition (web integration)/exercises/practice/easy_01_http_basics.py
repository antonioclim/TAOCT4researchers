#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Easy 01 - HTTP Request Basics

Difficulty: ★☆☆☆☆
Estimated Time: 15 minutes
Learning Objective: LO1, LO2

Task:
Implement a function that makes HTTP requests and extracts basic
information from responses. This exercise builds familiarity with
the requests library and HTTP response structure.

Requirements:
1. Make a GET request to a specified URL
2. Extract status code, headers and content
3. Handle basic request errors gracefully

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from typing import Any, Optional

import requests
from requests.exceptions import RequestException


def fetch_url_info(url: str, timeout: int = 10) -> dict[str, Any]:
    """
    Fetch a URL and return response information.
    
    This function makes a GET request to the specified URL and
    extracts key information from the response.
    
    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        
    Returns:
        Dictionary containing:
        - 'success': Boolean indicating if request succeeded
        - 'status_code': HTTP status code (or None if failed)
        - 'content_type': Content-Type header value
        - 'content_length': Response body length in bytes
        - 'headers_count': Number of response headers
        - 'error': Error message if request failed (or None)
        
    Example:
        >>> info = fetch_url_info('https://httpbin.org/get')
        >>> info['success']
        True
        >>> info['status_code']
        200
        >>> isinstance(info['content_type'], str)
        True
    """
    # TODO: Implement this function
    # HINT: Use requests.get() with timeout parameter
    # HINT: Use try/except to handle RequestException
    # HINT: Access response.status_code, response.headers, len(response.content)
    
    pass  # Replace with your implementation


def check_url_exists(url: str) -> bool:
    """
    Check if a URL exists (returns 2xx status).
    
    Uses a HEAD request for efficiency (doesn't download body).
    
    Args:
        url: The URL to check
        
    Returns:
        True if URL returns 2xx status code
        
    Example:
        >>> check_url_exists('https://httpbin.org/status/200')
        True
        >>> check_url_exists('https://httpbin.org/status/404')
        False
    """
    # TODO: Implement this function
    # HINT: Use requests.head() for efficiency
    # HINT: Check if 200 <= status_code < 300
    
    pass  # Replace with your implementation


def get_response_headers(url: str) -> Optional[dict[str, str]]:
    """
    Retrieve response headers from a URL.
    
    Args:
        url: The URL to fetch headers from
        
    Returns:
        Dictionary of headers (lowercase keys) or None if request failed
        
    Example:
        >>> headers = get_response_headers('https://httpbin.org/get')
        >>> 'content-type' in headers
        True
    """
    # TODO: Implement this function
    # HINT: Response headers are case-insensitive; convert to lowercase
    
    pass  # Replace with your implementation


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing HTTP Request Basics exercises...\n')
    
    # Test fetch_url_info
    print('Test 1: fetch_url_info')
    try:
        info = fetch_url_info('https://httpbin.org/get')
        assert info['success'] is True, 'Expected success=True'
        assert info['status_code'] == 200, 'Expected status_code=200'
        assert info['error'] is None, 'Expected no error'
        print('  PASSED: Basic fetch works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test with invalid URL
    try:
        info = fetch_url_info('https://invalid.invalid.invalid')
        assert info['success'] is False, 'Expected success=False'
        assert info['error'] is not None, 'Expected error message'
        print('  PASSED: Error handling works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test check_url_exists
    print('\nTest 2: check_url_exists')
    try:
        assert check_url_exists('https://httpbin.org/status/200') is True
        assert check_url_exists('https://httpbin.org/status/404') is False
        print('  PASSED: URL existence check works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test get_response_headers
    print('\nTest 3: get_response_headers')
    try:
        headers = get_response_headers('https://httpbin.org/get')
        assert headers is not None, 'Expected headers dict'
        assert 'content-type' in headers, 'Expected content-type header'
        print('  PASSED: Header retrieval works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_exercises()
