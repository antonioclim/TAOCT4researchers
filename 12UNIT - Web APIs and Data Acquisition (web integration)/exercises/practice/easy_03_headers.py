#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Practice Exercise: easy_03_headers.py

Difficulty: ★☆☆☆☆ (Easy)
Estimated Time: 15 minutes
Prerequisites: HTTP basics

Learning Objectives:
- LO1: Understand HTTP headers and their role

This exercise focuses on working with HTTP headers for both
requests and responses.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from typing import Any


# =============================================================================
# EXERCISE 1: Request Headers
# =============================================================================

def build_request_headers(
    content_type: str = 'application/json',
    accept: str = 'application/json',
    user_agent: str | None = None,
    auth_token: str | None = None,
    custom_headers: dict[str, str] | None = None
) -> dict[str, str]:
    """
    Build a dictionary of HTTP request headers.
    
    Construct headers dictionary with standard HTTP headers and optional
    custom headers. All provided values should be included.
    
    Args:
        content_type: Content-Type header value
        accept: Accept header value
        user_agent: Optional User-Agent header
        auth_token: Optional Bearer token for Authorization header
        custom_headers: Optional additional headers to include
        
    Returns:
        Dictionary of header name to value mappings
        
    Example:
        >>> headers = build_request_headers(
        ...     auth_token='abc123',
        ...     user_agent='MyBot/1.0'
        ... )
        >>> headers['Authorization']
        'Bearer abc123'
        >>> headers['User-Agent']
        'MyBot/1.0'
    """
    # TODO: Implement this function
    # 1. Start with Content-Type and Accept headers
    # 2. Add User-Agent if provided
    # 3. Add Authorization with 'Bearer ' prefix if token provided
    # 4. Merge custom_headers if provided
    pass


# =============================================================================
# EXERCISE 2: Parse Response Headers
# =============================================================================

def parse_content_type(content_type_header: str) -> dict[str, str]:
    """
    Parse a Content-Type header into its components.
    
    Content-Type headers may include media type and parameters like charset.
    Example: "application/json; charset=utf-8"
    
    Args:
        content_type_header: Raw Content-Type header value
        
    Returns:
        Dictionary with 'media_type' and any parameters
        
    Example:
        >>> parse_content_type('application/json; charset=utf-8')
        {'media_type': 'application/json', 'charset': 'utf-8'}
        >>> parse_content_type('text/html')
        {'media_type': 'text/html'}
    """
    # TODO: Implement this function
    # 1. Split on ';' to separate media type from parameters
    # 2. First part is media_type (strip whitespace)
    # 3. Parse remaining parts as key=value pairs
    pass


# =============================================================================
# EXERCISE 3: Rate Limit Headers
# =============================================================================

def extract_rate_limit_info(headers: dict[str, str]) -> dict[str, int | None]:
    """
    Extract rate limit information from response headers.
    
    Common rate limit headers:
    - X-RateLimit-Limit: Maximum requests per window
    - X-RateLimit-Remaining: Requests remaining
    - X-RateLimit-Reset: Unix timestamp when limit resets
    - Retry-After: Seconds to wait before retrying
    
    Args:
        headers: Dictionary of response headers (case-insensitive keys)
        
    Returns:
        Dictionary with keys 'limit', 'remaining', 'reset', 'retry_after'
        Values are integers or None if header not present
        
    Example:
        >>> headers = {
        ...     'X-RateLimit-Limit': '1000',
        ...     'X-RateLimit-Remaining': '847'
        ... }
        >>> info = extract_rate_limit_info(headers)
        >>> info['limit']
        1000
        >>> info['remaining']
        847
        >>> info['reset'] is None
        True
    """
    # TODO: Implement this function
    # 1. Create case-insensitive header lookup (convert keys to lowercase)
    # 2. Extract each rate limit header, converting to int if present
    # 3. Return dict with all four keys (use None for missing)
    pass


# =============================================================================
# EXERCISE 4: Cache Headers
# =============================================================================

def is_response_cacheable(headers: dict[str, str]) -> bool:
    """
    Determine if a response can be cached based on headers.
    
    A response is NOT cacheable if:
    - Cache-Control contains 'no-store' or 'no-cache'
    - Pragma contains 'no-cache'
    
    Args:
        headers: Dictionary of response headers
        
    Returns:
        True if response can be cached, False otherwise
        
    Example:
        >>> is_response_cacheable({'Cache-Control': 'max-age=3600'})
        True
        >>> is_response_cacheable({'Cache-Control': 'no-store'})
        False
        >>> is_response_cacheable({'Pragma': 'no-cache'})
        False
    """
    # TODO: Implement this function
    # 1. Check Cache-Control header for 'no-store' or 'no-cache'
    # 2. Check Pragma header for 'no-cache'
    # 3. Return True only if none of these are present
    pass


# =============================================================================
# TESTS
# =============================================================================

def test_build_request_headers():
    """Test request header building."""
    # Basic headers
    headers = build_request_headers()
    assert headers['Content-Type'] == 'application/json'
    assert headers['Accept'] == 'application/json'
    
    # With auth token
    headers = build_request_headers(auth_token='secret123')
    assert headers['Authorization'] == 'Bearer secret123'
    
    # With custom headers
    headers = build_request_headers(
        custom_headers={'X-Custom': 'value'}
    )
    assert headers['X-Custom'] == 'value'
    
    print('✓ build_request_headers tests passed')


def test_parse_content_type():
    """Test Content-Type parsing."""
    # Simple type
    result = parse_content_type('application/json')
    assert result['media_type'] == 'application/json'
    
    # With charset
    result = parse_content_type('text/html; charset=utf-8')
    assert result['media_type'] == 'text/html'
    assert result['charset'] == 'utf-8'
    
    # Multiple parameters
    result = parse_content_type('multipart/form-data; boundary=---abc')
    assert result['media_type'] == 'multipart/form-data'
    assert result['boundary'] == '---abc'
    
    print('✓ parse_content_type tests passed')


def test_extract_rate_limit_info():
    """Test rate limit extraction."""
    headers = {
        'X-RateLimit-Limit': '1000',
        'X-RateLimit-Remaining': '500',
        'X-RateLimit-Reset': '1609459200',
    }
    
    info = extract_rate_limit_info(headers)
    assert info['limit'] == 1000
    assert info['remaining'] == 500
    assert info['reset'] == 1609459200
    assert info['retry_after'] is None
    
    # Case insensitivity
    headers_lower = {'x-ratelimit-limit': '100'}
    info = extract_rate_limit_info(headers_lower)
    assert info['limit'] == 100
    
    print('✓ extract_rate_limit_info tests passed')


def test_is_response_cacheable():
    """Test cache determination."""
    assert is_response_cacheable({}) is True
    assert is_response_cacheable({'Cache-Control': 'max-age=3600'}) is True
    assert is_response_cacheable({'Cache-Control': 'no-store'}) is False
    assert is_response_cacheable({'Cache-Control': 'no-cache'}) is False
    assert is_response_cacheable({'Pragma': 'no-cache'}) is False
    
    print('✓ is_response_cacheable tests passed')


if __name__ == '__main__':
    test_build_request_headers()
    test_parse_content_type()
    test_extract_rate_limit_info()
    test_is_response_cacheable()
    print('\n✓ All tests passed!')
