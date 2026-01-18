#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Practice Exercise: hard_02_api_client.py

Difficulty: ★★★★★ (Hard)
Estimated Time: 45 minutes
Prerequisites: HTTP, authentication, error handling

Learning Objectives:
- LO2: Apply API consumption techniques
- LO3: Apply authentication mechanisms
- LO4: Handle API errors with retry logic

Build a complete, resilient API client class.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ClientConfig:
    """API client configuration."""
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    page_size: int = 100
    user_agent: str = 'ResearchClient/1.0'


# =============================================================================
# AUTHENTICATION INTERFACE
# =============================================================================

class Authenticator(ABC):
    """Abstract base class for authentication strategies."""
    
    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        """Return authentication headers."""
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if authentication is still valid."""
        pass
    
    @abstractmethod
    def refresh(self) -> None:
        """Refresh authentication if needed."""
        pass


# =============================================================================
# EXERCISE 1: API Key Authenticator
# =============================================================================

class APIKeyAuth(Authenticator):
    """
    API key authentication.
    
    Supports placing the API key in headers or query parameters.
    API keys don't expire, so is_valid always returns True.
    
    Attributes:
        api_key: The API key value
        header_name: Header name (e.g., 'X-API-Key')
        placement: 'header' or 'query'
        
    Example:
        >>> auth = APIKeyAuth('secret123', 'X-API-Key')
        >>> auth.get_headers()
        {'X-API-Key': 'secret123'}
    """
    
    def __init__(
        self, 
        api_key: str, 
        header_name: str = 'X-API-Key',
        placement: str = 'header'
    ):
        # TODO: Store attributes
        pass
    
    def get_headers(self) -> dict[str, str]:
        """Return API key header if placement is 'header'."""
        # TODO: Implement - return dict with header_name: api_key
        # Return empty dict if placement is 'query'
        pass
    
    def get_params(self) -> dict[str, str]:
        """Return API key as query param if placement is 'query'."""
        # TODO: Implement - return dict with header_name: api_key
        # Return empty dict if placement is 'header'
        pass
    
    def is_valid(self) -> bool:
        """API keys don't expire."""
        return True
    
    def refresh(self) -> None:
        """Nothing to refresh for API keys."""
        pass


# =============================================================================
# EXERCISE 2: Token Authenticator with Expiry
# =============================================================================

class TokenAuth(Authenticator):
    """
    Bearer token authentication with expiry tracking.
    
    Tracks token expiry and provides refresh capability.
    
    Attributes:
        token: Current access token
        expires_at: Datetime when token expires
        refresh_callback: Optional callable to get new token
        
    Example:
        >>> auth = TokenAuth('token123', datetime.now() + timedelta(hours=1))
        >>> auth.get_headers()
        {'Authorization': 'Bearer token123'}
        >>> auth.is_valid()
        True
    """
    
    def __init__(
        self,
        token: str,
        expires_at: datetime,
        refresh_callback: callable | None = None
    ):
        # TODO: Store attributes
        pass
    
    def get_headers(self) -> dict[str, str]:
        """Return Bearer token header."""
        # TODO: Implement
        pass
    
    def is_valid(self) -> bool:
        """Check if token is not expired (with 60s buffer)."""
        # TODO: Return True if expires_at > now + 60 seconds
        pass
    
    def refresh(self) -> None:
        """Refresh token using callback if available."""
        # TODO: If refresh_callback exists, call it to get new token/expiry
        # Expected callback return: {'token': str, 'expires_at': datetime}
        pass


# =============================================================================
# EXERCISE 3: Response Cache
# =============================================================================

@dataclass
class CacheEntry:
    """Single cache entry with expiry."""
    data: Any
    expires_at: datetime


class ResponseCache:
    """
    Simple in-memory response cache with TTL.
    
    Cache key is constructed from URL and parameters.
    
    Example:
        >>> cache = ResponseCache(ttl_seconds=300)
        >>> cache.set('/api/data', {'q': 'test'}, {'results': [1,2,3]})
        >>> cache.get('/api/data', {'q': 'test'})
        {'results': [1, 2, 3]}
        >>> cache.get('/api/other')  # Not cached
        None
    """
    
    def __init__(self, ttl_seconds: int = 300):
        # TODO: Initialise ttl and empty cache dict
        pass
    
    def _make_key(self, url: str, params: dict | None = None) -> str:
        """Create cache key from URL and params."""
        # TODO: Combine url and sorted params into unique string key
        pass
    
    def get(self, url: str, params: dict | None = None) -> Any | None:
        """Get cached response if exists and not expired."""
        # TODO: Look up key, check expiry, return data or None
        pass
    
    def set(self, url: str, params: dict | None, data: Any) -> None:
        """Cache response data."""
        # TODO: Create CacheEntry with expiry and store
        pass
    
    def clear(self) -> None:
        """Clear all cached entries."""
        # TODO: Empty the cache
        pass


# =============================================================================
# EXERCISE 4: Complete API Client
# =============================================================================

class ResearchAPIClient:
    """
    Complete API client with authentication, caching and retry.
    
    Features:
    - Configurable authentication
    - Response caching
    - Automatic retry with exponential backoff
    - Pagination support
    
    Example:
        >>> config = ClientConfig(base_url='https://api.example.com')
        >>> auth = APIKeyAuth('secret', 'X-API-Key')
        >>> client = ResearchAPIClient(config, auth)
        >>> data = client.get('/works', params={'query': 'python'})
    """
    
    def __init__(
        self,
        config: ClientConfig,
        auth: Authenticator | None = None,
        cache: ResponseCache | None = None
    ):
        # TODO: Store config, auth, cache
        # Create requests.Session with default headers
        pass
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        # TODO: Combine base_url and endpoint
        pass
    
    def _should_retry(self, status_code: int, attempt: int) -> bool:
        """Determine if request should be retried."""
        # TODO: Retry on 429 and 5xx if under max_retries
        pass
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate backoff delay."""
        # TODO: Exponential backoff capped at max_delay
        pass
    
    def get(
        self,
        endpoint: str,
        params: dict | None = None,
        use_cache: bool = True
    ) -> dict[str, Any]:
        """
        Make GET request with retry and caching.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_cache: Whether to use cached response
            
        Returns:
            Parsed JSON response
            
        Raises:
            APIError: If request fails after retries
        """
        # TODO: Implement with:
        # 1. Check cache if use_cache=True
        # 2. Refresh auth if invalid
        # 3. Build headers with auth
        # 4. Make request with retry loop
        # 5. Cache successful response
        # 6. Return parsed JSON
        pass
    
    def get_paginated(
        self,
        endpoint: str,
        params: dict | None = None,
        max_items: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """
        Iterate through paginated results.
        
        Yields individual items from paginated responses.
        Stops when no more results or max_items reached.
        
        Args:
            endpoint: API endpoint path
            params: Base query parameters
            max_items: Maximum items to retrieve
            
        Yields:
            Individual result items
        """
        # TODO: Implement pagination:
        # 1. Start with page=1 or cursor=*
        # 2. Make request, yield items
        # 3. Check for next page/cursor
        # 4. Stop at max_items or no more results
        pass
    
    def post(
        self,
        endpoint: str,
        data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Make POST request.
        
        Args:
            endpoint: API endpoint path
            data: JSON body data
            
        Returns:
            Parsed JSON response
        """
        # TODO: Implement POST request with retry
        pass


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class APIError(Exception):
    """API request error."""
    
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# =============================================================================
# TESTS
# =============================================================================

def test_api_key_auth():
    """Test API key authentication."""
    # Header placement
    auth = APIKeyAuth('secret123', 'X-API-Key', 'header')
    assert auth.get_headers() == {'X-API-Key': 'secret123'}
    assert auth.get_params() == {}
    assert auth.is_valid() is True
    
    # Query placement
    auth = APIKeyAuth('secret123', 'api_key', 'query')
    assert auth.get_headers() == {}
    assert auth.get_params() == {'api_key': 'secret123'}
    
    print('✓ APIKeyAuth tests passed')


def test_token_auth():
    """Test token authentication."""
    future = datetime.now() + timedelta(hours=1)
    auth = TokenAuth('token123', future)
    
    assert auth.get_headers() == {'Authorization': 'Bearer token123'}
    assert auth.is_valid() is True
    
    # Expired token
    past = datetime.now() - timedelta(hours=1)
    auth_expired = TokenAuth('old_token', past)
    assert auth_expired.is_valid() is False
    
    print('✓ TokenAuth tests passed')


def test_response_cache():
    """Test response caching."""
    cache = ResponseCache(ttl_seconds=300)
    
    # Cache miss
    assert cache.get('/api/data') is None
    
    # Cache set and hit
    cache.set('/api/data', {'q': 'test'}, {'results': [1, 2, 3]})
    assert cache.get('/api/data', {'q': 'test'}) == {'results': [1, 2, 3]}
    
    # Different params = different key
    assert cache.get('/api/data', {'q': 'other'}) is None
    
    # Clear cache
    cache.clear()
    assert cache.get('/api/data', {'q': 'test'}) is None
    
    print('✓ ResponseCache tests passed')


def test_client_url_building():
    """Test URL construction."""
    config = ClientConfig(base_url='https://api.example.com')
    client = ResearchAPIClient(config)
    
    assert client._build_url('/works') == 'https://api.example.com/works'
    assert client._build_url('works') == 'https://api.example.com/works'
    
    print('✓ URL building tests passed')


if __name__ == '__main__':
    test_api_key_auth()
    test_token_auth()
    test_response_cache()
    test_client_url_building()
    print('\n✓ All tests passed!')
