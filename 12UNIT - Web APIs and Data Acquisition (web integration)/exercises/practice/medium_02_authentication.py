#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Medium 02 - API Authentication

Difficulty: ★★★☆☆
Estimated Time: 25 minutes
Learning Objective: LO3

Task:
Implement authentication mechanisms for API access. Your implementation
must support multiple authentication strategies and securely manage
credentials.

Requirements:
1. Implement API key authentication (header and query)
2. Implement HTTP Basic authentication
3. Implement Bearer token authentication
4. Securely load credentials from environment variables

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Optional

import requests


@dataclass
class Credentials:
    """Secure credential container loaded from environment."""
    
    api_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    
    @classmethod
    def from_environment(cls, prefix: str = 'API') -> 'Credentials':
        """
        Load credentials from environment variables.
        
        Looks for variables with given prefix:
        - {PREFIX}_KEY for API key
        - {PREFIX}_USERNAME for username
        - {PREFIX}_PASSWORD for password
        - {PREFIX}_TOKEN for bearer token
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Credentials instance with available values
            
        Example:
            >>> # With API_KEY=secret in environment
            >>> creds = Credentials.from_environment('API')
            >>> creds.api_key is not None or creds.api_key is None  # Depends on env
            True
        """
        # TODO: Implement environment variable loading
        # HINT: Use os.environ.get() with None default
        
        pass  # Replace with your implementation


def authenticate_with_api_key(
    url: str,
    api_key: str,
    key_name: str = 'X-API-Key',
    in_header: bool = True
) -> Optional[dict[str, Any]]:
    """
    Make authenticated request using API key.
    
    Args:
        url: API endpoint URL
        api_key: The API key value
        key_name: Header or parameter name for key
        in_header: If True, send in header; if False, in query params
        
    Returns:
        Response JSON or None if request failed
        
    Example:
        >>> result = authenticate_with_api_key(
        ...     'https://httpbin.org/headers',
        ...     'test_key_123',
        ...     'X-API-Key',
        ...     in_header=True
        ... )
        >>> 'X-Api-Key' in str(result)
        True
    """
    # TODO: Implement API key authentication
    # HINT: Use headers={key_name: api_key} for header auth
    # HINT: Use params={key_name: api_key} for query auth
    
    pass  # Replace with your implementation


def authenticate_with_basic(
    url: str,
    username: str,
    password: str
) -> Optional[dict[str, Any]]:
    """
    Make authenticated request using HTTP Basic authentication.
    
    HTTP Basic auth sends credentials as base64-encoded header:
    Authorization: Basic base64(username:password)
    
    Args:
        url: API endpoint URL
        username: Account username
        password: Account password
        
    Returns:
        Response JSON or None if request failed
        
    Example:
        >>> result = authenticate_with_basic(
        ...     'https://httpbin.org/basic-auth/user/pass',
        ...     'user',
        ...     'pass'
        ... )
        >>> result.get('authenticated')
        True
    """
    # TODO: Implement Basic authentication
    # HINT: requests has built-in support: auth=(username, password)
    # OR manually: base64.b64encode(f'{username}:{password}'.encode())
    
    pass  # Replace with your implementation


def authenticate_with_bearer(
    url: str,
    token: str
) -> Optional[dict[str, Any]]:
    """
    Make authenticated request using Bearer token.
    
    Bearer authentication sends token in Authorization header:
    Authorization: Bearer <token>
    
    Args:
        url: API endpoint URL
        token: Bearer token value
        
    Returns:
        Response JSON or None if request failed
        
    Example:
        >>> result = authenticate_with_bearer(
        ...     'https://httpbin.org/bearer',
        ...     'test_token_abc'
        ... )
        >>> result.get('authenticated')
        True
    """
    # TODO: Implement Bearer token authentication
    # HINT: Set header 'Authorization': f'Bearer {token}'
    
    pass  # Replace with your implementation


class AuthenticatedClient:
    """
    Reusable client with configurable authentication.
    
    Supports multiple authentication strategies and maintains
    session for connection reuse.
    """
    
    def __init__(
        self,
        base_url: str,
        auth_type: str = 'none',
        credentials: Optional[Credentials] = None
    ):
        """
        Initialise authenticated client.
        
        Args:
            base_url: API base URL
            auth_type: One of 'none', 'api_key', 'basic', 'bearer'
            credentials: Credentials instance
        """
        self.base_url = base_url.rstrip('/')
        self.auth_type = auth_type
        self.credentials = credentials or Credentials()
        self.session = requests.Session()
        
        # TODO: Configure session based on auth_type
        # HINT: Set appropriate headers or auth tuple on self.session
    
    def get(self, endpoint: str, **kwargs: Any) -> Optional[dict[str, Any]]:
        """
        Make authenticated GET request.
        
        Args:
            endpoint: API endpoint path
            **kwargs: Additional request arguments
            
        Returns:
            Response JSON or None if failed
        """
        # TODO: Implement authenticated GET
        # HINT: Use self.session.get() with configured auth
        
        pass  # Replace with your implementation
    
    def post(
        self,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[dict[str, Any]]:
        """
        Make authenticated POST request.
        
        Args:
            endpoint: API endpoint path
            data: JSON data to send
            **kwargs: Additional request arguments
            
        Returns:
            Response JSON or None if failed
        """
        # TODO: Implement authenticated POST
        
        pass  # Replace with your implementation


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing API Authentication exercises...\n')
    
    # Test Credentials.from_environment
    print('Test 1: Credentials.from_environment')
    try:
        # Set test environment variable
        os.environ['TEST_API_KEY'] = 'secret123'
        creds = Credentials.from_environment('TEST_API')
        assert creds.api_key == 'secret123', 'Expected API key from env'
        del os.environ['TEST_API_KEY']
        print('  PASSED: Environment loading works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test API key authentication (header)
    print('\nTest 2: API key in header')
    try:
        result = authenticate_with_api_key(
            'https://httpbin.org/headers',
            'test_key',
            'X-API-Key',
            in_header=True
        )
        assert result is not None, 'Expected response'
        headers = result.get('headers', {})
        assert 'X-Api-Key' in headers or 'X-API-Key' in str(headers)
        print('  PASSED: Header auth works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test API key authentication (query)
    print('\nTest 3: API key in query')
    try:
        result = authenticate_with_api_key(
            'https://httpbin.org/get',
            'test_key',
            'api_key',
            in_header=False
        )
        assert result is not None, 'Expected response'
        assert 'api_key' in result.get('args', {})
        print('  PASSED: Query auth works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test Basic authentication
    print('\nTest 4: Basic authentication')
    try:
        result = authenticate_with_basic(
            'https://httpbin.org/basic-auth/testuser/testpass',
            'testuser',
            'testpass'
        )
        assert result is not None, 'Expected response'
        assert result.get('authenticated') is True
        print('  PASSED: Basic auth works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test Bearer authentication
    print('\nTest 5: Bearer authentication')
    try:
        result = authenticate_with_bearer(
            'https://httpbin.org/bearer',
            'test_token'
        )
        assert result is not None, 'Expected response'
        assert result.get('authenticated') is True
        print('  PASSED: Bearer auth works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_exercises()
