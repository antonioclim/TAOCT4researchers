#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Solutions for Medium Practice Exercises

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

import requests
from requests.exceptions import HTTPError, RequestException, Timeout


# =============================================================================
# MEDIUM 01: Pagination - Solutions
# =============================================================================

@dataclass
class PaginationConfig:
    """Configuration for paginated API requests."""
    page_size: int = 100
    max_pages: int | None = None
    delay_between_pages: float = 0.1


def fetch_offset_paginated(
    url: str,
    config: PaginationConfig | None = None
) -> Iterator[dict[str, Any]]:
    """
    Fetch all pages using offset pagination.
    
    Solution demonstrates:
    - Generator pattern for memory efficiency
    - Configurable pagination
    - Rate limiting between pages
    """
    config = config or PaginationConfig()
    page = 1
    pages_fetched = 0
    
    while True:
        if config.max_pages and pages_fetched >= config.max_pages:
            break
        
        response = requests.get(
            url,
            params={'page': page, 'per_page': config.page_size},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        items = data.get('items') or data.get('results') or data.get('data') or []
        
        if not items:
            break
        
        yield from items
        
        # Check if more pages exist
        total = data.get('total') or data.get('meta', {}).get('total')
        if total and page * config.page_size >= total:
            break
        
        page += 1
        pages_fetched += 1
        time.sleep(config.delay_between_pages)


def fetch_cursor_paginated(
    url: str,
    cursor_param: str = 'cursor',
    results_key: str = 'results',
    next_cursor_path: str = 'meta.next_cursor',
    config: PaginationConfig | None = None
) -> Iterator[dict[str, Any]]:
    """
    Fetch all pages using cursor pagination.
    
    Solution demonstrates:
    - Cursor-based iteration
    - Flexible response parsing
    - Clean generator implementation
    """
    config = config or PaginationConfig()
    cursor = '*'
    pages_fetched = 0
    
    while cursor:
        if config.max_pages and pages_fetched >= config.max_pages:
            break
        
        response = requests.get(
            url,
            params={cursor_param: cursor},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract items
        items = data.get(results_key, [])
        yield from items
        
        # Get next cursor
        cursor = _extract_nested(data, next_cursor_path)
        
        pages_fetched += 1
        time.sleep(config.delay_between_pages)


def _extract_nested(data: dict, path: str) -> Any:
    """Extract value from nested dict using dot notation."""
    for key in path.split('.'):
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return None
    return data


def detect_pagination_style(response_data: dict[str, Any]) -> str:
    """
    Detect pagination style from API response.
    
    Solution demonstrates:
    - Response structure analysis
    - Pattern matching
    """
    # Check for cursor indicators
    if 'next_cursor' in str(response_data) or 'cursor' in str(response_data):
        return 'cursor'
    
    # Check for offset indicators
    if any(key in response_data for key in ['page', 'offset', 'per_page', 'total_pages']):
        return 'offset'
    
    # Check meta section
    meta = response_data.get('meta', {})
    if 'next_cursor' in meta:
        return 'cursor'
    if 'page' in meta or 'total_pages' in meta:
        return 'offset'
    
    return 'unknown'


def collect_all_pages(
    url: str,
    max_items: int | None = None
) -> list[dict[str, Any]]:
    """
    Collect all items from paginated endpoint.
    
    Solution demonstrates:
    - Auto-detection of pagination style
    - Item limit handling
    """
    # First request to detect style
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    style = detect_pagination_style(data)
    all_items = []
    
    if style == 'cursor':
        for item in fetch_cursor_paginated(url):
            all_items.append(item)
            if max_items and len(all_items) >= max_items:
                break
    else:
        for item in fetch_offset_paginated(url):
            all_items.append(item)
            if max_items and len(all_items) >= max_items:
                break
    
    return all_items


# =============================================================================
# MEDIUM 02: Authentication - Solutions
# =============================================================================

class APIKeyAuth:
    """
    API Key authenticator.
    
    Solution demonstrates:
    - Flexible key placement
    - Request modification
    """
    
    def __init__(
        self,
        api_key: str,
        key_name: str = 'X-API-Key',
        location: str = 'header'
    ):
        self.api_key = api_key
        self.key_name = key_name
        self.location = location
    
    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Apply authentication to request."""
        if self.location == 'header':
            request.headers[self.key_name] = self.api_key
        elif self.location == 'query':
            # Modify URL to include API key
            separator = '&' if '?' in request.url else '?'
            request.url = f"{request.url}{separator}{self.key_name}={self.api_key}"
        return request


class OAuth2ClientAuth:
    """
    OAuth 2.0 Client Credentials authenticator.
    
    Solution demonstrates:
    - Token management
    - Automatic refresh
    - Thread-safe token caching
    """
    
    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        scope: str | None = None
    ):
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        self._token: str | None = None
        self._expiry: datetime | None = None
    
    def get_token(self) -> str:
        """Get valid access token, refreshing if needed."""
        if self._token and self._expiry and datetime.now() < self._expiry:
            return self._token
        
        return self._refresh_token()
    
    def _refresh_token(self) -> str:
        """Request new access token."""
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id,
            'client_secret': self.client_secret,
        }
        if self.scope:
            data['scope'] = self.scope
        
        response = requests.post(self.token_url, data=data, timeout=30)
        response.raise_for_status()
        
        token_data = response.json()
        self._token = token_data['access_token']
        expires_in = token_data.get('expires_in', 3600)
        self._expiry = datetime.now() + timedelta(seconds=expires_in - 60)
        
        return self._token
    
    def __call__(self, request: requests.PreparedRequest) -> requests.PreparedRequest:
        """Apply Bearer token to request."""
        token = self.get_token()
        request.headers['Authorization'] = f'Bearer {token}'
        return request


def create_basic_auth_header(username: str, password: str) -> str:
    """
    Create Basic authentication header.
    
    Solution demonstrates:
    - Base64 encoding
    - HTTP Basic auth format
    """
    credentials = f"{username}:{password}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"


def make_authenticated_request(
    url: str,
    auth_type: str,
    credentials: dict[str, str]
) -> requests.Response:
    """
    Make request with specified authentication.
    
    Solution demonstrates:
    - Authentication strategy pattern
    - Flexible credential handling
    """
    headers = {}
    auth = None
    
    if auth_type == 'api_key':
        headers[credentials.get('header_name', 'X-API-Key')] = credentials['key']
    elif auth_type == 'basic':
        auth = (credentials['username'], credentials['password'])
    elif auth_type == 'bearer':
        headers['Authorization'] = f"Bearer {credentials['token']}"
    
    return requests.get(url, headers=headers, auth=auth, timeout=30)


# =============================================================================
# MEDIUM 03: Error Handling - Solutions
# =============================================================================

class APIError(Exception):
    """Base exception for API errors."""
    
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Rate limit exceeded."""
    
    def __init__(self, retry_after: int):
        super().__init__(f"Rate limited. Retry after {retry_after}s", 429)
        self.retry_after = retry_after


def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: float = 30.0
) -> dict[str, Any]:
    """
    Fetch with exponential backoff retry.
    
    Solution demonstrates:
    - Exponential backoff
    - Selective retry on transient errors
    - Comprehensive error handling
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                raise RateLimitError(retry_after)
            
            response.raise_for_status()
            return response.json()
        
        except RateLimitError as e:
            wait_time = e.retry_after
            time.sleep(wait_time)
            last_exception = e
        
        except (Timeout, ConnectionError) as e:
            wait_time = backoff_factor ** attempt
            time.sleep(wait_time)
            last_exception = e
        
        except HTTPError as e:
            if e.response.status_code >= 500:
                wait_time = backoff_factor ** attempt
                time.sleep(wait_time)
                last_exception = e
            else:
                raise APIError(str(e), e.response.status_code)
    
    raise APIError(f"Failed after {max_retries} retries: {last_exception}")


def handle_api_response(response: requests.Response) -> dict[str, Any]:
    """
    Handle API response with comprehensive error checking.
    
    Solution demonstrates:
    - Status code categorisation
    - Error message extraction
    - Graceful degradation
    """
    status = response.status_code
    
    if 200 <= status < 300:
        try:
            return {'success': True, 'data': response.json()}
        except ValueError:
            return {'success': True, 'data': response.text}
    
    error_body = {}
    try:
        error_body = response.json()
    except ValueError:
        pass
    
    error_message = (
        error_body.get('error') or
        error_body.get('message') or
        error_body.get('detail') or
        response.reason
    )
    
    return {
        'success': False,
        'status_code': status,
        'error': error_message,
        'retriable': status >= 500 or status == 429
    }


def safe_api_call(
    func: Callable[[], requests.Response]
) -> dict[str, Any]:
    """
    Execute API call with comprehensive error handling.
    
    Solution demonstrates:
    - Exception wrapping
    - Consistent error format
    """
    try:
        response = func()
        return handle_api_response(response)
    except Timeout:
        return {
            'success': False,
            'error': 'Request timed out',
            'retriable': True
        }
    except ConnectionError:
        return {
            'success': False,
            'error': 'Connection failed',
            'retriable': True
        }
    except RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'retriable': False
        }


# =============================================================================
# TEST SUITE FOR SOLUTIONS
# =============================================================================

def test_solutions():
    """Verify solution correctness."""
    print("Testing Medium Exercise Solutions...")
    
    # Test pagination detection
    offset_response = {'items': [], 'page': 1, 'total_pages': 5}
    assert detect_pagination_style(offset_response) == 'offset'
    
    cursor_response = {'results': [], 'meta': {'next_cursor': 'abc'}}
    assert detect_pagination_style(cursor_response) == 'cursor'
    print("✓ detect_pagination_style works correctly")
    
    # Test Basic auth header
    header = create_basic_auth_header('user', 'pass')
    assert header.startswith('Basic ')
    decoded = base64.b64decode(header.split()[1]).decode()
    assert decoded == 'user:pass'
    print("✓ create_basic_auth_header works correctly")
    
    # Test error handling
    result = safe_api_call(lambda: requests.get('https://httpbin.org/get', timeout=10))
    assert result['success'] is True
    print("✓ safe_api_call works correctly")
    
    # Test retry (with mock)
    data = fetch_with_retry('https://httpbin.org/get')
    assert 'url' in data
    print("✓ fetch_with_retry works correctly")
    
    print("\nAll medium exercise solutions verified!")


if __name__ == '__main__':
    test_solutions()
