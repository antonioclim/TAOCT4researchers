#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Lab 12.01: API Consumption and HTTP Fundamentals

This laboratory develops proficiency in programmatic data acquisition from web
services. Through progressive exercises, you will master HTTP fundamentals,
REST patterns, authentication mechanisms and robust client implementation.

Duration: 50 minutes
Difficulty: ★★★★☆

Learning Objectives Addressed:
- LO1: Understand HTTP protocol and REST principles
- LO2: Apply API consumption techniques using requests
- LO3: Implement authentication mechanisms
- LO4: Handle API errors and implement retry logic

Prerequisites:
- 09UNIT: Exception handling
- 10UNIT: JSON serialisation

Required Packages:
    pip install requests

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Course: Computational Thinking for Researchers
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, RequestException, Timeout
from urllib3.util.retry import Retry

# Configure logging for API interactions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: HTTP FUNDAMENTALS
# Duration: ~10 minutes
# Learning Objective: LO1
# =============================================================================

def inspect_http_response(url: str) -> dict[str, Any]:
    """
    Inspect all components of an HTTP response.
    
    This function demonstrates the anatomy of an HTTP response, including
    status code, headers, encoding and content. Understanding these
    components is essential for effective API interaction.
    
    Args:
        url: The URL to request
        
    Returns:
        Dictionary containing response components
        
    Example:
        >>> info = inspect_http_response('https://httpbin.org/get')
        >>> print(info['status_code'])
        200
        >>> print('content-type' in info['headers'])
        True
    """
    # TODO: Make a GET request to the provided URL
    # HINT: Use requests.get() with a timeout
    response = ...  # Your code here
    
    # TODO: Extract response components
    result = {
        'status_code': ...,           # Integer status code
        'status_text': ...,           # e.g., 'OK', 'Not Found'
        'headers': ...,               # Dictionary of response headers
        'content_type': ...,          # Content-Type header value
        'encoding': ...,              # Response encoding
        'elapsed_ms': ...,            # Request duration in milliseconds
        'url': ...,                   # Final URL (after redirects)
        'is_redirect': ...,           # Whether response is a redirect
    }
    
    return result


def demonstrate_http_methods(base_url: str = 'https://httpbin.org') -> dict[str, int]:
    """
    Demonstrate different HTTP methods and their status codes.
    
    httpbin.org echoes back request information, making it ideal for
    exploring HTTP method behaviour. This function exercises the four
    primary REST methods.
    
    Args:
        base_url: Base URL for httpbin service
        
    Returns:
        Dictionary mapping method names to response status codes
        
    Example:
        >>> results = demonstrate_http_methods()
        >>> results['GET']
        200
        >>> results['POST']
        200
    """
    results = {}
    
    # TODO: Implement GET request
    # GET retrieves resource representation without side effects
    response = requests.get(f'{base_url}/get', timeout=10)
    results['GET'] = response.status_code
    
    # TODO: Implement POST request with JSON body
    # POST creates a new resource
    data = {'name': 'Research Dataset', 'year': 2024}
    response = ...  # Your code here
    results['POST'] = response.status_code
    
    # TODO: Implement PUT request with JSON body
    # PUT replaces a resource entirely
    response = ...  # Your code here
    results['PUT'] = response.status_code
    
    # TODO: Implement DELETE request
    # DELETE removes a resource
    response = ...  # Your code here
    results['DELETE'] = response.status_code
    
    return results


@dataclass
class HTTPResponseAnalyser:
    """
    Analyse HTTP responses for common patterns and issues.
    
    This class encapsulates response analysis logic, identifying successful
    responses, client errors, server errors and redirect chains.
    
    Attributes:
        response: The requests Response object to analyse
    """
    
    response: requests.Response
    
    @property
    def is_success(self) -> bool:
        """Check if response indicates success (2xx status)."""
        return 200 <= self.response.status_code < 300
    
    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error (4xx status)."""
        return 400 <= self.response.status_code < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error (5xx status)."""
        return 500 <= self.response.status_code < 600
    
    @property
    def is_rate_limited(self) -> bool:
        """Check if response indicates rate limiting (429 status)."""
        return self.response.status_code == 429
    
    def get_rate_limit_info(self) -> dict[str, Optional[int]]:
        """
        Extract rate limit information from response headers.
        
        Many APIs include rate limit headers indicating quota status.
        Common header patterns include X-RateLimit-* and RateLimit-*.
        
        Returns:
            Dictionary with limit, remaining and reset timestamp
        """
        headers = self.response.headers
        
        # TODO: Extract rate limit headers
        # HINT: Try both X-RateLimit-* and RateLimit-* patterns
        return {
            'limit': ...,       # Total allowed requests
            'remaining': ...,   # Requests remaining in window
            'reset': ...,       # Unix timestamp when limit resets
        }
    
    def get_retry_after(self) -> Optional[int]:
        """
        Parse Retry-After header if present.
        
        The Retry-After header indicates how long to wait before retrying.
        It may contain seconds (integer) or an HTTP date.
        
        Returns:
            Seconds to wait, or None if header not present
        """
        retry_after = self.response.headers.get('Retry-After')
        if retry_after is None:
            return None
        
        # TODO: Parse retry-after value
        # HINT: It might be an integer or a date string
        try:
            return int(retry_after)
        except ValueError:
            # Parse HTTP date format
            # e.g., "Wed, 21 Oct 2024 07:28:00 GMT"
            pass
        
        return None


# =============================================================================
# SECTION 2: REST PATTERNS AND PAGINATION
# Duration: ~15 minutes
# Learning Objective: LO2
# =============================================================================

@dataclass
class PaginatedAPIClient:
    """
    Client for consuming paginated REST APIs.
    
    Many APIs return results in pages to manage response sizes. This client
    abstracts pagination logic, yielding items from all pages transparently.
    
    Attributes:
        base_url: API base URL
        default_page_size: Number of items per page
        session: Requests session for connection reuse
    """
    
    base_url: str
    default_page_size: int = 100
    session: requests.Session = field(default_factory=requests.Session)
    
    def __post_init__(self) -> None:
        """Configure session with default headers."""
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'ResearchBot/1.0 (Educational; +https://ase.ro)'
        })
    
    def fetch_page(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        page: int = 1
    ) -> dict[str, Any]:
        """
        Fetch a single page of results.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            page: Page number (1-indexed)
            
        Returns:
            JSON response as dictionary
            
        Raises:
            HTTPError: If request fails
        """
        url = urljoin(self.base_url, endpoint)
        request_params = params.copy() if params else {}
        request_params['page'] = page
        request_params['per_page'] = self.default_page_size
        
        response = self.session.get(url, params=request_params, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def fetch_all_pages(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        max_pages: Optional[int] = None
    ) -> Iterator[dict[str, Any]]:
        """
        Iterate through all pages of results.
        
        This generator fetches pages lazily, yielding individual items
        from each page. Pagination continues until no more results or
        max_pages is reached.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            max_pages: Maximum pages to fetch (None for all)
            
        Yields:
            Individual items from paginated results
            
        Example:
            >>> client = PaginatedAPIClient('https://api.example.com')
            >>> for item in client.fetch_all_pages('/users'):
            ...     print(item['name'])
        """
        page = 1
        
        while max_pages is None or page <= max_pages:
            # TODO: Fetch the current page
            data = self.fetch_page(endpoint, params, page)
            
            # TODO: Extract items from response
            # HINT: Common patterns include 'results', 'items', 'data'
            items = data.get('results', data.get('items', data.get('data', [])))
            
            if not items:
                break
            
            yield from items
            
            # TODO: Check for next page
            # HINT: Look for 'next', 'has_more', or compare with total
            if not self._has_next_page(data):
                break
            
            page += 1
    
    def _has_next_page(self, response_data: dict[str, Any]) -> bool:
        """
        Determine if more pages exist.
        
        APIs indicate pagination status differently. This method checks
        common patterns including next links, has_more flags and
        total counts.
        """
        # Check for explicit next link
        if response_data.get('next') or response_data.get('links', {}).get('next'):
            return True
        
        # Check for has_more flag
        if response_data.get('has_more', False):
            return True
        
        # Check if we've reached total
        meta = response_data.get('meta', {})
        if 'total' in meta and 'page' in meta:
            current = meta['page'] * self.default_page_size
            return current < meta['total']
        
        return False


def fetch_crossref_works(
    query: str,
    rows: int = 100,
    filter_params: Optional[dict[str, str]] = None
) -> list[dict[str, Any]]:
    """
    Search CrossRef for scholarly works matching a query.
    
    CrossRef provides authoritative metadata for over 130 million scholarly
    works. This function demonstrates consuming a production research API.
    
    Args:
        query: Search query string
        rows: Maximum results to return
        filter_params: Optional filters (e.g., {'from-pub-date': '2023'})
        
    Returns:
        List of work metadata dictionaries
        
    Example:
        >>> works = fetch_crossref_works('machine learning', rows=10)
        >>> len(works) <= 10
        True
        >>> all('DOI' in w for w in works)
        True
    """
    url = 'https://api.crossref.org/works'
    
    params = {
        'query': query,
        'rows': rows,
    }
    
    # Add filters if provided
    if filter_params:
        filter_string = ','.join(f'{k}:{v}' for k, v in filter_params.items())
        params['filter'] = filter_string
    
    # TODO: Configure polite request headers
    # HINT: Include mailto in User-Agent for polite pool access
    headers = {
        'User-Agent': ...,  # Your code here
    }
    
    # TODO: Make request and handle response
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    
    # TODO: Extract works from response structure
    data = response.json()
    works = ...  # Your code here
    
    return works


# =============================================================================
# SECTION 3: AUTHENTICATION MECHANISMS
# Duration: ~10 minutes
# Learning Objective: LO3
# =============================================================================

@dataclass
class APIKeyAuthenticator:
    """
    Handle API key authentication for various placement strategies.
    
    APIs accept keys in different locations: headers, query parameters
    or HTTP Basic authentication. This class abstracts these variations.
    
    Attributes:
        api_key: The API key string
        key_name: Header or parameter name for the key
        placement: Where to place key ('header', 'query', 'basic')
    """
    
    api_key: str
    key_name: str = 'X-API-Key'
    placement: str = 'header'
    
    def apply_to_request(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None
    ) -> tuple[str, dict, dict]:
        """
        Apply authentication to request components.
        
        Args:
            url: Request URL
            params: Query parameters
            headers: Request headers
            
        Returns:
            Tuple of (url, params, headers) with auth applied
        """
        params = params.copy() if params else {}
        headers = headers.copy() if headers else {}
        
        if self.placement == 'header':
            headers[self.key_name] = self.api_key
        elif self.placement == 'query':
            params[self.key_name] = self.api_key
        elif self.placement == 'basic':
            # Basic auth encodes key as username with empty password
            import base64
            credentials = base64.b64encode(
                f'{self.api_key}:'.encode()
            ).decode()
            headers['Authorization'] = f'Basic {credentials}'
        
        return url, params, headers


@dataclass
class OAuth2ClientCredentials:
    """
    Implement OAuth 2.0 Client Credentials flow.
    
    The Client Credentials grant type is used for machine-to-machine
    authentication where no user context is required. The client
    exchanges its credentials for an access token.
    
    Attributes:
        token_url: OAuth token endpoint
        client_id: Application client ID
        client_secret: Application client secret
        scope: Requested permission scopes
    """
    
    token_url: str
    client_id: str
    client_secret: str
    scope: str = ''
    _access_token: Optional[str] = field(default=None, repr=False)
    _token_expiry: Optional[datetime] = field(default=None, repr=False)
    
    def get_access_token(self) -> str:
        """
        Obtain or return cached access token.
        
        Tokens are cached until expiry to avoid unnecessary token requests.
        A buffer of 60 seconds ensures tokens are refreshed before expiry.
        
        Returns:
            Valid access token string
            
        Raises:
            HTTPError: If token request fails
        """
        # Check if cached token is still valid
        if self._access_token and self._token_expiry:
            buffer = timedelta(seconds=60)
            if datetime.now() < self._token_expiry - buffer:
                return self._access_token
        
        # TODO: Request new access token
        # HINT: POST to token_url with grant_type=client_credentials
        response = requests.post(
            self.token_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': self.scope,
            },
            timeout=30
        )
        response.raise_for_status()
        
        # TODO: Parse token response
        token_data = response.json()
        self._access_token = token_data['access_token']
        
        # Calculate expiry time
        expires_in = token_data.get('expires_in', 3600)
        self._token_expiry = datetime.now() + timedelta(seconds=expires_in)
        
        return self._access_token
    
    def get_auth_header(self) -> dict[str, str]:
        """
        Get Authorization header with Bearer token.
        
        Returns:
            Dictionary with Authorization header
        """
        token = self.get_access_token()
        return {'Authorization': f'Bearer {token}'}


def demonstrate_authentication(
    api_key: Optional[str] = None
) -> dict[str, Any]:
    """
    Demonstrate various authentication mechanisms.
    
    This function shows how to authenticate with APIs using different
    mechanisms. Environment variables are used for credential storage.
    
    Args:
        api_key: Optional API key (defaults to environment variable)
        
    Returns:
        Dictionary of authentication test results
    """
    # TODO: Load API key from environment if not provided
    # HINT: Use os.environ.get() with a sensible default
    api_key = api_key or os.environ.get('DEMO_API_KEY', 'test_key')
    
    results = {}
    
    # Test header-based authentication
    auth_header = APIKeyAuthenticator(api_key, 'X-API-Key', 'header')
    url, params, headers = auth_header.apply_to_request(
        'https://httpbin.org/headers'
    )
    response = requests.get(url, params=params, headers=headers, timeout=10)
    results['header_auth'] = 'X-Api-Key' in response.json().get('headers', {})
    
    # Test query parameter authentication
    auth_query = APIKeyAuthenticator(api_key, 'api_key', 'query')
    url, params, headers = auth_query.apply_to_request(
        'https://httpbin.org/get'
    )
    response = requests.get(url, params=params, headers=headers, timeout=10)
    results['query_auth'] = 'api_key' in response.json().get('args', {})
    
    # Test Basic authentication
    auth_basic = APIKeyAuthenticator(api_key, '', 'basic')
    url, params, headers = auth_basic.apply_to_request(
        'https://httpbin.org/headers'
    )
    response = requests.get(url, params=params, headers=headers, timeout=10)
    results['basic_auth'] = 'Authorization' in response.json().get('headers', {})
    
    return results


# =============================================================================
# SECTION 4: RESEARCH API INTEGRATION
# Duration: ~10 minutes
# Learning Objective: LO2
# =============================================================================

@dataclass
class OpenAlexClient:
    """
    Client for the OpenAlex scholarly knowledge graph.
    
    OpenAlex provides free, open access to scholarly metadata including
    works, authors, institutions, concepts and venues. This client
    implements common query patterns for bibliometric research.
    
    Attributes:
        base_url: OpenAlex API base URL
        email: Contact email for polite pool access
        session: Requests session for connection reuse
    """
    
    base_url: str = 'https://api.openalex.org'
    email: Optional[str] = None
    session: requests.Session = field(default_factory=requests.Session)
    
    def __post_init__(self) -> None:
        """Configure session with polite headers."""
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'ResearchBot/1.0 (Educational)'
        }
        if self.email:
            headers['User-Agent'] += f' mailto:{self.email}'
        self.session.headers.update(headers)
    
    def get_work(self, work_id: str) -> dict[str, Any]:
        """
        Retrieve a single work by OpenAlex ID or DOI.
        
        Args:
            work_id: OpenAlex ID (W1234567890) or DOI
            
        Returns:
            Work metadata dictionary
        """
        # Handle DOI format
        if work_id.startswith('10.'):
            work_id = f'doi:{work_id}'
        
        url = f'{self.base_url}/works/{work_id}'
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def search_works(
        self,
        query: str,
        filters: Optional[dict[str, str]] = None,
        per_page: int = 100,
        max_results: Optional[int] = None
    ) -> Iterator[dict[str, Any]]:
        """
        Search for works matching query and filters.
        
        Args:
            query: Search query string
            filters: Filter parameters (e.g., {'publication_year': '2023'})
            per_page: Results per page
            max_results: Maximum total results (None for all)
            
        Yields:
            Work metadata dictionaries
            
        Example:
            >>> client = OpenAlexClient(email='user@example.com')
            >>> works = list(client.search_works('climate change', max_results=10))
            >>> len(works) <= 10
            True
        """
        params = {
            'search': query,
            'per-page': per_page,
        }
        
        # TODO: Build filter string from dictionary
        if filters:
            filter_parts = [f'{k}:{v}' for k, v in filters.items()]
            params['filter'] = ','.join(filter_parts)
        
        cursor = '*'
        count = 0
        
        while cursor:
            params['cursor'] = cursor
            
            response = self.session.get(
                f'{self.base_url}/works',
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            
            for work in data.get('results', []):
                yield work
                count += 1
                
                if max_results and count >= max_results:
                    return
            
            # Get next cursor
            cursor = data.get('meta', {}).get('next_cursor')
    
    def get_author(self, author_id: str) -> dict[str, Any]:
        """
        Retrieve author information by OpenAlex ID or ORCID.
        
        Args:
            author_id: OpenAlex ID (A1234567890) or ORCID
            
        Returns:
            Author metadata dictionary
        """
        # TODO: Handle ORCID format
        if author_id.startswith('0000-'):
            author_id = f'orcid:{author_id}'
        
        url = f'{self.base_url}/authors/{author_id}'
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        return response.json()
    
    def get_author_works(
        self,
        author_id: str,
        max_results: Optional[int] = None
    ) -> Iterator[dict[str, Any]]:
        """
        Retrieve all works by a specific author.
        
        Args:
            author_id: OpenAlex author ID
            max_results: Maximum works to retrieve
            
        Yields:
            Work metadata dictionaries
        """
        yield from self.search_works(
            query='',
            filters={'author.id': author_id},
            max_results=max_results
        )


# =============================================================================
# SECTION 5: ROBUST API CLIENT IMPLEMENTATION
# Duration: ~5 minutes
# Learning Objective: LO4
# =============================================================================

@dataclass
class RobustAPIClient:
    """
    Production-quality API client with retry logic and rate limiting.
    
    This client implements established patterns for reliable API consumption:
    - Exponential backoff retry for transient failures
    - Rate limit detection and compliance
    - Response caching to reduce redundant requests
    - Comprehensive logging for debugging
    
    Attributes:
        base_url: API base URL
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        rate_limit_delay: Default delay when rate limited
    """
    
    base_url: str
    max_retries: int = 3
    backoff_factor: float = 2.0
    rate_limit_delay: float = 60.0
    _session: Optional[requests.Session] = field(default=None, repr=False)
    
    @property
    def session(self) -> requests.Session:
        """Get or create session with retry configuration."""
        if self._session is None:
            self._session = requests.Session()
            
            # Configure automatic retry for certain status codes
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=self.backoff_factor,
                status_forcelist=[500, 502, 503, 504],
                allowed_methods=['GET', 'POST', 'PUT', 'DELETE']
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self._session.mount('http://', adapter)
            self._session.mount('https://', adapter)
            
            self._session.headers.update({
                'Accept': 'application/json',
                'User-Agent': 'RobustAPIClient/1.0'
            })
        
        return self._session
    
    def request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> requests.Response:
        """
        Make HTTP request with comprehensive error handling.
        
        This method wraps requests with:
        - Rate limit detection and waiting
        - Logging of all requests and responses
        - Timeout enforcement
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response object
            
        Raises:
            HTTPError: For non-retriable errors
            RequestException: For network failures after retries
        """
        url = urljoin(self.base_url, endpoint)
        kwargs.setdefault('timeout', 30)
        
        logger.info(f'{method} {url}')
        
        try:
            response = self.session.request(method, url, **kwargs)
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = self._get_retry_after(response)
                logger.warning(f'Rate limited. Waiting {retry_after}s')
                time.sleep(retry_after)
                return self.request(method, endpoint, **kwargs)
            
            response.raise_for_status()
            logger.info(f'Response: {response.status_code}')
            
            return response
            
        except Timeout:
            logger.error(f'Timeout: {url}')
            raise
        except HTTPError as e:
            logger.error(f'HTTP Error: {e.response.status_code} - {url}')
            raise
        except RequestException as e:
            logger.error(f'Request failed: {e}')
            raise
    
    def _get_retry_after(self, response: requests.Response) -> float:
        """Extract retry delay from response headers."""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return self.rate_limit_delay
    
    def get(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """Make GET request."""
        return self.request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """Make POST request."""
        return self.request('POST', endpoint, **kwargs)


@dataclass
class ResponseCache:
    """
    File-based cache for API responses.
    
    Caching reduces redundant API requests, improving performance and
    respecting rate limits. This implementation uses file-based storage
    with configurable time-to-live.
    
    Attributes:
        cache_dir: Directory for cache files
        default_ttl: Default time-to-live for cached responses
    """
    
    cache_dir: Path
    default_ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    def __post_init__(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, url: str, params: Optional[dict] = None) -> str:
        """Generate cache key from request parameters."""
        content = json.dumps({
            'url': url,
            'params': params or {}
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, url: str, params: Optional[dict] = None) -> Optional[Any]:
        """
        Retrieve cached response if valid.
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            Cached response data or None if not cached/expired
        """
        key = self._cache_key(url, params)
        cache_file = self.cache_dir / f'{key}.json'
        
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text())
            expires = datetime.fromisoformat(data['expires'])
            
            if datetime.now() < expires:
                logger.debug(f'Cache hit: {url}')
                return data['response']
            else:
                logger.debug(f'Cache expired: {url}')
                cache_file.unlink()
                
        except (json.JSONDecodeError, KeyError):
            cache_file.unlink()
        
        return None
    
    def set(
        self,
        url: str,
        params: Optional[dict],
        response: Any,
        ttl: Optional[timedelta] = None
    ) -> None:
        """
        Store response in cache.
        
        Args:
            url: Request URL
            params: Query parameters
            response: Response data to cache
            ttl: Time-to-live (uses default if not specified)
        """
        key = self._cache_key(url, params)
        cache_file = self.cache_dir / f'{key}.json'
        
        expires = datetime.now() + (ttl or self.default_ttl)
        
        data = {
            'url': url,
            'params': params,
            'response': response,
            'expires': expires.isoformat(),
            'cached_at': datetime.now().isoformat()
        }
        
        cache_file.write_text(json.dumps(data, indent=2))
        logger.debug(f'Cached: {url}')


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Run laboratory exercises demonstrating API consumption."""
    print('=' * 70)
    print('12UNIT: Web APIs and Data Acquisition')
    print('Lab 12.01: API Consumption and HTTP Fundamentals')
    print('=' * 70)
    
    # Section 1: HTTP Fundamentals
    print('\n--- Section 1: HTTP Fundamentals ---')
    print('Testing HTTP methods...')
    try:
        methods = demonstrate_http_methods()
        print(f'HTTP Methods tested: {methods}')
    except RequestException as e:
        print(f'HTTP test failed: {e}')
    
    # Section 2: REST Patterns
    print('\n--- Section 2: REST Patterns ---')
    print('Searching CrossRef for recent publications...')
    try:
        works = fetch_crossref_works(
            'computational thinking',
            rows=5,
            filter_params={'from-pub-date': '2023'}
        )
        print(f'Found {len(works)} works')
        for work in works[:3]:
            title = work.get('title', ['Unknown'])[0][:60]
            print(f'  - {title}...')
    except RequestException as e:
        print(f'CrossRef search failed: {e}')
    
    # Section 3: Authentication
    print('\n--- Section 3: Authentication ---')
    print('Testing authentication mechanisms...')
    try:
        auth_results = demonstrate_authentication()
        for mechanism, success in auth_results.items():
            status = 'OK' if success else 'FAILED'
            print(f'  {mechanism}: {status}')
    except RequestException as e:
        print(f'Authentication test failed: {e}')
    
    # Section 4: Research APIs
    print('\n--- Section 4: Research APIs ---')
    print('Querying OpenAlex...')
    try:
        client = OpenAlexClient(email='student@ase.ro')
        works = list(client.search_works('machine learning', max_results=3))
        print(f'Found {len(works)} works from OpenAlex')
    except RequestException as e:
        print(f'OpenAlex query failed: {e}')
    
    print('\n' + '=' * 70)
    print('Laboratory exercises complete.')
    print('Review TODOs and implement missing functionality.')
    print('=' * 70)


if __name__ == '__main__':
    main()
