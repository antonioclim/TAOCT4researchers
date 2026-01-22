#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Medium 01 - Pagination Handling

Difficulty: ★★★☆☆
Estimated Time: 25 minutes
Learning Objective: LO2

Task:
Implement functions that handle paginated API responses. Modern APIs
often split large result sets across multiple pages. Your implementation
must collect all results while respecting pagination conventions.

Requirements:
1. Detect pagination metadata in responses
2. Iterate through all available pages
3. Handle cursor-based and offset-based pagination
4. Implement early termination when max results reached

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from typing import Any, Iterator, Optional
from dataclasses import dataclass

import requests


@dataclass
class PaginationConfig:
    """Configuration for paginated API requests."""
    
    page_size: int = 100
    max_results: Optional[int] = None
    page_param: str = 'page'
    size_param: str = 'per_page'
    items_key: str = 'items'
    total_key: str = 'total'


def fetch_offset_paginated(
    url: str,
    config: PaginationConfig
) -> Iterator[dict[str, Any]]:
    """
    Iterate through offset-based paginated results.
    
    Offset pagination uses page numbers or skip/limit parameters.
    This function fetches successive pages until exhausted or
    max_results is reached.
    
    Args:
        url: Base API URL
        config: Pagination configuration
        
    Yields:
        Individual items from all pages
        
    Example:
        >>> config = PaginationConfig(page_size=10, max_results=25)
        >>> items = list(fetch_offset_paginated('https://api.example.com/items', config))
        >>> len(items) <= 25
        True
    """
    # TODO: Implement offset-based pagination
    # HINT: Start at page 1, increment until no more results
    # HINT: Track total items yielded to respect max_results
    # HINT: Check if response has fewer items than page_size (last page)
    
    pass  # Replace with your implementation


def fetch_cursor_paginated(
    url: str,
    cursor_param: str = 'cursor',
    initial_cursor: str = '*',
    items_key: str = 'results',
    next_cursor_path: str = 'meta.next_cursor',
    max_results: Optional[int] = None
) -> Iterator[dict[str, Any]]:
    """
    Iterate through cursor-based paginated results.
    
    Cursor pagination uses opaque tokens to track position.
    This is common in APIs that need consistent results during
    iteration (e.g., OpenAlex, Twitter).
    
    Args:
        url: Base API URL
        cursor_param: Query parameter name for cursor
        initial_cursor: Starting cursor value
        items_key: Key containing items in response
        next_cursor_path: Dot-notation path to next cursor
        max_results: Maximum items to return
        
    Yields:
        Individual items from all pages
        
    Example:
        >>> items = list(fetch_cursor_paginated(
        ...     'https://api.openalex.org/works',
        ...     max_results=50
        ... ))
    """
    # TODO: Implement cursor-based pagination
    # HINT: Continue while next_cursor is not None
    # HINT: Use dot-notation path to extract next cursor
    # HINT: Track count to respect max_results
    
    pass  # Replace with your implementation


def detect_pagination_style(response_data: dict[str, Any]) -> str:
    """
    Detect the pagination style used by an API response.
    
    Different APIs use different pagination conventions. This function
    analyses a response to determine which style is in use.
    
    Args:
        response_data: API response dictionary
        
    Returns:
        One of: 'offset', 'cursor', 'link', 'none'
        
    Example:
        >>> detect_pagination_style({'items': [], 'page': 1, 'total_pages': 5})
        'offset'
        >>> detect_pagination_style({'results': [], 'meta': {'next_cursor': 'abc123'}})
        'cursor'
    """
    # TODO: Implement pagination style detection
    # HINT: Check for common pagination indicators
    # - Offset: 'page', 'offset', 'skip', 'total_pages'
    # - Cursor: 'cursor', 'next_cursor', 'continuation_token'
    # - Link: '_links', 'links', 'next' URL
    
    pass  # Replace with your implementation


def collect_all_pages(
    url: str,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    max_pages: int = 100,
    delay: float = 0.1
) -> list[dict[str, Any]]:
    """
    Collect all items from a paginated endpoint.
    
    Automatically detects pagination style and collects all available
    results, with optional rate limiting between requests.
    
    Args:
        url: API endpoint URL
        params: Base query parameters
        headers: Request headers
        max_pages: Safety limit on pages to fetch
        delay: Delay between requests in seconds
        
    Returns:
        List of all items from all pages
        
    Example:
        >>> items = collect_all_pages(
        ...     'https://api.example.com/items',
        ...     params={'filter': 'active'},
        ...     max_pages=10
        ... )
    """
    import time
    
    # TODO: Implement automatic pagination collection
    # HINT: First request to detect pagination style
    # HINT: Use appropriate fetch function based on detected style
    # HINT: Include delay between requests
    
    pass  # Replace with your implementation


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing Pagination Handling exercises...\n')
    
    # Test detect_pagination_style
    print('Test 1: detect_pagination_style')
    try:
        # Offset style
        offset_response = {
            'items': [{'id': 1}],
            'page': 1,
            'total_pages': 5,
            'total': 50
        }
        assert detect_pagination_style(offset_response) == 'offset'
        
        # Cursor style
        cursor_response = {
            'results': [{'id': 1}],
            'meta': {'next_cursor': 'abc123'}
        }
        assert detect_pagination_style(cursor_response) == 'cursor'
        
        # Link style
        link_response = {
            'data': [{'id': 1}],
            '_links': {'next': {'href': '/api/items?page=2'}}
        }
        assert detect_pagination_style(link_response) == 'link'
        
        # No pagination
        simple_response = {'items': [{'id': 1}]}
        assert detect_pagination_style(simple_response) == 'none'
        
        print('  PASSED: Pagination detection works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test pagination config
    print('\nTest 2: PaginationConfig defaults')
    try:
        config = PaginationConfig()
        assert config.page_size == 100
        assert config.max_results is None
        assert config.page_param == 'page'
        print('  PASSED: Config defaults work')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nNote: Full pagination tests require API access.')
    print('Implement functions and test against real APIs.')
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_exercises()
