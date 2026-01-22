#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Easy 02 - GET Requests and JSON Parsing

Difficulty: ★☆☆☆☆
Estimated Time: 15 minutes
Learning Objective: LO2

Task:
Implement functions that make GET requests with query parameters
and parse JSON responses. This exercise develops skills in
constructing API requests and handling structured data.

Requirements:
1. Make GET requests with query parameters
2. Parse JSON responses and extract specific fields
3. Handle missing or malformed data gracefully

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from typing import Any, Optional

import requests


def search_with_params(
    base_url: str,
    params: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """
    Make a GET request with query parameters.
    
    Args:
        base_url: The base URL to request
        params: Dictionary of query parameters
        
    Returns:
        Parsed JSON response or None if request failed
        
    Example:
        >>> result = search_with_params(
        ...     'https://httpbin.org/get',
        ...     {'search': 'test', 'limit': 10}
        ... )
        >>> result['args']['search']
        'test'
    """
    # TODO: Implement this function
    # HINT: Use requests.get(url, params=params)
    # HINT: Use response.json() to parse JSON
    
    pass  # Replace with your implementation


def extract_nested_value(
    data: dict[str, Any],
    path: str,
    default: Any = None
) -> Any:
    """
    Extract a value from nested dictionary using dot notation.
    
    Args:
        data: Dictionary to extract from
        path: Dot-separated path (e.g., 'response.data.items')
        default: Value to return if path not found
        
    Returns:
        Value at path or default if not found
        
    Example:
        >>> data = {'response': {'data': {'count': 42}}}
        >>> extract_nested_value(data, 'response.data.count')
        42
        >>> extract_nested_value(data, 'response.missing', 'N/A')
        'N/A'
    """
    # TODO: Implement this function
    # HINT: Split path by '.' and traverse dictionary
    # HINT: Use dict.get() for safe access
    
    pass  # Replace with your implementation


def fetch_and_extract(
    url: str,
    params: dict[str, Any],
    extract_path: str
) -> Optional[Any]:
    """
    Fetch JSON from URL and extract value at specified path.
    
    Combines HTTP request with nested value extraction.
    
    Args:
        url: URL to fetch
        params: Query parameters
        extract_path: Dot-notation path to extract
        
    Returns:
        Extracted value or None if request failed or path not found
        
    Example:
        >>> value = fetch_and_extract(
        ...     'https://httpbin.org/get',
        ...     {'key': 'value'},
        ...     'args.key'
        ... )
        >>> value
        'value'
    """
    # TODO: Implement this function
    # HINT: Combine search_with_params() and extract_nested_value()
    
    pass  # Replace with your implementation


def parse_api_list_response(
    response_data: dict[str, Any],
    items_key: str = 'items',
    id_field: str = 'id'
) -> dict[str, Any]:
    """
    Parse a typical API list response into a lookup dictionary.
    
    Many APIs return lists of items with IDs. This function converts
    such responses into dictionaries keyed by ID for efficient lookup.
    
    Args:
        response_data: Raw API response dictionary
        items_key: Key containing the list of items
        id_field: Field name for item IDs
        
    Returns:
        Dictionary mapping IDs to items
        
    Example:
        >>> data = {'items': [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]}
        >>> result = parse_api_list_response(data)
        >>> result[1]['name']
        'A'
    """
    # TODO: Implement this function
    # HINT: Extract items list using items_key
    # HINT: Build dict comprehension with id_field as key
    
    pass  # Replace with your implementation


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing GET Requests and JSON Parsing exercises...\n')
    
    # Test search_with_params
    print('Test 1: search_with_params')
    try:
        result = search_with_params(
            'https://httpbin.org/get',
            {'query': 'test', 'limit': 5}
        )
        assert result is not None, 'Expected response'
        assert result['args']['query'] == 'test', 'Expected query param'
        print('  PASSED: Parameterised request works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test extract_nested_value
    print('\nTest 2: extract_nested_value')
    try:
        data = {'a': {'b': {'c': 'value'}}}
        assert extract_nested_value(data, 'a.b.c') == 'value'
        assert extract_nested_value(data, 'a.b.d', 'default') == 'default'
        assert extract_nested_value(data, 'x.y.z') is None
        print('  PASSED: Nested extraction works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test fetch_and_extract
    print('\nTest 3: fetch_and_extract')
    try:
        value = fetch_and_extract(
            'https://httpbin.org/get',
            {'test': 'hello'},
            'args.test'
        )
        assert value == 'hello', f'Expected "hello", got "{value}"'
        print('  PASSED: Fetch and extract works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test parse_api_list_response
    print('\nTest 4: parse_api_list_response')
    try:
        data = {
            'items': [
                {'id': 1, 'name': 'First'},
                {'id': 2, 'name': 'Second'}
            ]
        }
        result = parse_api_list_response(data)
        assert 1 in result, 'Expected ID 1 in result'
        assert result[1]['name'] == 'First'
        print('  PASSED: List response parsing works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_exercises()
