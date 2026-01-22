#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Solutions for Easy Practice Exercises

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from typing import Any

import requests
from requests.exceptions import RequestException


# =============================================================================
# EASY 01: HTTP Basics - Solutions
# =============================================================================

def fetch_url_info(url: str) -> dict[str, Any]:
    """
    Fetch a URL and return basic response information.
    
    Solution demonstrates:
    - Basic GET request
    - Accessing response properties
    - Timeout handling
    """
    response = requests.get(url, timeout=30)
    
    return {
        'status_code': response.status_code,
        'content_type': response.headers.get('Content-Type', ''),
        'content_length': len(response.content),
        'encoding': response.encoding,
    }


def check_url_exists(url: str) -> bool:
    """
    Check if a URL exists using HEAD request.
    
    Solution demonstrates:
    - HEAD requests for efficiency
    - Status code interpretation
    - Exception handling
    """
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return response.status_code < 400
    except RequestException:
        return False


def get_response_headers(url: str) -> dict[str, str]:
    """
    Retrieve all response headers as a dictionary.
    
    Solution demonstrates:
    - Header access
    - Dictionary conversion
    """
    response = requests.head(url, timeout=10)
    return dict(response.headers)


# =============================================================================
# EASY 02: GET with JSON - Solutions
# =============================================================================

def search_with_params(base_url: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Make GET request with query parameters.
    
    Solution demonstrates:
    - Query parameter passing
    - JSON response parsing
    """
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_nested_value(data: dict[str, Any], path: str) -> Any:
    """
    Extract value from nested dictionary using dot notation.
    
    Solution demonstrates:
    - Recursive dictionary traversal
    - Safe key access
    """
    keys = path.split('.')
    current = data
    
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and key.isdigit():
            index = int(key)
            current = current[index] if index < len(current) else None
        else:
            return None
        
        if current is None:
            return None
    
    return current


def fetch_and_extract(url: str, json_path: str) -> Any:
    """
    Fetch URL and extract specific value from JSON response.
    
    Solution demonstrates:
    - Combining HTTP request with data extraction
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    return extract_nested_value(data, json_path)


def parse_api_list_response(
    data: list[dict[str, Any]],
    key_field: str
) -> dict[str, dict[str, Any]]:
    """
    Convert list response to lookup dictionary.
    
    Solution demonstrates:
    - Dictionary comprehension
    - Data transformation
    """
    return {item[key_field]: item for item in data if key_field in item}


# =============================================================================
# EASY 03: POST Requests - Solutions
# =============================================================================

def create_resource(
    url: str,
    data: dict[str, Any],
    auth_header: str | None = None
) -> dict[str, Any]:
    """
    Create a resource using POST request.
    
    Solution demonstrates:
    - POST with JSON body
    - Authorization header
    - Status code checking
    """
    headers = {'Content-Type': 'application/json'}
    if auth_header:
        headers['Authorization'] = auth_header
    
    response = requests.post(url, json=data, headers=headers, timeout=30)
    response.raise_for_status()
    
    return {
        'status_code': response.status_code,
        'data': response.json() if response.content else {},
        'location': response.headers.get('Location'),
    }


def post_form_data(url: str, form_data: dict[str, str]) -> dict[str, Any]:
    """
    Submit form data using POST.
    
    Solution demonstrates:
    - Form-encoded data
    - Different content types
    """
    response = requests.post(url, data=form_data, timeout=30)
    response.raise_for_status()
    return response.json()


def upload_json_file(url: str, json_content: dict[str, Any]) -> int:
    """
    Upload JSON content to endpoint.
    
    Solution demonstrates:
    - JSON serialisation
    - Response status handling
    """
    response = requests.post(
        url,
        json=json_content,
        headers={'Content-Type': 'application/json'},
        timeout=30
    )
    return response.status_code


# =============================================================================
# TEST SUITE FOR SOLUTIONS
# =============================================================================

def test_solutions():
    """Verify solution correctness."""
    print("Testing Easy Exercise Solutions...")
    
    # Test fetch_url_info
    info = fetch_url_info('https://httpbin.org/get')
    assert info['status_code'] == 200
    assert 'application/json' in info['content_type']
    print("✓ fetch_url_info works correctly")
    
    # Test check_url_exists
    assert check_url_exists('https://httpbin.org/status/200') is True
    assert check_url_exists('https://httpbin.org/status/404') is False
    print("✓ check_url_exists works correctly")
    
    # Test extract_nested_value
    test_data = {'a': {'b': {'c': 42}}}
    assert extract_nested_value(test_data, 'a.b.c') == 42
    assert extract_nested_value(test_data, 'a.x.c') is None
    print("✓ extract_nested_value works correctly")
    
    # Test parse_api_list_response
    items = [{'id': 1, 'name': 'A'}, {'id': 2, 'name': 'B'}]
    lookup = parse_api_list_response(items, 'id')
    assert lookup[1]['name'] == 'A'
    assert lookup[2]['name'] == 'B'
    print("✓ parse_api_list_response works correctly")
    
    # Test create_resource
    result = create_resource(
        'https://httpbin.org/post',
        {'name': 'test'}
    )
    assert result['status_code'] == 200
    print("✓ create_resource works correctly")
    
    print("\nAll easy exercise solutions verified!")


if __name__ == '__main__':
    test_solutions()
