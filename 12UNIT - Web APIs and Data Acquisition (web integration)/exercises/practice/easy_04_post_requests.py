#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Easy 03 - POST Requests and Data Submission

Difficulty: ★☆☆☆☆
Estimated Time: 15 minutes
Learning Objective: LO2

Task:
Implement functions that submit data to APIs using POST requests.
This exercise builds skills in sending structured data and handling
creation responses.

Requirements:
1. Send JSON data in POST requests
2. Handle creation responses (201 Created)
3. Extract created resource information from responses

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

from typing import Any, Optional

import requests


def post_json_data(
    url: str,
    data: dict[str, Any],
    headers: Optional[dict[str, str]] = None
) -> tuple[int, Optional[dict[str, Any]]]:
    """
    Send JSON data via POST request.
    
    Args:
        url: The endpoint URL
        data: Dictionary to send as JSON
        headers: Optional additional headers
        
    Returns:
        Tuple of (status_code, response_json or None)
        
    Example:
        >>> status, response = post_json_data(
        ...     'https://httpbin.org/post',
        ...     {'name': 'Test', 'value': 42}
        ... )
        >>> status
        200
        >>> response['json']['name']
        'Test'
    """
    # TODO: Implement this function
    # HINT: Use requests.post(url, json=data, headers=headers)
    # HINT: Handle exceptions and return (0, None) on failure
    
    pass  # Replace with your implementation


def create_resource(
    base_url: str,
    resource_type: str,
    resource_data: dict[str, Any]
) -> Optional[str]:
    """
    Create a new resource and return its ID.
    
    This function follows REST conventions for resource creation:
    POST to collection endpoint, expect 201 response with ID.
    
    Args:
        base_url: API base URL
        resource_type: Resource collection name (e.g., 'users', 'posts')
        resource_data: Data for the new resource
        
    Returns:
        ID of created resource or None if creation failed
        
    Example:
        >>> resource_id = create_resource(
        ...     'https://jsonplaceholder.typicode.com',
        ...     'posts',
        ...     {'title': 'Test', 'body': 'Content', 'userId': 1}
        ... )
        >>> resource_id is not None
        True
    """
    # TODO: Implement this function
    # HINT: Construct URL as {base_url}/{resource_type}
    # HINT: Look for 'id' in response JSON
    
    pass  # Replace with your implementation


def submit_form_data(
    url: str,
    form_fields: dict[str, str]
) -> Optional[dict[str, Any]]:
    """
    Submit form-encoded data via POST.
    
    Some APIs expect form-encoded data rather than JSON.
    This uses application/x-www-form-urlencoded content type.
    
    Args:
        url: The endpoint URL
        form_fields: Dictionary of form field values
        
    Returns:
        Response JSON or None if request failed
        
    Example:
        >>> result = submit_form_data(
        ...     'https://httpbin.org/post',
        ...     {'username': 'test', 'password': 'secret'}
        ... )
        >>> 'username' in result['form']
        True
    """
    # TODO: Implement this function
    # HINT: Use requests.post(url, data=form_fields) for form encoding
    # HINT: This differs from json= parameter
    
    pass  # Replace with your implementation


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing POST Requests exercises...\n')
    
    # Test post_json_data
    print('Test 1: post_json_data')
    try:
        status, response = post_json_data(
            'https://httpbin.org/post',
            {'test': 'value', 'number': 123}
        )
        assert status == 200, f'Expected 200, got {status}'
        assert response is not None, 'Expected response data'
        assert response['json']['test'] == 'value'
        print('  PASSED: JSON POST works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test create_resource
    print('\nTest 2: create_resource')
    try:
        resource_id = create_resource(
            'https://jsonplaceholder.typicode.com',
            'posts',
            {'title': 'Test Post', 'body': 'Content', 'userId': 1}
        )
        assert resource_id is not None, 'Expected resource ID'
        print(f'  PASSED: Created resource with ID {resource_id}')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test submit_form_data
    print('\nTest 3: submit_form_data')
    try:
        result = submit_form_data(
            'https://httpbin.org/post',
            {'field1': 'value1', 'field2': 'value2'}
        )
        assert result is not None, 'Expected response'
        assert 'field1' in result.get('form', {}), 'Expected form data'
        print('  PASSED: Form submission works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_exercises()
