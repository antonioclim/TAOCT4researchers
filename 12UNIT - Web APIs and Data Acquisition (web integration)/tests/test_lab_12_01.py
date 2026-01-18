#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Tests for Lab 12.01: API Consumption

This module tests the API consumption functionality including
HTTP fundamentals, pagination, authentication and error handling.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lab'))

from lab_12_01_api_consumption import (
    HTTPResponseAnalyser,
    PaginatedAPIClient,
    APIKeyAuthenticator,
    OAuth2ClientCredentials,
    RobustAPIClient,
    ResponseCache,
    OpenAlexClient,
    fetch_crossref_works,
    demonstrate_http_methods,
    demonstrate_authentication,
)


# =============================================================================
# HTTP RESPONSE ANALYSER TESTS
# =============================================================================

class TestHTTPResponseAnalyser:
    """Tests for HTTPResponseAnalyser class."""
    
    def test_is_success_for_200(self, mock_response):
        """Test success detection for 200 status."""
        response = mock_response(status_code=200)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.is_success is True
    
    def test_is_success_for_201(self, mock_response):
        """Test success detection for 201 status."""
        response = mock_response(status_code=201)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.is_success is True
    
    def test_is_success_for_404(self, mock_response):
        """Test success detection for 404 status."""
        response = mock_response(status_code=404)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.is_success is False
    
    def test_is_client_error(self, mock_response):
        """Test client error detection."""
        response = mock_response(status_code=400)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.is_client_error is True
        assert analyser.is_server_error is False
    
    def test_is_server_error(self, mock_response):
        """Test server error detection."""
        response = mock_response(status_code=500)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.is_server_error is True
        assert analyser.is_client_error is False
    
    def test_is_rate_limited(self, mock_response):
        """Test rate limit detection."""
        response = mock_response(status_code=429)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.is_rate_limited is True
    
    def test_get_retry_after_integer(self, mock_response):
        """Test Retry-After parsing for integer value."""
        response = mock_response(
            status_code=429,
            headers={'Retry-After': '60'}
        )
        analyser = HTTPResponseAnalyser(response)
        assert analyser.get_retry_after() == 60
    
    def test_get_retry_after_missing(self, mock_response):
        """Test Retry-After when header is missing."""
        response = mock_response(status_code=429)
        analyser = HTTPResponseAnalyser(response)
        assert analyser.get_retry_after() is None


# =============================================================================
# PAGINATED API CLIENT TESTS
# =============================================================================

class TestPaginatedAPIClient:
    """Tests for PaginatedAPIClient class."""
    
    def test_client_initialization(self):
        """Test client initialises with correct defaults."""
        client = PaginatedAPIClient('https://api.example.com')
        assert client.base_url == 'https://api.example.com'
        assert client.default_page_size == 100
        assert 'Accept' in client.session.headers
    
    def test_has_next_page_with_next_link(self):
        """Test next page detection with explicit next link."""
        client = PaginatedAPIClient('https://api.example.com')
        data = {'next': 'https://api.example.com/items?page=2'}
        assert client._has_next_page(data) is True
    
    def test_has_next_page_with_has_more(self):
        """Test next page detection with has_more flag."""
        client = PaginatedAPIClient('https://api.example.com')
        data = {'has_more': True}
        assert client._has_next_page(data) is True
    
    def test_has_next_page_false(self):
        """Test next page detection when no more pages."""
        client = PaginatedAPIClient('https://api.example.com')
        data = {'results': [], 'has_more': False}
        assert client._has_next_page(data) is False


# =============================================================================
# AUTHENTICATION TESTS
# =============================================================================

class TestAPIKeyAuthenticator:
    """Tests for APIKeyAuthenticator class."""
    
    def test_header_placement(self):
        """Test API key placement in headers."""
        auth = APIKeyAuthenticator('test_key', 'X-API-Key', 'header')
        url, params, headers = auth.apply_to_request('https://api.example.com')
        
        assert headers['X-API-Key'] == 'test_key'
        assert 'X-API-Key' not in params
    
    def test_query_placement(self):
        """Test API key placement in query parameters."""
        auth = APIKeyAuthenticator('test_key', 'api_key', 'query')
        url, params, headers = auth.apply_to_request('https://api.example.com')
        
        assert params['api_key'] == 'test_key'
        assert 'api_key' not in headers
    
    def test_basic_placement(self):
        """Test API key placement as Basic auth."""
        auth = APIKeyAuthenticator('test_key', '', 'basic')
        url, params, headers = auth.apply_to_request('https://api.example.com')
        
        assert 'Authorization' in headers
        assert headers['Authorization'].startswith('Basic ')
    
    def test_preserves_existing_params(self):
        """Test that existing params are preserved."""
        auth = APIKeyAuthenticator('test_key', 'api_key', 'query')
        url, params, headers = auth.apply_to_request(
            'https://api.example.com',
            params={'existing': 'value'}
        )
        
        assert params['existing'] == 'value'
        assert params['api_key'] == 'test_key'


class TestOAuth2ClientCredentials:
    """Tests for OAuth2ClientCredentials class."""
    
    @patch('lab_12_01_api_consumption.requests.post')
    def test_get_access_token(self, mock_post, mock_response):
        """Test access token retrieval."""
        mock_post.return_value = mock_response(
            json_data={
                'access_token': 'test_token_123',
                'expires_in': 3600,
                'token_type': 'Bearer'
            }
        )
        
        oauth = OAuth2ClientCredentials(
            token_url='https://oauth.example.com/token',
            client_id='client_123',
            client_secret='secret_456'
        )
        
        token = oauth.get_access_token()
        assert token == 'test_token_123'
    
    @patch('lab_12_01_api_consumption.requests.post')
    def test_token_caching(self, mock_post, mock_response):
        """Test that tokens are cached."""
        mock_post.return_value = mock_response(
            json_data={
                'access_token': 'cached_token',
                'expires_in': 3600,
                'token_type': 'Bearer'
            }
        )
        
        oauth = OAuth2ClientCredentials(
            token_url='https://oauth.example.com/token',
            client_id='client_123',
            client_secret='secret_456'
        )
        
        token1 = oauth.get_access_token()
        token2 = oauth.get_access_token()
        
        assert token1 == token2
        assert mock_post.call_count == 1  # Only one request made
    
    def test_get_auth_header(self):
        """Test authorization header generation."""
        oauth = OAuth2ClientCredentials(
            token_url='https://oauth.example.com/token',
            client_id='client_123',
            client_secret='secret_456'
        )
        oauth._access_token = 'test_token'
        oauth._token_expiry = datetime.now() + timedelta(hours=1)
        
        header = oauth.get_auth_header()
        assert header == {'Authorization': 'Bearer test_token'}


# =============================================================================
# RESPONSE CACHE TESTS
# =============================================================================

class TestResponseCache:
    """Tests for ResponseCache class."""
    
    def test_cache_miss(self, temp_cache_dir):
        """Test cache returns None for missing entries."""
        cache = ResponseCache(temp_cache_dir)
        result = cache.get('https://api.example.com/data')
        assert result is None
    
    def test_cache_set_and_get(self, temp_cache_dir):
        """Test caching and retrieval."""
        cache = ResponseCache(temp_cache_dir)
        
        url = 'https://api.example.com/data'
        response_data = {'items': [1, 2, 3]}
        
        cache.set(url, None, response_data)
        result = cache.get(url)
        
        assert result == response_data
    
    def test_cache_with_params(self, temp_cache_dir):
        """Test cache key includes parameters."""
        cache = ResponseCache(temp_cache_dir)
        
        url = 'https://api.example.com/data'
        params1 = {'page': 1}
        params2 = {'page': 2}
        
        cache.set(url, params1, {'data': 'page1'})
        cache.set(url, params2, {'data': 'page2'})
        
        assert cache.get(url, params1) == {'data': 'page1'}
        assert cache.get(url, params2) == {'data': 'page2'}
    
    def test_cache_expiry(self, temp_cache_dir):
        """Test cache entries expire."""
        cache = ResponseCache(
            temp_cache_dir,
            default_ttl=timedelta(seconds=-1)  # Already expired
        )
        
        url = 'https://api.example.com/data'
        cache.set(url, None, {'data': 'test'})
        
        result = cache.get(url)
        assert result is None


# =============================================================================
# ROBUST API CLIENT TESTS
# =============================================================================

class TestRobustAPIClient:
    """Tests for RobustAPIClient class."""
    
    def test_client_initialization(self):
        """Test client initialises correctly."""
        client = RobustAPIClient('https://api.example.com')
        assert client.base_url == 'https://api.example.com'
        assert client.max_retries == 3
    
    def test_session_configuration(self):
        """Test session is properly configured."""
        client = RobustAPIClient('https://api.example.com')
        session = client.session
        
        assert 'Accept' in session.headers
        assert session.headers['Accept'] == 'application/json'


# =============================================================================
# OPEN ALEX CLIENT TESTS
# =============================================================================

class TestOpenAlexClient:
    """Tests for OpenAlexClient class."""
    
    def test_client_initialization(self):
        """Test client initialises with email."""
        client = OpenAlexClient(email='test@example.com')
        assert 'mailto:test@example.com' in client.session.headers['User-Agent']
    
    def test_client_initialization_without_email(self):
        """Test client initialises without email."""
        client = OpenAlexClient()
        assert client.email is None


# =============================================================================
# INTEGRATION TESTS (MARKED FOR OPTIONAL EXECUTION)
# =============================================================================

@pytest.mark.integration
@pytest.mark.slow
class TestHTTPIntegration:
    """Integration tests requiring network access."""
    
    def test_httpbin_get(self):
        """Test basic GET request against httpbin."""
        import requests
        response = requests.get('https://httpbin.org/get', timeout=10)
        assert response.status_code == 200
    
    def test_demonstrate_http_methods(self):
        """Test HTTP methods demonstration."""
        results = demonstrate_http_methods()
        assert results['GET'] == 200


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
