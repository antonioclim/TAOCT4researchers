#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Test Configuration and Fixtures

This module provides pytest fixtures for testing API consumption,
web scraping and Flask API functionality.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Generator

import pytest
import requests
from flask import Flask
from flask.testing import FlaskClient

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_works() -> list[dict[str, Any]]:
    """Load sample works dataset."""
    data_file = Path(__file__).parent.parent / 'resources' / 'datasets' / 'sample_works.json'
    if data_file.exists():
        with open(data_file) as f:
            data = json.load(f)
            return data.get('works', [])
    return []


@pytest.fixture
def sample_http_response() -> dict[str, Any]:
    """Sample HTTP response structure for testing."""
    return {
        'status_code': 200,
        'headers': {
            'content-type': 'application/json',
            'x-ratelimit-limit': '1000',
            'x-ratelimit-remaining': '999',
        },
        'json': {
            'items': [
                {'id': 1, 'name': 'Item 1'},
                {'id': 2, 'name': 'Item 2'},
            ],
            'meta': {
                'total': 100,
                'page': 1,
                'per_page': 10
            }
        }
    }


@pytest.fixture
def paginated_response() -> dict[str, Any]:
    """Sample paginated API response."""
    return {
        'results': [
            {'id': f'item_{i}', 'value': i * 10}
            for i in range(10)
        ],
        'meta': {
            'count': 100,
            'next_cursor': 'cursor_abc123',
            'per_page': 10
        }
    }


# =============================================================================
# MOCK RESPONSE FIXTURES
# =============================================================================

class MockResponse:
    """Mock requests.Response object for testing."""
    
    def __init__(
        self,
        status_code: int = 200,
        json_data: dict | None = None,
        text: str = '',
        headers: dict | None = None,
        raise_for_status_error: bool = False
    ):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or json.dumps(self._json_data)
        self.headers = headers or {'content-type': 'application/json'}
        self._raise_error = raise_for_status_error
        self.content = self.text.encode('utf-8')
        self.ok = 200 <= status_code < 400
    
    def json(self) -> dict:
        """Return JSON data."""
        return self._json_data
    
    def raise_for_status(self) -> None:
        """Raise HTTPError if configured."""
        if self._raise_error or self.status_code >= 400:
            error = requests.HTTPError()
            error.response = self
            raise error


@pytest.fixture
def mock_response() -> type[MockResponse]:
    """Provide MockResponse class for tests."""
    return MockResponse


@pytest.fixture
def mock_success_response() -> MockResponse:
    """Pre-configured successful response."""
    return MockResponse(
        status_code=200,
        json_data={'status': 'success', 'data': [1, 2, 3]}
    )


@pytest.fixture
def mock_error_response() -> MockResponse:
    """Pre-configured error response."""
    return MockResponse(
        status_code=404,
        json_data={'error': 'Not Found'},
        raise_for_status_error=True
    )


@pytest.fixture
def mock_rate_limited_response() -> MockResponse:
    """Pre-configured rate limit response."""
    return MockResponse(
        status_code=429,
        json_data={'error': 'Too Many Requests'},
        headers={
            'content-type': 'application/json',
            'retry-after': '60',
            'x-ratelimit-limit': '100',
            'x-ratelimit-remaining': '0'
        },
        raise_for_status_error=True
    )


# =============================================================================
# FLASK TEST FIXTURES
# =============================================================================

@pytest.fixture
def flask_app() -> Flask:
    """Create Flask test application."""
    from lab.lab_12_02_web_scraping_flask import create_api_app
    
    app = create_api_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def flask_client(flask_app: Flask) -> Generator[FlaskClient, None, None]:
    """Create Flask test client."""
    with flask_app.test_client() as client:
        yield client


# =============================================================================
# TEMPORARY FILE FIXTURES
# =============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary directory for cache testing."""
    cache_dir = tmp_path / 'cache'
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_data_file(tmp_path: Path, sample_works: list[dict]) -> Path:
    """Create temporary data file for testing."""
    data_file = tmp_path / 'test_data.json'
    data_file.write_text(json.dumps({
        'metadata': {'title': 'Test Dataset'},
        'works': sample_works[:5]
    }))
    return data_file


# =============================================================================
# HTML FIXTURES FOR SCRAPING TESTS
# =============================================================================

@pytest.fixture
def sample_html() -> str:
    """Sample HTML page for scraping tests."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sample Research Page</title>
    </head>
    <body>
        <h1>Research Publications</h1>
        <div class="publications">
            <article class="publication">
                <h2 class="title">Machine Learning for Climate</h2>
                <p class="authors">Smith, J.; Chen, L.</p>
                <p class="abstract">This paper presents...</p>
                <a href="/papers/1" class="link">Read More</a>
            </article>
            <article class="publication">
                <h2 class="title">Deep Learning in Genomics</h2>
                <p class="authors">Johnson, M.</p>
                <p class="abstract">We survey recent...</p>
                <a href="/papers/2" class="link">Read More</a>
            </article>
        </div>
        <table class="data-table">
            <tr>
                <th>Year</th>
                <th>Count</th>
            </tr>
            <tr>
                <td>2023</td>
                <td>150</td>
            </tr>
            <tr>
                <td>2024</td>
                <td>89</td>
            </tr>
        </table>
        <nav class="pagination">
            <a href="/page/1" class="prev">Previous</a>
            <a href="/page/3" class="next">Next</a>
        </nav>
    </body>
    </html>
    """


@pytest.fixture
def robots_txt_content() -> str:
    """Sample robots.txt content."""
    return """
User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /public/

User-agent: ResearchBot
Allow: /data/
Disallow: /internal/
"""


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def api_config() -> dict[str, Any]:
    """API configuration for tests."""
    return {
        'base_url': 'https://api.example.com',
        'timeout': 30,
        'max_retries': 3,
        'page_size': 100,
    }


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset any rate limiter state between tests."""
    yield
    # Cleanup if needed


# =============================================================================
# MARKERS
# =============================================================================

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        'markers', 'slow: marks tests as slow (deselect with -m "not slow")'
    )
    config.addinivalue_line(
        'markers', 'integration: marks tests requiring network access'
    )
    config.addinivalue_line(
        'markers', 'flask: marks Flask application tests'
    )
