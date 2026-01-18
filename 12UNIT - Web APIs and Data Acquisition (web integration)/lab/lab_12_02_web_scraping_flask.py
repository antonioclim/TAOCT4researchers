#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Lab 12.02: Web Scraping and Flask API Development

This laboratory develops skills in extracting data from HTML sources and
exposing research datasets through RESTful APIs. You will implement
ethical scraping patterns and build production-quality Flask endpoints.

Duration: 40 minutes
Difficulty: ★★★★☆

Learning Objectives Addressed:
- LO5: Perform ethical web scraping with BeautifulSoup
- LO6: Design and implement Flask REST APIs

Prerequisites:
- Lab 12.01: API Consumption
- 10UNIT: JSON serialisation

Required Packages:
    pip install beautifulsoup4 lxml requests flask

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Course: Computational Thinking for Researchers
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag
from flask import Flask, jsonify, request, abort, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: WEB SCRAPING FUNDAMENTALS
# Duration: ~15 minutes
# Learning Objective: LO5
# =============================================================================

@dataclass
class RobotsChecker:
    """
    Check robots.txt compliance before scraping.
    
    The Robots Exclusion Protocol specifies which paths crawlers may access.
    Ethical scrapers always check and respect these directives.
    
    Attributes:
        base_url: Website base URL
        user_agent: User-agent string for rule matching
    """
    
    base_url: str
    user_agent: str = '*'
    _rules: Optional[dict[str, list[str]]] = field(default=None, repr=False)
    
    def _fetch_robots(self) -> None:
        """Fetch and parse robots.txt file."""
        robots_url = urljoin(self.base_url, '/robots.txt')
        
        try:
            response = requests.get(robots_url, timeout=10)
            if response.status_code == 200:
                self._parse_robots(response.text)
            else:
                # No robots.txt means everything is allowed
                self._rules = {'allow': ['*'], 'disallow': []}
        except requests.RequestException:
            self._rules = {'allow': ['*'], 'disallow': []}
    
    def _parse_robots(self, content: str) -> None:
        """Parse robots.txt content."""
        self._rules = {'allow': [], 'disallow': []}
        current_agent = None
        
        for line in content.split('\n'):
            line = line.strip().lower()
            
            if line.startswith('user-agent:'):
                agent = line.split(':', 1)[1].strip()
                if agent == '*' or agent == self.user_agent.lower():
                    current_agent = agent
                else:
                    current_agent = None
            
            elif current_agent and line.startswith('disallow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    self._rules['disallow'].append(path)
            
            elif current_agent and line.startswith('allow:'):
                path = line.split(':', 1)[1].strip()
                if path:
                    self._rules['allow'].append(path)
    
    def is_allowed(self, path: str) -> bool:
        """
        Check if scraping a path is allowed.
        
        Args:
            path: URL path to check
            
        Returns:
            True if scraping is permitted
        """
        if self._rules is None:
            self._fetch_robots()
        
        # Check disallow rules
        for rule in self._rules.get('disallow', []):
            if path.startswith(rule) or rule == '/':
                # Check if explicitly allowed
                for allow in self._rules.get('allow', []):
                    if path.startswith(allow):
                        return True
                return False
        
        return True


@dataclass
class HTMLScraper:
    """
    Ethical HTML scraper with polite request patterns.
    
    This scraper implements professional standards:
    - Respects robots.txt
    - Rate limits requests
    - Identifies itself via User-Agent
    - Caches responses to reduce load
    
    Attributes:
        base_url: Target website base URL
        delay: Delay between requests in seconds
        user_agent: Identification string
    """
    
    base_url: str
    delay: float = 1.0
    user_agent: str = 'ResearchScraper/1.0 (+https://university.edu/bot)'
    _last_request: float = field(default=0.0, repr=False)
    _session: Optional[requests.Session] = field(default=None, repr=False)
    _robots: Optional[RobotsChecker] = field(default=None, repr=False)
    
    @property
    def session(self) -> requests.Session:
        """Get or create session with configured headers."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-GB,en;q=0.9',
            })
        return self._session
    
    @property
    def robots(self) -> RobotsChecker:
        """Get robots.txt checker."""
        if self._robots is None:
            self._robots = RobotsChecker(self.base_url, self.user_agent)
        return self._robots
    
    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request = time.time()
    
    def fetch(self, path: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse HTML from path.
        
        Args:
            path: URL path relative to base_url
            
        Returns:
            BeautifulSoup object or None if fetch failed
            
        Raises:
            PermissionError: If robots.txt disallows access
        """
        # TODO: Check robots.txt compliance
        if not self.robots.is_allowed(path):
            raise PermissionError(f'Scraping {path} disallowed by robots.txt')
        
        self._wait_for_rate_limit()
        
        url = urljoin(self.base_url, path)
        logger.info(f'Fetching: {url}')
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return BeautifulSoup(response.text, 'lxml')
            
        except requests.RequestException as e:
            logger.error(f'Fetch failed: {e}')
            return None
    
    def extract_text(self, soup: BeautifulSoup, selector: str) -> list[str]:
        """
        Extract text content from elements matching selector.
        
        Args:
            soup: BeautifulSoup document
            selector: CSS selector string
            
        Returns:
            List of text contents
        """
        elements = soup.select(selector)
        return [el.get_text(strip=True) for el in elements]
    
    def extract_links(
        self,
        soup: BeautifulSoup,
        selector: str = 'a[href]'
    ) -> list[dict[str, str]]:
        """
        Extract links from elements matching selector.
        
        Args:
            soup: BeautifulSoup document
            selector: CSS selector string
            
        Returns:
            List of dictionaries with 'text' and 'href' keys
        """
        links = []
        for element in soup.select(selector):
            href = element.get('href', '')
            if href:
                links.append({
                    'text': element.get_text(strip=True),
                    'href': urljoin(self.base_url, href)
                })
        return links


def scrape_example_page() -> dict[str, Any]:
    """
    Demonstrate scraping from a sample page.
    
    This function shows basic scraping patterns using a public
    example page. In practice, you would target specific research
    data sources.
    
    Returns:
        Dictionary of extracted data
    """
    # Use httpbin's HTML endpoint as example
    scraper = HTMLScraper('https://httpbin.org')
    
    soup = scraper.fetch('/html')
    
    if soup is None:
        return {'error': 'Failed to fetch page'}
    
    # TODO: Extract content from the page
    # HINT: Use scraper.extract_text() with appropriate selectors
    result = {
        'title': ...,      # Extract page title
        'paragraphs': ..., # Extract paragraph texts
        'headings': ...,   # Extract heading texts
    }
    
    return result


# =============================================================================
# SECTION 2: ADVANCED SCRAPING PATTERNS
# Duration: ~10 minutes
# Learning Objective: LO5
# =============================================================================

@dataclass
class TableScraper:
    """
    Specialised scraper for HTML tables.
    
    Many research data sources present information in HTML tables.
    This class provides utilities for extracting structured data
    from table elements.
    """
    
    soup: BeautifulSoup
    
    def find_tables(self) -> list[Tag]:
        """Find all table elements in document."""
        return self.soup.find_all('table')
    
    def extract_table(
        self,
        table: Tag,
        has_header: bool = True
    ) -> list[dict[str, str]]:
        """
        Extract table data as list of dictionaries.
        
        Args:
            table: BeautifulSoup table element
            has_header: Whether first row contains headers
            
        Returns:
            List of row dictionaries
        """
        rows = table.find_all('tr')
        
        if not rows:
            return []
        
        # Extract headers
        if has_header:
            header_row = rows[0]
            headers = [
                th.get_text(strip=True)
                for th in header_row.find_all(['th', 'td'])
            ]
            data_rows = rows[1:]
        else:
            # Generate column names
            first_row = rows[0].find_all(['th', 'td'])
            headers = [f'column_{i}' for i in range(len(first_row))]
            data_rows = rows
        
        # Extract data
        result = []
        for row in data_rows:
            cells = row.find_all(['th', 'td'])
            row_data = {}
            for i, cell in enumerate(cells):
                if i < len(headers):
                    row_data[headers[i]] = cell.get_text(strip=True)
            if row_data:
                result.append(row_data)
        
        return result
    
    def to_records(self) -> list[dict[str, str]]:
        """
        Extract all tables as flat list of records.
        
        Returns:
            Combined records from all tables
        """
        all_records = []
        for table in self.find_tables():
            all_records.extend(self.extract_table(table))
        return all_records


@dataclass
class PaginatedScraper(HTMLScraper):
    """
    Scraper that handles pagination.
    
    Many websites split content across multiple pages. This scraper
    automatically follows pagination links to collect all data.
    
    Attributes:
        next_selector: CSS selector for 'next page' link
        max_pages: Maximum pages to scrape
    """
    
    next_selector: str = 'a.next, a[rel="next"], .pagination a:last-child'
    max_pages: int = 10
    
    def scrape_all_pages(
        self,
        start_path: str,
        item_selector: str
    ) -> list[dict[str, Any]]:
        """
        Scrape items from all paginated pages.
        
        Args:
            start_path: Initial page path
            item_selector: CSS selector for items on each page
            
        Returns:
            List of extracted items from all pages
        """
        all_items = []
        current_path = start_path
        pages_scraped = 0
        
        while current_path and pages_scraped < self.max_pages:
            soup = self.fetch(current_path)
            
            if soup is None:
                break
            
            # Extract items from current page
            items = soup.select(item_selector)
            for item in items:
                all_items.append(self._extract_item(item))
            
            # Find next page link
            next_link = soup.select_one(self.next_selector)
            if next_link and next_link.get('href'):
                current_path = next_link['href']
            else:
                current_path = None
            
            pages_scraped += 1
            logger.info(f'Scraped page {pages_scraped}, found {len(items)} items')
        
        return all_items
    
    def _extract_item(self, element: Tag) -> dict[str, Any]:
        """
        Extract data from a single item element.
        
        Override this method for custom extraction logic.
        """
        return {
            'text': element.get_text(strip=True),
            'html': str(element)[:200],
        }


# =============================================================================
# SECTION 3: FLASK API DEVELOPMENT
# Duration: ~10 minutes
# Learning Objective: LO6
# =============================================================================

# Sample research dataset
SAMPLE_DATASETS = [
    {
        'id': 1,
        'name': 'Climate Observations 2023',
        'description': 'Global temperature and precipitation measurements',
        'records': 52560,
        'format': 'CSV',
        'created_at': '2023-01-15T00:00:00Z',
        'tags': ['climate', 'temperature', 'precipitation']
    },
    {
        'id': 2,
        'name': 'Species Distribution',
        'description': 'Biodiversity occurrence records',
        'records': 12847,
        'format': 'JSON',
        'created_at': '2023-03-22T00:00:00Z',
        'tags': ['biodiversity', 'ecology', 'species']
    },
    {
        'id': 3,
        'name': 'Economic Indicators',
        'description': 'Quarterly economic metrics by country',
        'records': 8640,
        'format': 'CSV',
        'created_at': '2023-06-01T00:00:00Z',
        'tags': ['economics', 'gdp', 'indicators']
    }
]


def create_api_app() -> Flask:
    """
    Create and configure Flask API application.
    
    This function demonstrates building a RESTful API for
    exposing research datasets. The API follows REST conventions
    for resource naming, HTTP methods and status codes.
    
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # In-memory storage (use database in production)
    datasets = {d['id']: d.copy() for d in SAMPLE_DATASETS}
    next_id = max(datasets.keys()) + 1
    
    # -------------------------------------------------------------------------
    # Health check endpoint
    # -------------------------------------------------------------------------
    
    @app.route('/api/health', methods=['GET'])
    def health_check() -> Response:
        """Return API health status."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    # -------------------------------------------------------------------------
    # Dataset collection endpoints
    # -------------------------------------------------------------------------
    
    @app.route('/api/datasets', methods=['GET'])
    def list_datasets() -> Response:
        """
        List all datasets with optional filtering.
        
        Query Parameters:
            tag: Filter by tag
            format: Filter by format
            limit: Maximum results (default 100)
            offset: Skip first N results
        
        Returns:
            JSON array of dataset objects
        """
        # TODO: Implement filtering logic
        tag_filter = request.args.get('tag')
        format_filter = request.args.get('format')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        results = list(datasets.values())
        
        # Apply filters
        if tag_filter:
            results = [d for d in results if tag_filter in d.get('tags', [])]
        
        if format_filter:
            results = [d for d in results if d.get('format') == format_filter]
        
        # Apply pagination
        total = len(results)
        results = results[offset:offset + limit]
        
        return jsonify({
            'data': results,
            'meta': {
                'total': total,
                'limit': limit,
                'offset': offset
            }
        })
    
    @app.route('/api/datasets', methods=['POST'])
    def create_dataset() -> tuple[Response, int]:
        """
        Create a new dataset.
        
        Request Body:
            name: Dataset name (required)
            description: Dataset description (required)
            records: Number of records
            format: Data format
            tags: List of tags
        
        Returns:
            Created dataset object with 201 status
        """
        nonlocal next_id
        
        # TODO: Validate request content type
        if not request.is_json:
            abort(400, description='Request must be JSON')
        
        data = request.get_json()
        
        # TODO: Validate required fields
        required = ['name', 'description']
        missing = [f for f in required if f not in data]
        if missing:
            abort(400, description=f'Missing required fields: {missing}')
        
        # Create new dataset
        new_dataset = {
            'id': next_id,
            'name': data['name'],
            'description': data['description'],
            'records': data.get('records', 0),
            'format': data.get('format', 'JSON'),
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'tags': data.get('tags', [])
        }
        
        datasets[next_id] = new_dataset
        next_id += 1
        
        return jsonify(new_dataset), 201
    
    # -------------------------------------------------------------------------
    # Individual dataset endpoints
    # -------------------------------------------------------------------------
    
    @app.route('/api/datasets/<int:dataset_id>', methods=['GET'])
    def get_dataset(dataset_id: int) -> Response:
        """
        Retrieve a specific dataset by ID.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Dataset object or 404 error
        """
        dataset = datasets.get(dataset_id)
        
        if dataset is None:
            abort(404, description=f'Dataset {dataset_id} not found')
        
        return jsonify(dataset)
    
    @app.route('/api/datasets/<int:dataset_id>', methods=['PUT'])
    def update_dataset(dataset_id: int) -> Response:
        """
        Update a dataset (full replacement).
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Updated dataset object
        """
        if dataset_id not in datasets:
            abort(404, description=f'Dataset {dataset_id} not found')
        
        if not request.is_json:
            abort(400, description='Request must be JSON')
        
        data = request.get_json()
        
        # Preserve ID and created_at
        updated = {
            'id': dataset_id,
            'created_at': datasets[dataset_id]['created_at'],
            'name': data.get('name', datasets[dataset_id]['name']),
            'description': data.get('description', datasets[dataset_id]['description']),
            'records': data.get('records', datasets[dataset_id]['records']),
            'format': data.get('format', datasets[dataset_id]['format']),
            'tags': data.get('tags', datasets[dataset_id]['tags']),
            'updated_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        datasets[dataset_id] = updated
        return jsonify(updated)
    
    @app.route('/api/datasets/<int:dataset_id>', methods=['DELETE'])
    def delete_dataset(dataset_id: int) -> tuple[str, int]:
        """
        Delete a dataset.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Empty response with 204 status
        """
        if dataset_id not in datasets:
            abort(404, description=f'Dataset {dataset_id} not found')
        
        del datasets[dataset_id]
        return '', 204
    
    # -------------------------------------------------------------------------
    # Error handlers
    # -------------------------------------------------------------------------
    
    @app.errorhandler(400)
    def bad_request(error: Any) -> tuple[Response, int]:
        """Handle bad request errors."""
        return jsonify({
            'error': 'Bad Request',
            'message': str(error.description)
        }), 400
    
    @app.errorhandler(404)
    def not_found(error: Any) -> tuple[Response, int]:
        """Handle not found errors."""
        return jsonify({
            'error': 'Not Found',
            'message': str(error.description)
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error: Any) -> tuple[Response, int]:
        """Handle internal server errors."""
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred'
        }), 500
    
    return app


# =============================================================================
# SECTION 4: RESEARCH DATA API
# Duration: ~5 minutes
# Learning Objective: LO6
# =============================================================================

def create_research_api(data_file: Path) -> Flask:
    """
    Create API exposing a research dataset from file.
    
    This function demonstrates building an API that serves
    actual research data from a JSON file. This pattern is
    common for sharing reproducible datasets.
    
    Args:
        data_file: Path to JSON data file
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load data from file
    if data_file.exists():
        with open(data_file) as f:
            research_data = json.load(f)
    else:
        research_data = {'works': [], 'metadata': {}}
    
    @app.route('/api/works', methods=['GET'])
    def list_works() -> Response:
        """List all research works with filtering."""
        year = request.args.get('year', type=int)
        author = request.args.get('author')
        limit = request.args.get('limit', 100, type=int)
        
        works = research_data.get('works', [])
        
        # Apply filters
        if year:
            works = [w for w in works if w.get('year') == year]
        
        if author:
            works = [
                w for w in works
                if author.lower() in str(w.get('authors', [])).lower()
            ]
        
        return jsonify({
            'data': works[:limit],
            'total': len(works)
        })
    
    @app.route('/api/works/<work_id>', methods=['GET'])
    def get_work(work_id: str) -> Response:
        """Get a specific work by ID or DOI."""
        works = research_data.get('works', [])
        
        for work in works:
            if str(work.get('id')) == work_id or work.get('doi') == work_id:
                return jsonify(work)
        
        abort(404, description=f'Work {work_id} not found')
    
    @app.route('/api/metadata', methods=['GET'])
    def get_metadata() -> Response:
        """Get dataset metadata."""
        return jsonify(research_data.get('metadata', {}))
    
    return app


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """Run laboratory exercises demonstrating scraping and Flask."""
    print('=' * 70)
    print('12UNIT: Web APIs and Data Acquisition')
    print('Lab 12.02: Web Scraping and Flask API Development')
    print('=' * 70)
    
    # Section 1-2: Web Scraping
    print('\n--- Sections 1-2: Web Scraping ---')
    print('Demonstrating scraping patterns...')
    
    try:
        result = scrape_example_page()
        print(f'Scraped example page: {result}')
    except Exception as e:
        print(f'Scraping example failed: {e}')
    
    # Section 3-4: Flask API
    print('\n--- Sections 3-4: Flask API ---')
    print('Creating Flask API application...')
    
    app = create_api_app()
    
    # Test the API endpoints using test client
    with app.test_client() as client:
        # Test health check
        response = client.get('/api/health')
        print(f'Health check: {response.status_code}')
        
        # Test list datasets
        response = client.get('/api/datasets')
        data = response.get_json()
        print(f'List datasets: {len(data["data"])} datasets')
        
        # Test get single dataset
        response = client.get('/api/datasets/1')
        print(f'Get dataset 1: {response.status_code}')
        
        # Test create dataset
        new_dataset = {
            'name': 'Test Dataset',
            'description': 'Created during testing',
            'records': 100,
            'tags': ['test']
        }
        response = client.post(
            '/api/datasets',
            json=new_dataset,
            content_type='application/json'
        )
        print(f'Create dataset: {response.status_code}')
        
        # Test 404
        response = client.get('/api/datasets/999')
        print(f'Get non-existent: {response.status_code}')
    
    print('\n' + '=' * 70)
    print('Laboratory exercises complete.')
    print('To run the API server:')
    print('  app = create_api_app()')
    print('  app.run(debug=True)')
    print('=' * 70)


if __name__ == '__main__':
    main()
