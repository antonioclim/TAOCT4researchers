#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Hard 02 - Ethical Web Scraper

Difficulty: ★★★★★
Estimated Time: 40 minutes
Learning Objective: LO5

Task:
Design and implement a production-quality web scraper that follows
ethical practices. Your scraper must respect robots.txt, implement
rate limiting, cache responses and extract structured data reliably.

Requirements:
1. Parse and respect robots.txt directives
2. Implement configurable rate limiting
3. Cache responses to reduce server load
4. Extract structured data using CSS selectors
5. Handle pagination across multiple pages
6. Log all operations for transparency

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RobotsParser:
    """
    Parse and enforce robots.txt rules.
    
    Implements a subset of the Robots Exclusion Protocol,
    supporting User-agent, Allow and Disallow directives.
    """
    
    base_url: str
    user_agent: str = '*'
    _rules: dict[str, list[str]] = field(default_factory=dict)
    _loaded: bool = False
    
    def load(self) -> None:
        """
        Fetch and parse robots.txt from the target site.
        
        If robots.txt is not found or cannot be parsed, assumes
        all paths are allowed.
        """
        # TODO: Implement robots.txt fetching and parsing
        # HINT: Fetch {base_url}/robots.txt
        # HINT: Parse User-agent, Allow, Disallow directives
        # HINT: Store rules for matching user-agent and '*'
        
        pass  # Replace with your implementation
    
    def is_allowed(self, path: str) -> bool:
        """
        Check if scraping a path is allowed.
        
        Args:
            path: URL path to check
            
        Returns:
            True if scraping is permitted
            
        Example:
            >>> parser = RobotsParser('https://example.com')
            >>> parser.is_allowed('/public/page')  # Depends on robots.txt
            True
        """
        if not self._loaded:
            self.load()
        
        # TODO: Implement path checking against rules
        # HINT: Check Disallow rules first
        # HINT: Allow rules take precedence over Disallow
        # HINT: More specific rules take precedence
        
        pass  # Replace with your implementation


@dataclass
class ResponseCache:
    """
    File-based cache for HTTP responses.
    
    Reduces server load by caching responses and serving from
    cache when valid.
    """
    
    cache_dir: Path
    default_ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))
    
    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _key(self, url: str) -> str:
        """Generate cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()
    
    def get(self, url: str) -> Optional[str]:
        """
        Retrieve cached response if valid.
        
        Args:
            url: Request URL
            
        Returns:
            Cached response text or None if not cached/expired
        """
        # TODO: Implement cache retrieval
        # HINT: Check if cache file exists
        # HINT: Verify cache has not expired
        # HINT: Return cached content or None
        
        pass  # Replace with your implementation
    
    def set(self, url: str, content: str, ttl: Optional[timedelta] = None) -> None:
        """
        Store response in cache.
        
        Args:
            url: Request URL
            content: Response content to cache
            ttl: Time-to-live (uses default if not specified)
        """
        # TODO: Implement cache storage
        # HINT: Store content with expiry timestamp
        # HINT: Use JSON for metadata
        
        pass  # Replace with your implementation


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.
    
    Controls request rate to avoid overwhelming servers.
    """
    
    requests_per_second: float = 1.0
    burst_size: int = 5
    _tokens: float = field(default=0.0, repr=False)
    _last_update: float = field(default=0.0, repr=False)
    
    def __post_init__(self) -> None:
        self._tokens = float(self.burst_size)
        self._last_update = time.time()
    
    def acquire(self) -> None:
        """
        Acquire permission to make a request.
        
        Blocks until a token is available in the bucket.
        """
        # TODO: Implement token bucket algorithm
        # HINT: Add tokens based on elapsed time
        # HINT: Cap tokens at burst_size
        # HINT: If tokens < 1, sleep until available
        # HINT: Consume one token
        
        pass  # Replace with your implementation


@dataclass
class EthicalScraper:
    """
    Production-quality ethical web scraper.
    
    Combines robots.txt compliance, rate limiting, caching and
    structured data extraction.
    """
    
    base_url: str
    user_agent: str = 'ResearchScraper/1.0 (+https://university.edu/bot-info)'
    requests_per_second: float = 0.5
    cache_dir: Optional[Path] = None
    
    _session: Optional[requests.Session] = field(default=None, repr=False)
    _robots: Optional[RobotsParser] = field(default=None, repr=False)
    _cache: Optional[ResponseCache] = field(default=None, repr=False)
    _limiter: Optional[RateLimiter] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        # Initialise components
        self._session = requests.Session()
        self._session.headers['User-Agent'] = self.user_agent
        
        self._robots = RobotsParser(self.base_url, self.user_agent)
        self._limiter = RateLimiter(self.requests_per_second)
        
        if self.cache_dir:
            self._cache = ResponseCache(self.cache_dir)
    
    def fetch(self, path: str, use_cache: bool = True) -> Optional[BeautifulSoup]:
        """
        Fetch and parse a page ethically.
        
        Args:
            path: URL path relative to base_url
            use_cache: Whether to use cached responses
            
        Returns:
            BeautifulSoup document or None if fetch failed
            
        Raises:
            PermissionError: If robots.txt disallows access
        """
        # TODO: Implement ethical fetching
        # HINT: Check robots.txt first
        # HINT: Check cache if enabled
        # HINT: Respect rate limiter
        # HINT: Log all operations
        
        pass  # Replace with your implementation
    
    def extract_items(
        self,
        soup: BeautifulSoup,
        item_selector: str,
        field_selectors: dict[str, str]
    ) -> list[dict[str, str]]:
        """
        Extract structured data from page.
        
        Args:
            soup: Parsed HTML document
            item_selector: CSS selector for item containers
            field_selectors: Map of field names to CSS selectors
            
        Returns:
            List of dictionaries with extracted fields
            
        Example:
            >>> scraper = EthicalScraper('https://example.com')
            >>> soup = BeautifulSoup('<div class="item"><h2>Title</h2></div>', 'html.parser')
            >>> items = scraper.extract_items(soup, 'div.item', {'title': 'h2'})
            >>> items[0]['title']
            'Title'
        """
        # TODO: Implement structured extraction
        # HINT: Find all items matching item_selector
        # HINT: For each item, extract fields using field_selectors
        # HINT: Handle missing fields gracefully
        
        pass  # Replace with your implementation
    
    def scrape_paginated(
        self,
        start_path: str,
        item_selector: str,
        field_selectors: dict[str, str],
        next_page_selector: str,
        max_pages: int = 10
    ) -> Iterator[dict[str, str]]:
        """
        Scrape items from multiple paginated pages.
        
        Args:
            start_path: Initial page path
            item_selector: CSS selector for item containers
            field_selectors: Map of field names to CSS selectors
            next_page_selector: CSS selector for next page link
            max_pages: Maximum pages to scrape
            
        Yields:
            Dictionaries with extracted item data
        """
        # TODO: Implement paginated scraping
        # HINT: Loop through pages up to max_pages
        # HINT: Find next page link and follow
        # HINT: Yield items from each page
        
        pass  # Replace with your implementation


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    print('Testing Ethical Scraper exercises...\n')
    
    # Test RateLimiter
    print('Test 1: RateLimiter')
    try:
        limiter = RateLimiter(requests_per_second=10.0, burst_size=2)
        
        # Should allow burst
        start = time.time()
        limiter.acquire()
        limiter.acquire()
        elapsed = time.time() - start
        assert elapsed < 0.5, f'Burst should be fast, took {elapsed}s'
        
        # Third request should wait
        start = time.time()
        limiter.acquire()
        elapsed = time.time() - start
        assert elapsed >= 0.05, f'Should have waited, only took {elapsed}s'
        
        print('  PASSED: Rate limiter works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test ResponseCache
    print('\nTest 2: ResponseCache')
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ResponseCache(Path(tmpdir))
            
            # Store and retrieve
            cache.set('https://example.com/test', '<html>Test</html>')
            content = cache.get('https://example.com/test')
            assert content == '<html>Test</html>', 'Cache miss'
            
            # Miss for different URL
            content = cache.get('https://example.com/other')
            assert content is None, 'Should be cache miss'
            
        print('  PASSED: Cache works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test extract_items
    print('\nTest 3: extract_items')
    try:
        scraper = EthicalScraper('https://example.com')
        html = '''
        <div class="results">
            <article class="item">
                <h2 class="title">First Item</h2>
                <span class="author">Alice</span>
            </article>
            <article class="item">
                <h2 class="title">Second Item</h2>
                <span class="author">Bob</span>
            </article>
        </div>
        '''
        soup = BeautifulSoup(html, 'html.parser')
        items = scraper.extract_items(
            soup,
            'article.item',
            {'title': 'h2.title', 'author': 'span.author'}
        )
        
        assert len(items) == 2, f'Expected 2 items, got {len(items)}'
        assert items[0]['title'] == 'First Item'
        assert items[1]['author'] == 'Bob'
        
        print('  PASSED: Extraction works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')
    print('Note: Full scraping tests require live website access.')


if __name__ == '__main__':
    test_exercises()
