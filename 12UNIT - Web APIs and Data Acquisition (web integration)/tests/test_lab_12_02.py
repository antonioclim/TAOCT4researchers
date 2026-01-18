#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Tests for Lab 12.02: Web Scraping and Flask

This module tests web scraping functionality and Flask API endpoints.

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from bs4 import BeautifulSoup

# Import modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lab'))

from lab_12_02_web_scraping_flask import (
    RobotsChecker,
    HTMLScraper,
    TableScraper,
    create_api_app,
    create_research_api,
    SAMPLE_DATASETS,
)


# =============================================================================
# ROBOTS CHECKER TESTS
# =============================================================================

class TestRobotsChecker:
    """Tests for RobotsChecker class."""
    
    def test_parse_simple_robots(self):
        """Test parsing simple robots.txt."""
        checker = RobotsChecker('https://example.com')
        content = """
User-agent: *
Disallow: /admin/
Disallow: /private/
"""
        checker._parse_robots(content)
        
        assert '/admin/' in checker._rules['disallow']
        assert '/private/' in checker._rules['disallow']
    
    def test_is_allowed_permitted_path(self):
        """Test allowed path returns True."""
        checker = RobotsChecker('https://example.com')
        checker._rules = {
            'allow': [],
            'disallow': ['/admin/', '/private/']
        }
        
        assert checker.is_allowed('/public/page') is True
    
    def test_is_allowed_blocked_path(self):
        """Test blocked path returns False."""
        checker = RobotsChecker('https://example.com')
        checker._rules = {
            'allow': [],
            'disallow': ['/admin/', '/private/']
        }
        
        assert checker.is_allowed('/admin/users') is False
    
    def test_is_allowed_with_explicit_allow(self):
        """Test explicit allow overrides disallow."""
        checker = RobotsChecker('https://example.com')
        checker._rules = {
            'allow': ['/admin/public/'],
            'disallow': ['/admin/']
        }
        
        assert checker.is_allowed('/admin/public/data') is True


# =============================================================================
# TABLE SCRAPER TESTS
# =============================================================================

class TestTableScraper:
    """Tests for TableScraper class."""
    
    def test_extract_table_with_headers(self, sample_html):
        """Test table extraction with headers."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        scraper = TableScraper(soup)
        
        tables = scraper.find_tables()
        assert len(tables) == 1
        
        data = scraper.extract_table(tables[0])
        assert len(data) == 2
        assert data[0]['Year'] == '2023'
        assert data[0]['Count'] == '150'
    
    def test_extract_table_without_headers(self):
        """Test table extraction without headers."""
        html = """
        <table>
            <tr><td>A</td><td>1</td></tr>
            <tr><td>B</td><td>2</td></tr>
        </table>
        """
        soup = BeautifulSoup(html, 'html.parser')
        scraper = TableScraper(soup)
        
        tables = scraper.find_tables()
        data = scraper.extract_table(tables[0], has_header=False)
        
        assert len(data) == 2
        assert 'column_0' in data[0]


# =============================================================================
# FLASK API TESTS
# =============================================================================

@pytest.mark.flask
class TestFlaskAPI:
    """Tests for Flask API endpoints."""
    
    def test_health_check(self, flask_client):
        """Test health check endpoint."""
        response = flask_client.get('/api/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_list_datasets(self, flask_client):
        """Test dataset listing endpoint."""
        response = flask_client.get('/api/datasets')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'data' in data
        assert 'meta' in data
        assert len(data['data']) == len(SAMPLE_DATASETS)
    
    def test_list_datasets_with_tag_filter(self, flask_client):
        """Test dataset filtering by tag."""
        response = flask_client.get('/api/datasets?tag=climate')
        assert response.status_code == 200
        
        data = response.get_json()
        for dataset in data['data']:
            assert 'climate' in dataset['tags']
    
    def test_list_datasets_with_format_filter(self, flask_client):
        """Test dataset filtering by format."""
        response = flask_client.get('/api/datasets?format=CSV')
        assert response.status_code == 200
        
        data = response.get_json()
        for dataset in data['data']:
            assert dataset['format'] == 'CSV'
    
    def test_list_datasets_pagination(self, flask_client):
        """Test dataset pagination."""
        response = flask_client.get('/api/datasets?limit=1&offset=0')
        assert response.status_code == 200
        
        data = response.get_json()
        assert len(data['data']) == 1
        assert data['meta']['limit'] == 1
        assert data['meta']['offset'] == 0
    
    def test_get_dataset(self, flask_client):
        """Test single dataset retrieval."""
        response = flask_client.get('/api/datasets/1')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['id'] == 1
        assert 'name' in data
    
    def test_get_dataset_not_found(self, flask_client):
        """Test 404 for non-existent dataset."""
        response = flask_client.get('/api/datasets/999')
        assert response.status_code == 404
        
        data = response.get_json()
        assert 'error' in data
    
    def test_create_dataset(self, flask_client):
        """Test dataset creation."""
        new_dataset = {
            'name': 'Test Dataset',
            'description': 'A test dataset',
            'records': 100,
            'format': 'JSON',
            'tags': ['test']
        }
        
        response = flask_client.post(
            '/api/datasets',
            json=new_dataset,
            content_type='application/json'
        )
        assert response.status_code == 201
        
        data = response.get_json()
        assert data['name'] == 'Test Dataset'
        assert 'id' in data
        assert 'created_at' in data
    
    def test_create_dataset_missing_fields(self, flask_client):
        """Test 400 for missing required fields."""
        response = flask_client.post(
            '/api/datasets',
            json={'name': 'Incomplete'},
            content_type='application/json'
        )
        assert response.status_code == 400
    
    def test_create_dataset_invalid_content_type(self, flask_client):
        """Test 400 for non-JSON content type."""
        response = flask_client.post(
            '/api/datasets',
            data='not json',
            content_type='text/plain'
        )
        assert response.status_code == 400
    
    def test_update_dataset(self, flask_client):
        """Test dataset update."""
        update = {
            'name': 'Updated Name',
            'description': 'Updated description',
            'records': 200
        }
        
        response = flask_client.put(
            '/api/datasets/1',
            json=update,
            content_type='application/json'
        )
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['name'] == 'Updated Name'
        assert 'updated_at' in data
    
    def test_update_dataset_not_found(self, flask_client):
        """Test 404 for updating non-existent dataset."""
        response = flask_client.put(
            '/api/datasets/999',
            json={'name': 'Test'},
            content_type='application/json'
        )
        assert response.status_code == 404
    
    def test_delete_dataset(self, flask_client):
        """Test dataset deletion."""
        # First create a dataset to delete
        new_dataset = {
            'name': 'To Delete',
            'description': 'Will be deleted'
        }
        create_response = flask_client.post(
            '/api/datasets',
            json=new_dataset,
            content_type='application/json'
        )
        dataset_id = create_response.get_json()['id']
        
        # Delete it
        response = flask_client.delete(f'/api/datasets/{dataset_id}')
        assert response.status_code == 204
        
        # Verify it's gone
        get_response = flask_client.get(f'/api/datasets/{dataset_id}')
        assert get_response.status_code == 404
    
    def test_delete_dataset_not_found(self, flask_client):
        """Test 404 for deleting non-existent dataset."""
        response = flask_client.delete('/api/datasets/999')
        assert response.status_code == 404


# =============================================================================
# RESEARCH API TESTS
# =============================================================================

@pytest.mark.flask
class TestResearchAPI:
    """Tests for research data API."""
    
    def test_create_research_api(self, temp_data_file):
        """Test research API creation."""
        app = create_research_api(temp_data_file)
        assert app is not None
    
    def test_list_works(self, temp_data_file):
        """Test works listing endpoint."""
        app = create_research_api(temp_data_file)
        
        with app.test_client() as client:
            response = client.get('/api/works')
            assert response.status_code == 200
            
            data = response.get_json()
            assert 'data' in data
            assert 'total' in data
    
    def test_metadata_endpoint(self, temp_data_file):
        """Test metadata endpoint."""
        app = create_research_api(temp_data_file)
        
        with app.test_client() as client:
            response = client.get('/api/metadata')
            assert response.status_code == 200


# =============================================================================
# HTML SCRAPING TESTS
# =============================================================================

class TestHTMLScraping:
    """Tests for HTML scraping functionality."""
    
    def test_extract_titles(self, sample_html):
        """Test extracting titles from HTML."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        titles = [h.get_text(strip=True) for h in soup.select('h2.title')]
        
        assert len(titles) == 2
        assert 'Machine Learning for Climate' in titles
    
    def test_extract_links(self, sample_html):
        """Test extracting links from HTML."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        links = soup.select('a.link')
        
        assert len(links) == 2
        assert links[0]['href'] == '/papers/1'
    
    def test_extract_nested_content(self, sample_html):
        """Test extracting nested content."""
        soup = BeautifulSoup(sample_html, 'html.parser')
        articles = soup.select('article.publication')
        
        for article in articles:
            title = article.select_one('h2.title')
            authors = article.select_one('p.authors')
            assert title is not None
            assert authors is not None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
