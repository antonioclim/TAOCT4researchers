#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Hard 03 - Complete Data Pipeline

Difficulty: ★★★★★
Estimated Time: 45 minutes
Learning Objective: LO6, LO7

Task:
Design and implement a complete data acquisition and exposure pipeline.
Your system must retrieve data from a public API, process and store it,
then expose it through a custom REST API with full CRUD operations.

Requirements:
1. Consume data from a public research API (OpenAlex or CrossRef)
2. Process and normalise the acquired data
3. Store data persistently (JSON file or SQLite)
4. Expose data through a Flask REST API
5. Implement filtering, pagination and search
6. Include comprehensive error handling

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

import requests
from flask import Flask, jsonify, request, abort, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Work:
    """Scholarly work data model."""
    
    id: str
    title: str
    authors: list[str]
    year: int
    doi: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    citations_count: int = 0
    keywords: list[str] = field(default_factory=list)
    source_api: str = 'unknown'
    acquired_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Work':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# =============================================================================
# DATA ACQUISITION
# =============================================================================

class OpenAlexAcquirer:
    """
    Acquire scholarly work data from OpenAlex API.
    
    OpenAlex is a free, open catalogue of scholarly papers,
    authors, institutions and more.
    """
    
    BASE_URL = 'https://api.openalex.org'
    
    def __init__(self, email: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers['Accept'] = 'application/json'
        if email:
            self.session.headers['User-Agent'] = f'ResearchPipeline/1.0 (mailto:{email})'
    
    def search_works(
        self,
        query: str,
        max_results: int = 100,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None
    ) -> Iterator[Work]:
        """
        Search for works matching query.
        
        Args:
            query: Search query string
            max_results: Maximum works to retrieve
            year_from: Filter by publication year (minimum)
            year_to: Filter by publication year (maximum)
            
        Yields:
            Work objects from search results
        """
        # TODO: Implement OpenAlex search
        # HINT: Use cursor-based pagination
        # HINT: Build filter string for year range
        # HINT: Transform API response to Work objects
        
        pass  # Replace with your implementation
    
    def _transform_work(self, raw: dict[str, Any]) -> Work:
        """
        Transform raw OpenAlex response to Work model.
        
        Args:
            raw: Raw work data from API
            
        Returns:
            Normalised Work object
        """
        # TODO: Implement data transformation
        # HINT: Handle missing fields gracefully
        # HINT: Extract authors from authorships
        # HINT: Get keywords from concepts
        
        pass  # Replace with your implementation


# =============================================================================
# DATA STORAGE
# =============================================================================

class WorksRepository:
    """
    Persistent storage for scholarly works.
    
    Uses SQLite for reliable storage with query capabilities.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialise database schema."""
        # TODO: Create works table with appropriate schema
        # HINT: Store JSON for complex fields (authors, keywords)
        
        pass  # Replace with your implementation
    
    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save(self, work: Work) -> None:
        """
        Save or update a work in the repository.
        
        Args:
            work: Work object to save
        """
        # TODO: Implement upsert logic
        # HINT: Use INSERT OR REPLACE
        
        pass  # Replace with your implementation
    
    def save_many(self, works: list[Work]) -> int:
        """
        Save multiple works efficiently.
        
        Args:
            works: List of Work objects
            
        Returns:
            Number of works saved
        """
        # TODO: Implement batch insert
        
        pass  # Replace with your implementation
    
    def get(self, work_id: str) -> Optional[Work]:
        """
        Retrieve a work by ID.
        
        Args:
            work_id: Work identifier
            
        Returns:
            Work object or None if not found
        """
        # TODO: Implement retrieval
        
        pass  # Replace with your implementation
    
    def search(
        self,
        query: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> tuple[list[Work], int]:
        """
        Search works with filtering.
        
        Args:
            query: Text search in title and abstract
            year_from: Minimum publication year
            year_to: Maximum publication year
            limit: Maximum results
            offset: Results to skip
            
        Returns:
            Tuple of (works list, total count)
        """
        # TODO: Implement search with filters
        # HINT: Use LIKE for text search
        # HINT: Build WHERE clause dynamically
        
        pass  # Replace with your implementation
    
    def delete(self, work_id: str) -> bool:
        """
        Delete a work from repository.
        
        Args:
            work_id: Work identifier
            
        Returns:
            True if work was deleted
        """
        # TODO: Implement deletion
        
        pass  # Replace with your implementation
    
    def count(self) -> int:
        """Return total number of works."""
        # TODO: Implement count
        
        pass  # Replace with your implementation


# =============================================================================
# API LAYER
# =============================================================================

def create_pipeline_api(repository: WorksRepository) -> Flask:
    """
    Create REST API for the data pipeline.
    
    Endpoints:
        GET    /api/works          - List/search works (paginated)
        POST   /api/works          - Create work
        GET    /api/works/<id>     - Get single work
        PUT    /api/works/<id>     - Update work
        DELETE /api/works/<id>     - Delete work
        GET    /api/stats          - Get repository statistics
        POST   /api/acquire        - Trigger data acquisition
        
    Args:
        repository: Works repository instance
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # TODO: Implement all endpoints
    
    @app.route('/api/works', methods=['GET'])
    def list_works() -> Response:
        """
        List works with filtering and pagination.
        
        Query Parameters:
            q: Text search query
            year_from: Minimum year filter
            year_to: Maximum year filter
            limit: Results per page (default 20, max 100)
            offset: Results to skip
        """
        # TODO: Implement list with pagination
        
        pass  # Replace with your implementation
    
    @app.route('/api/works', methods=['POST'])
    def create_work() -> tuple[Response, int]:
        """Create a new work."""
        # TODO: Implement creation with validation
        
        pass  # Replace with your implementation
    
    @app.route('/api/works/<work_id>', methods=['GET'])
    def get_work(work_id: str) -> Response:
        """Get a single work by ID."""
        # TODO: Implement retrieval
        
        pass  # Replace with your implementation
    
    @app.route('/api/works/<work_id>', methods=['PUT'])
    def update_work(work_id: str) -> Response:
        """Update an existing work."""
        # TODO: Implement update
        
        pass  # Replace with your implementation
    
    @app.route('/api/works/<work_id>', methods=['DELETE'])
    def delete_work(work_id: str) -> tuple[str, int]:
        """Delete a work."""
        # TODO: Implement deletion
        
        pass  # Replace with your implementation
    
    @app.route('/api/stats', methods=['GET'])
    def get_stats() -> Response:
        """Get repository statistics."""
        # TODO: Return count, year distribution, etc.
        
        pass  # Replace with your implementation
    
    @app.route('/api/acquire', methods=['POST'])
    def acquire_data() -> Response:
        """
        Trigger data acquisition from external API.
        
        Request Body:
            query: Search query
            max_results: Maximum works to acquire
        """
        # TODO: Implement acquisition trigger
        # HINT: Create acquirer and fetch works
        # HINT: Save to repository
        # HINT: Return acquisition summary
        
        pass  # Replace with your implementation
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error: Any) -> tuple[Response, int]:
        return jsonify({'error': 'Bad Request', 'message': str(error.description)}), 400
    
    @app.errorhandler(404)
    def not_found(error: Any) -> tuple[Response, int]:
        return jsonify({'error': 'Not Found', 'message': str(error.description)}), 404
    
    @app.errorhandler(500)
    def server_error(error: Any) -> tuple[Response, int]:
        return jsonify({'error': 'Internal Server Error'}), 500
    
    return app


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

class DataPipeline:
    """
    Orchestrate the complete data pipeline.
    
    Coordinates acquisition, storage and API exposure.
    """
    
    def __init__(self, db_path: Path, email: Optional[str] = None):
        self.repository = WorksRepository(db_path)
        self.acquirer = OpenAlexAcquirer(email)
        self.app = create_pipeline_api(self.repository)
    
    def acquire(self, query: str, max_results: int = 100) -> int:
        """
        Acquire and store works matching query.
        
        Args:
            query: Search query
            max_results: Maximum works to acquire
            
        Returns:
            Number of works acquired
        """
        logger.info(f'Acquiring works for query: {query}')
        
        works = list(self.acquirer.search_works(query, max_results))
        saved = self.repository.save_many(works)
        
        logger.info(f'Acquired and saved {saved} works')
        return saved
    
    def run_api(self, host: str = '127.0.0.1', port: int = 5000) -> None:
        """Start the API server."""
        logger.info(f'Starting API server at http://{host}:{port}')
        self.app.run(host=host, port=port, debug=True)


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_exercises() -> None:
    """Run basic tests on exercise implementations."""
    import tempfile
    
    print('Testing Data Pipeline exercises...\n')
    
    # Test Work model
    print('Test 1: Work model')
    try:
        work = Work(
            id='test-123',
            title='Test Work',
            authors=['Alice', 'Bob'],
            year=2024,
            doi='10.1234/test'
        )
        d = work.to_dict()
        assert d['id'] == 'test-123'
        assert len(d['authors']) == 2
        
        work2 = Work.from_dict(d)
        assert work2.title == work.title
        
        print('  PASSED: Work model works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test WorksRepository
    print('\nTest 2: WorksRepository')
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'
            repo = WorksRepository(db_path)
            
            # Save work
            work = Work(
                id='test-1',
                title='Repository Test',
                authors=['Test Author'],
                year=2024
            )
            repo.save(work)
            
            # Retrieve work
            retrieved = repo.get('test-1')
            assert retrieved is not None, 'Work not found'
            assert retrieved.title == 'Repository Test'
            
            # Count
            assert repo.count() == 1
            
            # Delete
            assert repo.delete('test-1') is True
            assert repo.count() == 0
            
        print('  PASSED: Repository works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    # Test API
    print('\nTest 3: Pipeline API')
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / 'test.db'
            repo = WorksRepository(db_path)
            
            # Add test data
            repo.save(Work(
                id='api-test-1',
                title='API Test Work',
                authors=['Test'],
                year=2024
            ))
            
            app = create_pipeline_api(repo)
            
            with app.test_client() as client:
                # Test list
                response = client.get('/api/works')
                assert response.status_code == 200
                data = response.get_json()
                assert 'data' in data or 'works' in data
                
                # Test get
                response = client.get('/api/works/api-test-1')
                assert response.status_code == 200
                
                # Test 404
                response = client.get('/api/works/nonexistent')
                assert response.status_code == 404
                
        print('  PASSED: API works')
    except Exception as e:
        print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')
    print('Note: Full integration tests require OpenAlex API access.')


if __name__ == '__main__':
    test_exercises()
