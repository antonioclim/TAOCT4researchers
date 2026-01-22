#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Exercise: Hard 01 - Complete Flask REST API

Difficulty: ★★★★★
Estimated Time: 40 minutes
Learning Objective: LO6

Task:
Design and implement a complete REST API for a research publication
database. Your API must follow REST conventions, handle errors gracefully
and support filtering, pagination and CRUD operations.

Requirements:
1. Implement all CRUD endpoints (Create, Read, Update, Delete)
2. Support query-based filtering
3. Implement cursor-based pagination
4. Validate input data with meaningful error messages
5. Return appropriate HTTP status codes
6. Include HATEOAS-style links in responses

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional

from flask import Flask, jsonify, request, abort, url_for, Response


@dataclass
class Publication:
    """Research publication data model."""
    
    id: str
    title: str
    authors: list[str]
    abstract: str
    year: int
    doi: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    citations: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    updated_at: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Publication':
        """Create Publication from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def create_publications_api() -> Flask:
    """
    Create a complete REST API for research publications.
    
    Endpoints:
        GET    /api/publications          - List publications (paginated)
        POST   /api/publications          - Create publication
        GET    /api/publications/<id>     - Get single publication
        PUT    /api/publications/<id>     - Update publication
        DELETE /api/publications/<id>     - Delete publication
        GET    /api/publications/<id>/citations - Get citing publications
    
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # In-memory storage
    publications: dict[str, Publication] = {}
    
    # Seed some sample data
    samples = [
        Publication(
            id=str(uuid.uuid4()),
            title='Machine Learning for Climate Prediction',
            authors=['Smith, J.', 'Chen, L.'],
            abstract='We present a novel approach to climate prediction...',
            year=2023,
            doi='10.1234/example.2023.001',
            keywords=['machine learning', 'climate', 'prediction'],
            citations=15
        ),
        Publication(
            id=str(uuid.uuid4()),
            title='Deep Learning in Genomics',
            authors=['Johnson, M.', 'Williams, K.', 'Brown, R.'],
            abstract='This paper surveys deep learning applications...',
            year=2024,
            doi='10.1234/example.2024.002',
            keywords=['deep learning', 'genomics', 'bioinformatics'],
            citations=8
        )
    ]
    for pub in samples:
        publications[pub.id] = pub
    
    # -------------------------------------------------------------------------
    # TODO: Implement list publications endpoint with pagination
    # -------------------------------------------------------------------------
    
    @app.route('/api/publications', methods=['GET'])
    def list_publications() -> Response:
        """
        List publications with filtering and pagination.
        
        Query Parameters:
            year: Filter by publication year
            author: Filter by author name (partial match)
            keyword: Filter by keyword
            cursor: Pagination cursor
            limit: Results per page (default 10, max 100)
            
        Returns:
            JSON with 'data', 'meta' and '_links' keys
        """
        # TODO: Implement filtering
        # HINT: Get parameters from request.args
        # HINT: Apply filters to publications.values()
        
        # TODO: Implement cursor-based pagination
        # HINT: Use publication ID as cursor
        # HINT: Sort consistently for stable pagination
        
        # TODO: Build response with HATEOAS links
        # HINT: Include 'self', 'next', 'prev' links
        
        pass  # Replace with your implementation
    
    # -------------------------------------------------------------------------
    # TODO: Implement create publication endpoint
    # -------------------------------------------------------------------------
    
    @app.route('/api/publications', methods=['POST'])
    def create_publication() -> tuple[Response, int]:
        """
        Create a new publication.
        
        Request Body:
            title: Publication title (required)
            authors: List of author names (required)
            abstract: Publication abstract (required)
            year: Publication year (required)
            doi: Optional DOI
            keywords: Optional list of keywords
            
        Returns:
            Created publication with 201 status
        """
        # TODO: Validate content type
        # TODO: Validate required fields
        # TODO: Create Publication object
        # TODO: Return with 201 status and Location header
        
        pass  # Replace with your implementation
    
    # -------------------------------------------------------------------------
    # TODO: Implement get single publication endpoint
    # -------------------------------------------------------------------------
    
    @app.route('/api/publications/<publication_id>', methods=['GET'])
    def get_publication(publication_id: str) -> Response:
        """
        Get a single publication by ID.
        
        Returns:
            Publication object with _links
        """
        # TODO: Look up publication
        # TODO: Return 404 if not found
        # TODO: Include HATEOAS links
        
        pass  # Replace with your implementation
    
    # -------------------------------------------------------------------------
    # TODO: Implement update publication endpoint
    # -------------------------------------------------------------------------
    
    @app.route('/api/publications/<publication_id>', methods=['PUT'])
    def update_publication(publication_id: str) -> Response:
        """
        Update an existing publication.
        
        Returns:
            Updated publication object
        """
        # TODO: Validate publication exists
        # TODO: Validate request body
        # TODO: Update fields (preserve id and created_at)
        # TODO: Set updated_at timestamp
        
        pass  # Replace with your implementation
    
    # -------------------------------------------------------------------------
    # TODO: Implement delete publication endpoint
    # -------------------------------------------------------------------------
    
    @app.route('/api/publications/<publication_id>', methods=['DELETE'])
    def delete_publication(publication_id: str) -> tuple[str, int]:
        """
        Delete a publication.
        
        Returns:
            Empty response with 204 status
        """
        # TODO: Validate publication exists
        # TODO: Remove from storage
        # TODO: Return 204 No Content
        
        pass  # Replace with your implementation
    
    # -------------------------------------------------------------------------
    # Error handlers
    # -------------------------------------------------------------------------
    
    @app.errorhandler(400)
    def bad_request(error: Any) -> tuple[Response, int]:
        return jsonify({
            'error': 'Bad Request',
            'message': str(error.description),
            'status': 400
        }), 400
    
    @app.errorhandler(404)
    def not_found(error: Any) -> tuple[Response, int]:
        return jsonify({
            'error': 'Not Found',
            'message': str(error.description),
            'status': 404
        }), 404
    
    @app.errorhandler(422)
    def unprocessable(error: Any) -> tuple[Response, int]:
        return jsonify({
            'error': 'Unprocessable Entity',
            'message': str(error.description),
            'status': 422
        }), 422
    
    return app


# =============================================================================
# TEST YOUR IMPLEMENTATION
# =============================================================================

def test_api() -> None:
    """Test the publications API."""
    print('Testing Publications API...\n')
    
    app = create_publications_api()
    
    with app.test_client() as client:
        # Test list publications
        print('Test 1: List publications')
        try:
            response = client.get('/api/publications')
            assert response.status_code == 200, f'Expected 200, got {response.status_code}'
            data = response.get_json()
            assert 'data' in data, 'Expected data key'
            assert 'meta' in data, 'Expected meta key'
            print(f'  PASSED: Listed {len(data["data"])} publications')
        except Exception as e:
            print(f'  FAILED: {e}')
        
        # Test create publication
        print('\nTest 2: Create publication')
        try:
            new_pub = {
                'title': 'Test Publication',
                'authors': ['Test Author'],
                'abstract': 'This is a test abstract.',
                'year': 2024
            }
            response = client.post(
                '/api/publications',
                json=new_pub,
                content_type='application/json'
            )
            assert response.status_code == 201, f'Expected 201, got {response.status_code}'
            created = response.get_json()
            assert 'id' in created, 'Expected id in response'
            pub_id = created['id']
            print(f'  PASSED: Created publication {pub_id}')
        except Exception as e:
            print(f'  FAILED: {e}')
            pub_id = None
        
        # Test get single publication
        if pub_id:
            print('\nTest 3: Get publication')
            try:
                response = client.get(f'/api/publications/{pub_id}')
                assert response.status_code == 200
                data = response.get_json()
                assert data['title'] == 'Test Publication'
                print('  PASSED: Retrieved publication')
            except Exception as e:
                print(f'  FAILED: {e}')
        
        # Test update publication
        if pub_id:
            print('\nTest 4: Update publication')
            try:
                update = {
                    'title': 'Updated Title',
                    'authors': ['Test Author'],
                    'abstract': 'Updated abstract.',
                    'year': 2024,
                    'citations': 5
                }
                response = client.put(
                    f'/api/publications/{pub_id}',
                    json=update,
                    content_type='application/json'
                )
                assert response.status_code == 200
                data = response.get_json()
                assert data['title'] == 'Updated Title'
                assert data['citations'] == 5
                print('  PASSED: Updated publication')
            except Exception as e:
                print(f'  FAILED: {e}')
        
        # Test delete publication
        if pub_id:
            print('\nTest 5: Delete publication')
            try:
                response = client.delete(f'/api/publications/{pub_id}')
                assert response.status_code == 204
                # Verify deletion
                response = client.get(f'/api/publications/{pub_id}')
                assert response.status_code == 404
                print('  PASSED: Deleted publication')
            except Exception as e:
                print(f'  FAILED: {e}')
        
        # Test 404 for non-existent
        print('\nTest 6: Handle non-existent publication')
        try:
            response = client.get('/api/publications/non-existent-id')
            assert response.status_code == 404
            print('  PASSED: Returns 404 for missing')
        except Exception as e:
            print(f'  FAILED: {e}')
        
        # Test validation
        print('\nTest 7: Validate required fields')
        try:
            invalid = {'title': 'Missing required fields'}
            response = client.post(
                '/api/publications',
                json=invalid,
                content_type='application/json'
            )
            assert response.status_code == 400, f'Expected 400, got {response.status_code}'
            print('  PASSED: Validates required fields')
        except Exception as e:
            print(f'  FAILED: {e}')
    
    print('\nAll tests complete.')


if __name__ == '__main__':
    test_api()
