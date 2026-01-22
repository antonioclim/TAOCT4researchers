#!/usr/bin/env python3
"""
12UNIT: Web APIs and Data Acquisition
Solutions for Hard Practice Exercises

Author: Dr Antonio Clim
Institution: Academy of Economic Studies, Bucharest
Licence: Restrictive — see repository root for terms
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from flask import Flask, jsonify, request, abort


# =============================================================================
# HARD 01: FLASK API SOLUTIONS
# =============================================================================

@dataclass
class Publication:
    """Publication data model."""
    id: str
    title: str
    authors: list[str]
    year: int
    doi: str | None = None
    abstract: str | None = None
    citations: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    updated_at: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_publications_api() -> Flask:
    """Create complete publications REST API."""
    app = Flask(__name__)
    
    # In-memory storage
    publications: dict[str, Publication] = {}
    next_id = 1
    
    def get_next_id() -> str:
        nonlocal next_id
        pub_id = f'PUB{next_id:05d}'
        next_id += 1
        return pub_id
    
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'publication_count': len(publications)
        })
    
    @app.route('/api/publications', methods=['GET'])
    def list_publications():
        # Filtering
        year = request.args.get('year', type=int)
        min_citations = request.args.get('min_citations', type=int)
        author = request.args.get('author')
        
        results = list(publications.values())
        
        if year:
            results = [p for p in results if p.year == year]
        if min_citations:
            results = [p for p in results if p.citations >= min_citations]
        if author:
            results = [p for p in results if any(author.lower() in a.lower() for a in p.authors)]
        
        # Pagination
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        total = len(results)
        results = results[offset:offset + limit]
        
        return jsonify({
            'data': [p.to_dict() for p in results],
            'meta': {
                'total': total,
                'limit': limit,
                'offset': offset
            },
            'links': {
                'self': f'/api/publications?limit={limit}&offset={offset}',
                'next': f'/api/publications?limit={limit}&offset={offset + limit}' if offset + limit < total else None
            }
        })
    
    @app.route('/api/publications', methods=['POST'])
    def create_publication():
        if not request.is_json:
            abort(400, description='Request must be JSON')
        
        data = request.get_json()
        
        required = ['title', 'authors', 'year']
        missing = [f for f in required if f not in data]
        if missing:
            abort(400, description=f'Missing required fields: {missing}')
        
        if not isinstance(data['authors'], list) or len(data['authors']) == 0:
            abort(400, description='authors must be a non-empty list')
        
        pub_id = get_next_id()
        pub = Publication(
            id=pub_id,
            title=data['title'],
            authors=data['authors'],
            year=data['year'],
            doi=data.get('doi'),
            abstract=data.get('abstract'),
            citations=data.get('citations', 0)
        )
        
        publications[pub_id] = pub
        
        response = jsonify(pub.to_dict())
        response.status_code = 201
        response.headers['Location'] = f'/api/publications/{pub_id}'
        return response
    
    @app.route('/api/publications/<pub_id>', methods=['GET'])
    def get_publication(pub_id: str):
        pub = publications.get(pub_id)
        if not pub:
            abort(404, description=f'Publication {pub_id} not found')
        return jsonify(pub.to_dict())
    
    @app.route('/api/publications/<pub_id>', methods=['PUT'])
    def update_publication(pub_id: str):
        pub = publications.get(pub_id)
        if not pub:
            abort(404, description=f'Publication {pub_id} not found')
        
        if not request.is_json:
            abort(400, description='Request must be JSON')
        
        data = request.get_json()
        
        # Update allowed fields
        if 'title' in data:
            pub.title = data['title']
        if 'authors' in data:
            pub.authors = data['authors']
        if 'year' in data:
            pub.year = data['year']
        if 'doi' in data:
            pub.doi = data['doi']
        if 'abstract' in data:
            pub.abstract = data['abstract']
        if 'citations' in data:
            pub.citations = data['citations']
        
        pub.updated_at = datetime.utcnow().isoformat() + 'Z'
        
        return jsonify(pub.to_dict())
    
    @app.route('/api/publications/<pub_id>', methods=['DELETE'])
    def delete_publication(pub_id: str):
        if pub_id not in publications:
            abort(404, description=f'Publication {pub_id} not found')
        
        del publications[pub_id]
        return '', 204
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': str(error.description),
            'status': 400
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': str(error.description),
            'status': 404
        }), 404
    
    return app


# =============================================================================
# HARD 02: API CLIENT SOLUTIONS
# =============================================================================

class Authenticator(ABC):
    """Abstract authenticator."""
    
    @abstractmethod
    def get_headers(self) -> dict[str, str]:
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        pass
    
    @abstractmethod
    def refresh(self) -> None:
        pass


class APIKeyAuth(Authenticator):
    """API key authentication."""
    
    def __init__(self, api_key: str, header_name: str = 'X-API-Key', placement: str = 'header'):
        self.api_key = api_key
        self.header_name = header_name
        self.placement = placement
    
    def get_headers(self) -> dict[str, str]:
        if self.placement == 'header':
            return {self.header_name: self.api_key}
        return {}
    
    def get_params(self) -> dict[str, str]:
        if self.placement == 'query':
            return {self.header_name: self.api_key}
        return {}
    
    def is_valid(self) -> bool:
        return True
    
    def refresh(self) -> None:
        pass


class TokenAuth(Authenticator):
    """Bearer token authentication with expiry."""
    
    def __init__(self, token: str, expires_at: datetime, refresh_callback=None):
        self.token = token
        self.expires_at = expires_at
        self.refresh_callback = refresh_callback
    
    def get_headers(self) -> dict[str, str]:
        return {'Authorization': f'Bearer {self.token}'}
    
    def is_valid(self) -> bool:
        return datetime.now() + timedelta(seconds=60) < self.expires_at
    
    def refresh(self) -> None:
        if self.refresh_callback:
            result = self.refresh_callback()
            self.token = result['token']
            self.expires_at = result['expires_at']


@dataclass
class CacheEntry:
    """Cache entry with expiry."""
    data: Any
    expires_at: datetime


class ResponseCache:
    """Response cache with TTL."""
    
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: dict[str, CacheEntry] = {}
    
    def _make_key(self, url: str, params: dict | None = None) -> str:
        param_str = json.dumps(sorted((params or {}).items()))
        return f'{url}|{param_str}'
    
    def get(self, url: str, params: dict | None = None) -> Any | None:
        key = self._make_key(url, params)
        entry = self._cache.get(key)
        
        if entry and datetime.now() < entry.expires_at:
            return entry.data
        
        if entry:
            del self._cache[key]
        
        return None
    
    def set(self, url: str, params: dict | None, data: Any) -> None:
        key = self._make_key(url, params)
        self._cache[key] = CacheEntry(
            data=data,
            expires_at=datetime.now() + self.ttl
        )
    
    def clear(self) -> None:
        self._cache.clear()


@dataclass
class ClientConfig:
    """API client configuration."""
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    page_size: int = 100
    user_agent: str = 'ResearchClient/1.0'


class APIError(Exception):
    """API error."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class ResearchAPIClient:
    """Complete API client with auth, caching and retry."""
    
    def __init__(
        self,
        config: ClientConfig,
        auth: Authenticator | None = None,
        cache: ResponseCache | None = None
    ):
        self.config = config
        self.auth = auth
        self.cache = cache or ResponseCache()
        
        import requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'application/json'
        })
    
    def _build_url(self, endpoint: str) -> str:
        endpoint = endpoint.lstrip('/')
        base = self.config.base_url.rstrip('/')
        return f'{base}/{endpoint}'
    
    def _should_retry(self, status_code: int, attempt: int) -> bool:
        if attempt >= self.config.max_retries:
            return False
        return status_code == 429 or status_code >= 500
    
    def _calculate_delay(self, attempt: int) -> float:
        delay = self.config.base_delay * (2 ** (attempt - 1))
        return min(delay, self.config.max_delay)
    
    def get(
        self,
        endpoint: str,
        params: dict | None = None,
        use_cache: bool = True
    ) -> dict[str, Any]:
        url = self._build_url(endpoint)
        
        # Check cache
        if use_cache:
            cached = self.cache.get(url, params)
            if cached:
                return cached
        
        # Refresh auth if needed
        if self.auth and not self.auth.is_valid():
            self.auth.refresh()
        
        # Build headers
        headers = {}
        if self.auth:
            headers.update(self.auth.get_headers())
        
        # Request with retry
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                if response.status_code < 400:
                    data = response.json()
                    if use_cache:
                        self.cache.set(url, params, data)
                    return data
                
                if self._should_retry(response.status_code, attempt):
                    time.sleep(self._calculate_delay(attempt))
                    continue
                
                raise APIError(
                    f'Request failed: {response.status_code}',
                    response.status_code
                )
            
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise APIError(f'Request failed after {attempt} attempts: {e}')
                time.sleep(self._calculate_delay(attempt))
        
        raise APIError('Max retries exceeded')
    
    def get_paginated(
        self,
        endpoint: str,
        params: dict | None = None,
        max_items: int | None = None
    ) -> Iterator[dict[str, Any]]:
        params = dict(params or {})
        params.setdefault('limit', self.config.page_size)
        
        offset = 0
        yielded = 0
        
        while True:
            params['offset'] = offset
            response = self.get(endpoint, params, use_cache=False)
            
            items = response.get('data', response.get('results', []))
            
            for item in items:
                yield item
                yielded += 1
                if max_items and yielded >= max_items:
                    return
            
            if len(items) < params['limit']:
                break
            
            offset += len(items)
    
    def post(self, endpoint: str, data: dict[str, Any]) -> dict[str, Any]:
        url = self._build_url(endpoint)
        
        headers = {'Content-Type': 'application/json'}
        if self.auth:
            headers.update(self.auth.get_headers())
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                if response.status_code < 400:
                    return response.json()
                
                if self._should_retry(response.status_code, attempt):
                    time.sleep(self._calculate_delay(attempt))
                    continue
                
                raise APIError(
                    f'POST failed: {response.status_code}',
                    response.status_code
                )
            
            except Exception as e:
                if attempt == self.config.max_retries:
                    raise APIError(f'POST failed after {attempt} attempts: {e}')
                time.sleep(self._calculate_delay(attempt))
        
        raise APIError('Max retries exceeded')


# =============================================================================
# HARD 03: DATA PIPELINE SOLUTIONS
# =============================================================================

@dataclass
class PipelinePublication:
    """Publication for pipeline."""
    id: str
    title: str
    authors: list[str]
    year: int
    doi: str | None = None
    abstract: str | None = None
    citations: int = 0
    keywords: list[str] = field(default_factory=list)
    source: str = 'unknown'
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'PipelinePublication':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class MockDataSource:
    """Mock data source."""
    
    _author_names = ['Smith, J.', 'Chen, L.', 'Johnson, M.', 'Garcia, A.', 'Lee, S.', 'Brown, R.']
    
    @property
    def source_name(self) -> str:
        return 'mock'
    
    def fetch(self, query: str, max_results: int = 100) -> list[PipelinePublication]:
        publications = []
        
        for i in range(max_results):
            num_authors = random.randint(1, 3)
            authors = random.sample(self._author_names, num_authors)
            
            pub = PipelinePublication(
                id=f'MOCK{i+1:05d}',
                title=f'{query.title()} Study {i+1}',
                authors=authors,
                year=random.randint(2020, 2024),
                citations=random.randint(0, 100),
                source=self.source_name
            )
            publications.append(pub)
        
        return publications


class DataTransformer:
    """Transform and validate data."""
    
    def transform(self, pub: PipelinePublication) -> PipelinePublication:
        pub.title = pub.title.strip()
        pub.authors = [a.strip() for a in pub.authors]
        
        if pub.abstract:
            pub.abstract = pub.abstract.strip()
        
        current_year = datetime.now().year
        pub.year = max(1900, min(pub.year, current_year))
        pub.citations = max(0, pub.citations)
        
        return pub
    
    def transform_batch(self, pubs: list[PipelinePublication]) -> list[PipelinePublication]:
        return [self.transform(p) for p in pubs]
    
    def validate(self, pub: PipelinePublication) -> list[str]:
        issues = []
        
        if not pub.id:
            issues.append('ID is required')
        if not pub.title:
            issues.append('Title is required')
        if not pub.authors:
            issues.append('At least one author required')
        
        current_year = datetime.now().year
        if not (1900 <= pub.year <= current_year):
            issues.append(f'Year must be between 1900 and {current_year}')
        
        return issues


class DataStore:
    """Persistent publication storage."""
    
    def __init__(self, path: Path):
        self.path = path
    
    def save(self, publications: list[PipelinePublication]) -> None:
        data = {
            'metadata': {
                'count': len(publications),
                'updated_at': datetime.utcnow().isoformat() + 'Z'
            },
            'publications': [p.to_dict() for p in publications]
        }
        
        self.path.write_text(json.dumps(data, indent=2))
    
    def load(self) -> list[PipelinePublication]:
        if not self.path.exists():
            return []
        
        data = json.loads(self.path.read_text())
        return [PipelinePublication.from_dict(p) for p in data.get('publications', [])]
    
    def append(self, publications: list[PipelinePublication]) -> None:
        existing = {p.id: p for p in self.load()}
        
        for pub in publications:
            existing[pub.id] = pub
        
        self.save(list(existing.values()))
    
    def query(
        self,
        year: int | None = None,
        min_citations: int | None = None,
        keyword: str | None = None
    ) -> list[PipelinePublication]:
        results = self.load()
        
        if year:
            results = [p for p in results if p.year == year]
        if min_citations:
            results = [p for p in results if p.citations >= min_citations]
        if keyword:
            keyword = keyword.lower()
            results = [p for p in results if keyword in p.title.lower() or keyword in ' '.join(p.keywords).lower()]
        
        return results


@dataclass
class PipelineStats:
    """Pipeline execution statistics."""
    fetched: int = 0
    transformed: int = 0
    valid: int = 0
    invalid: int = 0
    stored: int = 0
    errors: list[str] = field(default_factory=list)


class DataPipeline:
    """Data acquisition pipeline."""
    
    def __init__(
        self,
        source,
        store: DataStore,
        transformer: DataTransformer | None = None
    ):
        self.source = source
        self.store = store
        self.transformer = transformer or DataTransformer()
    
    def run(
        self,
        query: str,
        max_results: int = 100,
        validate: bool = True
    ) -> PipelineStats:
        stats = PipelineStats()
        
        # Fetch
        try:
            publications = self.source.fetch(query, max_results)
            stats.fetched = len(publications)
        except Exception as e:
            stats.errors.append(f'Fetch error: {e}')
            return stats
        
        # Transform
        publications = self.transformer.transform_batch(publications)
        stats.transformed = len(publications)
        
        # Validate
        valid_pubs = []
        if validate:
            for pub in publications:
                issues = self.transformer.validate(pub)
                if issues:
                    stats.invalid += 1
                    stats.errors.extend(issues)
                else:
                    stats.valid += 1
                    valid_pubs.append(pub)
        else:
            valid_pubs = publications
            stats.valid = len(publications)
        
        # Store
        self.store.save(valid_pubs)
        stats.stored = len(valid_pubs)
        
        return stats
    
    def run_incremental(
        self,
        query: str,
        max_results: int = 100
    ) -> PipelineStats:
        stats = PipelineStats()
        
        try:
            publications = self.source.fetch(query, max_results)
            stats.fetched = len(publications)
        except Exception as e:
            stats.errors.append(f'Fetch error: {e}')
            return stats
        
        publications = self.transformer.transform_batch(publications)
        stats.transformed = len(publications)
        
        valid_pubs = []
        for pub in publications:
            issues = self.transformer.validate(pub)
            if issues:
                stats.invalid += 1
            else:
                stats.valid += 1
                valid_pubs.append(pub)
        
        self.store.append(valid_pubs)
        stats.stored = len(valid_pubs)
        
        return stats


def create_pipeline_api(store: DataStore) -> Flask:
    """Create Flask API for pipeline data."""
    app = Flask(__name__)
    
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
    
    @app.route('/api/publications', methods=['GET'])
    def list_publications():
        year = request.args.get('year', type=int)
        min_citations = request.args.get('min_citations', type=int)
        keyword = request.args.get('keyword')
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        results = store.query(year=year, min_citations=min_citations, keyword=keyword)
        total = len(results)
        results = results[offset:offset + limit]
        
        return jsonify({
            'data': [p.to_dict() for p in results],
            'meta': {'total': total, 'limit': limit, 'offset': offset}
        })
    
    @app.route('/api/publications/<pub_id>', methods=['GET'])
    def get_publication(pub_id: str):
        pubs = store.load()
        pub = next((p for p in pubs if p.id == pub_id), None)
        if not pub:
            abort(404)
        return jsonify(pub.to_dict())
    
    @app.route('/api/stats', methods=['GET'])
    def get_stats():
        pubs = store.load()
        return jsonify({
            'total': len(pubs),
            'by_year': {},
            'total_citations': sum(p.citations for p in pubs)
        })
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not Found'}), 404
    
    return app


# =============================================================================
# TEST ALL SOLUTIONS
# =============================================================================

if __name__ == '__main__':
    # Test Flask API
    app = create_publications_api()
    with app.test_client() as client:
        resp = client.get('/api/health')
        assert resp.status_code == 200
        print('✓ Flask API health check passed')
    
    # Test API Client components
    auth = APIKeyAuth('test_key', 'X-API-Key')
    assert auth.get_headers() == {'X-API-Key': 'test_key'}
    print('✓ API Key Auth passed')
    
    token_auth = TokenAuth('token123', datetime.now() + timedelta(hours=1))
    assert token_auth.is_valid() is True
    print('✓ Token Auth passed')
    
    cache = ResponseCache(300)
    cache.set('/test', {'q': 'a'}, {'data': 'cached'})
    assert cache.get('/test', {'q': 'a'}) == {'data': 'cached'}
    print('✓ Response Cache passed')
    
    # Test Pipeline components
    source = MockDataSource()
    pubs = source.fetch('test', 5)
    assert len(pubs) == 5
    print('✓ Mock Data Source passed')
    
    transformer = DataTransformer()
    pub = PipelinePublication(id='1', title='  Test  ', authors=['A'], year=2024)
    clean = transformer.transform(pub)
    assert clean.title == 'Test'
    print('✓ Data Transformer passed')
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        store = DataStore(Path(tmp) / 'test.json')
        store.save(pubs)
        loaded = store.load()
        assert len(loaded) == 5
        print('✓ Data Store passed')
        
        pipeline = DataPipeline(source, store)
        stats = pipeline.run('test', 10)
        assert stats.stored > 0
        print('✓ Data Pipeline passed')
    
    print('\n✓ All hard solutions verified!')
