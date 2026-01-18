# 12UNIT: Web APIs and Data Acquisition — Homework Assignment

## Overview

**Title:** Research Data Pipeline: Acquisition and Exposure  
**Due:** One week after laboratory completion  
**Weight:** 30% of unit grade  
**Estimated Time:** 2-3 hours

This assignment synthesises the core competencies developed throughout the unit: HTTP protocol understanding, RESTful API consumption, error handling and Flask-based API development. The deliverable represents a complete data pipeline suitable for integration into research workflows.

---

## Learning Objectives Assessed

- LO2: Apply API consumption techniques using requests
- LO4: Handle API errors and implement retry logic
- LO6: Design and implement Flask REST APIs
- LO7: Evaluate data acquisition strategies

---

## Task Description

You will build a complete data pipeline that:

1. **Acquires** research-relevant data from a public API
2. **Processes** the data into a consistent format
3. **Exposes** the processed data through your own REST API
4. **Documents** your design decisions and trade-offs

This assignment simulates a common research computing task: making external data accessible to colleagues through a standardised interface. The skills developed transfer directly to bibliometric analysis, environmental data collection, social science research and numerous other computational research domains.

---

## Requirements

### Part 1: Data Acquisition (10 marks)

Select ONE of the following public APIs as your data source:

**Option A: OpenAlex (Recommended)**
- Endpoint: `https://api.openalex.org/works`
- Documentation: https://docs.openalex.org/
- Task: Retrieve publications matching a research topic of your choice
- Advantages: No authentication required, excellent documentation, rich metadata

**Option B: CrossRef**
- Endpoint: `https://api.crossref.org/works`
- Documentation: https://api.crossref.org/swagger-ui/index.html
- Task: Retrieve publications from a specific journal or author
- Advantages: Comprehensive citation data, well-established service

**Option C: Open Meteo (Weather)**
- Endpoint: `https://api.open-meteo.com/v1/forecast`
- Documentation: https://open-meteo.com/en/docs
- Task: Retrieve historical weather data for research locations
- Advantages: No API key required, supports historical data queries

**Implementation Requirements:**

```python
# Your acquisition module must implement this interface:

from pathlib import Path
from typing import Optional


class APIError(Exception):
    """Custom exception for API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def fetch_data(
    query: str,
    max_results: int = 100,
    cache_dir: Path | None = None
) -> list[dict]:
    """
    Fetch data from the selected API.
    
    This function handles the complete data acquisition workflow,
    including pagination, rate limiting and caching. It should be
    idempotent for the same query parameters when caching is enabled.
    
    Args:
        query: Search query or filter parameters. The exact format
               depends on your chosen API (e.g., "machine learning"
               for OpenAlex topic search).
        max_results: Maximum items to retrieve. The function should
                     handle pagination automatically to collect up to
                     this many records.
        cache_dir: Optional directory for response caching. When
                   provided, responses should be cached to avoid
                   redundant API calls during development.
        
    Returns:
        List of processed data records in dictionary format.
        Each dictionary should contain consistent keys regardless
        of missing fields in the source data.
        
    Raises:
        APIError: If API request fails after all retry attempts.
        ValueError: If query parameters are invalid.
    
    Example:
        >>> records = fetch_data("climate change", max_results=50)
        >>> len(records) <= 50
        True
        >>> all("title" in r for r in records)
        True
    """
    ...
```

**Must Include:**
- Pagination handling (collect all results up to max_results)
- Exponential backoff retry logic (minimum 3 retries with increasing delays)
- Rate limit compliance (respect API-specified limits or default to 1 request/second)
- Response caching (optional but strongly recommended for development efficiency)
- Proper error handling with custom exceptions
- Type hints throughout all functions and classes

---

### Part 2: Data Processing (5 marks)

Transform the raw API responses into a consistent format suitable for your Flask API. This separation of concerns isolates the API-specific parsing logic from the data serving logic.

**Implementation Requirements:**

```python
from dataclasses import dataclass, asdict
from datetime import date
from typing import Any, Optional


@dataclass
class ProcessedRecord:
    """
    Standardised data record.
    
    This dataclass defines the canonical format for records served
    by your API. Fields should be meaningful for your chosen data
    domain whilst remaining general enough for various use cases.
    """
    id: str                          # Unique identifier
    title: str                       # Human-readable title
    created_date: Optional[date]     # When the record was created
    # Add 3-5 additional fields appropriate to your chosen API
    # Example for publications: authors, journal, citation_count
    # Example for weather: location, temperature, precipitation
    
    def to_dict(self) -> dict[str, Any]:
        """
        Convert to JSON-serialisable dictionary.
        
        Handles date serialisation and any other non-JSON types.
        
        Returns:
            Dictionary ready for JSON encoding.
        """
        result = asdict(self)
        # Handle date serialisation
        if self.created_date:
            result['created_date'] = self.created_date.isoformat()
        return result


def process_raw_data(raw_records: list[dict]) -> list[ProcessedRecord]:
    """
    Transform raw API data into standardised format.
    
    This function handles all the messy details of API response
    parsing: missing fields, inconsistent formats, malformed data.
    The output should be clean, consistent and ready for serving.
    
    Args:
        raw_records: Raw records from API in their original format.
        
    Returns:
        List of processed records. Records that cannot be processed
        (e.g., missing required fields) should be skipped with a
        warning logged.
    """
    ...
```

**Must Include:**
- Dataclass or TypedDict for structured data
- Field validation and cleaning (handle None, empty strings, malformed data)
- Handling of missing/malformed data (skip with warning, do not crash)
- Consistent date/time formatting (ISO 8601)

---

### Part 3: Flask API (10 marks)

Expose your processed data through a RESTful API following industry conventions.

**Required Endpoints:**

| Method | Endpoint | Description | Response Code |
|--------|----------|-------------|---------------|
| GET | `/api/health` | Health check | 200 |
| GET | `/api/records` | List records (paginated, filterable) | 200 |
| GET | `/api/records/<id>` | Get single record | 200, 404 |
| GET | `/api/statistics` | Aggregate statistics | 200 |
| POST | `/api/refresh` | Trigger data refresh | 202, 500 |

**Implementation Requirements:**

```python
from flask import Flask, jsonify, request


def create_app(data_source: DataSource) -> Flask:
    """
    Create Flask application with configured routes.
    
    Uses the application factory pattern to enable testing
    with different data sources.
    
    Args:
        data_source: Data source providing records. This abstraction
                     allows testing with mock data.
        
    Returns:
        Configured Flask application ready to run.
    """
    app = Flask(__name__)
    
    @app.route('/api/health')
    def health():
        """Return service health status."""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'record_count': data_source.count()
        })
    
    # Implement remaining endpoints...
    
    return app
```

**Must Include:**
- Proper HTTP status codes (200, 201, 202, 400, 404, 500)
- JSON error responses with meaningful messages
- Query parameter filtering (at least 2 filter options, e.g., `?year=2024&topic=AI`)
- Pagination with limit/offset or cursor-based approach
- CORS headers if appropriate for browser access
- Request logging for debugging

---

### Part 4: Documentation (5 marks)

Create a `DESIGN.md` file documenting your implementation decisions:

1. **API Selection Rationale** (1 mark)
   - Why you chose this particular API
   - How the data relates to potential research scenarios
   - Any challenges you anticipated based on the API documentation

2. **Architecture Decisions** (2 marks)
   - How the three main components (acquisition, processing, API) interact
   - Caching strategy: what is cached, for how long, cache invalidation
   - Error handling approach: how failures propagate, what gets logged

3. **Trade-offs and Limitations** (1 mark)
   - What you would implement differently given more time
   - Known limitations of your current implementation
   - Scalability considerations for larger datasets

4. **Usage Examples** (1 mark)
   - curl commands demonstrating each endpoint
   - Example request/response pairs
   - Instructions for running the test suite

---

## Deliverables

Submit a ZIP archive containing:

```
homework_12/
├── README.md           # Setup and running instructions
├── DESIGN.md           # Design documentation
├── requirements.txt    # Python dependencies (pinned versions)
├── acquisition.py      # Data acquisition module
├── processing.py       # Data processing module
├── api.py              # Flask API
├── models.py           # Data models (dataclasses/TypedDicts)
├── tests/
│   ├── __init__.py
│   ├── conftest.py     # Shared fixtures
│   ├── test_acquisition.py
│   ├── test_processing.py
│   └── test_api.py
└── data/               # Sample cached data (optional)
    └── .gitkeep
```

---

## Evaluation Criteria

### Technical Correctness (15 marks)

| Criterion | Marks | Description |
|-----------|-------|-------------|
| API consumption works correctly | 4 | Successfully retrieves data from chosen API |
| Pagination and retry logic | 3 | Handles multi-page results and transient failures |
| Data processing is reliable | 3 | Transforms data consistently, handles edge cases |
| Flask endpoints function correctly | 3 | All five endpoints work as specified |
| Error handling is comprehensive | 2 | Meaningful errors, no unhandled exceptions |

### Code Quality (10 marks)

| Criterion | Marks | Description |
|-----------|-------|-------------|
| Type hints throughout | 2 | All functions have complete type annotations |
| Clear function/class documentation | 2 | Docstrings explain purpose, args, returns |
| PEP 8 style compliance | 2 | Consistent style, passes ruff or flake8 |
| Appropriate abstraction | 2 | Clean separation of concerns, no duplication |
| Test coverage >60% | 2 | Meaningful tests covering main paths |

### Documentation (5 marks)

| Criterion | Marks | Description |
|-----------|-------|-------------|
| Clear setup instructions | 1 | Can run the project following README |
| Design rationale explained | 2 | Decisions are justified, not just described |
| Usage examples provided | 1 | Working curl commands for all endpoints |
| Trade-offs acknowledged | 1 | Honest assessment of limitations |

---

## Getting Started

1. **Choose your API** and explore its documentation thoroughly
2. **Fetch sample data** manually using curl or a browser to understand the response structure
3. **Design your data model** based on the fields most useful for research scenarios
4. **Implement acquisition** with basic functionality first (single page, no retry)
5. **Add reliability features** (retry, caching, error handling)
6. **Build Flask API** incrementally, testing each endpoint before moving on
7. **Write tests** as you develop—do not leave testing until the end
8. **Document** your decisions whilst they are fresh in your mind

---

## Troubleshooting Guide

| Problem | Likely Cause | Solution |
|---------|--------------|----------|
| `ConnectionError` | Network unavailable | Verify internet connection; check firewall settings |
| `Timeout` | API server slow | Increase timeout value; implement retry with backoff |
| `403 Forbidden` | Missing/invalid authentication | Verify API key; check rate limit status |
| `429 Too Many Requests` | Rate limited | Implement exponential backoff; reduce request frequency |
| Empty results | Incorrect query format | Review API documentation; test query in browser first |
| JSON decode error | Non-JSON response | Check status code first; API may return HTML errors |

---

## Tips for Success

- Start with the simplest working version, then add features iteratively
- Use httpbin.org to test retry logic without hitting real APIs
- Keep API credentials in environment variables, never in code
- Test edge cases: empty results, malformed data, network failures
- Review the lab solutions for patterns to follow
- Run your code frequently during development—do not write 200 lines before testing
- Use logging liberally during development; reduce verbosity before submission

---

## Submission Checklist

Before submitting, verify the following:

- [ ] All files are present in the correct directory structure
- [ ] `requirements.txt` contains all dependencies with pinned versions
- [ ] Code runs without errors: `python -c "from api import create_app; create_app(...)"`
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Type checking passes: `mypy *.py`
- [ ] Style checking passes: `ruff check .`
- [ ] README.md contains clear setup instructions
- [ ] DESIGN.md addresses all four required sections
- [ ] No API keys or credentials are included in the submission

---

## Academic Integrity

This is an individual assignment. You may:
- Reference documentation and tutorials
- Discuss concepts with classmates at a high level
- Use code from the laboratory exercises as a starting point

You may NOT:
- Share code with other students
- Submit code written by others
- Use AI code generation tools without explicit attribution

---

## Submission

Submit via the course portal by the deadline. Late submissions incur a 10% penalty per day up to 5 days, after which no submissions are accepted.

Questions about the assignment should be posted to the course forum so all students benefit from the answers.

---

*Licence: Restrictive — see repository root for terms.*
