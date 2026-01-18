# 12UNIT: Web APIs and Data Acquisition

## The Art of Computational Thinking for Researchers

![UNIT Badge](assets/images/12UNIT_badge.svg)

> **UNIT 12 of 14** | **Bloom Level**: Apply/Create | **Est. Time**: 8-10 hours

---

## 1. UNIT Synopsis

### 1.1 Abstract

The contemporary research field increasingly depends upon programmatic access to web-based data sources—from bibliographic repositories and social media platforms to governmental statistical portals and institutional research databases. This unit develops comprehensive competencies for web-based data acquisition, encompassing RESTful API consumption, authentication mechanisms, rate limiting strategies and the creation of lightweight APIs to disseminate research outputs.

Computational approaches have transformed how researchers collaborate and communicate. Shared codebases enable researchers to build on each other's work, creating cumulative knowledge that extends beyond what any individual could achieve. These collaborative aspects highlight a broader transformation in scientific practice—from individual insight to collective intelligence. Where traditional research might rely on the brilliance of individual scientists, computational approaches often utilise the distributed expertise of diverse teams.

The laboratory components instantiate these principles through the `requests` library for HTTP interactions, `BeautifulSoup` for HTML parsing where APIs prove unavailable, and `Flask` for exposing research data through custom endpoints. Ethical considerations permeate the treatment—respecting rate limits, honouring `robots.txt` directives and acknowledging the terms of service that govern data access represent professional obligations rather than mere technical constraints.

**Keywords**: REST API, HTTP methods, JSON, authentication, OAuth2, web scraping, BeautifulSoup, Flask, rate limiting, robots.txt

**Estimated Duration**: 8-10 hours (theory: 2h, laboratory: 4h, exercises: 3h)

**Difficulty Level**: ★★★★☆ (4/5)

### 1.2 Position in Curriculum

```mermaid
graph TB
    subgraph "TIER 1: Foundation"
        U01[01UNIT<br/>Epistemology]
        U02[02UNIT<br/>Abstraction]
        U03[03UNIT<br/>Complexity]
        U04[04UNIT<br/>Data Structures]
    end
    
    subgraph "TIER 2: Application"
        U05[05UNIT<br/>Scientific Computing]
        U06[06UNIT<br/>Visualisation]
        U07[07UNIT<br/>Reproducibility]
    end
    
    subgraph "TIER 3: Extension"
        U08[08UNIT<br/>Recursion & DP]
        U09[09UNIT<br/>Exceptions]
        U10[10UNIT<br/>Persistence]
        U11[11UNIT<br/>Text & NLP]
        U12[12UNIT<br/>Web APIs]
        U13[13UNIT<br/>Machine Learning]
        U14[14UNIT<br/>Parallel Computing]
    end
    
    U01 --> U02
    U02 --> U03
    U03 --> U04
    U03 --> U05
    U04 --> U05
    U05 --> U06
    U01 & U02 & U03 & U04 & U05 & U06 --> U07
    
    U03 --> U08
    U04 --> U09
    U09 --> U10
    U10 --> U11
    U09 & U10 --> U12
    U06 & U11 --> U13
    U03 & U08 --> U14
    U13 --> U14
    
    style U12 fill:#4a9eff,stroke:#fff,stroke-width:3px
```

This unit occupies a central position in the extension tier, bridging data persistence skills from 10UNIT (JSON serialisation, database fundamentals) with exception handling resilience from 09UNIT. The techniques developed here feed directly into 13UNIT's machine learning pipelines, where programmatic data acquisition from research APIs enables the construction of training datasets at scale.

| Dependency Type | Source | Concepts Required |
|-----------------|--------|-------------------|
| Hard prerequisite | 09UNIT | Exception handling, retry patterns |
| Hard prerequisite | 10UNIT | JSON serialisation, file I/O |
| Soft prerequisite | 11UNIT | Text processing (for HTML parsing) |
| Prepares for | 13UNIT | Data pipelines for ML |

### 1.3 Pedagogical Rationale

The instructional design follows experiential learning principles, immersing participants in authentic data acquisition scenarios drawn from bibliometric research, environmental monitoring and social science applications. The progression moves from foundational HTTP concepts through increasingly sophisticated patterns: simple GET requests, authenticated API calls, pagination handling, rate-limited bulk retrieval and ultimately the creation of custom APIs to share research outputs.

Unlike pure mathematical functions where behaviour depends solely on inputs, API interactions introduce external dependencies—network conditions, server availability, authentication state. As noted in software engineering literature: "Unlike the pure rectangle area function, this function's behaviour depends on external factors (the file system) and it modifies external state (the log file). These dependencies make the function less predictable and more challenging to test in isolation" (Peyton Jones, 2007, p. 34). This reality necessitates resilient error handling, timeout management and graceful degradation strategies.

**Instructional Strategies Employed:**

1. **Authentic Tasks**: Real API queries against CrossRef and OpenAlex for bibliometric data
2. **Scaffolded Complexity**: Progressive introduction of authentication and rate limiting
3. **Ethical Reasoning**: Explicit treatment of scraping ethics and terms of service
4. **Transfer Opportunities**: Custom Flask API creation for research data dissemination

---

## 2. Visual Overview

### 2.1 Conceptual Architecture

```mermaid
mindmap
  root((12UNIT: Web APIs))
    HTTP Fundamentals
      Methods
        GET
        POST
        PUT
        DELETE
      Status Codes
        2xx Success
        4xx Client Error
        5xx Server Error
      Headers
        Content-Type
        Authorization
        Accept
    API Consumption
      requests Library
        Session Management
        Timeout Handling
        SSL Verification
      Authentication
        API Keys
        Basic Auth
        Bearer Tokens
        OAuth2 Flow
      Pagination
        Offset-based
        Cursor-based
        Link Headers
    Web Scraping
      BeautifulSoup
        CSS Selectors
        Tag Navigation
      Ethics
        robots.txt
        Rate Limiting
        Terms of Service
    Flask APIs
      Routes
      JSON Responses
      Error Handlers
      Blueprints
```

### 2.2 HTTP Request-Response Cycle

The foundation of web-based data acquisition rests upon the HTTP protocol's request-response cycle. A client (research script) transmits a request containing method, URL, headers and optional body; the server processes this request and returns a response with status code, headers and body (typically JSON or HTML).

```
┌─────────────────┐                    ┌─────────────────┐
│   Client        │                    │   Server        │
│   (requests)    │                    │   (API)         │
└────────┬────────┘                    └────────┬────────┘
         │                                      │
         │  ───────── HTTP Request ──────────▶  │
         │  Method: GET                         │
         │  URL: /api/v1/works                  │
         │  Headers: Accept: application/json  │
         │  Authorization: Bearer xxx          │
         │                                      │
         │  ◀──────── HTTP Response ─────────  │
         │  Status: 200 OK                      │
         │  Headers: Content-Type: json         │
         │  Body: {"results": [...]}            │
         │                                      │
```

---

## 3. Learning Objectives

Upon successful completion of this unit, participants will demonstrate the following competencies:

| ID | Bloom Level | Verb | Objective Statement | Assessment |
|----|-------------|------|---------------------|------------|
| LO1 | Understand | Explain | HTTP methods (GET, POST, PUT, DELETE), status codes (2xx-5xx), headers and REST principles | Quiz Q1-Q3 |
| LO2 | Apply | Implement | API consumption using requests library: GET, POST, authentication and pagination | Lab 01, Ex easy_01-medium_01 |
| LO3 | Apply | Build | Web scrapers using BeautifulSoup with CSS selectors and ethical practices | Lab 02 §1-2, Ex medium_02 |
| LO4 | Apply | Handle | API errors, rate limiting and retry logic with exponential backoff | Lab 01 §5, Ex medium_03 |
| LO5 | Create | Design | Flask REST API to expose research data with proper error handling | Lab 02 §3-4, Ex hard_01-02 |
| LO6 | Evaluate | Evaluate | Data acquisition strategies for different research scenarios | Homework Part 3, Quiz Q9-10 |

### 3.1 Alignment Matrix

| Assessment Component | LO1 | LO2 | LO3 | LO4 | LO5 | LO6 |
|---------------------|-----|-----|-----|-----|-----|-----|
| Lab 01: API Consumption | ✓ | ✓ | | ✓ | | |
| Lab 02: Scraping & Flask | | | ✓ | | ✓ | |
| Easy Exercises | ✓ | ✓ | | | | |
| Medium Exercises | | ✓ | ✓ | ✓ | | |
| Hard Exercises | | | | ✓ | ✓ | ✓ |
| Quiz | ✓ | ✓ | ✓ | ✓ | | ✓ |
| Homework | | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 4. Theoretical Foundations

### 4.1 HTTP Protocol Fundamentals

The Hypertext Transfer Protocol (HTTP) constitutes the foundational communication protocol for the World Wide Web. Introduced by Tim Berners-Lee in 1991, HTTP operates as a stateless request-response protocol where clients transmit requests and servers return responses. HTTP/1.1 (RFC 2616, 1999) remains widely deployed, whilst HTTP/2 (2015) and HTTP/3 (2022) introduce performance optimisations.

**HTTP Methods and Their Semantics:**

| Method | Idempotent | Safe | Typical Use |
|--------|------------|------|-------------|
| GET | Yes | Yes | Retrieve resource |
| POST | No | No | Create resource |
| PUT | Yes | No | Replace resource |
| PATCH | No | No | Partial update |
| DELETE | Yes | No | Remove resource |
| HEAD | Yes | Yes | Retrieve headers only |
| OPTIONS | Yes | Yes | Discover allowed methods |

A method is *safe* if it does not modify server state; it is *idempotent* if multiple identical requests produce the same effect as a single request.

**HTTP Status Codes:**

Status codes communicate the outcome of request processing:

- **2xx Success**: Request processed successfully (200 OK, 201 Created, 204 No Content)
- **3xx Redirection**: Further action required (301 Moved Permanently, 304 Not Modified)
- **4xx Client Error**: Invalid request (400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 429 Too Many Requests)
- **5xx Server Error**: Server failure (500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable)

### 4.2 REST Architectural Style

Representational State Transfer (REST) defines an architectural style for networked hypermedia applications (Fielding, 2000). RESTful APIs expose resources (nouns) at URIs and manipulate them through HTTP methods (verbs).

**REST Principles:**

1. **Statelessness**: Each request contains all information necessary for processing; the server maintains no client session state
2. **Uniform Interface**: Standard HTTP methods operate on resources identified by URIs
3. **Resource-Based**: Everything is a resource with a unique identifier
4. **HATEOAS**: Hypermedia As The Engine Of Application State—responses include links to related resources

**Example REST Endpoints:**

```
GET    /api/v1/works              # List all works
GET    /api/v1/works/{id}         # Retrieve specific work
POST   /api/v1/works              # Create new work
PUT    /api/v1/works/{id}         # Replace work
PATCH  /api/v1/works/{id}         # Partial update
DELETE /api/v1/works/{id}         # Delete work
```

### 4.3 Authentication Mechanisms

Web APIs employ various authentication schemes to control access:

**API Keys**: Simple tokens included in headers or query parameters. Suitable for public APIs with usage tracking.

```python
headers = {"X-API-Key": "your_api_key_here"}
```

**HTTP Basic Authentication**: Username and password encoded in Base64 within the Authorization header.

```python
from requests.auth import HTTPBasicAuth
auth = HTTPBasicAuth("username", "password")
```

**Bearer Token Authentication**: Tokens (often JWTs) prefixed with "Bearer" in the Authorization header.

```python
headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIs..."}
```

**OAuth 2.0**: Industry-standard protocol for delegated authorisation. The client obtains an access token through an authorisation flow, then includes it in subsequent requests.

### 4.4 Rate Limiting and Backoff Strategies

APIs impose rate limits to protect infrastructure and ensure fair usage. Common implementations include:

- **Fixed Window**: N requests per time window (e.g., 100/minute)
- **Sliding Window**: N requests in any rolling time period
- **Token Bucket**: Tokens accumulate over time; each request consumes one token

When rate limited (HTTP 429), implement exponential backoff:

$$T_{wait} = T_{base} \times 2^{attempt} + jitter$$

where $T_{base}$ is the initial wait time (e.g., 1 second), $attempt$ is the retry count, and $jitter$ adds randomisation to prevent thundering herd effects.

---

## 5. Algorithmic Content

### 5.1 Pagination Algorithms

APIs returning large result sets implement pagination to manage response size. Three common patterns exist:

**Offset-Based Pagination:**
```
GET /api/works?offset=0&limit=20    # Page 1
GET /api/works?offset=20&limit=20   # Page 2
```

**Cursor-Based Pagination:**
```
GET /api/works?cursor=abc123&limit=20
```

**Link Header Pagination (RFC 5988):**
```
Link: <https://api.example.com/works?page=2>; rel="next",
      <https://api.example.com/works?page=5>; rel="last"
```

### 5.2 Algorithm Catalogue

| Algorithm | Problem | Time Complexity | Space Complexity |
|-----------|---------|-----------------|------------------|
| Linear pagination | Retrieve all pages | O(n/page_size) requests | O(n) |
| Exponential backoff | Rate limit handling | O(log max_retries) | O(1) |
| Recursive scraping | Site traversal | O(V + E) | O(V) |
| Token bucket | Rate limiting | O(1) per request | O(1) |

---

## 6. Practical Applications

### 6.1 Research Domain Applications

| Domain | Application | API/Source | Reference |
|--------|-------------|------------|-----------|
| Bibliometrics | Citation analysis | CrossRef, OpenAlex, Semantic Scholar | Priem et al. (2022) |
| Social Science | Survey distribution | Qualtrics API | — |
| Environmental | Weather data | OpenWeatherMap, NOAA APIs | — |
| Economics | Financial data | Alpha Vantage, FRED API | — |
| Biology | Sequence databases | NCBI Entrez, UniProt | Sayers et al. (2021) |

### 6.2 Case Study: Bibliometric Data Collection

**Context**: A research group studying publication trends requires metadata for 50,000 academic papers.

**Challenge**: Manual collection infeasible; CrossRef API provides programmatic access but enforces rate limits.

**Solution**: Implement paginated retrieval with polite rate limiting:

```python
def collect_crossref_works(query: str, max_results: int = 1000) -> list[dict]:
    """
    Collect works from CrossRef API with polite rate limiting.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to retrieve
        
    Returns:
        List of work metadata dictionaries
    """
    works = []
    cursor = "*"
    
    while len(works) < max_results:
        response = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "cursor": cursor, "rows": 100},
            headers={"User-Agent": "ResearchBot/1.0 (mailto:research@example.edu)"}
        )
        response.raise_for_status()
        data = response.json()
        
        works.extend(data["message"]["items"])
        cursor = data["message"]["next-cursor"]
        
        if not data["message"]["items"]:
            break
            
        time.sleep(1)  # Polite delay
        
    return works[:max_results]
```

**Results**: Complete dataset collected over 8 hours with zero rate limit violations.

**Lessons**: Polite crawling with appropriate delays and user-agent identification enables sustained API access.

---

## 7. Laboratory Guide

### 7.1 Laboratory Overview

| Lab | Title | Duration | Objective | Difficulty |
|-----|-------|----------|-----------|------------|
| 01 | API Consumption | 50 min | Master requests library patterns | ★★★☆☆ |
| 02 | Web Scraping and Flask | 40 min | Build scrapers and REST APIs | ★★★★☆ |

### 7.2 Lab 01: API Consumption

**Learning Trajectory:**

```mermaid
flowchart LR
    A[HTTP Fundamentals] --> B[REST Patterns]
    B --> C[Authentication]
    C --> D[Research APIs]
    D --> E[Resilient Consumption]
```

**Key Files:**
- `lab/lab_12_01_api_consumption.py` (500+ lines)
- `lab/solutions/lab_12_01_solution.py`

**Sections:**
1. HTTP Fundamentals (~80 lines): Request/response inspection, headers
2. REST Patterns (~120 lines): Pagination, CRUD operations, error handling
3. Authentication (~100 lines): API keys, Basic, Bearer, OAuth2 client credentials
4. Research APIs (~100 lines): CrossRef and OpenAlex examples
5. Resilient Consumption (~100 lines): Retry logic, rate limiting, caching

### 7.3 Lab 02: Web Scraping and Flask

**Learning Trajectory:**

```mermaid
flowchart LR
    A[BeautifulSoup Basics] --> B[Scraping Patterns]
    B --> C[Flask Routes]
    C --> D[Research API]
```

**Key Files:**
- `lab/lab_12_02_web_scraping_flask.py` (350+ lines)
- `lab/solutions/lab_12_02_solution.py`

**Sections:**
1. Web Scraping (~100 lines): BeautifulSoup, CSS selectors, ethical practices
2. Scraping Patterns (~80 lines): Sessions, pagination, robots.txt compliance
3. Flask API Basics (~120 lines): Routes, JSON responses, error handlers
4. Research API (~50 lines): Expose dataset via REST endpoints

---

## 8. Exercise Framework

### 8.1 Difficulty Progression

```
EASY (10-15 min each)     ████░░░░░░ 3 exercises
MEDIUM (20-25 min each)   ██████░░░░ 3 exercises  
HARD (30-40 min each)     ██████████ 3 exercises
```

### 8.2 Exercise-Objective Mapping

| Exercise | Objectives | Key Concepts | Est. Time |
|----------|------------|--------------|-----------|
| easy_01_requests_basics.py | LO1, LO2 | GET requests, JSON parsing | 15 min |
| easy_02_simple_post.py | LO2 | POST requests, request body | 15 min |
| easy_03_status_codes.py | LO1 | Status code handling | 10 min |
| medium_01_pagination.py | LO2 | Paginated retrieval | 25 min |
| medium_02_authentication.py | LO2 | Auth mechanisms | 25 min |
| medium_03_error_handling.py | LO4 | Retry with backoff | 20 min |
| hard_01_flask_api.py | LO5 | Complete REST API | 40 min |
| hard_02_ethical_scraper.py | LO3, LO6 | robots.txt, rate limiting | 35 min |
| hard_03_data_pipeline.py | LO2-LO6 | End-to-end acquisition | 40 min |

### 8.3 Hints Policy

Progressive hints unlock after failed attempts:

1. **Level 1**: Conceptual direction
2. **Level 2**: Relevant function/method names
3. **Level 3**: Partial code structure

---

## 9. Technical Reference

### 9.1 Essential Libraries

| Library | Import | Purpose |
|---------|--------|---------|
| requests | `import requests` | HTTP client |
| BeautifulSoup | `from bs4 import BeautifulSoup` | HTML parsing |
| Flask | `from flask import Flask, jsonify, request` | Web framework |
| urllib.robotparser | `from urllib.robotparser import RobotFileParser` | robots.txt parsing |

### 9.2 Common Code Patterns

**Making a GET Request:**
```python
response = requests.get(url, headers=headers, timeout=30)
response.raise_for_status()
data = response.json()
```

**Session Management:**
```python
with requests.Session() as session:
    session.headers.update({"Authorization": "Bearer token"})
    response = session.get(url)
```

**BeautifulSoup Parsing:**
```python
soup = BeautifulSoup(html, "html.parser")
titles = [h2.text for h2 in soup.select("h2.article-title")]
```

**Flask JSON Endpoint:**
```python
@app.route("/api/v1/data", methods=["GET"])
def get_data():
    return jsonify({"results": data, "count": len(data)})
```

---

## 10. Visual Assets

### 10.1 Diagram Inventory

| Diagram | Type | Purpose |
|---------|------|---------|
| http_request_response.puml | PlantUML | HTTP cycle visualisation |
| rest_api_design.puml | PlantUML | REST endpoint structure |
| oauth2_flow.puml | PlantUML | OAuth2 authorisation flow |
| scraping_workflow.puml | PlantUML | Web scraping decision tree |
| flask_app_structure.puml | PlantUML | Flask application architecture |
| http_status_codes.svg | SVG | Status code categories |
| api_pagination_patterns.svg | SVG | Pagination approaches |
| rate_limiting.svg | SVG | Rate limiting visualisation |

### 10.2 Interactive Demo

The `12UNIT_api_explorer.html` provides an interactive environment for:
- Testing API endpoints with customisable parameters
- Visualising request/response headers
- Experimenting with authentication methods
- Observing rate limiting behaviour

---

## 11. Assessment Overview

### 11.1 Formative Assessment

- Lab completion with self-verification
- Practice exercises with automated tests
- Quiz with immediate feedback

### 11.2 Summative Assessment

- Homework assignment (graded)
- Integration into capstone project

---

## 12. Connections

### 12.1 Prerequisites Review

**From 09UNIT (Exception Handling):**
- try/except patterns for network errors
- Custom exception hierarchies
- Retry decorators with backoff

**From 10UNIT (Data Persistence):**
- JSON serialisation with custom encoders
- File I/O for caching responses
- Database storage for collected data

### 12.2 Forward Connections

**To 13UNIT (Machine Learning):**
- API-acquired datasets as ML training data
- Feature extraction from web sources
- Automated data pipeline construction

---

## 13. Quick Start

### 13.1 Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install requests beautifulsoup4 flask pytest

# Verify installation
python -c "import requests; print(requests.__version__)"
```

### 13.2 Running Labs

```bash
# Run lab in demo mode
python lab/lab_12_01_api_consumption.py --demo

# Run specific section
python lab/lab_12_01_api_consumption.py --section 3

# Run tests
pytest tests/ -v
```

---

## 14. Directory Structure

```
12UNIT - Web APIs and Data Acquisition (web)/
│
├── 📄 README.md                              # This document
│
├── 📚 theory/
│   ├── 12UNIT_slides.html                    # reveal.js presentation (40+ slides)
│   ├── lecture_notes.md                      # Detailed notes (2,500+ words)
│   └── learning_objectives.md                # LOs with alignment matrix
│
├── 🔬 lab/
│   ├── __init__.py
│   ├── lab_12_01_api_consumption.py          # Primary lab (500+ lines)
│   ├── lab_12_02_web_scraping_flask.py       # Secondary lab (350+ lines)
│   └── solutions/
│       ├── lab_12_01_solution.py
│       └── lab_12_02_solution.py
│
├── 📝 exercises/
│   ├── homework.md                           # Homework assignment
│   ├── practice/
│   │   ├── easy_01_requests_basics.py
│   │   ├── easy_02_simple_post.py
│   │   ├── easy_03_status_codes.py
│   │   ├── medium_01_pagination.py
│   │   ├── medium_02_authentication.py
│   │   ├── medium_03_error_handling.py
│   │   ├── hard_01_flask_api.py
│   │   ├── hard_02_ethical_scraper.py
│   │   └── hard_03_data_pipeline.py
│   └── solutions/
│       ├── easy_01_solution.py
│       ├── easy_02_solution.py
│       ├── easy_03_solution.py
│       ├── medium_01_solution.py
│       ├── medium_02_solution.py
│       ├── medium_03_solution.py
│       ├── hard_01_solution.py
│       ├── hard_02_solution.py
│       └── hard_03_solution.py
│
├── 📊 assessments/
│   ├── quiz.md                               # 10 questions
│   ├── rubric.md                             # Grading criteria
│   └── self_check.md                         # Self-assessment checklist
│
├── 📖 resources/
│   ├── cheatsheet.md                         # Quick reference (2 A4 pages)
│   ├── further_reading.md                    # Annotated bibliography (15+ entries)
│   ├── glossary.md                           # Definitions (20+ terms)
│   └── datasets/
│       └── sample_works.json                 # Sample API response data
│
├── 🎨 assets/
│   ├── diagrams/
│   │   ├── http_request_response.puml
│   │   ├── rest_api_design.puml
│   │   ├── oauth2_flow.puml
│   │   ├── scraping_workflow.puml
│   │   ├── flask_app_structure.puml
│   │   ├── http_status_codes.svg
│   │   ├── api_pagination_patterns.svg
│   │   └── rate_limiting.svg
│   ├── animations/
│   │   └── 12UNIT_api_explorer.html
│   └── images/
│       └── 12UNIT_badge.svg
│
├── 🧪 tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_lab_12_01.py
│   └── test_lab_12_02.py
│
└── Makefile
```

---

## 15. Progress Tracking

### 15.1 Completion Checklist

| Component | Status | Estimated Duration |
|-----------|--------|-------------------|
| Lecture notes studied | ⬜ | ~30 min |
| Presentation slides viewed | ⬜ | ~60 min |
| Lab 01: API Consumption completed | ⬜ | ~50 min |
| Lab 02: Scraping & Flask completed | ⬜ | ~40 min |
| Easy exercises (3) completed | ⬜ | ~40 min |
| Medium exercises (3) completed | ⬜ | ~70 min |
| Hard exercises (3) completed | ⬜ | ~115 min |
| Quiz passed (≥70%) | ⬜ | ~15 min |
| Self-assessment completed | ⬜ | ~15 min |
| Homework submitted | ⬜ | ~90 min |

---

## 16. Troubleshooting

### 16.1 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ConnectionError` | Network unavailable | Check internet connection |
| `Timeout` | Server unresponsive | Increase timeout, retry later |
| `403 Forbidden` | Authentication failure | Verify credentials/API key |
| `429 Too Many Requests` | Rate limited | Implement exponential backoff |
| `SSLError` | Certificate issues | Update certifi: `pip install --upgrade certifi` |

### 16.2 FAQ

**Q: Why does my API request work in the browser but fail in Python?**
A: Browsers automatically handle cookies, redirects and authentication. Ensure your Python code explicitly handles these. Check the User-Agent header—some APIs block generic user agents.

**Q: How do I respect rate limits without knowing the exact limit?**
A: Start conservatively (1 request/second), monitor for 429 responses, and implement exponential backoff. Include a polite User-Agent with contact information.

---

## 17. Further Reading

### 17.1 Primary Sources

- Fielding, R. T. (2000). *Architectural Styles and the Design of Network-based Software Architectures*. Doctoral dissertation, UC Irvine.
- RFC 7231: Hypertext Transfer Protocol (HTTP/1.1): Semantics and Content

### 17.2 Textbooks

- Masse, M. (2011). *REST API Design Rulebook*. O'Reilly Media.
- Richardson, L., & Ruby, S. (2013). *RESTful Web APIs*. O'Reilly Media.

### 17.3 Online Resources

- requests documentation: https://docs.python-requests.org/
- BeautifulSoup documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- Flask documentation: https://flask.palletsprojects.com/

---

## 18. Glossary

| Term | Definition |
|------|------------|
| **API** | Application Programming Interface—a defined contract for software component interaction |
| **Bearer Token** | Authentication token included in Authorization header |
| **CORS** | Cross-Origin Resource Sharing—security mechanism for browser requests |
| **Endpoint** | Specific URL where an API receives requests |
| **HTTP** | Hypertext Transfer Protocol—the foundation of web communication |
| **Idempotent** | Operation producing the same result regardless of repetition count |
| **JSON** | JavaScript Object Notation—lightweight data interchange format |
| **OAuth** | Open Authorisation—standard protocol for delegated access |
| **Pagination** | Dividing large result sets into manageable pages |
| **Rate Limiting** | Restricting request frequency to protect server resources |
| **REST** | Representational State Transfer—architectural style for web services |
| **robots.txt** | File specifying which parts of a site may be crawled |
| **Scraping** | Extracting data from web pages programmatically |
| **Session** | Persistent connection maintaining state across requests |
| **Status Code** | Three-digit number indicating request processing outcome |
| **Timeout** | Maximum duration to wait for server response |
| **URI** | Uniform Resource Identifier—string identifying a resource |
| **URL** | Uniform Resource Locator—URI specifying resource location |
| **User-Agent** | Header identifying the client making a request |
| **Webhook** | HTTP callback triggered by events |

---

## 📜 Licence and Terms of Use

<div align="center">

<table>
<tr>
<td>

<div align="center">
<h3>🔒 RESTRICTIVE LICENCE</h3>
<p><strong>Version 4.1.0 — January 2025</strong></p>
</div>

---

**© 2025 Antonio Clim. All rights reserved.**

<table>
<tr>
<th>✅ PERMITTED</th>
<th>❌ PROHIBITED</th>
</tr>
<tr>
<td>

- Personal use for self-study
- Viewing and running code for personal educational purposes
- Local modifications for personal experimentation

</td>
<td>

- Publishing materials (online or offline)
- Use in formal teaching activities
- Teaching or presenting to third parties
- Redistribution in any form
- Creating derivative works for public use
- Commercial use of any kind

</td>
</tr>
</table>

---

<p><em>For requests regarding educational use or publication,<br>
please contact the author to obtain written consent.</em></p>

</td>
</tr>
</table>

</div>

### Terms and Conditions

1. **Intellectual Property**: All materials, including code, documentation,
   presentations and exercises, are the intellectual property of Antonio Clim.

2. **No Warranty**: Materials are provided "as is" without warranty of any kind,
   express or implied.

3. **Limitation of Liability**: The author shall not be liable for any damages
   arising from the use of these materials.

4. **Governing Law**: These terms are governed by the laws of Romania.

5. **Contact**: For permissions and enquiries, contact the author through
   official academic channels.

### Technology Stack

<div align="center">

| Technology | Version | Purpose |
|:----------:|:-------:|:--------|
| Python | 3.12+ | Primary programming language |
| requests | ≥2.28 | HTTP client library |
| BeautifulSoup4 | ≥4.12 | HTML parsing |
| Flask | ≥3.0 | Web framework |
| pytest | ≥7.0 | Testing framework |
| pytest-cov | ≥4.0 | Coverage reporting |
| ruff | ≥0.1 | Linting and formatting |
| mypy | ≥1.0 | Type checking |
| responses | ≥0.23 | HTTP mocking for tests |
| reveal.js | 5.0 | Presentation framework |
| D3.js | 7.8+ | Interactive visualisations |

</div>
