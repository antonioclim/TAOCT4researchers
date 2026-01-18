# 12UNIT: Web APIs and Data Acquisition — Assessment Rubric

## Overview

This rubric evaluates student work on the Unit 12 laboratory exercises and
homework assignment. Assessment focuses on technical correctness, code quality,
error handling and adherence to REST/HTTP conventions.

---

## Laboratory Assessment (40 marks total)

### Lab 12.01: API Consumption (20 marks)

#### Section 1: HTTP Fundamentals (4 marks)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|-----------|---------------|----------|------------------|----------------|
| HTTP method usage | Correct methods for all operations with proper semantics | Minor issues with method selection | Some confusion between methods | Incorrect method usage |
| Status code handling | Comprehensive handling of all status categories | Handles common codes correctly | Basic success/error distinction | Minimal status handling |

#### Section 2: REST Patterns (4 marks)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|-----------|---------------|----------|------------------|----------------|
| Pagination implementation | Complete cursor and offset support with edge cases | Working pagination with minor issues | Basic pagination works | Pagination incomplete |
| Query parameter handling | Proper encoding and multi-value support | Correct basic parameter usage | Parameters work in simple cases | Parameter handling errors |

#### Section 3: Authentication (4 marks)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|-----------|---------------|----------|------------------|----------------|
| Credential security | Environment variables with fallbacks | Environment variables used | Credentials not hardcoded | Credentials exposed in code |
| Token management | Caching with expiry handling | Basic token caching | Token retrieval works | No token management |

#### Section 4: Research APIs (4 marks)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|-----------|---------------|----------|------------------|----------------|
| API integration | Multiple APIs with proper headers | One API fully integrated | Basic API calls work | API calls fail |
| Data extraction | Comprehensive field extraction with validation | Correct extraction of key fields | Some fields extracted | Minimal extraction |

#### Section 5: Resilience (4 marks)

| Criterion | Excellent (4) | Good (3) | Satisfactory (2) | Needs Work (1) |
|-----------|---------------|----------|------------------|----------------|
| Error handling | Exponential backoff with jitter | Retry logic with backoff | Basic retry attempts | No retry logic |
| Rate limiting | Proactive rate limit respect | Reactive 429 handling | Some rate awareness | No rate consideration |

---

### Lab 12.02: Web Scraping and Flask (20 marks)

#### Sections 1-2: Web Scraping (10 marks)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (1-2) |
|-----------|---------------|----------|------------------|------------------|
| robots.txt compliance | Full parser with path matching | Basic disallow checking | Checks robots.txt exists | No robots.txt check |
| HTML parsing | Complex selectors with error handling | CSS selectors work correctly | Basic element extraction | Parsing fails |
| Ethical practices | Rate limiting, identification, caching | Rate limiting implemented | User-agent set | No ethical measures |

#### Sections 3-4: Flask API (10 marks)

| Criterion | Excellent (5) | Good (4) | Satisfactory (3) | Needs Work (1-2) |
|-----------|---------------|----------|------------------|------------------|
| Endpoint implementation | Full CRUD with validation | CRUD works with minor issues | Basic endpoints work | Endpoints fail |
| HTTP semantics | Correct status codes and headers | Most status codes correct | Success/error distinction | Wrong status codes |
| Error handling | Comprehensive error handlers | Common errors handled | Basic error responses | No error handling |

---

## Homework Assessment (30 marks)

### Task: Complete Data Pipeline

Students must implement a complete data acquisition and exposure pipeline
that retrieves data from a public API, processes it and exposes it through
a custom Flask API.

#### Data Acquisition (10 marks)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (1-5) |
|-----------|----------------|----------|------------------|------------------|
| API consumption | Efficient pagination, caching, error handling | Working pagination and error handling | Basic data retrieval | Retrieval fails |
| Data processing | Comprehensive cleaning and transformation | Appropriate processing | Some processing | Raw data only |

#### API Design (10 marks)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (1-5) |
|-----------|----------------|----------|------------------|------------------|
| REST compliance | Full REST semantics with HATEOAS | Proper resource-oriented design | Basic REST patterns | Not RESTful |
| Documentation | OpenAPI spec or comprehensive docstrings | Clear endpoint documentation | Basic documentation | Undocumented |

#### Code Quality (10 marks)

| Criterion | Excellent (10) | Good (8) | Satisfactory (6) | Needs Work (1-5) |
|-----------|----------------|----------|------------------|------------------|
| Type hints | Complete annotations with TypedDict/dataclasses | Most functions annotated | Some type hints | No type hints |
| Testing | Unit tests with mocking, >80% coverage | Tests for main paths | Some tests present | No tests |
| Style | PEP 8 compliant, clear naming | Minor style issues | Readable code | Poor style |

---

## Grade Boundaries

| Grade | Marks | Percentage |
|-------|-------|------------|
| Excellent | 63-70 | 90-100% |
| Good | 56-62 | 80-89% |
| Satisfactory | 49-55 | 70-79% |
| Pass | 42-48 | 60-69% |
| Fail | <42 | <60% |

---

## Feedback Template

```
Student: ________________
Date: __________________

Lab 12.01: __/20
  - HTTP Fundamentals: __/4
  - REST Patterns: __/4
  - Authentication: __/4
  - Research APIs: __/4
  - Resilience: __/4

Lab 12.02: __/20
  - Web Scraping: __/10
  - Flask API: __/10

Homework: __/30
  - Data Acquisition: __/10
  - API Design: __/10
  - Code Quality: __/10

TOTAL: __/70

Strengths:
1.
2.

Areas for Improvement:
1.
2.

Additional Comments:
```

---

*Licence: Restrictive — see repository root for terms.*
