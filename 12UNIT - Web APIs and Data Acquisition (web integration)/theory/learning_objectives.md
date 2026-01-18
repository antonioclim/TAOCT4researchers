# 12UNIT: Web APIs and Data Acquisition — Learning Objectives

## Unit Overview

This unit equips doctoral researchers with the theoretical understanding and practical skills necessary to programmatically acquire data from web sources. Participants will master HTTP protocol fundamentals, REST architectural principles, API consumption patterns and Flask-based API development. The unit emphasises resilient error handling, authentication mechanisms and ethical considerations essential for reproducible research workflows.

---

## Learning Objectives

### LO1: Understand HTTP Protocol and REST Principles
**Cognitive Level**: Understand

*Upon completion, learners will be able to:*
- Explain the HTTP request-response cycle and its stateless nature
- Differentiate between HTTP methods (GET, POST, PUT, DELETE) and their semantic meanings
- Interpret HTTP status codes (2xx, 3xx, 4xx, 5xx) and their implications for client behaviour
- Describe REST architectural constraints and their benefits for API design
- Analyse HTTP headers for content negotiation, caching and rate limiting information

**Assessment Criteria**:
- Correctly identify appropriate HTTP methods for given operations
- Accurately interpret status codes in API responses
- Explain REST principles without reference to documentation

---

### LO2: Apply API Consumption Techniques
**Cognitive Level**: Apply

*Upon completion, learners will be able to:*
- Construct HTTP requests using the `requests` library with appropriate parameters and headers
- Implement GET requests with query parameters for filtering and pagination
- Execute POST requests with JSON payloads for resource creation
- Configure request sessions for connection pooling and persistent authentication
- Parse JSON responses and extract relevant data fields

**Assessment Criteria**:
- Successfully retrieve data from public APIs (CrossRef, OpenAlex)
- Handle paginated responses collecting all available results
- Demonstrate proper header configuration for authentication and content type

---

### LO3: Implement Authentication Mechanisms
**Cognitive Level**: Apply

*Upon completion, learners will be able to:*
- Configure API key authentication via headers and query parameters
- Implement HTTP Basic Authentication with the `requests` library
- Execute OAuth 2.0 Client Credentials flow for machine-to-machine authentication
- Handle Bearer token authentication for protected resources
- Securely manage credentials using environment variables

**Assessment Criteria**:
- Successfully authenticate against APIs requiring various mechanisms
- Demonstrate secure credential storage without hardcoding
- Implement token refresh for expiring credentials

---

### LO4: Handle API Errors and Implement Retry Logic
**Cognitive Level**: Apply

*Upon completion, learners will be able to:*
- Implement comprehensive exception handling for network failures
- Construct exponential backoff retry strategies for transient errors
- Respect rate limits by parsing response headers and implementing delays
- Distinguish retriable errors (5xx, 429) from non-retriable errors (4xx)
- Log API interactions for debugging and audit purposes

**Assessment Criteria**:
- Build fault-tolerant clients that recover from temporary failures
- Demonstrate rate limit compliance without manual intervention
- Produce diagnostic logs enabling failure analysis

---

### LO5: Perform Ethical Web Scraping
**Cognitive Level**: Apply

*Upon completion, learners will be able to:*
- Parse HTML documents using BeautifulSoup with CSS selectors
- Navigate document trees to extract structured data from web pages
- Check and comply with robots.txt restrictions
- Implement polite scraping with appropriate delays and user-agent identification
- Handle pagination and session management in scraping workflows

**Assessment Criteria**:
- Extract structured data from HTML pages accurately
- Demonstrate robots.txt compliance verification
- Implement rate-limited scraping respecting server resources

---

### LO6: Design and Implement Flask REST APIs
**Cognitive Level**: Create

*Upon completion, learners will be able to:*
- Structure Flask applications with route decorators and blueprints
- Implement CRUD endpoints following REST conventions
- Validate request data and return appropriate error responses
- Serialise Python objects to JSON for API responses
- Document API endpoints with clear specifications

**Assessment Criteria**:
- Build functional APIs exposing research datasets
- Return appropriate status codes for success and error conditions
- Implement input validation with informative error messages

---

### LO7: Evaluate Data Acquisition Strategies
**Cognitive Level**: Evaluate

*Upon completion, learners will be able to:*
- Compare API-based and scraping-based data acquisition approaches
- Assess API quality based on documentation, reliability and rate limits
- Evaluate authentication mechanisms for security and usability trade-offs
- Critique API designs against REST principles and usability heuristics
- Select appropriate data sources for specific research requirements

**Assessment Criteria**:
- Justify data acquisition strategy choices for given research scenarios
- Identify limitations and risks of different approaches
- Propose improvements to existing API designs

---

## Alignment Matrix

| Learning Objective | Laboratory | Exercises | Assessment |
|-------------------|------------|-----------|------------|
| LO1: HTTP/REST    | Lab 12.01 §1-2 | Easy 1-2 | Quiz Q1-5 |
| LO2: API Consumption | Lab 12.01 §2,4 | Easy 3, Medium 1 | Quiz Q6-8 |
| LO3: Authentication | Lab 12.01 §3 | Medium 2 | Quiz Q9-11 |
| LO4: Error Handling | Lab 12.01 §5 | Medium 3 | Quiz Q12-14 |
| LO5: Web Scraping | Lab 12.02 §1-2 | Hard 1 | Quiz Q15-17 |
| LO6: Flask APIs | Lab 12.02 §3-4 | Hard 2 | Quiz Q18-20 |
| LO7: Evaluation | Homework | Hard 3 | Rubric |

---

## Prerequisites

Successful engagement with this unit assumes mastery of:

- **09UNIT**: Exception handling patterns and defensive programming
- **10UNIT**: JSON serialisation and data persistence
- **Python fundamentals**: Functions, classes, dictionaries, file I/O
- **Command line**: Environment variables, package installation

---

## Estimated Time Investment

| Component | Duration |
|-----------|----------|
| Lecture notes study | 2 hours |
| Laboratory 12.01: API Consumption | 50 minutes |
| Laboratory 12.02: Web Scraping and Flask | 40 minutes |
| Practice exercises (9 total) | 3 hours |
| Homework assignment | 2 hours |
| Self-assessment and review | 30 minutes |
| **Total** | **8-10 hours** |

---

*Licence: Restrictive — see repository root for terms.*
