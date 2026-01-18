# 12UNIT: Web APIs and Data Acquisition — Self-Assessment Checklist

## Instructions

Use this checklist to verify your understanding before proceeding to assessments.
For each item, honestly evaluate whether you can perform the task without assistance.

**Rating Scale:**
- ✓ Confident — Can do independently
- ~ Developing — Need occasional reference
- ✗ Not Yet — Requires significant support

---

## Learning Objective 1: HTTP Protocol and REST Principles

### HTTP Fundamentals

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Explain the HTTP request-response cycle | | |
| Identify appropriate HTTP methods (GET, POST, PUT, DELETE) | | |
| Interpret status codes (200, 201, 400, 401, 404, 429, 500) | | |
| Read and write HTTP headers | | |
| Distinguish safe and idempotent methods | | |

### REST Principles

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Describe the six REST constraints | | |
| Design resource-oriented URIs | | |
| Apply HATEOAS principles to responses | | |
| Explain statelessness benefits | | |
| Differentiate REST from RPC-style APIs | | |

**Self-Test:** Without reference, explain why PUT is idempotent but POST is not.

---

## Learning Objective 2: API Consumption with Requests

### Basic Requests

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Make GET requests with query parameters | | |
| Send POST requests with JSON bodies | | |
| Parse JSON responses | | |
| Configure request headers | | |
| Use Session objects for efficiency | | |

### Advanced Patterns

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Implement offset-based pagination | | |
| Implement cursor-based pagination | | |
| Handle streaming responses | | |
| Configure SSL verification | | |
| Use proxy servers | | |

**Self-Test:** Write code to fetch all pages from a paginated API endpoint.

---

## Learning Objective 3: Authentication Mechanisms

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Configure API key authentication (header) | | |
| Configure API key authentication (query) | | |
| Implement HTTP Basic authentication | | |
| Execute OAuth 2.0 Client Credentials flow | | |
| Handle Bearer token authentication | | |
| Manage token refresh | | |
| Store credentials securely | | |

**Self-Test:** Explain the difference between OAuth 2.0 authorization code and client credentials flows.

---

## Learning Objective 4: Error Handling and Retry Logic

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Implement try/except for network errors | | |
| Distinguish retriable from non-retriable errors | | |
| Implement exponential backoff | | |
| Parse Retry-After headers | | |
| Respect rate limit headers | | |
| Log API interactions for debugging | | |
| Implement circuit breaker pattern | | |

**Self-Test:** Implement a function that retries on 5xx errors with exponential backoff.

---

## Learning Objective 5: Web Scraping

### Ethical Scraping

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Check and comply with robots.txt | | |
| Set appropriate User-Agent headers | | |
| Implement rate limiting | | |
| Cache responses to reduce load | | |
| Identify when scraping is inappropriate | | |

### BeautifulSoup

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Parse HTML with BeautifulSoup | | |
| Use CSS selectors to find elements | | |
| Navigate the document tree | | |
| Extract text and attributes | | |
| Handle malformed HTML | | |

**Self-Test:** Write a scraper that extracts all article titles and links from a news page.

---

## Learning Objective 6: Flask API Development

### Basic Flask

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Create a Flask application | | |
| Define routes with decorators | | |
| Handle different HTTP methods | | |
| Access request data (JSON, form, args) | | |
| Return JSON responses | | |

### API Design

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Implement CRUD endpoints | | |
| Validate request data | | |
| Return appropriate status codes | | |
| Handle errors with error handlers | | |
| Document endpoints | | |

**Self-Test:** Build a complete CRUD API for a simple resource.

---

## Learning Objective 7: Evaluation

| Skill | Rating | Evidence/Notes |
|-------|--------|----------------|
| Compare API vs scraping approaches | | |
| Evaluate API documentation quality | | |
| Assess authentication security | | |
| Identify rate limiting strategies | | |
| Critique API designs | | |

**Self-Test:** Given a research scenario, justify your choice of data acquisition strategy.

---

## Overall Readiness

### Before Laboratory

- [ ] Installed required packages (requests, beautifulsoup4, flask)
- [ ] Reviewed HTTP method semantics
- [ ] Understood JSON data structures
- [ ] Familiar with exception handling patterns

### Before Homework

- [ ] Completed both laboratory exercises
- [ ] Can consume a real-world API independently
- [ ] Can build basic Flask endpoints
- [ ] Understand pagination patterns

### Before Quiz

- [ ] All learning objectives at "Confident" level
- [ ] Can explain concepts without code reference
- [ ] Reviewed lecture notes
- [ ] Completed practice exercises

---

## Reflection Questions

1. What is the most challenging aspect of API consumption for you?

2. How does understanding HTTP improve your API debugging?

3. What ethical considerations are most important in web scraping?

4. How would you design an API differently after this unit?

---

*Licence: Restrictive — see repository root for terms.*
