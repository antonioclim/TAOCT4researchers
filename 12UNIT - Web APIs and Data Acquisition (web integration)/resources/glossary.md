# 12UNIT: Web APIs and Data Acquisition — Glossary

## Technical Terms and Definitions

---

### A

**API (Application Programming Interface)**
A set of protocols, routines and tools that specify how software components should interact. In web contexts, APIs typically expose functionality over HTTP, allowing programs to request data or trigger actions on remote servers.

**Authentication**
The process of verifying the identity of a client making a request. Common mechanisms include API keys, username/password (Basic Auth), tokens (Bearer/JWT) and OAuth flows. Authentication answers "Who are you?"

**Authorisation**
The process of determining whether an authenticated client has permission to perform a requested action. Authorisation answers "What may you do?" Often implemented through scopes, roles or permissions.

---

### B

**Backoff (Exponential)**
A retry strategy where wait time between attempts increases exponentially (e.g., 1s, 2s, 4s, 8s). This prevents overwhelming recovering servers and reduces contention during outages.

**Bearer Token**
An authentication token included in the `Authorization` header as `Bearer <token>`. The server validates the token to authenticate requests without requiring credentials on each call.

**BeautifulSoup**
A Python library for parsing HTML and XML documents. It creates parse trees that can be navigated, searched and modified, making it useful for web scraping tasks.

---

### C

**Cache**
Temporary storage of responses to reduce redundant requests. Caching improves performance and reduces server load. HTTP caching is controlled via headers like `Cache-Control`, `ETag` and `Last-Modified`.

**Client**
In HTTP, the program that initiates requests. Web browsers, mobile apps and Python scripts using `requests` all act as clients when communicating with servers.

**Content-Type**
An HTTP header specifying the media type of the request or response body. For JSON APIs, this is typically `application/json`.

**CORS (Cross-Origin Resource Sharing)**
A mechanism allowing servers to specify which origins may access their resources. Important for browser-based API clients where same-origin policy would otherwise block requests.

**CRUD**
Acronym for Create, Read, Update, Delete—the four basic operations on persistent data. REST APIs typically map these to POST, GET, PUT/PATCH and DELETE methods.

**Cursor-Based Pagination**
A pagination strategy using opaque tokens to mark positions in result sets. More reliable than offset pagination for frequently changing data, as it maintains consistent iteration.

---

### D

**DNS (Domain Name System)**
The hierarchical naming system that translates human-readable domain names (api.example.com) to IP addresses. DNS resolution is the first step in HTTP request processing.

---

### E

**Endpoint**
A specific URL path that accepts requests and returns responses. For example, `/api/users` might be an endpoint for user-related operations.

**ETags**
Entity tags—opaque identifiers assigned to specific versions of resources. Used for cache validation and conditional requests (If-None-Match header).

---

### F

**Flask**
A lightweight Python web framework for building web applications and APIs. Flask provides routing, request handling and response generation with minimal boilerplate.

---

### G

**GET**
HTTP method for retrieving resources. GET requests should be safe (no side effects) and idempotent (repeatable without different outcomes).

---

### H

**HATEOAS (Hypermedia as the Engine of Application State)**
A REST constraint stating that responses should include links enabling discovery of related resources and actions, allowing clients to navigate APIs without hardcoded knowledge.

**Header**
Metadata transmitted with HTTP requests and responses. Headers convey information about content type, caching, authentication, encoding and more.

**HTTP (Hypertext Transfer Protocol)**
The application-layer protocol underlying web communication. HTTP defines how messages are formatted and transmitted, and how servers and clients should respond to various commands.

**HTTPS**
HTTP Secure—HTTP over TLS/SSL encryption. HTTPS protects data in transit from eavesdropping and tampering. Modern APIs should always use HTTPS.

---

### I

**Idempotent**
A property of operations where multiple identical executions produce the same result as a single execution. GET, PUT and DELETE are idempotent; POST is not.

---

### J

**JSON (JavaScript Object Notation)**
A lightweight data interchange format using human-readable text to represent objects as key-value pairs and arrays. The dominant format for web APIs.

**JWT (JSON Web Token)**
A compact, URL-safe token format encoding claims as a signed JSON object. JWTs enable stateless authentication—servers can verify tokens without database lookups.

---

### L

**Latency**
The time delay between sending a request and receiving the first byte of the response. Network latency, server processing time and distance all contribute to overall latency.

---

### M

**Method**
The HTTP verb indicating the desired action. Primary methods are GET, POST, PUT, PATCH, DELETE, HEAD and OPTIONS.

---

### O

**OAuth 2.0**
An authorisation framework enabling applications to obtain limited access to user resources without sharing credentials. OAuth defines flows for different scenarios (web apps, mobile apps, server-to-server).

**Offset Pagination**
A pagination strategy using page numbers or skip/limit parameters. Simple but can produce inconsistent results if data changes during iteration.

---

### P

**Pagination**
The practice of dividing large result sets across multiple responses. Clients request successive pages until all data is retrieved. Common patterns include offset-based, cursor-based and keyset pagination.

**Payload**
The data carried by a request or response, typically in the body. For JSON APIs, the payload is a JSON object.

**POST**
HTTP method for creating resources or triggering actions. POST is neither safe nor idempotent—each request may create new state.

**PUT**
HTTP method for replacing a resource entirely. PUT is idempotent—multiple identical PUTs produce the same final state.

---

### R

**Rate Limiting**
Restricting the number of requests a client may make within a time window. Protects servers from abuse and ensures fair resource allocation. Rate limit information is often conveyed via headers.

**Request**
A message sent from client to server specifying a method, target resource, headers and optional body.

**Response**
A message sent from server to client containing status code, headers and optional body.

**REST (Representational State Transfer)**
An architectural style for distributed systems emphasising statelessness, uniform interfaces, cacheability and layered architecture. RESTful APIs expose resources via URIs and manipulate them through standard HTTP methods.

**Retry-After**
An HTTP header indicating how long a client should wait before retrying a request, typically used with 429 (Too Many Requests) or 503 (Service Unavailable) responses.

**robots.txt**
A text file at a website's root specifying which paths web crawlers may access. Ethical scrapers check and respect robots.txt directives.

---

### S

**Safe (HTTP Method)**
A method is safe if it does not modify server state. GET, HEAD and OPTIONS are safe—they only retrieve information.

**Scraping**
Programmatically extracting data from web pages by parsing HTML. Used when APIs are unavailable but requires ethical consideration regarding rate limits, terms of service and robots.txt.

**Session**
In `requests`, a Session object persists configuration (headers, cookies, authentication) across multiple requests and enables connection pooling for efficiency.

**SSL/TLS**
Protocols providing encrypted communication between clients and servers. TLS (Transport Layer Security) is the modern successor to SSL (Secure Sockets Layer).

**Status Code**
A three-digit number in HTTP responses indicating the result category: 1xx (informational), 2xx (success), 3xx (redirection), 4xx (client error), 5xx (server error).

**Stateless**
A property of HTTP where each request is independent—the server maintains no memory of previous requests. Clients must include all necessary information (authentication, context) in each request.

---

### T

**Timeout**
The maximum time to wait for a response before abandoning a request. Setting appropriate timeouts prevents programs from hanging indefinitely on unresponsive servers.

**Token**
A string granting access to resources. Tokens may be opaque (random strings validated by lookup) or self-contained (JWTs validated by signature verification).

---

### U

**URI (Uniform Resource Identifier)**
A string identifying a resource. URLs (Uniform Resource Locators) are URIs that also specify how to access the resource (protocol, host, path).

**URL Encoding**
The process of converting special characters in URLs to percent-encoded format (e.g., space becomes %20). Query parameters must be URL-encoded.

**User-Agent**
An HTTP header identifying the client software. Polite web clients include descriptive User-Agent strings, often with contact information for operators to report issues.

---

### W

**Webhook**
A mechanism for servers to push notifications to clients when events occur, inverting the typical request-response pattern. Clients register callback URLs that servers invoke.

---

### X

**X- Headers**
Custom HTTP headers, traditionally prefixed with X- (though this convention is now deprecated). Common examples include X-API-Key, X-RateLimit-Limit and X-Request-ID.

---

*Licence: Restrictive — see repository root for terms.*
