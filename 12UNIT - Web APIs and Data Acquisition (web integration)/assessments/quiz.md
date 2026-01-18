# 12UNIT: Web APIs and Data Acquisition — Quiz

## Instructions

This quiz assesses your understanding of HTTP fundamentals, REST principles,
API consumption patterns and web scraping ethics. Answer all questions.
Total marks: 40 (20 questions × 2 marks each).

Time allowed: 30 minutes

---

## Section A: HTTP Fundamentals (Questions 1-5)

### Question 1
Which HTTP method should be used to retrieve a resource without modifying server state?

A) POST  
B) GET  
C) PUT  
D) PATCH

---

### Question 2
An API returns status code 429. What does this indicate?

A) The resource was not found  
B) The request was malformed  
C) The client has exceeded rate limits  
D) The server encountered an internal error

---

### Question 3
Which of the following is NOT a characteristic of HTTP's stateless nature?

A) Each request contains all information needed for processing  
B) The server maintains no session state between requests  
C) Load balancing is simplified across multiple servers  
D) Authentication tokens persist automatically between requests

---

### Question 4
What is the primary purpose of the `Content-Type` header in HTTP requests?

A) Specify the encoding of the URL  
B) Indicate the media type of the request body  
C) Define the maximum response size accepted  
D) Set the connection timeout value

---

### Question 5
Which status code category indicates that the request was received and is being processed?

A) 1xx  
B) 2xx  
C) 3xx  
D) 4xx

---

## Section B: REST Principles (Questions 6-10)

### Question 6
According to REST architectural principles, which constraint enables horizontal scaling of servers?

A) Layered system  
B) Statelessness  
C) Code on demand  
D) Uniform interface

---

### Question 7
Which URI design follows REST conventions for retrieving a specific author's publications?

A) `/getPublicationsByAuthor?id=123`  
B) `/authors/123/publications`  
C) `/publications/author/123/get`  
D) `/api/fetchAuthorPubs/123`

---

### Question 8
What does HATEOAS stand for in REST architecture?

A) Hypertext As The Engine Of Application State  
B) HTTP Application Transfer Engine Over API Services  
C) Hierarchical API Transfer with Enhanced Object Access State  
D) Hyperlink-Accessible Text Encoding for API Services

---

### Question 9
Which statement about REST resources is correct?

A) Resources must correspond directly to database tables  
B) Resources are identified by URIs and accessed through representations  
C) Resources can only be formatted as JSON  
D) Resources must include executable code for client-side processing

---

### Question 10
What is the correct response status code when a POST request successfully creates a new resource?

A) 200 OK  
B) 201 Created  
C) 204 No Content  
D) 202 Accepted

---

## Section C: API Consumption (Questions 11-14)

### Question 11
In the `requests` library, which method is most appropriate for sending JSON data in a POST request?

A) `requests.post(url, data=json_string)`  
B) `requests.post(url, json=data_dict)`  
C) `requests.post(url, body=json_string)`  
D) `requests.post(url, content=data_dict)`

---

### Question 12
What is the primary advantage of using a `requests.Session` object for multiple API calls?

A) Automatic JSON parsing of all responses  
B) Connection pooling and persistent configuration  
C) Built-in rate limiting enforcement  
D) Automatic retry on all errors

---

### Question 13
When implementing exponential backoff retry logic, which type of errors should typically be retried?

A) 400 Bad Request  
B) 401 Unauthorised  
C) 404 Not Found  
D) 503 Service Unavailable

---

### Question 14
How should API credentials be securely stored in a Python application?

A) Hardcoded in the source file for convenience  
B) In a public configuration file in the repository  
C) Using environment variables or secure credential stores  
D) Embedded in URL query parameters

---

## Section D: Authentication (Questions 15-17)

### Question 15
In OAuth 2.0 Client Credentials flow, what is exchanged for an access token?

A) Username and password  
B) Client ID and client secret  
C) Refresh token and scope  
D) API key and user email

---

### Question 16
What is the correct format for a Bearer token in the Authorization header?

A) `Authorization: Bearer=<token>`  
B) `Authorization: Bearer <token>`  
C) `Authorization: Token <token>`  
D) `Bearer-Token: <token>`

---

### Question 17
Which authentication mechanism allows token verification without a database lookup?

A) Session-based authentication  
B) API key authentication  
C) JWT (JSON Web Tokens)  
D) Basic authentication

---

## Section E: Web Scraping and Flask (Questions 18-20)

### Question 18
What should an ethical web scraper check before accessing a website programmatically?

A) The website's CSS framework  
B) The `robots.txt` file  
C) The server's operating system  
D) The database schema

---

### Question 19
In Flask, which decorator is used to define a route that handles both GET and POST requests?

A) `@app.route('/path', method=['GET', 'POST'])`  
B) `@app.route('/path', methods=['GET', 'POST'])`  
C) `@app.methods('/path', ['GET', 'POST'])`  
D) `@app.endpoint('/path', type=['GET', 'POST'])`

---

### Question 20
When a Flask endpoint cannot find a requested resource, which function should be called?

A) `raise NotFound()`  
B) `return None`  
C) `abort(404)`  
D) `exit(404)`

---

## Answer Key

*(For instructor use only)*

1. B
2. C
3. D
4. B
5. A
6. B
7. B
8. A
9. B
10. B
11. B
12. B
13. D
14. C
15. B
16. B
17. C
18. B
19. B
20. C

---

*Licence: Restrictive — see repository root for terms.*
