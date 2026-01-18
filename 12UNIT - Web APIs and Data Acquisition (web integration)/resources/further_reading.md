# 12UNIT: Web APIs and Data Acquisition — Further Reading

## Curated Resources for Deeper Study

---

## Primary Sources and Standards

### HTTP Protocol

**RFC 9110: HTTP Semantics** (2022)
https://www.rfc-editor.org/rfc/rfc9110
The authoritative specification for HTTP methods, status codes, headers and protocol semantics. Essential reference for understanding HTTP behaviour at a fundamental level.

**RFC 9111: HTTP Caching** (2022)
https://www.rfc-editor.org/rfc/rfc9111
Specifies HTTP caching mechanisms, including Cache-Control directives and validation procedures. Critical for building efficient API clients.

**MDN Web Docs: HTTP**
https://developer.mozilla.org/en-US/docs/Web/HTTP
Comprehensive, accessible documentation covering all aspects of HTTP. Excellent balance between depth and readability.

### REST Architecture

**Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software Architectures.** Doctoral dissertation, University of California, Irvine.
https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm
The original articulation of REST principles by one of HTTP's principal architects. Chapter 5 defines REST constraints; essential reading for understanding architectural rationale.

**Richardson, L. (2008). Justice Will Take Us Millions of Intricate Moves.**
https://www.crummy.com/writing/speaking/2008-QCon/
Introduces the Richardson Maturity Model, a useful framework for evaluating API "RESTfulness" across levels from plain HTTP to full hypermedia.

---

## Textbooks and Comprehensive Guides

### API Design

**Masse, M. (2011). *REST API Design Rulebook.* O'Reilly Media.**
Concise handbook of REST API design patterns and anti-patterns. Practical guidance on URI design, HTTP method usage, representation formats and error handling.

**Lauret, A. (2019). *The Design of Web APIs.* Manning Publications.**
Comprehensive guide to API design from user perspective. Emphasises API usability, documentation and developer experience alongside technical correctness.

### Python Web Development

**Grinberg, M. (2018). *Flask Web Development*, 2nd Edition. O'Reilly Media.**
Authoritative guide to Flask, covering application structure, templates, databases, RESTful services and deployment. Essential for extending beyond basic API development.

**Reitz, K. & Schlusser, T. (2016). *The Hitchhiker's Guide to Python.* O'Reilly Media.**
Best practices for Python development, including sections on HTTP libraries, testing and project structure. Available free online at https://docs.python-guide.org/

### Web Scraping

**Mitchell, R. (2018). *Web Scraping with Python*, 2nd Edition. O'Reilly Media.**
Comprehensive coverage of web scraping techniques, from basic HTML parsing through JavaScript rendering, legal considerations and large-scale scraping architecture.

---

## Library Documentation

### requests

**Official Documentation**
https://requests.readthedocs.io/
Complete reference for the requests library, including advanced features like streaming, SSL configuration and custom authentication.

**Requests Toolbelt**
https://toolbelt.readthedocs.io/
Collection of utilities extending requests: multipart encoding, streaming uploads, session mocking and more.

### BeautifulSoup

**Beautiful Soup Documentation**
https://www.crummy.com/software/BeautifulSoup/bs4/doc/
Official documentation with extensive examples. Covers parsing, navigation, search methods and output formatting.

### Flask

**Flask Documentation**
https://flask.palletsprojects.com/
Official Flask documentation, including tutorials, API reference and deployment guidance.

**Flask-RESTful**
https://flask-restful.readthedocs.io/
Extension simplifying REST API construction in Flask with resource classes and request parsing.

---

## Research APIs

### Bibliometric Data

**CrossRef API Documentation**
https://api.crossref.org/swagger-ui/index.html
REST API providing access to scholarly metadata for over 130 million works. Excellent for bibliometric research.

**OpenAlex Documentation**
https://docs.openalex.org/
Open scholarly knowledge graph with comprehensive API. Covers works, authors, institutions, concepts and venues with extensive filtering and grouping.

**Semantic Scholar API**
https://api.semanticscholar.org/
AI-powered scholarly search API with citation analysis, paper recommendations and author disambiguation.

### Other Research APIs

**NASA APIs**
https://api.nasa.gov/
Collection of APIs providing access to NASA data, including astronomy imagery, Mars rover photos and Earth observation data.

**OpenWeather API**
https://openweathermap.org/api
Weather data API useful for environmental research, including historical data, forecasts and air quality.

**World Bank Data API**
https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
Economic indicators, development statistics and demographic data for research applications.

---

## Online Courses and Tutorials

### API Development

**Real Python: Python and REST APIs**
https://realpython.com/api-integration-in-python/
Comprehensive tutorial series covering API consumption and Flask API development with practical examples.

**Postman Learning Center**
https://learning.postman.com/
Tutorials on API testing, documentation and development workflows using Postman.

### Web Scraping

**Scrapy Documentation**
https://docs.scrapy.org/
Framework for large-scale web scraping. More complex than BeautifulSoup but essential for production scraping systems.

---

## Ethics and Legal Considerations

**Krotov, V. & Silva, L. (2018). Legality and Ethics of Web Scraping.** *Twenty-fourth Americas Conference on Information Systems*.
Academic analysis of legal and ethical considerations in web scraping, including terms of service, copyright and the Computer Fraud and Abuse Act.

**robots.txt Specification**
https://www.robotstxt.org/
Documentation for the Robots Exclusion Protocol, including syntax, examples and best practices for compliance.

---

## Tools and Testing

**httpbin.org**
https://httpbin.org/
HTTP request/response testing service. Invaluable for testing client behaviour without affecting real APIs.

**Postman**
https://www.postman.com/
API development platform for testing, documenting and monitoring APIs. Free tier available for individuals.

**HTTPie**
https://httpie.io/
Command-line HTTP client designed for testing and debugging APIs. More user-friendly than curl for manual testing.

**mitmproxy**
https://mitmproxy.org/
Interactive HTTPS proxy for inspecting, modifying and replaying HTTP traffic. Useful for debugging complex API interactions.

---

## Academic Papers

**Fielding, R. T. & Taylor, R. N. (2002). Principled Design of the Modern Web Architecture.** *ACM Transactions on Internet Technology*, 2(2), 115-150.
Expanded treatment of REST principles with discussion of design alternatives and trade-offs.

**Perego, A., et al. (2017). Survey on Web APIs.** *Future Generation Computer Systems*, 74, 227-241.
Comprehensive survey of Web API technologies, including REST, GraphQL and emerging patterns.

---

## Community Resources

**Stack Overflow**
https://stackoverflow.com/questions/tagged/python-requests
Active community for troubleshooting requests, Flask and general API development questions.

**Reddit r/learnpython**
https://www.reddit.com/r/learnpython/
Supportive community for Python learners with frequent API-related discussions.

**Python Discord**
https://pythondiscord.com/
Real-time help for Python development, including web development channels.

---

*Licence: Restrictive — see repository root for terms.*
