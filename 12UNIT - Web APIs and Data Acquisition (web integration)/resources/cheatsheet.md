# 12UNIT: Web APIs and Data Acquisition — Cheatsheet

## Quick Reference

### HTTP Methods

| Method | Purpose | Idempotent | Safe | Has Body |
|--------|---------|------------|------|----------|
| GET | Retrieve resource | Yes | Yes | No |
| POST | Create resource | No | No | Yes |
| PUT | Replace resource | Yes | No | Yes |
| PATCH | Partial update | No | No | Yes |
| DELETE | Remove resource | Yes | No | Optional |
| HEAD | Get headers only | Yes | Yes | No |

### HTTP Status Codes

```
1xx - Informational (100 Continue)
2xx - Success (200 OK, 201 Created, 204 No Content)
3xx - Redirection (301 Moved, 302 Found, 304 Not Modified)
4xx - Client Error (400 Bad Request, 401 Unauthorised, 403 Forbidden, 404 Not Found, 429 Too Many Requests)
5xx - Server Error (500 Internal, 502 Bad Gateway, 503 Unavailable)
```

---

## Python `requests` Library

### Basic Requests

```python
import requests

# GET request
response = requests.get('https://api.example.com/data')
response = requests.get(url, params={'key': 'value'})

# POST request with JSON
response = requests.post(url, json={'field': 'value'})

# POST with form data
response = requests.post(url, data={'field': 'value'})

# With headers
response = requests.get(url, headers={'Authorization': 'Bearer TOKEN'})

# With timeout
response = requests.get(url, timeout=30)
```

### Response Handling

```python
response.status_code      # 200
response.ok               # True if 2xx
response.headers          # Response headers dict
response.text             # Response body as string
response.json()           # Parse JSON response
response.content          # Raw bytes
response.raise_for_status()  # Raise HTTPError if 4xx/5xx
```

### Session Usage

```python
with requests.Session() as session:
    session.headers.update({'Authorization': 'Bearer TOKEN'})
    response = session.get(url)  # Reuses connection and headers
```

### Error Handling

```python
from requests.exceptions import RequestException, HTTPError, Timeout

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
except Timeout:
    print('Request timed out')
except HTTPError as e:
    print(f'HTTP error: {e.response.status_code}')
except RequestException as e:
    print(f'Request failed: {e}')
```

---

## Authentication Patterns

### API Key (Header)

```python
headers = {'X-API-Key': 'your_api_key'}
response = requests.get(url, headers=headers)
```

### API Key (Query Parameter)

```python
params = {'api_key': 'your_api_key'}
response = requests.get(url, params=params)
```

### HTTP Basic Auth

```python
response = requests.get(url, auth=('username', 'password'))
```

### Bearer Token

```python
headers = {'Authorization': 'Bearer your_access_token'}
response = requests.get(url, headers=headers)
```

### OAuth 2.0 Client Credentials

```python
token_response = requests.post(
    'https://oauth.example.com/token',
    data={
        'grant_type': 'client_credentials',
        'client_id': 'CLIENT_ID',
        'client_secret': 'CLIENT_SECRET'
    }
)
access_token = token_response.json()['access_token']
```

---

## Pagination Patterns

### Offset-Based

```python
page = 1
while True:
    response = requests.get(url, params={'page': page, 'per_page': 100})
    items = response.json()['items']
    if not items:
        break
    for item in items:
        yield item
    page += 1
```

### Cursor-Based

```python
cursor = '*'
while cursor:
    response = requests.get(url, params={'cursor': cursor})
    data = response.json()
    for item in data['results']:
        yield item
    cursor = data.get('meta', {}).get('next_cursor')
```

---

## BeautifulSoup

### Basic Parsing

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_content, 'html.parser')
# Or with lxml: BeautifulSoup(html_content, 'lxml')
```

### Finding Elements

```python
# By tag
soup.find('h1')                 # First h1
soup.find_all('p')              # All paragraphs

# By class
soup.find('div', class_='container')
soup.find_all('span', class_='highlight')

# By ID
soup.find(id='main-content')

# CSS selectors
soup.select('div.item > h2')    # Direct child
soup.select('article.post')     # Class selector
soup.select('#header a')        # ID then descendant
```

### Extracting Data

```python
element.text                    # Text content
element.get_text(strip=True)    # Stripped text
element['href']                 # Attribute value
element.get('href', '')         # Safe attribute access
```

---

## Flask API

### Basic Application

```python
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

@app.route('/api/items', methods=['GET'])
def list_items():
    return jsonify({'items': []})

@app.route('/api/items', methods=['POST'])
def create_item():
    data = request.get_json()
    return jsonify(data), 201

@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    # Return 404 if not found
    abort(404, description='Item not found')
    return jsonify(item)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': str(error.description)}), 404

if __name__ == '__main__':
    app.run(debug=True)
```

### Request Data Access

```python
request.args.get('param')       # Query parameter
request.args.get('limit', 10, type=int)  # With default and type
request.get_json()              # JSON body
request.form['field']           # Form data
request.headers['Authorization']  # Header
request.is_json                 # Check content type
```

---

## Rate Limiting

### Exponential Backoff

```python
import time

def fetch_with_backoff(url, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 429:
            wait = 2 ** attempt
            time.sleep(wait)
            continue
        return response
    raise Exception('Max retries exceeded')
```

### Respect Retry-After

```python
if response.status_code == 429:
    wait = int(response.headers.get('Retry-After', 60))
    time.sleep(wait)
```

---

## Environment Variables

```python
import os

# Load credentials from environment
API_KEY = os.environ.get('API_KEY')
API_SECRET = os.environ.get('API_SECRET', 'default_value')

# Check if set
if not API_KEY:
    raise ValueError('API_KEY environment variable not set')
```

---

## Common Headers

### Request Headers

```
Accept: application/json
Content-Type: application/json
Authorization: Bearer <token>
User-Agent: MyApp/1.0 (contact@example.com)
```

### Response Headers

```
Content-Type: application/json
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
Retry-After: 60
```

---

## Research APIs

### CrossRef

```python
response = requests.get(
    'https://api.crossref.org/works',
    params={'query': 'machine learning', 'rows': 100},
    headers={'User-Agent': 'MyApp/1.0 (mailto:user@example.com)'}
)
works = response.json()['message']['items']
```

### OpenAlex

```python
response = requests.get(
    'https://api.openalex.org/works',
    params={'search': 'climate change', 'per-page': 100}
)
works = response.json()['results']
```

---

*Licence: Restrictive — see repository root for terms.*
