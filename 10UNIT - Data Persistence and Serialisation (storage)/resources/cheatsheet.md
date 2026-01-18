# Data Persistence Cheatsheet

> Quick reference for file operations, serialisation formats and database fundamentals.

---

## File Modes

| Mode | Operation | Creates | Truncates | Pointer |
|------|-----------|---------|-----------|---------|
| `'r'` | Read | No | No | Start |
| `'w'` | Write | Yes | Yes | Start |
| `'a'` | Append | Yes | No | End |
| `'x'` | Exclusive create | Yes | — | Start |
| `'b'` | Binary modifier | — | — | — |
| `'+'` | Read+Write | — | — | — |

**Mode combinations**: `'rb'` (read binary), `'wb'` (write binary), `'r+'` (read and write existing), `'w+'` (write and read, truncates), `'a+'` (append and read).

---

## File Operations

```python
# Read entire file
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Read lines into list
with open('file.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Iterate lines (memory-efficient)
with open('file.txt', 'r', encoding='utf-8') as f:
    for line in f:
        process(line.strip())

# Write file
with open('file.txt', 'w', encoding='utf-8') as f:
    f.write('content')

# Write multiple lines
with open('file.txt', 'w', encoding='utf-8') as f:
    f.writelines(['line1\n', 'line2\n'])

# Binary read/write
with open('file.bin', 'rb') as f:
    data = f.read()

with open('file.bin', 'wb') as f:
    f.write(bytes_data)
```

---

## pathlib Essentials

```python
from pathlib import Path

# Path construction
path = Path('data') / 'file.csv'  # Join paths
path = Path.cwd() / 'output'      # Current working directory
path = Path.home() / '.config'    # User home directory

# Path inspection
path.exists()                      # Check existence
path.is_file()                     # Is it a file?
path.is_dir()                      # Is it a directory?
path.parent                        # Parent directory
path.name                          # Filename with extension
path.stem                          # Filename without extension
path.suffix                        # Extension (.csv)
path.suffixes                      # All extensions ['.tar', '.gz']

# Directory operations
path.mkdir(parents=True, exist_ok=True)  # Create directories
path.rmdir()                              # Remove empty directory

# File discovery
list(path.glob('*.csv'))           # Find matching files
list(path.rglob('*.csv'))          # Recursive glob
list(path.iterdir())               # List directory contents

# File operations
path.read_text(encoding='utf-8')   # Read text content
path.write_text(content, encoding='utf-8')  # Write text
path.read_bytes()                  # Read binary
path.write_bytes(data)             # Write binary
path.rename(new_path)              # Rename/move
path.unlink()                      # Delete file
```

---

## JSON Operations

```python
import json

# String conversion
json_str = json.dumps(data, indent=2)
data = json.loads(json_str)

# File operations
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Custom serialisation (datetime, Decimal, etc.)
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json.dumps(data, cls=CustomEncoder)
```

---

## CSV Operations

```python
import csv

# Read as dictionaries
with open('data.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['column_name'])

# Write from dictionaries
with open('out.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['col1', 'col2'])
    writer.writeheader()
    writer.writerows(data)

# Read as lists
with open('data.csv', 'r', encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header
    for row in reader:
        process(row)

# Handle different delimiters
reader = csv.reader(f, delimiter='\t')    # TSV
reader = csv.reader(f, delimiter=';')     # European format
```

---

## SQLite Operations

```python
import sqlite3

# Connect and setup
conn = sqlite3.connect('data.db')
conn.row_factory = sqlite3.Row        # Access columns by name
conn.execute("PRAGMA foreign_keys = ON")

# Parameterised query (SAFE)
cursor = conn.execute(
    "SELECT * FROM users WHERE name = ?",
    (user_input,)
)

# Named parameters
cursor = conn.execute(
    "SELECT * FROM users WHERE name = :name AND age > :age",
    {"name": user_name, "age": min_age}
)

# Insert and get ID
cursor = conn.execute(
    "INSERT INTO items (name) VALUES (?)",
    (name,)
)
item_id = cursor.lastrowid
conn.commit()

# Bulk insert
conn.executemany(
    "INSERT INTO items (name, value) VALUES (?, ?)",
    [(n, v) for n, v in data]
)
conn.commit()

# Context manager pattern
with sqlite3.connect('data.db') as conn:
    conn.execute("INSERT INTO ...")
    # Auto-commits on success, rollback on exception
```

---

## SQL Quick Reference

```sql
-- Create table with constraints
CREATE TABLE measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sensor TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(sensor, timestamp)
);

-- Insert
INSERT INTO measurements (sensor, value) VALUES ('T001', 23.5);

-- Select with filter and ordering
SELECT * FROM measurements 
WHERE sensor = 'T001' 
ORDER BY timestamp DESC
LIMIT 100;

-- Aggregation with grouping
SELECT sensor, AVG(value) AS avg_value, COUNT(*) AS count
FROM measurements 
GROUP BY sensor
HAVING COUNT(*) > 10;

-- Join tables
SELECT m.*, s.name, s.location
FROM measurements m 
JOIN sensors s ON m.sensor_id = s.id
WHERE s.location = 'Lab A';

-- Update with condition
UPDATE measurements SET value = 24.0 WHERE id = 1;

-- Delete with condition
DELETE FROM measurements WHERE timestamp < '2024-01-01';

-- Create index for performance
CREATE INDEX idx_sensor_time ON measurements(sensor, timestamp);
```

---

## Pickle (Use Carefully!)

```python
import pickle

# Save (trusted data only)
with open('data.pkl', 'wb') as f:
    pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load (NEVER from untrusted sources!)
with open('data.pkl', 'rb') as f:
    obj = pickle.load(f)
```

⚠️ **Security Warning**: Never unpickle data from unknown sources. Pickle can execute arbitrary code during deserialisation.

---

## Compression

```python
import gzip
import lzma

# gzip compression
with gzip.open('data.gz', 'wt', encoding='utf-8') as f:
    f.write(text_content)

with gzip.open('data.gz', 'rt', encoding='utf-8') as f:
    content = f.read()

# Binary compression
with gzip.open('data.gz', 'wb') as f:
    f.write(data_bytes)

# lzma (better ratio, slower)
with lzma.open('data.xz', 'wt') as f:
    f.write(content)
```

---

## Checksums and Integrity

```python
import hashlib

def file_checksum(path, algorithm='sha256'):
    """Calculate cryptographic checksum of file."""
    hasher = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

# Verify integrity
expected = "a1b2c3..."
actual = file_checksum('data.csv')
if actual != expected:
    raise ValueError("Data integrity check failed")
```

---

## Common Patterns

### Atomic Write (Prevent Corruption)

```python
import tempfile
from pathlib import Path

def atomic_write(path, content):
    """Write file atomically to prevent corruption."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        mode='w', dir=path.parent, delete=False, encoding='utf-8'
    ) as f:
        f.write(content)
        temp = Path(f.name)
    temp.replace(path)  # Atomic rename
```

### Streaming Large Files

```python
def process_large_file(path):
    """Process file line by line (memory-efficient)."""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield process(line)

# Chunked binary reading
def read_chunks(path, chunk_size=65536):
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk
```

### Transaction Pattern

```python
import sqlite3

def safe_batch_insert(db_path, records):
    """Insert records with transaction safety."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO data (col1, col2) VALUES (?, ?)",
            records
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
```

---

## Format Selection Guide

| Format | Use When | Avoid When |
|--------|----------|------------|
| **JSON** | Configuration, APIs, metadata | Large numerical arrays |
| **CSV** | Tabular data, interoperability | Complex nested structures |
| **Pickle** | Python-only, complex objects | Untrusted sources, long-term storage |
| **Parquet** | Large datasets, analytics | Small files, simple structures |
| **SQLite** | Relational data, querying | Distributed systems, write-heavy |

---

## Error Handling Patterns

```python
from pathlib import Path
import json

def safe_read_json(path):
    """Read JSON with comprehensive error handling."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")
    except PermissionError:
        raise PermissionError(f"Cannot read {path}: permission denied")
```

---

## pandas Integration

```python
import pandas as pd

# Read various formats
df = pd.read_csv('data.csv', encoding='utf-8')
df = pd.read_json('data.json')
df = pd.read_parquet('data.parquet')
df = pd.read_sql('SELECT * FROM table', conn)

# Write various formats
df.to_csv('output.csv', index=False, encoding='utf-8')
df.to_json('output.json', orient='records', indent=2)
df.to_parquet('output.parquet', engine='pyarrow')
df.to_sql('table', conn, if_exists='replace', index=False)
```

---

© 2025 Antonio Clim. All rights reserved.
