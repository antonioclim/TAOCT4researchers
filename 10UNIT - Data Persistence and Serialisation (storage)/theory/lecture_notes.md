# Lecture Notes: Data Persistence and Serialisation

## Introduction

The computational researcher faces a fundamental tension between the transient nature of program execution and the permanent demands of scientific reproducibility. Every simulation, analysis and experiment generates data that exists momentarily in volatile memory before evaporating when the program terminates. Data persistence bridges this gap, transforming ephemeral computational states into durable artefacts suitable for verification, sharing and long-term archival.

As Wilson et al. (2017) observe in their foundational work on scientific computing practices, the practical constraints of research computing often necessitate persistent storage: "Large datasets may not fit entirely in memory, requiring streaming processing. Simulations may run for days or weeks, necessitating checkpointing—saving intermediate states to files." These requirements pervade modern research across every domain.

This unit examines the mechanisms by which Python programs interact with persistent storage, from elementary file operations through structured serialisation formats to relational database systems. We shall discover that the choice of persistence strategy profoundly influences research workflows—affecting not merely storage efficiency but also data integrity, interoperability and the fundamental reproducibility of scientific findings.

---

## Part I: File Input/Output Fundamentals

### The File Abstraction

Operating systems present files as sequential streams of bytes, abstracting away the complexities of physical storage media—the spinning platters, flash memory cells and network protocols that constitute modern storage infrastructure. This abstraction, inherited from Unix's elegant "everything is a file" philosophy, allows programmers to interact with diverse storage media through a uniform interface. Python's `open()` function creates file objects that mediate between program memory and these byte streams, providing methods for reading, writing and navigating file contents.

Understanding the distinction between text and binary modes proves essential for correct file handling. Text mode (`'r'`, `'w'`, `'a'`) interprets bytes as character sequences according to a specified encoding (defaulting to platform-specific encodings if not explicitly provided—a source of countless portability bugs). Binary mode (`'rb'`, `'wb'`) treats files as raw byte sequences, appropriate for images, compressed archives and any format where character interpretation would corrupt the data.

The mode parameter to `open()` governs the operational semantics of the resulting file object:

```python
from pathlib import Path

# Read mode: file must exist
with open('existing.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Write mode: creates or truncates
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write('New content')

# Append mode: creates or extends
with open('log.txt', 'a', encoding='utf-8') as f:
    f.write('Additional entry\n')

# Exclusive creation: fails if file exists
with open('unique.txt', 'x', encoding='utf-8') as f:
    f.write('Guaranteed fresh')
```

The `pathlib.Path` class provides an object-oriented interface to filesystem operations that surpasses the traditional `os.path` module in clarity and cross-platform reliability:

```python
from pathlib import Path

data_dir = Path('experiments') / 'trial_001' / 'results'
data_dir.mkdir(parents=True, exist_ok=True)

for csv_file in data_dir.glob('*.csv'):
    print(f"Processing: {csv_file.name}")
```

### Context Managers and Resource Safety

File handles constitute operating system resources that must be explicitly released. Each open file consumes a file descriptor—a limited resource that, when exhausted, prevents any further file operations. More insidiously, holding file handles prevents other processes from accessing the same files on certain operating systems, leading to mysterious "file in use" errors in collaborative research environments.

The context manager protocol ensures cleanup occurs even when exceptions interrupt normal execution. The `with` statement invokes the file object's `__enter__` method to acquire the resource, then guarantees invocation of `__exit__` regardless of how control exits the block:

```python
# Without context manager: leak risk
f = open('data.txt', 'r')
try:
    content = f.read()
    process(content)  # If this raises, file handle leaks
finally:
    f.close()

# With context manager: guaranteed cleanup
with open('data.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    process(content)  # Even if this raises, file closes
# File automatically closed here, guaranteed
```

This pattern extends beyond files to any resource requiring cleanup: database connections, network sockets, locks and temporary directories. Custom resources implement context management through the `__enter__` and `__exit__` methods or utilise the `contextlib` module's `@contextmanager` decorator for simpler cases.

The principle generalises: **resource acquisition should be paired with guaranteed release**. Context managers encode this pairing syntactically, making resource leaks visible as code structure violations rather than subtle runtime bugs.

### Streaming and Memory Efficiency

Loading entire files into memory becomes impractical for datasets exceeding available RAM. Streaming approaches process data incrementally, maintaining constant memory usage regardless of file size:

```python
def process_large_file(filepath: Path) -> dict[str, int]:
    """Count occurrences of each word in a large text file."""
    word_counts: dict[str, int] = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:  # Iterate line by line
            for word in line.lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
    
    return word_counts
```

For binary data, the `read(size)` method enables chunked processing:

```python
def compute_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute SHA-256 hash of file contents efficiently."""
    import hashlib
    
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    
    return hasher.hexdigest()
```

### Atomic Write Operations

Write operations that fail midway leave files in corrupted partial states. Atomic writes address this by composing the complete output in a temporary location before an instantaneous rename operation:

```python
import tempfile
from pathlib import Path

def atomic_write(filepath: Path, content: str) -> None:
    """Write content atomically to prevent partial updates."""
    directory = filepath.parent
    
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding='utf-8',
        dir=directory,
        delete=False
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    
    tmp_path.replace(filepath)  # Atomic on POSIX systems
```

---

## Part II: Serialisation Formats

### JSON: The Universal Interchange Format

JavaScript Object Notation has emerged as the dominant format for data interchange, supported natively by virtually every programming language and platform. JSON maps directly to Python's fundamental data types: objects become dictionaries, arrays become lists and primitives (strings, numbers, booleans, null) map to their Python equivalents.

```python
import json
from datetime import datetime
from dataclasses import dataclass, asdict

@dataclass
class ExperimentResult:
    experiment_id: str
    timestamp: str
    measurements: list[float]
    parameters: dict[str, float]

def save_results(results: list[ExperimentResult], filepath: Path) -> None:
    """Serialise experiment results to JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(
            [asdict(r) for r in results],
            f,
            indent=2,
            ensure_ascii=False
        )

def load_results(filepath: Path) -> list[ExperimentResult]:
    """Deserialise experiment results from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return [ExperimentResult(**item) for item in data]
```

JSON's limitations include the inability to represent datetime objects, complex numbers or arbitrary Python objects directly. Custom encoders and decoders extend JSON's capabilities:

```python
class ResearchJSONEncoder(json.JSONEncoder):
    """Extended JSON encoder for research data types."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return {'__datetime__': obj.isoformat()}
        if isinstance(obj, complex):
            return {'__complex__': [obj.real, obj.imag]}
        if isinstance(obj, Path):
            return {'__path__': str(obj)}
        return super().default(obj)

def research_decoder(dct: dict):
    """Decode research-specific JSON types."""
    if '__datetime__' in dct:
        return datetime.fromisoformat(dct['__datetime__'])
    if '__complex__' in dct:
        return complex(*dct['__complex__'])
    if '__path__' in dct:
        return Path(dct['__path__'])
    return dct
```

### CSV: Tabular Data Exchange

Comma-separated values remain ubiquitous for tabular research data despite—or perhaps because of—their simplicity. The `csv` module handles the tedious details of quoting, escaping and delimiter management:

```python
import csv
from pathlib import Path

def read_measurements(filepath: Path) -> list[dict[str, str]]:
    """Read CSV measurements as list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_measurements(
    data: list[dict[str, float]],
    filepath: Path,
    fieldnames: list[str]
) -> None:
    """Write measurements to CSV with specified column order."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
```

Edge cases abound in CSV processing: embedded commas within quoted fields, newlines within cell values, varying delimiters (tabs, semicolons) and inconsistent quoting strategies. Explicit configuration defends against ambiguity:

```python
# European-style CSV with semicolon delimiter
reader = csv.DictReader(f, delimiter=';', quotechar='"')

# Tab-separated values
reader = csv.DictReader(f, delimiter='\t')
```

### Binary Formats: Pickle and Beyond

Python's `pickle` module serialises arbitrary object graphs, preserving class instances, circular references and function objects:

```python
import pickle

def checkpoint_state(state: dict, filepath: Path) -> None:
    """Save computation state for later resumption."""
    with open(filepath, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

def restore_state(filepath: Path) -> dict:
    """Restore previously checkpointed state."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
```

**Security Warning**: Pickle executes arbitrary code during deserialisation. Never unpickle data from untrusted sources. This vulnerability makes Pickle unsuitable for data interchange between systems; reserve it for local checkpointing within controlled environments.

### Columnar Formats: Parquet

Apache Parquet stores tabular data in a column-oriented format that enables efficient compression and selective column retrieval:

```python
import pandas as pd
from pathlib import Path

def save_dataframe_parquet(
    df: pd.DataFrame,
    filepath: Path,
    compression: str = 'snappy'
) -> None:
    """Save DataFrame to Parquet with compression."""
    df.to_parquet(filepath, compression=compression, index=False)

def load_columns(filepath: Path, columns: list[str]) -> pd.DataFrame:
    """Load only specified columns from Parquet file."""
    return pd.read_parquet(filepath, columns=columns)
```

Parquet's columnar storage yields dramatic benefits for analytical queries that access subsets of columns, achieving both compression ratios of 5–10× and read speeds exceeding row-oriented formats by orders of magnitude for appropriate workloads.

---

## Part III: Relational Databases with SQLite

### The Relational Model

Relational databases organise data into tables (relations) where each row (tuple) represents an entity and each column (attribute) represents a property. This structure provides mathematical foundations for data manipulation through relational algebra and its practical expression, SQL. Unlike hierarchical or document-oriented storage, the relational model excels at representing complex relationships between entities—precisely the characteristic of most research data.

Consider a longitudinal study tracking patients across multiple clinic visits with various measurements at each visit. File-based storage would require either massive redundancy (storing patient demographics with every measurement) or complex cross-referencing schemes. The relational model naturally separates patients, visits and measurements into distinct tables connected by foreign keys, eliminating redundancy whilst preserving data integrity through referential constraints.

SQLite provides a complete relational database engine in a single file, requiring no server configuration or administration. Unlike PostgreSQL or MySQL, SQLite operates as a library linked directly into the application, eliminating network latency and configuration complexity. Despite its simplicity, SQLite handles datasets of hundreds of gigabytes reliably and powers applications from smartphones to aircraft—and is perfectly suited for research data management.

Python's `sqlite3` module offers native access:

```python
import sqlite3
from pathlib import Path

def create_research_database(db_path: Path) -> None:
    """Initialise research database schema."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS measurements (
                measurement_id INTEGER PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                sensor_name TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TEXT NOT NULL,
                FOREIGN KEY (experiment_id) 
                    REFERENCES experiments(experiment_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_measurements_experiment 
                ON measurements(experiment_id);
        """)
```

### Parameterised Queries and Security

String concatenation in SQL queries creates injection vulnerabilities—one of the most prevalent and dangerous classes of security flaws in software. When user-provided data is interpolated directly into SQL strings, malicious input can alter the query's meaning entirely. A username input of `'; DROP TABLE users; --` transforms an innocent login query into database destruction.

Parameterised queries separate code from data, passing user inputs as bound parameters that the database engine treats as literal values regardless of their content. This separation makes injection attacks impossible by design:

```python
def add_measurement(
    conn: sqlite3.Connection,
    experiment_id: int,
    sensor: str,
    value: float,
    timestamp: str
) -> int:
    """Insert measurement with parameterised query.
    
    The ? placeholders are bound to parameter values by the
    database engine, ensuring safe handling of any input.
    """
    cursor = conn.execute(
        """
        INSERT INTO measurements 
            (experiment_id, sensor_name, value, recorded_at)
        VALUES (?, ?, ?, ?)
        """,
        (experiment_id, sensor, value, timestamp)
    )
    return cursor.lastrowid
```

Every SQL query that incorporates external data—whether from user input, file contents or API responses—must use parameterised queries. This requirement admits no exceptions in production code.

### Transactions and ACID Properties

Database transactions group operations into atomic units that either complete entirely or leave no trace:

```python
def transfer_samples(
    conn: sqlite3.Connection,
    from_batch: int,
    to_batch: int,
    sample_ids: list[int]
) -> None:
    """Atomically transfer samples between batches."""
    try:
        conn.execute("BEGIN TRANSACTION")
        
        for sample_id in sample_ids:
            conn.execute(
                "UPDATE samples SET batch_id = ? WHERE sample_id = ?",
                (to_batch, sample_id)
            )
        
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
```

---

## Part IV: Data Versioning and Integrity

### Cryptographic Checksums

Hash functions transform arbitrary data into fixed-length fingerprints that detect even single-bit modifications:

```python
import hashlib
from pathlib import Path

def compute_checksum(filepath: Path, algorithm: str = 'sha256') -> str:
    """Compute cryptographic checksum of file."""
    hasher = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    
    return hasher.hexdigest()
```

### Manifest Files

Data manifests record checksums and metadata for collections of files, enabling verification of dataset integrity:

```python
import json
from datetime import datetime
from pathlib import Path

def generate_manifest(data_dir: Path) -> dict:
    """Generate manifest for all files in directory."""
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'files': {}
    }
    
    for filepath in data_dir.rglob('*'):
        if filepath.is_file():
            relative_path = filepath.relative_to(data_dir)
            manifest['files'][str(relative_path)] = {
                'size': filepath.stat().st_size,
                'sha256': compute_checksum(filepath),
                'modified': datetime.fromtimestamp(
                    filepath.stat().st_mtime
                ).isoformat()
            }
    
    return manifest
```

---

## Summary

Data persistence transforms ephemeral computations into permanent scientific artefacts. The choice of persistence mechanism—from simple text files through structured serialisation formats to relational databases—should align with the specific requirements of research workflows: human readability, storage efficiency, query performance and long-term accessibility.

The techniques covered in this unit address distinct requirements along this spectrum. Plain text files with UTF-8 encoding offer maximum portability and human readability for configuration and simple data interchange. JSON provides structure whilst retaining editability, ideal for metadata and configuration hierarchies. CSV remains the lingua franca for tabular data exchange between tools and collaborators. SQLite offers query capabilities and data integrity guarantees essential for complex, relational research data. Binary formats like Pickle enable efficient checkpointing of computation state, though their security implications restrict appropriate use cases to trusted, local contexts.

The preprocessing trade-off—described by Garcia-Molina as the balance between storage cost and query efficiency—pervades every persistence decision. Maintaining database indices accelerates queries at the cost of increased storage and write overhead. Compressing archives reduces storage requirements whilst increasing access latency. Understanding these trade-offs enables informed decisions appropriate to specific research contexts.

Key principles to remember:

1. **Explicit encodings** prevent silent data corruption across platforms—always specify UTF-8 for text files
2. **Context managers** ensure resource cleanup even during exceptions—never write `open()` without `with`
3. **Streaming approaches** enable processing datasets larger than memory—iterate rather than load
4. **Atomic operations** prevent partial updates that corrupt data integrity—use temporary files with rename
5. **Parameterised queries** defend against injection vulnerabilities—never concatenate user input into SQL
6. **Cryptographic checksums** verify data integrity across time and space—hash early, hash often

These techniques form the foundation for reproducible computational research, ensuring that today's analyses can be verified, extended and built upon by future investigators. As Wilson et al. emphasise in their discussion of scientific computing practices, "Large datasets may not fit entirely in memory, requiring streaming processing. Simulations may run for days or weeks, necessitating checkpointing—saving intermediate states to files." The persistence strategies introduced here address precisely these practical constraints whilst maintaining the rigour that reproducible research demands.

---

## Self-Assessment Questions

Before proceeding to the laboratory exercises, consider the following questions:

1. What encoding would you specify when opening a text file for cross-platform sharing?
2. Why are context managers preferable to explicit `close()` calls?
3. Under what circumstances would you choose JSON over CSV for data storage?
4. Why should Pickle never be used with data from untrusted sources?
5. What problem do parameterised SQL queries solve?
6. How do cryptographic checksums contribute to research reproducibility?

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Unit 10 — Lecture Notes*

© 2025 Antonio Clim. All rights reserved.
