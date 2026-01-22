# Glossary: Data Persistence and Serialisation

## Core Concepts

### ACID Properties
The four guarantees provided by database transactions: **Atomicity** (all-or-nothing execution), **Consistency** (valid state transitions), **Isolation** (concurrent transactions don't interfere), and **Durability** (committed changes survive failures).

### Atomic Operation
An operation that either completes entirely or has no effect. In file systems, atomic writes prevent partial updates that could corrupt data.

### Binary Format
A file format that stores data as raw bytes rather than human-readable text. Examples include Pickle, Parquet, and HDF5. Typically more space-efficient but not human-readable.

### Checkpointing
The practice of saving intermediate computation states to persistent storage, enabling recovery from failures without restarting from scratch. Essential for long-running simulations.

### Checksum
A fixed-size value computed from data using a hash function. Used to verify data integrity by comparing checksums before and after transfer or storage.

### Context Manager
A Python object implementing `__enter__` and `__exit__` methods, used with the `with` statement to ensure proper resource management (e.g., automatic file closing).

### CRUD Operations
The four basic operations on persistent data: **Create**, **Read**, **Update**, and **Delete**. Forms the foundation of data management interfaces.

### CSV (Comma-Separated Values)
A text-based tabular data format where values are separated by commas (or other delimiters). Universal support but limited type information.

### Deserialisation
The process of reconstructing in-memory objects from their serialised (stored) representation. The inverse of serialisation.

### DOI (Digital Object Identifier)
A persistent identifier used to uniquely identify digital objects, commonly used for academic publications and datasets.

### Encoding
The scheme used to represent characters as bytes. UTF-8 is the dominant standard, supporting all Unicode characters whilst being backward-compatible with ASCII.

### File Handle
A reference to an open file provided by the operating system. Must be explicitly closed (or managed via context managers) to release system resources.

### Foreign Key
A column in a database table that references the primary key of another table, establishing a relationship between the tables.

### Hash Function
A function that maps arbitrary-size data to fixed-size values (hashes). Cryptographic hash functions (SHA-256, SHA-512) provide collision resistance for integrity verification.

### HDF5 (Hierarchical Data Format version 5)
A binary format for storing large numerical datasets with support for hierarchical organisation, compression, and metadata. Common in scientific computing.

### Index (Database)
A data structure that improves query performance by maintaining sorted access paths to table data. Accelerates lookups at the cost of storage and write overhead.

### Integrity Constraint
A rule enforced by a database to maintain data validity. Examples include primary keys, foreign keys, unique constraints, and check constraints.

### JSON (JavaScript Object Notation)
A human-readable text format for structured data, supporting objects (dictionaries), arrays (lists), strings, numbers, booleans, and null values.

### Manifest File
A file containing metadata about a collection of files, typically including file paths, sizes, and checksums. Used for data versioning and integrity verification.

### NetCDF (Network Common Data Form)
A self-describing binary format for array-oriented scientific data, widely used in climate science and meteorology.

### Normalisation
The process of organising database tables to reduce redundancy and dependency. Normal forms (1NF, 2NF, 3NF) represent progressive levels of organisation.

### ORM (Object-Relational Mapping)
A technique that maps database tables to programming language objects, allowing database operations using native language constructs rather than raw SQL.

### Parameterised Query
A SQL query where variable values are passed as parameters rather than concatenated into the query string. Prevents SQL injection attacks.

### Parquet
A columnar binary format optimised for analytical queries. Provides efficient compression and selective column reading.

### Persistence
The characteristic of data that outlives the process that created it. Persistent data survives program termination and system restarts.

### Pickle
Python's native serialisation format, capable of encoding arbitrary Python objects. Fast and flexible but insecure for untrusted data.

### Primary Key
A column or combination of columns that uniquely identifies each row in a database table. Cannot contain NULL values.

### Relational Database
A database organised into tables (relations) with rows (tuples) and columns (attributes), with relationships established through foreign keys.

### Schema
The structure of a database, defining tables, columns, data types, constraints, and relationships.

### Serialisation
The process of converting in-memory data structures into a format suitable for storage or transmission. The inverse of deserialisation.

### SQL (Structured Query Language)
The standard language for interacting with relational databases, supporting data definition (DDL), manipulation (DML), and querying.

### SQL Injection
A security vulnerability where malicious SQL code is inserted through user input. Prevented by using parameterised queries.

### SQLite
A self-contained, serverless relational database engine. The database is stored in a single file, making it ideal for embedded applications.

### Streaming I/O
Processing data incrementally as it is read, rather than loading the entire dataset into memory. Essential for handling files larger than available RAM.

### Transaction
A sequence of database operations that are treated as a single atomic unit. Either all operations succeed or none take effect.

### TSV (Tab-Separated Values)
A variant of CSV using tab characters as delimiters. Useful when data values may contain commas.

### UTF-8
A variable-width character encoding capable of representing all Unicode characters. The most common encoding for text files and web content.

### YAML (YAML Ain't Markup Language)
A human-readable data serialisation format, often used for configuration files. Supports complex data structures with cleaner syntax than JSON.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Unit 10 — Glossary*

© 2025 Antonio Clim. All rights reserved.
