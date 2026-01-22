# Unit 10: Self-Assessment Checklist

## Pre-Lab Readiness Check

Before starting the lab exercises, verify you can answer these questions:

### File I/O Fundamentals

- [ ] What is the difference between `'r'`, `'w'`, `'a'`, and `'x'` file modes?
- [ ] Why must you always specify `encoding='utf-8'` for text files?
- [ ] What happens if you forget to close a file handle?
- [ ] How do context managers (`with` statement) help with file operations?
- [ ] When should you use binary mode (`'rb'`, `'wb'`)?

### Serialisation Formats

- [ ] What data types can JSON represent natively?
- [ ] Why does CSV lose type information?
- [ ] What is the security risk with Pickle?
- [ ] When would you choose Parquet over CSV?
- [ ] How do you handle datetime objects in JSON?

### Database Concepts

- [ ] What does ACID stand for and why does it matter?
- [ ] What is SQL injection and how do you prevent it?
- [ ] When should you use a database instead of flat files?
- [ ] What is the purpose of database normalisation?
- [ ] How do foreign keys maintain referential integrity?

---

## Lab Completion Checklist

### Lab 10.1: File I/O and Serialisation

#### Section 1: File Operations
- [ ] Implemented `read_text_file()` with proper encoding
- [ ] Implemented `write_text_file()` with append support
- [ ] Implemented `read_lines_generator()` for streaming
- [ ] Implemented `atomic_write_text()` for safe writes
- [ ] Implemented `compute_file_checksum()` with chunked reading

#### Section 2: JSON Serialisation
- [ ] Created custom JSON encoder for datetime and Path
- [ ] Implemented `save_json()` and `load_json()` with custom hooks
- [ ] Added schema validation function

#### Section 3: CSV Processing
- [ ] Implemented CSV reading as dictionaries
- [ ] Implemented CSV writing from dictionaries
- [ ] Created streaming CSV reader
- [ ] Added type conversion utility

#### Section 4: Binary Formats
- [ ] Implemented Pickle save/load (with security awareness)
- [ ] Implemented gzip compression utilities

#### Section 5: Benchmarking
- [ ] Created format comparison benchmark
- [ ] Analysed trade-offs between formats

### Lab 10.2: Database Fundamentals

#### Section 1: SQLite Basics
- [ ] Created database connection context manager
- [ ] Implemented schema creation with proper constraints
- [ ] Added appropriate indices

#### Section 2: CRUD Operations
- [ ] Implemented insert functions with parameterised queries
- [ ] Implemented read functions returning dataclass instances
- [ ] Implemented update functions
- [ ] Implemented delete functions
- [ ] Added bulk insert with transaction

#### Section 3: Advanced Queries
- [ ] Implemented time range query
- [ ] Implemented aggregation with GROUP BY
- [ ] Implemented anomaly detection query
- [ ] Created daily summary report

#### Section 4: Data Integrity
- [ ] Implemented checksum computation for database tables
- [ ] Created manifest generation
- [ ] Implemented integrity verification

---

## Knowledge Verification

### Conceptual Understanding

Rate your confidence (1-5) on each topic:

| Topic | Confidence | Notes |
|-------|------------|-------|
| File modes and when to use each | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Encoding and UTF-8 importance | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Context managers for resources | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| JSON limitations and workarounds | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| CSV processing patterns | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Pickle security concerns | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Streaming vs loading all data | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| SQL parameterised queries | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Database normalisation basics | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| ACID properties | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Cryptographic checksums | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |
| Data versioning concepts | ☐ 1 ☐ 2 ☐ 3 ☐ 4 ☐ 5 | |

### Practical Skills

Can you perform these tasks without reference material?

- [ ] Open a file safely with proper encoding
- [ ] Read a CSV file into a list of dictionaries
- [ ] Write JSON with custom datetime handling
- [ ] Connect to SQLite and create a table
- [ ] Execute a parameterised SQL query
- [ ] Compute SHA-256 checksum of a file
- [ ] Create a data manifest with checksums

---

## Common Mistakes to Avoid

### File Operations
- ❌ Forgetting to specify encoding
- ❌ Not using context managers
- ❌ Using string concatenation for paths
- ❌ Ignoring file not found errors

### Serialisation
- ❌ Assuming JSON handles all Python types
- ❌ Unpickling data from untrusted sources
- ❌ Loading entire large files into memory

### Database
- ❌ Using string formatting in SQL queries
- ❌ Forgetting to commit transactions
- ❌ Not creating indices for frequent queries
- ❌ Ignoring foreign key constraints

---

## Ready for Homework?

Before starting the homework assignment, ensure:

- [ ] All lab exercises completed successfully
- [ ] Confidence rating ≥3 on all topics
- [ ] Can explain each common mistake
- [ ] Reviewed the cheatsheet
- [ ] Understand the rubric criteria

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Unit 10 — Self-Assessment*

© 2025 Antonio Clim. All rights reserved.
