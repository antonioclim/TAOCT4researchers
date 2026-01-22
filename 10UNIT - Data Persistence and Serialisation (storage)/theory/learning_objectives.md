# 10UNIT — Learning Objectives

## Data Persistence and Serialisation

This document specifies the measurable learning objectives for Unit 10, organised by cognitive level according to educational taxonomy principles. Each objective is accompanied by assessment criteria and evidence requirements.

---

## Objective Hierarchy

### Level 1: Remember (Knowledge Recall)

**LO-1.1**: List the fundamental file modes available in Python (`'r'`, `'w'`, `'a'`, `'x'`, `'b'`, `'+'`) and state their operational behaviours.

- **Assessment**: Quiz questions requiring mode identification
- **Evidence**: Correctly match mode specifiers to operations (read, write, append, exclusive create)
- **Threshold**: 100% accuracy on mode definitions

**LO-1.2**: Recall the ACID properties (Atomicity, Consistency, Isolation, Durability) and their significance for database transactions.

- **Assessment**: Short-answer questions defining each property
- **Evidence**: Provide accurate definitions with research-relevant examples
- **Threshold**: All four properties correctly defined

**LO-1.3**: Identify common serialisation formats (JSON, CSV, Pickle, Parquet, HDF5) and their primary characteristics.

- **Assessment**: Format identification exercises
- **Evidence**: Match formats to appropriate use cases
- **Threshold**: 90% accuracy on format characteristics

---

### Level 2: Understand (Comprehension)

**LO-2.1**: Explain the trade-offs between human-readable formats (JSON, CSV) and binary formats (Pickle, Parquet) for research data storage.

- **Assessment**: Written comparison essay (300-500 words)
- **Evidence**: Articulate performance, interoperability and security considerations
- **Threshold**: Address at least three distinct trade-off dimensions

**LO-2.2**: Describe how database normalisation eliminates data redundancy and prevents update anomalies through progressive refinement (1NF, 2NF, 3NF).

- **Assessment**: Schema analysis exercises
- **Evidence**: Identify normalisation violations and explain remediation
- **Threshold**: Correctly diagnose normalisation level of given schemas

**LO-2.3**: Interpret the relationship between encoding specifications and cross-platform data portability.

- **Assessment**: Encoding troubleshooting scenarios
- **Evidence**: Diagnose encoding-related failures and propose solutions
- **Threshold**: Resolve encoding issues in provided case studies

---

### Level 3: Apply (Implementation)

**LO-3.1**: Implement file operations using context managers and `pathlib` for reliable, cross-platform resource management.

- **Assessment**: Lab exercise completion (lab_10_01)
- **Evidence**: Working code demonstrating proper resource handling
- **Threshold**: All file operations use context managers; no resource leaks

**LO-3.2**: Construct CRUD (Create, Read, Update, Delete) operations with SQLite using parameterised queries exclusively.

- **Assessment**: Lab exercise completion (lab_10_02)
- **Evidence**: Database operations without SQL injection vulnerabilities
- **Threshold**: Zero string concatenation in SQL queries

**LO-3.3**: Apply JSON and CSV modules to serialise and deserialise complex nested data structures.

- **Assessment**: Practice exercises (easy_01 through easy_03)
- **Evidence**: Correct round-trip serialisation preserving data integrity
- **Threshold**: Data equivalence before and after serialisation

**LO-3.4**: Implement streaming file processing for datasets exceeding available memory.

- **Assessment**: Hard exercise completion (hard_01)
- **Evidence**: Memory-efficient processing of multi-gigabyte files
- **Threshold**: Constant memory usage regardless of file size

---

### Level 4: Analyse (Evaluation and Selection)

**LO-4.1**: Analyse data storage requirements to select appropriate serialisation strategies based on access patterns, performance needs and interoperability constraints.

- **Assessment**: Format selection case studies
- **Evidence**: Justified recommendations with quantitative reasoning
- **Threshold**: Correct format selection for 4/5 scenarios with valid justification

**LO-4.2**: Evaluate database schema designs against normalisation principles and identify functional dependencies.

- **Assessment**: Schema review exercises
- **Evidence**: Detect redundancy and propose normalised alternatives
- **Threshold**: Identify all normalisation violations in provided schemas

**LO-4.3**: Compare transaction isolation levels and their implications for concurrent data access in research applications.

- **Assessment**: Written analysis of isolation trade-offs
- **Evidence**: Explain phantom reads, dirty reads and lost updates
- **Threshold**: Accurate description of at least three isolation phenomena

---

### Level 5: Evaluate (Judgement)

**LO-5.1**: Assess when relational database solutions (SQLite) outperform flat-file storage for research data management.

- **Assessment**: Decision framework application
- **Evidence**: Apply systematic criteria to storage technology selection
- **Threshold**: Correct recommendation in 4/5 scenarios with reasoned justification

**LO-5.2**: Critique data storage implementations for security vulnerabilities (SQL injection, Pickle deserialisation attacks).

- **Assessment**: Code review exercises
- **Evidence**: Identify vulnerabilities and propose mitigations
- **Threshold**: Detect all critical vulnerabilities in provided code samples

---

### Level 6: Create (Synthesis)

**LO-6.1**: Design data versioning systems using cryptographic checksums and manifest files to ensure research reproducibility.

- **Assessment**: Hard exercise completion (hard_03)
- **Evidence**: Functional versioning system with integrity verification
- **Threshold**: System detects all data modifications reliably

**LO-6.2**: Construct database schemas satisfying Third Normal Form for multi-entity research domains.

- **Assessment**: Schema design project (hard_02)
- **Evidence**: ERD and SQL DDL statements demonstrating normalisation
- **Threshold**: Schema passes 3NF validation with no redundancy

**LO-6.3**: Develop data migration utilities transforming between storage formats whilst preserving semantic integrity.

- **Assessment**: Medium exercise completion (medium_03)
- **Evidence**: Bidirectional format conversion with data validation
- **Threshold**: Zero data loss during migration operations

---

## Mapping to Assessment Components

| Objective | Quiz | Lab 1 | Lab 2 | Easy | Medium | Hard | Homework |
|-----------|------|-------|-------|------|--------|------|----------|
| LO-1.1 | ✓ | | | ✓ | | | |
| LO-1.2 | ✓ | | ✓ | | | | |
| LO-1.3 | ✓ | ✓ | | | | | |
| LO-2.1 | ✓ | | | | | | ✓ |
| LO-2.2 | ✓ | | ✓ | | ✓ | | |
| LO-2.3 | | ✓ | | ✓ | | | |
| LO-3.1 | | ✓ | | ✓ | | | |
| LO-3.2 | | | ✓ | | ✓ | | |
| LO-3.3 | | ✓ | | ✓ | | | |
| LO-3.4 | | | | | | ✓ | |
| LO-4.1 | | | | | | | ✓ |
| LO-4.2 | | | ✓ | | | ✓ | |
| LO-4.3 | ✓ | | ✓ | | | | |
| LO-5.1 | | | | | | | ✓ |
| LO-5.2 | ✓ | | | | | | |
| LO-6.1 | | | | | | ✓ | |
| LO-6.2 | | | | | | ✓ | |
| LO-6.3 | | | | | ✓ | | |

---

## Prerequisite Mapping

This unit builds upon competencies from previous units:

| Prerequisite Objective | Source Unit | Application in 10UNIT |
|------------------------|-------------|----------------------|
| Exception handling patterns | 09UNIT | I/O error recovery, transaction rollback |
| Dictionary operations | 06UNIT | JSON structure manipulation |
| Function design principles | 05UNIT | File processing utilities |
| Context manager comprehension | 09UNIT | Resource management |

---

## Forward References

Competencies developed in this unit support subsequent learning:

| Current Objective | Target Unit | Application |
|-------------------|-------------|-------------|
| LO-3.3 (Serialisation) | 11UNIT | Text data storage and retrieval |
| LO-3.2 (SQL operations) | 12UNIT | API response caching |
| LO-6.1 (Versioning) | 07UNIT | Reproducibility infrastructure |

---

## Verification Methods

### Automated Assessment

- **Unit tests**: All lab and exercise solutions verified against test suites
- **Linting**: Code quality verified through ruff and mypy
- **Coverage**: Minimum 80% test coverage for submitted solutions

### Manual Assessment

- **Code review**: Instructor evaluation of design decisions
- **Written responses**: Assessment of conceptual understanding
- **Peer review**: Collaborative evaluation following established rubrics

---

© 2025 Antonio Clim. All rights reserved.
