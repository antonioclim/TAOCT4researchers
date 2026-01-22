"""
10UNIT — Data Persistence and Serialisation Test Suite
=======================================================

This package contains comprehensive test modules verifying the correctness,
reliability and performance characteristics of the laboratory implementations.

Test Modules
------------
test_lab_10_01
    Validates file I/O operations, serialisation formats and encoding handling.
    Covers: text/binary modes, JSON/CSV/Pickle operations, streaming I/O,
    atomic writes and checksum verification.

test_lab_10_02
    Validates database operations, schema design and transaction integrity.
    Covers: CRUD operations, parameterised queries, normalisation validation,
    ACID properties and concurrent access patterns.

Test Execution
--------------
Execute the full test suite with coverage reporting:

    pytest tests/ --cov=lab --cov-report=term-missing

Execute specific test modules:

    pytest tests/test_lab_10_01.py -v
    pytest tests/test_lab_10_02.py -v

Generate HTML coverage report:

    pytest tests/ --cov=lab --cov-report=html

Test Categories
---------------
Tests are organised by functionality and marked for selective execution:

    @pytest.mark.unit       - Fast, isolated unit tests
    @pytest.mark.integration - Tests requiring file system or database
    @pytest.mark.slow        - Performance and stress tests

Fixtures
--------
Common test fixtures are defined in conftest.py:

    tmp_data_dir     - Temporary directory for file operations
    sample_json_data - Dictionary structure for serialisation tests
    sample_csv_data  - List of records for tabular data tests
    sqlite_connection - Pre-configured SQLite connection
    research_schema  - Standard research database schema

Author
------
Antonio Clim <antonio.clim@ie.ase.ro>

Licence
-------
© 2025 Antonio Clim. All rights reserved.
"""

__version__ = "1.0.0"
