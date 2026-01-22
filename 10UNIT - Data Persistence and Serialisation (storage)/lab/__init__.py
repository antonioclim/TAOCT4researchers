"""
10UNIT — Data Persistence and Serialisation Laboratory Package
================================================================

This package provides comprehensive laboratory modules for exploring data
persistence concepts, serialisation formats and database fundamentals in
research computing contexts.

Modules
-------
lab_10_01_file_io_serialisation
    File input/output operations, encoding management, serialisation formats
    (JSON, CSV, Pickle), streaming I/O and atomic write patterns.

lab_10_02_database_fundamentals
    SQLite database operations, CRUD patterns, schema design, normalisation
    principles and transaction management.

Technology Stack
----------------
- Python 3.12+
- Standard library modules: json, csv, pickle, sqlite3, pathlib, hashlib
- External packages: pandas, pyarrow (optional for Parquet support)

Usage Example
-------------
>>> from lab import lab_10_01_file_io_serialisation as file_io
>>> from lab import lab_10_02_database_fundamentals as db
>>>
>>> # File operations
>>> data = file_io.read_json_file('config.json')
>>> file_io.write_csv_file('output.csv', records)
>>>
>>> # Database operations
>>> with db.DatabaseConnection('research.db') as conn:
...     results = db.execute_query(conn, "SELECT * FROM experiments")

Research Applications
---------------------
- Climate science: NetCDF/HDF5 for multidimensional data
- Genomics: FASTA/FASTQ sequence file processing
- Social science: Survey data in relational databases
- Computational experiments: Checkpoint/recovery mechanisms

Author
------
Antonio Clim <antonio.clim@ie.ase.ro>

Licence
-------
© 2025 Antonio Clim. All rights reserved.
This material is provided for educational purposes as part of
"The Art of Computational Thinking for Researchers" course at ASE-CSIE.
"""

__version__ = "1.0.0"
__author__ = "Antonio Clim"
__all__ = [
    "lab_10_01_file_io_serialisation",
    "lab_10_02_database_fundamentals",
]
