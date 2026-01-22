"""
Pytest Configuration and Fixtures for 10UNIT Test Suite
========================================================

This module provides shared fixtures and configuration for testing
data persistence and serialisation functionality.

Fixtures defined here are automatically available to all test modules
without explicit import.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTORY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory for file operation tests.
    
    The directory is automatically cleaned up after each test.
    
    Returns
    -------
    Path
        Path object pointing to a temporary directory.
    
    Example
    -------
    >>> def test_file_creation(tmp_data_dir):
    ...     output_file = tmp_data_dir / "output.txt"
    ...     output_file.write_text("test content")
    ...     assert output_file.exists()
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """
    Provide a temporary path for SQLite database files.
    
    Returns
    -------
    Path
        Path object for a temporary database file.
    """
    return tmp_path / "test_database.db"


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_json_data() -> dict[str, Any]:
    """
    Provide sample JSON-compatible data structure for serialisation tests.
    
    Returns
    -------
    dict
        Nested dictionary structure representing research experiment metadata.
    
    Example
    -------
    >>> def test_json_roundtrip(sample_json_data, tmp_data_dir):
    ...     path = tmp_data_dir / "data.json"
    ...     with open(path, 'w') as f:
    ...         json.dump(sample_json_data, f)
    ...     with open(path, 'r') as f:
    ...         loaded = json.load(f)
    ...     assert loaded == sample_json_data
    """
    return {
        "experiment_id": "EXP-2025-001",
        "title": "Temperature Response Analysis",
        "researcher": {
            "name": "Dr. Jane Smith",
            "institution": "Research Institute",
            "orcid": "0000-0001-2345-6789",
        },
        "parameters": {
            "temperature_range": [15.0, 35.0],
            "humidity": 65.5,
            "sample_count": 100,
        },
        "measurements": [
            {"time": 0, "value": 23.4, "unit": "celsius"},
            {"time": 60, "value": 24.1, "unit": "celsius"},
            {"time": 120, "value": 24.8, "unit": "celsius"},
        ],
        "metadata": {
            "created": "2025-01-15T10:30:00Z",
            "version": "1.0.0",
            "tags": ["temperature", "calibration", "pilot"],
        },
    }


@pytest.fixture
def sample_csv_data() -> list[dict[str, Any]]:
    """
    Provide sample tabular data for CSV serialisation tests.
    
    Returns
    -------
    list[dict]
        List of record dictionaries representing sensor measurements.
    
    Example
    -------
    >>> def test_csv_write(sample_csv_data, tmp_data_dir):
    ...     import csv
    ...     path = tmp_data_dir / "data.csv"
    ...     with open(path, 'w', newline='') as f:
    ...         writer = csv.DictWriter(f, fieldnames=sample_csv_data[0].keys())
    ...         writer.writeheader()
    ...         writer.writerows(sample_csv_data)
    """
    return [
        {
            "sensor_id": "T001",
            "timestamp": "2025-01-15 10:00:00",
            "temperature": 22.5,
            "humidity": 45.2,
            "status": "active",
        },
        {
            "sensor_id": "T002",
            "timestamp": "2025-01-15 10:00:00",
            "temperature": 23.1,
            "humidity": 44.8,
            "status": "active",
        },
        {
            "sensor_id": "T003",
            "timestamp": "2025-01-15 10:00:00",
            "temperature": 21.9,
            "humidity": 46.1,
            "status": "calibrating",
        },
        {
            "sensor_id": "T001",
            "timestamp": "2025-01-15 10:05:00",
            "temperature": 22.7,
            "humidity": 45.0,
            "status": "active",
        },
        {
            "sensor_id": "T002",
            "timestamp": "2025-01-15 10:05:00",
            "temperature": 23.3,
            "humidity": 44.5,
            "status": "active",
        },
    ]


@pytest.fixture
def sample_binary_data() -> bytes:
    """
    Provide sample binary data for binary file operation tests.
    
    Returns
    -------
    bytes
        Binary data representing a simplified research data header.
    """
    # Simulated binary format: magic bytes + version + data count + values
    magic = b"RSRC"  # Research data magic bytes
    version = (1).to_bytes(2, byteorder="little")
    count = (5).to_bytes(4, byteorder="little")
    values = b"".join(
        (v).to_bytes(4, byteorder="little", signed=True)
        for v in [100, 200, 300, 400, 500]
    )
    return magic + version + count + values


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sqlite_connection(tmp_db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Provide a configured SQLite connection for database tests.
    
    The connection is automatically closed after each test.
    
    Yields
    ------
    sqlite3.Connection
        Configured database connection with Row factory.
    
    Example
    -------
    >>> def test_query(sqlite_connection):
    ...     cursor = sqlite_connection.execute("SELECT 1 + 1 AS result")
    ...     row = cursor.fetchone()
    ...     assert row["result"] == 2
    """
    conn = sqlite3.connect(str(tmp_db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    yield conn
    conn.close()


@pytest.fixture
def research_schema(sqlite_connection: sqlite3.Connection) -> sqlite3.Connection:
    """
    Provide a database with pre-created research schema.
    
    Creates tables for experiments, measurements and sensors following
    normalisation principles.
    
    Yields
    ------
    sqlite3.Connection
        Connection to database with research schema.
    """
    sqlite_connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            start_date TEXT NOT NULL,
            end_date TEXT,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'cancelled'))
        );
        
        CREATE TABLE IF NOT EXISTS sensors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_code TEXT NOT NULL UNIQUE,
            sensor_type TEXT NOT NULL,
            location TEXT,
            calibration_date TEXT
        );
        
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            sensor_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            quality_flag TEXT DEFAULT 'valid',
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
            FOREIGN KEY (sensor_id) REFERENCES sensors(id) ON DELETE RESTRICT
        );
        
        CREATE INDEX IF NOT EXISTS idx_measurements_experiment ON measurements(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_measurements_timestamp ON measurements(timestamp);
        """
    )
    sqlite_connection.commit()
    return sqlite_connection


@pytest.fixture
def populated_database(research_schema: sqlite3.Connection) -> sqlite3.Connection:
    """
    Provide a database populated with sample research data.
    
    Yields
    ------
    sqlite3.Connection
        Connection to database with sample data.
    """
    # Insert experiment
    research_schema.execute(
        """
        INSERT INTO experiments (name, description, start_date, status)
        VALUES (?, ?, ?, ?)
        """,
        ("Climate Study 2025", "Temperature monitoring pilot", "2025-01-01", "active"),
    )
    
    # Insert sensors
    sensors = [
        ("T001", "thermometer", "Lab A", "2025-01-01"),
        ("T002", "thermometer", "Lab B", "2025-01-01"),
        ("H001", "hygrometer", "Lab A", "2025-01-01"),
    ]
    research_schema.executemany(
        "INSERT INTO sensors (sensor_code, sensor_type, location, calibration_date) VALUES (?, ?, ?, ?)",
        sensors,
    )
    
    # Insert measurements
    measurements = [
        (1, 1, "2025-01-15 10:00:00", 22.5, "celsius", "valid"),
        (1, 1, "2025-01-15 10:05:00", 22.7, "celsius", "valid"),
        (1, 2, "2025-01-15 10:00:00", 23.1, "celsius", "valid"),
        (1, 3, "2025-01-15 10:00:00", 45.2, "percent", "valid"),
    ]
    research_schema.executemany(
        """
        INSERT INTO measurements (experiment_id, sensor_id, timestamp, value, unit, quality_flag)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        measurements,
    )
    
    research_schema.commit()
    return research_schema


# ═══════════════════════════════════════════════════════════════════════════════
# FILE CONTENT FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def json_file(tmp_data_dir: Path, sample_json_data: dict) -> Path:
    """
    Provide a JSON file containing sample data.
    
    Returns
    -------
    Path
        Path to the created JSON file.
    """
    path = tmp_data_dir / "sample_data.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample_json_data, f, indent=2)
    return path


@pytest.fixture
def csv_file(tmp_data_dir: Path, sample_csv_data: list) -> Path:
    """
    Provide a CSV file containing sample tabular data.
    
    Returns
    -------
    Path
        Path to the created CSV file.
    """
    import csv

    path = tmp_data_dir / "sample_data.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sample_csv_data[0].keys())
        writer.writeheader()
        writer.writerows(sample_csv_data)
    return path


@pytest.fixture
def binary_file(tmp_data_dir: Path, sample_binary_data: bytes) -> Path:
    """
    Provide a binary file containing sample data.
    
    Returns
    -------
    Path
        Path to the created binary file.
    """
    path = tmp_data_dir / "sample_data.bin"
    with open(path, "wb") as f:
        f.write(sample_binary_data)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def large_dataset_generator():
    """
    Provide a generator for creating large datasets for streaming tests.
    
    Returns
    -------
    callable
        Function that generates records up to specified count.
    """

    def generate_records(count: int) -> Generator[dict, None, None]:
        for i in range(count):
            yield {
                "id": i,
                "timestamp": datetime.now().isoformat(),
                "sensor": f"S{i % 100:03d}",
                "value": 20.0 + (i % 100) / 10,
                "status": "active" if i % 10 != 0 else "calibrating",
            }

    return generate_records


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_configure(config):
    """Register custom markers for test categorisation."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow-running")
    config.addinivalue_line("markers", "database: mark test as requiring database")
    config.addinivalue_line("markers", "filesystem: mark test as requiring file system")
