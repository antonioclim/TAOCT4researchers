#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
10UNIT, Lab 2: Database Fundamentals
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Database systems epitomize this pattern, maintaining indices (In practice
sorted or structured access paths) to accelerate queries at the cost of
increased storage and update overhead."
— Garcia-Molina et al., 2008, p. 145

Relational databases provide structured, queryable storage for research data.
This laboratory introduces SQLite—a self-contained database engine embedded
in Python's standard library—demonstrating schema design, CRUD operations,
advanced queries, and data integrity mechanisms.

PREREQUISITES
─────────────
- Lab 10.1: File I/O and Serialisation
- Basic understanding of SQL syntax
- Familiarity with relational data concepts (tables, rows, columns)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Design normalised database schemas with appropriate constraints
2. Execute parameterised queries safely, avoiding SQL injection
3. Implement transactions for atomic multi-statement operations
4. Use aggregation and JOIN queries for data analysis
5. Create data versioning systems with checksums and manifests

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 45 minutes
- Total: 75 minutes

DEPENDENCIES
────────────
- Python 3.12+
- Standard library: sqlite3, hashlib, json, dataclasses

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Iterator

# Configure module-level logger
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SQLITE BASICS AND CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@contextmanager
def database_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for safe database connection handling.

    Ensures connections are properly closed even if exceptions occur.
    Enables foreign key enforcement by default.

    Args:
        db_path: Path to SQLite database file.

    Yields:
        Active database connection.

    Example:
        >>> with database_connection(Path('research.db')) as conn:
        ...     cursor = conn.execute("SELECT * FROM experiments")
    """
    logger.debug("Opening database connection: %s", db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    conn.execute("PRAGMA foreign_keys = ON")  # Enforce referential integrity

    try:
        yield conn
    finally:
        conn.close()
        logger.debug("Database connection closed")


def create_research_schema(conn: sqlite3.Connection) -> None:
    """
    Create the research database schema.

    Implements a normalised schema with three related tables:
    - experiments: Experimental metadata
    - sensors: Measurement instruments
    - measurements: Individual data points

    Args:
        conn: Active database connection.

    Schema Design Rationale:
        The schema follows Third Normal Form (3NF) to eliminate redundancy:
        - Each fact is stored exactly once
        - Foreign keys establish relationships between entities
        - Indices accelerate common query patterns
    """
    logger.info("Creating research database schema")

    conn.executescript("""
        -- Experiments table: stores experimental metadata
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            researcher TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'cancelled')),
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Sensors table: measurement instruments
        CREATE TABLE IF NOT EXISTS sensors (
            sensor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            serial_number TEXT NOT NULL UNIQUE,
            sensor_type TEXT NOT NULL,
            location TEXT,
            calibration_date TEXT,
            accuracy REAL,
            unit TEXT NOT NULL
        );

        -- Measurements table: individual data points
        CREATE TABLE IF NOT EXISTS measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            sensor_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            value REAL NOT NULL,
            quality_flag TEXT DEFAULT 'valid' CHECK(quality_flag IN ('valid', 'suspect', 'invalid')),
            notes TEXT,
            FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE,
            FOREIGN KEY (sensor_id) REFERENCES sensors(sensor_id)
        );

        -- Indices for common query patterns
        CREATE INDEX IF NOT EXISTS idx_measurements_experiment 
            ON measurements(experiment_id);
        CREATE INDEX IF NOT EXISTS idx_measurements_timestamp 
            ON measurements(timestamp);
        CREATE INDEX IF NOT EXISTS idx_measurements_sensor 
            ON measurements(sensor_id);
        
        -- Composite index for time-range queries within experiments
        CREATE INDEX IF NOT EXISTS idx_measurements_exp_time 
            ON measurements(experiment_id, timestamp);
    """)

    conn.commit()
    logger.info("Schema created successfully")


@dataclass
class Experiment:
    """Data class representing an experiment record."""

    name: str
    researcher: str
    start_date: str
    description: str = ''
    end_date: str | None = None
    status: str = 'active'
    experiment_id: int | None = None


@dataclass
class Sensor:
    """Data class representing a sensor record."""

    serial_number: str
    sensor_type: str
    unit: str
    location: str = ''
    calibration_date: str | None = None
    accuracy: float | None = None
    sensor_id: int | None = None


@dataclass
class Measurement:
    """Data class representing a measurement record."""

    experiment_id: int
    sensor_id: int
    timestamp: str
    value: float
    quality_flag: str = 'valid'
    notes: str = ''
    measurement_id: int | None = None


def insert_experiment(conn: sqlite3.Connection, experiment: Experiment) -> int:
    """
    Insert new experiment record.

    Uses parameterised queries to prevent SQL injection attacks.

    Args:
        conn: Database connection.
        experiment: Experiment data to insert.

    Returns:
        ID of the newly inserted experiment.

    Example:
        >>> exp = Experiment(name='Climate Study', researcher='Dr. Smith',
        ...                  start_date='2024-01-15')
        >>> exp_id = insert_experiment(conn, exp)
    """
    logger.debug("Inserting experiment: %s", experiment.name)

    cursor = conn.execute(
        """
        INSERT INTO experiments (name, description, researcher, start_date, end_date, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            experiment.name,
            experiment.description,
            experiment.researcher,
            experiment.start_date,
            experiment.end_date,
            experiment.status,
        )
    )
    conn.commit()

    experiment_id = cursor.lastrowid
    logger.debug("Inserted experiment with ID: %d", experiment_id)
    return experiment_id


def insert_sensor(conn: sqlite3.Connection, sensor: Sensor) -> int:
    """
    Insert new sensor record.

    Args:
        conn: Database connection.
        sensor: Sensor data to insert.

    Returns:
        ID of the newly inserted sensor.
    """
    logger.debug("Inserting sensor: %s", sensor.serial_number)

    cursor = conn.execute(
        """
        INSERT INTO sensors (serial_number, sensor_type, location, calibration_date, accuracy, unit)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            sensor.serial_number,
            sensor.sensor_type,
            sensor.location,
            sensor.calibration_date,
            sensor.accuracy,
            sensor.unit,
        )
    )
    conn.commit()

    sensor_id = cursor.lastrowid
    logger.debug("Inserted sensor with ID: %d", sensor_id)
    return sensor_id


def bulk_insert_measurements(
    conn: sqlite3.Connection,
    measurements: list[Measurement]
) -> int:
    """
    Insert multiple measurements in a single transaction.

    Bulk operations within a transaction are dramatically faster than
    individual inserts, as they avoid per-statement commit overhead.

    Args:
        conn: Database connection.
        measurements: List of measurements to insert.

    Returns:
        Number of records inserted.

    Example:
        >>> measurements = [Measurement(...) for _ in range(1000)]
        >>> count = bulk_insert_measurements(conn, measurements)
    """
    if not measurements:
        return 0

    logger.debug("Bulk inserting %d measurements", len(measurements))

    # Prepare data tuples
    data = [
        (
            m.experiment_id,
            m.sensor_id,
            m.timestamp,
            m.value,
            m.quality_flag,
            m.notes,
        )
        for m in measurements
    ]

    conn.executemany(
        """
        INSERT INTO measurements (experiment_id, sensor_id, timestamp, value, quality_flag, notes)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        data
    )
    conn.commit()

    logger.debug("Bulk insert completed: %d records", len(measurements))
    return len(measurements)


def get_experiment_by_id(
    conn: sqlite3.Connection,
    experiment_id: int
) -> Experiment | None:
    """
    Retrieve experiment by ID.

    Args:
        conn: Database connection.
        experiment_id: ID of experiment to retrieve.

    Returns:
        Experiment object or None if not found.
    """
    cursor = conn.execute(
        "SELECT * FROM experiments WHERE experiment_id = ?",
        (experiment_id,)
    )
    row = cursor.fetchone()

    if row is None:
        return None

    return Experiment(
        experiment_id=row['experiment_id'],
        name=row['name'],
        description=row['description'] or '',
        researcher=row['researcher'],
        start_date=row['start_date'],
        end_date=row['end_date'],
        status=row['status'],
    )


def update_experiment_status(
    conn: sqlite3.Connection,
    experiment_id: int,
    status: str,
    end_date: str | None = None
) -> bool:
    """
    Update experiment status.

    Args:
        conn: Database connection.
        experiment_id: ID of experiment to update.
        status: New status value.
        end_date: Optional end date for completed experiments.

    Returns:
        True if update succeeded, False if experiment not found.
    """
    logger.debug("Updating experiment %d status to %s", experiment_id, status)

    cursor = conn.execute(
        """
        UPDATE experiments 
        SET status = ?, end_date = ?
        WHERE experiment_id = ?
        """,
        (status, end_date, experiment_id)
    )
    conn.commit()

    return cursor.rowcount > 0


def delete_experiment(conn: sqlite3.Connection, experiment_id: int) -> bool:
    """
    Delete experiment and all associated measurements.

    Uses ON DELETE CASCADE to automatically remove related measurements.

    Args:
        conn: Database connection.
        experiment_id: ID of experiment to delete.

    Returns:
        True if deletion succeeded.
    """
    logger.debug("Deleting experiment: %d", experiment_id)

    cursor = conn.execute(
        "DELETE FROM experiments WHERE experiment_id = ?",
        (experiment_id,)
    )
    conn.commit()

    return cursor.rowcount > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SCHEMA DESIGN AND NORMALISATION
# ═══════════════════════════════════════════════════════════════════════════════


def get_table_info(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    """
    Retrieve schema information for a table.

    Args:
        conn: Database connection.
        table_name: Name of table to inspect.

    Returns:
        List of column information dictionaries.
    """
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    columns = []

    for row in cursor.fetchall():
        columns.append({
            'column_id': row[0],
            'name': row[1],
            'type': row[2],
            'not_null': bool(row[3]),
            'default_value': row[4],
            'primary_key': bool(row[5]),
        })

    return columns


def get_foreign_keys(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    """
    Retrieve foreign key constraints for a table.

    Args:
        conn: Database connection.
        table_name: Name of table to inspect.

    Returns:
        List of foreign key information dictionaries.
    """
    cursor = conn.execute(f"PRAGMA foreign_key_list({table_name})")
    foreign_keys = []

    for row in cursor.fetchall():
        foreign_keys.append({
            'id': row[0],
            'seq': row[1],
            'table': row[2],
            'from': row[3],
            'to': row[4],
            'on_update': row[5],
            'on_delete': row[6],
        })

    return foreign_keys


def get_indices(conn: sqlite3.Connection, table_name: str) -> list[dict[str, Any]]:
    """
    Retrieve index information for a table.

    Args:
        conn: Database connection.
        table_name: Name of table to inspect.

    Returns:
        List of index information dictionaries.
    """
    cursor = conn.execute(f"PRAGMA index_list({table_name})")
    indices = []

    for row in cursor.fetchall():
        index_name = row[1]
        # Get columns in this index
        col_cursor = conn.execute(f"PRAGMA index_info({index_name})")
        columns = [col[2] for col in col_cursor.fetchall()]

        indices.append({
            'name': index_name,
            'unique': bool(row[2]),
            'columns': columns,
        })

    return indices


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ADVANCED QUERIES
# ═══════════════════════════════════════════════════════════════════════════════


def query_measurements_by_time_range(
    conn: sqlite3.Connection,
    experiment_id: int,
    start_time: str,
    end_time: str
) -> list[dict[str, Any]]:
    """
    Retrieve measurements within a time range.

    Args:
        conn: Database connection.
        experiment_id: Experiment to query.
        start_time: Start of time range (ISO format).
        end_time: End of time range (ISO format).

    Returns:
        List of measurement dictionaries.
    """
    cursor = conn.execute(
        """
        SELECT m.*, s.serial_number, s.sensor_type, s.unit
        FROM measurements m
        JOIN sensors s ON m.sensor_id = s.sensor_id
        WHERE m.experiment_id = ?
          AND m.timestamp >= ?
          AND m.timestamp <= ?
        ORDER BY m.timestamp
        """,
        (experiment_id, start_time, end_time)
    )

    return [dict(row) for row in cursor.fetchall()]


def compute_statistics_by_sensor(
    conn: sqlite3.Connection,
    experiment_id: int
) -> list[dict[str, Any]]:
    """
    Compute aggregate statistics for each sensor in an experiment.

    Uses SQL aggregation functions for efficient computation.

    Args:
        conn: Database connection.
        experiment_id: Experiment to analyse.

    Returns:
        List of statistics dictionaries per sensor.
    """
    cursor = conn.execute(
        """
        SELECT 
            s.serial_number,
            s.sensor_type,
            s.unit,
            COUNT(m.measurement_id) as measurement_count,
            MIN(m.value) as min_value,
            MAX(m.value) as max_value,
            AVG(m.value) as mean_value,
            SUM(CASE WHEN m.quality_flag = 'valid' THEN 1 ELSE 0 END) as valid_count
        FROM sensors s
        LEFT JOIN measurements m ON s.sensor_id = m.sensor_id
            AND m.experiment_id = ?
        GROUP BY s.sensor_id
        HAVING measurement_count > 0
        ORDER BY s.serial_number
        """,
        (experiment_id,)
    )

    return [dict(row) for row in cursor.fetchall()]


def find_anomalous_measurements(
    conn: sqlite3.Connection,
    experiment_id: int,
    std_threshold: float = 3.0
) -> list[dict[str, Any]]:
    """
    Find measurements that deviate significantly from the mean.

    Uses a Common Table Expression (CTE) to compute statistics before
    filtering anomalous values.

    Args:
        conn: Database connection.
        experiment_id: Experiment to analyse.
        std_threshold: Number of standard deviations for anomaly detection.

    Returns:
        List of anomalous measurement dictionaries.
    """
    cursor = conn.execute(
        """
        WITH sensor_stats AS (
            SELECT 
                sensor_id,
                AVG(value) as mean_val,
                -- SQLite doesn't have STDDEV, so we compute variance manually
                AVG(value * value) - AVG(value) * AVG(value) as variance
            FROM measurements
            WHERE experiment_id = ?
              AND quality_flag = 'valid'
            GROUP BY sensor_id
        )
        SELECT 
            m.measurement_id,
            m.timestamp,
            m.value,
            s.serial_number,
            ss.mean_val,
            SQRT(ss.variance) as std_dev,
            ABS(m.value - ss.mean_val) / SQRT(ss.variance) as z_score
        FROM measurements m
        JOIN sensors s ON m.sensor_id = s.sensor_id
        JOIN sensor_stats ss ON m.sensor_id = ss.sensor_id
        WHERE m.experiment_id = ?
          AND ss.variance > 0
          AND ABS(m.value - ss.mean_val) / SQRT(ss.variance) > ?
        ORDER BY z_score DESC
        """,
        (experiment_id, experiment_id, std_threshold)
    )

    return [dict(row) for row in cursor.fetchall()]


def generate_daily_summary(
    conn: sqlite3.Connection,
    experiment_id: int
) -> list[dict[str, Any]]:
    """
    Generate daily summary statistics for an experiment.

    Aggregates measurements by date, providing daily overviews.

    Args:
        conn: Database connection.
        experiment_id: Experiment to summarise.

    Returns:
        List of daily summary dictionaries.
    """
    cursor = conn.execute(
        """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as measurement_count,
            COUNT(DISTINCT sensor_id) as active_sensors,
            AVG(value) as mean_value,
            MIN(value) as min_value,
            MAX(value) as max_value,
            SUM(CASE WHEN quality_flag = 'valid' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as valid_percentage
        FROM measurements
        WHERE experiment_id = ?
        GROUP BY DATE(timestamp)
        ORDER BY date
        """,
        (experiment_id,)
    )

    return [dict(row) for row in cursor.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DATA VERSIONING AND INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataManifest:
    """
    Manifest record for data versioning.

    Captures metadata about the state of research data at a specific point
    in time, enabling integrity verification and change tracking.
    """

    manifest_id: str
    created_at: str
    experiment_id: int
    record_count: int
    checksum: str
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_data_checksum(
    conn: sqlite3.Connection,
    experiment_id: int
) -> str:
    """
    Compute checksum of all measurements for an experiment.

    Creates a deterministic hash of all measurement data, enabling
    detection of any modifications to the dataset.

    Args:
        conn: Database connection.
        experiment_id: Experiment to checksum.

    Returns:
        SHA-256 checksum of measurement data.
    """
    cursor = conn.execute(
        """
        SELECT sensor_id, timestamp, value, quality_flag
        FROM measurements
        WHERE experiment_id = ?
        ORDER BY timestamp, sensor_id
        """,
        (experiment_id,)
    )

    hasher = hashlib.sha256()

    for row in cursor:
        # Create deterministic string representation
        row_str = f"{row[0]}|{row[1]}|{row[2]:.10f}|{row[3]}"
        hasher.update(row_str.encode('utf-8'))

    return hasher.hexdigest()


def create_manifest(
    conn: sqlite3.Connection,
    experiment_id: int,
    additional_metadata: dict[str, Any] | None = None
) -> DataManifest:
    """
    Create a manifest for current experiment data state.

    Args:
        conn: Database connection.
        experiment_id: Experiment to manifest.
        additional_metadata: Optional extra metadata to include.

    Returns:
        DataManifest capturing current data state.
    """
    # Count records
    cursor = conn.execute(
        "SELECT COUNT(*) FROM measurements WHERE experiment_id = ?",
        (experiment_id,)
    )
    record_count = cursor.fetchone()[0]

    # Compute checksum
    checksum = compute_data_checksum(conn, experiment_id)

    # Generate manifest ID
    timestamp = datetime.now().isoformat()
    manifest_id = f"MAN-{experiment_id}-{timestamp[:10]}-{checksum[:8]}"

    manifest = DataManifest(
        manifest_id=manifest_id,
        created_at=timestamp,
        experiment_id=experiment_id,
        record_count=record_count,
        checksum=checksum,
        metadata=additional_metadata or {},
    )

    logger.info("Created manifest %s: %d records, checksum %s...",
                manifest_id, record_count, checksum[:16])

    return manifest


def verify_data_integrity(
    conn: sqlite3.Connection,
    manifest: DataManifest
) -> tuple[bool, str]:
    """
    Verify data integrity against a saved manifest.

    Args:
        conn: Database connection.
        manifest: Previously saved manifest.

    Returns:
        Tuple of (is_valid, message).
    """
    # Count current records
    cursor = conn.execute(
        "SELECT COUNT(*) FROM measurements WHERE experiment_id = ?",
        (manifest.experiment_id,)
    )
    current_count = cursor.fetchone()[0]

    if current_count != manifest.record_count:
        return False, f"Record count mismatch: {current_count} vs {manifest.record_count}"

    # Compute current checksum
    current_checksum = compute_data_checksum(conn, manifest.experiment_id)

    if current_checksum != manifest.checksum:
        return False, f"Checksum mismatch: data has been modified"

    return True, "Data integrity verified successfully"


def save_manifest_to_file(manifest: DataManifest, filepath: Path) -> None:
    """
    Save manifest to JSON file.

    Args:
        manifest: Manifest to save.
        filepath: Destination file path.
    """
    data = {
        'manifest_id': manifest.manifest_id,
        'created_at': manifest.created_at,
        'experiment_id': manifest.experiment_id,
        'record_count': manifest.record_count,
        'checksum': manifest.checksum,
        'metadata': manifest.metadata,
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    logger.debug("Manifest saved to %s", filepath)


def load_manifest_from_file(filepath: Path) -> DataManifest:
    """
    Load manifest from JSON file.

    Args:
        filepath: Path to manifest file.

    Returns:
        Loaded DataManifest.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return DataManifest(
        manifest_id=data['manifest_id'],
        created_at=data['created_at'],
        experiment_id=data['experiment_id'],
        record_count=data['record_count'],
        checksum=data['checksum'],
        metadata=data.get('metadata', {}),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    Demonstrate database fundamentals.

    Creates a sample research database, populates it with test data,
    and demonstrates query and integrity verification operations.
    """
    import random
    import tempfile

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    logger.info("Starting Database Fundamentals Lab Demonstration")

    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'research.db'

        with database_connection(db_path) as conn:
            # Create schema
            create_research_schema(conn)

            # Insert sample experiment
            experiment = Experiment(
                name='Climate Monitoring Study',
                researcher='Dr. Jane Smith',
                start_date='2024-01-15',
                description='Long-term temperature and humidity monitoring',
            )
            exp_id = insert_experiment(conn, experiment)
            logger.info("Created experiment with ID: %d", exp_id)

            # Insert sample sensors
            sensors = [
                Sensor(serial_number='TEMP-001', sensor_type='temperature',
                       unit='celsius', location='Lab A', accuracy=0.1),
                Sensor(serial_number='HUMID-001', sensor_type='humidity',
                       unit='percent', location='Lab A', accuracy=1.0),
                Sensor(serial_number='TEMP-002', sensor_type='temperature',
                       unit='celsius', location='Lab B', accuracy=0.1),
            ]

            sensor_ids = [insert_sensor(conn, s) for s in sensors]
            logger.info("Created %d sensors", len(sensor_ids))

            # Generate sample measurements
            measurements = []
            base_time = datetime(2024, 1, 15, 0, 0, 0)

            for hour in range(48):  # 48 hours of data
                timestamp = base_time.replace(hour=hour % 24, day=15 + hour // 24)

                for sensor_id in sensor_ids:
                    value = 20.0 + random.gauss(0, 2)
                    measurements.append(Measurement(
                        experiment_id=exp_id,
                        sensor_id=sensor_id,
                        timestamp=timestamp.isoformat(),
                        value=round(value, 2),
                    ))

            count = bulk_insert_measurements(conn, measurements)
            logger.info("Inserted %d measurements", count)

            # Demonstrate queries
            logger.info("\n--- Query Demonstrations ---")

            # Statistics by sensor
            stats = compute_statistics_by_sensor(conn, exp_id)
            logger.info("Statistics by sensor:")
            for s in stats:
                logger.info("  %s: count=%d, mean=%.2f, range=[%.2f, %.2f]",
                            s['serial_number'], s['measurement_count'],
                            s['mean_value'], s['min_value'], s['max_value'])

            # Daily summary
            daily = generate_daily_summary(conn, exp_id)
            logger.info("\nDaily summary:")
            for d in daily:
                logger.info("  %s: %d measurements, mean=%.2f, valid=%.1f%%",
                            d['date'], d['measurement_count'],
                            d['mean_value'], d['valid_percentage'])

            # Create and verify manifest
            logger.info("\n--- Data Versioning ---")
            manifest = create_manifest(conn, exp_id)

            is_valid, message = verify_data_integrity(conn, manifest)
            logger.info("Integrity check: %s - %s", is_valid, message)

            # Save manifest
            manifest_path = Path(temp_dir) / 'manifest.json'
            save_manifest_to_file(manifest, manifest_path)
            logger.info("Manifest saved to %s", manifest_path)

    logger.info("\nDemonstration completed successfully")


if __name__ == '__main__':
    main()
