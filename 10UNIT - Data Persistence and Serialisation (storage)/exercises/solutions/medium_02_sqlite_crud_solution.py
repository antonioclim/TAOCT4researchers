#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Medium 02 - SQLite CRUD Operations
═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class Measurement:
    """Measurement record."""
    sensor_id: str
    timestamp: str
    value: float
    unit: str
    measurement_id: int | None = None


def create_measurements_table(conn: sqlite3.Connection) -> None:
    """Create measurements table with proper schema."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            measurement_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_sensor 
        ON measurements(sensor_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_measurements_timestamp 
        ON measurements(timestamp)
    """)
    conn.commit()


def insert_measurement(conn: sqlite3.Connection, measurement: Measurement) -> int:
    """Insert a measurement and return its ID."""
    cursor = conn.execute(
        """
        INSERT INTO measurements (sensor_id, timestamp, value, unit)
        VALUES (?, ?, ?, ?)
        """,
        (measurement.sensor_id, measurement.timestamp, 
         measurement.value, measurement.unit)
    )
    conn.commit()
    return cursor.lastrowid


def get_measurement_by_id(
    conn: sqlite3.Connection, 
    measurement_id: int
) -> Measurement | None:
    """Retrieve a measurement by ID."""
    cursor = conn.execute(
        "SELECT * FROM measurements WHERE measurement_id = ?",
        (measurement_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    
    return Measurement(
        measurement_id=row[0],
        sensor_id=row[1],
        timestamp=row[2],
        value=row[3],
        unit=row[4]
    )


def update_measurement_value(
    conn: sqlite3.Connection,
    measurement_id: int,
    new_value: float
) -> bool:
    """Update a measurement's value."""
    cursor = conn.execute(
        "UPDATE measurements SET value = ? WHERE measurement_id = ?",
        (new_value, measurement_id)
    )
    conn.commit()
    return cursor.rowcount > 0


def delete_measurement(conn: sqlite3.Connection, measurement_id: int) -> bool:
    """Delete a measurement by ID."""
    cursor = conn.execute(
        "DELETE FROM measurements WHERE measurement_id = ?",
        (measurement_id,)
    )
    conn.commit()
    return cursor.rowcount > 0


def get_measurements_by_sensor(
    conn: sqlite3.Connection,
    sensor_id: str,
    limit: int = 100
) -> list[Measurement]:
    """Get all measurements for a sensor."""
    cursor = conn.execute(
        """
        SELECT * FROM measurements 
        WHERE sensor_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (sensor_id, limit)
    )
    
    return [
        Measurement(
            measurement_id=row[0],
            sensor_id=row[1],
            timestamp=row[2],
            value=row[3],
            unit=row[4]
        )
        for row in cursor.fetchall()
    ]


def get_sensor_statistics(
    conn: sqlite3.Connection,
    sensor_id: str
) -> dict[str, Any]:
    """Get statistics for a sensor."""
    cursor = conn.execute(
        """
        SELECT 
            COUNT(*) as count,
            MIN(value) as min_value,
            MAX(value) as max_value,
            AVG(value) as avg_value,
            MIN(timestamp) as first_reading,
            MAX(timestamp) as last_reading
        FROM measurements
        WHERE sensor_id = ?
        """,
        (sensor_id,)
    )
    row = cursor.fetchone()
    
    return {
        'count': row[0],
        'min_value': row[1],
        'max_value': row[2],
        'avg_value': row[3],
        'first_reading': row[4],
        'last_reading': row[5]
    }


def bulk_insert_measurements(
    conn: sqlite3.Connection,
    measurements: list[Measurement]
) -> int:
    """Insert multiple measurements in a single transaction."""
    if not measurements:
        return 0
    
    data = [
        (m.sensor_id, m.timestamp, m.value, m.unit)
        for m in measurements
    ]
    
    conn.executemany(
        """
        INSERT INTO measurements (sensor_id, timestamp, value, unit)
        VALUES (?, ?, ?, ?)
        """,
        data
    )
    conn.commit()
    return len(measurements)


def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / 'test.db'
        conn = sqlite3.connect(db_path)

        print("Testing create_measurements_table")
        create_measurements_table(conn)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert 'measurements' in tables
        print("  ✓ Passed")

        print("Testing insert_measurement")
        m = Measurement(
            sensor_id='T001',
            timestamp='2024-01-15T10:00:00',
            value=23.5,
            unit='celsius'
        )
        m_id = insert_measurement(conn, m)
        assert m_id == 1
        print("  ✓ Passed")

        print("Testing get_measurement_by_id")
        loaded = get_measurement_by_id(conn, m_id)
        assert loaded is not None
        assert loaded.value == 23.5
        print("  ✓ Passed")

        print("Testing update_measurement_value")
        updated = update_measurement_value(conn, m_id, 24.0)
        assert updated
        loaded = get_measurement_by_id(conn, m_id)
        assert loaded.value == 24.0
        print("  ✓ Passed")

        print("Testing bulk_insert_measurements")
        measurements = [
            Measurement(sensor_id='T001', timestamp=f'2024-01-15T1{i}:00:00',
                       value=20.0 + i, unit='celsius')
            for i in range(5)
        ]
        count = bulk_insert_measurements(conn, measurements)
        assert count == 5
        print("  ✓ Passed")

        print("Testing get_measurements_by_sensor")
        results = get_measurements_by_sensor(conn, 'T001')
        assert len(results) == 6  # 1 original + 5 bulk
        print("  ✓ Passed")

        print("Testing get_sensor_statistics")
        stats = get_sensor_statistics(conn, 'T001')
        assert stats['count'] == 6
        print("  ✓ Passed")

        print("Testing delete_measurement")
        deleted = delete_measurement(conn, m_id)
        assert deleted
        assert get_measurement_by_id(conn, m_id) is None
        print("  ✓ Passed")

        conn.close()
        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
