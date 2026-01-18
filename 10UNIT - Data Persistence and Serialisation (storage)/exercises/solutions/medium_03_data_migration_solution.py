#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Medium 03 - Data Migration
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MigrationRecord:
    """Record for migration tracking."""
    id: str
    timestamp: str
    sensor: str
    value: float
    unit: str


def csv_to_json(csv_path: Path, json_path: Path) -> int:
    """Convert CSV file to JSON format."""
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    
    return len(records)


def json_to_csv(json_path: Path, csv_path: Path) -> int:
    """Convert JSON file to CSV format."""
    with open(json_path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    if not records:
        return 0
    
    fieldnames = list(records[0].keys())
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    return len(records)


def csv_to_sqlite(
    csv_path: Path,
    db_path: Path,
    table_name: str
) -> int:
    """Import CSV data into SQLite table."""
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        records = list(reader)
    
    if not records:
        return 0
    
    conn = sqlite3.connect(db_path)
    columns = list(records[0].keys())
    
    # Create table
    column_defs = ', '.join(f'"{col}" TEXT' for col in columns)
    conn.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({column_defs})')
    
    # Insert data
    placeholders = ', '.join('?' for _ in columns)
    column_names = ', '.join(f'"{col}"' for col in columns)
    
    for record in records:
        values = [record[col] for col in columns]
        conn.execute(
            f'INSERT INTO "{table_name}" ({column_names}) VALUES ({placeholders})',
            values
        )
    
    conn.commit()
    conn.close()
    return len(records)


def sqlite_to_csv(
    db_path: Path,
    table_name: str,
    csv_path: Path
) -> int:
    """Export SQLite table to CSV format."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.execute(f'SELECT * FROM "{table_name}"')
    rows = cursor.fetchall()
    
    if not rows:
        conn.close()
        return 0
    
    columns = rows[0].keys()
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    
    conn.close()
    return len(rows)


def transform_records(
    records: list[dict[str, Any]],
    transformations: dict[str, callable]
) -> list[dict[str, Any]]:
    """Apply transformations to record fields."""
    result = []
    
    for record in records:
        transformed = {}
        for key, value in record.items():
            if key in transformations:
                transformed[key] = transformations[key](value)
            else:
                transformed[key] = value
        result.append(transformed)
    
    return result


def validate_migration(
    source_count: int,
    dest_count: int,
    source_checksum: str | None = None,
    dest_checksum: str | None = None
) -> tuple[bool, str]:
    """Validate migration by comparing counts and optional checksums."""
    if source_count != dest_count:
        return False, f"Count mismatch: {source_count} vs {dest_count}"
    
    if source_checksum and dest_checksum:
        if source_checksum != dest_checksum:
            return False, "Checksum mismatch"
    
    return True, "Migration validated successfully"


def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # Create test CSV
        csv_path = test_dir / 'data.csv'
        csv_path.write_text(
            'id,name,value\n1,sensor_a,23.5\n2,sensor_b,24.1\n',
            encoding='utf-8'
        )

        print("Testing csv_to_json")
        json_path = test_dir / 'data.json'
        count = csv_to_json(csv_path, json_path)
        assert count == 2
        assert json_path.exists()
        print("  ✓ Passed")

        print("Testing json_to_csv")
        csv_out = test_dir / 'data_out.csv'
        count = json_to_csv(json_path, csv_out)
        assert count == 2
        print("  ✓ Passed")

        print("Testing csv_to_sqlite")
        db_path = test_dir / 'data.db'
        count = csv_to_sqlite(csv_path, db_path, 'measurements')
        assert count == 2
        print("  ✓ Passed")

        print("Testing sqlite_to_csv")
        csv_export = test_dir / 'exported.csv'
        count = sqlite_to_csv(db_path, 'measurements', csv_export)
        assert count == 2
        print("  ✓ Passed")

        print("Testing transform_records")
        records = [{'value': '23.5'}, {'value': '24.1'}]
        transformed = transform_records(records, {'value': float})
        assert transformed[0]['value'] == 23.5
        assert isinstance(transformed[0]['value'], float)
        print("  ✓ Passed")

        print("Testing validate_migration")
        valid, msg = validate_migration(2, 2)
        assert valid
        invalid, msg = validate_migration(2, 3)
        assert not invalid
        print("  ✓ Passed")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
