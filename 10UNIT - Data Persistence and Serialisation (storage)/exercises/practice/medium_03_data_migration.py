#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Medium 03 - Data Migration
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Research data often needs to be converted between formats for different tools,
collaborators, or archival requirements. This exercise implements data
migration pipelines between CSV, JSON, and SQLite formats.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Transform data between different serialisation formats
2. Handle schema mapping and type conversion
3. Validate data during migration processes

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 35 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: CSV to JSON Conversion
# ═══════════════════════════════════════════════════════════════════════════════

def csv_to_json(
    csv_path: Path,
    json_path: Path,
    type_converters: dict[str, Callable[[str], Any]] | None = None
) -> int:
    """
    Convert a CSV file to JSON format.

    Args:
        csv_path: Path to source CSV file.
        json_path: Path for output JSON file.
        type_converters: Optional mapping of column names to conversion
                        functions (e.g., {'value': float, 'count': int}).

    Returns:
        Number of records converted.

    Example:
        >>> count = csv_to_json(
        ...     Path('data.csv'),
        ...     Path('data.json'),
        ...     {'temperature': float, 'count': int}
        ... )
    """
    # TODO: Implement this function
    # Read CSV, apply type converters, write as JSON array
    pass


def json_to_csv(
    json_path: Path,
    csv_path: Path,
    fieldnames: list[str] | None = None
) -> int:
    """
    Convert a JSON array to CSV format.

    Args:
        json_path: Path to source JSON file (must be an array of objects).
        csv_path: Path for output CSV file.
        fieldnames: Optional column order (uses first object's keys if None).

    Returns:
        Number of records converted.

    Example:
        >>> count = json_to_csv(Path('data.json'), Path('data.csv'))
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: CSV to SQLite Migration
# ═══════════════════════════════════════════════════════════════════════════════

def infer_sql_type(sample_values: list[str]) -> str:
    """
    Infer SQL type from sample string values.

    Rules:
        - If all values are integers -> INTEGER
        - If all values are numbers -> REAL
        - Otherwise -> TEXT

    Args:
        sample_values: List of string values to analyse.

    Returns:
        SQL type name ('INTEGER', 'REAL', or 'TEXT').

    Example:
        >>> infer_sql_type(['1', '2', '3'])
        'INTEGER'
        >>> infer_sql_type(['1.5', '2.0', '3.7'])
        'REAL'
    """
    # TODO: Implement this function
    pass


def csv_to_sqlite(
    csv_path: Path,
    db_path: Path,
    table_name: str,
    primary_key: str | None = None
) -> int:
    """
    Import a CSV file into a SQLite table.

    Automatically creates the table schema based on CSV headers and
    inferred column types.

    Args:
        csv_path: Path to source CSV file.
        db_path: Path to SQLite database file.
        table_name: Name for the destination table.
        primary_key: Optional column to use as primary key.

    Returns:
        Number of records imported.

    Example:
        >>> count = csv_to_sqlite(
        ...     Path('measurements.csv'),
        ...     Path('research.db'),
        ...     'measurements',
        ...     primary_key='id'
        ... )
    """
    # TODO: Implement this function
    # 1. Read CSV headers
    # 2. Sample data to infer types
    # 3. Create table with inferred schema
    # 4. Insert all records
    pass


def sqlite_to_csv(
    db_path: Path,
    table_name: str,
    csv_path: Path,
    where_clause: str | None = None
) -> int:
    """
    Export a SQLite table to CSV format.

    Args:
        db_path: Path to SQLite database.
        table_name: Name of table to export.
        csv_path: Path for output CSV file.
        where_clause: Optional SQL WHERE clause for filtering.

    Returns:
        Number of records exported.

    Example:
        >>> count = sqlite_to_csv(
        ...     Path('research.db'),
        ...     'measurements',
        ...     Path('export.csv'),
        ...     where_clause="status = 'valid'"
        ... )
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: JSON to SQLite Migration
# ═══════════════════════════════════════════════════════════════════════════════

def json_to_sqlite(
    json_path: Path,
    db_path: Path,
    table_name: str
) -> int:
    """
    Import a JSON array into a SQLite table.

    Flattens nested objects using dot notation for column names.

    Args:
        json_path: Path to source JSON file.
        db_path: Path to SQLite database.
        table_name: Name for the destination table.

    Returns:
        Number of records imported.

    Example:
        >>> # JSON: [{"name": "A", "data": {"value": 1}}]
        >>> # Creates columns: name, data.value
        >>> count = json_to_sqlite(Path('data.json'), Path('db.sqlite'), 'data')
    """
    # TODO: Implement this function
    pass


def sqlite_to_json(
    db_path: Path,
    table_name: str,
    json_path: Path,
    query: str | None = None
) -> int:
    """
    Export SQLite data to JSON format.

    Args:
        db_path: Path to SQLite database.
        table_name: Name of table (ignored if query provided).
        json_path: Path for output JSON file.
        query: Optional custom SELECT query.

    Returns:
        Number of records exported.

    Example:
        >>> count = sqlite_to_json(
        ...     Path('research.db'),
        ...     'measurements',
        ...     Path('export.json')
        ... )
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Migration with Validation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MigrationResult:
    """Results of a data migration operation."""
    total_records: int
    successful_records: int
    failed_records: int
    errors: list[str]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_records == 0:
            return 100.0
        return (self.successful_records / self.total_records) * 100


def validated_csv_to_sqlite(
    csv_path: Path,
    db_path: Path,
    table_name: str,
    validators: dict[str, Callable[[str], bool]]
) -> MigrationResult:
    """
    Migrate CSV to SQLite with validation.

    Records that fail validation are skipped and logged.

    Args:
        csv_path: Path to source CSV file.
        db_path: Path to SQLite database.
        table_name: Name for the destination table.
        validators: Mapping of column names to validation functions.
                   Each function should return True if the value is valid.

    Returns:
        MigrationResult with statistics and error messages.

    Example:
        >>> validators = {
        ...     'temperature': lambda v: -50 < float(v) < 100,
        ...     'humidity': lambda v: 0 <= float(v) <= 100
        ... }
        >>> result = validated_csv_to_sqlite(
        ...     Path('data.csv'),
        ...     Path('clean.db'),
        ...     'measurements',
        ...     validators
        ... )
        >>> print(f"Success rate: {result.success_rate:.1f}%")
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # Create test CSV
        csv_file = test_dir / 'test.csv'
        csv_content = "id,name,value,active\n1,Alice,23.5,true\n2,Bob,24.1,false\n3,Charlie,22.8,true\n"
        csv_file.write_text(csv_content, encoding='utf-8')

        print("Testing Exercise 1: CSV to JSON Conversion")
        json_file = test_dir / 'test.json'
        count = csv_to_json(
            csv_file, json_file,
            type_converters={'id': int, 'value': float}
        )
        assert count == 3, f"Expected 3 records, got {count}"
        with open(json_file, 'r') as f:
            data = json.load(f)
        assert data[0]['id'] == 1, "ID should be integer"
        assert data[0]['value'] == 23.5, "Value should be float"

        # JSON to CSV
        csv_out = test_dir / 'roundtrip.csv'
        count = json_to_csv(json_file, csv_out)
        assert count == 3, "Should convert 3 records"
        print("  ✓ CSV <-> JSON conversion works correctly")

        print("\nTesting Exercise 2: CSV to SQLite Migration")
        db_file = test_dir / 'test.db'
        count = csv_to_sqlite(csv_file, db_file, 'data', primary_key='id')
        assert count == 3, "Should import 3 records"

        # Export back to CSV
        export_csv = test_dir / 'export.csv'
        count = sqlite_to_csv(db_file, 'data', export_csv)
        assert count == 3, "Should export 3 records"
        print("  ✓ CSV <-> SQLite migration works correctly")

        print("\nTesting Exercise 3: JSON to SQLite Migration")
        # Create test JSON with nested structure
        nested_json = test_dir / 'nested.json'
        nested_data = [
            {'id': 1, 'name': 'Test1', 'meta': {'source': 'A'}},
            {'id': 2, 'name': 'Test2', 'meta': {'source': 'B'}}
        ]
        with open(nested_json, 'w') as f:
            json.dump(nested_data, f)

        db_file2 = test_dir / 'test2.db'
        count = json_to_sqlite(nested_json, db_file2, 'nested_data')
        assert count == 2, "Should import 2 records"

        # Export to JSON
        export_json = test_dir / 'export.json'
        count = sqlite_to_json(db_file2, 'nested_data', export_json)
        assert count == 2, "Should export 2 records"
        print("  ✓ JSON <-> SQLite migration works correctly")

        print("\nTesting Exercise 4: Validated Migration")
        # Create CSV with some invalid data
        invalid_csv = test_dir / 'invalid.csv'
        invalid_content = "id,temperature,humidity\n1,25.0,50\n2,150.0,60\n3,22.0,120\n4,20.0,45\n"
        invalid_csv.write_text(invalid_content, encoding='utf-8')

        validators = {
            'temperature': lambda v: -50 < float(v) < 100,
            'humidity': lambda v: 0 <= float(v) <= 100
        }
        db_clean = test_dir / 'clean.db'
        result = validated_csv_to_sqlite(invalid_csv, db_clean, 'data', validators)
        assert result.total_records == 4, "Should process 4 records"
        assert result.successful_records == 2, "Should succeed for 2 valid records"
        assert result.failed_records == 2, "Should fail for 2 invalid records"
        print(f"  ✓ Validated migration: {result.success_rate:.1f}% success rate")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
