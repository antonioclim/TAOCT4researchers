#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Easy 03 - CSV Processing
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
CSV (Comma-Separated Values) remains the most common format for tabular
research data. This exercise introduces Python's csv module for reading
and writing structured tabular data.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Read CSV files as lists of dictionaries using DictReader
2. Write dictionaries to CSV files using DictWriter
3. Handle CSV files with custom delimiters

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import csv
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Read CSV as Dictionaries
# ═══════════════════════════════════════════════════════════════════════════════

def read_csv_as_dicts(filepath: Path) -> list[dict[str, str]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.

    Uses the first row as column headers. All values are returned as strings.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of dictionaries, one per row.

    Example:
        >>> # Given measurements.csv:
        >>> # sensor,value,unit
        >>> # T001,23.5,celsius
        >>> read_csv_as_dicts(Path('measurements.csv'))
        [{'sensor': 'T001', 'value': '23.5', 'unit': 'celsius'}]
    """
    # TODO: Implement this function
    # Hint: Use csv.DictReader and remember newline='' parameter
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Write Dictionaries to CSV
# ═══════════════════════════════════════════════════════════════════════════════

def write_dicts_to_csv(
    data: list[dict[str, Any]],
    filepath: Path,
    fieldnames: list[str] | None = None
) -> int:
    """
    Write a list of dictionaries to a CSV file.

    Args:
        data: List of dictionaries to write.
        filepath: Destination file path.
        fieldnames: Column order (uses first dict's keys if None).

    Returns:
        Number of rows written (excluding header).

    Example:
        >>> data = [{'name': 'A', 'value': 1}, {'name': 'B', 'value': 2}]
        >>> write_dicts_to_csv(data, Path('output.csv'))
        2
    """
    # TODO: Implement this function
    # Hint: Use csv.DictWriter with writeheader() and writerows()
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Filter CSV Rows
# ═══════════════════════════════════════════════════════════════════════════════

def filter_csv_by_column(
    filepath: Path,
    column: str,
    value: str
) -> list[dict[str, str]]:
    """
    Read CSV and return only rows where the specified column matches the value.

    Args:
        filepath: Path to CSV file.
        column: Column name to filter on.
        value: Value to match.

    Returns:
        List of matching row dictionaries.

    Example:
        >>> # Given data with 'status' column
        >>> filter_csv_by_column(Path('data.csv'), 'status', 'active')
        [{'id': '1', 'status': 'active'}, ...]
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: CSV with Custom Delimiter
# ═══════════════════════════════════════════════════════════════════════════════

def read_tsv_file(filepath: Path) -> list[dict[str, str]]:
    """
    Read a tab-separated values (TSV) file.

    Args:
        filepath: Path to TSV file.

    Returns:
        List of row dictionaries.

    Example:
        >>> read_tsv_file(Path('data.tsv'))
    """
    # TODO: Implement this function
    # Hint: Use delimiter='\t' parameter
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Convert CSV Types
# ═══════════════════════════════════════════════════════════════════════════════

def read_csv_with_types(
    filepath: Path,
    type_map: dict[str, type]
) -> list[dict[str, Any]]:
    """
    Read CSV file and convert specified columns to given types.

    Args:
        filepath: Path to CSV file.
        type_map: Mapping of column names to type conversion functions.
                  e.g., {'value': float, 'count': int}

    Returns:
        List of dictionaries with converted values.

    Example:
        >>> type_map = {'temperature': float, 'count': int}
        >>> data = read_csv_with_types(Path('data.csv'), type_map)
        >>> data[0]['temperature']  # Returns float, not string
        23.5
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
        csv_content = "sensor,value,status\nT001,23.5,active\nT002,24.1,active\nT003,22.8,inactive\n"
        csv_file.write_text(csv_content, encoding='utf-8')

        print("Testing Exercise 1: read_csv_as_dicts")
        rows = read_csv_as_dicts(csv_file)
        assert len(rows) == 3, f"Expected 3 rows, got {len(rows)}"
        assert rows[0]['sensor'] == 'T001'
        assert rows[0]['value'] == '23.5'
        print("  ✓ read_csv_as_dicts works correctly")

        print("\nTesting Exercise 2: write_dicts_to_csv")
        output_file = test_dir / 'output.csv'
        data = [{'id': '1', 'name': 'Alice'}, {'id': '2', 'name': 'Bob'}]
        count = write_dicts_to_csv(data, output_file, ['id', 'name'])
        assert count == 2, f"Expected 2 rows written, got {count}"
        content = output_file.read_text(encoding='utf-8')
        assert 'id,name' in content, "Header missing"
        print("  ✓ write_dicts_to_csv works correctly")

        print("\nTesting Exercise 3: filter_csv_by_column")
        active = filter_csv_by_column(csv_file, 'status', 'active')
        assert len(active) == 2, "Should find 2 active sensors"
        assert all(r['status'] == 'active' for r in active)
        print("  ✓ filter_csv_by_column works correctly")

        print("\nTesting Exercise 4: read_tsv_file")
        tsv_file = test_dir / 'test.tsv'
        tsv_file.write_text("col1\tcol2\nval1\tval2\n", encoding='utf-8')
        tsv_rows = read_tsv_file(tsv_file)
        assert len(tsv_rows) == 1
        assert tsv_rows[0]['col1'] == 'val1'
        print("  ✓ read_tsv_file works correctly")

        print("\nTesting Exercise 5: read_csv_with_types")
        typed_rows = read_csv_with_types(csv_file, {'value': float})
        assert isinstance(typed_rows[0]['value'], float)
        assert typed_rows[0]['value'] == 23.5
        print("  ✓ read_csv_with_types works correctly")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
