#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Easy 03 - CSV Processing
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
from pathlib import Path
from typing import Any


def read_csv_as_dicts(filepath: Path) -> list[dict[str, str]]:
    """Read a CSV file and return its contents as a list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_dicts_to_csv(
    data: list[dict[str, Any]],
    filepath: Path,
    fieldnames: list[str] | None = None
) -> int:
    """Write a list of dictionaries to a CSV file."""
    if not data:
        return 0
    
    if fieldnames is None:
        fieldnames = list(data[0].keys())
    
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return len(data)


def filter_csv_by_column(
    filepath: Path,
    column: str,
    value: str
) -> list[dict[str, str]]:
    """Read CSV and return only rows where the specified column matches the value."""
    rows = read_csv_as_dicts(filepath)
    return [row for row in rows if row.get(column) == value]


def read_tsv_file(filepath: Path) -> list[dict[str, str]]:
    """Read a tab-separated values (TSV) file."""
    with open(filepath, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return [dict(row) for row in reader]


def read_csv_with_types(
    filepath: Path,
    type_map: dict[str, type]
) -> list[dict[str, Any]]:
    """Read CSV file and convert specified columns to given types."""
    rows = read_csv_as_dicts(filepath)
    result: list[dict[str, Any]] = []
    
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in type_map:
                try:
                    converted[key] = type_map[key](value)
                except (ValueError, TypeError):
                    converted[key] = value
            else:
                converted[key] = value
        result.append(converted)
    
    return result


def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        csv_file = test_dir / 'test.csv'
        csv_content = "sensor,value,status\nT001,23.5,active\nT002,24.1,active\nT003,22.8,inactive\n"
        csv_file.write_text(csv_content, encoding='utf-8')

        print("Testing read_csv_as_dicts")
        rows = read_csv_as_dicts(csv_file)
        assert len(rows) == 3
        assert rows[0]['sensor'] == 'T001'
        print("  ✓ Passed")

        print("Testing write_dicts_to_csv")
        output_file = test_dir / 'output.csv'
        data = [{'id': '1', 'name': 'Alice'}, {'id': '2', 'name': 'Bob'}]
        count = write_dicts_to_csv(data, output_file, ['id', 'name'])
        assert count == 2
        print("  ✓ Passed")

        print("Testing filter_csv_by_column")
        active = filter_csv_by_column(csv_file, 'status', 'active')
        assert len(active) == 2
        print("  ✓ Passed")

        print("Testing read_tsv_file")
        tsv_file = test_dir / 'test.tsv'
        tsv_file.write_text("col1\tcol2\nval1\tval2\n", encoding='utf-8')
        tsv_rows = read_tsv_file(tsv_file)
        assert tsv_rows[0]['col1'] == 'val1'
        print("  ✓ Passed")

        print("Testing read_csv_with_types")
        typed_rows = read_csv_with_types(csv_file, {'value': float})
        assert isinstance(typed_rows[0]['value'], float)
        assert typed_rows[0]['value'] == 23.5
        print("  ✓ Passed")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
