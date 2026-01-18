#!/usr/bin/env python3
"""Solution for Exercise 04: Exception Hierarchy Design.

This module provides reference implementations for designing
comprehensive exception hierarchies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    
    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            return f"{self.message} (context: {self.context})"
        return self.message


class FileProcessingError(DataProcessingError):
    """Exception for file-related processing errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Path | str,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.operation = operation
        
        full_context = context or {}
        full_context["file_path"] = str(self.file_path)
        full_context["operation"] = operation
        
        super().__init__(message, full_context)


class ParseError(FileProcessingError):
    """Exception for parsing errors in data files."""
    
    def __init__(
        self,
        message: str,
        file_path: Path | str,
        line_number: int,
        column: int | None = None,
        raw_content: str | None = None,
    ) -> None:
        self.line_number = line_number
        self.column = column
        self.raw_content = raw_content
        
        context = {
            "line_number": line_number,
            "column": column,
            "raw_content": raw_content,
        }
        
        super().__init__(message, file_path, "parse", context)
    
    @property
    def location(self) -> str:
        if self.column is not None:
            return f"line {self.line_number}, column {self.column}"
        return f"line {self.line_number}"


class ValidationError(DataProcessingError):
    """Exception for data validation failures."""
    
    def __init__(
        self,
        message: str,
        field_name: str,
        invalid_value: Any,
        constraint: str,
    ) -> None:
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.constraint = constraint
        
        context = {
            "field_name": field_name,
            "invalid_value": invalid_value,
            "constraint": constraint,
        }
        
        super().__init__(message, context)


class TransformationError(DataProcessingError):
    """Exception for data transformation failures."""
    
    def __init__(
        self,
        message: str,
        source_type: str,
        target_type: str,
        transformation: str,
    ) -> None:
        self.source_type = source_type
        self.target_type = target_type
        self.transformation = transformation
        
        context = {
            "source_type": source_type,
            "target_type": target_type,
            "transformation": transformation,
        }
        
        super().__init__(message, context)


def read_data_file(file_path: Path) -> str:
    """Read content from a data file with exception chaining."""
    try:
        return file_path.read_text()
    except FileNotFoundError as e:
        raise FileProcessingError(
            f"Cannot read file: {file_path}",
            file_path,
            "read",
        ) from e
    except PermissionError as e:
        raise FileProcessingError(
            f"Permission denied reading: {file_path}",
            file_path,
            "read",
        ) from e


def parse_csv_line(line: str, line_number: int, file_path: Path) -> list[str]:
    """Parse a single CSV line."""
    if not line.strip():
        raise ParseError(
            "Empty line",
            file_path,
            line_number,
        )
    
    return [field.strip() for field in line.split(",")]


def validate_record(
    record: dict[str, str],
    required_fields: set[str],
    line_number: int,
) -> None:
    """Validate that a record contains required fields."""
    for field in required_fields:
        if field not in record:
            raise ValidationError(
                f"Missing required field: {field}",
                field,
                None,
                f"field must be present",
            )
        if not record[field].strip():
            raise ValidationError(
                f"Empty required field: {field}",
                field,
                record[field],
                "field must not be empty",
            )


def process_data_file(
    file_path: Path,
    required_fields: set[str],
) -> list[dict[str, str]]:
    """Process a CSV data file with full error handling."""
    content = read_data_file(file_path)
    lines = content.strip().split("\n")
    
    if not lines:
        raise ParseError("Empty file", file_path, 1)
    
    # Parse header
    header = parse_csv_line(lines[0], 1, file_path)
    
    records = []
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        
        values = parse_csv_line(line, i, file_path)
        
        if len(values) != len(header):
            raise ParseError(
                f"Column count mismatch: expected {len(header)}, got {len(values)}",
                file_path,
                i,
            )
        
        record = dict(zip(header, values))
        validate_record(record, required_fields, i)
        records.append(record)
    
    return records


def main() -> None:
    """Verify solution implementations."""
    import tempfile
    
    print("Testing exception hierarchy...")
    
    # Test base exception
    try:
        raise DataProcessingError("Test error", {"key": "value"})
    except DataProcessingError as e:
        assert e.message == "Test error"
        assert e.context["key"] == "value"
        assert "context" in str(e)
    print("  ✓ DataProcessingError passed")
    
    # Test FileProcessingError
    try:
        raise FileProcessingError("Cannot read", "data.csv", "read")
    except FileProcessingError as e:
        assert e.file_path == Path("data.csv")
        assert e.operation == "read"
    print("  ✓ FileProcessingError passed")
    
    # Test ParseError
    try:
        raise ParseError("Invalid format", "data.csv", 5, column=10)
    except ParseError as e:
        assert e.line_number == 5
        assert "line 5" in e.location
        assert "column 10" in e.location
    print("  ✓ ParseError passed")
    
    # Test ValidationError
    try:
        raise ValidationError("Out of range", "age", 200, "must be 0-150")
    except ValidationError as e:
        assert e.field_name == "age"
        assert e.invalid_value == 200
    print("  ✓ ValidationError passed")
    
    # Test read_data_file with nonexistent file
    try:
        read_data_file(Path("nonexistent_file.csv"))
        assert False
    except FileProcessingError as e:
        assert e.__cause__ is not None
    print("  ✓ read_data_file exception chaining passed")
    
    # Test with actual file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,age,email\n")
        f.write("Alice,30,alice@example.com\n")
        f.write("Bob,25,bob@example.com\n")
        temp_path = Path(f.name)
    
    try:
        records = process_data_file(temp_path, {"name", "email"})
        assert len(records) == 2
        assert records[0]["name"] == "Alice"
        assert records[1]["email"] == "bob@example.com"
        print("  ✓ process_data_file passed")
    finally:
        temp_path.unlink()
    
    print("\n✅ All solutions verified!")


if __name__ == "__main__":
    main()
