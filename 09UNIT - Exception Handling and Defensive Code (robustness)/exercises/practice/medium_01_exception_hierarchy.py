#!/usr/bin/env python3
"""Exercise 04: Exception Hierarchy Design (Medium).

This exercise develops skills in designing comprehensive exception
hierarchies for domain-specific applications.

Learning Objectives:
    - Design multi-level exception hierarchies
    - Implement exception chaining
    - Create exceptions with rich context information

Estimated Time: 15 minutes
Difficulty: Medium (★★☆)

Instructions:
    1. Complete the exception hierarchy for a data processing library
    2. Implement exception chaining in the processing functions
    3. Run tests to verify your implementation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# =============================================================================
# TASK 1: Design Exception Hierarchy
# =============================================================================

class DataProcessingError(Exception):
    """Base exception for data processing errors.
    
    All domain-specific exceptions should inherit from this class.
    
    Attributes:
        message: Human-readable error description.
        context: Dictionary of additional context information.
    """
    
    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialise with message and optional context.
        
        Args:
            message: Error description.
            context: Additional context for debugging.
        """
        # TODO: Implement __init__
        # - Store message and context attributes
        # - Call super().__init__(message)
        pass
    
    def __str__(self) -> str:
        """Return formatted error message with context."""
        # TODO: Return message, and if context exists, append it
        # Format: "{message} (context: {context})" or just "{message}"
        pass


class FileProcessingError(DataProcessingError):
    """Exception for file-related processing errors.
    
    Attributes:
        file_path: Path to the problematic file.
        operation: Operation that failed (read, write, parse, etc.).
    """
    
    def __init__(
        self,
        message: str,
        file_path: Path | str,
        operation: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Initialise file processing error.
        
        Args:
            message: Error description.
            file_path: Path to the file.
            operation: Failed operation name.
            context: Additional context.
        """
        # TODO: Implement __init__
        # - Store file_path (as Path) and operation
        # - Add file_path and operation to context
        # - Call super().__init__()
        pass


class ParseError(FileProcessingError):
    """Exception for parsing errors in data files.
    
    Attributes:
        line_number: Line where error occurred (1-indexed).
        column: Column position if available.
        raw_content: The problematic content.
    """
    
    def __init__(
        self,
        message: str,
        file_path: Path | str,
        line_number: int,
        column: int | None = None,
        raw_content: str | None = None,
    ) -> None:
        """Initialise parse error.
        
        Args:
            message: Error description.
            file_path: Path to the file.
            line_number: Line number (1-indexed).
            column: Column position.
            raw_content: The content that failed to parse.
        """
        # TODO: Implement __init__
        # - Store line_number, column, raw_content
        # - Build context with location info
        # - Call super().__init__() with operation="parse"
        pass
    
    @property
    def location(self) -> str:
        """Return formatted location string."""
        # TODO: Return "line {line_number}" or "line {line_number}, column {column}"
        pass


class ValidationError(DataProcessingError):
    """Exception for data validation failures.
    
    Attributes:
        field_name: Name of the invalid field.
        invalid_value: The value that failed validation.
        constraint: Description of the violated constraint.
    """
    
    def __init__(
        self,
        message: str,
        field_name: str,
        invalid_value: Any,
        constraint: str,
    ) -> None:
        """Initialise validation error.
        
        Args:
            message: Error description.
            field_name: Name of the field.
            invalid_value: The invalid value.
            constraint: The constraint that was violated.
        """
        # TODO: Implement __init__
        pass


class TransformationError(DataProcessingError):
    """Exception for data transformation failures.
    
    Attributes:
        source_type: Original data type.
        target_type: Intended result type.
        transformation: Name of the transformation.
    """
    
    def __init__(
        self,
        message: str,
        source_type: str,
        target_type: str,
        transformation: str,
    ) -> None:
        """Initialise transformation error.
        
        Args:
            message: Error description.
            source_type: Type of source data.
            target_type: Intended type of result.
            transformation: Name of transformation.
        """
        # TODO: Implement __init__
        pass


# =============================================================================
# TASK 2: Implement Exception Chaining
# =============================================================================

def read_data_file(file_path: Path) -> str:
    """Read content from a data file.
    
    Args:
        file_path: Path to the file.
        
    Returns:
        File content as string.
        
    Raises:
        FileProcessingError: If file cannot be read, chaining original exception.
        
    Example:
        >>> read_data_file(Path("nonexistent.txt"))
        Traceback (most recent call last):
            ...
        FileProcessingError: Cannot read file: nonexistent.txt ...
    """
    # TODO: Implement with exception chaining
    # - Try to read file
    # - Catch FileNotFoundError and PermissionError
    # - Raise FileProcessingError with 'from' to chain original exception
    pass


def parse_csv_line(line: str, line_number: int, file_path: Path) -> list[str]:
    """Parse a single CSV line.
    
    Simple CSV parsing (no quoted fields with commas).
    
    Args:
        line: Line to parse.
        line_number: Line number for error reporting.
        file_path: File path for error reporting.
        
    Returns:
        List of field values.
        
    Raises:
        ParseError: If line cannot be parsed.
    """
    # TODO: Implement CSV line parsing
    # - Split by comma
    # - If line is empty, raise ParseError
    # - Return list of stripped fields
    pass


def validate_record(
    record: dict[str, str],
    required_fields: set[str],
    line_number: int,
) -> None:
    """Validate that a record contains required fields.
    
    Args:
        record: Dictionary of field names to values.
        required_fields: Fields that must be present and non-empty.
        line_number: Line number for error context.
        
    Raises:
        ValidationError: If validation fails.
    """
    # TODO: Implement validation
    # - Check each required field exists
    # - Check each required field is not empty
    # - Raise ValidationError with appropriate details
    pass


def process_data_file(
    file_path: Path,
    required_fields: set[str],
) -> list[dict[str, str]]:
    """Process a CSV data file with full error handling.
    
    Reads the file, parses each line, validates records, and
    returns a list of validated records.
    
    Args:
        file_path: Path to CSV file.
        required_fields: Required field names.
        
    Returns:
        List of validated record dictionaries.
        
    Raises:
        FileProcessingError: If file cannot be read.
        ParseError: If parsing fails.
        ValidationError: If validation fails.
        
    Example:
        File content:
        name,age,email
        Alice,30,alice@example.com
        
        >>> records = process_data_file(Path("data.csv"), {"name", "email"})
        >>> records[0]["name"]
        'Alice'
    """
    # TODO: Implement complete processing pipeline
    # 1. Read file using read_data_file
    # 2. Split into lines
    # 3. Extract header from first line
    # 4. Parse each data line
    # 5. Create record dict from header and values
    # 6. Validate each record
    # 7. Return list of valid records
    pass


# =============================================================================
# TEST YOUR IMPLEMENTATIONS
# =============================================================================


def main() -> None:
    """Test exercise implementations."""
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
        raise ParseError("Invalid format", "data.csv", 5, column=10, raw_content="bad")
    except ParseError as e:
        assert e.line_number == 5
        assert e.column == 10
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
    
    print("\nTesting processing functions...")
    
    # Test read_data_file with nonexistent file
    try:
        read_data_file(Path("nonexistent_file.csv"))
        assert False, "Should have raised FileProcessingError"
    except FileProcessingError as e:
        assert e.__cause__ is not None  # Should have chained exception
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
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
