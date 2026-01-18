#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Unit 10: Tests for Lab 10.01 - File I/O and Serialisation
═══════════════════════════════════════════════════════════════════════════════

Test suite providing ≥80% coverage for lab_10_01_file_io_serialisation.py

Run with:
    pytest tests/test_lab_10_01.py -v --cov=lab --cov-report=term-missing

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lab'))

from lab_10_01_file_io_serialisation import (
    read_text_file,
    write_text_file,
    read_lines_generator,
    read_binary_file,
    write_binary_file,
    atomic_write_text,
    compute_file_checksum,
    ResearchJSONEncoder,
    research_json_decoder,
    save_json,
    load_json,
    validate_json_schema,
    MeasurementRecord,
    read_csv_as_dicts,
    write_csv_from_dicts,
    stream_csv_records,
    convert_csv_types,
    save_pickle,
    load_pickle,
    save_compressed,
    load_compressed,
    generate_sample_records,
    compare_formats,
    FormatBenchmark,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file for testing."""
    filepath = temp_dir / 'sample.txt'
    content = "Line 1: Hello\nLine 2: World\nLine 3: Test\n"
    filepath.write_text(content, encoding='utf-8')
    return filepath, content


@pytest.fixture
def sample_csv_file(temp_dir):
    """Create a sample CSV file for testing."""
    filepath = temp_dir / 'sample.csv'
    content = "id,name,value\n1,alpha,10.5\n2,beta,20.3\n3,gamma,30.1\n"
    filepath.write_text(content, encoding='utf-8')
    return filepath


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT FILE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTextFileOperations:
    """Tests for text file read/write operations."""

    def test_read_text_file_success(self, sample_text_file):
        """Test reading an existing text file."""
        filepath, expected_content = sample_text_file
        content = read_text_file(filepath)
        assert content == expected_content

    def test_read_text_file_not_found(self, temp_dir):
        """Test reading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_text_file(temp_dir / 'nonexistent.txt')

    def test_write_text_file_creates_file(self, temp_dir):
        """Test writing creates a new file."""
        filepath = temp_dir / 'output.txt'
        content = "Test content"
        chars = write_text_file(filepath, content)
        
        assert filepath.exists()
        assert chars == len(content)
        assert filepath.read_text(encoding='utf-8') == content

    def test_write_text_file_overwrites(self, temp_dir):
        """Test writing overwrites existing content."""
        filepath = temp_dir / 'output.txt'
        write_text_file(filepath, "Original")
        write_text_file(filepath, "New")
        
        assert filepath.read_text(encoding='utf-8') == "New"

    def test_write_text_file_append(self, temp_dir):
        """Test appending to a file."""
        filepath = temp_dir / 'output.txt'
        write_text_file(filepath, "First\n")
        write_text_file(filepath, "Second\n", append=True)
        
        assert filepath.read_text(encoding='utf-8') == "First\nSecond\n"

    def test_read_lines_generator(self, sample_text_file):
        """Test reading file line by line."""
        filepath, _ = sample_text_file
        lines = list(read_lines_generator(filepath))
        
        assert len(lines) == 3
        assert lines[0] == "Line 1: Hello"
        assert lines[2] == "Line 3: Test"

    def test_read_lines_generator_no_strip(self, sample_text_file):
        """Test reading lines without stripping newlines."""
        filepath, _ = sample_text_file
        lines = list(read_lines_generator(filepath, strip_newlines=False))
        
        assert lines[0].endswith('\n')


# ═══════════════════════════════════════════════════════════════════════════════
# BINARY FILE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryFileOperations:
    """Tests for binary file operations."""

    def test_binary_round_trip(self, temp_dir):
        """Test writing and reading binary data."""
        filepath = temp_dir / 'data.bin'
        original = b'\x00\x01\x02\xff\xfe\xfd'
        
        bytes_written = write_binary_file(filepath, original)
        loaded = read_binary_file(filepath)
        
        assert bytes_written == len(original)
        assert loaded == original

    def test_atomic_write_creates_file(self, temp_dir):
        """Test atomic write creates the file."""
        filepath = temp_dir / 'atomic.txt'
        content = "Atomic content"
        
        atomic_write_text(filepath, content)
        
        assert filepath.exists()
        assert filepath.read_text(encoding='utf-8') == content


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKSUM TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestChecksums:
    """Tests for checksum computation."""

    def test_checksum_deterministic(self, sample_text_file):
        """Test that checksums are deterministic."""
        filepath, _ = sample_text_file
        
        checksum1 = compute_file_checksum(filepath)
        checksum2 = compute_file_checksum(filepath)
        
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 produces 64 hex chars

    def test_checksum_different_content(self, temp_dir):
        """Test different content produces different checksums."""
        file1 = temp_dir / 'file1.txt'
        file2 = temp_dir / 'file2.txt'
        
        file1.write_text("Content A", encoding='utf-8')
        file2.write_text("Content B", encoding='utf-8')
        
        assert compute_file_checksum(file1) != compute_file_checksum(file2)


# ═══════════════════════════════════════════════════════════════════════════════
# JSON TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestJSONOperations:
    """Tests for JSON serialisation."""

    def test_json_round_trip_simple(self, temp_dir):
        """Test basic JSON round-trip."""
        filepath = temp_dir / 'data.json'
        data = {'name': 'test', 'values': [1, 2, 3]}
        
        save_json(data, filepath)
        loaded = load_json(filepath)
        
        assert loaded == data

    def test_json_datetime_encoding(self, temp_dir):
        """Test datetime encoding and decoding."""
        filepath = temp_dir / 'data.json'
        now = datetime(2024, 1, 15, 10, 30, 0)
        data = {'timestamp': now}
        
        save_json(data, filepath)
        loaded = load_json(filepath)
        
        assert loaded['timestamp'] == now

    def test_json_path_encoding(self, temp_dir):
        """Test Path encoding and decoding."""
        filepath = temp_dir / 'data.json'
        data = {'path': Path('/data/file.csv')}
        
        save_json(data, filepath)
        loaded = load_json(filepath)
        
        assert loaded['path'] == Path('/data/file.csv')

    def test_validate_json_schema_valid(self):
        """Test schema validation with valid data."""
        data = {'id': '123', 'name': 'test'}
        errors = validate_json_schema(
            data,
            required_fields=['id', 'name'],
            field_types={'id': str, 'name': str}
        )
        assert len(errors) == 0

    def test_validate_json_schema_missing_field(self):
        """Test schema validation with missing field."""
        data = {'id': '123'}
        errors = validate_json_schema(data, required_fields=['id', 'name'])
        assert len(errors) == 1
        assert 'name' in errors[0]


# ═══════════════════════════════════════════════════════════════════════════════
# CSV TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCSVOperations:
    """Tests for CSV operations."""

    def test_read_csv_as_dicts(self, sample_csv_file):
        """Test reading CSV as list of dictionaries."""
        records = read_csv_as_dicts(sample_csv_file)
        
        assert len(records) == 3
        assert records[0]['id'] == '1'
        assert records[0]['name'] == 'alpha'

    def test_write_csv_from_dicts(self, temp_dir):
        """Test writing dictionaries to CSV."""
        filepath = temp_dir / 'output.csv'
        data = [
            {'a': '1', 'b': '2'},
            {'a': '3', 'b': '4'},
        ]
        
        count = write_csv_from_dicts(data, filepath)
        
        assert count == 2
        loaded = read_csv_as_dicts(filepath)
        assert loaded == data

    def test_stream_csv_records(self, sample_csv_file):
        """Test streaming CSV records."""
        records = list(stream_csv_records(sample_csv_file))
        
        assert len(records) == 3
        assert records[1]['name'] == 'beta'

    def test_convert_csv_types(self):
        """Test type conversion for CSV records."""
        record = {'id': '42', 'value': '3.14', 'name': 'test'}
        converted = convert_csv_types(
            record,
            {'id': int, 'value': float}
        )
        
        assert converted['id'] == 42
        assert converted['value'] == 3.14
        assert converted['name'] == 'test'


# ═══════════════════════════════════════════════════════════════════════════════
# PICKLE AND COMPRESSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBinaryFormats:
    """Tests for Pickle and compression."""

    def test_pickle_round_trip(self, temp_dir):
        """Test Pickle serialisation round-trip."""
        filepath = temp_dir / 'data.pkl'
        data = {'key': 'value', 'list': [1, 2, 3]}
        
        save_pickle(data, filepath)
        loaded = load_pickle(filepath)
        
        assert loaded == data

    def test_compression_round_trip(self, temp_dir):
        """Test gzip compression round-trip."""
        filepath = temp_dir / 'data.gz'
        original = b'Test data ' * 1000
        
        compressed_size = save_compressed(original, filepath)
        decompressed = load_compressed(filepath)
        
        assert compressed_size < len(original)
        assert decompressed == original


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarks:
    """Tests for benchmarking utilities."""

    def test_generate_sample_records(self):
        """Test sample record generation."""
        records = generate_sample_records(100)
        
        assert len(records) == 100
        assert 'id' in records[0]
        assert 'temperature' in records[0]

    def test_format_benchmark_properties(self):
        """Test FormatBenchmark property calculations."""
        bench = FormatBenchmark(
            format_name='Test',
            write_time_seconds=1.0,
            read_time_seconds=0.5,
            file_size_bytes=10000,
            record_count=100
        )
        
        assert bench.write_records_per_second == 100.0
        assert bench.read_records_per_second == 200.0
        assert bench.bytes_per_record == 100.0

    def test_compare_formats(self, temp_dir):
        """Test format comparison benchmark."""
        records = generate_sample_records(10)
        benchmarks = compare_formats(records, temp_dir)
        
        assert len(benchmarks) >= 3  # JSON, CSV, Pickle at minimum
        assert all(b.record_count == 10 for b in benchmarks)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_csv_write(self, temp_dir):
        """Test writing empty list to CSV."""
        filepath = temp_dir / 'empty.csv'
        count = write_csv_from_dicts([], filepath)
        assert count == 0

    def test_unicode_in_text_file(self, temp_dir):
        """Test Unicode content in text files."""
        filepath = temp_dir / 'unicode.txt'
        content = "Hello 世界 🌍 Ελληνικά"
        
        write_text_file(filepath, content)
        loaded = read_text_file(filepath)
        
        assert loaded == content


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
