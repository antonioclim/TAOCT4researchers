#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
10UNIT, Lab 1: File I/O and Serialisation
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Large datasets may not fit entirely in memory, requiring streaming processing.
Simulations may run for days or weeks, necessitating checkpointing—saving
intermediate states to files."
— Wilson et al., 2017

Data persistence forms the bridge between ephemeral computation and permanent
scientific record. This laboratory explores Python's file handling mechanisms,
from elementary text operations through structured serialisation formats to
efficient binary representations for large-scale research data.

PREREQUISITES
─────────────
- Week 9: Exception Handling (for reliable I/O error recovery)
- Python: Intermediate proficiency with context managers and generators
- Libraries: json, csv, pathlib, pickle, hashlib

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement reliable file I/O operations with proper resource management
2. Serialise and deserialise complex data structures using JSON and CSV
3. Handle binary formats including Pickle with appropriate security awareness
4. Compare format characteristics to select appropriate storage strategies

ESTIMATED TIME
──────────────
- Reading: 40 minutes
- Coding: 50 minutes
- Total: 90 minutes

DEPENDENCIES
────────────
- Python 3.12+
- pandas ≥2.0 (for Parquet examples)
- pyarrow ≥14.0 (for Parquet support)
- Standard library: json, csv, pathlib, pickle, hashlib, gzip

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import pickle
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Generator, Iterator, TextIO, TypeVar

# Configure module-level logger
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: FILE OPERATIONS FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════


def read_text_file(filepath: Path, encoding: str = 'utf-8') -> str:
    """
    Read entire text file contents with explicit encoding.

    This function demonstrates the fundamental pattern for safe file reading:
    using context managers for automatic resource cleanup and explicit encoding
    declarations for cross-platform consistency.

    Args:
        filepath: Path to the text file to read.
        encoding: Character encoding (default UTF-8).

    Returns:
        Complete file contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        UnicodeDecodeError: If encoding is incorrect for file contents.

    Example:
        >>> content = read_text_file(Path('data/notes.txt'))
        >>> print(f"Read {len(content)} characters")
    """
    logger.debug("Reading text file: %s with encoding %s", filepath, encoding)

    with open(filepath, 'r', encoding=encoding) as file_handle:
        content = file_handle.read()

    logger.debug("Successfully read %d characters", len(content))
    return content


def write_text_file(
    filepath: Path,
    content: str,
    encoding: str = 'utf-8',
    append: bool = False
) -> int:
    """
    Write string content to text file with explicit encoding.

    Args:
        filepath: Destination file path.
        content: String content to write.
        encoding: Character encoding (default UTF-8).
        append: If True, append to existing file; otherwise overwrite.

    Returns:
        Number of characters written.

    Example:
        >>> chars_written = write_text_file(Path('output.txt'), 'Hello, World!')
        >>> print(f"Wrote {chars_written} characters")
    """
    mode = 'a' if append else 'w'
    logger.debug("Writing to %s in mode '%s'", filepath, mode)

    with open(filepath, mode, encoding=encoding) as file_handle:
        chars_written = file_handle.write(content)

    logger.debug("Wrote %d characters to %s", chars_written, filepath)
    return chars_written


def read_lines_generator(
    filepath: Path,
    encoding: str = 'utf-8',
    strip_newlines: bool = True
) -> Generator[str, None, None]:
    """
    Generate lines from file without loading entire contents into memory.

    This streaming approach enables processing of files larger than available
    RAM by yielding one line at a time. Essential for large log files,
    genomic sequences, and multi-gigabyte datasets.

    Args:
        filepath: Path to the text file.
        encoding: Character encoding.
        strip_newlines: If True, remove trailing newline characters.

    Yields:
        Individual lines from the file.

    Example:
        >>> for line_num, line in enumerate(read_lines_generator(Path('large.txt'))):
        ...     if 'ERROR' in line:
        ...         print(f"Error at line {line_num + 1}")
    """
    logger.debug("Opening file for streaming: %s", filepath)

    with open(filepath, 'r', encoding=encoding) as file_handle:
        for line in file_handle:
            if strip_newlines:
                yield line.rstrip('\n\r')
            else:
                yield line


def read_binary_file(filepath: Path) -> bytes:
    """
    Read entire binary file contents.

    Binary mode bypasses character encoding, treating file contents as raw
    byte sequences. Essential for images, compressed archives, and serialised
    binary formats.

    Args:
        filepath: Path to the binary file.

    Returns:
        Raw file contents as bytes.

    Example:
        >>> image_data = read_binary_file(Path('photo.jpg'))
        >>> print(f"Image size: {len(image_data)} bytes")
    """
    logger.debug("Reading binary file: %s", filepath)

    with open(filepath, 'rb') as file_handle:
        content = file_handle.read()

    logger.debug("Read %d bytes from %s", len(content), filepath)
    return content


def write_binary_file(filepath: Path, data: bytes) -> int:
    """
    Write binary data to file.

    Args:
        filepath: Destination file path.
        data: Binary data to write.

    Returns:
        Number of bytes written.

    Example:
        >>> bytes_written = write_binary_file(Path('output.bin'), b'\\x00\\x01\\x02')
    """
    logger.debug("Writing %d bytes to %s", len(data), filepath)

    with open(filepath, 'wb') as file_handle:
        bytes_written = file_handle.write(data)

    return bytes_written


def atomic_write_text(
    filepath: Path,
    content: str,
    encoding: str = 'utf-8'
) -> None:
    """
    Write content atomically to prevent partial updates on failure.

    Atomic writes work by first writing to a temporary file in the same
    directory, then renaming it to the target path. The rename operation
    is atomic on POSIX systems, ensuring the file either contains the
    complete new content or remains unchanged.

    Args:
        filepath: Final destination path.
        content: String content to write.
        encoding: Character encoding.

    Example:
        >>> atomic_write_text(Path('critical_config.json'), json_content)
        # File is guaranteed to be complete or unchanged
    """
    logger.debug("Performing atomic write to %s", filepath)

    # Create temporary file in same directory for same-filesystem rename
    directory = filepath.parent
    directory.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding=encoding,
        dir=directory,
        delete=False,
        suffix='.tmp'
    ) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    # Atomic rename (on POSIX systems)
    temp_path.replace(filepath)
    logger.debug("Atomic write completed successfully")


def compute_file_checksum(
    filepath: Path,
    algorithm: str = 'sha256',
    chunk_size: int = 65536
) -> str:
    """
    Compute cryptographic checksum of file contents efficiently.

    Processes the file in chunks to handle arbitrarily large files without
    loading them entirely into memory.

    Args:
        filepath: Path to file.
        algorithm: Hash algorithm name (sha256, sha512, md5, etc.).
        chunk_size: Size of chunks to read at a time.

    Returns:
        Hexadecimal checksum string.

    Example:
        >>> checksum = compute_file_checksum(Path('dataset.csv'))
        >>> print(f"SHA-256: {checksum}")
    """
    logger.debug("Computing %s checksum for %s", algorithm, filepath)

    hasher = hashlib.new(algorithm)

    with open(filepath, 'rb') as file_handle:
        while True:
            chunk = file_handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    checksum = hasher.hexdigest()
    logger.debug("Checksum computed: %s...%s", checksum[:8], checksum[-8:])
    return checksum


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: JSON SERIALISATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentMetadata:
    """
    Metadata record for a scientific experiment.

    This dataclass demonstrates structured data that benefits from JSON
    serialisation for human readability and cross-platform interchange.
    """

    experiment_id: str
    researcher: str
    start_time: datetime
    parameters: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    notes: str = ''


class ResearchJSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder for research data types.

    Standard JSON cannot represent Python-specific types like datetime,
    Path, or complex numbers. This encoder provides custom serialisation
    for common research data types.

    Supported types:
        - datetime: ISO 8601 format string
        - date: ISO format string
        - Path: String representation
        - complex: Tuple of (real, imaginary)
        - bytes: Base64-encoded string
        - set: List representation
        - dataclass instances: Dictionary via asdict()

    Example:
        >>> data = {'timestamp': datetime.now(), 'path': Path('/data')}
        >>> json.dumps(data, cls=ResearchJSONEncoder)
    """

    def default(self, obj: Any) -> Any:
        """Convert non-standard types to JSON-serialisable representations."""
        if isinstance(obj, datetime):
            return {'__type__': 'datetime', 'value': obj.isoformat()}

        if isinstance(obj, Path):
            return {'__type__': 'path', 'value': str(obj)}

        if isinstance(obj, complex):
            return {'__type__': 'complex', 'real': obj.real, 'imag': obj.imag}

        if isinstance(obj, bytes):
            import base64
            return {'__type__': 'bytes', 'value': base64.b64encode(obj).decode('ascii')}

        if isinstance(obj, set):
            return {'__type__': 'set', 'value': list(obj)}

        if hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)

        return super().default(obj)


def research_json_decoder(dct: dict[str, Any]) -> Any:
    """
    Object hook for decoding research-specific JSON types.

    Args:
        dct: Dictionary from JSON parsing.

    Returns:
        Decoded object or original dictionary.

    Example:
        >>> data = json.loads(json_string, object_hook=research_json_decoder)
    """
    if '__type__' not in dct:
        return dct

    type_tag = dct['__type__']

    if type_tag == 'datetime':
        return datetime.fromisoformat(dct['value'])

    if type_tag == 'path':
        return Path(dct['value'])

    if type_tag == 'complex':
        return complex(dct['real'], dct['imag'])

    if type_tag == 'bytes':
        import base64
        return base64.b64decode(dct['value'])

    if type_tag == 'set':
        return set(dct['value'])

    return dct


def save_json(
    data: Any,
    filepath: Path,
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Save data to JSON file with research-friendly encoding.

    Args:
        data: Data structure to serialise.
        filepath: Destination file path.
        indent: Indentation level for pretty-printing.
        ensure_ascii: If False, allow non-ASCII characters.

    Example:
        >>> experiment = {'name': 'Trial α', 'results': [1.5, 2.3, 3.1]}
        >>> save_json(experiment, Path('results.json'))
    """
    logger.debug("Saving JSON to %s", filepath)

    json_string = json.dumps(
        data,
        cls=ResearchJSONEncoder,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=True
    )

    atomic_write_text(filepath, json_string)
    logger.debug("JSON saved successfully")


def load_json(filepath: Path) -> Any:
    """
    Load data from JSON file with research-specific type decoding.

    Args:
        filepath: Path to JSON file.

    Returns:
        Deserialised data structure.

    Example:
        >>> data = load_json(Path('results.json'))
    """
    logger.debug("Loading JSON from %s", filepath)

    with open(filepath, 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle, object_hook=research_json_decoder)

    logger.debug("JSON loaded successfully")
    return data


def validate_json_schema(
    data: dict[str, Any],
    required_fields: list[str],
    field_types: dict[str, type] | None = None
) -> list[str]:
    """
    Validate JSON data against a simple schema.

    Args:
        data: Dictionary to validate.
        required_fields: List of field names that must be present.
        field_types: Optional mapping of field names to expected types.

    Returns:
        List of validation error messages (empty if valid).

    Example:
        >>> errors = validate_json_schema(
        ...     data,
        ...     required_fields=['id', 'timestamp'],
        ...     field_types={'id': str, 'timestamp': str}
        ... )
    """
    errors: list[str] = []

    for field_name in required_fields:
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")

    if field_types:
        for field_name, expected_type in field_types.items():
            if field_name in data and not isinstance(data[field_name], expected_type):
                actual_type = type(data[field_name]).__name__
                errors.append(
                    f"Field '{field_name}' has type {actual_type}, "
                    f"expected {expected_type.__name__}"
                )

    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CSV PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MeasurementRecord:
    """
    Individual measurement from a research instrument.

    Demonstrates the typical structure of tabular research data suitable
    for CSV serialisation.
    """

    timestamp: str
    sensor_id: str
    value: float
    unit: str
    quality_flag: str = 'valid'


def read_csv_as_dicts(
    filepath: Path,
    encoding: str = 'utf-8',
    delimiter: str = ','
) -> list[dict[str, str]]:
    """
    Read CSV file as list of dictionaries.

    Uses the first row as column headers. All values are returned as
    strings; type conversion is the caller's responsibility.

    Args:
        filepath: Path to CSV file.
        encoding: Character encoding.
        delimiter: Field separator character.

    Returns:
        List of dictionaries, one per row.

    Example:
        >>> records = read_csv_as_dicts(Path('measurements.csv'))
        >>> for record in records:
        ...     print(record['sensor_id'], float(record['value']))
    """
    logger.debug("Reading CSV from %s", filepath)

    records: list[dict[str, str]] = []

    with open(filepath, 'r', encoding=encoding, newline='') as file_handle:
        reader = csv.DictReader(file_handle, delimiter=delimiter)
        for row in reader:
            records.append(dict(row))

    logger.debug("Read %d records from CSV", len(records))
    return records


def write_csv_from_dicts(
    records: list[dict[str, Any]],
    filepath: Path,
    fieldnames: list[str] | None = None,
    encoding: str = 'utf-8',
    delimiter: str = ','
) -> int:
    """
    Write list of dictionaries to CSV file.

    Args:
        records: List of dictionaries to write.
        filepath: Destination file path.
        fieldnames: Column order (uses first record's keys if None).
        encoding: Character encoding.
        delimiter: Field separator character.

    Returns:
        Number of records written.

    Example:
        >>> records = [{'id': '001', 'value': 23.5}, {'id': '002', 'value': 24.1}]
        >>> write_csv_from_dicts(records, Path('output.csv'))
    """
    if not records:
        logger.warning("No records to write")
        return 0

    if fieldnames is None:
        fieldnames = list(records[0].keys())

    logger.debug("Writing %d records to CSV: %s", len(records), filepath)

    with open(filepath, 'w', encoding=encoding, newline='') as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction='ignore'
        )
        writer.writeheader()
        writer.writerows(records)

    logger.debug("CSV write completed")
    return len(records)


def stream_csv_records(
    filepath: Path,
    encoding: str = 'utf-8',
    delimiter: str = ','
) -> Generator[dict[str, str], None, None]:
    """
    Stream CSV records one at a time for memory-efficient processing.

    Args:
        filepath: Path to CSV file.
        encoding: Character encoding.
        delimiter: Field separator.

    Yields:
        Individual row dictionaries.

    Example:
        >>> total = 0.0
        >>> for record in stream_csv_records(Path('large_dataset.csv')):
        ...     total += float(record['value'])
    """
    logger.debug("Streaming CSV from %s", filepath)

    with open(filepath, 'r', encoding=encoding, newline='') as file_handle:
        reader = csv.DictReader(file_handle, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def convert_csv_types(
    record: dict[str, str],
    type_mapping: dict[str, Callable[[str], Any]]
) -> dict[str, Any]:
    """
    Convert CSV string values to appropriate Python types.

    Args:
        record: Dictionary with string values.
        type_mapping: Mapping of field names to conversion functions.

    Returns:
        Dictionary with converted values.

    Example:
        >>> type_map = {'value': float, 'count': int, 'active': lambda x: x == 'true'}
        >>> converted = convert_csv_types(raw_record, type_map)
    """
    converted: dict[str, Any] = {}

    for key, value in record.items():
        if key in type_mapping:
            try:
                converted[key] = type_mapping[key](value)
            except (ValueError, TypeError) as exc:
                logger.warning("Failed to convert %s='%s': %s", key, value, exc)
                converted[key] = value
        else:
            converted[key] = value

    return converted


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BINARY FORMATS AND PICKLE
# ═══════════════════════════════════════════════════════════════════════════════


def save_pickle(
    obj: Any,
    filepath: Path,
    protocol: int | None = None
) -> None:
    """
    Serialise Python object to Pickle file.

    WARNING: Pickle files can execute arbitrary code during loading.
    NEVER unpickle data from untrusted sources.

    Args:
        obj: Any Python object to serialise.
        filepath: Destination file path.
        protocol: Pickle protocol version (None = highest available).

    Example:
        >>> save_pickle(trained_model, Path('model_checkpoint.pkl'))
    """
    logger.debug("Saving pickle to %s", filepath)

    if protocol is None:
        protocol = pickle.HIGHEST_PROTOCOL

    with open(filepath, 'wb') as file_handle:
        pickle.dump(obj, file_handle, protocol=protocol)

    logger.debug("Pickle saved with protocol %d", protocol)


def load_pickle(filepath: Path) -> Any:
    """
    Load Python object from Pickle file.

    WARNING: Only load Pickle files from trusted sources.
    Malicious Pickle files can execute arbitrary code.

    Args:
        filepath: Path to Pickle file.

    Returns:
        Deserialised Python object.

    Example:
        >>> model = load_pickle(Path('model_checkpoint.pkl'))
    """
    logger.debug("Loading pickle from %s", filepath)

    with open(filepath, 'rb') as file_handle:
        obj = pickle.load(file_handle)

    logger.debug("Pickle loaded successfully")
    return obj


def save_compressed(
    data: bytes,
    filepath: Path,
    compression_level: int = 9
) -> int:
    """
    Save binary data with gzip compression.

    Args:
        data: Binary data to compress and save.
        filepath: Destination path (should end in .gz).
        compression_level: Compression level (1-9, higher = smaller).

    Returns:
        Size of compressed file in bytes.

    Example:
        >>> json_bytes = json.dumps(large_data).encode('utf-8')
        >>> compressed_size = save_compressed(json_bytes, Path('data.json.gz'))
    """
    logger.debug("Saving compressed data to %s", filepath)

    with gzip.open(filepath, 'wb', compresslevel=compression_level) as file_handle:
        file_handle.write(data)

    compressed_size = filepath.stat().st_size
    compression_ratio = len(data) / compressed_size if compressed_size > 0 else 0

    logger.debug(
        "Compressed %d bytes to %d bytes (ratio: %.2fx)",
        len(data), compressed_size, compression_ratio
    )

    return compressed_size


def load_compressed(filepath: Path) -> bytes:
    """
    Load and decompress gzip-compressed binary data.

    Args:
        filepath: Path to compressed file.

    Returns:
        Decompressed binary data.

    Example:
        >>> data = load_compressed(Path('data.json.gz'))
        >>> parsed = json.loads(data.decode('utf-8'))
    """
    logger.debug("Loading compressed data from %s", filepath)

    with gzip.open(filepath, 'rb') as file_handle:
        data = file_handle.read()

    logger.debug("Decompressed to %d bytes", len(data))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: FORMAT COMPARISON AND BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FormatBenchmark:
    """Results of format comparison benchmark."""

    format_name: str
    write_time_seconds: float
    read_time_seconds: float
    file_size_bytes: int
    record_count: int

    @property
    def write_records_per_second(self) -> float:
        """Calculate write throughput."""
        if self.write_time_seconds > 0:
            return self.record_count / self.write_time_seconds
        return 0.0

    @property
    def read_records_per_second(self) -> float:
        """Calculate read throughput."""
        if self.read_time_seconds > 0:
            return self.record_count / self.read_time_seconds
        return 0.0

    @property
    def bytes_per_record(self) -> float:
        """Calculate storage efficiency."""
        if self.record_count > 0:
            return self.file_size_bytes / self.record_count
        return 0.0


def generate_sample_records(count: int) -> list[dict[str, Any]]:
    """
    Generate sample measurement records for benchmarking.

    Args:
        count: Number of records to generate.

    Returns:
        List of sample measurement dictionaries.
    """
    import random

    records = []
    base_time = datetime(2024, 1, 1, 0, 0, 0)

    for i in range(count):
        record = {
            'id': f'REC{i:08d}',
            'timestamp': (base_time.isoformat()),
            'sensor': f'SENSOR_{i % 10:03d}',
            'temperature': round(20.0 + random.gauss(0, 5), 3),
            'humidity': round(50.0 + random.gauss(0, 15), 2),
            'pressure': round(1013.25 + random.gauss(0, 10), 2),
            'quality': random.choice(['good', 'fair', 'poor']),
        }
        records.append(record)

    return records


def compare_formats(
    records: list[dict[str, Any]],
    output_dir: Path
) -> list[FormatBenchmark]:
    """
    Compare serialisation formats for the given records.

    Tests JSON, CSV, and Pickle formats, measuring write/read times
    and file sizes.

    Args:
        records: Data to serialise.
        output_dir: Directory for temporary output files.

    Returns:
        List of benchmark results for each format.

    Example:
        >>> records = generate_sample_records(10000)
        >>> results = compare_formats(records, Path('/tmp/benchmarks'))
        >>> for r in results:
        ...     print(f"{r.format_name}: {r.file_size_bytes} bytes")
    """
    import time

    output_dir.mkdir(parents=True, exist_ok=True)
    benchmarks: list[FormatBenchmark] = []

    # JSON benchmark
    json_path = output_dir / 'benchmark.json'
    start = time.perf_counter()
    save_json(records, json_path)
    write_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = load_json(json_path)
    read_time = time.perf_counter() - start

    benchmarks.append(FormatBenchmark(
        format_name='JSON',
        write_time_seconds=write_time,
        read_time_seconds=read_time,
        file_size_bytes=json_path.stat().st_size,
        record_count=len(records)
    ))

    # CSV benchmark
    csv_path = output_dir / 'benchmark.csv'
    start = time.perf_counter()
    write_csv_from_dicts(records, csv_path)
    write_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = read_csv_as_dicts(csv_path)
    read_time = time.perf_counter() - start

    benchmarks.append(FormatBenchmark(
        format_name='CSV',
        write_time_seconds=write_time,
        read_time_seconds=read_time,
        file_size_bytes=csv_path.stat().st_size,
        record_count=len(records)
    ))

    # Pickle benchmark
    pickle_path = output_dir / 'benchmark.pkl'
    start = time.perf_counter()
    save_pickle(records, pickle_path)
    write_time = time.perf_counter() - start

    start = time.perf_counter()
    _ = load_pickle(pickle_path)
    read_time = time.perf_counter() - start

    benchmarks.append(FormatBenchmark(
        format_name='Pickle',
        write_time_seconds=write_time,
        read_time_seconds=read_time,
        file_size_bytes=pickle_path.stat().st_size,
        record_count=len(records)
    ))

    # Compressed JSON benchmark
    gzip_path = output_dir / 'benchmark.json.gz'
    json_bytes = json.dumps(records, cls=ResearchJSONEncoder).encode('utf-8')

    start = time.perf_counter()
    save_compressed(json_bytes, gzip_path)
    write_time = time.perf_counter() - start

    start = time.perf_counter()
    loaded_bytes = load_compressed(gzip_path)
    _ = json.loads(loaded_bytes.decode('utf-8'))
    read_time = time.perf_counter() - start

    benchmarks.append(FormatBenchmark(
        format_name='JSON+gzip',
        write_time_seconds=write_time,
        read_time_seconds=read_time,
        file_size_bytes=gzip_path.stat().st_size,
        record_count=len(records)
    ))

    return benchmarks


def print_benchmark_report(benchmarks: list[FormatBenchmark]) -> None:
    """
    Print formatted benchmark comparison report.

    Args:
        benchmarks: List of benchmark results to display.
    """
    logger.info("=" * 70)
    logger.info("FORMAT COMPARISON BENCHMARK REPORT")
    logger.info("=" * 70)

    header = f"{'Format':<12} {'Size (KB)':<12} {'Write (ms)':<12} {'Read (ms)':<12} {'B/rec':<10}"
    logger.info(header)
    logger.info("-" * 70)

    for bench in benchmarks:
        row = (
            f"{bench.format_name:<12} "
            f"{bench.file_size_bytes / 1024:<12.2f} "
            f"{bench.write_time_seconds * 1000:<12.2f} "
            f"{bench.read_time_seconds * 1000:<12.2f} "
            f"{bench.bytes_per_record:<10.2f}"
        )
        logger.info(row)

    logger.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    Demonstrate file I/O and serialisation capabilities.

    This function provides a comprehensive walkthrough of the module's
    functionality, suitable for both learning and verification.
    """
    # Configure logging for demonstration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    logger.info("Starting File I/O and Serialisation Lab Demonstration")

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Demonstrate text file operations
        logger.info("\n--- Text File Operations ---")
        sample_text = "Line 1: Introduction to computational thinking\n"
        sample_text += "Line 2: Data persistence fundamentals\n"
        sample_text += "Line 3: Serialisation strategies\n"

        text_path = output_dir / 'sample.txt'
        write_text_file(text_path, sample_text)
        content = read_text_file(text_path)
        logger.info("Text file round-trip successful: %d chars", len(content))

        # Demonstrate JSON serialisation
        logger.info("\n--- JSON Serialisation ---")
        experiment = {
            'id': 'EXP-2024-001',
            'timestamp': datetime.now(),
            'parameters': {'learning_rate': 0.001, 'epochs': 100},
            'results': [0.95, 0.96, 0.97],
            'data_path': Path('/data/training'),
        }

        json_path = output_dir / 'experiment.json'
        save_json(experiment, json_path)
        loaded = load_json(json_path)
        logger.info("JSON round-trip: timestamp type = %s", type(loaded['timestamp']))

        # Demonstrate CSV operations
        logger.info("\n--- CSV Operations ---")
        measurements = [
            {'timestamp': '2024-01-01T10:00:00', 'sensor': 'T001', 'value': '23.5'},
            {'timestamp': '2024-01-01T10:01:00', 'sensor': 'T001', 'value': '23.7'},
            {'timestamp': '2024-01-01T10:02:00', 'sensor': 'T001', 'value': '23.6'},
        ]

        csv_path = output_dir / 'measurements.csv'
        write_csv_from_dicts(measurements, csv_path)
        loaded_csv = read_csv_as_dicts(csv_path)
        logger.info("CSV round-trip: %d records", len(loaded_csv))

        # Demonstrate checksum computation
        logger.info("\n--- Checksum Verification ---")
        checksum = compute_file_checksum(csv_path)
        logger.info("SHA-256 checksum: %s", checksum)

        # Run format comparison benchmark
        logger.info("\n--- Format Comparison Benchmark ---")
        sample_records = generate_sample_records(1000)
        benchmarks = compare_formats(sample_records, output_dir / 'benchmarks')
        print_benchmark_report(benchmarks)

    logger.info("\nDemonstration completed successfully")


if __name__ == '__main__':
    main()
