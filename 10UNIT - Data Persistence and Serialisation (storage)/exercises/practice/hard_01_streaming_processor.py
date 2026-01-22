#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Hard 01 - Streaming Processor
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Large datasets may not fit entirely in memory, requiring streaming processing."
— Wilson et al., 2017

This exercise implements memory-efficient streaming processors for handling
datasets that exceed available RAM. Essential techniques for processing
genomic sequences, climate data archives, and large-scale simulations.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Process arbitrarily large files with constant memory usage
2. Implement chunked reading and aggregation patterns
3. Design streaming pipelines with filtering and transformation

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 45 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, Iterator, TypeVar

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Streaming Line Processor
# ═══════════════════════════════════════════════════════════════════════════════

def stream_lines(
    filepath: Path,
    encoding: str = 'utf-8',
    skip_blank: bool = True
) -> Generator[tuple[int, str], None, None]:
    """
    Stream lines from a file with line numbers.

    Yields one line at a time, maintaining constant memory regardless of
    file size. Essential for processing multi-gigabyte log files.

    Args:
        filepath: Path to text file.
        encoding: File encoding.
        skip_blank: If True, skip empty lines.

    Yields:
        Tuples of (line_number, line_content) where line_number is 1-indexed.

    Example:
        >>> for line_num, line in stream_lines(Path('huge.log')):
        ...     if 'ERROR' in line:
        ...         print(f"Error at line {line_num}")
    """
    # TODO: Implement this function
    pass


def count_pattern_occurrences(
    filepath: Path,
    pattern: str,
    case_sensitive: bool = True
) -> dict[str, int]:
    """
    Count occurrences of a pattern in a large file using streaming.

    Returns both total count and count per line (for lines containing pattern).
    Must use constant memory regardless of file size.

    Args:
        filepath: Path to text file.
        pattern: String pattern to search for.
        case_sensitive: Whether to match case.

    Returns:
        Dictionary with 'total_count', 'lines_with_pattern', 'first_occurrence'.

    Example:
        >>> stats = count_pattern_occurrences(Path('log.txt'), 'ERROR')
        >>> print(f"Found {stats['total_count']} errors")
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Streaming CSV Processor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StreamingStats:
    """Statistics computed via streaming aggregation."""
    count: int = 0
    sum_value: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    # For computing variance using Welford's online algorithm
    mean: float = 0.0
    m2: float = 0.0  # Sum of squared differences from mean

    def update(self, value: float) -> None:
        """Update statistics with a new value using Welford's algorithm."""
        self.count += 1
        self.sum_value += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

        # Welford's online algorithm for variance
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        """Compute variance from accumulated statistics."""
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Compute standard deviation."""
        return self.variance ** 0.5


def stream_csv_stats(
    filepath: Path,
    value_column: str,
    group_by: str | None = None
) -> dict[str, StreamingStats]:
    """
    Compute statistics on a CSV column using streaming aggregation.

    Processes the file line-by-line, never loading the entire file into
    memory. Optionally groups statistics by another column.

    Args:
        filepath: Path to CSV file.
        value_column: Name of the column containing numeric values.
        group_by: Optional column name to group statistics by.

    Returns:
        Dictionary mapping group names (or '_all' if no grouping) to stats.

    Example:
        >>> stats = stream_csv_stats(
        ...     Path('measurements.csv'),
        ...     'temperature',
        ...     group_by='sensor_id'
        ... )
        >>> for sensor, s in stats.items():
        ...     print(f"{sensor}: mean={s.mean:.2f}, std={s.std_dev:.2f}")
    """
    # TODO: Implement this function
    pass


def stream_csv_filter(
    input_path: Path,
    output_path: Path,
    filter_func: Callable[[dict[str, str]], bool]
) -> tuple[int, int]:
    """
    Filter a large CSV file using streaming, writing matching rows to output.

    Args:
        input_path: Path to source CSV file.
        output_path: Path for filtered output.
        filter_func: Function that returns True for rows to keep.

    Returns:
        Tuple of (total_rows, kept_rows).

    Example:
        >>> total, kept = stream_csv_filter(
        ...     Path('huge.csv'),
        ...     Path('filtered.csv'),
        ...     lambda row: float(row['value']) > 100
        ... )
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Chunked Binary Processor
# ═══════════════════════════════════════════════════════════════════════════════

def stream_binary_chunks(
    filepath: Path,
    chunk_size: int = 65536
) -> Generator[tuple[int, bytes], None, None]:
    """
    Stream binary file in chunks.

    Args:
        filepath: Path to binary file.
        chunk_size: Size of each chunk in bytes.

    Yields:
        Tuples of (chunk_index, chunk_data).

    Example:
        >>> for idx, chunk in stream_binary_chunks(Path('large.bin')):
        ...     process_chunk(chunk)
    """
    # TODO: Implement this function
    pass


def compute_rolling_hash(
    filepath: Path,
    window_size: int = 1024,
    step: int = 512
) -> Generator[tuple[int, str], None, None]:
    """
    Compute rolling hash values for detecting duplicate segments.

    Uses a sliding window approach common in deduplication systems
    and similarity detection algorithms.

    Args:
        filepath: Path to binary file.
        window_size: Size of the rolling window in bytes.
        step: Step size between windows.

    Yields:
        Tuples of (offset, md5_hash) for each window.

    Example:
        >>> hashes = list(compute_rolling_hash(Path('data.bin')))
        >>> # Find duplicate windows
        >>> seen = {}
        >>> for offset, h in hashes:
        ...     if h in seen:
        ...         print(f"Duplicate at {offset}, same as {seen[h]}")
        ...     seen[h] = offset
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Streaming JSON Lines Processor
# ═══════════════════════════════════════════════════════════════════════════════

def stream_jsonl(filepath: Path) -> Generator[dict[str, Any], None, None]:
    """
    Stream JSON Lines (JSONL) file, yielding one object per line.

    JSONL format has one JSON object per line, enabling streaming without
    loading the entire array into memory.

    Args:
        filepath: Path to JSONL file.

    Yields:
        Parsed JSON objects.

    Example:
        >>> for record in stream_jsonl(Path('events.jsonl')):
        ...     process_event(record)
    """
    # TODO: Implement this function
    pass


def aggregate_jsonl(
    filepath: Path,
    group_key: str,
    value_key: str,
    aggregation: str = 'sum'
) -> dict[str, float]:
    """
    Aggregate values from a JSONL file by group key.

    Supported aggregations: 'sum', 'count', 'mean', 'min', 'max'.

    Args:
        filepath: Path to JSONL file.
        group_key: Key to group by.
        value_key: Key containing numeric values.
        aggregation: Aggregation function name.

    Returns:
        Dictionary mapping group values to aggregated results.

    Example:
        >>> totals = aggregate_jsonl(
        ...     Path('sales.jsonl'),
        ...     group_key='region',
        ...     value_key='amount',
        ...     aggregation='sum'
        ... )
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Streaming Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineStage:
    """A stage in the streaming pipeline."""
    name: str
    transform: Callable[[dict[str, Any]], dict[str, Any] | None]
    stats: dict[str, int] = field(default_factory=lambda: {'input': 0, 'output': 0})


class StreamingPipeline:
    """
    Configurable streaming data pipeline.

    Processes records through a series of transformation stages,
    filtering out records where any stage returns None.
    """

    def __init__(self) -> None:
        """Initialise empty pipeline."""
        self.stages: list[PipelineStage] = []

    def add_stage(
        self,
        name: str,
        transform: Callable[[dict[str, Any]], dict[str, Any] | None]
    ) -> 'StreamingPipeline':
        """
        Add a transformation stage to the pipeline.

        Args:
            name: Stage name for logging.
            transform: Function that transforms or filters records.
                      Return None to filter out a record.

        Returns:
            Self for method chaining.

        Example:
            >>> pipeline.add_stage('validate', lambda r: r if r['value'] > 0 else None)
        """
        # TODO: Implement this method
        pass

    def process_csv(
        self,
        input_path: Path,
        output_path: Path
    ) -> dict[str, dict[str, int]]:
        """
        Process a CSV file through the pipeline.

        Args:
            input_path: Source CSV file.
            output_path: Destination for processed records.

        Returns:
            Statistics for each stage.

        Example:
            >>> stats = pipeline.process_csv(Path('raw.csv'), Path('clean.csv'))
            >>> for stage, s in stats.items():
            ...     print(f"{stage}: {s['output']}/{s['input']} records")
        """
        # TODO: Implement this method
        pass

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Return statistics for all stages."""
        return {stage.name: dict(stage.stats) for stage in self.stages}


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing Exercise 1: Streaming Line Processor")
        # Create test file
        log_file = test_dir / 'test.log'
        lines = ["INFO: Starting\n", "ERROR: Something failed\n", "\n",
                 "WARNING: Low memory\n", "ERROR: Another error\n"]
        log_file.write_text(''.join(lines), encoding='utf-8')

        streamed = list(stream_lines(log_file))
        assert len(streamed) == 4, "Should skip blank line"
        assert streamed[0] == (1, "INFO: Starting")

        stats = count_pattern_occurrences(log_file, 'ERROR')
        assert stats['total_count'] == 2, "Should find 2 ERRORs"
        print("  ✓ Streaming line processor works correctly")

        print("\nTesting Exercise 2: Streaming CSV Processor")
        csv_file = test_dir / 'data.csv'
        csv_content = "sensor,value\nA,10\nB,20\nA,30\nB,40\nA,50\n"
        csv_file.write_text(csv_content, encoding='utf-8')

        stats = stream_csv_stats(csv_file, 'value', group_by='sensor')
        assert 'A' in stats and 'B' in stats
        assert stats['A'].count == 3
        assert stats['A'].mean == 30.0
        assert stats['B'].mean == 30.0

        filtered = test_dir / 'filtered.csv'
        total, kept = stream_csv_filter(
            csv_file, filtered,
            lambda row: float(row['value']) > 25
        )
        assert total == 5 and kept == 3
        print("  ✓ Streaming CSV processor works correctly")

        print("\nTesting Exercise 3: Chunked Binary Processor")
        bin_file = test_dir / 'test.bin'
        bin_file.write_bytes(b'A' * 1000 + b'B' * 1000 + b'A' * 1000)

        chunks = list(stream_binary_chunks(bin_file, chunk_size=500))
        assert len(chunks) == 6, "Should have 6 chunks of 500 bytes"

        hashes = list(compute_rolling_hash(bin_file, window_size=100, step=100))
        assert len(hashes) > 0, "Should produce hash values"
        print("  ✓ Chunked binary processor works correctly")

        print("\nTesting Exercise 4: Streaming JSON Lines Processor")
        jsonl_file = test_dir / 'events.jsonl'
        events = [
            '{"region": "A", "amount": 100}\n',
            '{"region": "B", "amount": 200}\n',
            '{"region": "A", "amount": 150}\n',
        ]
        jsonl_file.write_text(''.join(events), encoding='utf-8')

        records = list(stream_jsonl(jsonl_file))
        assert len(records) == 3

        totals = aggregate_jsonl(jsonl_file, 'region', 'amount', 'sum')
        assert totals['A'] == 250
        assert totals['B'] == 200
        print("  ✓ Streaming JSONL processor works correctly")

        print("\nTesting Exercise 5: Streaming Pipeline")
        pipeline = StreamingPipeline()
        pipeline.add_stage('parse', lambda r: {**r, 'value': float(r['value'])})
        pipeline.add_stage('filter', lambda r: r if r['value'] > 15 else None)
        pipeline.add_stage('transform', lambda r: {**r, 'value': r['value'] * 2})

        output_csv = test_dir / 'pipeline_out.csv'
        stats = pipeline.process_csv(csv_file, output_csv)
        assert stats['filter']['output'] == 4, "Should filter to 4 records"
        print("  ✓ Streaming pipeline works correctly")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
