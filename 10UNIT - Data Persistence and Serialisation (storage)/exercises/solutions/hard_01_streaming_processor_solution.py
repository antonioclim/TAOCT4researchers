#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Hard 01 - Streaming Data Processor
═══════════════════════════════════════════════════════════════════════════════
"""

import csv
import gzip
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics from data processing."""
    records_processed: int = 0
    records_filtered: int = 0
    errors_encountered: int = 0
    bytes_read: int = 0
    processing_time_seconds: float = 0.0


def stream_csv_file(
    filepath: Path,
    encoding: str = 'utf-8'
) -> Iterator[dict[str, str]]:
    """Stream records from a CSV file one at a time."""
    with open(filepath, 'r', encoding=encoding, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def stream_compressed_csv(filepath: Path) -> Iterator[dict[str, str]]:
    """Stream records from a gzip-compressed CSV file."""
    import io
    with gzip.open(filepath, 'rt', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def stream_jsonl_file(filepath: Path) -> Iterator[dict[str, Any]]:
    """Stream records from a JSON Lines file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")


def filter_records(
    records: Iterator[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool]
) -> Iterator[dict[str, Any]]:
    """Filter records based on a predicate function."""
    for record in records:
        if predicate(record):
            yield record


def transform_records(
    records: Iterator[dict[str, Any]],
    transformer: Callable[[dict[str, Any]], dict[str, Any]]
) -> Iterator[dict[str, Any]]:
    """Apply transformation function to each record."""
    for record in records:
        try:
            yield transformer(record)
        except Exception as e:
            logger.warning(f"Transform error: {e}")


def batch_records(
    records: Iterator[dict[str, Any]],
    batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    """Group records into batches."""
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_csv_streaming(
    records: Iterator[dict[str, Any]],
    filepath: Path,
    fieldnames: list[str]
) -> int:
    """Write records to CSV file with streaming."""
    count = 0
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for record in records:
            writer.writerow(record)
            count += 1
    return count


def write_jsonl_streaming(
    records: Iterator[dict[str, Any]],
    filepath: Path
) -> int:
    """Write records to JSON Lines file with streaming."""
    count = 0
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
            count += 1
    return count


@dataclass
class StreamingAggregator:
    """Compute running statistics without storing all data."""
    count: int = 0
    sum_value: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    sum_squared: float = 0.0

    def update(self, value: float) -> None:
        """Update statistics with a new value."""
        self.count += 1
        self.sum_value += value
        self.sum_squared += value * value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def mean(self) -> float:
        """Compute mean."""
        return self.sum_value / self.count if self.count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Compute variance using Welford's algorithm."""
        if self.count < 2:
            return 0.0
        return (self.sum_squared - (self.sum_value ** 2) / self.count) / (self.count - 1)

    @property
    def std_dev(self) -> float:
        """Compute standard deviation."""
        return self.variance ** 0.5


def aggregate_by_key(
    records: Iterator[dict[str, Any]],
    key_field: str,
    value_field: str
) -> dict[str, StreamingAggregator]:
    """Aggregate values grouped by a key field."""
    aggregators: dict[str, StreamingAggregator] = {}
    
    for record in records:
        key = record.get(key_field, 'unknown')
        value = record.get(value_field)
        
        if value is not None:
            try:
                value = float(value)
                if key not in aggregators:
                    aggregators[key] = StreamingAggregator()
                aggregators[key].update(value)
            except (ValueError, TypeError):
                pass
    
    return aggregators


def compute_rolling_checksum(
    records: Iterator[dict[str, Any]]
) -> tuple[str, int]:
    """Compute checksum while streaming records."""
    hasher = hashlib.sha256()
    count = 0
    
    for record in records:
        record_bytes = json.dumps(record, sort_keys=True).encode('utf-8')
        hasher.update(record_bytes)
        count += 1
    
    return hasher.hexdigest(), count


def process_with_checkpointing(
    records: Iterator[dict[str, Any]],
    processor: Callable[[dict[str, Any]], dict[str, Any]],
    checkpoint_path: Path,
    checkpoint_interval: int = 1000
) -> Iterator[dict[str, Any]]:
    """Process records with periodic checkpointing."""
    count = 0
    last_processed: dict[str, Any] | None = None
    
    for record in records:
        try:
            result = processor(record)
            count += 1
            last_processed = result
            
            if count % checkpoint_interval == 0:
                checkpoint = {
                    'count': count,
                    'timestamp': datetime.now().isoformat(),
                    'last_record': last_processed
                }
                with open(checkpoint_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f)
                logger.info(f"Checkpoint at record {count}")
            
            yield result
            
        except Exception as e:
            logger.error(f"Error processing record {count}: {e}")
            raise


def run_tests() -> None:
    """Run basic tests for the streaming processor."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        # Create test CSV
        csv_path = test_dir / 'test.csv'
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'sensor', 'value'])
            for i in range(1000):
                writer.writerow([i, f'sensor_{i % 5}', 20.0 + i * 0.1])

        print("Testing stream_csv_file")
        count = sum(1 for _ in stream_csv_file(csv_path))
        assert count == 1000
        print("  ✓ Passed")

        print("Testing filter_records")
        records = stream_csv_file(csv_path)
        filtered = filter_records(records, lambda r: int(r['id']) < 100)
        count = sum(1 for _ in filtered)
        assert count == 100
        print("  ✓ Passed")

        print("Testing batch_records")
        records = stream_csv_file(csv_path)
        batches = list(batch_records(records, 100))
        assert len(batches) == 10
        assert all(len(b) == 100 for b in batches)
        print("  ✓ Passed")

        print("Testing aggregate_by_key")
        records = stream_csv_file(csv_path)
        aggregators = aggregate_by_key(records, 'sensor', 'value')
        assert len(aggregators) == 5
        print(f"  ✓ Aggregated {len(aggregators)} groups")

        print("Testing StreamingAggregator")
        agg = StreamingAggregator()
        for i in range(100):
            agg.update(float(i))
        assert agg.count == 100
        assert agg.mean == 49.5
        print(f"  ✓ Mean: {agg.mean}, StdDev: {agg.std_dev:.2f}")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
