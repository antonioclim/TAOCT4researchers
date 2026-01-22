#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Medium 01 - Binary Formats
═══════════════════════════════════════════════════════════════════════════════
"""

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentState:
    """Sample complex object for serialisation testing."""
    name: str
    iteration: int
    weights: list[float]
    metadata: dict[str, Any]


@dataclass
class CompressionResult:
    """Result of compression analysis."""
    original_size: int
    compressed_size: int
    compression_ratio: float


def write_binary_file(filepath: Path, data: bytes) -> int:
    """Write binary data to a file."""
    with open(filepath, 'wb') as f:
        return f.write(data)


def read_binary_file(filepath: Path) -> bytes:
    """Read binary data from a file."""
    with open(filepath, 'rb') as f:
        return f.read()


def save_object_pickle(obj: Any, filepath: Path) -> None:
    """Save any Python object to a Pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_object_pickle(filepath: Path) -> Any:
    """Load a Python object from a Pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_compressed(data: bytes, filepath: Path, level: int = 9) -> int:
    """Save binary data with gzip compression."""
    with gzip.open(filepath, 'wb', compresslevel=level) as f:
        f.write(data)
    return filepath.stat().st_size


def load_compressed(filepath: Path) -> bytes:
    """Load and decompress gzip-compressed data."""
    with gzip.open(filepath, 'rb') as f:
        return f.read()


def save_json_compressed(data: dict[str, Any], filepath: Path) -> int:
    """Save JSON data with gzip compression."""
    json_bytes = json.dumps(data).encode('utf-8')
    return save_compressed(json_bytes, filepath)


def load_json_compressed(filepath: Path) -> dict[str, Any]:
    """Load compressed JSON data."""
    data_bytes = load_compressed(filepath)
    return json.loads(data_bytes.decode('utf-8'))


def analyse_compression(data: bytes) -> CompressionResult:
    """Analyse how well data compresses with gzip."""
    compressed = gzip.compress(data, compresslevel=9)
    original_size = len(data)
    compressed_size = len(compressed)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    return CompressionResult(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=ratio
    )


def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing binary file operations")
        bin_file = test_dir / 'test.bin'
        test_bytes = b'\x00\x01\x02\x03\xff'
        written = write_binary_file(bin_file, test_bytes)
        assert written == 5
        read_data = read_binary_file(bin_file)
        assert read_data == test_bytes
        print("  ✓ Passed")

        print("Testing Pickle serialisation")
        pkl_file = test_dir / 'state.pkl'
        state = ExperimentState('test', 42, [0.1, 0.2, 0.3], {'key': 'value'})
        save_object_pickle(state, pkl_file)
        loaded_state = load_object_pickle(pkl_file)
        assert loaded_state.name == 'test'
        assert loaded_state.iteration == 42
        print("  ✓ Passed")

        print("Testing Gzip compression")
        gz_file = test_dir / 'data.gz'
        original = b'Hello World! ' * 1000
        compressed_size = save_compressed(original, gz_file)
        assert compressed_size < len(original)
        decompressed = load_compressed(gz_file)
        assert decompressed == original
        print(f"  ✓ Compressed {len(original)} -> {compressed_size} bytes")

        print("Testing Compressed JSON")
        json_gz = test_dir / 'data.json.gz'
        test_data = {'numbers': list(range(1000)), 'name': 'test'}
        save_json_compressed(test_data, json_gz)
        loaded_data = load_json_compressed(json_gz)
        assert loaded_data == test_data
        print("  ✓ Passed")

        print("Testing Compression analysis")
        compressible = b'A' * 10000
        result = analyse_compression(compressible)
        assert result.original_size == 10000
        assert result.compression_ratio > 10
        print(f"  ✓ Compression ratio: {result.compression_ratio:.1f}x")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
