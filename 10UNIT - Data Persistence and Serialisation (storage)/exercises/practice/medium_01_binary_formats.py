#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Medium 01 - Binary Formats
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Binary serialisation formats offer performance advantages over text-based
formats for large datasets. This exercise explores Python's pickle module
and gzip compression for efficient data storage.

SECURITY WARNING: Pickle can execute arbitrary code during deserialisation.
Never load pickle files from untrusted sources.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Serialise Python objects using pickle with protocol selection
2. Compress data using gzip for storage efficiency
3. Compare format characteristics for informed selection

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 30 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Basic Pickle Operations
# ═══════════════════════════════════════════════════════════════════════════════

def save_object_pickle(obj: Any, filepath: Path) -> None:
    """
    Save any Python object to a pickle file.

    Uses the highest available protocol for best performance.

    Args:
        obj: Any Python object to serialise.
        filepath: Destination file path.

    Example:
        >>> data = {'model': 'v1', 'weights': [0.1, 0.2, 0.3]}
        >>> save_object_pickle(data, Path('model.pkl'))
    """
    # TODO: Implement this function
    # Hint: Use pickle.dump() with protocol=pickle.HIGHEST_PROTOCOL
    pass


def load_object_pickle(filepath: Path) -> Any:
    """
    Load a Python object from a pickle file.

    WARNING: Only load pickle files from trusted sources!

    Args:
        filepath: Path to pickle file.

    Returns:
        Deserialised Python object.

    Example:
        >>> data = load_object_pickle(Path('model.pkl'))
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Compressed Storage
# ═══════════════════════════════════════════════════════════════════════════════

def save_compressed_text(content: str, filepath: Path) -> int:
    """
    Save text content with gzip compression.

    Args:
        content: String content to compress and save.
        filepath: Destination path (should end in .gz).

    Returns:
        Size of the compressed file in bytes.

    Example:
        >>> size = save_compressed_text('x' * 10000, Path('data.txt.gz'))
        >>> print(f"Compressed to {size} bytes")
    """
    # TODO: Implement this function
    # Hint: Encode string to bytes, then use gzip.open() in 'wb' mode
    pass


def load_compressed_text(filepath: Path) -> str:
    """
    Load and decompress text content from a gzip file.

    Args:
        filepath: Path to compressed file.

    Returns:
        Decompressed string content.

    Example:
        >>> content = load_compressed_text(Path('data.txt.gz'))
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Compressed Pickle
# ═══════════════════════════════════════════════════════════════════════════════

def save_compressed_pickle(obj: Any, filepath: Path) -> int:
    """
    Save a Python object with both pickle serialisation and gzip compression.

    Combines the flexibility of pickle with the space efficiency of compression.

    Args:
        obj: Python object to serialise.
        filepath: Destination path (should end in .pkl.gz).

    Returns:
        Size of the compressed file in bytes.

    Example:
        >>> large_data = list(range(100000))
        >>> size = save_compressed_pickle(large_data, Path('data.pkl.gz'))
    """
    # TODO: Implement this function
    # Hint: Use gzip.open() with 'wb' mode and pickle.dump()
    pass


def load_compressed_pickle(filepath: Path) -> Any:
    """
    Load a compressed pickle file.

    Args:
        filepath: Path to compressed pickle file.

    Returns:
        Deserialised Python object.

    Example:
        >>> data = load_compressed_pickle(Path('data.pkl.gz'))
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Format Comparison
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FormatStats:
    """Statistics for a serialisation format."""
    format_name: str
    file_size_bytes: int
    original_size_bytes: int

    @property
    def compression_ratio(self) -> float:
        """Calculate compression ratio (higher = better compression)."""
        if self.file_size_bytes == 0:
            return 0.0
        return self.original_size_bytes / self.file_size_bytes


def compare_formats(
    data: Any,
    output_dir: Path
) -> list[FormatStats]:
    """
    Compare different serialisation formats for the same data.

    Tests: JSON, Pickle, and Compressed Pickle.
    Reports file sizes for comparison.

    Args:
        data: Data to serialise (must be JSON-compatible).
        output_dir: Directory for output files.

    Returns:
        List of FormatStats for each format tested.

    Example:
        >>> data = [{'id': i, 'value': i * 0.1} for i in range(1000)]
        >>> stats = compare_formats(data, Path('/tmp/comparison'))
        >>> for s in stats:
        ...     print(f"{s.format_name}: {s.file_size_bytes} bytes")
    """
    # TODO: Implement this function
    # Test JSON, Pickle, and Compressed Pickle
    # Return list of FormatStats
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Checkpointing System
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComputationState:
    """State of a long-running computation."""
    iteration: int
    accumulated_result: float
    intermediate_values: list[float]
    metadata: dict[str, Any]


def save_checkpoint(
    state: ComputationState,
    checkpoint_dir: Path,
    checkpoint_name: str
) -> Path:
    """
    Save computation state as a checkpoint.

    Creates timestamped checkpoint files for recovery.

    Args:
        state: Current computation state.
        checkpoint_dir: Directory for checkpoint files.
        checkpoint_name: Base name for checkpoint file.

    Returns:
        Path to the saved checkpoint file.

    Example:
        >>> state = ComputationState(100, 42.5, [1.0, 2.0], {'seed': 42})
        >>> path = save_checkpoint(state, Path('checkpoints'), 'simulation')
    """
    # TODO: Implement this function
    # Create checkpoint_dir if needed
    # Save as compressed pickle with timestamp in filename
    pass


def load_latest_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str
) -> ComputationState | None:
    """
    Load the most recent checkpoint matching the given name.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        checkpoint_name: Base name to match.

    Returns:
        Most recent checkpoint state, or None if no checkpoints found.

    Example:
        >>> state = load_latest_checkpoint(Path('checkpoints'), 'simulation')
        >>> if state:
        ...     print(f"Resuming from iteration {state.iteration}")
    """
    # TODO: Implement this function
    # Find matching checkpoint files, load the most recent
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing Exercise 1: Basic Pickle Operations")
        test_data = {'name': 'test', 'values': [1, 2, 3], 'nested': {'a': 1}}
        pkl_file = test_dir / 'test.pkl'
        save_object_pickle(test_data, pkl_file)
        loaded = load_object_pickle(pkl_file)
        assert loaded == test_data, "Pickle round-trip failed"
        print("  ✓ Pickle operations work correctly")

        print("\nTesting Exercise 2: Compressed Storage")
        text_content = "Hello, World! " * 1000
        gz_file = test_dir / 'test.txt.gz'
        compressed_size = save_compressed_text(text_content, gz_file)
        assert compressed_size < len(text_content), "Compression should reduce size"
        loaded_text = load_compressed_text(gz_file)
        assert loaded_text == text_content, "Decompression failed"
        print(f"  ✓ Compressed {len(text_content)} bytes to {compressed_size} bytes")

        print("\nTesting Exercise 3: Compressed Pickle")
        large_list = list(range(10000))
        cpkl_file = test_dir / 'test.pkl.gz'
        save_compressed_pickle(large_list, cpkl_file)
        loaded_list = load_compressed_pickle(cpkl_file)
        assert loaded_list == large_list, "Compressed pickle round-trip failed"
        print("  ✓ Compressed pickle works correctly")

        print("\nTesting Exercise 4: Format Comparison")
        comparison_data = [{'id': i, 'value': float(i)} for i in range(1000)]
        stats = compare_formats(comparison_data, test_dir / 'comparison')
        assert len(stats) >= 3, "Should test at least 3 formats"
        for s in stats:
            print(f"  {s.format_name}: {s.file_size_bytes} bytes, ratio: {s.compression_ratio:.2f}")
        print("  ✓ Format comparison works correctly")

        print("\nTesting Exercise 5: Checkpointing System")
        checkpoint_dir = test_dir / 'checkpoints'
        state1 = ComputationState(50, 25.0, [1.0, 2.0], {'seed': 42})
        save_checkpoint(state1, checkpoint_dir, 'sim')
        state2 = ComputationState(100, 50.0, [1.0, 2.0, 3.0], {'seed': 42})
        save_checkpoint(state2, checkpoint_dir, 'sim')
        
        loaded_state = load_latest_checkpoint(checkpoint_dir, 'sim')
        assert loaded_state is not None, "Should find checkpoint"
        assert loaded_state.iteration == 100, "Should load latest checkpoint"
        print("  ✓ Checkpointing system works correctly")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
