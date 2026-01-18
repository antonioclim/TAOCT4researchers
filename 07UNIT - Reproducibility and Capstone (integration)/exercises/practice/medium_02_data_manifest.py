#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Medium 02 - Data Manifests and Integrity
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Data integrity is essential for reproducible research. This exercise focuses on
creating data manifests that track file hashes, enabling verification that
data has not been corrupted or modified between experiments.

PREREQUISITES
─────────────
- Understanding of file I/O in Python
- Basic knowledge of hashing concepts
- Familiarity with JSON serialisation

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Calculate cryptographic hashes of files
2. Create and validate data manifests
3. Detect data corruption or modification

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 45 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import hashlib
import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: File Hashing
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_file_hash(
    filepath: Path,
    algorithm: str = "sha256",
    chunk_size: int = 8192,
) -> str:
    """
    Calculate the cryptographic hash of a file.

    This function should read the file in chunks to handle large files
    efficiently without loading the entire file into memory.

    Args:
        filepath: Path to the file to hash.
        algorithm: Hash algorithm to use (sha256, md5, sha1).
        chunk_size: Size of chunks to read at a time.

    Returns:
        Hexadecimal string representation of the hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the algorithm is not supported.

    Example:
        >>> # Create a test file
        >>> Path("test.txt").write_text("Hello, World!")
        >>> calculate_file_hash(Path("test.txt"))
        'dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f'
    """
    # TODO: Implement this function
    # Hint: Use hashlib.new(algorithm) to create the hash object
    # Read file in chunks and update the hash with each chunk
    pass


def calculate_directory_hash(
    directory: Path,
    algorithm: str = "sha256",
    pattern: str = "*",
) -> dict[str, str]:
    """
    Calculate hashes for all files matching pattern in a directory.

    Args:
        directory: Path to the directory.
        algorithm: Hash algorithm to use.
        pattern: Glob pattern for files to include.

    Returns:
        Dictionary mapping relative file paths to their hashes.

    Example:
        >>> hashes = calculate_directory_hash(Path("data"), pattern="*.csv")
        >>> hashes
        {'file1.csv': 'abc123...', 'subdir/file2.csv': 'def456...'}
    """
    # TODO: Implement this function
    # Use directory.rglob(pattern) to find files recursively
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Data Manifest Class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileEntry:
    """Entry for a single file in the manifest."""

    path: str
    hash: str
    size: int
    modified: str  # ISO format timestamp


@dataclass
class DataManifest:
    """
    Manifest tracking files and their hashes for reproducibility.

    TODO: Implement the methods below.

    Attributes:
        version: Manifest format version.
        created: Creation timestamp (ISO format).
        algorithm: Hash algorithm used.
        entries: Dictionary of path -> FileEntry.
    """

    version: str = "1.0"
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    algorithm: str = "sha256"
    entries: dict[str, FileEntry] = field(default_factory=dict)

    def add_file(self, filepath: Path, base_dir: Path | None = None) -> None:
        """
        Add a file to the manifest.

        Args:
            filepath: Path to the file to add.
            base_dir: Base directory for relative paths (optional).

        Raises:
            FileNotFoundError: If file does not exist.

        Example:
            >>> manifest = DataManifest()
            >>> manifest.add_file(Path("data/input.csv"), Path("data"))
            >>> "input.csv" in manifest.entries
            True
        """
        # TODO: Implement this method
        # 1. Calculate the file hash
        # 2. Get file size and modification time
        # 3. Create FileEntry and add to entries
        pass

    def add_directory(
        self,
        directory: Path,
        pattern: str = "*",
    ) -> int:
        """
        Add all files matching pattern in directory.

        Args:
            directory: Directory to scan.
            pattern: Glob pattern for files.

        Returns:
            Number of files added.

        Example:
            >>> manifest = DataManifest()
            >>> count = manifest.add_directory(Path("data"), "*.csv")
            >>> count > 0
            True
        """
        # TODO: Implement this method
        pass

    def verify_file(self, filepath: Path, base_dir: Path | None = None) -> bool:
        """
        Verify a file against the manifest.

        Args:
            filepath: Path to the file to verify.
            base_dir: Base directory for relative paths.

        Returns:
            True if file matches manifest, False otherwise.

        Raises:
            KeyError: If file is not in manifest.
        """
        # TODO: Implement this method
        pass

    def verify_all(self, base_dir: Path) -> dict[str, bool]:
        """
        Verify all files in the manifest.

        Args:
            base_dir: Base directory containing the files.

        Returns:
            Dictionary mapping paths to verification results.
        """
        # TODO: Implement this method
        pass

    def to_json(self) -> str:
        """
        Serialise manifest to JSON string.

        Returns:
            JSON string representation of the manifest.
        """
        # TODO: Implement this method
        # Convert entries to serialisable format
        pass

    @classmethod
    def from_json(cls, json_str: str) -> "DataManifest":
        """
        Create manifest from JSON string.

        Args:
            json_str: JSON string to parse.

        Returns:
            DataManifest instance.
        """
        # TODO: Implement this method
        pass

    def save(self, filepath: Path) -> None:
        """
        Save manifest to file.

        Args:
            filepath: Path to save the manifest.
        """
        # TODO: Implement this method
        pass

    @classmethod
    def load(cls, filepath: Path) -> "DataManifest":
        """
        Load manifest from file.

        Args:
            filepath: Path to the manifest file.

        Returns:
            DataManifest instance.
        """
        # TODO: Implement this method
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Integrity Checking Workflow
# ═══════════════════════════════════════════════════════════════════════════════

def create_manifest_for_experiment(
    data_dir: Path,
    output_path: Path,
    patterns: list[str] | None = None,
) -> DataManifest:
    """
    Create a manifest for an experiment's data directory.

    This function should:
    1. Create a new DataManifest
    2. Add all files matching the patterns (default: all files)
    3. Save the manifest to output_path
    4. Return the manifest

    Args:
        data_dir: Directory containing experiment data.
        output_path: Where to save the manifest.
        patterns: List of glob patterns (default: ["*"]).

    Returns:
        The created DataManifest.

    Example:
        >>> manifest = create_manifest_for_experiment(
        ...     Path("experiment_001/data"),
        ...     Path("experiment_001/MANIFEST.json"),
        ...     patterns=["*.csv", "*.npy"]
        ... )
    """
    # TODO: Implement this function
    pass


def verify_experiment_data(
    data_dir: Path,
    manifest_path: Path,
) -> tuple[bool, list[str]]:
    """
    Verify experiment data against its manifest.

    Args:
        data_dir: Directory containing experiment data.
        manifest_path: Path to the manifest file.

    Returns:
        Tuple of (all_valid, list_of_invalid_files).

    Example:
        >>> valid, invalid = verify_experiment_data(
        ...     Path("experiment_001/data"),
        ...     Path("experiment_001/MANIFEST.json")
        ... )
        >>> if not valid:
        ...     print(f"Invalid files: {invalid}")
    """
    # TODO: Implement this function
    pass


def find_modified_files(
    data_dir: Path,
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """
    Find files that have been modified since manifest creation.

    Args:
        data_dir: Directory containing experiment data.
        manifest_path: Path to the manifest file.

    Returns:
        Dictionary with modification details:
        {
            "path": {
                "expected_hash": "...",
                "actual_hash": "...",
                "expected_size": 1234,
                "actual_size": 5678,
            }
        }
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run tests for all exercises."""
    import tempfile

    print("Testing file hashing...")

    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create test files
        (tmppath / "file1.txt").write_text("Hello, World!")
        (tmppath / "file2.txt").write_text("Test data for hashing")
        (tmppath / "subdir").mkdir()
        (tmppath / "subdir" / "file3.txt").write_text("Nested file")

        # Test Exercise 1
        print("\n--- Exercise 1: File Hashing ---")
        hash1 = calculate_file_hash(tmppath / "file1.txt")
        print(f"  Hash of 'Hello, World!': {hash1[:16]}...")
        assert len(hash1) == 64, "SHA-256 hash should be 64 characters"

        dir_hashes = calculate_directory_hash(tmppath, pattern="*.txt")
        print(f"  Found {len(dir_hashes)} files in directory")

        # Test Exercise 2
        print("\n--- Exercise 2: DataManifest ---")
        manifest = DataManifest()
        manifest.add_file(tmppath / "file1.txt", tmppath)
        print(f"  Added file to manifest: {list(manifest.entries.keys())}")

        count = manifest.add_directory(tmppath, "*.txt")
        print(f"  Added {count} files from directory")

        # Test serialisation
        json_str = manifest.to_json()
        loaded = DataManifest.from_json(json_str)
        print(f"  Serialisation test: {len(loaded.entries)} entries")

        # Test verification
        is_valid = manifest.verify_file(tmppath / "file1.txt", tmppath)
        print(f"  Verification (unchanged): {is_valid}")

        # Modify file and test again
        (tmppath / "file1.txt").write_text("Modified content!")
        is_valid = manifest.verify_file(tmppath / "file1.txt", tmppath)
        print(f"  Verification (modified): {is_valid}")

        # Test Exercise 3
        print("\n--- Exercise 3: Workflow ---")
        manifest_path = tmppath / "MANIFEST.json"
        manifest = create_manifest_for_experiment(
            tmppath,
            manifest_path,
            patterns=["*.txt"],
        )
        print(f"  Created manifest with {len(manifest.entries)} entries")

        valid, invalid = verify_experiment_data(tmppath, manifest_path)
        print(f"  Verification result: valid={valid}, invalid={invalid}")

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
