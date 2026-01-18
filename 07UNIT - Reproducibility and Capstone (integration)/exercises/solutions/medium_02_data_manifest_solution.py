#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Medium Exercise 2 — Data Manifest System
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for data integrity and manifest management exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: HASH CALCULATION — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """
    Calculate the cryptographic hash of a file.

    Reads the file in chunks to handle large files efficiently without
    loading the entire file into memory.

    Args:
        filepath: Path to the file to hash.
        algorithm: Hash algorithm to use. Defaults to "sha256".
            Supported: "md5", "sha1", "sha256", "sha512".

    Returns:
        Hexadecimal string of the file's hash.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If an unsupported algorithm is specified.

    Example:
        >>> calculate_file_hash(Path("data.csv"))
        'a3f2b8c9d4e5f6...'
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if algorithm not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    hasher = hashlib.new(algorithm)
    chunk_size = 8192  # 8 KB chunks

    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def calculate_directory_hash(
    directory: Path,
    algorithm: str = "sha256",
    pattern: str = "*"
) -> dict[str, str]:
    """
    Calculate hashes for all matching files in a directory.

    Recursively processes all files matching the pattern and returns
    a dictionary mapping relative paths to their hashes.

    Args:
        directory: Path to the directory to process.
        algorithm: Hash algorithm to use. Defaults to "sha256".
        pattern: Glob pattern for file matching. Defaults to "*" (all files).

    Returns:
        Dictionary mapping relative file paths (as strings) to their hashes.

    Raises:
        NotADirectoryError: If the path is not a directory.

    Example:
        >>> calculate_directory_hash(Path("data/"))
        {'file1.csv': 'abc123...', 'subdir/file2.json': 'def456...'}
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    hashes: dict[str, str] = {}

    for filepath in directory.rglob(pattern):
        if filepath.is_file():
            relative_path = filepath.relative_to(directory)
            hashes[str(relative_path)] = calculate_file_hash(filepath, algorithm)

    return hashes


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: DATA MANIFEST CLASS — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FileEntry:
    """
    Entry for a single file in the manifest.

    Attributes:
        path: Relative path to the file.
        hash: Cryptographic hash of the file contents.
        size: File size in bytes.
        modified: Last modification timestamp in ISO format.
    """

    path: str
    hash: str
    size: int
    modified: str


@dataclass
class DataManifest:
    """
    A manifest tracking files and their integrity information.

    Attributes:
        name: Name identifier for this manifest.
        created: Creation timestamp in ISO format.
        algorithm: Hash algorithm used for all entries.
        entries: Dictionary mapping paths to FileEntry objects.
    """

    name: str
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    algorithm: str = "sha256"
    entries: dict[str, FileEntry] = field(default_factory=dict)

    def add_file(self, filepath: Path, base_dir: Path | None = None) -> FileEntry:
        """
        Add a file to the manifest.

        Args:
            filepath: Path to the file to add.
            base_dir: Optional base directory for relative path calculation.
                If None, uses the file's parent directory.

        Returns:
            The created FileEntry.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if base_dir is None:
            base_dir = filepath.parent
            relative_path = filepath.name
        else:
            relative_path = str(filepath.relative_to(base_dir))

        stat = filepath.stat()
        entry = FileEntry(
            path=relative_path,
            hash=calculate_file_hash(filepath, self.algorithm),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime).isoformat()
        )

        self.entries[relative_path] = entry
        return entry

    def add_directory(
        self,
        directory: Path,
        pattern: str = "*"
    ) -> list[FileEntry]:
        """
        Add all matching files from a directory to the manifest.

        Args:
            directory: Path to the directory.
            pattern: Glob pattern for file matching.

        Returns:
            List of created FileEntry objects.
        """
        added: list[FileEntry] = []

        for filepath in directory.rglob(pattern):
            if filepath.is_file():
                entry = self.add_file(filepath, base_dir=directory)
                added.append(entry)

        return added

    def verify_file(self, filepath: Path, base_dir: Path | None = None) -> bool:
        """
        Verify a file against its manifest entry.

        Args:
            filepath: Path to the file to verify.
            base_dir: Base directory for relative path calculation.

        Returns:
            True if the file matches its manifest entry, False otherwise.
        """
        if base_dir is None:
            relative_path = filepath.name
        else:
            relative_path = str(filepath.relative_to(base_dir))

        if relative_path not in self.entries:
            return False

        if not filepath.exists():
            return False

        current_hash = calculate_file_hash(filepath, self.algorithm)
        return current_hash == self.entries[relative_path].hash

    def verify_all(self, base_dir: Path) -> dict[str, bool]:
        """
        Verify all files in the manifest.

        Args:
            base_dir: Base directory containing the files.

        Returns:
            Dictionary mapping paths to verification results.
        """
        results: dict[str, bool] = {}

        for path in self.entries:
            filepath = base_dir / path
            results[path] = self.verify_file(filepath, base_dir)

        return results

    def to_json(self) -> str:
        """
        Serialise the manifest to JSON format.

        Returns:
            JSON string representation of the manifest.
        """
        data = {
            "name": self.name,
            "created": self.created,
            "algorithm": self.algorithm,
            "entries": {
                path: {
                    "path": entry.path,
                    "hash": entry.hash,
                    "size": entry.size,
                    "modified": entry.modified
                }
                for path, entry in self.entries.items()
            }
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_string: str) -> DataManifest:
        """
        Create a manifest from JSON string.

        Args:
            json_string: JSON representation of a manifest.

        Returns:
            Reconstructed DataManifest instance.
        """
        data = json.loads(json_string)
        manifest = cls(
            name=data["name"],
            created=data["created"],
            algorithm=data["algorithm"]
        )

        for path, entry_data in data["entries"].items():
            manifest.entries[path] = FileEntry(
                path=entry_data["path"],
                hash=entry_data["hash"],
                size=entry_data["size"],
                modified=entry_data["modified"]
            )

        return manifest

    def save(self, filepath: Path) -> None:
        """
        Save the manifest to a file.

        Args:
            filepath: Path where to save the manifest.
        """
        with open(filepath, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, filepath: Path) -> DataManifest:
        """
        Load a manifest from a file.

        Args:
            filepath: Path to the manifest file.

        Returns:
            Loaded DataManifest instance.
        """
        with open(filepath) as f:
            return cls.from_json(f.read())


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: WORKFLOW FUNCTIONS — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


def create_manifest_for_experiment(
    experiment_dir: Path,
    experiment_name: str,
    patterns: list[str] | None = None
) -> DataManifest:
    """
    Create a complete manifest for an experiment directory.

    Args:
        experiment_dir: Path to the experiment directory.
        experiment_name: Name for the manifest.
        patterns: List of glob patterns to include. Defaults to common data files.

    Returns:
        A DataManifest containing all matching files.

    Example:
        >>> manifest = create_manifest_for_experiment(
        ...     Path("experiments/exp-001"),
        ...     "Experiment 001"
        ... )
    """
    if patterns is None:
        patterns = ["*.csv", "*.json", "*.npy", "*.pkl", "*.txt"]

    manifest = DataManifest(name=experiment_name)

    for pattern in patterns:
        for filepath in experiment_dir.rglob(pattern):
            if filepath.is_file():
                manifest.add_file(filepath, base_dir=experiment_dir)

    return manifest


def verify_experiment_data(
    experiment_dir: Path,
    manifest: DataManifest
) -> tuple[bool, dict[str, Any]]:
    """
    Verify all experiment data against a manifest.

    Args:
        experiment_dir: Path to the experiment directory.
        manifest: The manifest to verify against.

    Returns:
        Tuple of (all_valid, details) where details contains:
        - verified: List of paths that passed verification
        - failed: List of paths that failed verification
        - missing: List of paths in manifest but not on disk
        - extra: List of paths on disk but not in manifest
    """
    results = manifest.verify_all(experiment_dir)

    verified = [p for p, v in results.items() if v]
    failed = [p for p, v in results.items() if not v]

    # Check for missing files
    missing = []
    for path in manifest.entries:
        if not (experiment_dir / path).exists():
            missing.append(path)
            if path in failed:
                failed.remove(path)

    # Check for extra files (not in manifest)
    extra = []
    for filepath in experiment_dir.rglob("*"):
        if filepath.is_file():
            relative = str(filepath.relative_to(experiment_dir))
            if relative not in manifest.entries:
                extra.append(relative)

    all_valid = len(failed) == 0 and len(missing) == 0

    return all_valid, {
        "verified": verified,
        "failed": failed,
        "missing": missing,
        "extra": extra
    }


def find_modified_files(
    experiment_dir: Path,
    manifest: DataManifest
) -> list[dict[str, Any]]:
    """
    Find files that have been modified since the manifest was created.

    Args:
        experiment_dir: Path to the experiment directory.
        manifest: The manifest to compare against.

    Returns:
        List of dictionaries containing modification details:
        - path: Relative path to the modified file
        - original_hash: Hash from manifest
        - current_hash: Current hash of file
        - size_changed: Boolean indicating if size changed
    """
    modified: list[dict[str, Any]] = []

    for path, entry in manifest.entries.items():
        filepath = experiment_dir / path

        if not filepath.exists():
            continue

        current_hash = calculate_file_hash(filepath, manifest.algorithm)

        if current_hash != entry.hash:
            current_size = filepath.stat().st_size
            modified.append({
                "path": path,
                "original_hash": entry.hash,
                "current_hash": current_hash,
                "size_changed": current_size != entry.size
            })

    return modified


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION AND TESTING
# ═══════════════════════════════════════════════════════════════════════════════


def run_tests() -> None:
    """Run all validation tests for the exercises."""
    print("=" * 70)
    print("SOLUTION VALIDATION: Medium Exercise 2 — Data Manifest")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir)

        # Create test files
        (test_dir / "data").mkdir()
        file1 = test_dir / "data" / "test1.csv"
        file2 = test_dir / "data" / "test2.json"

        file1.write_text("a,b,c\n1,2,3\n4,5,6\n")
        file2.write_text('{"values": [1, 2, 3]}')

        # Test Exercise 1: Hash calculation
        print("\n--- Exercise 1: Hash Calculation ---")
        hash1 = calculate_file_hash(file1)
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars
        print(f"✓ File hash calculated: {hash1[:16]}...")

        hashes = calculate_directory_hash(test_dir / "data")
        assert "test1.csv" in hashes
        assert "test2.json" in hashes
        print("✓ Directory hashes calculated")

        # Test Exercise 2: DataManifest class
        print("\n--- Exercise 2: DataManifest Class ---")
        manifest = DataManifest(name="test_manifest")
        manifest.add_file(file1, base_dir=test_dir / "data")
        manifest.add_file(file2, base_dir=test_dir / "data")

        assert len(manifest.entries) == 2
        print("✓ Files added to manifest")

        # Test serialisation
        json_str = manifest.to_json()
        loaded = DataManifest.from_json(json_str)
        assert loaded.name == manifest.name
        assert len(loaded.entries) == 2
        print("✓ JSON serialisation works")

        # Test verification
        assert manifest.verify_file(file1, base_dir=test_dir / "data")
        print("✓ File verification works")

        results = manifest.verify_all(test_dir / "data")
        assert all(results.values())
        print("✓ Bulk verification works")

        # Test Exercise 3: Workflow functions
        print("\n--- Exercise 3: Workflow Functions ---")
        exp_manifest = create_manifest_for_experiment(
            test_dir / "data",
            "Test Experiment",
            patterns=["*.csv", "*.json"]
        )
        assert len(exp_manifest.entries) == 2
        print("✓ Experiment manifest created")

        is_valid, details = verify_experiment_data(test_dir / "data", exp_manifest)
        assert is_valid
        assert len(details["verified"]) == 2
        print("✓ Experiment verification works")

        # Modify a file
        file1.write_text("modified content\n")
        modified = find_modified_files(test_dir / "data", exp_manifest)
        assert len(modified) == 1
        assert modified[0]["path"] == "test1.csv"
        print("✓ Modified file detection works")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
