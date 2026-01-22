#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Hard 03 - Data Versioning System
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Reproducible research requires tracking changes to datasets over time.
This exercise implements a complete data versioning system using
cryptographic checksums, manifest files, and change detection.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Compute cryptographic checksums for file integrity verification
2. Generate and validate data manifests
3. Detect and report changes between dataset versions
4. Implement a simple version control system for research data

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 45 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: File Checksums
# ═══════════════════════════════════════════════════════════════════════════════

def compute_file_checksum(
    filepath: Path,
    algorithm: str = 'sha256',
    chunk_size: int = 65536
) -> str:
    """
    Compute cryptographic checksum of a file.

    Must process the file in chunks for memory efficiency.

    Args:
        filepath: Path to the file.
        algorithm: Hash algorithm (sha256, sha512, md5).
        chunk_size: Bytes to read at a time.

    Returns:
        Hexadecimal checksum string.

    Example:
        >>> checksum = compute_file_checksum(Path('data.csv'))
        >>> len(checksum)  # SHA-256 produces 64 hex characters
        64
    """
    # TODO: Implement this function
    # Hint: Use hashlib.new(algorithm) and update in chunks
    pass


def compute_string_checksum(content: str, algorithm: str = 'sha256') -> str:
    """
    Compute checksum of a string.

    Args:
        content: String to hash.
        algorithm: Hash algorithm.

    Returns:
        Hexadecimal checksum string.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Manifest Generation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileEntry:
    """Entry for a single file in the manifest."""
    relative_path: str
    size_bytes: int
    checksum: str
    modified_at: str


@dataclass
class DataManifest:
    """Complete data manifest for a directory."""
    manifest_id: str
    created_at: str
    root_directory: str
    total_files: int
    total_size_bytes: int
    files: list[FileEntry] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialisation."""
        return {
            'manifest_id': self.manifest_id,
            'created_at': self.created_at,
            'root_directory': self.root_directory,
            'total_files': self.total_files,
            'total_size_bytes': self.total_size_bytes,
            'files': [asdict(f) for f in self.files],
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'DataManifest':
        """Create from dictionary."""
        files = [FileEntry(**f) for f in data.get('files', [])]
        return cls(
            manifest_id=data['manifest_id'],
            created_at=data['created_at'],
            root_directory=data['root_directory'],
            total_files=data['total_files'],
            total_size_bytes=data['total_size_bytes'],
            files=files,
            metadata=data.get('metadata', {}),
        )


def generate_manifest(
    directory: Path,
    patterns: list[str] | None = None,
    metadata: dict[str, Any] | None = None
) -> DataManifest:
    """
    Generate a manifest for all files in a directory.

    Args:
        directory: Root directory to scan.
        patterns: Glob patterns to include (e.g., ['*.csv', '*.json']).
                  If None, includes all files.
        metadata: Optional additional metadata to include.

    Returns:
        DataManifest with all file entries.

    Example:
        >>> manifest = generate_manifest(Path('data/'), ['*.csv'])
        >>> manifest.total_files
        5
    """
    # TODO: Implement this function
    # Hint: Use directory.rglob() to find files
    pass


def save_manifest(manifest: DataManifest, filepath: Path) -> None:
    """
    Save manifest to JSON file.

    Args:
        manifest: Manifest to save.
        filepath: Destination path.
    """
    # TODO: Implement this function
    pass


def load_manifest(filepath: Path) -> DataManifest:
    """
    Load manifest from JSON file.

    Args:
        filepath: Path to manifest file.

    Returns:
        Loaded DataManifest.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Integrity Verification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationResult:
    """Result of integrity verification."""
    is_valid: bool
    verified_files: list[str]
    modified_files: list[str]
    missing_files: list[str]
    new_files: list[str]
    error_message: str = ''


def verify_integrity(
    manifest: DataManifest,
    directory: Path
) -> VerificationResult:
    """
    Verify directory contents against a manifest.

    Checks:
    - All manifest files exist
    - Checksums match
    - Detects new files not in manifest

    Args:
        manifest: Reference manifest.
        directory: Directory to verify.

    Returns:
        VerificationResult with detailed comparison.

    Example:
        >>> result = verify_integrity(manifest, Path('data/'))
        >>> result.is_valid
        True
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Change Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FileChange:
    """Description of a file change."""
    path: str
    change_type: str  # 'added', 'removed', 'modified', 'unchanged'
    old_checksum: str | None = None
    new_checksum: str | None = None
    old_size: int | None = None
    new_size: int | None = None


def compare_manifests(
    old_manifest: DataManifest,
    new_manifest: DataManifest
) -> list[FileChange]:
    """
    Compare two manifests to detect changes.

    Args:
        old_manifest: Previous version manifest.
        new_manifest: Current version manifest.

    Returns:
        List of FileChange objects describing all changes.

    Example:
        >>> changes = compare_manifests(v1_manifest, v2_manifest)
        >>> added = [c for c in changes if c.change_type == 'added']
    """
    # TODO: Implement this function
    pass


def generate_change_report(changes: list[FileChange]) -> str:
    """
    Generate a human-readable change report.

    Args:
        changes: List of file changes.

    Returns:
        Formatted report string.

    Example:
        >>> report = generate_change_report(changes)
        >>> print(report)
        === Data Change Report ===
        Added: 2 files
        - data/new_file.csv
        ...
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Version History
# ═══════════════════════════════════════════════════════════════════════════════

class DataVersionManager:
    """
    Manages version history for a data directory.

    Stores manifests and enables rollback verification.
    """

    def __init__(self, data_dir: Path, versions_dir: Path) -> None:
        """
        Initialise version manager.

        Args:
            data_dir: Directory containing data files.
            versions_dir: Directory to store version manifests.
        """
        self.data_dir = data_dir
        self.versions_dir = versions_dir
        # TODO: Create versions_dir if needed

    def create_version(
        self,
        version_tag: str,
        metadata: dict[str, Any] | None = None
    ) -> DataManifest:
        """
        Create a new version snapshot.

        Args:
            version_tag: Human-readable version identifier (e.g., 'v1.0').
            metadata: Optional metadata for this version.

        Returns:
            Created manifest.
        """
        # TODO: Implement this function
        pass

    def list_versions(self) -> list[dict[str, Any]]:
        """
        List all available versions.

        Returns:
            List of version info dictionaries with tag, date, file_count.
        """
        # TODO: Implement this function
        pass

    def get_version(self, version_tag: str) -> DataManifest | None:
        """
        Retrieve a specific version's manifest.

        Args:
            version_tag: Version identifier.

        Returns:
            Manifest or None if not found.
        """
        # TODO: Implement this function
        pass

    def verify_version(self, version_tag: str) -> VerificationResult:
        """
        Verify current data against a specific version.

        Args:
            version_tag: Version to verify against.

        Returns:
            Verification result.
        """
        # TODO: Implement this function
        pass

    def compare_versions(
        self,
        old_tag: str,
        new_tag: str
    ) -> list[FileChange]:
        """
        Compare two versions.

        Args:
            old_tag: Older version tag.
            new_tag: Newer version tag.

        Returns:
            List of changes between versions.
        """
        # TODO: Implement this function
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run comprehensive tests for the versioning system."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        data_dir = test_dir / 'data'
        versions_dir = test_dir / 'versions'
        data_dir.mkdir()

        # Create test files
        (data_dir / 'file1.csv').write_text('a,b,c\n1,2,3\n', encoding='utf-8')
        (data_dir / 'file2.json').write_text('{"key": "value"}', encoding='utf-8')

        print("Testing Exercise 1: File checksums")
        checksum1 = compute_file_checksum(data_dir / 'file1.csv')
        assert len(checksum1) == 64, "SHA-256 should produce 64 hex chars"
        checksum2 = compute_file_checksum(data_dir / 'file1.csv')
        assert checksum1 == checksum2, "Same file should produce same checksum"
        print("  ✓ File checksums work correctly")

        print("\nTesting Exercise 2: Manifest generation")
        manifest = generate_manifest(data_dir, ['*.csv', '*.json'])
        assert manifest.total_files == 2
        assert manifest.total_size_bytes > 0
        print(f"  ✓ Generated manifest with {manifest.total_files} files")

        manifest_path = test_dir / 'manifest.json'
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        assert loaded.total_files == manifest.total_files
        print("  ✓ Manifest save/load works correctly")

        print("\nTesting Exercise 3: Integrity verification")
        result = verify_integrity(manifest, data_dir)
        assert result.is_valid, "Unmodified directory should verify"
        assert len(result.verified_files) == 2
        print("  ✓ Integrity verification passes for unmodified data")

        # Modify a file
        (data_dir / 'file1.csv').write_text('modified content', encoding='utf-8')
        result2 = verify_integrity(manifest, data_dir)
        assert not result2.is_valid, "Modified file should fail verification"
        assert 'file1.csv' in str(result2.modified_files)
        print("  ✓ Modification detection works correctly")

        # Restore original
        (data_dir / 'file1.csv').write_text('a,b,c\n1,2,3\n', encoding='utf-8')

        print("\nTesting Exercise 4: Change detection")
        # Create v2 with changes
        (data_dir / 'file3.txt').write_text('new file', encoding='utf-8')
        manifest2 = generate_manifest(data_dir)
        manifest2.manifest_id = 'v2'

        changes = compare_manifests(manifest, manifest2)
        added = [c for c in changes if c.change_type == 'added']
        assert len(added) == 1, "Should detect 1 added file"
        print("  ✓ Change detection works correctly")

        report = generate_change_report(changes)
        assert 'Added' in report or 'added' in report.lower()
        print("  ✓ Change report generation works")

        print("\nTesting Exercise 5: Version management")
        # Reset data
        (data_dir / 'file3.txt').unlink()

        vm = DataVersionManager(data_dir, versions_dir)
        vm.create_version('v1.0', {'description': 'Initial version'})
        versions = vm.list_versions()
        assert len(versions) == 1
        print("  ✓ Version creation works")

        # Add file and create new version
        (data_dir / 'new_data.csv').write_text('x,y\n1,2\n', encoding='utf-8')
        vm.create_version('v1.1', {'description': 'Added new_data.csv'})
        versions = vm.list_versions()
        assert len(versions) == 2
        print("  ✓ Multiple versions supported")

        changes = vm.compare_versions('v1.0', 'v1.1')
        assert len([c for c in changes if c.change_type == 'added']) == 1
        print("  ✓ Version comparison works")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
