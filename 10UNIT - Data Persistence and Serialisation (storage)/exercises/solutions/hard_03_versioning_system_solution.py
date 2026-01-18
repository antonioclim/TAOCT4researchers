#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Hard 03 - Data Versioning System
═══════════════════════════════════════════════════════════════════════════════
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


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


@dataclass
class VerificationResult:
    """Result of integrity verification."""
    is_valid: bool
    verified_files: list[str]
    modified_files: list[str]
    missing_files: list[str]
    new_files: list[str]
    error_message: str = ''


@dataclass
class FileChange:
    """Description of a file change."""
    path: str
    change_type: str
    old_checksum: str | None = None
    new_checksum: str | None = None
    old_size: int | None = None
    new_size: int | None = None


def compute_file_checksum(
    filepath: Path,
    algorithm: str = 'sha256',
    chunk_size: int = 65536
) -> str:
    """Compute cryptographic checksum of a file."""
    hasher = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_string_checksum(content: str, algorithm: str = 'sha256') -> str:
    """Compute checksum of a string."""
    hasher = hashlib.new(algorithm)
    hasher.update(content.encode('utf-8'))
    return hasher.hexdigest()


def generate_manifest(
    directory: Path,
    patterns: list[str] | None = None,
    metadata: dict[str, Any] | None = None
) -> DataManifest:
    """Generate a manifest for all files in a directory."""
    files: list[FileEntry] = []
    total_size = 0
    
    if patterns:
        all_files: set[Path] = set()
        for pattern in patterns:
            all_files.update(directory.rglob(pattern))
        file_paths = sorted(all_files)
    else:
        file_paths = sorted(f for f in directory.rglob('*') if f.is_file())
    
    for filepath in file_paths:
        if filepath.is_file():
            stat = filepath.stat()
            checksum = compute_file_checksum(filepath)
            relative = str(filepath.relative_to(directory))
            
            entry = FileEntry(
                relative_path=relative,
                size_bytes=stat.st_size,
                checksum=checksum,
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
            )
            files.append(entry)
            total_size += stat.st_size
    
    return DataManifest(
        manifest_id=str(uuid.uuid4()),
        created_at=datetime.now().isoformat(),
        root_directory=str(directory.absolute()),
        total_files=len(files),
        total_size_bytes=total_size,
        files=files,
        metadata=metadata or {}
    )


def save_manifest(manifest: DataManifest, filepath: Path) -> None:
    """Save manifest to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(manifest.to_dict(), f, indent=2)


def load_manifest(filepath: Path) -> DataManifest:
    """Load manifest from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return DataManifest.from_dict(data)


def verify_integrity(
    manifest: DataManifest,
    directory: Path
) -> VerificationResult:
    """Verify directory contents against a manifest."""
    verified: list[str] = []
    modified: list[str] = []
    missing: list[str] = []
    
    manifest_files = {f.relative_path: f for f in manifest.files}
    
    for rel_path, entry in manifest_files.items():
        filepath = directory / rel_path
        
        if not filepath.exists():
            missing.append(rel_path)
        else:
            current_checksum = compute_file_checksum(filepath)
            if current_checksum == entry.checksum:
                verified.append(rel_path)
            else:
                modified.append(rel_path)
    
    # Find new files
    new_files: list[str] = []
    for filepath in directory.rglob('*'):
        if filepath.is_file():
            rel_path = str(filepath.relative_to(directory))
            if rel_path not in manifest_files:
                new_files.append(rel_path)
    
    is_valid = len(modified) == 0 and len(missing) == 0
    
    return VerificationResult(
        is_valid=is_valid,
        verified_files=verified,
        modified_files=modified,
        missing_files=missing,
        new_files=new_files,
        error_message='' if is_valid else 'Data integrity check failed'
    )


def compare_manifests(
    old_manifest: DataManifest,
    new_manifest: DataManifest
) -> list[FileChange]:
    """Compare two manifests to detect changes."""
    changes: list[FileChange] = []
    
    old_files = {f.relative_path: f for f in old_manifest.files}
    new_files = {f.relative_path: f for f in new_manifest.files}
    
    all_paths = set(old_files.keys()) | set(new_files.keys())
    
    for path in sorted(all_paths):
        old_entry = old_files.get(path)
        new_entry = new_files.get(path)
        
        if old_entry is None:
            # Added
            changes.append(FileChange(
                path=path,
                change_type='added',
                new_checksum=new_entry.checksum,
                new_size=new_entry.size_bytes
            ))
        elif new_entry is None:
            # Removed
            changes.append(FileChange(
                path=path,
                change_type='removed',
                old_checksum=old_entry.checksum,
                old_size=old_entry.size_bytes
            ))
        elif old_entry.checksum != new_entry.checksum:
            # Modified
            changes.append(FileChange(
                path=path,
                change_type='modified',
                old_checksum=old_entry.checksum,
                new_checksum=new_entry.checksum,
                old_size=old_entry.size_bytes,
                new_size=new_entry.size_bytes
            ))
        else:
            # Unchanged
            changes.append(FileChange(
                path=path,
                change_type='unchanged',
                old_checksum=old_entry.checksum,
                new_checksum=new_entry.checksum
            ))
    
    return changes


def generate_change_report(changes: list[FileChange]) -> str:
    """Generate a human-readable change report."""
    lines = ["=== Data Change Report ===", ""]
    
    added = [c for c in changes if c.change_type == 'added']
    removed = [c for c in changes if c.change_type == 'removed']
    modified = [c for c in changes if c.change_type == 'modified']
    unchanged = [c for c in changes if c.change_type == 'unchanged']
    
    lines.append(f"Added: {len(added)} files")
    for c in added:
        lines.append(f"  + {c.path}")
    
    lines.append(f"\nRemoved: {len(removed)} files")
    for c in removed:
        lines.append(f"  - {c.path}")
    
    lines.append(f"\nModified: {len(modified)} files")
    for c in modified:
        lines.append(f"  ~ {c.path}")
    
    lines.append(f"\nUnchanged: {len(unchanged)} files")
    
    return '\n'.join(lines)


class DataVersionManager:
    """Manages version history for a data directory."""

    def __init__(self, data_dir: Path, versions_dir: Path) -> None:
        """Initialise version manager."""
        self.data_dir = data_dir
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    def create_version(
        self,
        version_tag: str,
        metadata: dict[str, Any] | None = None
    ) -> DataManifest:
        """Create a new version snapshot."""
        meta = metadata or {}
        meta['version_tag'] = version_tag
        
        manifest = generate_manifest(self.data_dir, metadata=meta)
        manifest_path = self.versions_dir / f"{version_tag}.json"
        save_manifest(manifest, manifest_path)
        
        return manifest

    def list_versions(self) -> list[dict[str, Any]]:
        """List all available versions."""
        versions = []
        
        for manifest_path in sorted(self.versions_dir.glob('*.json')):
            manifest = load_manifest(manifest_path)
            versions.append({
                'tag': manifest.metadata.get('version_tag', manifest_path.stem),
                'date': manifest.created_at,
                'file_count': manifest.total_files
            })
        
        return versions

    def get_version(self, version_tag: str) -> DataManifest | None:
        """Retrieve a specific version's manifest."""
        manifest_path = self.versions_dir / f"{version_tag}.json"
        if manifest_path.exists():
            return load_manifest(manifest_path)
        return None

    def verify_version(self, version_tag: str) -> VerificationResult:
        """Verify current data against a specific version."""
        manifest = self.get_version(version_tag)
        if manifest is None:
            return VerificationResult(
                is_valid=False,
                verified_files=[],
                modified_files=[],
                missing_files=[],
                new_files=[],
                error_message=f"Version {version_tag} not found"
            )
        
        return verify_integrity(manifest, self.data_dir)

    def compare_versions(
        self,
        old_tag: str,
        new_tag: str
    ) -> list[FileChange]:
        """Compare two versions."""
        old_manifest = self.get_version(old_tag)
        new_manifest = self.get_version(new_tag)
        
        if old_manifest is None or new_manifest is None:
            return []
        
        return compare_manifests(old_manifest, new_manifest)


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

        print("Testing file checksums")
        checksum1 = compute_file_checksum(data_dir / 'file1.csv')
        checksum2 = compute_file_checksum(data_dir / 'file1.csv')
        assert checksum1 == checksum2
        assert len(checksum1) == 64
        print("  ✓ Checksums work correctly")

        print("Testing manifest generation")
        manifest = generate_manifest(data_dir)
        assert manifest.total_files == 2
        print(f"  ✓ Generated manifest with {manifest.total_files} files")

        manifest_path = test_dir / 'manifest.json'
        save_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        assert loaded.total_files == manifest.total_files
        print("  ✓ Manifest save/load works")

        print("Testing integrity verification")
        result = verify_integrity(manifest, data_dir)
        assert result.is_valid
        print("  ✓ Integrity verification passes")

        # Modify a file
        (data_dir / 'file1.csv').write_text('modified', encoding='utf-8')
        result2 = verify_integrity(manifest, data_dir)
        assert not result2.is_valid
        print("  ✓ Modification detected")

        # Restore
        (data_dir / 'file1.csv').write_text('a,b,c\n1,2,3\n', encoding='utf-8')

        print("Testing change detection")
        (data_dir / 'file3.txt').write_text('new file', encoding='utf-8')
        manifest2 = generate_manifest(data_dir)
        changes = compare_manifests(manifest, manifest2)
        added = [c for c in changes if c.change_type == 'added']
        assert len(added) == 1
        print("  ✓ Change detection works")

        print("Testing version management")
        (data_dir / 'file3.txt').unlink()
        vm = DataVersionManager(data_dir, versions_dir)
        vm.create_version('v1.0')
        versions = vm.list_versions()
        assert len(versions) == 1
        print("  ✓ Version creation works")

        (data_dir / 'new_data.csv').write_text('x,y\n1,2\n', encoding='utf-8')
        vm.create_version('v1.1')
        versions = vm.list_versions()
        assert len(versions) == 2
        print("  ✓ Multiple versions supported")

        changes = vm.compare_versions('v1.0', 'v1.1')
        assert len([c for c in changes if c.change_type == 'added']) == 1
        print("  ✓ Version comparison works")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
