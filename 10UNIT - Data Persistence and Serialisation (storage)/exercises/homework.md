# Week 10 Homework: Research Data Management System

## 📋 Metadata

| Property | Value |
|----------|-------|
| **Deadline** | Friday, 23:59 |
| **Total Points** | 100 |
| **Estimated Time** | 4–5 hours |
| **Difficulty** | ⭐⭐⭐☆☆ (3/5) |

## 🔗 Prerequisites

- [x] Completed Lab 10.1: File I/O and Serialisation
- [x] Completed Lab 10.2: Database Fundamentals
- [x] Read lecture notes on data persistence
- [x] Reviewed JSON and CSV processing examples

## 🎯 Objectives Assessed

1. **Apply** file I/O operations with proper resource management and encoding
2. **Analyse** data storage requirements to select appropriate formats
3. **Create** a normalised database schema for research data
4. **Evaluate** data integrity using checksum verification

---

## Context

You are developing a data management system for a multi-site climate research project. The system must handle sensor measurements from weather stations, store experimental metadata and ensure data integrity across distributed collection points. Your implementation will demonstrate mastery of Python's persistence mechanisms whilst addressing real-world research data challenges.

---

## Part 1: Configuration Management (20 points)

### Background

Research projects require flexible configuration that persists across sessions. Your first task implements a hierarchical configuration system using JSON serialisation.

### Requirements

Create a module `config_manager.py` with the following functionality:

```python
from pathlib import Path
from typing import Any

class ConfigManager:
    """
    Hierarchical configuration manager with file persistence.
    
    Supports nested configuration values with dot-notation access,
    automatic saving, and default value fallbacks.
    """
    
    def __init__(self, config_path: Path) -> None:
        """Initialise manager, loading existing config if present."""
        ...
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve configuration value using dot notation.
        
        Args:
            key: Dot-separated path (e.g., 'database.host')
            default: Value returned if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            >>> cfg.get('sensors.temperature.unit', 'celsius')
            'celsius'
        """
        ...
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value, creating nested structure as needed.
        
        Args:
            key: Dot-separated path
            value: Value to store (must be JSON-serialisable)
        """
        ...
    
    def save(self) -> None:
        """Persist current configuration to file."""
        ...
```

### Grading Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Dot-notation access | 6 | Correctly navigates nested dictionaries |
| Auto-creation of paths | 4 | Creates missing intermediate keys |
| JSON persistence | 5 | Proper encoding, atomic writes |
| Error handling | 3 | Graceful handling of missing files |
| Type hints and docstrings | 2 | Complete documentation |

### Test Cases

```python
def test_config_manager():
    cfg = ConfigManager(Path('test_config.json'))
    
    # Test nested set
    cfg.set('database.sqlite.path', '/data/research.db')
    cfg.set('sensors.temperature.precision', 2)
    
    # Test retrieval
    assert cfg.get('database.sqlite.path') == '/data/research.db'
    assert cfg.get('sensors.temperature.precision') == 2
    assert cfg.get('missing.key', 'default') == 'default'
    
    # Test persistence
    cfg.save()
    cfg2 = ConfigManager(Path('test_config.json'))
    assert cfg2.get('database.sqlite.path') == '/data/research.db'
```

---

## Part 2: Multi-Format Data Importer (25 points)

### Background

Research data arrives in various formats from different sources. Your importer must handle CSV, JSON and custom text formats whilst normalising data into a consistent internal representation.

### Requirements

Implement `data_importer.py` with a unified import interface:

```python
@dataclass
class Measurement:
    """Normalised measurement record."""
    station_id: str
    timestamp: datetime
    parameter: str
    value: float
    unit: str
    quality_flag: str = 'valid'

class DataImporter:
    """Multi-format research data importer."""
    
    def import_csv(
        self, 
        filepath: Path,
        column_mapping: dict[str, str] | None = None
    ) -> list[Measurement]:
        """
        Import measurements from CSV file.
        
        Handles various CSV dialects and column naming conventions.
        Missing columns raise informative errors.
        """
        ...
    
    def import_json(
        self,
        filepath: Path,
        records_path: str = 'measurements'
    ) -> list[Measurement]:
        """
        Import measurements from JSON file.
        
        Supports nested JSON structures with configurable record location.
        """
        ...
    
    def import_fixed_width(
        self,
        filepath: Path,
        field_specs: list[tuple[str, int, int]]
    ) -> list[Measurement]:
        """
        Import measurements from fixed-width text files.
        
        Args:
            field_specs: List of (field_name, start_col, end_col) tuples
        """
        ...
    
    def validate_measurements(
        self,
        measurements: list[Measurement]
    ) -> tuple[list[Measurement], list[str]]:
        """
        Validate measurements, returning valid records and error messages.
        
        Checks:
        - Timestamp validity
        - Numeric value ranges
        - Required field presence
        """
        ...
```

### Sample Input Files

**stations_data.csv**:
```csv
site,datetime,temp_c,humidity_pct,pressure_hpa
WS001,2024-06-15T10:30:00,23.5,65.2,1013.25
WS001,2024-06-15T10:45:00,24.1,64.8,1013.20
WS002,2024-06-15T10:30:00,21.8,72.1,1015.50
```

**observations.json**:
```json
{
  "source": "weather_network",
  "measurements": [
    {
      "station": "WS001",
      "time": "2024-06-15T10:30:00",
      "readings": {
        "temperature": {"value": 23.5, "unit": "celsius"},
        "humidity": {"value": 65.2, "unit": "percent"}
      }
    }
  ]
}
```

### Grading Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| CSV import with mapping | 8 | Handles column renaming and type conversion |
| JSON import with nesting | 7 | Navigates nested structures correctly |
| Fixed-width parsing | 5 | Accurate field extraction |
| Validation logic | 3 | Comprehensive error detection |
| Documentation | 2 | Clear usage examples |

---

## Part 3: Research Database (35 points)

### Background

The research project requires a structured database for efficient querying across stations, time ranges and parameters. You will design and implement a normalised schema.

### Requirements

Create `research_database.py` implementing the following schema and operations:

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│   stations   │     │  measurements   │     │  parameters  │
├──────────────┤     ├─────────────────┤     ├──────────────┤
│ station_id   │◄───┤│ measurement_id  │     │ parameter_id │
│ name         │     │ station_id (FK) │     │ name         │
│ latitude     │     │ parameter_id(FK)│────►│ unit         │
│ longitude    │     │ timestamp       │     │ min_valid    │
│ elevation    │     │ value           │     │ max_valid    │
│ installed_at │     │ quality_flag    │     └──────────────┘
└──────────────┘     └─────────────────┘

┌──────────────────┐
│  data_manifests  │
├──────────────────┤
│ manifest_id      │
│ created_at       │
│ file_count       │
│ total_records    │
│ checksum         │
└──────────────────┘
```

```python
class ResearchDatabase:
    """SQLite-based research data repository."""
    
    def __init__(self, db_path: Path) -> None:
        """Initialise database, creating schema if needed."""
        ...
    
    def register_station(
        self,
        station_id: str,
        name: str,
        latitude: float,
        longitude: float,
        elevation: float
    ) -> None:
        """Register new weather station."""
        ...
    
    def bulk_insert_measurements(
        self,
        measurements: list[Measurement]
    ) -> int:
        """
        Insert multiple measurements in single transaction.
        
        Returns:
            Number of records inserted
        """
        ...
    
    def query_time_range(
        self,
        station_id: str,
        parameter: str,
        start_time: datetime,
        end_time: datetime
    ) -> list[tuple[datetime, float]]:
        """Retrieve measurements within time range."""
        ...
    
    def get_station_statistics(
        self,
        station_id: str
    ) -> dict[str, dict[str, float]]:
        """
        Compute statistics per parameter for station.
        
        Returns:
            Dict mapping parameter name to {min, max, mean, count}
        """
        ...
    
    def export_to_csv(
        self,
        station_id: str,
        output_path: Path
    ) -> int:
        """Export station data to CSV, returning record count."""
        ...
```

### Grading Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Schema design | 10 | Proper normalisation, foreign keys, indices |
| Bulk insert efficiency | 8 | Single transaction, parameterised queries |
| Query correctness | 7 | Accurate filtering and aggregation |
| Statistics computation | 5 | Correct SQL aggregates |
| Export functionality | 3 | Proper CSV formatting |
| Documentation | 2 | Schema explanation |

---

## Part 4: Data Integrity System (20 points)

### Background

Ensuring data integrity across distributed collection sites requires systematic verification. You will implement a manifest-based integrity checking system.

### Requirements

Create `integrity_checker.py`:

```python
@dataclass
class FileManifest:
    """Manifest entry for single file."""
    relative_path: str
    size_bytes: int
    sha256_checksum: str
    record_count: int
    modified_at: datetime

class IntegrityChecker:
    """Data integrity verification system."""
    
    def generate_manifest(
        self,
        data_dir: Path,
        patterns: list[str] = ['*.csv', '*.json']
    ) -> dict[str, FileManifest]:
        """
        Generate manifest for all matching files in directory.
        
        Computes checksums and counts records per file.
        """
        ...
    
    def save_manifest(
        self,
        manifest: dict[str, FileManifest],
        output_path: Path
    ) -> None:
        """Save manifest to JSON file."""
        ...
    
    def verify_integrity(
        self,
        manifest_path: Path,
        data_dir: Path
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Verify data directory against saved manifest.
        
        Returns:
            Tuple of (verified_files, modified_files, missing_files)
        """
        ...
    
    def detect_duplicates(
        self,
        manifest: dict[str, FileManifest]
    ) -> list[list[str]]:
        """
        Find files with identical checksums.
        
        Returns:
            List of groups of duplicate file paths
        """
        ...
```

### Grading Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Checksum computation | 6 | Correct SHA-256, efficient chunked reading |
| Manifest generation | 5 | Complete file enumeration |
| Verification logic | 5 | Accurate change detection |
| Duplicate detection | 2 | Correct grouping by checksum |
| Documentation | 2 | Usage examples |

---

## Submission Requirements

### Directory Structure

```
homework_10/
├── config_manager.py
├── data_importer.py
├── research_database.py
├── integrity_checker.py
├── tests/
│   ├── test_config.py
│   ├── test_importer.py
│   ├── test_database.py
│   └── test_integrity.py
└── sample_data/
    ├── stations_data.csv
    └── observations.json
```

### Quality Checklist

- [ ] All functions have complete type hints
- [ ] All public functions have Google-style docstrings
- [ ] No `print()` statements (use `logging` module)
- [ ] All paths use `pathlib.Path`, not strings
- [ ] Tests achieve ≥80% coverage
- [ ] `ruff check` passes with no errors
- [ ] Files use UTF-8 encoding with explicit declaration

### Submission Command

```bash
# Create submission archive
zip -r homework_10_$(whoami).zip homework_10/

# Verify archive contents
unzip -l homework_10_*.zip
```

---

## Hints

<details>
<summary>💡 Hint 1: Dot-notation parsing</summary>

Split the key on dots and iterate through the nested structure:

```python
def get_nested(data: dict, keys: list[str], default: Any) -> Any:
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
```
</details>

<details>
<summary>💡 Hint 2: Bulk insert performance</summary>

Use `executemany` with a single transaction for efficient bulk inserts:

```python
with conn:
    conn.executemany(
        "INSERT INTO measurements VALUES (?, ?, ?, ?)",
        [(m.station, m.timestamp, m.value, m.unit) for m in measurements]
    )
```
</details>

<details>
<summary>💡 Hint 3: Checksum streaming</summary>

Process large files in chunks to avoid memory issues:

```python
hasher = hashlib.sha256()
with open(filepath, 'rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        hasher.update(chunk)
```
</details>

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS*
*Week 10 — Homework Assignment*

© 2025 Antonio Clim. All rights reserved.
