#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Unit 10: Tests for Lab 10.02 - Database Fundamentals
═══════════════════════════════════════════════════════════════════════════════

Test suite providing ≥80% coverage for lab_10_02_database_fundamentals.py

Run with:
    pytest tests/test_lab_10_02.py -v --cov=lab --cov-report=term-missing

═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Import module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'lab'))

from lab_10_02_database_fundamentals import (
    database_connection,
    create_research_schema,
    Experiment,
    Sensor,
    Measurement,
    insert_experiment,
    insert_sensor,
    bulk_insert_measurements,
    get_experiment_by_id,
    update_experiment_status,
    delete_experiment,
    get_table_info,
    get_foreign_keys,
    get_indices,
    query_measurements_by_time_range,
    compute_statistics_by_sensor,
    find_anomalous_measurements,
    generate_daily_summary,
    DataManifest,
    compute_data_checksum,
    create_manifest,
    verify_data_integrity,
    save_manifest_to_file,
    load_manifest_from_file,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_db():
    """Provide a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / 'test.db'
        with database_connection(db_path) as conn:
            create_research_schema(conn)
            yield conn, Path(tmpdir)


@pytest.fixture
def populated_db(temp_db):
    """Provide a database with sample data."""
    conn, tmpdir = temp_db
    
    # Insert experiment
    exp = Experiment(
        name='Test Experiment',
        researcher='Dr. Test',
        start_date='2024-01-15',
        description='A test experiment'
    )
    exp_id = insert_experiment(conn, exp)
    
    # Insert sensors
    sensor1 = Sensor(
        serial_number='TEMP-001',
        sensor_type='temperature',
        unit='celsius',
        location='Lab A'
    )
    sensor2 = Sensor(
        serial_number='HUMID-001',
        sensor_type='humidity',
        unit='percent',
        location='Lab A'
    )
    sensor1_id = insert_sensor(conn, sensor1)
    sensor2_id = insert_sensor(conn, sensor2)
    
    # Insert measurements
    measurements = []
    for hour in range(24):
        measurements.append(Measurement(
            experiment_id=exp_id,
            sensor_id=sensor1_id,
            timestamp=f'2024-01-15T{hour:02d}:00:00',
            value=20.0 + hour * 0.5
        ))
        measurements.append(Measurement(
            experiment_id=exp_id,
            sensor_id=sensor2_id,
            timestamp=f'2024-01-15T{hour:02d}:00:00',
            value=50.0 + hour * 0.2
        ))
    
    bulk_insert_measurements(conn, measurements)
    
    return conn, tmpdir, exp_id, sensor1_id, sensor2_id


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSchemaCreation:
    """Tests for database schema creation."""

    def test_creates_experiments_table(self, temp_db):
        """Test that experiments table is created."""
        conn, _ = temp_db
        info = get_table_info(conn, 'experiments')
        column_names = [col['name'] for col in info]
        
        assert 'experiment_id' in column_names
        assert 'name' in column_names
        assert 'researcher' in column_names

    def test_creates_sensors_table(self, temp_db):
        """Test that sensors table is created."""
        conn, _ = temp_db
        info = get_table_info(conn, 'sensors')
        column_names = [col['name'] for col in info]
        
        assert 'sensor_id' in column_names
        assert 'serial_number' in column_names
        assert 'sensor_type' in column_names

    def test_creates_measurements_table(self, temp_db):
        """Test that measurements table is created."""
        conn, _ = temp_db
        info = get_table_info(conn, 'measurements')
        column_names = [col['name'] for col in info]
        
        assert 'measurement_id' in column_names
        assert 'experiment_id' in column_names
        assert 'sensor_id' in column_names

    def test_foreign_keys_defined(self, temp_db):
        """Test that foreign keys are properly defined."""
        conn, _ = temp_db
        fks = get_foreign_keys(conn, 'measurements')
        
        tables = [fk['table'] for fk in fks]
        assert 'experiments' in tables
        assert 'sensors' in tables

    def test_indices_created(self, temp_db):
        """Test that indices are created."""
        conn, _ = temp_db
        indices = get_indices(conn, 'measurements')
        
        assert len(indices) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CRUD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCRUDOperations:
    """Tests for CRUD operations."""

    def test_insert_experiment(self, temp_db):
        """Test inserting an experiment."""
        conn, _ = temp_db
        exp = Experiment(
            name='New Experiment',
            researcher='Dr. Smith',
            start_date='2024-02-01'
        )
        
        exp_id = insert_experiment(conn, exp)
        
        assert exp_id is not None
        assert exp_id > 0

    def test_get_experiment_by_id(self, temp_db):
        """Test retrieving an experiment by ID."""
        conn, _ = temp_db
        exp = Experiment(
            name='Retrievable Experiment',
            researcher='Dr. Jones',
            start_date='2024-03-01'
        )
        exp_id = insert_experiment(conn, exp)
        
        loaded = get_experiment_by_id(conn, exp_id)
        
        assert loaded is not None
        assert loaded.name == 'Retrievable Experiment'
        assert loaded.researcher == 'Dr. Jones'

    def test_get_nonexistent_experiment(self, temp_db):
        """Test retrieving a non-existent experiment returns None."""
        conn, _ = temp_db
        loaded = get_experiment_by_id(conn, 99999)
        assert loaded is None

    def test_update_experiment_status(self, temp_db):
        """Test updating experiment status."""
        conn, _ = temp_db
        exp = Experiment(
            name='Status Test',
            researcher='Dr. Test',
            start_date='2024-01-01'
        )
        exp_id = insert_experiment(conn, exp)
        
        result = update_experiment_status(conn, exp_id, 'completed', '2024-01-31')
        
        assert result is True
        loaded = get_experiment_by_id(conn, exp_id)
        assert loaded.status == 'completed'

    def test_delete_experiment(self, temp_db):
        """Test deleting an experiment."""
        conn, _ = temp_db
        exp = Experiment(
            name='Deletable',
            researcher='Dr. Test',
            start_date='2024-01-01'
        )
        exp_id = insert_experiment(conn, exp)
        
        result = delete_experiment(conn, exp_id)
        
        assert result is True
        assert get_experiment_by_id(conn, exp_id) is None

    def test_bulk_insert_measurements(self, temp_db):
        """Test bulk inserting measurements."""
        conn, _ = temp_db
        
        exp_id = insert_experiment(conn, Experiment(
            name='Bulk Test', researcher='Dr. Test', start_date='2024-01-01'
        ))
        sensor_id = insert_sensor(conn, Sensor(
            serial_number='BULK-001', sensor_type='test', unit='units'
        ))
        
        measurements = [
            Measurement(experiment_id=exp_id, sensor_id=sensor_id,
                       timestamp=f'2024-01-01T{i:02d}:00:00', value=float(i))
            for i in range(10)
        ]
        
        count = bulk_insert_measurements(conn, measurements)
        
        assert count == 10


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdvancedQueries:
    """Tests for advanced query functions."""

    def test_query_by_time_range(self, populated_db):
        """Test querying measurements by time range."""
        conn, _, exp_id, _, _ = populated_db
        
        results = query_measurements_by_time_range(
            conn, exp_id,
            '2024-01-15T06:00:00',
            '2024-01-15T12:00:00'
        )
        
        assert len(results) > 0
        for r in results:
            assert r['timestamp'] >= '2024-01-15T06:00:00'
            assert r['timestamp'] <= '2024-01-15T12:00:00'

    def test_compute_statistics(self, populated_db):
        """Test computing statistics by sensor."""
        conn, _, exp_id, _, _ = populated_db
        
        stats = compute_statistics_by_sensor(conn, exp_id)
        
        assert len(stats) == 2  # Two sensors
        for s in stats:
            assert 'measurement_count' in s
            assert 'min_value' in s
            assert 'max_value' in s
            assert 'mean_value' in s

    def test_generate_daily_summary(self, populated_db):
        """Test generating daily summary."""
        conn, _, exp_id, _, _ = populated_db
        
        summary = generate_daily_summary(conn, exp_id)
        
        assert len(summary) == 1  # One day of data
        assert summary[0]['date'] == '2024-01-15'
        assert summary[0]['measurement_count'] == 48  # 24 hours × 2 sensors


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INTEGRITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataIntegrity:
    """Tests for data integrity verification."""

    def test_compute_checksum_deterministic(self, populated_db):
        """Test that checksums are deterministic."""
        conn, _, exp_id, _, _ = populated_db
        
        checksum1 = compute_data_checksum(conn, exp_id)
        checksum2 = compute_data_checksum(conn, exp_id)
        
        assert checksum1 == checksum2

    def test_create_manifest(self, populated_db):
        """Test creating a data manifest."""
        conn, _, exp_id, _, _ = populated_db
        
        manifest = create_manifest(conn, exp_id)
        
        assert manifest.experiment_id == exp_id
        assert manifest.record_count == 48
        assert len(manifest.checksum) == 64

    def test_verify_integrity_valid(self, populated_db):
        """Test integrity verification on unmodified data."""
        conn, _, exp_id, _, _ = populated_db
        
        manifest = create_manifest(conn, exp_id)
        is_valid, message = verify_data_integrity(conn, manifest)
        
        assert is_valid is True

    def test_verify_integrity_after_modification(self, populated_db):
        """Test integrity verification after data modification."""
        conn, _, exp_id, sensor_id, _ = populated_db
        
        manifest = create_manifest(conn, exp_id)
        
        # Modify data
        conn.execute(
            "UPDATE measurements SET value = 999.0 WHERE measurement_id = 1"
        )
        conn.commit()
        
        is_valid, message = verify_data_integrity(conn, manifest)
        
        assert is_valid is False
        assert 'checksum' in message.lower() or 'modified' in message.lower()

    def test_manifest_save_load(self, populated_db):
        """Test saving and loading manifests."""
        conn, tmpdir, exp_id, _, _ = populated_db
        
        manifest = create_manifest(conn, exp_id)
        filepath = tmpdir / 'manifest.json'
        
        save_manifest_to_file(manifest, filepath)
        loaded = load_manifest_from_file(filepath)
        
        assert loaded.manifest_id == manifest.manifest_id
        assert loaded.checksum == manifest.checksum
        assert loaded.record_count == manifest.record_count


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_bulk_insert(self, temp_db):
        """Test bulk insert with empty list."""
        conn, _ = temp_db
        count = bulk_insert_measurements(conn, [])
        assert count == 0

    def test_delete_nonexistent_experiment(self, temp_db):
        """Test deleting non-existent experiment returns False."""
        conn, _ = temp_db
        result = delete_experiment(conn, 99999)
        assert result is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
