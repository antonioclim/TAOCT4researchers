#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Medium 02 - SQLite CRUD Operations
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Database systems epitomize this pattern, maintaining indices to accelerate
queries at the cost of increased storage and update overhead."
— Garcia-Molina et al., 2008

This exercise introduces fundamental database operations: Create, Read,
Update, and Delete (CRUD) using SQLite and Python's sqlite3 module.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Create database tables with appropriate constraints
2. Insert, query, update, and delete records safely
3. Use parameterised queries to prevent SQL injection

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 35 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Database Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def create_experiments_table(conn: sqlite3.Connection) -> None:
    """
    Create the experiments table with appropriate schema.

    Schema:
        - experiment_id: INTEGER PRIMARY KEY AUTOINCREMENT
        - name: TEXT NOT NULL UNIQUE
        - researcher: TEXT NOT NULL
        - start_date: TEXT NOT NULL
        - status: TEXT DEFAULT 'active'

    Args:
        conn: Active database connection.

    Example:
        >>> conn = sqlite3.connect(':memory:')
        >>> create_experiments_table(conn)
    """
    # TODO: Implement this function
    # Hint: Use conn.execute() with CREATE TABLE IF NOT EXISTS
    pass


def create_measurements_table(conn: sqlite3.Connection) -> None:
    """
    Create the measurements table with foreign key constraint.

    Schema:
        - measurement_id: INTEGER PRIMARY KEY AUTOINCREMENT
        - experiment_id: INTEGER NOT NULL (FK to experiments)
        - timestamp: TEXT NOT NULL
        - value: REAL NOT NULL
        - sensor_name: TEXT NOT NULL

    Args:
        conn: Active database connection.

    Example:
        >>> create_measurements_table(conn)
    """
    # TODO: Implement this function
    # Include FOREIGN KEY constraint
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Create Operations (INSERT)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Experiment:
    """Experiment record."""
    name: str
    researcher: str
    start_date: str
    status: str = 'active'
    experiment_id: int | None = None


def insert_experiment(conn: sqlite3.Connection, experiment: Experiment) -> int:
    """
    Insert a new experiment record.

    Must use parameterised queries for safety.

    Args:
        conn: Database connection.
        experiment: Experiment data to insert.

    Returns:
        ID of the newly inserted experiment.

    Example:
        >>> exp = Experiment('Climate Study', 'Dr. Smith', '2024-01-15')
        >>> exp_id = insert_experiment(conn, exp)
    """
    # TODO: Implement this function
    # Use parameterised query (?, ?, ?) syntax
    pass


def insert_measurement(
    conn: sqlite3.Connection,
    experiment_id: int,
    timestamp: str,
    value: float,
    sensor_name: str
) -> int:
    """
    Insert a measurement record.

    Args:
        conn: Database connection.
        experiment_id: ID of the parent experiment.
        timestamp: Measurement timestamp (ISO format).
        value: Measured value.
        sensor_name: Name of the sensor.

    Returns:
        ID of the newly inserted measurement.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Read Operations (SELECT)
# ═══════════════════════════════════════════════════════════════════════════════

def get_experiment_by_id(
    conn: sqlite3.Connection,
    experiment_id: int
) -> Experiment | None:
    """
    Retrieve an experiment by its ID.

    Args:
        conn: Database connection.
        experiment_id: ID to look up.

    Returns:
        Experiment object or None if not found.

    Example:
        >>> exp = get_experiment_by_id(conn, 1)
        >>> if exp:
        ...     print(exp.name)
    """
    # TODO: Implement this function
    pass


def get_all_experiments(conn: sqlite3.Connection) -> list[Experiment]:
    """
    Retrieve all experiments from the database.

    Returns experiments ordered by start_date descending.

    Args:
        conn: Database connection.

    Returns:
        List of all experiments.
    """
    # TODO: Implement this function
    pass


def get_measurements_for_experiment(
    conn: sqlite3.Connection,
    experiment_id: int
) -> list[dict[str, Any]]:
    """
    Retrieve all measurements for a specific experiment.

    Args:
        conn: Database connection.
        experiment_id: Experiment ID to query.

    Returns:
        List of measurement dictionaries.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Update Operations
# ═══════════════════════════════════════════════════════════════════════════════

def update_experiment_status(
    conn: sqlite3.Connection,
    experiment_id: int,
    new_status: str
) -> bool:
    """
    Update the status of an experiment.

    Args:
        conn: Database connection.
        experiment_id: ID of experiment to update.
        new_status: New status value.

    Returns:
        True if update succeeded, False if experiment not found.

    Example:
        >>> success = update_experiment_status(conn, 1, 'completed')
    """
    # TODO: Implement this function
    # Check rowcount to determine if update succeeded
    pass


def update_measurement_value(
    conn: sqlite3.Connection,
    measurement_id: int,
    new_value: float
) -> bool:
    """
    Correct a measurement value.

    Args:
        conn: Database connection.
        measurement_id: ID of measurement to update.
        new_value: Corrected value.

    Returns:
        True if update succeeded.
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Delete Operations
# ═══════════════════════════════════════════════════════════════════════════════

def delete_experiment(conn: sqlite3.Connection, experiment_id: int) -> bool:
    """
    Delete an experiment and all its measurements.

    Uses cascading delete through foreign key constraint, or
    manually deletes measurements first.

    Args:
        conn: Database connection.
        experiment_id: ID of experiment to delete.

    Returns:
        True if deletion succeeded.

    Example:
        >>> success = delete_experiment(conn, 1)
    """
    # TODO: Implement this function
    # Delete measurements first, then experiment
    pass


def delete_measurements_before(
    conn: sqlite3.Connection,
    experiment_id: int,
    cutoff_timestamp: str
) -> int:
    """
    Delete measurements older than a cutoff timestamp.

    Args:
        conn: Database connection.
        experiment_id: Experiment to clean up.
        cutoff_timestamp: Delete measurements before this time.

    Returns:
        Number of measurements deleted.

    Example:
        >>> count = delete_measurements_before(conn, 1, '2024-01-01T00:00:00')
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""

    # Use in-memory database for testing
    conn = sqlite3.connect(':memory:')
    conn.execute("PRAGMA foreign_keys = ON")

    print("Testing Exercise 1: Database Initialisation")
    create_experiments_table(conn)
    create_measurements_table(conn)
    # Verify tables exist
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert 'experiments' in tables, "experiments table not created"
    assert 'measurements' in tables, "measurements table not created"
    print("  ✓ Tables created successfully")

    print("\nTesting Exercise 2: Create Operations")
    exp1 = Experiment('Climate Study', 'Dr. Smith', '2024-01-15')
    exp_id = insert_experiment(conn, exp1)
    assert exp_id == 1, f"Expected ID 1, got {exp_id}"

    m_id = insert_measurement(conn, exp_id, '2024-01-15T10:00:00', 23.5, 'TEMP-01')
    assert m_id == 1, "Measurement insert failed"
    print("  ✓ Insert operations work correctly")

    print("\nTesting Exercise 3: Read Operations")
    exp = get_experiment_by_id(conn, exp_id)
    assert exp is not None, "Should find experiment"
    assert exp.name == 'Climate Study', "Name mismatch"

    all_exps = get_all_experiments(conn)
    assert len(all_exps) == 1, "Should have 1 experiment"

    measurements = get_measurements_for_experiment(conn, exp_id)
    assert len(measurements) == 1, "Should have 1 measurement"
    print("  ✓ Read operations work correctly")

    print("\nTesting Exercise 4: Update Operations")
    success = update_experiment_status(conn, exp_id, 'completed')
    assert success, "Update should succeed"
    exp = get_experiment_by_id(conn, exp_id)
    assert exp.status == 'completed', "Status not updated"

    success = update_measurement_value(conn, m_id, 24.0)
    assert success, "Measurement update should succeed"
    print("  ✓ Update operations work correctly")

    print("\nTesting Exercise 5: Delete Operations")
    # Add more measurements for deletion test
    insert_measurement(conn, exp_id, '2023-12-01T10:00:00', 22.0, 'TEMP-01')
    insert_measurement(conn, exp_id, '2024-02-01T10:00:00', 25.0, 'TEMP-01')

    deleted = delete_measurements_before(conn, exp_id, '2024-01-01T00:00:00')
    assert deleted == 1, f"Should delete 1 old measurement, deleted {deleted}"

    success = delete_experiment(conn, exp_id)
    assert success, "Delete should succeed"
    assert get_experiment_by_id(conn, exp_id) is None, "Experiment should be deleted"
    print("  ✓ Delete operations work correctly")

    conn.close()

    print("\n" + "=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
