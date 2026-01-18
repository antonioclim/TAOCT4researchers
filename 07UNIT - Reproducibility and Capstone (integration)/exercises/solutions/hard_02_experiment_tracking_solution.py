#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Hard Exercise 2 — Experiment Tracking System
═══════════════════════════════════════════════════════════════════════════════

This solution implements a comprehensive experiment tracking system including:
1. SQLite-based experiment storage
2. Context manager for automatic run tracking
3. Experiment comparison and reporting

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import random
import shutil
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RunStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for an experiment.

    Attributes:
        name: Name of the experiment.
        description: Optional description.
        tags: List of tags for categorisation.
        metadata: Additional metadata dictionary.
    """

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class MetricEntry:
    """A single metric measurement.

    Attributes:
        name: Name of the metric.
        value: Numeric value.
        step: Optional step/epoch number.
        timestamp: When the metric was recorded.
    """

    name: str
    value: float
    step: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Artifact:
    """An artifact associated with a run.

    Attributes:
        name: Name of the artifact.
        path: Path to the artifact file.
        artifact_type: Type of artifact (e.g. "model", "data", "plot").
        hash: SHA-256 hash of the artifact content.
        size_bytes: Size of the artifact in bytes.
    """

    name: str
    path: Path
    artifact_type: str = "file"
    hash: str = ""
    size_bytes: int = 0


@dataclass
class ExperimentRun:
    """Represents a single experiment run.

    Attributes:
        run_id: Unique identifier for the run.
        experiment_id: ID of the parent experiment.
        status: Current status of the run.
        parameters: Hyperparameters for this run.
        metrics: Dictionary of metric name to list of entries.
        artifacts: List of artifacts.
        start_time: When the run started.
        end_time: When the run ended.
        error_message: Error message if run failed.
    """

    run_id: str
    experiment_id: str
    status: RunStatus = RunStatus.PENDING
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[MetricEntry]] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_message: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        """Calculate run duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: EXPERIMENT STORE WITH SQLITE
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentStore:
    """SQLite-based storage for experiments and runs.

    This class provides persistent storage for experiment metadata,
    run information, metrics and artifacts using SQLite.

    Attributes:
        db_path: Path to the SQLite database file.

    Example:
        >>> store = ExperimentStore(Path("experiments.db"))
        >>> exp_id = store.create_experiment(ExperimentConfig(name="test"))
        >>> run_id = store.start_run(exp_id, {"learning_rate": 0.001})
        >>> store.log_metric(run_id, "loss", 0.5, step=1)
        >>> store.end_run(run_id, RunStatus.COMPLETED)
    """

    def __init__(self, db_path: Path) -> None:
        """Initialise the experiment store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialise the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Experiments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            # Runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parameters TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    error_message TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            # Metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    step INTEGER,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            # Artifacts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    artifact_type TEXT,
                    hash TEXT,
                    size_bytes INTEGER,
                    FOREIGN KEY (run_id) REFERENCES runs(run_id)
                )
            """)

            # Tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                )
            """)

            # Create indices for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_runs_experiment
                ON runs(experiment_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_run
                ON metrics(run_id, name)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with context management.

        Yields:
            SQLite connection object.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _generate_id(self, prefix: str = "") -> str:
        """Generate a unique identifier.

        Args:
            prefix: Optional prefix for the ID.

        Returns:
            Unique identifier string.
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        random_part = hashlib.sha256(
            f"{timestamp}{random.random()}".encode()
        ).hexdigest()[:8]
        return f"{prefix}{timestamp}_{random_part}" if prefix else f"{timestamp}_{random_part}"

    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment.

        Args:
            config: Configuration for the experiment.

        Returns:
            Unique experiment ID.
        """
        experiment_id = self._generate_id("exp_")
        created_at = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO experiments (experiment_id, name, description, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    config.name,
                    config.description,
                    created_at,
                    json.dumps(config.metadata),
                ),
            )

            # Insert tags
            for tag in config.tags:
                cursor.execute(
                    "INSERT INTO tags (experiment_id, tag) VALUES (?, ?)",
                    (experiment_id, tag),
                )

            conn.commit()

        logger.info(f"Created experiment '{config.name}' with ID: {experiment_id}")
        return experiment_id

    def start_run(
        self,
        experiment_id: str,
        parameters: dict[str, Any] | None = None,
    ) -> str:
        """Start a new run within an experiment.

        Args:
            experiment_id: ID of the parent experiment.
            parameters: Hyperparameters for this run.

        Returns:
            Unique run ID.
        """
        run_id = self._generate_id("run_")
        start_time = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (run_id, experiment_id, status, parameters, start_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    experiment_id,
                    RunStatus.RUNNING.value,
                    json.dumps(parameters or {}),
                    start_time,
                ),
            )
            conn.commit()

        logger.info(f"Started run {run_id} for experiment {experiment_id}")
        return run_id

    def end_run(
        self,
        run_id: str,
        status: RunStatus,
        error_message: str | None = None,
    ) -> None:
        """End a run with the given status.

        Args:
            run_id: ID of the run to end.
            status: Final status of the run.
            error_message: Error message if run failed.
        """
        end_time = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE runs
                SET status = ?, end_time = ?, error_message = ?
                WHERE run_id = ?
                """,
                (status.value, end_time, error_message, run_id),
            )
            conn.commit()

        logger.info(f"Ended run {run_id} with status: {status.value}")

    def log_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a metric value for a run.

        Args:
            run_id: ID of the run.
            name: Name of the metric.
            value: Numeric value.
            step: Optional step/epoch number.
        """
        timestamp = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO metrics (run_id, name, value, step, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, name, value, step, timestamp),
            )
            conn.commit()

    def log_artifact(
        self,
        run_id: str,
        name: str,
        path: Path,
        artifact_type: str = "file",
    ) -> None:
        """Log an artifact for a run.

        Args:
            run_id: ID of the run.
            name: Name of the artifact.
            path: Path to the artifact file.
            artifact_type: Type of artifact.
        """
        # Calculate hash and size
        file_hash = ""
        size_bytes = 0
        if path.exists():
            size_bytes = path.stat().st_size
            with path.open("rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO artifacts (run_id, name, path, artifact_type, hash, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (run_id, name, str(path), artifact_type, file_hash, size_bytes),
            )
            conn.commit()

        logger.debug(f"Logged artifact '{name}' for run {run_id}")

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Retrieve a run by ID.

        Args:
            run_id: ID of the run to retrieve.

        Returns:
            ExperimentRun object or None if not found.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get run data
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None

            # Get metrics
            cursor.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY step, timestamp",
                (run_id,),
            )
            metrics_rows = cursor.fetchall()

            metrics: dict[str, list[MetricEntry]] = {}
            for m in metrics_rows:
                entry = MetricEntry(
                    name=m["name"],
                    value=m["value"],
                    step=m["step"],
                    timestamp=datetime.fromisoformat(m["timestamp"]),
                )
                if m["name"] not in metrics:
                    metrics[m["name"]] = []
                metrics[m["name"]].append(entry)

            # Get artifacts
            cursor.execute("SELECT * FROM artifacts WHERE run_id = ?", (run_id,))
            artifact_rows = cursor.fetchall()
            artifacts = [
                Artifact(
                    name=a["name"],
                    path=Path(a["path"]),
                    artifact_type=a["artifact_type"],
                    hash=a["hash"],
                    size_bytes=a["size_bytes"],
                )
                for a in artifact_rows
            ]

            return ExperimentRun(
                run_id=row["run_id"],
                experiment_id=row["experiment_id"],
                status=RunStatus(row["status"]),
                parameters=json.loads(row["parameters"]) if row["parameters"] else {},
                metrics=metrics,
                artifacts=artifacts,
                start_time=datetime.fromisoformat(row["start_time"]) if row["start_time"] else None,
                end_time=datetime.fromisoformat(row["end_time"]) if row["end_time"] else None,
                error_message=row["error_message"],
            )

    def list_runs(
        self,
        experiment_id: str,
        status: RunStatus | None = None,
    ) -> list[ExperimentRun]:
        """List all runs for an experiment.

        Args:
            experiment_id: ID of the experiment.
            status: Optional status filter.

        Returns:
            List of ExperimentRun objects.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute(
                    "SELECT run_id FROM runs WHERE experiment_id = ? AND status = ?",
                    (experiment_id, status.value),
                )
            else:
                cursor.execute(
                    "SELECT run_id FROM runs WHERE experiment_id = ?",
                    (experiment_id,),
                )

            rows = cursor.fetchall()

        runs = []
        for row in rows:
            run = self.get_run(row["run_id"])
            if run:
                runs.append(run)

        return runs

    def get_metrics_history(
        self,
        run_id: str,
        metric_name: str,
    ) -> list[tuple[int | None, float]]:
        """Get the history of a metric for a run.

        Args:
            run_id: ID of the run.
            metric_name: Name of the metric.

        Returns:
            List of (step, value) tuples.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT step, value FROM metrics
                WHERE run_id = ? AND name = ?
                ORDER BY step, timestamp
                """,
                (run_id, metric_name),
            )
            return [(row["step"], row["value"]) for row in cursor.fetchall()]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: EXPERIMENT TRACKER WITH CONTEXT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentTracker:
    """High-level API for tracking experiments with context management.

    Provides a convenient interface for starting runs, logging metrics
    and handling run lifecycle automatically.

    Attributes:
        store: Underlying experiment store.
        experiment_id: Current experiment ID.
        active_run_id: Currently active run ID.

    Example:
        >>> tracker = ExperimentTracker(Path("experiments.db"))
        >>> tracker.set_experiment("my_experiment")
        >>> with tracker.start_run({"lr": 0.001}) as run_id:
        ...     tracker.log_metric("loss", 0.5, step=1)
        ...     tracker.log_metric("accuracy", 0.8, step=1)
    """

    def __init__(self, db_path: Path) -> None:
        """Initialise the experiment tracker.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.store = ExperimentStore(db_path)
        self.experiment_id: str | None = None
        self.active_run_id: str | None = None

    def set_experiment(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """Set or create the current experiment.

        Args:
            name: Name of the experiment.
            description: Optional description.
            tags: Optional list of tags.

        Returns:
            Experiment ID.
        """
        config = ExperimentConfig(
            name=name,
            description=description,
            tags=tags or [],
        )
        self.experiment_id = self.store.create_experiment(config)
        return self.experiment_id

    @contextmanager
    def start_run(
        self,
        parameters: dict[str, Any] | None = None,
    ) -> Generator[str, None, None]:
        """Start a new run as a context manager.

        Automatically handles run lifecycle, setting status to COMPLETED
        on normal exit or FAILED on exception.

        Args:
            parameters: Hyperparameters for this run.

        Yields:
            Run ID.

        Raises:
            ValueError: If no experiment is set.
        """
        if self.experiment_id is None:
            raise ValueError("No experiment set. Call set_experiment() first.")

        run_id = self.store.start_run(self.experiment_id, parameters)
        self.active_run_id = run_id

        try:
            yield run_id
            self.store.end_run(run_id, RunStatus.COMPLETED)
        except Exception as e:
            self.store.end_run(run_id, RunStatus.FAILED, str(e))
            raise
        finally:
            self.active_run_id = None

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log a metric value for the active run.

        Args:
            name: Name of the metric.
            value: Numeric value.
            step: Optional step/epoch number.

        Raises:
            ValueError: If no run is active.
        """
        if self.active_run_id is None:
            raise ValueError("No active run. Use start_run() context manager.")
        self.store.log_metric(self.active_run_id, name, value, step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step/epoch number.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """Log additional parameters to the active run.

        Note: This updates the parameters in the database.

        Args:
            parameters: Dictionary of parameter name to value.

        Raises:
            ValueError: If no run is active.
        """
        if self.active_run_id is None:
            raise ValueError("No active run.")

        run = self.store.get_run(self.active_run_id)
        if run:
            updated_params = {**run.parameters, **parameters}
            with self.store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE runs SET parameters = ? WHERE run_id = ?",
                    (json.dumps(updated_params), self.active_run_id),
                )
                conn.commit()

    def log_artifact(
        self,
        name: str,
        path: Path,
        artifact_type: str = "file",
    ) -> None:
        """Log an artifact for the active run.

        Args:
            name: Name of the artifact.
            path: Path to the artifact file.
            artifact_type: Type of artifact.

        Raises:
            ValueError: If no run is active.
        """
        if self.active_run_id is None:
            raise ValueError("No active run.")
        self.store.log_artifact(self.active_run_id, name, path, artifact_type)

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Get a run by ID.

        Args:
            run_id: ID of the run.

        Returns:
            ExperimentRun object or None.
        """
        return self.store.get_run(run_id)


def track_experiment(
    experiment_name: str,
    parameter_names: list[str] | None = None,
) -> Callable[[F], F]:
    """Decorator for automatic experiment tracking.

    Wraps a function to automatically track it as an experiment run,
    capturing parameters and return values.

    Args:
        experiment_name: Name of the experiment.
        parameter_names: Names of function parameters to log.

    Returns:
        Decorator function.

    Example:
        >>> @track_experiment("training", ["epochs", "lr"])
        ... def train(epochs: int, lr: float) -> float:
        ...     return 0.95  # accuracy
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import tempfile

            # Create temporary tracker
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                tracker = ExperimentTracker(Path(f.name))

            tracker.set_experiment(experiment_name)

            # Extract parameters to log
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            params_to_log = {}
            if parameter_names:
                for name in parameter_names:
                    if name in bound.arguments:
                        params_to_log[name] = bound.arguments[name]
            else:
                params_to_log = dict(bound.arguments)

            with tracker.start_run(params_to_log):
                result = func(*args, **kwargs)

                # Log result if numeric
                if isinstance(result, (int, float)):
                    tracker.log_metric("result", float(result))

                return result

        return wrapper  # type: ignore

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: EXPERIMENT COMPARATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Result of comparing two runs.

    Attributes:
        run_id_a: ID of first run.
        run_id_b: ID of second run.
        parameter_diff: Dictionary of differing parameters.
        metric_comparison: Dictionary of metric comparisons.
        winner: Run ID of the better run (based on primary metric).
    """

    run_id_a: str
    run_id_b: str
    parameter_diff: dict[str, tuple[Any, Any]]
    metric_comparison: dict[str, dict[str, float]]
    winner: str | None = None


class ExperimentComparator:
    """Utilities for comparing experiment runs.

    Provides methods for comparing runs, finding the best run
    and generating comparison reports.

    Attributes:
        store: Underlying experiment store.

    Example:
        >>> comparator = ExperimentComparator(store)
        >>> result = comparator.compare_runs(run_id_1, run_id_2, "accuracy")
        >>> print(f"Winner: {result.winner}")
    """

    def __init__(self, store: ExperimentStore) -> None:
        """Initialise the comparator.

        Args:
            store: Experiment store to use.
        """
        self.store = store

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
        primary_metric: str,
        higher_is_better: bool = True,
    ) -> ComparisonResult:
        """Compare two runs.

        Args:
            run_id_a: ID of first run.
            run_id_b: ID of second run.
            primary_metric: Metric to use for determining winner.
            higher_is_better: Whether higher metric values are better.

        Returns:
            ComparisonResult with detailed comparison.
        """
        run_a = self.store.get_run(run_id_a)
        run_b = self.store.get_run(run_id_b)

        if not run_a or not run_b:
            raise ValueError("One or both runs not found")

        # Compare parameters
        param_diff: dict[str, tuple[Any, Any]] = {}
        all_params = set(run_a.parameters.keys()) | set(run_b.parameters.keys())
        for param in all_params:
            val_a = run_a.parameters.get(param)
            val_b = run_b.parameters.get(param)
            if val_a != val_b:
                param_diff[param] = (val_a, val_b)

        # Compare metrics
        all_metrics = set(run_a.metrics.keys()) | set(run_b.metrics.keys())
        metric_comparison: dict[str, dict[str, float]] = {}

        for metric in all_metrics:
            entries_a = run_a.metrics.get(metric, [])
            entries_b = run_b.metrics.get(metric, [])

            final_a = entries_a[-1].value if entries_a else float("nan")
            final_b = entries_b[-1].value if entries_b else float("nan")

            metric_comparison[metric] = {
                "run_a": final_a,
                "run_b": final_b,
                "diff": final_b - final_a,
                "pct_diff": ((final_b - final_a) / final_a * 100) if final_a != 0 else 0,
            }

        # Determine winner
        winner = None
        if primary_metric in metric_comparison:
            val_a = metric_comparison[primary_metric]["run_a"]
            val_b = metric_comparison[primary_metric]["run_b"]
            if higher_is_better:
                winner = run_id_a if val_a > val_b else run_id_b
            else:
                winner = run_id_a if val_a < val_b else run_id_b

        return ComparisonResult(
            run_id_a=run_id_a,
            run_id_b=run_id_b,
            parameter_diff=param_diff,
            metric_comparison=metric_comparison,
            winner=winner,
        )

    def find_best_run(
        self,
        experiment_id: str,
        metric_name: str,
        higher_is_better: bool = True,
    ) -> ExperimentRun | None:
        """Find the best run for an experiment based on a metric.

        Args:
            experiment_id: ID of the experiment.
            metric_name: Metric to optimise.
            higher_is_better: Whether higher values are better.

        Returns:
            Best ExperimentRun or None if no runs found.
        """
        runs = self.store.list_runs(experiment_id, RunStatus.COMPLETED)

        if not runs:
            return None

        best_run = None
        best_value = float("-inf") if higher_is_better else float("inf")

        for run in runs:
            if metric_name in run.metrics and run.metrics[metric_name]:
                final_value = run.metrics[metric_name][-1].value
                if higher_is_better:
                    if final_value > best_value:
                        best_value = final_value
                        best_run = run
                else:
                    if final_value < best_value:
                        best_value = final_value
                        best_run = run

        return best_run

    def get_metric_progression(
        self,
        run_ids: list[str],
        metric_name: str,
    ) -> dict[str, list[tuple[int | None, float]]]:
        """Get metric progression for multiple runs.

        Args:
            run_ids: List of run IDs.
            metric_name: Name of the metric.

        Returns:
            Dictionary mapping run ID to list of (step, value) tuples.
        """
        result = {}
        for run_id in run_ids:
            history = self.store.get_metrics_history(run_id, metric_name)
            result[run_id] = history
        return result

    def generate_report(
        self,
        experiment_id: str,
        metrics: list[str] | None = None,
    ) -> str:
        """Generate a comprehensive report for an experiment.

        Args:
            experiment_id: ID of the experiment.
            metrics: Optional list of metrics to include.

        Returns:
            Formatted report string.
        """
        runs = self.store.list_runs(experiment_id)

        if not runs:
            return f"No runs found for experiment {experiment_id}"

        lines = [
            "=" * 70,
            f"EXPERIMENT REPORT: {experiment_id}",
            "=" * 70,
            f"Total runs: {len(runs)}",
            f"Completed: {sum(1 for r in runs if r.status == RunStatus.COMPLETED)}",
            f"Failed: {sum(1 for r in runs if r.status == RunStatus.FAILED)}",
            "-" * 70,
        ]

        # Determine metrics to report
        all_metrics: set[str] = set()
        for run in runs:
            all_metrics.update(run.metrics.keys())

        if metrics:
            all_metrics = all_metrics.intersection(metrics)

        # Header
        header = f"{'Run ID':<20} {'Status':<12}"
        for metric in sorted(all_metrics):
            header += f" {metric:>12}"
        lines.append(header)
        lines.append("-" * 70)

        # Rows
        for run in runs:
            row = f"{run.run_id[:18]:<20} {run.status.value:<12}"
            for metric in sorted(all_metrics):
                if metric in run.metrics and run.metrics[metric]:
                    value = run.metrics[metric][-1].value
                    row += f" {value:>12.4f}"
                else:
                    row += f" {'N/A':>12}"
            lines.append(row)

        lines.append("=" * 70)

        # Find best runs
        for metric in sorted(all_metrics):
            best = self.find_best_run(experiment_id, metric)
            if best and metric in best.metrics:
                best_val = best.metrics[metric][-1].value
                lines.append(f"Best {metric}: {best_val:.4f} (run {best.run_id[:18]})")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_training(
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Simulate a training process with fake metrics.

    Args:
        epochs: Number of epochs.
        learning_rate: Learning rate (affects convergence).
        batch_size: Batch size (affects noise).

    Returns:
        Tuple of (losses, accuracies) lists.
    """
    random.seed(42)

    losses = []
    accuracies = []

    base_loss = 2.0
    base_acc = 0.3

    for epoch in range(epochs):
        # Simulate decreasing loss
        noise = random.gauss(0, 0.1 / batch_size * 32)
        loss = base_loss * (0.9 ** (epoch * learning_rate * 100)) + noise
        loss = max(0.01, loss)
        losses.append(loss)

        # Simulate increasing accuracy
        noise = random.gauss(0, 0.02)
        acc = 1.0 - (1.0 - base_acc) * (0.9 ** (epoch * learning_rate * 100)) + noise
        acc = min(0.99, max(0.0, acc))
        accuracies.append(acc)

    return losses, accuracies


def demonstrate_tracking() -> None:
    """Demonstrate the experiment tracking system."""
    import tempfile

    logger.info("=" * 60)
    logger.info("DEMONSTRATING EXPERIMENT TRACKING SYSTEM")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "experiments.db"

        # Create tracker
        tracker = ExperimentTracker(db_path)
        tracker.set_experiment(
            name="hyperparameter_search",
            description="Searching for optimal learning rate",
            tags=["ml", "hyperparameter"],
        )

        # Run multiple experiments with different parameters
        configurations = [
            {"learning_rate": 0.001, "batch_size": 32, "epochs": 10},
            {"learning_rate": 0.01, "batch_size": 32, "epochs": 10},
            {"learning_rate": 0.1, "batch_size": 64, "epochs": 10},
        ]

        run_ids = []
        for config in configurations:
            logger.info(f"Running with config: {config}")

            with tracker.start_run(config) as run_id:
                run_ids.append(run_id)

                losses, accuracies = simulate_training(
                    config["epochs"],
                    config["learning_rate"],
                    config["batch_size"],
                )

                for epoch, (loss, acc) in enumerate(zip(losses, accuracies)):
                    tracker.log_metrics(
                        {"loss": loss, "accuracy": acc},
                        step=epoch,
                    )
                    time.sleep(0.01)  # Small delay for realistic timing

        # Compare runs
        logger.info("\n" + "=" * 60)
        logger.info("COMPARING RUNS")
        logger.info("=" * 60)

        comparator = ExperimentComparator(tracker.store)

        # Compare first two runs
        comparison = comparator.compare_runs(
            run_ids[0],
            run_ids[1],
            primary_metric="accuracy",
            higher_is_better=True,
        )
        logger.info(f"Winner between runs 1 and 2: {comparison.winner}")
        logger.info(f"Parameter differences: {comparison.parameter_diff}")

        # Find best run
        best_run = comparator.find_best_run(
            tracker.experiment_id,
            "accuracy",
            higher_is_better=True,
        )
        if best_run:
            logger.info(f"Best run overall: {best_run.run_id}")
            logger.info(f"  Parameters: {best_run.parameters}")
            final_acc = best_run.metrics["accuracy"][-1].value
            logger.info(f"  Final accuracy: {final_acc:.4f}")

        # Generate report
        report = comparator.generate_report(tracker.experiment_id)
        print("\n" + report)


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demonstrate_tracking()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Experiment Tracking System - Solution"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        run_all_demos()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
