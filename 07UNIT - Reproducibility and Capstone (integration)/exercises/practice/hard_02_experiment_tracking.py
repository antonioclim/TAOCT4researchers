#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Hard 02 - Experiment Tracking System
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Tracking experiments is crucial for reproducible research. This exercise
challenges you to build a comprehensive experiment tracking system similar
to MLflow or Weights & Biases, but simplified for educational purposes.

PREREQUISITES
─────────────
- Completed all previous exercises
- Understanding of database concepts
- Familiarity with context managers and decorators

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Design experiment tracking schemas
2. Implement automatic metric logging
3. Create experiment comparison tools

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 90 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Generator
from typing import TypeVar

import numpy as np

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricEntry:
    """A single metric measurement."""

    name: str
    value: float
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Artifact:
    """An artifact produced by an experiment."""

    name: str
    path: str
    artifact_type: str  # "model", "data", "figure", "log"
    size_bytes: int
    created: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentRun:
    """A single run of an experiment."""

    run_id: str
    experiment_name: str
    status: str  # "running", "completed", "failed"
    start_time: str
    end_time: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: list[MetricEntry] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    git_commit: str | None = None
    error_message: str | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Experiment Store
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentStore:
    """
    TODO: Implement a persistent experiment store.

    This class manages storage and retrieval of experiment data.
    Use SQLite for persistence.

    Database schema:
    - experiments: id, name, description, created
    - runs: id, experiment_id, status, start_time, end_time, parameters, git_commit
    - metrics: id, run_id, name, value, step, timestamp
    - artifacts: id, run_id, name, path, type, size_bytes, created
    - tags: id, run_id, tag
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        """
        Initialise the experiment store.

        Args:
            db_path: Path to SQLite database or ":memory:" for in-memory.
        """
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """
        TODO: Initialise database schema.

        Create all necessary tables if they don't exist.
        """
        # TODO: Implement
        pass

    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new experiment.

        Args:
            config: Experiment configuration.

        Returns:
            Experiment ID.
        """
        # TODO: Implement
        pass

    def start_run(
        self,
        experiment_name: str,
        parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> ExperimentRun:
        """
        Start a new experiment run.

        Args:
            experiment_name: Name of the experiment.
            parameters: Run parameters.
            tags: Run tags.

        Returns:
            ExperimentRun object.
        """
        # TODO: Implement
        pass

    def end_run(
        self,
        run_id: str,
        status: str = "completed",
        error_message: str | None = None,
    ) -> None:
        """
        End an experiment run.

        Args:
            run_id: Run ID.
            status: Final status ("completed" or "failed").
            error_message: Error message if failed.
        """
        # TODO: Implement
        pass

    def log_metric(
        self,
        run_id: str,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Log a metric for a run.

        Args:
            run_id: Run ID.
            name: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        # TODO: Implement
        pass

    def log_artifact(
        self,
        run_id: str,
        name: str,
        path: Path,
        artifact_type: str = "file",
    ) -> None:
        """
        Log an artifact for a run.

        Args:
            run_id: Run ID.
            name: Artifact name.
            path: Path to artifact file.
            artifact_type: Type of artifact.
        """
        # TODO: Implement
        pass

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """
        Get a run by ID.

        Args:
            run_id: Run ID.

        Returns:
            ExperimentRun or None if not found.
        """
        # TODO: Implement
        pass

    def list_runs(
        self,
        experiment_name: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ExperimentRun]:
        """
        List runs with optional filters.

        Args:
            experiment_name: Filter by experiment name.
            status: Filter by status.
            tags: Filter by tags (all must match).

        Returns:
            List of matching runs.
        """
        # TODO: Implement
        pass

    def get_metrics_history(
        self,
        run_id: str,
        metric_name: str,
    ) -> list[MetricEntry]:
        """
        Get metric history for a run.

        Args:
            run_id: Run ID.
            metric_name: Name of metric.

        Returns:
            List of MetricEntry objects.
        """
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Experiment Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentTracker:
    """
    TODO: Implement an experiment tracker.

    This class provides a high-level interface for tracking experiments
    with automatic context management and decorators.
    """

    _instance: "ExperimentTracker | None" = None
    _current_run: ExperimentRun | None = None

    def __init__(self, store: ExperimentStore) -> None:
        """
        Initialise tracker with a store.

        Args:
            store: ExperimentStore for persistence.
        """
        self.store = store
        ExperimentTracker._instance = self

    @classmethod
    def get_instance(cls) -> "ExperimentTracker":
        """Get the singleton tracker instance."""
        if cls._instance is None:
            raise RuntimeError("ExperimentTracker not initialised")
        return cls._instance

    @contextmanager
    def start_run(
        self,
        experiment_name: str,
        parameters: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Generator[ExperimentRun, None, None]:
        """
        TODO: Implement context manager for experiment runs.

        Usage:
            with tracker.start_run("my_experiment", {"lr": 0.01}) as run:
                # ... experiment code ...
                tracker.log_metric("accuracy", 0.95)

        Should automatically:
        1. Start the run
        2. Set as current run
        3. End run with appropriate status on exit
        4. Handle exceptions and log errors
        """
        # TODO: Implement
        pass

    def log_metric(
        self,
        name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Log a metric to the current run.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step number.

        Raises:
            RuntimeError: If no run is active.
        """
        # TODO: Implement
        pass

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric name -> value.
            step: Optional step number.
        """
        # TODO: Implement
        pass

    def log_parameters(self, parameters: dict[str, Any]) -> None:
        """
        Log parameters to current run.

        Args:
            parameters: Dictionary of parameters.
        """
        # TODO: Implement
        pass

    def log_artifact(
        self,
        name: str,
        path: Path,
        artifact_type: str = "file",
    ) -> None:
        """
        Log an artifact to current run.

        Args:
            name: Artifact name.
            path: Path to artifact.
            artifact_type: Type of artifact.
        """
        # TODO: Implement
        pass


def track_experiment(
    experiment_name: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """
    TODO: Implement decorator for tracking experiments.

    Usage:
        @track_experiment("my_experiment", {"lr": 0.01})
        def train_model(data):
            # ... training code ...
            return accuracy

    The decorator should:
    1. Start a run before function execution
    2. Log return value as "result" metric if numeric
    3. Handle exceptions and log errors
    4. End run after function completes
    """
    # TODO: Implement
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # TODO: Implement wrapper
            pass
        return wrapper  # type: ignore
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Experiment Comparison
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Result of comparing two experiment runs."""

    run_id_a: str
    run_id_b: str
    parameter_diffs: dict[str, tuple[Any, Any]]
    metric_diffs: dict[str, tuple[float, float]]
    best_run: str
    comparison_metric: str


class ExperimentComparator:
    """
    TODO: Implement experiment comparison tools.

    Features:
    1. Compare parameters between runs
    2. Compare metrics between runs
    3. Find best run by metric
    4. Generate comparison reports
    """

    def __init__(self, store: ExperimentStore) -> None:
        """
        Initialise comparator.

        Args:
            store: ExperimentStore for accessing runs.
        """
        self.store = store

    def compare_runs(
        self,
        run_id_a: str,
        run_id_b: str,
        comparison_metric: str,
        higher_is_better: bool = True,
    ) -> ComparisonResult:
        """
        Compare two experiment runs.

        Args:
            run_id_a: First run ID.
            run_id_b: Second run ID.
            comparison_metric: Metric to determine best run.
            higher_is_better: Whether higher metric is better.

        Returns:
            ComparisonResult with differences.
        """
        # TODO: Implement
        pass

    def find_best_run(
        self,
        experiment_name: str,
        metric: str,
        higher_is_better: bool = True,
    ) -> ExperimentRun | None:
        """
        Find the best run by a metric.

        Args:
            experiment_name: Experiment to search.
            metric: Metric to optimise.
            higher_is_better: Whether higher is better.

        Returns:
            Best ExperimentRun or None.
        """
        # TODO: Implement
        pass

    def get_metric_progression(
        self,
        experiment_name: str,
        metric: str,
    ) -> dict[str, list[float]]:
        """
        Get metric progression across all runs.

        Args:
            experiment_name: Experiment name.
            metric: Metric name.

        Returns:
            Dictionary mapping run_id to metric values.
        """
        # TODO: Implement
        pass

    def generate_report(
        self,
        experiment_name: str,
        metrics: list[str],
    ) -> str:
        """
        Generate a comparison report for an experiment.

        Args:
            experiment_name: Experiment name.
            metrics: Metrics to include.

        Returns:
            Markdown-formatted report.
        """
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_training(
    epochs: int,
    learning_rate: float,
    tracker: ExperimentTracker,
) -> float:
    """Simulate a training run with metric logging."""
    accuracy = 0.5

    for epoch in range(epochs):
        # Simulate training
        accuracy += np.random.uniform(0, 0.1) * (1 - accuracy)
        loss = 1 - accuracy + np.random.uniform(-0.05, 0.05)

        tracker.log_metrics({
            "accuracy": accuracy,
            "loss": max(0, loss),
        }, step=epoch)

        time.sleep(0.01)  # Simulate computation

    return accuracy


def demonstrate_tracking() -> None:
    """Demonstrate the experiment tracking system."""
    print("=" * 60)
    print("Experiment Tracking System Demo")
    print("=" * 60)

    # Create store and tracker
    store = ExperimentStore()
    tracker = ExperimentTracker(store)

    # Create experiment
    store.create_experiment(ExperimentConfig(
        name="demo_experiment",
        description="Demonstrating the tracking system",
        tags=["demo", "testing"],
    ))

    # Run multiple experiments with different parameters
    configs = [
        {"epochs": 10, "learning_rate": 0.01},
        {"epochs": 10, "learning_rate": 0.001},
        {"epochs": 20, "learning_rate": 0.01},
    ]

    for config in configs:
        print(f"\nRunning with config: {config}")

        with tracker.start_run(
            "demo_experiment",
            parameters=config,
        ) as run:
            accuracy = simulate_training(
                config["epochs"],
                config["learning_rate"],
                tracker,
            )
            print(f"  Final accuracy: {accuracy:.3f}")

    # Compare experiments
    print("\n" + "=" * 60)
    print("Experiment Comparison")
    print("=" * 60)

    comparator = ExperimentComparator(store)
    best_run = comparator.find_best_run(
        "demo_experiment",
        metric="accuracy",
        higher_is_better=True,
    )

    if best_run:
        print(f"Best run: {best_run.run_id}")
        print(f"  Parameters: {best_run.parameters}")
        final_metrics = {m.name: m.value for m in best_run.metrics}
        print(f"  Final metrics: {final_metrics}")

    # Generate report
    report = comparator.generate_report(
        "demo_experiment",
        metrics=["accuracy", "loss"],
    )
    print("\n" + report)


if __name__ == "__main__":
    demonstrate_tracking()
