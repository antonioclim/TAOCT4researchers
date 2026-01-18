#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
07UNIT, Lab 1: Reproducibility Toolkit
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Non-reproducible single occurrences are of no significance to science."
— Karl Popper

Reproducibility is the foundation of the scientific method. In the digital era,
code is part of the methodology and must be treated with the same rigour as any
experimental protocol. This toolkit provides essential utilities for ensuring
your computational research can be reliably reproduced by others—or by yourself
months later.

PREREQUISITES
─────────────
- Week 6: Visualisation for Research (output generation)
- Python: Intermediate proficiency with decorators and context managers
- Libraries: hashlib, json, dataclasses, typing

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Configure reproducible random seed management across multiple libraries
2. Create and verify data integrity manifests using cryptographic hashes
3. Implement structured experiment logging with comprehensive metadata
4. Generate standardised project documentation automatically

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 90 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
- Python 3.12+
- NumPy ≥1.24 (optional, for seed management)
- Standard library: hashlib, json, dataclasses, typing, pathlib

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

# Configure module-level logger
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: REPRODUCIBILITY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ReproducibilityConfig:
    """
    Centralised configuration for reproducibility settings.

    This class provides a unified interface for setting random seeds across
    multiple libraries commonly used in computational research. By centralising
    seed management, we ensure consistent behaviour across all random operations.

    Attributes:
        seed: The master random seed for all libraries.
        deterministic: Whether to enforce deterministic behaviour in deep learning.
        log_level: Logging verbosity level.

    Example:
        >>> config = ReproducibilityConfig(seed=42)
        >>> config.apply()  # Sets all seeds consistently
        >>> random.random()  # Now reproducible
        0.6394267984578837

    Research Context:
        In computational biology, setting consistent seeds is essential for
        reproducing simulation results. A study by Hutson (2018) found that
        over 70% of AI research papers could not be reproduced due to missing
        seed information.
    """

    seed: int = 42
    deterministic: bool = True
    log_level: str = "INFO"

    def apply(self) -> None:
        """
        Apply the configuration to all supported libraries.

        This method sets seeds for:
        - Python's built-in random module
        - NumPy's random number generator
        - PyTorch (if installed)
        - TensorFlow (if installed)

        It also sets the PYTHONHASHSEED environment variable to ensure
        consistent hash-based operations.
        """
        # Python's built-in random
        random.seed(self.seed)

        # Environment variable for hash seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # NumPy (if available)
        try:
            import numpy as np

            np.random.seed(self.seed)
            logger.debug("NumPy seed configured")
        except ImportError:
            logger.debug("NumPy not available, skipping")

        # PyTorch (if available)
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            if self.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.debug("PyTorch seed configured")
        except ImportError:
            logger.debug("PyTorch not available, skipping")

        # TensorFlow (if available)
        try:
            import tensorflow as tf

            tf.random.set_seed(self.seed)
            logger.debug("TensorFlow seed configured")
        except ImportError:
            logger.debug("TensorFlow not available, skipping")

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        logger.info(f"Reproducibility configured: seed={self.seed}")


def set_all_seeds(seed: int) -> None:
    """
    Convenience function for setting all random seeds.

    Args:
        seed: The master random seed to apply.

    Example:
        >>> set_all_seeds(42)
        >>> random.random()
        0.6394267984578837
    """
    ReproducibilityConfig(seed=seed).apply()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA INTEGRITY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def compute_file_hash(filepath: str | Path, algorithm: str = "sha256") -> str:
    """
    Compute the cryptographic hash of a file for integrity verification.

    This function reads files in chunks to handle large files efficiently
    without loading them entirely into memory.

    Args:
        filepath: Path to the file to hash.
        algorithm: Hash algorithm to use ('md5', 'sha256', 'sha512').

    Returns:
        Hexadecimal string representation of the hash.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If an unsupported algorithm is specified.

    Example:
        >>> hash_value = compute_file_hash("data/train.csv")
        >>> print(hash_value[:16])  # First 16 characters
        'a1b2c3d4e5f6g7h8'

    Research Context:
        Data integrity verification is essential in bioinformatics where
        sequence databases are frequently updated. The NCBI recommends
        SHA-256 checksums for all downloaded datasets.
    """
    supported_algorithms = {"md5", "sha256", "sha512", "sha1"}
    if algorithm not in supported_algorithms:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}. "
            f"Use one of: {supported_algorithms}"
        )

    hash_func = hashlib.new(algorithm)

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def verify_file_hash(
    filepath: str | Path, expected_hash: str, algorithm: str = "sha256"
) -> bool:
    """
    Verify that a file's hash matches the expected value.

    Args:
        filepath: Path to the file to verify.
        expected_hash: Expected hash value as hexadecimal string.
        algorithm: Hash algorithm used to compute the expected hash.

    Returns:
        True if hashes match, False otherwise.

    Example:
        >>> is_valid = verify_file_hash("data/train.csv", expected_hash)
        >>> assert is_valid, "Data corruption detected!"
    """
    actual_hash = compute_file_hash(filepath, algorithm)
    return actual_hash == expected_hash


@dataclass
class DataManifest:
    """
    Manifest for tracking and verifying data file integrity.

    A data manifest records cryptographic hashes of all data files used in
    a research project. This allows verification that data has not been
    corrupted or modified since the manifest was created.

    Attributes:
        files: Dictionary mapping file paths to their hash values.
        created_at: ISO format timestamp of manifest creation.
        algorithm: Hash algorithm used for all files.

    Example:
        >>> manifest = DataManifest()
        >>> manifest.add_file("data/train.csv")
        >>> manifest.add_file("data/test.csv")
        >>> manifest.save("data/MANIFEST.json")
        >>>
        >>> # Later, verify integrity:
        >>> loaded = DataManifest.load("data/MANIFEST.json")
        >>> results = loaded.verify_all()
        >>> assert all(results.values()), "Data integrity check failed!"

    Research Context:
        The Open Science Framework recommends including data manifests
        with all research publications to ensure long-term reproducibility.
    """

    files: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    algorithm: str = "sha256"

    def add_file(self, filepath: str | Path) -> None:
        """
        Add a file to the manifest.

        Args:
            filepath: Path to the file to add.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        hash_value = compute_file_hash(path, self.algorithm)
        self.files[str(path)] = hash_value
        logger.debug(f"Added to manifest: {path} ({hash_value[:8]}...)")

    def verify_file(self, filepath: str | Path) -> bool:
        """
        Verify a single file against the manifest.

        Args:
            filepath: Path to the file to verify.

        Returns:
            True if the file hash matches the manifest.

        Raises:
            KeyError: If the file is not in the manifest.
        """
        path = str(Path(filepath))
        if path not in self.files:
            raise KeyError(f"File not in manifest: {filepath}")

        return verify_file_hash(path, self.files[path], self.algorithm)

    def verify_all(self) -> dict[str, bool]:
        """
        Verify all files in the manifest.

        Returns:
            Dictionary mapping file paths to verification results.
        """
        results = {}
        for filepath in self.files:
            try:
                results[filepath] = self.verify_file(filepath)
            except FileNotFoundError:
                results[filepath] = False
                logger.warning(f"File missing: {filepath}")
        return results

    def save(self, filepath: str | Path) -> None:
        """Save the manifest as JSON."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Manifest saved: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "DataManifest":
        """Load a manifest from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EXPERIMENT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentConfig:
    """
    Configuration for a single experiment.

    Attributes:
        name: Unique identifier for the experiment.
        description: Human-readable description.
        parameters: Dictionary of hyperparameters and settings.
        tags: List of categorical tags for organisation.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """
    Container for experiment results.

    Attributes:
        metrics: Dictionary of metric names to values.
        artifacts: List of paths to generated artifacts.
        logs: List of timestamped log messages.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)

    def add_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        self.metrics[name] = value
        logger.debug(f"Metric recorded: {name}={value}")

    def add_artifact(self, filepath: str) -> None:
        """Record an artifact path."""
        self.artifacts.append(filepath)

    def log(self, message: str) -> None:
        """Add a timestamped log entry."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")


@dataclass
class Experiment:
    """
    Complete experiment container with configuration and results.

    This class provides a structured way to track computational experiments,
    including their configuration, results and metadata. It supports context
    manager usage for automatic timing and status tracking.

    Attributes:
        config: Experiment configuration.
        result: Experiment results container.
        started_at: ISO timestamp of experiment start.
        finished_at: ISO timestamp of experiment completion.
        duration_seconds: Total execution time in seconds.
        status: Current status (pending, running, completed, failed).
        error_message: Error message if experiment failed.
        python_version: Python version used.
        platform: Operating system platform.

    Example:
        >>> exp = Experiment(
        ...     config=ExperimentConfig(
        ...         name="baseline_model",
        ...         parameters={"learning_rate": 0.001, "epochs": 100}
        ...     )
        ... )
        >>>
        >>> with exp.run():
        ...     # Training code here
        ...     exp.result.add_metric("accuracy", 0.95)
        ...     exp.result.add_metric("loss", 0.05)
        >>>
        >>> exp.save("experiments/exp001.json")

    Research Context:
        Structured experiment logging is recommended by the Machine Learning
        Reproducibility Checklist (Pineau et al., 2021) and is essential
        for conducting systematic hyperparameter studies.
    """

    config: ExperimentConfig
    result: ExperimentResult = field(default_factory=ExperimentResult)

    # Metadata
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    status: str = "pending"
    error_message: str | None = None

    # Environment
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=lambda: sys.platform)

    def run(self) -> "ExperimentContext":
        """
        Return a context manager for running the experiment.

        Returns:
            ExperimentContext for use with 'with' statement.
        """
        return ExperimentContext(self)

    def save(self, filepath: str | Path) -> None:
        """
        Save the experiment as JSON.

        Args:
            filepath: Destination path for the JSON file.
        """
        data = {
            "config": asdict(self.config),
            "result": asdict(self.result),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error_message": self.error_message,
            "python_version": self.python_version,
            "platform": self.platform,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Experiment saved: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "Experiment":
        """
        Load an experiment from JSON.

        Args:
            filepath: Path to the JSON file.

        Returns:
            Loaded Experiment instance.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = ExperimentConfig(**data.pop("config"))
        result_data = data.pop("result")
        result = ExperimentResult(**result_data)

        return cls(config=config, result=result, **data)


class ExperimentContext:
    """
    Context manager for experiment execution.

    Automatically tracks timing and status, handling exceptions gracefully.
    """

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self._start_time: float = 0.0

    def __enter__(self) -> Experiment:
        self.experiment.started_at = datetime.now().isoformat()
        self.experiment.status = "running"
        self._start_time = time.time()

        logger.info(f"Experiment started: {self.experiment.config.name}")
        return self.experiment

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        self.experiment.finished_at = datetime.now().isoformat()
        self.experiment.duration_seconds = time.time() - self._start_time

        if exc_type is not None:
            self.experiment.status = "failed"
            self.experiment.error_message = str(exc_val)
            logger.error(f"Experiment failed: {exc_val}")
        else:
            self.experiment.status = "completed"
            logger.info(
                f"Experiment completed in {self.experiment.duration_seconds:.2f}s"
            )

        return False  # Do not suppress exceptions


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: UTILITY DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for measuring function execution time.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that logs execution time.

    Example:
        >>> @timed
        ... def train_model():
        ...     time.sleep(1)
        ...
        >>> train_model()  # Logs: "train_model executed in 1.0001s"
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result

    return wrapper


def retry(
    max_attempts: int = 3, delay: float = 1.0, exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying failed function calls.

    Useful for network operations or other operations that may fail
    transiently.

    Args:
        max_attempts: Maximum number of attempts before giving up.
        delay: Seconds to wait between attempts.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorator function.

    Example:
        >>> @retry(max_attempts=3, delay=2.0)
        ... def download_data():
        ...     # May fail due to network issues
        ...     pass
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: BaseException | None = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}"
                    )
                    if attempt < max_attempts - 1:
                        time.sleep(delay)

            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected retry state")

        return wrapper

    return decorator


def cached(func: Callable[..., T]) -> Callable[..., T]:
    """
    Simple caching decorator for expensive computations.

    Note: For production use, consider functools.lru_cache or a proper
    caching library.

    Args:
        func: Function to cache.

    Returns:
        Cached function.
    """
    cache: dict[str, T] = {}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
            logger.debug(f"Cache miss for {func.__name__}")
        else:
            logger.debug(f"Cache hit for {func.__name__}")
        return cache[key]

    return wrapper


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: DOCUMENTATION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_readme(
    project_name: str,
    description: str,
    requirements: list[str],
    usage: str,
    author: str = "",
    licence: str = "MIT",
) -> str:
    """
    Generate a standardised README.md for research projects.

    Args:
        project_name: Name of the project.
        description: Brief project description.
        requirements: List of dependencies.
        usage: Usage instructions.
        author: Author name.
        licence: Licence type.

    Returns:
        Complete README content as string.

    Example:
        >>> readme = generate_readme(
        ...     project_name="My Research",
        ...     description="A computational study",
        ...     requirements=["numpy>=1.24", "pandas>=2.0"],
        ...     usage="python run.py --config default.yaml"
        ... )
        >>> Path("README.md").write_text(readme)
    """
    requirements_str = "\n".join(f"- {r}" for r in requirements)
    slug = project_name.lower().replace(" ", "-")

    readme = f"""# {project_name}

{description}

## Installation

```bash
git clone https://github.com/username/{slug}.git
cd {slug}
pip install -e ".[dev]"
```

## Requirements

{requirements_str}

## Usage

{usage}

## Reproducibility

This project follows reproducibility established conventions:

1. **Random seeds**: All random operations are seeded for reproducibility
2. **Data versioning**: Data files are checksummed in `MANIFEST.json`
3. **Environment**: Dependencies are pinned in `pyproject.toml`
4. **Experiments**: All experiments are logged with parameters and results

To reproduce results:

```bash
python run.py --seed 42 --config config/default.yaml
```

## Project Structure

```
{slug}/
├── src/                # Source code
├── tests/              # Test files
├── data/               # Data files
├── experiments/        # Experiment logs
├── notebooks/          # Jupyter notebooks
└── docs/               # Documentation
```

## Licence

{licence}

## Author

{author}
"""

    return readme


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def demo_reproducibility() -> None:
    """Demonstrate reproducibility configuration."""
    logger.info("=" * 60)
    logger.info("DEMO: Reproducibility Configuration")
    logger.info("=" * 60)

    config = ReproducibilityConfig(seed=42)
    config.apply()

    # Verify reproducibility
    logger.info("Random numbers with seed=42:")
    numbers_first = [random.random() for _ in range(5)]
    for i, num in enumerate(numbers_first):
        logger.info(f"  random.random()[{i}] = {num:.6f}")

    # Reset and verify
    config.apply()
    logger.info("After reset with same seed:")
    numbers_second = [random.random() for _ in range(5)]
    for i, num in enumerate(numbers_second):
        logger.info(f"  random.random()[{i}] = {num:.6f}")

    assert numbers_first == numbers_second, "Reproducibility failed!"
    logger.info("✓ Reproducibility verified!")


def demo_experiment_logging() -> None:
    """Demonstrate experiment logging."""
    logger.info("=" * 60)
    logger.info("DEMO: Experiment Logging")
    logger.info("=" * 60)

    exp = Experiment(
        config=ExperimentConfig(
            name="demo_experiment",
            description="Demonstration of experiment logging",
            parameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
            },
            tags=["demo", "test"],
        )
    )

    with exp.run():
        # Simulate training
        for epoch in range(3):
            time.sleep(0.1)  # Simulate work
            accuracy = 0.8 + epoch * 0.05 + random.random() * 0.02
            exp.result.add_metric(f"accuracy_epoch_{epoch}", accuracy)

        exp.result.add_metric("final_accuracy", 0.92)
        exp.result.log("Training completed successfully")

    logger.info(f"Experiment: {exp.config.name}")
    logger.info(f"Status: {exp.status}")
    logger.info(f"Duration: {exp.duration_seconds:.2f}s")
    logger.info(f"Metrics: {exp.result.metrics}")


def demo_data_integrity() -> None:
    """Demonstrate data integrity verification."""
    logger.info("=" * 60)
    logger.info("DEMO: Data Integrity Verification")
    logger.info("=" * 60)

    # Create a temporary test file
    test_file = Path("/tmp/test_data.txt")
    test_file.write_text("Hello, this is test data for integrity check!")

    # Compute hash
    file_hash = compute_file_hash(test_file)
    logger.info(f"File: {test_file}")
    logger.info(f"SHA256: {file_hash}")

    # Verify
    is_valid = verify_file_hash(test_file, file_hash)
    logger.info(f"Verification: {'PASS' if is_valid else 'FAIL'}")

    # Demonstrate manifest
    manifest = DataManifest()
    manifest.add_file(test_file)
    manifest.save("/tmp/test_manifest.json")

    # Verify via manifest
    loaded = DataManifest.load("/tmp/test_manifest.json")
    results = loaded.verify_all()
    logger.info(f"Manifest verification: {results}")

    # Cleanup
    test_file.unlink()
    Path("/tmp/test_manifest.json").unlink()
    logger.info("✓ Data integrity demo completed!")


def run_all_demos() -> None:
    """Execute all demonstrations."""
    demo_reproducibility()
    print()
    demo_experiment_logging()
    print()
    demo_data_integrity()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="07UNIT Lab 1: Reproducibility Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_7_01_reproducibility.py --demo
  python lab_7_01_reproducibility.py --seed 42 --verbose

Established Conventions:
  1. Set ALL seeds with set_all_seeds(42)
  2. Verify data with DataManifest
  3. Log experiments with Experiment
  4. Generate documentation with generate_readme()
        """,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.demo:
        set_all_seeds(args.seed)
        run_all_demos()
    else:
        logger.info("07UNIT Lab 1: Reproducibility Toolkit")
        logger.info("Use --demo to run demonstrations")
        logger.info("Use --help for more options")


if __name__ == "__main__":
    main()
