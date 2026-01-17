#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Lab: Reproducibility Toolkit
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
"Non-reproducible single occurrences are of no significance to science."
— Karl Popper

Reproducibilitatea este fundamentul metodei științifice. În era digitală,
codul e parte din metodologie și trebuie tratat cu aceeași rigoare.

ACEST TOOLKIT OFERĂ:
1. Helpers pentru seeding consistent
2. Logging structurat pentru experimente
3. Verificare integritate date
4. Generare automată documentație

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
import logging
import random
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, TypeVar
from pathlib import Path
from functools import wraps
import traceback


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: REPRODUCIBILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ReproducibilityConfig:
    """
    Configurație centralizată pentru reproducibilitate.
    
    Folosire:
        config = ReproducibilityConfig(seed=42)
        config.apply()  # Setează toate seed-urile
    """
    seed: int = 42
    deterministic: bool = True
    log_level: str = "INFO"
    
    def apply(self) -> None:
        """Aplică configurația la toate bibliotecile."""
        # Python random
        random.seed(self.seed)
        
        # Environment variable pentru alte biblioteci
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # NumPy (dacă e disponibil)
        try:
            import numpy as np
            np.random.seed(self.seed)
        except ImportError:
            pass
        
        # PyTorch (dacă e disponibil)
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            if self.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
        
        # TensorFlow (dacă e disponibil)
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
        except ImportError:
            pass
        
        # Logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        logging.info(f"Reproducibility configured: seed={self.seed}")


def set_all_seeds(seed: int) -> None:
    """Shortcut pentru setarea tuturor seed-urilor."""
    ReproducibilityConfig(seed=seed).apply()


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: DATA INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_file_hash(filepath: str | Path, algorithm: str = 'sha256') -> str:
    """
    Calculează hash-ul unui fișier pentru verificare integritate.
    
    Args:
        filepath: Calea către fișier
        algorithm: 'md5', 'sha256', 'sha512'
        
    Returns:
        Hash-ul ca string hex
    """
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def verify_file_hash(filepath: str | Path, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """Verifică dacă hash-ul fișierului corespunde."""
    actual_hash = compute_file_hash(filepath, algorithm)
    return actual_hash == expected_hash


@dataclass
class DataManifest:
    """
    Manifest pentru verificarea integrității datelor.
    
    Folosire:
        manifest = DataManifest()
        manifest.add_file("data/train.csv")
        manifest.add_file("data/test.csv")
        manifest.save("data/MANIFEST.json")
        
        # Mai târziu, verificare:
        manifest = DataManifest.load("data/MANIFEST.json")
        assert manifest.verify_all()
    """
    files: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    algorithm: str = 'sha256'
    
    def add_file(self, filepath: str | Path) -> None:
        """Adaugă un fișier la manifest."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        hash_value = compute_file_hash(path, self.algorithm)
        self.files[str(path)] = hash_value
    
    def verify_file(self, filepath: str | Path) -> bool:
        """Verifică un singur fișier."""
        path = str(Path(filepath))
        if path not in self.files:
            raise KeyError(f"File not in manifest: {filepath}")
        
        return verify_file_hash(path, self.files[path], self.algorithm)
    
    def verify_all(self) -> dict[str, bool]:
        """Verifică toate fișierele."""
        results = {}
        for filepath in self.files:
            try:
                results[filepath] = self.verify_file(filepath)
            except FileNotFoundError:
                results[filepath] = False
        return results
    
    def save(self, filepath: str | Path) -> None:
        """Salvează manifest-ul ca JSON."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'DataManifest':
        """Încarcă manifest-ul din JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: EXPERIMENT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Configurație pentru un experiment."""
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Rezultatele unui experiment."""
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    
    def add_metric(self, name: str, value: float) -> None:
        self.metrics[name] = value
    
    def add_artifact(self, filepath: str) -> None:
        self.artifacts.append(filepath)
    
    def log(self, message: str) -> None:
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")


@dataclass
class Experiment:
    """
    Container pentru un experiment complet.
    
    Folosire:
        exp = Experiment(
            config=ExperimentConfig(
                name="baseline_model",
                parameters={"learning_rate": 0.001, "epochs": 100}
            )
        )
        
        with exp.run():
            # Training code
            exp.result.add_metric("accuracy", 0.95)
            exp.result.add_metric("loss", 0.05)
        
        exp.save("experiments/exp001.json")
    """
    config: ExperimentConfig
    result: ExperimentResult = field(default_factory=ExperimentResult)
    
    # Metadata
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: str | None = None
    
    # Environment
    python_version: str = field(default_factory=lambda: sys.version)
    platform: str = field(default_factory=lambda: sys.platform)
    
    def run(self) -> 'ExperimentContext':
        """Context manager pentru rularea experimentului."""
        return ExperimentContext(self)
    
    def save(self, filepath: str | Path) -> None:
        """Salvează experimentul ca JSON."""
        data = {
            'config': self.config.to_dict(),
            'result': asdict(self.result),
            'started_at': self.started_at,
            'finished_at': self.finished_at,
            'duration_seconds': self.duration_seconds,
            'status': self.status,
            'error_message': self.error_message,
            'python_version': self.python_version,
            'platform': self.platform,
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str | Path) -> 'Experiment':
        """Încarcă experimentul din JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        exp = cls(
            config=ExperimentConfig(**data['config']),
            result=ExperimentResult(**data['result']),
        )
        exp.started_at = data.get('started_at')
        exp.finished_at = data.get('finished_at')
        exp.duration_seconds = data.get('duration_seconds')
        exp.status = data.get('status', 'unknown')
        exp.error_message = data.get('error_message')
        
        return exp


class ExperimentContext:
    """Context manager pentru experimente."""
    
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
    
    def __enter__(self) -> Experiment:
        self.experiment.started_at = datetime.now().isoformat()
        self.experiment.status = "running"
        self._start_time = time.time()
        
        logging.info(f"Experiment started: {self.experiment.config.name}")
        return self.experiment
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.experiment.finished_at = datetime.now().isoformat()
        self.experiment.duration_seconds = time.time() - self._start_time
        
        if exc_type is not None:
            self.experiment.status = "failed"
            self.experiment.error_message = str(exc_val)
            logging.error(f"Experiment failed: {exc_val}")
        else:
            self.experiment.status = "completed"
            logging.info(f"Experiment completed in {self.experiment.duration_seconds:.2f}s")
        
        return False  # Nu suprimăm excepțiile


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: TIMING DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator pentru măsurarea timpului de execuție.
    
    Folosire:
        @timed
        def train_model():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} executed in {elapsed:.4f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0):
    """
    Decorator pentru retry la erori.
    
    Folosire:
        @retry(max_attempts=3, delay=2.0)
        def download_data():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logging.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: README GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_readme(
    project_name: str,
    description: str,
    requirements: list[str],
    usage: str,
    author: str = "",
    license: str = "MIT"
) -> str:
    """
    Generează un README.md standard.
    """
    requirements_str = '\n'.join(f'- {r}' for r in requirements)
    
    readme = f'''# {project_name}

{description}

## Installation

```bash
git clone https://github.com/username/{project_name.lower().replace(" ", "-")}.git
cd {project_name.lower().replace(" ", "-")}
pip install -e ".[dev]"
```

## Requirements

{requirements_str}

## Usage

{usage}

## Reproducibility

This project follows reproducibility best practices:

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
{project_name.lower().replace(" ", "-")}/
├── src/                # Source code
├── tests/              # Test files
├── data/               # Data files
├── experiments/        # Experiment logs
├── notebooks/          # Jupyter notebooks
└── docs/               # Documentation
```

## License

{license}

## Author

{author}
'''
    
    return readme


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA VI: DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_reproducibility() -> None:
    """Demonstrație: configurare reproducibilitate."""
    print("=" * 60)
    print("DEMO: Reproducibility Configuration")
    print("=" * 60)
    print()
    
    config = ReproducibilityConfig(seed=42)
    config.apply()
    
    # Verificare
    print("Random numbers with seed=42:")
    for i in range(5):
        print(f"  random.random() = {random.random():.6f}")
    
    print()
    
    # Reset și verificare reproducibilitate
    config.apply()
    print("After reset with same seed:")
    for i in range(5):
        print(f"  random.random() = {random.random():.6f}")
    
    print()


def demo_experiment_logging() -> None:
    """Demonstrație: logging experimente."""
    print("=" * 60)
    print("DEMO: Experiment Logging")
    print("=" * 60)
    print()
    
    exp = Experiment(
        config=ExperimentConfig(
            name="demo_experiment",
            description="Demonstration of experiment logging",
            parameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
            },
            tags=["demo", "test"]
        )
    )
    
    with exp.run():
        # Simulare training
        for epoch in range(3):
            time.sleep(0.1)  # Simulare work
            accuracy = 0.8 + epoch * 0.05 + random.random() * 0.02
            exp.result.add_metric(f"accuracy_epoch_{epoch}", accuracy)
        
        exp.result.add_metric("final_accuracy", 0.92)
        exp.result.log("Training completed successfully")
    
    print(f"Experiment: {exp.config.name}")
    print(f"Status: {exp.status}")
    print(f"Duration: {exp.duration_seconds:.2f}s")
    print(f"Metrics: {exp.result.metrics}")
    print()


def demo_data_integrity() -> None:
    """Demonstrație: verificare integritate date."""
    print("=" * 60)
    print("DEMO: Data Integrity Verification")
    print("=" * 60)
    print()
    
    # Creăm un fișier temporar
    test_file = Path("/tmp/test_data.txt")
    test_file.write_text("Hello, this is test data for integrity check!")
    
    # Calculăm hash
    file_hash = compute_file_hash(test_file)
    print(f"File: {test_file}")
    print(f"SHA256: {file_hash}")
    
    # Verificare
    is_valid = verify_file_hash(test_file, file_hash)
    print(f"Verification: {'PASS' if is_valid else 'FAIL'}")
    
    # Cleanup
    test_file.unlink()
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 7 LAB: REPRODUCIBILITY TOOLKIT")
    print("═" * 60 + "\n")
    
    demo_reproducibility()
    demo_experiment_logging()
    demo_data_integrity()
    
    print("=" * 60)
    print("Best practices pentru reproducibilitate:")
    print("  1. Setați TOATE seed-urile cu set_all_seeds(42)")
    print("  2. Verificați datele cu DataManifest")
    print("  3. Logați experimentele cu Experiment")
    print("  4. Generați documentație cu generate_readme()")
    print("=" * 60)
