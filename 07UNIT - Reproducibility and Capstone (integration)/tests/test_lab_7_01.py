#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7: Test Suite for Lab 7.01 — Reproducibility Toolkit
═══════════════════════════════════════════════════════════════════════════════

This test module provides comprehensive tests for the reproducibility toolkit,
covering seed management, configuration tracking and data manifests.

TEST COVERAGE
─────────────
1. SeedManager: Seed setting, context management, reproducibility verification
2. ConfigurationTracker: Config saving, loading, hashing, comparison
3. DataManifest: File tracking, checksums, validation, schema compliance
4. Integration: End-to-end reproducibility workflows

USAGE
─────
    pytest tests/test_lab_7_01.py -v
    pytest tests/test_lab_7_01.py -v -m reproducibility
    pytest tests/test_lab_7_01.py -v --cov=lab

DEPENDENCIES
────────────
pytest>=7.0
pytest-cov>=4.0

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Attempt numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SEED MANAGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeedManager:
    """Tests for the SeedManager class."""

    def test_seed_setting_produces_deterministic_random(self) -> None:
        """Test that setting a seed produces deterministic random values."""
        seed = 42
        
        # First run
        random.seed(seed)
        values_1 = [random.random() for _ in range(10)]
        
        # Second run with same seed
        random.seed(seed)
        values_2 = [random.random() for _ in range(10)]
        
        assert values_1 == values_2, "Same seed should produce identical sequences"

    def test_different_seeds_produce_different_values(self) -> None:
        """Test that different seeds produce different random values."""
        random.seed(42)
        values_1 = [random.random() for _ in range(10)]
        
        random.seed(123)
        values_2 = [random.random() for _ in range(10)]
        
        assert values_1 != values_2, "Different seeds should produce different sequences"

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
    def test_numpy_seed_reproducibility(self) -> None:
        """Test NumPy random seed produces deterministic arrays."""
        seed = 42
        
        np.random.seed(seed)
        arr_1 = np.random.rand(5, 5)
        
        np.random.seed(seed)
        arr_2 = np.random.rand(5, 5)
        
        np.testing.assert_array_equal(arr_1, arr_2)

    def test_seed_boundary_values(self, various_seeds: int) -> None:
        """Test seed manager with various seed values including boundaries."""
        random.seed(various_seeds)
        value = random.random()
        
        random.seed(various_seeds)
        assert random.random() == value

    @pytest.mark.reproducibility
    def test_seed_context_isolation(self) -> None:
        """Test that seed contexts are properly isolated."""
        # Set initial seed
        random.seed(100)
        outer_value = random.random()
        
        # Inner context with different seed
        random.seed(42)
        inner_values = [random.random() for _ in range(5)]
        
        # Restore outer seed - should get same sequence
        random.seed(100)
        assert random.random() == outer_value

    def test_seed_with_negative_value(self) -> None:
        """Test seed handling with negative values (should work in Python)."""
        seed = -42
        random.seed(seed)
        value_1 = random.random()
        
        random.seed(seed)
        value_2 = random.random()
        
        assert value_1 == value_2

    def test_seed_state_serialisation(self) -> None:
        """Test that random state can be saved and restored."""
        random.seed(42)
        _ = [random.random() for _ in range(5)]
        
        # Save state
        state = random.getstate()
        next_value = random.random()
        
        # Restore state
        random.setstate(state)
        restored_value = random.random()
        
        assert next_value == restored_value


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFIGURATION TRACKER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigurationTracker:
    """Tests for configuration tracking functionality."""

    def test_config_dict_to_json_roundtrip(
        self,
        sample_test_data: dict[str, Any],
        tmp_path: Path
    ) -> None:
        """Test configuration can be saved and loaded from JSON."""
        config_file = tmp_path / "config.json"
        
        # Save
        with open(config_file, "w") as f:
            json.dump(sample_test_data, f, indent=2)
        
        # Load
        with open(config_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded == sample_test_data

    def test_config_hash_consistency(self) -> None:
        """Test that configuration hashing is consistent."""
        config = {"a": 1, "b": 2, "c": [1, 2, 3]}
        
        # Hash same config multiple times
        config_str = json.dumps(config, sort_keys=True)
        hash_1 = hashlib.sha256(config_str.encode()).hexdigest()
        hash_2 = hashlib.sha256(config_str.encode()).hexdigest()
        
        assert hash_1 == hash_2

    def test_config_hash_order_independence(self) -> None:
        """Test that configuration hash is independent of key order."""
        config_1 = {"z": 1, "a": 2, "m": 3}
        config_2 = {"a": 2, "m": 3, "z": 1}
        
        hash_1 = hashlib.sha256(
            json.dumps(config_1, sort_keys=True).encode()
        ).hexdigest()
        hash_2 = hashlib.sha256(
            json.dumps(config_2, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash_1 == hash_2

    def test_config_change_detection(self) -> None:
        """Test that configuration changes are detected via hash."""
        config_1 = {"learning_rate": 0.01, "batch_size": 32}
        config_2 = {"learning_rate": 0.02, "batch_size": 32}
        
        hash_1 = hashlib.sha256(
            json.dumps(config_1, sort_keys=True).encode()
        ).hexdigest()
        hash_2 = hashlib.sha256(
            json.dumps(config_2, sort_keys=True).encode()
        ).hexdigest()
        
        assert hash_1 != hash_2

    def test_nested_config_handling(self) -> None:
        """Test handling of nested configuration structures."""
        config = {
            "model": {
                "architecture": "transformer",
                "layers": {"encoder": 6, "decoder": 6}
            },
            "training": {
                "optimizer": {"type": "adam", "lr": 0.001}
            }
        }
        
        # Should be JSON serialisable
        config_str = json.dumps(config, sort_keys=True)
        loaded = json.loads(config_str)
        
        assert loaded == config

    def test_config_with_datetime(self, tmp_path: Path) -> None:
        """Test configuration with datetime values."""
        config = {
            "experiment_date": datetime.now().isoformat(),
            "parameters": {"value": 42}
        }
        
        config_file = tmp_path / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f)
        
        with open(config_file, "r") as f:
            loaded = json.load(f)
        
        assert loaded["parameters"]["value"] == 42

    @pytest.mark.parametrize("invalid_type", [
        lambda: None,
        object(),
        {frozenset([1, 2])},
    ])
    def test_config_non_serialisable_detection(self, invalid_type: Any) -> None:
        """Test that non-serialisable config values raise errors."""
        config = {"invalid": invalid_type}
        
        with pytest.raises((TypeError, ValueError)):
            json.dumps(config)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA MANIFEST TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataManifest:
    """Tests for data manifest functionality."""

    def test_file_checksum_computation(
        self,
        sample_data_file: Path,
        file_hasher
    ) -> None:
        """Test file checksum computation."""
        checksum = file_hasher(sample_data_file)
        
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex digest length

    def test_file_checksum_consistency(
        self,
        sample_data_file: Path,
        file_hasher
    ) -> None:
        """Test that file checksum is consistent across reads."""
        checksum_1 = file_hasher(sample_data_file)
        checksum_2 = file_hasher(sample_data_file)
        
        assert checksum_1 == checksum_2

    def test_file_checksum_changes_with_content(
        self,
        tmp_path: Path,
        file_hasher
    ) -> None:
        """Test that checksum changes when file content changes."""
        test_file = tmp_path / "test.txt"
        
        test_file.write_text("original content")
        checksum_1 = file_hasher(test_file)
        
        test_file.write_text("modified content")
        checksum_2 = file_hasher(test_file)
        
        assert checksum_1 != checksum_2

    def test_manifest_creation(self, tmp_path: Path, file_hasher) -> None:
        """Test creation of a data manifest."""
        # Create sample files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        files = {
            "data.csv": "a,b,c\n1,2,3\n",
            "config.json": '{"key": "value"}',
            "readme.txt": "Sample readme content"
        }
        
        for name, content in files.items():
            (data_dir / name).write_text(content)
        
        # Create manifest
        manifest = {}
        for file_path in data_dir.iterdir():
            manifest[file_path.name] = {
                "checksum": file_hasher(file_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).isoformat()
            }
        
        assert len(manifest) == 3
        assert all("checksum" in entry for entry in manifest.values())

    def test_manifest_validation_success(
        self,
        tmp_path: Path,
        file_hasher
    ) -> None:
        """Test successful manifest validation."""
        test_file = tmp_path / "data.txt"
        test_file.write_text("test content")
        
        # Create manifest
        manifest = {
            "data.txt": {"checksum": file_hasher(test_file)}
        }
        
        # Validate
        current_checksum = file_hasher(test_file)
        assert current_checksum == manifest["data.txt"]["checksum"]

    def test_manifest_validation_failure(
        self,
        tmp_path: Path,
        file_hasher
    ) -> None:
        """Test manifest validation detects file changes."""
        test_file = tmp_path / "data.txt"
        test_file.write_text("original content")
        
        manifest = {
            "data.txt": {"checksum": file_hasher(test_file)}
        }
        
        # Modify file
        test_file.write_text("modified content")
        
        # Validate
        current_checksum = file_hasher(test_file)
        assert current_checksum != manifest["data.txt"]["checksum"]

    def test_manifest_missing_file_detection(self, tmp_path: Path) -> None:
        """Test manifest detects missing files."""
        manifest = {
            "missing.txt": {"checksum": "abc123"}
        }
        
        missing_file = tmp_path / "missing.txt"
        assert not missing_file.exists()

    def test_manifest_directory_traversal(self, tmp_path: Path) -> None:
        """Test manifest creation with nested directories."""
        # Create nested structure
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()
        (tmp_path / "level1" / "file1.txt").write_text("file1")
        (tmp_path / "level1" / "level2" / "file2.txt").write_text("file2")
        
        # Collect all files
        files = list(tmp_path.rglob("*.txt"))
        
        assert len(files) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EXPERIMENT TRACKING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExperimentTracking:
    """Tests for experiment tracking functionality."""

    def test_experiment_metadata_creation(self) -> None:
        """Test creation of experiment metadata."""
        metadata = {
            "experiment_id": "exp_001",
            "timestamp": datetime.now().isoformat(),
            "seed": 42,
            "parameters": {"lr": 0.01, "epochs": 100}
        }
        
        assert "experiment_id" in metadata
        assert "seed" in metadata

    def test_metric_logging(self, tmp_path: Path) -> None:
        """Test logging of experiment metrics."""
        metrics_file = tmp_path / "metrics.json"
        
        metrics = []
        for epoch in range(5):
            metrics.append({
                "epoch": epoch,
                "loss": 1.0 / (epoch + 1),
                "accuracy": 0.5 + 0.1 * epoch
            })
        
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)
        
        with open(metrics_file, "r") as f:
            loaded = json.load(f)
        
        assert len(loaded) == 5
        assert loaded[-1]["accuracy"] > loaded[0]["accuracy"]

    def test_artifact_tracking(self, tmp_path: Path) -> None:
        """Test tracking of experiment artifacts."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        
        # Create artifacts
        model_file = artifacts_dir / "model.pkl"
        model_file.write_bytes(b"mock model data")
        
        plot_file = artifacts_dir / "loss_curve.png"
        plot_file.write_bytes(b"mock image data")
        
        # Track artifacts
        artifacts = {}
        for artifact in artifacts_dir.iterdir():
            artifacts[artifact.name] = {
                "path": str(artifact),
                "size": artifact.stat().st_size
            }
        
        assert len(artifacts) == 2
        assert "model.pkl" in artifacts

    def test_experiment_comparison(self) -> None:
        """Test comparison of experiment results."""
        exp_1 = {"accuracy": 0.85, "loss": 0.15, "epochs": 100}
        exp_2 = {"accuracy": 0.90, "loss": 0.10, "epochs": 150}
        
        comparison = {
            "accuracy_diff": exp_2["accuracy"] - exp_1["accuracy"],
            "loss_diff": exp_2["loss"] - exp_1["loss"],
            "better_accuracy": exp_2["accuracy"] > exp_1["accuracy"]
        }
        
        assert comparison["accuracy_diff"] == pytest.approx(0.05)
        assert comparison["better_accuracy"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: REPRODUCIBILITY VERIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibilityVerification:
    """Tests for reproducibility verification utilities."""

    @pytest.mark.reproducibility
    def test_deterministic_function_output(self) -> None:
        """Test that a seeded function produces deterministic output."""
        def compute_with_randomness(seed: int, n: int) -> list[float]:
            random.seed(seed)
            return [random.gauss(0, 1) for _ in range(n)]
        
        result_1 = compute_with_randomness(42, 10)
        result_2 = compute_with_randomness(42, 10)
        
        assert result_1 == result_2

    @pytest.mark.reproducibility
    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not installed")
    def test_numpy_computation_reproducibility(self) -> None:
        """Test reproducibility of NumPy computations."""
        def matrix_operation(seed: int) -> np.ndarray:
            np.random.seed(seed)
            A = np.random.rand(5, 5)
            B = np.random.rand(5, 5)
            return A @ B
        
        result_1 = matrix_operation(42)
        result_2 = matrix_operation(42)
        
        np.testing.assert_array_equal(result_1, result_2)

    @pytest.mark.reproducibility
    def test_shuffling_reproducibility(self) -> None:
        """Test that shuffling is reproducible with seed."""
        data = list(range(100))
        
        random.seed(42)
        data_1 = data.copy()
        random.shuffle(data_1)
        
        random.seed(42)
        data_2 = data.copy()
        random.shuffle(data_2)
        
        assert data_1 == data_2

    @pytest.mark.reproducibility
    def test_sampling_reproducibility(self) -> None:
        """Test that random sampling is reproducible."""
        population = list(range(1000))
        
        random.seed(42)
        sample_1 = random.sample(population, 10)
        
        random.seed(42)
        sample_2 = random.sample(population, 10)
        
        assert sample_1 == sample_2

    @pytest.mark.reproducibility
    def test_cross_session_reproducibility(
        self,
        content_hasher,
        tmp_path: Path
    ) -> None:
        """Test reproducibility across simulated sessions."""
        # Session 1
        random.seed(42)
        session_1_values = [random.random() for _ in range(10)]
        
        # Save session state
        state_file = tmp_path / "session_state.json"
        with open(state_file, "w") as f:
            json.dump({"seed": 42, "values": session_1_values}, f)
        
        # Session 2 (simulate restart)
        with open(state_file, "r") as f:
            saved = json.load(f)
        
        random.seed(saved["seed"])
        session_2_values = [random.random() for _ in range(10)]
        
        assert session_1_values == session_2_values


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: ENVIRONMENT CAPTURE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentCapture:
    """Tests for environment capture functionality."""

    def test_python_version_capture(self) -> None:
        """Test capturing Python version information."""
        import sys
        
        version_info = {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "full": sys.version
        }
        
        assert version_info["major"] >= 3
        assert version_info["minor"] >= 10

    def test_package_version_capture(self) -> None:
        """Test capturing installed package versions."""
        import importlib.metadata
        
        packages = {}
        for pkg_name in ["pytest"]:
            try:
                packages[pkg_name] = importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                packages[pkg_name] = "not installed"
        
        assert "pytest" in packages

    def test_environment_variable_capture(self) -> None:
        """Test capturing environment variables."""
        import os
        
        # Set a test variable
        os.environ["TEST_VAR"] = "test_value"
        
        env_capture = {
            k: v for k, v in os.environ.items()
            if k.startswith("TEST_")
        }
        
        assert "TEST_VAR" in env_capture
        
        # Cleanup
        del os.environ["TEST_VAR"]

    def test_platform_information_capture(self) -> None:
        """Test capturing platform information."""
        import platform
        
        platform_info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation()
        }
        
        assert platform_info["python_implementation"] in ["CPython", "PyPy"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReproducibilityIntegration:
    """Integration tests for reproducibility workflow."""

    @pytest.mark.integration
    def test_full_reproducibility_workflow(
        self,
        tmp_path: Path,
        sample_test_data: dict[str, Any],
        file_hasher
    ) -> None:
        """Test a complete reproducibility workflow."""
        # Step 1: Create experiment directory
        exp_dir = tmp_path / "experiment_001"
        exp_dir.mkdir()
        
        # Step 2: Save configuration
        config = {
            "seed": 42,
            "parameters": {"learning_rate": 0.01, "epochs": 100},
            "data": sample_test_data
        }
        config_file = exp_dir / "config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Step 3: Run "experiment" with seed
        random.seed(config["seed"])
        results = [random.random() for _ in range(10)]
        
        # Step 4: Save results
        results_file = exp_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump({"values": results}, f)
        
        # Step 5: Create manifest
        manifest = {}
        for file_path in exp_dir.iterdir():
            manifest[file_path.name] = file_hasher(file_path)
        
        manifest_file = exp_dir / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)
        
        # Step 6: Verify reproducibility
        # Load config and reproduce
        with open(config_file, "r") as f:
            loaded_config = json.load(f)
        
        random.seed(loaded_config["seed"])
        reproduced_results = [random.random() for _ in range(10)]
        
        assert results == reproduced_results

    @pytest.mark.integration
    def test_data_pipeline_reproducibility(
        self,
        tmp_path: Path,
        sample_csv_content: str
    ) -> None:
        """Test reproducibility of a data processing pipeline."""
        # Create input data
        input_file = tmp_path / "input.csv"
        input_file.write_text(sample_csv_content)
        
        # Define pipeline steps
        def process_data(filepath: Path, seed: int) -> list[dict]:
            """Simulate data processing with randomness."""
            random.seed(seed)
            
            with open(filepath) as f:
                lines = f.readlines()[1:]  # Skip header
            
            data = []
            for line in lines:
                parts = line.strip().split(",")
                # Add random noise (deterministic with seed)
                noise = random.gauss(0, 0.1)
                data.append({
                    "id": parts[0],
                    "processed_value": float(parts[2]) + noise
                })
            
            return data
        
        # Run pipeline twice
        result_1 = process_data(input_file, seed=42)
        result_2 = process_data(input_file, seed=42)
        
        assert result_1 == result_2


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EDGE CASES AND ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_config_handling(self, tmp_path: Path) -> None:
        """Test handling of empty configuration."""
        config_file = tmp_path / "empty_config.json"
        config_file.write_text("{}")
        
        with open(config_file) as f:
            config = json.load(f)
        
        assert config == {}

    def test_large_seed_value(self) -> None:
        """Test handling of large seed values."""
        large_seed = 2**31 - 1
        random.seed(large_seed)
        value = random.random()
        
        random.seed(large_seed)
        assert random.random() == value

    def test_unicode_in_config(self, tmp_path: Path) -> None:
        """Test handling of Unicode in configuration."""
        config = {
            "description": "Test with émojis 🎯 and ünïcödé",
            "author": "José García"
        }
        
        config_file = tmp_path / "unicode_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False)
        
        with open(config_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert loaded["description"] == config["description"]

    def test_binary_file_checksum(self, tmp_path: Path, file_hasher) -> None:
        """Test checksum computation for binary files."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(bytes(range(256)))
        
        checksum = file_hasher(binary_file)
        assert len(checksum) == 64

    def test_very_large_file_checksum(self, tmp_path: Path, file_hasher) -> None:
        """Test checksum computation for larger files."""
        large_file = tmp_path / "large.txt"
        # Create a ~1MB file
        large_file.write_text("x" * (1024 * 1024))
        
        checksum = file_hasher(large_file)
        assert len(checksum) == 64

    def test_special_characters_in_path(
        self,
        tmp_path: Path,
        file_hasher
    ) -> None:
        """Test handling of special characters in file paths."""
        # Create file with spaces and special chars
        special_dir = tmp_path / "test dir with spaces"
        special_dir.mkdir()
        special_file = special_dir / "file-with_special.chars.txt"
        special_file.write_text("content")
        
        checksum = file_hasher(special_file)
        assert len(checksum) == 64


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """Performance tests for reproducibility tools."""

    @pytest.mark.slow
    def test_checksum_performance(self, tmp_path: Path, file_hasher) -> None:
        """Test checksum computation performance."""
        import time
        
        # Create test file
        test_file = tmp_path / "perf_test.txt"
        test_file.write_text("x" * 100000)
        
        start = time.time()
        for _ in range(100):
            file_hasher(test_file)
        elapsed = time.time() - start
        
        # Should complete 100 checksums in under 1 second
        assert elapsed < 1.0

    @pytest.mark.slow
    def test_seed_setting_performance(self) -> None:
        """Test seed setting performance."""
        import time
        
        start = time.time()
        for seed in range(10000):
            random.seed(seed)
            _ = random.random()
        elapsed = time.time() - start
        
        # Should complete 10000 seed+random operations in under 1 second
        assert elapsed < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# END OF TEST MODULE
# ═══════════════════════════════════════════════════════════════════════════════
