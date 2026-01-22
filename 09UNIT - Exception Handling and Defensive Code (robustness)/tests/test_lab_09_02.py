"""Tests for lab_09_02_defensive_programming module.

This module contains comprehensive tests for defensive programming
implementations including design by contract, input validation,
numerical resilience and defensive data processing.

Run with: pytest tests/test_lab_09_02.py -v
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import pytest

from lab.lab_09_02_defensive_programming import (
    CheckpointManager,
    ValidationResult,
    checkpoint_processor,
    detect_numerical_instability,
    invariant,
    kahan_summation,
    postcondition,
    precondition,
    resilient_csv_reader,
    safe_float_comparison,
    safe_json_load,
    stable_mean,
    stable_variance,
    validate_collection,
    validate_dataframe_schema,
    validate_numeric_range,
    validate_string_pattern,
)
from lab.lab_09_01_exception_handling import ContractViolationError


# =============================================================================
# TEST DESIGN BY CONTRACT
# =============================================================================


class TestDesignByContract:
    """Tests for design by contract decorators."""
    
    def test_precondition_valid_input(self) -> None:
        """Test precondition decorator with valid input."""
        @precondition(lambda x: x > 0, "x must be positive")
        def square_root(x: float) -> float:
            return math.sqrt(x)
        
        result = square_root(4.0)
        assert result == 2.0
    
    def test_precondition_invalid_input(self) -> None:
        """Test precondition decorator with invalid input."""
        @precondition(lambda x: x > 0, "x must be positive")
        def square_root(x: float) -> float:
            return math.sqrt(x)
        
        with pytest.raises(ContractViolationError, match="Precondition failed"):
            square_root(-1.0)
    
    def test_postcondition_valid_result(self) -> None:
        """Test postcondition decorator with valid result."""
        @postcondition(lambda r: r >= 0, "result must be non-negative")
        def absolute(x: float) -> float:
            return abs(x)
        
        result = absolute(-5.0)
        assert result == 5.0
    
    def test_postcondition_invalid_result(self) -> None:
        """Test postcondition decorator with invalid result."""
        @postcondition(lambda r: r > 0, "result must be positive")
        def buggy_abs(x: float) -> float:
            return x  # Bug: doesn't actually compute absolute value
        
        with pytest.raises(ContractViolationError, match="Postcondition failed"):
            buggy_abs(-5.0)
    
    def test_invariant_maintained(self) -> None:
        """Test invariant decorator on class."""
        @invariant(lambda self: self.value >= 0, "value must be non-negative")
        class Counter:
            def __init__(self, initial: int = 0) -> None:
                self.value = initial
            
            def increment(self) -> None:
                self.value += 1
            
            def decrement(self) -> None:
                self.value -= 1
        
        counter = Counter(5)
        counter.increment()
        assert counter.value == 6
        
        counter.decrement()
        assert counter.value == 5
    
    def test_invariant_violated(self) -> None:
        """Test invariant decorator detects violation."""
        @invariant(lambda self: self.value >= 0, "value must be non-negative")
        class Counter:
            def __init__(self, initial: int = 0) -> None:
                self.value = initial
            
            def reset_negative(self) -> None:
                self.value = -1
        
        counter = Counter(5)
        
        with pytest.raises(ContractViolationError, match="Invariant violated"):
            counter.reset_negative()
    
    def test_combined_contracts(self) -> None:
        """Test combining precondition and postcondition."""
        @precondition(lambda x, y: y != 0, "divisor must not be zero")
        @postcondition(lambda r: not math.isnan(r), "result must not be NaN")
        def safe_divide(x: float, y: float) -> float:
            return x / y
        
        result = safe_divide(10.0, 2.0)
        assert result == 5.0
        
        with pytest.raises(ContractViolationError):
            safe_divide(10.0, 0.0)


# =============================================================================
# TEST INPUT VALIDATION
# =============================================================================


class TestInputValidation:
    """Tests for input validation functions."""
    
    def test_validate_numeric_range_valid(self) -> None:
        """Test validate_numeric_range with valid values."""
        result = validate_numeric_range(50, min_value=0, max_value=100)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_numeric_range_too_low(self) -> None:
        """Test validate_numeric_range with value below minimum."""
        result = validate_numeric_range(-5, min_value=0, max_value=100)
        assert not result.is_valid
        assert "below minimum" in result.errors[0].lower()
    
    def test_validate_numeric_range_too_high(self) -> None:
        """Test validate_numeric_range with value above maximum."""
        result = validate_numeric_range(150, min_value=0, max_value=100)
        assert not result.is_valid
        assert "above maximum" in result.errors[0].lower()
    
    def test_validate_numeric_range_nan(self) -> None:
        """Test validate_numeric_range with NaN."""
        result = validate_numeric_range(float("nan"), min_value=0, max_value=100)
        assert not result.is_valid
    
    def test_validate_string_pattern_valid(self) -> None:
        """Test validate_string_pattern with matching string."""
        result = validate_string_pattern(
            "ABC123",
            pattern=r"^[A-Z]{3}\d{3}$",
            field_name="code",
        )
        assert result.is_valid
    
    def test_validate_string_pattern_invalid(self) -> None:
        """Test validate_string_pattern with non-matching string."""
        result = validate_string_pattern(
            "abc",
            pattern=r"^[A-Z]{3}\d{3}$",
            field_name="code",
        )
        assert not result.is_valid
        assert "code" in result.errors[0].lower()
    
    def test_validate_collection_valid(self) -> None:
        """Test validate_collection with valid collection."""
        result = validate_collection(
            [1, 2, 3, 4, 5],
            min_length=1,
            max_length=10,
            element_validator=lambda x: x > 0,
        )
        assert result.is_valid
    
    def test_validate_collection_too_short(self) -> None:
        """Test validate_collection with collection too short."""
        result = validate_collection(
            [],
            min_length=1,
            max_length=10,
        )
        assert not result.is_valid
        assert "length" in result.errors[0].lower()
    
    def test_validate_collection_invalid_element(self) -> None:
        """Test validate_collection with invalid element."""
        result = validate_collection(
            [1, 2, -3, 4],
            min_length=1,
            max_length=10,
            element_validator=lambda x: x > 0,
        )
        assert not result.is_valid
    
    def test_validation_result_merge(self) -> None:
        """Test ValidationResult merge functionality."""
        result1 = ValidationResult(is_valid=False, errors=["Error 1"])
        result2 = ValidationResult(is_valid=False, errors=["Error 2"])
        
        merged = result1.merge(result2)
        assert not merged.is_valid
        assert len(merged.errors) == 2


# =============================================================================
# TEST NUMERICAL resilience
# =============================================================================


class TestNumericalResilience:
    """Tests for numerical resilience functions."""
    
    def test_safe_float_comparison_equal(self) -> None:
        """Test safe_float_comparison with equal values."""
        assert safe_float_comparison(0.1 + 0.2, 0.3)
    
    def test_safe_float_comparison_not_equal(self) -> None:
        """Test safe_float_comparison with different values."""
        assert not safe_float_comparison(1.0, 2.0)
    
    def test_safe_float_comparison_custom_tolerance(self) -> None:
        """Test safe_float_comparison with custom tolerance."""
        assert safe_float_comparison(1.0, 1.001, rel_tol=0.01)
        assert not safe_float_comparison(1.0, 1.001, rel_tol=0.0001)
    
    def test_safe_float_comparison_near_zero(self) -> None:
        """Test safe_float_comparison near zero."""
        assert safe_float_comparison(1e-15, 2e-15, abs_tol=1e-14)
    
    def test_detect_numerical_instability_nan(self) -> None:
        """Test detect_numerical_instability with NaN."""
        result = detect_numerical_instability([1.0, float("nan"), 3.0])
        assert not result["is_stable"]
        assert result["has_nan"]
    
    def test_detect_numerical_instability_inf(self) -> None:
        """Test detect_numerical_instability with infinity."""
        result = detect_numerical_instability([1.0, float("inf"), 3.0])
        assert not result["is_stable"]
        assert result["has_inf"]
    
    def test_detect_numerical_instability_stable(self) -> None:
        """Test detect_numerical_instability with stable values."""
        result = detect_numerical_instability([1.0, 2.0, 3.0])
        assert result["is_stable"]
    
    def test_detect_numerical_instability_extreme_values(self) -> None:
        """Test detect_numerical_instability with extreme values."""
        result = detect_numerical_instability([1e-320, 1e308])
        assert "warnings" in result
    
    def test_kahan_summation_accuracy(self) -> None:
        """Test kahan_summation accuracy for many small values."""
        # Sum of 0.1 ten thousand times
        values = [0.1] * 10000
        
        naive_sum = sum(values)
        kahan_sum = kahan_summation(values)
        
        expected = 1000.0
        
        # Kahan should be more accurate
        kahan_error = abs(kahan_sum - expected)
        naive_error = abs(naive_sum - expected)
        
        assert kahan_error <= naive_error
    
    def test_kahan_summation_empty(self) -> None:
        """Test kahan_summation with empty list."""
        result = kahan_summation([])
        assert result == 0.0
    
    def test_kahan_summation_single(self) -> None:
        """Test kahan_summation with single value."""
        result = kahan_summation([42.0])
        assert result == 42.0
    
    def test_stable_mean_accuracy(self) -> None:
        """Test stable_mean accuracy."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = stable_mean(values)
        assert result == pytest.approx(3.0)
    
    def test_stable_mean_empty(self) -> None:
        """Test stable_mean with empty list."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            stable_mean([])
    
    def test_stable_variance_accuracy(self) -> None:
        """Test stable_variance accuracy."""
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        result = stable_variance(values)
        # Population variance
        expected = 4.0
        assert result == pytest.approx(expected, rel=0.01)
    
    def test_stable_variance_constant(self) -> None:
        """Test stable_variance with constant values."""
        values = [5.0, 5.0, 5.0, 5.0]
        result = stable_variance(values)
        assert result == pytest.approx(0.0)


# =============================================================================
# TEST DEFENSIVE DATA PROCESSING
# =============================================================================


class TestDefensiveDataProcessing:
    """Tests for defensive data processing functions."""
    
    def test_safe_json_load_valid(self, valid_json_file: Path) -> None:
        """Test safe_json_load with valid JSON."""
        result = safe_json_load(valid_json_file)
        assert result is not None
        assert "database" in result
    
    def test_safe_json_load_invalid(self, invalid_json_file: Path) -> None:
        """Test safe_json_load with invalid JSON."""
        result = safe_json_load(invalid_json_file)
        assert result is None
    
    def test_safe_json_load_missing_file(self, temp_dir: Path) -> None:
        """Test safe_json_load with missing file."""
        result = safe_json_load(temp_dir / "nonexistent.json")
        assert result is None
    
    def test_safe_json_load_with_default(self, temp_dir: Path) -> None:
        """Test safe_json_load returns default on error."""
        default = {"default": "value"}
        result = safe_json_load(temp_dir / "nonexistent.json", default=default)
        assert result == default
    
    def test_resilient_csv_reader_valid(self, valid_csv_file: Path) -> None:
        """Test resilient_csv_reader with valid CSV."""
        rows = list(resilient_csv_reader(valid_csv_file))
        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"
        assert rows[1]["age"] == "25"
    
    def test_resilient_csv_reader_malformed(self, malformed_csv_file: Path) -> None:
        """Test resilient_csv_reader handles malformed CSV gracefully."""
        rows = list(resilient_csv_reader(malformed_csv_file, skip_errors=True))
        # Should skip malformed rows
        assert len(rows) < 3
    
    def test_resilient_csv_reader_missing_file(self, temp_dir: Path) -> None:
        """Test resilient_csv_reader with missing file."""
        rows = list(resilient_csv_reader(temp_dir / "nonexistent.csv"))
        assert len(rows) == 0


# =============================================================================
# TEST CHECKPOINT MANAGER
# =============================================================================


class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    def test_checkpoint_save_load(self, checkpoint_path: Path) -> None:
        """Test checkpoint save and load cycle."""
        manager = CheckpointManager(checkpoint_path)
        manager.state["key"] = "value"
        manager.state["count"] = 42
        manager.save()
        
        # Load in new manager
        manager2 = CheckpointManager(checkpoint_path)
        manager2.load()
        
        assert manager2.state["key"] == "value"
        assert manager2.state["count"] == 42
    
    def test_checkpoint_atomic_save(self, checkpoint_path: Path) -> None:
        """Test checkpoint atomic save mechanism."""
        manager = CheckpointManager(checkpoint_path)
        manager.state["data"] = "test"
        manager.save()
        
        assert checkpoint_path.exists()
    
    def test_checkpoint_missing_file(self, checkpoint_path: Path) -> None:
        """Test checkpoint load with missing file."""
        manager = CheckpointManager(checkpoint_path)
        loaded = manager.load()
        
        assert not loaded
        assert manager.state == {}
    
    def test_checkpoint_update_state(self, checkpoint_path: Path) -> None:
        """Test checkpoint state updates."""
        manager = CheckpointManager(checkpoint_path)
        
        manager.state["iteration"] = 0
        manager.save()
        
        manager.state["iteration"] = 100
        manager.save()
        
        manager2 = CheckpointManager(checkpoint_path)
        manager2.load()
        
        assert manager2.state["iteration"] == 100


# =============================================================================
# TEST CHECKPOINT PROCESSOR
# =============================================================================


class TestCheckpointProcessor:
    """Tests for checkpoint_processor function."""
    
    def test_checkpoint_processor_completes(self, checkpoint_path: Path) -> None:
        """Test checkpoint_processor completes successfully."""
        items = list(range(10))
        
        def process(x: int) -> int:
            return x * 2
        
        results = checkpoint_processor(
            items,
            process,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=3,
        )
        
        assert len(results) == 10
        assert results[0] == 0
        assert results[9] == 18
    
    def test_checkpoint_processor_creates_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> None:
        """Test checkpoint_processor creates checkpoint file."""
        items = list(range(20))
        
        checkpoint_processor(
            items,
            lambda x: x,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=5,
        )
        
        assert checkpoint_path.exists()
    
    def test_checkpoint_processor_handles_errors(
        self,
        checkpoint_path: Path,
    ) -> None:
        """Test checkpoint_processor handles processing errors."""
        items = list(range(10))
        
        def process(x: int) -> int:
            if x == 5:
                raise ValueError(f"Cannot process {x}")
            return x * 2
        
        with pytest.raises(ValueError):
            checkpoint_processor(
                items,
                process,
                checkpoint_path=checkpoint_path,
                checkpoint_interval=3,
            )
        
        # Checkpoint should exist with partial progress
        assert checkpoint_path.exists()


# =============================================================================
# TEST DATAFRAME VALIDATION (IF AVAILABLE)
# =============================================================================


class TestDataframeValidation:
    """Tests for dataframe validation (requires pandas)."""
    
    @pytest.fixture
    def sample_dataframe(self) -> Any:
        """Create sample dataframe for testing."""
        try:
            import pandas as pd
            return pd.DataFrame({
                "name": ["Alice", "Bob", "Charlie"],
                "age": [30, 25, 35],
                "score": [0.95, 0.87, 0.92],
            })
        except ImportError:
            pytest.skip("pandas not available")
    
    def test_validate_dataframe_schema_valid(
        self,
        sample_dataframe: Any,
    ) -> None:
        """Test validate_dataframe_schema with valid dataframe."""
        schema = {
            "name": str,
            "age": int,
            "score": float,
        }
        
        result = validate_dataframe_schema(sample_dataframe, schema)
        assert result.is_valid
    
    def test_validate_dataframe_schema_missing_column(
        self,
        sample_dataframe: Any,
    ) -> None:
        """Test validate_dataframe_schema with missing column."""
        schema = {
            "name": str,
            "age": int,
            "score": float,
            "missing_column": str,
        }
        
        result = validate_dataframe_schema(sample_dataframe, schema)
        assert not result.is_valid
        assert "missing" in result.errors[0].lower()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple defensive programming techniques."""
    
    def test_validated_numerical_pipeline(self) -> None:
        """Test pipeline combining validation and numerical resilience."""
        # Input data
        raw_data = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Validate input
        validation = validate_collection(
            raw_data,
            min_length=1,
            max_length=1000,
            element_validator=lambda x: 0 <= x <= 1,
        )
        assert validation.is_valid
        
        # Process with numerical stability
        mean = stable_mean(raw_data)
        variance = stable_variance(raw_data)
        
        # Validate output
        assert safe_float_comparison(mean, 0.3, rel_tol=0.01)
        assert variance >= 0
    
    def test_resilient_file_processing(
        self,
        temp_dir: Path,
        sample_config: dict[str, Any],
    ) -> None:
        """Test consistent file processing with validation."""
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(sample_config))
        
        # Load with safe function
        config = safe_json_load(config_path)
        assert config is not None
        
        # Validate structure
        validation = validate_collection(
            list(config.keys()),
            min_length=1,
            max_length=100,
        )
        assert validation.is_valid
    
    def test_checkpoint_with_validation(
        self,
        checkpoint_path: Path,
    ) -> None:
        """Test checkpointed processing with validation."""
        items = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        @precondition(lambda x: 0 <= x <= 1, "x must be in [0, 1]")
        @postcondition(lambda r: 0 <= r <= 1, "result must be in [0, 1]")
        def normalise(x: float) -> float:
            return x / max(items)
        
        results = checkpoint_processor(
            items,
            normalise,
            checkpoint_path=checkpoint_path,
            checkpoint_interval=2,
        )
        
        assert len(results) == 5
        assert max(results) == pytest.approx(1.0)
