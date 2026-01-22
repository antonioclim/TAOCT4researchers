"""Tests for lab_09_01_exception_handling module.

This module contains comprehensive tests for exception handling
implementations including custom exceptions, context managers
and resilience patterns.

Run with: pytest tests/test_lab_09_01.py -v
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lab.lab_09_01_exception_handling import (
    CircuitBreaker,
    CircuitOpenError,
    ComputationError,
    ConfigurationError,
    ContractViolationError,
    ConvergenceError,
    DatabaseConnection,
    DataValidationError,
    FileFormatError,
    ManagedFile,
    NumericalInstabilityError,
    ResearchError,
    Timer,
    bulk_operation_with_partial_failure,
    demonstrate_exception_hierarchy,
    exception_chaining_demo,
    exitstack_demo,
    graceful_degradation,
    logged_function,
    parse_config,
    retry_with_backoff,
    safe_divide,
    setup_logging,
    temporary_directory,
)


# =============================================================================
# TEST EXCEPTION HIERARCHY
# =============================================================================


class TestExceptionHierarchy:
    """Tests for custom exception classes."""
    
    def test_research_error_base(self) -> None:
        """Test ResearchError base exception."""
        error = ResearchError("Test error", details={"key": "value"})
        assert str(error) == "Test error"
        assert error.details == {"key": "value"}
    
    def test_research_error_without_details(self) -> None:
        """Test ResearchError without details."""
        error = ResearchError("Simple error")
        assert error.details == {}
    
    def test_data_validation_error(self) -> None:
        """Test DataValidationError attributes."""
        error = DataValidationError(
            "Invalid field",
            field_name="temperature",
            expected_type=float,
            actual_value="not a number",
        )
        assert error.field_name == "temperature"
        assert error.expected_type is float
        assert error.actual_value == "not a number"
        assert "temperature" in str(error)
    
    def test_file_format_error(self) -> None:
        """Test FileFormatError attributes."""
        error = FileFormatError(
            "Parse failed",
            file_path=Path("data.csv"),
            line_number=42,
            expected_format="CSV",
        )
        assert error.file_path == Path("data.csv")
        assert error.line_number == 42
        assert error.expected_format == "CSV"
    
    def test_computation_error(self) -> None:
        """Test ComputationError attributes."""
        error = ComputationError(
            "Division failed",
            operation="matrix_inverse",
            input_shape=(3, 3),
        )
        assert error.operation == "matrix_inverse"
        assert error.input_shape == (3, 3)
    
    def test_numerical_instability_error(self) -> None:
        """Test NumericalInstabilityError attributes."""
        error = NumericalInstabilityError(
            "Overflow detected",
            value=float("inf"),
            threshold=1e308,
        )
        assert error.value == float("inf")
        assert error.threshold == 1e308
    
    def test_convergence_error(self) -> None:
        """Test ConvergenceError attributes."""
        error = ConvergenceError(
            "Failed to converge",
            iterations=1000,
            final_error=0.01,
            tolerance=0.001,
        )
        assert error.iterations == 1000
        assert error.final_error == 0.01
        assert error.tolerance == 0.001
    
    def test_exception_inheritance(self) -> None:
        """Test exception inheritance hierarchy."""
        assert issubclass(DataValidationError, ResearchError)
        assert issubclass(FileFormatError, ResearchError)
        assert issubclass(ComputationError, ResearchError)
        assert issubclass(NumericalInstabilityError, ComputationError)
        assert issubclass(ConvergenceError, ComputationError)
    
    def test_demonstrate_exception_hierarchy(self) -> None:
        """Test exception hierarchy demonstration function."""
        # Should not raise any exceptions
        demonstrate_exception_hierarchy()


# =============================================================================
# TEST SAFE OPERATIONS
# =============================================================================


class TestSafeOperations:
    """Tests for safe operation functions."""
    
    def test_safe_divide_normal(self) -> None:
        """Test safe_divide with normal inputs."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(7, 3) == pytest.approx(2.333, rel=0.01)
        assert safe_divide(-10, 2) == -5.0
    
    def test_safe_divide_by_zero(self) -> None:
        """Test safe_divide with zero denominator."""
        assert safe_divide(10, 0) is None
        assert safe_divide(10, 0, default=-1) == -1
    
    def test_safe_divide_with_floats(self) -> None:
        """Test safe_divide with float inputs."""
        assert safe_divide(1.5, 0.5) == 3.0
        assert safe_divide(0.1, 0.1) == pytest.approx(1.0)
    
    def test_parse_config_valid(self, valid_json_file: Path) -> None:
        """Test parse_config with valid JSON file."""
        config = parse_config(valid_json_file)
        assert config is not None
        assert "database" in config
    
    def test_parse_config_missing_file(self, temp_dir: Path) -> None:
        """Test parse_config with missing file."""
        result = parse_config(temp_dir / "nonexistent.json")
        assert result is None
    
    def test_parse_config_invalid_json(self, invalid_json_file: Path) -> None:
        """Test parse_config with invalid JSON."""
        result = parse_config(invalid_json_file)
        assert result is None


# =============================================================================
# TEST CONTEXT MANAGERS
# =============================================================================


class TestContextManagers:
    """Tests for context manager implementations."""
    
    def test_managed_file_write_read(self, temp_file: Path) -> None:
        """Test ManagedFile for write and read operations."""
        # Write
        with ManagedFile(temp_file, "w") as f:
            f.write("test content")
        
        assert temp_file.exists()
        
        # Read
        with ManagedFile(temp_file, "r") as f:
            content = f.read()
        
        assert content == "test content"
    
    def test_managed_file_exception_cleanup(self, temp_file: Path) -> None:
        """Test ManagedFile cleans up on exception."""
        with pytest.raises(ValueError):
            with ManagedFile(temp_file, "w") as f:
                f.write("partial")
                raise ValueError("Intentional error")
        
        # File should still be closed properly
        # (content may or may not be written depending on buffering)
    
    def test_temporary_directory(self) -> None:
        """Test temporary_directory context manager."""
        created_path: Path | None = None
        
        with temporary_directory() as tmpdir:
            created_path = tmpdir
            assert tmpdir.exists()
            assert tmpdir.is_dir()
            
            # Create a file inside
            test_file = tmpdir / "test.txt"
            test_file.write_text("content")
            assert test_file.exists()
        
        # Directory should be cleaned up
        assert created_path is not None
        assert not created_path.exists()
    
    def test_temporary_directory_with_prefix(self) -> None:
        """Test temporary_directory with custom prefix."""
        with temporary_directory(prefix="test_prefix_") as tmpdir:
            assert "test_prefix_" in tmpdir.name
    
    def test_database_connection_success(self) -> None:
        """Test DatabaseConnection with successful operations."""
        with DatabaseConnection("test_db") as conn:
            conn.execute("SELECT 1")
            conn.execute("SELECT 2")
        
        # Connection should be committed
    
    def test_database_connection_rollback(self) -> None:
        """Test DatabaseConnection rolls back on exception."""
        with pytest.raises(ValueError):
            with DatabaseConnection("test_db") as conn:
                conn.execute("SELECT 1")
                raise ValueError("Simulated error")
        
        # Connection should be rolled back
    
    def test_timer_context_manager(self) -> None:
        """Test Timer context manager."""
        with Timer("test_operation") as timer:
            time.sleep(0.05)
        
        assert timer.elapsed >= 0.05
        assert timer.elapsed < 0.5  # Should not be too long
    
    def test_timer_with_exception(self) -> None:
        """Test Timer records time even with exception."""
        with pytest.raises(ValueError):
            with Timer("failing_operation") as timer:
                time.sleep(0.02)
                raise ValueError("Error")
        
        assert timer.elapsed >= 0.02
    
    def test_exitstack_demo(self, temp_dir: Path) -> None:
        """Test exitstack_demo function."""
        paths = [temp_dir / f"file_{i}.txt" for i in range(3)]
        for p in paths:
            p.write_text(f"content of {p.name}")
        
        # Should not raise
        exitstack_demo(paths)


# =============================================================================
# TEST RESILIENCE PATTERNS
# =============================================================================


class TestResiliencePatterns:
    """Tests for resilience pattern implementations."""
    
    def test_retry_with_backoff_success_first_attempt(self) -> None:
        """Test retry_with_backoff succeeds on first attempt."""
        call_count = [0]
        
        def success_func() -> str:
            call_count[0] += 1
            return "success"
        
        result = retry_with_backoff(success_func, max_attempts=3)
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retry_with_backoff_eventual_success(self) -> None:
        """Test retry_with_backoff eventually succeeds."""
        call_count = [0]
        
        def flaky_func() -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = retry_with_backoff(
            flaky_func,
            max_attempts=5,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_with_backoff_all_failures(self) -> None:
        """Test retry_with_backoff raises after all attempts fail."""
        def always_fails() -> None:
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            retry_with_backoff(
                always_fails,
                max_attempts=3,
                base_delay=0.01,
            )
    
    def test_retry_with_backoff_non_retryable(self) -> None:
        """Test retry_with_backoff does not retry non-retryable exceptions."""
        call_count = [0]
        
        def fails_with_type_error() -> None:
            call_count[0] += 1
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            retry_with_backoff(
                fails_with_type_error,
                max_attempts=3,
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
            )
        
        assert call_count[0] == 1  # Should not have retried
    
    def test_circuit_breaker_closed_state(self) -> None:
        """Test CircuitBreaker in closed state."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == "closed"
    
    def test_circuit_breaker_opens_after_failures(self) -> None:
        """Test CircuitBreaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, reset_timeout=0.1)
        
        def failing_func() -> None:
            raise ConnectionError("Service down")
        
        for _ in range(3):
            with pytest.raises(ConnectionError):
                breaker.call(failing_func)
        
        assert breaker.state == "open"
    
    def test_circuit_breaker_rejects_when_open(self) -> None:
        """Test CircuitBreaker rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, reset_timeout=1.0)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        # Should reject immediately
        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: "should not execute")
    
    def test_circuit_breaker_half_open_recovery(self) -> None:
        """Test CircuitBreaker recovers through half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            success_threshold=2,
            reset_timeout=0.05,
        )
        
        # Open the circuit
        for _ in range(2):
            try:
                breaker.call(lambda: (_ for _ in ()).throw(ValueError()))
            except ValueError:
                pass
        
        assert breaker.state == "open"
        
        # Wait for reset timeout
        time.sleep(0.1)
        
        # Successful calls should close circuit
        breaker.call(lambda: "success")
        breaker.call(lambda: "success")
        
        assert breaker.state == "closed"
    
    def test_graceful_degradation(self) -> None:
        """Test graceful_degradation function."""
        def primary() -> str:
            raise ConnectionError("Primary failed")
        
        def fallback() -> str:
            return "fallback_result"
        
        result = graceful_degradation(primary, fallback)
        assert result == "fallback_result"
    
    def test_graceful_degradation_primary_succeeds(self) -> None:
        """Test graceful_degradation uses primary when it succeeds."""
        def primary() -> str:
            return "primary_result"
        
        def fallback() -> str:
            return "fallback_result"
        
        result = graceful_degradation(primary, fallback)
        assert result == "primary_result"
    
    def test_bulk_operation_with_partial_failure(self) -> None:
        """Test bulk_operation_with_partial_failure."""
        items = [1, 2, 3, 4, 5]
        
        def process(x: int) -> int:
            if x == 3:
                raise ValueError(f"Cannot process {x}")
            return x * 2
        
        results = bulk_operation_with_partial_failure(items, process)
        
        assert len(results["successful"]) == 4
        assert len(results["failed"]) == 1
        assert 3 in [f["item"] for f in results["failed"]]


# =============================================================================
# TEST LOGGING INTEGRATION
# =============================================================================


class TestLoggingIntegration:
    """Tests for logging integration."""
    
    def test_setup_logging(self, temp_dir: Path) -> None:
        """Test setup_logging function."""
        log_file = temp_dir / "test.log"
        logger = setup_logging("test_logger", log_file=log_file)
        
        logger.info("Test message")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
    
    def test_logged_function_decorator(self) -> None:
        """Test logged_function decorator."""
        @logged_function
        def sample_function(x: int, y: int) -> int:
            return x + y
        
        result = sample_function(2, 3)
        assert result == 5
    
    def test_logged_function_with_exception(self) -> None:
        """Test logged_function logs exceptions."""
        @logged_function
        def failing_function() -> None:
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()


# =============================================================================
# TEST EXCEPTION CHAINING
# =============================================================================


class TestExceptionChaining:
    """Tests for exception chaining functionality."""
    
    def test_exception_chaining_demo(self) -> None:
        """Test exception_chaining_demo function."""
        with pytest.raises(ConfigurationError) as exc_info:
            exception_chaining_demo()
        
        # Should have a cause
        assert exc_info.value.__cause__ is not None
    
    def test_exception_cause_preserved(self) -> None:
        """Test that exception cause is properly preserved."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise RuntimeError("Wrapped error") from e
        except RuntimeError as e:
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow_with_error_handling(
        self,
        temp_dir: Path,
        sample_config: dict[str, Any],
    ) -> None:
        """Test complete workflow with comprehensive error handling."""
        import json
        
        # Create config file
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(sample_config))
        
        # Load and validate
        config = parse_config(config_path)
        assert config is not None
        
        # Process with timing
        with Timer("processing") as timer:
            with temporary_directory() as work_dir:
                # Create work file
                work_file = work_dir / "data.txt"
                work_file.write_text("processed data")
                
                # Verify
                assert work_file.exists()
        
        assert timer.elapsed > 0
    
    def test_resilient_processing_pipeline(self) -> None:
        """Test consistent processing with retry and circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=5, reset_timeout=0.1)
        call_count = [0]
        
        def unreliable_service() -> str:
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Service temporarily unavailable")
            return "data"
        
        # Use retry with circuit breaker
        result = retry_with_backoff(
            lambda: breaker.call(unreliable_service),
            max_attempts=5,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        
        assert result == "data"
        assert breaker.state == "closed"
