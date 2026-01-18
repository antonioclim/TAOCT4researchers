"""09UNIT Laboratory Package: Exception Handling and Defensive Programming.

This package contains laboratory modules for exploring Python's exception
handling mechanisms and defensive programming techniques.

Modules:
    lab_09_01_exception_handling: Exception fundamentals, custom exceptions,
        context managers and resilience patterns.
    lab_09_02_defensive_programming: Design by contract, input validation,
        numerical resilience and defensive data processing.

Example:
    >>> from lab import lab_09_01_exception_handling as lab01
    >>> from lab import lab_09_02_defensive_programming as lab02
    >>> result = lab01.safe_divide(10, 2)
    >>> result
    5.0
"""

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

__all__ = [
    # Lab 01: Exception Handling
    "CircuitBreaker",
    "CircuitOpenError",
    "ComputationError",
    "ConfigurationError",
    "ContractViolationError",
    "ConvergenceError",
    "DatabaseConnection",
    "DataValidationError",
    "FileFormatError",
    "ManagedFile",
    "NumericalInstabilityError",
    "ResearchError",
    "Timer",
    "bulk_operation_with_partial_failure",
    "demonstrate_exception_hierarchy",
    "exception_chaining_demo",
    "exitstack_demo",
    "graceful_degradation",
    "logged_function",
    "parse_config",
    "retry_with_backoff",
    "safe_divide",
    "setup_logging",
    "temporary_directory",
    # Lab 02: Defensive Programming
    "CheckpointManager",
    "ValidationResult",
    "checkpoint_processor",
    "detect_numerical_instability",
    "invariant",
    "kahan_summation",
    "postcondition",
    "precondition",
    "resilient_csv_reader",
    "safe_float_comparison",
    "safe_json_load",
    "stable_mean",
    "stable_variance",
    "validate_collection",
    "validate_dataframe_schema",
    "validate_numeric_range",
    "validate_string_pattern",
]

__version__ = "1.0.0"
