# 09UNIT: Homework Assignment

## Exception Handling and Defensive Programming

**Due Date**: One week from assignment  
**Submission**: Submit all Python files via course platform  
**Weight**: 20% of unit grade

---

## Overview

This assignment develops practical competence in exception handling and defensive programming through three progressive parts. You will implement reliable error handling for a scientific data processing pipeline, design custom exception hierarchies and apply resilience patterns to external service integration.

**Total Points**: 100

| Part | Topic | Points | Est. Time |
|------|-------|--------|-----------|
| Part 1 | Exception Handling Pipeline | 35 | 60 min |
| Part 2 | Context Managers and Validation | 35 | 60 min |
| Part 3 | Resilience Patterns | 30 | 45 min |
| Bonus | Advanced Checkpoint Recovery | +10 | 30 min |

---

## Part 1: Exception Handling Pipeline (35 points)

### Background

Research data often arrives in inconsistent formats with missing values, encoding issues and structural problems. Your task is to implement a resilient data processing pipeline that handles these challenges gracefully.

### Requirements

Create a module `homework_part1.py` containing:

#### 1.1 Custom Exception Hierarchy (10 points)

Design a hierarchy of exceptions for data processing:

```python
class DataPipelineError(Exception):
    """Base exception for data pipeline errors."""
    pass

class DataSourceError(DataPipelineError):
    """Raised when data cannot be retrieved from source."""
    pass

class DataParseError(DataPipelineError):
    """Raised when data parsing fails."""
    pass

class DataValidationError(DataPipelineError):
    """Raised when data fails validation."""
    pass

class DataTransformError(DataPipelineError):
    """Raised when data transformation fails."""
    pass
```

Each exception class must:
- Include informative attributes (source, field, value as appropriate)
- Provide a meaningful `__str__` representation
- Include proper type hints

#### 1.2 Resilient Data Loader (15 points)

Implement a function that loads data from various sources:

```python
def load_research_data(
    source: Path | str,
    *,
    format: str = "auto",
    encoding: str = "utf-8",
    fallback_encodings: tuple[str, ...] = ("latin-1", "cp1252"),
) -> list[dict[str, Any]]:
    """Load research data from file with reliable error handling.
    
    Args:
        source: Path to data file.
        format: Data format ("csv", "json", "auto" for detection).
        encoding: Primary encoding.
        fallback_encodings: Encodings to try if primary fails.
        
    Returns:
        List of data records as dictionaries.
        
    Raises:
        DataSourceError: If file cannot be accessed.
        DataParseError: If file cannot be parsed.
    """
```

Requirements:
- Auto-detect format from file extension when format="auto"
- Try multiple encodings before failing
- Use exception chaining to preserve original errors
- Log warnings for recovered errors
- Handle CSV and JSON formats

#### 1.3 Validation Pipeline (10 points)

Implement data validation:

```python
def validate_research_record(
    record: dict[str, Any],
    schema: dict[str, type],
    *,
    required_fields: set[str] | None = None,
    value_ranges: dict[str, tuple[float, float]] | None = None,
) -> ValidationResult:
    """Validate a single research record against schema.
    
    Args:
        record: Data record to validate.
        schema: Expected types for fields.
        required_fields: Fields that must be present.
        value_ranges: Numeric ranges for specified fields.
        
    Returns:
        ValidationResult with is_valid, errors and warnings.
    """
```

### Deliverables

- `homework_part1.py` with all implementations
- Docstrings with examples for each function
- Type hints throughout
- At least 5 unit tests in `test_homework_part1.py`

### Grading Rubric

| Criterion | Points |
|-----------|--------|
| Exception hierarchy design | 10 |
| Data loader implementation | 15 |
| Validation implementation | 10 |
| **Part 1 Total** | **35** |

---

## Part 2: Context Managers and Validation (35 points)

### Background

Research in software engineering consistently shows that modular designs with high cohesion (functions focused on single tasks) and low coupling (minimal dependencies between functions) lead to more maintainable, resilient code (Yourdon & Constantine, 1979, p. 85). This part develops context managers for resource management and validation frameworks.

### Requirements

Create a module `homework_part2.py` containing:

#### 2.1 Experiment Context Manager (15 points)

Implement a context manager for research experiments:

```python
@dataclass
class ExperimentContext:
    """Context manager for research experiment execution.
    
    Manages experiment setup, logging and cleanup with guaranteed
    resource release and result persistence.
    
    Attributes:
        experiment_id: Unique experiment identifier.
        output_dir: Directory for experiment outputs.
        config: Experiment configuration dictionary.
        start_time: Experiment start timestamp.
        results: Collected experiment results.
    """
    
    experiment_id: str
    output_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    
    def __enter__(self) -> ExperimentContext:
        """Set up experiment environment."""
        ...
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Clean up and save results.
        
        On success: Save results to output_dir/results.json
        On failure: Save partial results and error details
        """
        ...
    
    def record_result(self, key: str, value: Any) -> None:
        """Record an experiment result."""
        ...
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a numeric metric with timestamp."""
        ...
```

Requirements:
- Create output directory if it does not exist
- Record start and end timestamps
- Save results to JSON on exit
- Log exception details if context exits with error
- Implement both `__enter__`/`__exit__` methods

#### 2.2 Validation Decorator Framework (12 points)

Implement composable validation decorators:

```python
def validate_args(**validators: Callable[[Any], bool]) -> Callable:
    """Decorator for validating function arguments.
    
    Args:
        **validators: Mapping of argument names to validation functions.
        
    Example:
        >>> @validate_args(x=lambda v: v > 0, name=lambda v: len(v) > 0)
        ... def process(x: float, name: str) -> str:
        ...     return f"{name}: {x}"
    """
    ...

def validate_return(validator: Callable[[Any], bool], message: str = "") -> Callable:
    """Decorator for validating function return value."""
    ...

def validate_types(**type_specs: type) -> Callable:
    """Decorator for runtime type checking of arguments."""
    ...
```

#### 2.3 Schema Validator Class (8 points)

Implement a reusable schema validator:

```python
class SchemaValidator:
    """Reusable schema validator for research data.
    
    Example:
        >>> validator = SchemaValidator({
        ...     "temperature": (float, {"min": -273.15}),
        ...     "pressure": (float, {"min": 0}),
        ...     "label": (str, {"pattern": r"^[A-Z]{3}[0-9]+$"}),
        ... })
        >>> validator.validate({"temperature": 25.0, "pressure": 101.3, "label": "ABC123"})
        ValidationResult(is_valid=True, errors=[], warnings=[])
    """
    
    def __init__(self, schema: dict[str, tuple[type, dict[str, Any]]]) -> None:
        """Initialise with schema specification."""
        ...
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        ...
    
    def validate_many(
        self,
        records: Sequence[dict[str, Any]],
        *,
        stop_on_first_error: bool = False,
    ) -> list[ValidationResult]:
        """Validate multiple records."""
        ...
```

### Deliverables

- `homework_part2.py` with all implementations
- Comprehensive docstrings with examples
- At least 6 unit tests in `test_homework_part2.py`

### Grading Rubric

| Criterion | Points |
|-----------|--------|
| ExperimentContext implementation | 15 |
| Validation decorators | 12 |
| SchemaValidator class | 8 |
| **Part 2 Total** | **35** |

---

## Part 3: Resilience Patterns (30 points)

### Background

Scientific computing workflows often depend on external services: databases, APIs, file systems and compute clusters. These dependencies introduce failure modes that require resilience patterns.

### Requirements

Create a module `homework_part3.py` containing:

#### 3.1 Retry Mechanism with Statistics (12 points)

Implement an enhanced retry decorator:

```python
@dataclass
class RetryStatistics:
    """Statistics from retry operations."""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_delay: float = 0.0
    exceptions: list[Exception] = field(default_factory=list)

def retry_with_statistics(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Callable:
    """Decorator with retry statistics collection.
    
    The decorated function gains a `.stats` attribute containing
    RetryStatistics from the most recent call.
    
    Args:
        max_attempts: Maximum attempts before giving up.
        base_delay: Initial delay between retries.
        max_delay: Maximum delay cap.
        exponential_base: Multiplier for exponential backoff.
        retryable_exceptions: Exceptions to retry on.
        on_retry: Callback invoked on each retry.
        
    Returns:
        Decorated function with .stats attribute.
    """
    ...
```

#### 3.2 Circuit Breaker with Monitoring (10 points)

Extend the circuit breaker with monitoring:

```python
class MonitoredCircuitBreaker:
    """Circuit breaker with metrics collection.
    
    Tracks call counts, failure rates and state transitions
    for monitoring and alerting.
    
    Attributes:
        name: Circuit breaker identifier.
        failure_threshold: Failures before opening.
        reset_timeout: Seconds before reset attempt.
        metrics: Collected metrics dictionary.
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        *,
        on_state_change: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialise monitored circuit breaker."""
        ...
    
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute function through circuit breaker."""
        ...
    
    def get_metrics(self) -> dict[str, Any]:
        """Return current metrics.
        
        Returns:
            Dictionary with:
            - total_calls: int
            - successful_calls: int
            - failed_calls: int
            - rejected_calls: int
            - state_transitions: list[tuple[str, str, float]]
            - current_state: str
            - failure_rate: float
        """
        ...
    
    def reset(self) -> None:
        """Reset circuit breaker and clear metrics."""
        ...
```

#### 3.3 Graceful Degradation Service (8 points)

Implement a service wrapper with graceful degradation:

```python
class ResilientService:
    """Service wrapper with multiple fallback levels.
    
    Attempts primary operation, then fallbacks in order,
    returning first successful result.
    
    Example:
        >>> service = ResilientService(
        ...     primary=fetch_from_api,
        ...     fallbacks=[fetch_from_cache, return_default],
        ...     circuit_breaker=MonitoredCircuitBreaker("api"),
        ... )
        >>> result = service.call(item_id=123)
    """
    
    def __init__(
        self,
        primary: Callable[..., T],
        fallbacks: list[Callable[..., T]],
        *,
        circuit_breaker: MonitoredCircuitBreaker | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialise resilient service."""
        ...
    
    def call(self, *args: Any, **kwargs: Any) -> T:
        """Attempt primary then fallbacks."""
        ...
    
    @property
    def last_source(self) -> str:
        """Return name of last successful source."""
        ...
```

### Deliverables

- `homework_part3.py` with all implementations
- Comprehensive docstrings
- At least 5 unit tests in `test_homework_part3.py`

### Grading Rubric

| Criterion | Points |
|-----------|--------|
| Retry with statistics | 12 |
| Monitored circuit breaker | 10 |
| Resilient service wrapper | 8 |
| **Part 3 Total** | **30** |

---

## Bonus: Advanced Checkpoint Recovery (+10 points)

### Requirements

Implement an advanced checkpoint system supporting:

```python
class TransactionalCheckpoint:
    """Checkpoint manager with transactional semantics.
    
    Supports atomic updates, rollback capability and
    versioned state snapshots.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        max_versions: int = 5,
    ) -> None:
        """Initialise transactional checkpoint."""
        ...
    
    @contextmanager
    def transaction(self) -> Generator[dict[str, Any], None, None]:
        """Context manager for transactional state updates.
        
        Changes are only committed if context exits successfully.
        On exception, state rolls back to pre-transaction state.
        """
        ...
    
    def get_version_history(self) -> list[tuple[int, datetime, str]]:
        """Return list of (version, timestamp, description) tuples."""
        ...
    
    def rollback_to_version(self, version: int) -> None:
        """Rollback state to specified version."""
        ...
```

---

## Submission Checklist

Before submitting, verify:

- [ ] All files follow naming convention (`homework_part1.py`, etc.)
- [ ] All functions have complete docstrings with examples
- [ ] All functions have type hints
- [ ] Code passes `ruff check` without errors
- [ ] Code passes `mypy --strict` without errors
- [ ] All unit tests pass with `pytest`
- [ ] No hardcoded file paths (use `pathlib.Path`)
- [ ] No `print()` statements (use `logging` module)
- [ ] British English spelling in comments and docstrings

---

## Academic Integrity

This is an individual assignment. You may:
- Consult course materials and documentation
- Discuss concepts with classmates
- Use standard library and course-provided utilities

You may not:
- Share code with classmates
- Submit code written by others
- Use AI code generation tools without disclosure

---

## Support

- Office hours: [See course schedule]
- Discussion forum: Post questions about requirements (not solutions)
- Email: For personal circumstances affecting submission

---

**Produce work of the highest standard.** Remember that reliable error handling is not an afterthought—it is integral to producing trustworthy research software.
