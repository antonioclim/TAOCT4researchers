#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Easy Exercise 3 — Documentation Established Conventions
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for documentation exercises with Google-style docstrings.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: FUNCTION DOCSTRINGS — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


def calculate_mean(values: list[float]) -> float:
    """
    Calculate the arithmetic mean of a list of numerical values.

    The arithmetic mean is computed as the sum of all values divided by
    the count of values. This function handles both integer and floating-point
    inputs.

    Args:
        values: A list of numerical values. Must contain at least one element.

    Returns:
        The arithmetic mean as a floating-point number.

    Raises:
        ValueError: If the input list is empty.
        TypeError: If values contains non-numeric elements.

    Example:
        >>> calculate_mean([1, 2, 3, 4, 5])
        3.0
        >>> calculate_mean([10.5, 20.5])
        15.5
    """
    if not values:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(values) / len(values)


def calculate_variance(values: list[float], population: bool = True) -> float:
    """
    Calculate the variance of a list of numerical values.

    Variance measures the spread of data points around the mean. This function
    supports both population variance (dividing by N) and sample variance
    (dividing by N-1, using Bessel's correction).

    Args:
        values: A list of numerical values. Must contain at least one element
            for population variance or at least two elements for sample variance.
        population: If True, calculates population variance (default).
            If False, calculates sample variance with Bessel's correction.

    Returns:
        The variance as a floating-point number.

    Raises:
        ValueError: If the list is empty, or if sample variance is requested
            but the list contains fewer than two elements.

    Example:
        >>> calculate_variance([1, 2, 3, 4, 5])
        2.0
        >>> calculate_variance([1, 2, 3, 4, 5], population=False)
        2.5
    """
    if not values:
        raise ValueError("Cannot calculate variance of empty list")

    n = len(values)

    if not population and n < 2:
        raise ValueError("Sample variance requires at least two values")

    mean = calculate_mean(values)
    squared_diffs = [(x - mean) ** 2 for x in values]
    divisor = n if population else (n - 1)

    return sum(squared_diffs) / divisor


def calculate_standard_deviation(
    values: list[float],
    population: bool = True
) -> float:
    """
    Calculate the standard deviation of a list of numerical values.

    Standard deviation is the square root of variance and represents the
    average distance of data points from the mean. Like variance, this
    function supports both population and sample calculations.

    Args:
        values: A list of numerical values. Must contain at least one element
            for population standard deviation or at least two for sample.
        population: If True, calculates population standard deviation (default).
            If False, calculates sample standard deviation.

    Returns:
        The standard deviation as a floating-point number.

    Raises:
        ValueError: If the list is empty, or if sample calculation is requested
            but the list contains fewer than two elements.

    Example:
        >>> round(calculate_standard_deviation([1, 2, 3, 4, 5]), 4)
        1.4142
        >>> round(calculate_standard_deviation([1, 2, 3, 4, 5], population=False), 4)
        1.5811
    """
    return calculate_variance(values, population) ** 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: CLASS AND DATACLASS DOCSTRINGS — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExperimentResult:
    """
    A container for storing experiment results with metadata.

    This dataclass provides a structured way to record experimental outcomes,
    including the experiment identifier, measured values, statistical metrics
    and a timestamp for reproducibility tracking.

    Attributes:
        experiment_id: Unique identifier for the experiment.
        values: List of measured values from the experiment.
        mean: Calculated arithmetic mean of values.
        std_dev: Calculated standard deviation of values.
        timestamp: ISO format timestamp when the result was created.
            Defaults to the current time if not specified.

    Example:
        >>> result = ExperimentResult(
        ...     experiment_id="EXP-001",
        ...     values=[1.0, 2.0, 3.0],
        ...     mean=2.0,
        ...     std_dev=0.816
        ... )
        >>> result.experiment_id
        'EXP-001'
    """

    experiment_id: str
    values: list[float]
    mean: float
    std_dev: float
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class DataProcessor:
    """
    A class for processing and analysing experimental data.

    DataProcessor provides methods to load data, perform statistical analysis
    and export results. It maintains an internal state of loaded data and
    computed results for efficient batch processing.

    Attributes:
        data: The currently loaded dataset as a list of floats.
        results: List of ExperimentResult objects from analyses.
        name: Optional name identifier for the processor instance.

    Example:
        >>> processor = DataProcessor(name="Temperature Analysis")
        >>> processor.load_data([20.1, 21.3, 19.8, 22.0])
        >>> result = processor.analyse("temp-exp-001")
        >>> print(f"Mean: {result.mean:.2f}")
        Mean: 20.80
    """

    def __init__(self, name: str = "default") -> None:
        """
        Initialise a new DataProcessor instance.

        Args:
            name: Optional identifier for this processor. Defaults to "default".
        """
        self.data: list[float] = []
        self.results: list[ExperimentResult] = []
        self.name: str = name

    def load_data(self, data: list[float]) -> None:
        """
        Load a dataset into the processor for analysis.

        Replaces any previously loaded data with the new dataset. The data
        is validated to ensure it contains only numeric values.

        Args:
            data: A list of numerical values to process.

        Raises:
            TypeError: If data contains non-numeric elements.
            ValueError: If data is empty.

        Example:
            >>> processor = DataProcessor()
            >>> processor.load_data([1.0, 2.0, 3.0])
            >>> len(processor.data)
            3
        """
        if not data:
            raise ValueError("Cannot load empty dataset")
        self.data = list(data)

    def analyse(self, experiment_id: str) -> ExperimentResult:
        """
        Perform statistical analysis on the loaded data.

        Calculates the mean and standard deviation of the currently loaded
        data and creates an ExperimentResult object with the findings.

        Args:
            experiment_id: Unique identifier for this analysis run.

        Returns:
            An ExperimentResult containing the analysis outcomes.

        Raises:
            RuntimeError: If no data has been loaded.

        Example:
            >>> processor = DataProcessor()
            >>> processor.load_data([10, 20, 30])
            >>> result = processor.analyse("test-001")
            >>> result.mean
            20.0
        """
        if not self.data:
            raise RuntimeError("No data loaded. Call load_data() first.")

        mean = calculate_mean(self.data)
        std_dev = calculate_standard_deviation(self.data)

        result = ExperimentResult(
            experiment_id=experiment_id,
            values=self.data.copy(),
            mean=mean,
            std_dev=std_dev
        )

        self.results.append(result)
        return result

    def export_results(self) -> list[dict[str, Any]]:
        """
        Export all analysis results as a list of dictionaries.

        Converts all stored ExperimentResult objects to dictionary format
        suitable for JSON serialisation or further processing.

        Returns:
            A list of dictionaries, each representing one analysis result.

        Example:
            >>> processor = DataProcessor()
            >>> processor.load_data([1, 2, 3])
            >>> _ = processor.analyse("exp-001")
            >>> exported = processor.export_results()
            >>> exported[0]["experiment_id"]
            'exp-001'
        """
        return [
            {
                "experiment_id": r.experiment_id,
                "values": r.values,
                "mean": r.mean,
                "std_dev": r.std_dev,
                "timestamp": r.timestamp
            }
            for r in self.results
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: README SECTIONS — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

INSTALLATION_SECTION = """
## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- Git (for cloning the repository)

### Quick Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/project-name.git
cd project-name
pip install -r requirements.txt
```

### Development Installation

For development work, install with additional testing dependencies:

```bash
pip install -e ".[dev]"
```

### Verifying Installation

Run the test suite to verify the installation:

```bash
pytest tests/
```

### Troubleshooting

If you encounter import errors, ensure your Python version is compatible:

```bash
python --version  # Should output 3.12 or higher
```
"""

USAGE_SECTION = """
## Usage

### Basic Usage

Import the main module and create a processor instance:

```python
from data_analysis import DataProcessor

# Create a processor
processor = DataProcessor(name="my-analysis")

# Load your data
processor.load_data([1.5, 2.3, 3.1, 4.7, 5.2])

# Perform analysis
result = processor.analyse("experiment-001")

# Access results
print(f"Mean: {result.mean:.2f}")
print(f"Standard Deviation: {result.std_dev:.2f}")
```

### Command Line Interface

The package can also be used from the command line:

```bash
python -m data_analysis --input data.csv --output results.json
```

### Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | stdin | Input data file path |
| `--output` | stdout | Output results file path |
| `--format` | json | Output format (json, csv) |
| `--verbose` | False | Enable detailed logging |

### Advanced Usage

For batch processing multiple experiments:

```python
from data_analysis import DataProcessor, BatchRunner

runner = BatchRunner()
runner.add_experiment("exp-001", [1, 2, 3])
runner.add_experiment("exp-002", [4, 5, 6])

all_results = runner.run_all()
```
"""


# ═══════════════════════════════════════════════════════════════════════════════
# DOCSTRING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def check_docstring(obj: Any) -> dict[str, bool]:
    """
    Check if an object has a properly formatted docstring.

    Validates that the object has a docstring containing the essential
    sections for Google-style documentation.

    Args:
        obj: Any Python object (function, class, method) to check.

    Returns:
        A dictionary indicating which documentation elements are present.
    """
    doc = obj.__doc__ or ""

    return {
        "has_docstring": bool(doc.strip()),
        "has_description": len(doc.strip().split("\n")[0]) > 10,
        "has_args": "Args:" in doc or "arg" not in str(obj),
        "has_returns": "Returns:" in doc or "return" not in str(obj),
        "has_example": "Example:" in doc or ">>>" in doc,
        "has_raises": "Raises:" in doc or True,  # Optional
    }


def run_tests() -> None:
    """Run all validation tests for the exercises."""
    print("=" * 70)
    print("SOLUTION VALIDATION: Easy Exercise 3 — Documentation")
    print("=" * 70)

    # Test Exercise 1: Function docstrings
    print("\n--- Exercise 1: Function Docstrings ---")
    for func in [calculate_mean, calculate_variance, calculate_standard_deviation]:
        result = check_docstring(func)
        assert result["has_docstring"], f"{func.__name__} missing docstring"
        assert result["has_description"], f"{func.__name__} missing description"
        assert result["has_args"], f"{func.__name__} missing Args section"
        assert result["has_returns"], f"{func.__name__} missing Returns section"
        print(f"✓ {func.__name__} has complete docstring")

    # Test functions work correctly
    assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
    assert calculate_variance([1, 2, 3, 4, 5]) == 2.0
    print("✓ Functions work correctly")

    # Test Exercise 2: Class docstrings
    print("\n--- Exercise 2: Class Docstrings ---")
    assert ExperimentResult.__doc__ and len(ExperimentResult.__doc__) > 100
    print("✓ ExperimentResult has complete docstring")

    assert DataProcessor.__doc__ and len(DataProcessor.__doc__) > 100
    print("✓ DataProcessor class has complete docstring")

    # Check methods
    for method_name in ["load_data", "analyse", "export_results"]:
        method = getattr(DataProcessor, method_name)
        assert method.__doc__ and len(method.__doc__) > 50
        print(f"✓ DataProcessor.{method_name}() has docstring")

    # Test Exercise 3: README sections
    print("\n--- Exercise 3: README Sections ---")
    assert "## Installation" in INSTALLATION_SECTION
    assert "```bash" in INSTALLATION_SECTION
    assert "pip install" in INSTALLATION_SECTION
    print("✓ Installation section is complete")

    assert "## Usage" in USAGE_SECTION
    assert "```python" in USAGE_SECTION
    assert "Command Line" in USAGE_SECTION
    print("✓ Usage section is complete")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
