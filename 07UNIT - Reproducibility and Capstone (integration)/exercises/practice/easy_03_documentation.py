#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Easy 03 - Writing Documentation
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Good documentation is essential for reproducible research. This exercise
focuses on writing proper docstrings following the Google style guide and
creating clear README sections.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Write Google-style docstrings with all required sections
2. Document function parameters and return values
3. Include usage examples in docstrings

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Add Docstrings to Functions
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_mean(numbers: list[float]) -> float:
    """
    TODO: Write a proper Google-style docstring for this function.

    The docstring should include:
    - A brief description (one line)
    - An extended description (if needed)
    - Args section documenting 'numbers'
    - Returns section documenting the return value
    - Raises section for ValueError (empty list)
    - Example section showing usage

    Template:
        Brief description.

        Extended description if needed.

        Args:
            param_name: Description of parameter.

        Returns:
            Description of return value.

        Raises:
            ExceptionType: When this exception is raised.

        Example:
            >>> function_call()
            expected_result
    """
    if not numbers:
        raise ValueError("Cannot calculate mean of empty list")
    return sum(numbers) / len(numbers)


def calculate_variance(
    numbers: list[float],
    population: bool = True,
) -> float:
    """
    TODO: Write a proper Google-style docstring.

    This function calculates variance. When population=True, it calculates
    population variance (divides by N). When population=False, it calculates
    sample variance (divides by N-1).

    Include all sections: description, Args, Returns, Raises, Example.
    """
    if not numbers:
        raise ValueError("Cannot calculate variance of empty list")
    if len(numbers) == 1 and not population:
        raise ValueError("Sample variance requires at least 2 values")

    mean = sum(numbers) / len(numbers)
    squared_diffs = [(x - mean) ** 2 for x in numbers]
    divisor = len(numbers) if population else len(numbers) - 1
    return sum(squared_diffs) / divisor


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Document a Class
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """
    TODO: Write a proper class docstring.

    Document:
    - What this class represents
    - Each attribute with its type and description
    - Example of creating an instance

    Template:
        Brief description of the class.

        Extended description explaining purpose and usage.

        Attributes:
            attr_name: Description of attribute.

        Example:
            >>> result = ExperimentResult(...)
    """

    experiment_id: str
    accuracy: float
    loss: float
    epochs: int
    parameters: dict[str, Any]


class DataProcessor:
    """
    TODO: Write a proper class docstring.

    This class processes experimental data. Document the class purpose,
    its initialisation parameters, and public methods.
    """

    def __init__(self, normalise: bool = True) -> None:
        """
        TODO: Document the initialiser.

        Args:
            normalise: Whether to normalise data during processing.
        """
        self.normalise = normalise
        self._data: list[float] = []

    def add_data(self, values: list[float]) -> None:
        """
        TODO: Document this method.

        This method adds data points to the processor.
        """
        self._data.extend(values)

    def process(self) -> list[float]:
        """
        TODO: Document this method.

        This method processes and optionally normalises the data.
        Returns the processed data.
        """
        if not self._data:
            return []

        if self.normalise:
            min_val = min(self._data)
            max_val = max(self._data)
            if max_val == min_val:
                return [0.5] * len(self._data)
            return [(x - min_val) / (max_val - min_val) for x in self._data]

        return self._data.copy()


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Write README Sections
# ═══════════════════════════════════════════════════════════════════════════════

README_INSTALLATION_TEMPLATE = """
## Installation

TODO: Write clear installation instructions including:
1. Prerequisites (Python version, OS requirements)
2. Clone command
3. Virtual environment setup (optional but recommended)
4. pip install command
5. Verification step

Example format:

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Steps

```bash
# Clone the repository
git clone https://github.com/username/project.git
cd project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e .

# Verify installation
python -c "import mypackage; print(mypackage.__version__)"
```
"""


README_USAGE_TEMPLATE = """
## Usage

TODO: Write a usage section including:
1. Basic example (simplest use case)
2. Common use case with explanation
3. Advanced example (optional)

Example format:

### Basic Usage

```python
from mypackage import process_data

# Load and process data
result = process_data("input.csv")
print(result.summary())
```

### Processing with Options

```python
from mypackage import process_data, Config

# Configure processing
config = Config(
    normalise=True,
    remove_outliers=True,
)

# Process with configuration
result = process_data("input.csv", config=config)
```
"""


def generate_readme_section(section_type: str, project_name: str) -> str:
    """
    Generate a README section template.

    TODO: Implement this function to generate README sections.

    Args:
        section_type: Type of section ("installation", "usage", "contributing").
        project_name: Name of the project.

    Returns:
        Formatted README section as a string.

    Raises:
        ValueError: If section_type is not recognised.

    Example:
        >>> section = generate_readme_section("installation", "MyProject")
        >>> "pip install" in section
        True
    """
    # TODO: Implement this function
    # Return appropriate template based on section_type
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def check_docstring(obj: Any) -> dict[str, bool]:
    """
    Check if an object has a proper docstring.

    Args:
        obj: Function, class, or method to check.

    Returns:
        Dictionary with checks for each docstring section.
    """
    doc = obj.__doc__ or ""

    checks = {
        "has_docstring": bool(doc.strip()),
        "has_args": "Args:" in doc or "Parameters:" in doc,
        "has_returns": "Returns:" in doc,
        "has_example": "Example:" in doc or ">>>" in doc,
        "not_todo": "TODO" not in doc,
    }

    return checks


def run_validation() -> None:
    """Run validation on all documented items."""
    print("Checking documentation quality...\n")

    items_to_check = [
        ("calculate_mean", calculate_mean),
        ("calculate_variance", calculate_variance),
        ("ExperimentResult", ExperimentResult),
        ("DataProcessor", DataProcessor),
        ("DataProcessor.add_data", DataProcessor.add_data),
        ("DataProcessor.process", DataProcessor.process),
    ]

    all_passed = True

    for name, obj in items_to_check:
        checks = check_docstring(obj)
        status = "✓" if all(checks.values()) else "✗"
        print(f"{status} {name}")

        if not all(checks.values()):
            all_passed = False
            for check, passed in checks.items():
                if not passed:
                    print(f"    Missing: {check}")

    print("\n" + "=" * 60)
    if all_passed:
        print("All documentation checks passed! 🎉")
    else:
        print("Some documentation needs improvement.")
        print("Complete the TODO items in each docstring.")
    print("=" * 60)


if __name__ == "__main__":
    run_validation()
