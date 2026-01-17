#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Hard 01 - Comprehensive Testing Framework
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Production-quality research software requires comprehensive testing strategies
including property-based testing, regression testing and performance testing.
This exercise challenges you to build a complete testing framework.

PREREQUISITES
─────────────
- Completed all easy and medium exercises
- Strong understanding of pytest fixtures and parametrisation
- Familiarity with hypothesis library concepts

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Implement property-based testing with hypothesis
2. Build regression test suites with golden files
3. Create performance benchmarking frameworks

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 90 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Callable
from typing import TypeVar

import numpy as np

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# CODE UNDER TEST
# ═══════════════════════════════════════════════════════════════════════════════

def sort_data(data: list[float]) -> list[float]:
    """Sort a list of floats in ascending order."""
    return sorted(data)


def normalise_data(data: list[float]) -> list[float]:
    """Normalise data to [0, 1] range."""
    if not data:
        return []
    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def compute_statistics(data: list[float]) -> dict[str, float]:
    """Compute basic statistics for a dataset."""
    if not data:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    arr = np.array(data)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two matrices."""
    return np.dot(a, b)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Property-Based Testing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PropertyTest:
    """Represents a property-based test."""

    name: str
    property_fn: Callable[[Any], bool]
    generator: Callable[[], Any]
    num_examples: int = 100


def generate_random_list(
    min_length: int = 0,
    max_length: int = 100,
    min_value: float = -1000.0,
    max_value: float = 1000.0,
) -> list[float]:
    """
    Generate a random list of floats.

    This is a simple generator for property-based testing.
    In practice, you would use hypothesis library.

    Args:
        min_length: Minimum list length.
        max_length: Maximum list length.
        min_value: Minimum float value.
        max_value: Maximum float value.

    Returns:
        Random list of floats.
    """
    length = np.random.randint(min_length, max_length + 1)
    return list(np.random.uniform(min_value, max_value, length))


def property_sorted_output_is_sorted(data: list[float]) -> bool:
    """
    Property: sorted output should be in ascending order.

    Args:
        data: Input list.

    Returns:
        True if property holds.
    """
    result = sort_data(data)
    return all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def property_sorted_preserves_elements(data: list[float]) -> bool:
    """
    Property: sorting should preserve all elements.

    Args:
        data: Input list.

    Returns:
        True if property holds.
    """
    result = sort_data(data)
    return sorted(data) == sorted(result) and len(data) == len(result)


def property_normalised_in_range(data: list[float]) -> bool:
    """
    TODO: Implement property that normalised data is in [0, 1].

    Property: all normalised values should be between 0 and 1.

    Args:
        data: Input list.

    Returns:
        True if all normalised values are in [0, 1].
    """
    # TODO: Implement this property
    pass


def property_normalised_preserves_order(data: list[float]) -> bool:
    """
    TODO: Implement property that normalisation preserves relative order.

    Property: if a < b in original, then normalise(a) <= normalise(b).

    Args:
        data: Input list.

    Returns:
        True if relative ordering is preserved.
    """
    # TODO: Implement this property
    pass


def run_property_test(test: PropertyTest) -> tuple[bool, list[Any]]:
    """
    Run a property-based test with multiple random examples.

    Args:
        test: PropertyTest specification.

    Returns:
        Tuple of (all_passed, failing_examples).
    """
    failing_examples = []

    for _ in range(test.num_examples):
        example = test.generator()
        try:
            if not test.property_fn(example):
                failing_examples.append(example)
        except Exception as e:
            failing_examples.append((example, str(e)))

    return len(failing_examples) == 0, failing_examples


class PropertyTestSuite:
    """
    TODO: Implement a property test suite.

    This class should:
    1. Collect PropertyTest instances
    2. Run all tests and collect results
    3. Report failures with shrunk examples
    """

    def __init__(self) -> None:
        """Initialise empty test suite."""
        self.tests: list[PropertyTest] = []
        self.results: dict[str, tuple[bool, list[Any]]] = {}

    def add_test(self, test: PropertyTest) -> None:
        """Add a test to the suite."""
        # TODO: Implement
        pass

    def run_all(self) -> dict[str, tuple[bool, list[Any]]]:
        """
        Run all tests and return results.

        Returns:
            Dictionary mapping test name to (passed, failing_examples).
        """
        # TODO: Implement
        pass

    def shrink_example(
        self,
        example: list[float],
        property_fn: Callable[[list[float]], bool],
    ) -> list[float]:
        """
        TODO: Implement example shrinking.

        Shrinking finds a minimal failing example by:
        1. Trying to remove elements
        2. Trying to simplify values (towards 0)
        3. Trying to reduce list length

        Args:
            example: Failing example.
            property_fn: Property that should fail.

        Returns:
            Minimal failing example.
        """
        # TODO: Implement shrinking algorithm
        pass

    def report(self) -> str:
        """Generate a test report."""
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Regression Testing with Golden Files
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoldenTest:
    """Specification for a golden file test."""

    name: str
    function: Callable[..., Any]
    inputs: dict[str, Any]
    golden_file: Path


class GoldenTestFramework:
    """
    TODO: Implement a golden file testing framework.

    Golden file testing compares function output against stored "golden"
    reference outputs. This is useful for regression testing complex
    computations.

    Workflow:
    1. First run: Generate golden files
    2. Subsequent runs: Compare against golden files
    3. If different: Either test fails or update golden file
    """

    def __init__(self, golden_dir: Path) -> None:
        """
        Initialise framework with golden file directory.

        Args:
            golden_dir: Directory for golden files.
        """
        self.golden_dir = golden_dir
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.tests: list[GoldenTest] = []

    def add_test(
        self,
        name: str,
        function: Callable[..., Any],
        inputs: dict[str, Any],
    ) -> None:
        """
        Add a test to the framework.

        Args:
            name: Test name (used for golden file name).
            function: Function to test.
            inputs: Dictionary of inputs to pass to function.
        """
        # TODO: Implement
        pass

    def generate_golden(self, test: GoldenTest) -> None:
        """
        Generate golden file for a test.

        Args:
            test: Test specification.
        """
        # TODO: Implement
        # 1. Run the function with inputs
        # 2. Serialise output to JSON
        # 3. Save to golden file
        pass

    def compare_with_golden(
        self,
        test: GoldenTest,
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Compare test output with golden file.

        Args:
            test: Test specification.

        Returns:
            Tuple of (matches, diff) where diff is None if matches.
        """
        # TODO: Implement
        pass

    def run_test(
        self,
        test: GoldenTest,
        update: bool = False,
    ) -> tuple[bool, str]:
        """
        Run a single test.

        Args:
            test: Test to run.
            update: If True, update golden file instead of comparing.

        Returns:
            Tuple of (passed, message).
        """
        # TODO: Implement
        pass

    def run_all(self, update: bool = False) -> dict[str, tuple[bool, str]]:
        """
        Run all tests.

        Args:
            update: If True, update all golden files.

        Returns:
            Dictionary mapping test name to (passed, message).
        """
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Performance Benchmarking
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time: float
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    input_size: int


@dataclass
class PerformanceBaseline:
    """Baseline performance expectations."""

    max_mean_time: float  # Maximum acceptable mean time
    max_std_time: float  # Maximum acceptable standard deviation
    complexity_class: str  # Expected complexity: O(1), O(n), O(n^2), etc.


class BenchmarkFramework:
    """
    TODO: Implement a benchmarking framework.

    Features:
    1. Run functions multiple times for statistical significance
    2. Compare against baseline expectations
    3. Detect performance regressions
    4. Estimate algorithmic complexity
    """

    def __init__(self, warmup_iterations: int = 3) -> None:
        """
        Initialise benchmark framework.

        Args:
            warmup_iterations: Number of warmup runs before measuring.
        """
        self.warmup_iterations = warmup_iterations
        self.results: dict[str, list[BenchmarkResult]] = {}

    def benchmark(
        self,
        name: str,
        function: Callable[[], Any],
        iterations: int = 100,
        input_size: int = 0,
    ) -> BenchmarkResult:
        """
        Benchmark a function.

        Args:
            name: Benchmark name.
            function: Function to benchmark (takes no arguments).
            iterations: Number of iterations.
            input_size: Size of input (for complexity analysis).

        Returns:
            BenchmarkResult with timing statistics.
        """
        # TODO: Implement
        # 1. Run warmup iterations
        # 2. Run timed iterations
        # 3. Compute statistics
        # 4. Return BenchmarkResult
        pass

    def benchmark_scaling(
        self,
        name: str,
        function_factory: Callable[[int], Callable[[], Any]],
        sizes: list[int],
        iterations: int = 50,
    ) -> list[BenchmarkResult]:
        """
        Benchmark function at different input sizes.

        Args:
            name: Benchmark name.
            function_factory: Function that creates benchmark function for size.
            sizes: List of input sizes to test.
            iterations: Iterations per size.

        Returns:
            List of BenchmarkResults for each size.
        """
        # TODO: Implement
        pass

    def estimate_complexity(
        self,
        results: list[BenchmarkResult],
    ) -> str:
        """
        Estimate algorithmic complexity from benchmark results.

        Uses curve fitting to estimate whether the function is:
        - O(1): constant time
        - O(log n): logarithmic
        - O(n): linear
        - O(n log n): linearithmic
        - O(n^2): quadratic

        Args:
            results: Benchmark results at different input sizes.

        Returns:
            Estimated complexity class string.
        """
        # TODO: Implement
        # Hint: Fit different complexity models and compare R^2 values
        pass

    def check_baseline(
        self,
        result: BenchmarkResult,
        baseline: PerformanceBaseline,
    ) -> tuple[bool, str]:
        """
        Check if benchmark meets baseline expectations.

        Args:
            result: Benchmark result.
            baseline: Expected baseline.

        Returns:
            Tuple of (passed, message).
        """
        # TODO: Implement
        pass

    def generate_report(self) -> str:
        """Generate a performance report."""
        # TODO: Implement
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_property_testing() -> None:
    """Demonstrate property-based testing."""
    print("\n" + "=" * 60)
    print("Property-Based Testing Demo")
    print("=" * 60)

    # Test sort properties
    suite = PropertyTestSuite()
    suite.add_test(PropertyTest(
        name="sorted_is_sorted",
        property_fn=property_sorted_output_is_sorted,
        generator=generate_random_list,
    ))
    suite.add_test(PropertyTest(
        name="sorted_preserves_elements",
        property_fn=property_sorted_preserves_elements,
        generator=generate_random_list,
    ))

    if property_normalised_in_range is not None:
        suite.add_test(PropertyTest(
            name="normalised_in_range",
            property_fn=property_normalised_in_range,
            generator=generate_random_list,
        ))

    results = suite.run_all()
    print(suite.report())


def demonstrate_golden_testing() -> None:
    """Demonstrate golden file testing."""
    print("\n" + "=" * 60)
    print("Golden File Testing Demo")
    print("=" * 60)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        framework = GoldenTestFramework(Path(tmpdir))

        framework.add_test(
            name="statistics_basic",
            function=compute_statistics,
            inputs={"data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        )

        # First run: generate golden files
        print("Generating golden files...")
        results = framework.run_all(update=True)
        for name, (passed, msg) in results.items():
            print(f"  {name}: {msg}")

        # Second run: compare
        print("\nComparing with golden files...")
        results = framework.run_all(update=False)
        for name, (passed, msg) in results.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {name}: {msg}")


def demonstrate_benchmarking() -> None:
    """Demonstrate performance benchmarking."""
    print("\n" + "=" * 60)
    print("Performance Benchmarking Demo")
    print("=" * 60)

    framework = BenchmarkFramework()

    # Benchmark sorting at different sizes
    def create_sort_benchmark(size: int) -> Callable[[], Any]:
        data = list(np.random.random(size))
        return lambda: sort_data(data)

    sizes = [100, 500, 1000, 2000, 5000]
    results = framework.benchmark_scaling(
        "sort_data",
        create_sort_benchmark,
        sizes,
        iterations=20,
    )

    if results:
        print("\nBenchmark Results:")
        for result in results:
            print(f"  n={result.input_size}: {result.mean_time*1000:.3f}ms "
                  f"(±{result.std_time*1000:.3f}ms)")

        complexity = framework.estimate_complexity(results)
        print(f"\nEstimated complexity: {complexity}")

    print(framework.generate_report())


if __name__ == "__main__":
    demonstrate_property_testing()
    demonstrate_golden_testing()
    demonstrate_benchmarking()
