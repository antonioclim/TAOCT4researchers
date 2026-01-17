#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Hard Exercise 1 — Building a Comprehensive Testing Framework
═══════════════════════════════════════════════════════════════════════════════

This solution implements a multi-paradigm testing framework including:
1. Property-based testing with shrinking
2. Regression testing with golden files
3. Performance benchmarking with complexity estimation

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# CODE UNDER TEST
# ═══════════════════════════════════════════════════════════════════════════════

def sort_data(data: list[float]) -> list[float]:
    """Sort a list of numbers in ascending order."""
    return sorted(data)


def normalise_data(data: list[float]) -> list[float]:
    """Normalise data to [0, 1] range using min-max scaling."""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if min_val == max_val:
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def compute_statistics(data: list[float]) -> dict[str, float]:
    """Compute basic statistics for a dataset."""
    if not data:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(data),
        "mean": statistics.mean(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "min": min(data),
        "max": max(data),
    }


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices using naive O(n³) algorithm."""
    if not a or not b or not a[0] or not b[0]:
        return []
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError("Incompatible matrix dimensions")
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: PROPERTY-BASED TESTING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PropertyTest:
    """Represents a property-based test case.

    Attributes:
        name: Descriptive name for the property.
        property_fn: Function that takes generated data and returns bool.
        generator: Function that produces random test inputs.
        num_examples: Number of random examples to test.
    """

    name: str
    property_fn: Callable[[Any], bool]
    generator: Callable[[], Any]
    num_examples: int = 100


def generate_random_list(
    min_length: int = 0,
    max_length: int = 100,
    min_value: float = -1000.0,
    max_value: float = 1000.0,
) -> Callable[[], list[float]]:
    """Create a generator function for random float lists.

    Args:
        min_length: Minimum list length (inclusive).
        max_length: Maximum list length (inclusive).
        min_value: Minimum value for elements.
        max_value: Maximum value for elements.

    Returns:
        A callable that generates random lists when invoked.

    Example:
        >>> gen = generate_random_list(5, 10, 0, 100)
        >>> data = gen()
        >>> 5 <= len(data) <= 10
        True
    """

    def generator() -> list[float]:
        length = random.randint(min_length, max_length)
        return [random.uniform(min_value, max_value) for _ in range(length)]

    return generator


def sorted_output_is_sorted(data: list[float]) -> bool:
    """Property: Output of sort_data should always be sorted.

    Args:
        data: Input list to test sorting on.

    Returns:
        True if output is properly sorted, False otherwise.
    """
    result = sort_data(data)
    return all(result[i] <= result[i + 1] for i in range(len(result) - 1))


def sorted_preserves_elements(data: list[float]) -> bool:
    """Property: Sorting should preserve all elements (multiset equality).

    Args:
        data: Input list to test sorting on.

    Returns:
        True if sorted output contains exactly the same elements.
    """
    result = sort_data(data)
    return sorted(data) == sorted(result)


def normalised_in_range(data: list[float]) -> bool:
    """Property: Normalised values should be in [0, 1] range.

    Args:
        data: Input list to test normalisation on.

    Returns:
        True if all normalised values are within [0, 1].
    """
    if not data:
        return True
    result = normalise_data(data)
    return all(0.0 <= x <= 1.0 for x in result)


def normalised_preserves_order(data: list[float]) -> bool:
    """Property: Normalisation should preserve relative ordering.

    Args:
        data: Input list to test normalisation on.

    Returns:
        True if relative ordering is preserved after normalisation.
    """
    if len(data) < 2:
        return True
    original = data
    normalised = normalise_data(data)
    for i in range(len(original)):
        for j in range(i + 1, len(original)):
            orig_relation = (original[i] < original[j], original[i] == original[j])
            norm_relation = (normalised[i] < normalised[j], normalised[i] == normalised[j])
            # If original has strict ordering, normalised should too
            # (allowing for floating-point tolerance)
            if original[i] < original[j] - 1e-10:
                if not normalised[i] <= normalised[j]:
                    return False
            elif original[i] > original[j] + 1e-10:
                if not normalised[i] >= normalised[j]:
                    return False
    return True


@dataclass
class PropertyTestResult:
    """Result of running a property test.

    Attributes:
        name: Name of the property test.
        passed: Whether all examples passed.
        num_passed: Number of examples that passed.
        num_failed: Number of examples that failed.
        counterexample: First failing input, if any.
        shrunk_counterexample: Minimised failing input, if available.
    """

    name: str
    passed: bool
    num_passed: int
    num_failed: int
    counterexample: Any | None = None
    shrunk_counterexample: Any | None = None


class PropertyTestSuite:
    """Suite for running property-based tests with shrinking.

    This class manages a collection of property tests and provides
    functionality for running them and shrinking counterexamples.

    Attributes:
        tests: List of property tests to run.
        results: Results from the most recent run.

    Example:
        >>> suite = PropertyTestSuite()
        >>> suite.add_test(PropertyTest(
        ...     name="sort_is_sorted",
        ...     property_fn=sorted_output_is_sorted,
        ...     generator=generate_random_list(1, 50),
        ... ))
        >>> suite.run_all()
        >>> suite.results[0].passed
        True
    """

    def __init__(self) -> None:
        """Initialise an empty test suite."""
        self.tests: list[PropertyTest] = []
        self.results: list[PropertyTestResult] = []

    def add_test(self, test: PropertyTest) -> None:
        """Add a property test to the suite.

        Args:
            test: The property test to add.
        """
        self.tests.append(test)

    def run_all(self, seed: int | None = None) -> list[PropertyTestResult]:
        """Run all property tests in the suite.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            List of results for each test.
        """
        if seed is not None:
            random.seed(seed)

        self.results = []
        for test in self.tests:
            result = self._run_single_test(test)
            self.results.append(result)
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            logger.info(f"Property '{test.name}': {status}")
            if not result.passed and result.shrunk_counterexample is not None:
                logger.info(f"  Shrunk counterexample: {result.shrunk_counterexample}")

        return self.results

    def _run_single_test(self, test: PropertyTest) -> PropertyTestResult:
        """Run a single property test.

        Args:
            test: The property test to run.

        Returns:
            Result of running the test.
        """
        num_passed = 0
        num_failed = 0
        first_counterexample = None

        for _ in range(test.num_examples):
            example = test.generator()
            try:
                if test.property_fn(example):
                    num_passed += 1
                else:
                    num_failed += 1
                    if first_counterexample is None:
                        first_counterexample = example
            except Exception as e:
                num_failed += 1
                if first_counterexample is None:
                    first_counterexample = example
                    logger.warning(f"Exception during test: {e}")

        shrunk = None
        if first_counterexample is not None:
            shrunk = self.shrink_example(first_counterexample, test.property_fn)

        return PropertyTestResult(
            name=test.name,
            passed=num_failed == 0,
            num_passed=num_passed,
            num_failed=num_failed,
            counterexample=first_counterexample,
            shrunk_counterexample=shrunk,
        )

    def shrink_example(
        self,
        example: list[float],
        property_fn: Callable[[list[float]], bool],
        max_iterations: int = 100,
    ) -> list[float]:
        """Attempt to shrink a counterexample to a minimal failing case.

        Uses several shrinking strategies:
        1. Remove elements one at a time
        2. Reduce element values toward zero
        3. Simplify to round numbers

        Args:
            example: The failing input to shrink.
            property_fn: The property function being tested.
            max_iterations: Maximum shrinking iterations.

        Returns:
            A smaller or simpler counterexample that still fails.
        """
        if not isinstance(example, list):
            return example

        current = example.copy()
        iterations = 0

        while iterations < max_iterations:
            improved = False
            iterations += 1

            # Strategy 1: Try removing elements
            for i in range(len(current) - 1, -1, -1):
                if len(current) <= 1:
                    break
                candidate = current[:i] + current[i + 1:]
                try:
                    if not property_fn(candidate):
                        current = candidate
                        improved = True
                        break
                except Exception:
                    current = candidate
                    improved = True
                    break

            if improved:
                continue

            # Strategy 2: Try shrinking values toward zero
            for i in range(len(current)):
                for factor in [0, 0.5, -0.5]:
                    candidate = current.copy()
                    if factor == 0:
                        candidate[i] = 0.0
                    else:
                        candidate[i] = current[i] * factor
                    try:
                        if not property_fn(candidate):
                            current = candidate
                            improved = True
                            break
                    except Exception:
                        current = candidate
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Strategy 3: Try rounding to integers
            for i in range(len(current)):
                candidate = current.copy()
                candidate[i] = round(current[i])
                try:
                    if not property_fn(candidate):
                        if candidate != current:
                            current = candidate
                            improved = True
                            break
                except Exception:
                    current = candidate
                    improved = True
                    break

            if not improved:
                break

        return current

    def summary(self) -> str:
        """Generate a summary report of test results.

        Returns:
            Formatted string summarising all test results.
        """
        if not self.results:
            return "No tests have been run."

        lines = ["=" * 60, "PROPERTY TEST SUMMARY", "=" * 60]
        total_passed = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)

        for result in self.results:
            status = "✓" if result.passed else "✗"
            lines.append(f"{status} {result.name}: {result.num_passed}/{result.num_passed + result.num_failed}")
            if not result.passed and result.shrunk_counterexample is not None:
                lines.append(f"   Counterexample: {result.shrunk_counterexample[:5]}...")

        lines.append("-" * 60)
        lines.append(f"Total: {total_passed}/{total_tests} passed")
        lines.append("=" * 60)

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: REGRESSION TESTING WITH GOLDEN FILES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GoldenTest:
    """Represents a golden file test case.

    Attributes:
        name: Descriptive name for the test.
        function: Function to test.
        inputs: Input arguments for the function.
        golden_file: Path to the golden file.
    """

    name: str
    function: Callable[..., Any]
    inputs: tuple[Any, ...]
    golden_file: Path


@dataclass
class GoldenTestResult:
    """Result of a golden file test.

    Attributes:
        name: Name of the test.
        passed: Whether the test passed.
        expected: Expected output from golden file.
        actual: Actual output from function.
        diff: Description of differences if test failed.
    """

    name: str
    passed: bool
    expected: Any | None = None
    actual: Any | None = None
    diff: str | None = None


class GoldenTestFramework:
    """Framework for regression testing using golden files.

    Golden files store expected outputs that are compared against
    actual function outputs. This enables detection of unintended
    behavioural changes.

    Attributes:
        golden_dir: Directory for storing golden files.
        tests: List of golden tests to run.
        results: Results from the most recent run.

    Example:
        >>> framework = GoldenTestFramework(Path("./golden"))
        >>> framework.add_test(GoldenTest(
        ...     name="stats_test",
        ...     function=compute_statistics,
        ...     inputs=([1.0, 2.0, 3.0, 4.0, 5.0],),
        ...     golden_file=Path("stats_basic.json"),
        ... ))
        >>> framework.generate_all_golden()  # First time setup
        >>> results = framework.run_all()
    """

    def __init__(self, golden_dir: Path) -> None:
        """Initialise the golden test framework.

        Args:
            golden_dir: Directory for storing golden files.
        """
        self.golden_dir = golden_dir
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.tests: list[GoldenTest] = []
        self.results: list[GoldenTestResult] = []

    def add_test(self, test: GoldenTest) -> None:
        """Add a golden test to the framework.

        Args:
            test: The golden test to add.
        """
        self.tests.append(test)

    def generate_golden(self, test: GoldenTest) -> None:
        """Generate or update the golden file for a test.

        Args:
            test: The test to generate golden output for.
        """
        result = test.function(*test.inputs)
        golden_path = self.golden_dir / test.golden_file

        # Serialise result based on type
        serialised = self._serialise(result)

        with golden_path.open("w", encoding="utf-8") as f:
            json.dump(serialised, f, indent=2, sort_keys=True)

        logger.info(f"Generated golden file: {golden_path}")

    def generate_all_golden(self) -> None:
        """Generate golden files for all tests."""
        for test in self.tests:
            self.generate_golden(test)

    def compare_with_golden(self, test: GoldenTest) -> GoldenTestResult:
        """Compare function output with golden file.

        Args:
            test: The test to run.

        Returns:
            Result of the comparison.
        """
        golden_path = self.golden_dir / test.golden_file

        if not golden_path.exists():
            return GoldenTestResult(
                name=test.name,
                passed=False,
                diff=f"Golden file not found: {golden_path}",
            )

        # Load expected output
        with golden_path.open("r", encoding="utf-8") as f:
            expected = json.load(f)

        # Get actual output
        actual_raw = test.function(*test.inputs)
        actual = self._serialise(actual_raw)

        # Compare
        passed = self._deep_compare(expected, actual)

        diff = None
        if not passed:
            diff = self._generate_diff(expected, actual)

        return GoldenTestResult(
            name=test.name,
            passed=passed,
            expected=expected,
            actual=actual,
            diff=diff,
        )

    def run_test(self, test: GoldenTest) -> GoldenTestResult:
        """Run a single golden test.

        Args:
            test: The test to run.

        Returns:
            Result of running the test.
        """
        return self.compare_with_golden(test)

    def run_all(self) -> list[GoldenTestResult]:
        """Run all golden tests.

        Returns:
            List of results for each test.
        """
        self.results = []
        for test in self.tests:
            result = self.run_test(test)
            self.results.append(result)
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            logger.info(f"Golden test '{test.name}': {status}")
            if not result.passed and result.diff:
                logger.info(f"  {result.diff}")

        return self.results

    def _serialise(self, value: Any) -> Any:
        """Serialise a value to JSON-compatible format.

        Args:
            value: Value to serialise.

        Returns:
            JSON-serialisable representation.
        """
        if isinstance(value, (list, tuple)):
            return [self._serialise(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialise(v) for k, v in value.items()}
        elif isinstance(value, float):
            # Round to avoid floating-point precision issues
            return round(value, 10)
        elif isinstance(value, np.ndarray):
            return self._serialise(value.tolist())
        else:
            return value

    def _deep_compare(
        self,
        expected: Any,
        actual: Any,
        tolerance: float = 1e-9,
    ) -> bool:
        """Deep comparison with floating-point tolerance.

        Args:
            expected: Expected value.
            actual: Actual value.
            tolerance: Floating-point comparison tolerance.

        Returns:
            True if values are equal within tolerance.
        """
        if type(expected) != type(actual):
            return False

        if isinstance(expected, dict):
            if expected.keys() != actual.keys():
                return False
            return all(
                self._deep_compare(expected[k], actual[k], tolerance)
                for k in expected
            )
        elif isinstance(expected, list):
            if len(expected) != len(actual):
                return False
            return all(
                self._deep_compare(e, a, tolerance)
                for e, a in zip(expected, actual)
            )
        elif isinstance(expected, float):
            return abs(expected - actual) < tolerance
        else:
            return expected == actual

    def _generate_diff(self, expected: Any, actual: Any) -> str:
        """Generate a human-readable diff between expected and actual.

        Args:
            expected: Expected value.
            actual: Actual value.

        Returns:
            Description of differences.
        """
        diffs = []

        def compare_recursive(path: str, exp: Any, act: Any) -> None:
            if type(exp) != type(act):
                diffs.append(f"{path}: type mismatch ({type(exp).__name__} vs {type(act).__name__})")
                return

            if isinstance(exp, dict):
                all_keys = set(exp.keys()) | set(act.keys())
                for key in all_keys:
                    if key not in exp:
                        diffs.append(f"{path}.{key}: unexpected key")
                    elif key not in act:
                        diffs.append(f"{path}.{key}: missing key")
                    else:
                        compare_recursive(f"{path}.{key}", exp[key], act[key])
            elif isinstance(exp, list):
                if len(exp) != len(act):
                    diffs.append(f"{path}: length mismatch ({len(exp)} vs {len(act)})")
                else:
                    for i, (e, a) in enumerate(zip(exp, act)):
                        compare_recursive(f"{path}[{i}]", e, a)
            elif isinstance(exp, float):
                if abs(exp - act) >= 1e-9:
                    diffs.append(f"{path}: {exp} != {act}")
            elif exp != act:
                diffs.append(f"{path}: {exp} != {act}")

        compare_recursive("root", expected, actual)
        return "; ".join(diffs[:5]) + ("..." if len(diffs) > 5 else "")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: PERFORMANCE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes:
        name: Name of the benchmark.
        input_size: Size of the input.
        iterations: Number of iterations run.
        mean_time: Mean execution time in seconds.
        std_time: Standard deviation of execution time.
        min_time: Minimum execution time.
        max_time: Maximum execution time.
        throughput: Operations per second.
    """

    name: str
    input_size: int
    iterations: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    throughput: float


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection.

    Attributes:
        name: Name of the benchmark.
        input_size: Size of input used for baseline.
        mean_time: Expected mean execution time.
        tolerance: Acceptable deviation as fraction (e.g. 0.2 = 20%).
    """

    name: str
    input_size: int
    mean_time: float
    tolerance: float = 0.2


@dataclass
class ComplexityEstimate:
    """Estimated time complexity.

    Attributes:
        name: Name of the algorithm.
        complexity_class: Estimated complexity (e.g. "O(n)", "O(n log n)").
        coefficient: Estimated coefficient.
        r_squared: R² fit quality (1.0 = perfect fit).
    """

    name: str
    complexity_class: str
    coefficient: float
    r_squared: float


class BenchmarkFramework:
    """Framework for performance benchmarking and complexity estimation.

    Provides tools for measuring execution time, detecting performance
    regressions and estimating algorithmic complexity.

    Attributes:
        results: List of benchmark results.
        baselines: Dictionary of performance baselines.

    Example:
        >>> framework = BenchmarkFramework()
        >>> result = framework.benchmark(
        ...     name="sort_benchmark",
        ...     function=sort_data,
        ...     input_generator=lambda n: [random.random() for _ in range(n)],
        ...     input_size=1000,
        ... )
        >>> result.throughput > 0
        True
    """

    def __init__(self) -> None:
        """Initialise the benchmark framework."""
        self.results: list[BenchmarkResult] = []
        self.baselines: dict[str, PerformanceBaseline] = {}

    def benchmark(
        self,
        name: str,
        function: Callable[[Any], Any],
        input_generator: Callable[[int], Any],
        input_size: int,
        iterations: int = 10,
        warmup: int = 3,
    ) -> BenchmarkResult:
        """Run a benchmark for a given function.

        Args:
            name: Name for this benchmark.
            function: Function to benchmark.
            input_generator: Function that generates input of given size.
            input_size: Size of input to generate.
            iterations: Number of timed iterations.
            warmup: Number of warmup iterations (not timed).

        Returns:
            Benchmark result with timing statistics.
        """
        # Generate input once
        test_input = input_generator(input_size)

        # Warmup runs
        for _ in range(warmup):
            function(test_input)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            function(test_input)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0

        result = BenchmarkResult(
            name=name,
            input_size=input_size,
            iterations=iterations,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min(times),
            max_time=max(times),
            throughput=1.0 / mean_time if mean_time > 0 else float("inf"),
        )

        self.results.append(result)
        logger.info(
            f"Benchmark '{name}' (n={input_size}): "
            f"{mean_time*1000:.3f}ms ± {std_time*1000:.3f}ms"
        )

        return result

    def benchmark_scaling(
        self,
        name: str,
        function: Callable[[Any], Any],
        input_generator: Callable[[int], Any],
        sizes: list[int],
        iterations: int = 5,
    ) -> list[BenchmarkResult]:
        """Benchmark a function across multiple input sizes.

        Args:
            name: Base name for benchmarks.
            function: Function to benchmark.
            input_generator: Function that generates input of given size.
            sizes: List of input sizes to test.
            iterations: Number of iterations per size.

        Returns:
            List of benchmark results for each size.
        """
        results = []
        for size in sizes:
            result = self.benchmark(
                name=f"{name}_n{size}",
                function=function,
                input_generator=input_generator,
                input_size=size,
                iterations=iterations,
            )
            results.append(result)
        return results

    def estimate_complexity(
        self,
        results: list[BenchmarkResult],
    ) -> ComplexityEstimate:
        """Estimate algorithmic complexity from benchmark results.

        Fits the timing data to common complexity classes and returns
        the best-fitting complexity.

        Args:
            results: List of benchmark results at different input sizes.

        Returns:
            Estimated complexity class with fit quality.
        """
        if len(results) < 3:
            return ComplexityEstimate(
                name=results[0].name if results else "unknown",
                complexity_class="O(?)",
                coefficient=0.0,
                r_squared=0.0,
            )

        sizes = np.array([r.input_size for r in results], dtype=float)
        times = np.array([r.mean_time for r in results])

        # Define complexity functions to fit
        complexity_functions = {
            "O(1)": lambda n: np.ones_like(n),
            "O(log n)": lambda n: np.log(n + 1),
            "O(n)": lambda n: n,
            "O(n log n)": lambda n: n * np.log(n + 1),
            "O(n²)": lambda n: n ** 2,
            "O(n³)": lambda n: n ** 3,
        }

        best_fit = None
        best_r_squared = -float("inf")
        best_coefficient = 0.0

        for name, func in complexity_functions.items():
            # Compute transformed x values
            x = func(sizes)

            # Linear regression: time = coefficient * x
            # Using least squares: coefficient = sum(x*t) / sum(x*x)
            coefficient = np.sum(x * times) / (np.sum(x * x) + 1e-10)

            # Predicted values
            predicted = coefficient * x

            # R² calculation
            ss_res = np.sum((times - predicted) ** 2)
            ss_tot = np.sum((times - np.mean(times)) ** 2)
            r_squared = 1 - (ss_res / (ss_tot + 1e-10))

            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_fit = name
                best_coefficient = coefficient

        base_name = results[0].name.split("_n")[0] if results else "unknown"

        estimate = ComplexityEstimate(
            name=base_name,
            complexity_class=best_fit or "O(?)",
            coefficient=best_coefficient,
            r_squared=max(0.0, best_r_squared),
        )

        logger.info(
            f"Complexity estimate for '{base_name}': "
            f"{estimate.complexity_class} (R² = {estimate.r_squared:.4f})"
        )

        return estimate

    def set_baseline(self, baseline: PerformanceBaseline) -> None:
        """Set a performance baseline for regression detection.

        Args:
            baseline: The baseline to set.
        """
        self.baselines[baseline.name] = baseline
        logger.info(
            f"Set baseline for '{baseline.name}': "
            f"{baseline.mean_time*1000:.3f}ms ± {baseline.tolerance*100:.0f}%"
        )

    def check_baseline(self, result: BenchmarkResult) -> tuple[bool, str]:
        """Check if a benchmark result meets its baseline.

        Args:
            result: The benchmark result to check.

        Returns:
            Tuple of (passed, message).
        """
        baseline = self.baselines.get(result.name)
        if baseline is None:
            return True, "No baseline set"

        if result.input_size != baseline.input_size:
            return True, f"Input size mismatch (expected {baseline.input_size})"

        max_acceptable = baseline.mean_time * (1 + baseline.tolerance)
        passed = result.mean_time <= max_acceptable

        if passed:
            message = (
                f"Within baseline: {result.mean_time*1000:.3f}ms "
                f"<= {max_acceptable*1000:.3f}ms"
            )
        else:
            regression = (result.mean_time / baseline.mean_time - 1) * 100
            message = (
                f"REGRESSION: {result.mean_time*1000:.3f}ms "
                f"> {max_acceptable*1000:.3f}ms ({regression:.1f}% slower)"
            )

        return passed, message

    def summary(self) -> str:
        """Generate a summary report of benchmark results.

        Returns:
            Formatted string summarising all results.
        """
        if not self.results:
            return "No benchmarks have been run."

        lines = [
            "=" * 70,
            "BENCHMARK SUMMARY",
            "=" * 70,
            f"{'Name':<30} {'Size':>8} {'Mean (ms)':>12} {'Std (ms)':>10}",
            "-" * 70,
        ]

        for result in self.results:
            lines.append(
                f"{result.name:<30} {result.input_size:>8} "
                f"{result.mean_time*1000:>12.3f} {result.std_time*1000:>10.3f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMONSTRATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demonstrate_property_testing() -> None:
    """Demonstrate property-based testing functionality."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING PROPERTY-BASED TESTING")
    logger.info("=" * 60)

    suite = PropertyTestSuite()

    # Add tests for sorting
    suite.add_test(PropertyTest(
        name="sort_output_is_sorted",
        property_fn=sorted_output_is_sorted,
        generator=generate_random_list(0, 50, -100, 100),
        num_examples=100,
    ))

    suite.add_test(PropertyTest(
        name="sort_preserves_elements",
        property_fn=sorted_preserves_elements,
        generator=generate_random_list(0, 50, -100, 100),
        num_examples=100,
    ))

    # Add tests for normalisation
    suite.add_test(PropertyTest(
        name="normalised_values_in_range",
        property_fn=normalised_in_range,
        generator=generate_random_list(1, 50, -1000, 1000),
        num_examples=100,
    ))

    suite.add_test(PropertyTest(
        name="normalised_preserves_order",
        property_fn=normalised_preserves_order,
        generator=generate_random_list(2, 20, -100, 100),
        num_examples=50,
    ))

    # Run all tests
    suite.run_all(seed=42)
    print(suite.summary())


def demonstrate_golden_testing() -> None:
    """Demonstrate golden file testing functionality."""
    import tempfile

    logger.info("=" * 60)
    logger.info("DEMONSTRATING GOLDEN FILE TESTING")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        framework = GoldenTestFramework(Path(tmpdir))

        # Add tests
        framework.add_test(GoldenTest(
            name="statistics_basic",
            function=compute_statistics,
            inputs=([1.0, 2.0, 3.0, 4.0, 5.0],),
            golden_file=Path("stats_basic.json"),
        ))

        framework.add_test(GoldenTest(
            name="sort_basic",
            function=sort_data,
            inputs=([5.0, 2.0, 8.0, 1.0, 9.0],),
            golden_file=Path("sort_basic.json"),
        ))

        framework.add_test(GoldenTest(
            name="normalise_basic",
            function=normalise_data,
            inputs=([10.0, 20.0, 30.0, 40.0, 50.0],),
            golden_file=Path("normalise_basic.json"),
        ))

        # Generate golden files
        logger.info("Generating golden files...")
        framework.generate_all_golden()

        # Run tests
        logger.info("Running golden tests...")
        framework.run_all()


def demonstrate_benchmarking() -> None:
    """Demonstrate performance benchmarking functionality."""
    logger.info("=" * 60)
    logger.info("DEMONSTRATING PERFORMANCE BENCHMARKING")
    logger.info("=" * 60)

    framework = BenchmarkFramework()

    # Define input generator
    def list_generator(n: int) -> list[float]:
        random.seed(42)  # Consistent inputs
        return [random.random() for _ in range(n)]

    # Benchmark at multiple sizes
    sizes = [100, 500, 1000, 2000, 5000]
    results = framework.benchmark_scaling(
        name="sort_data",
        function=sort_data,
        input_generator=list_generator,
        sizes=sizes,
        iterations=5,
    )

    # Estimate complexity
    estimate = framework.estimate_complexity(results)
    logger.info(f"Estimated complexity: {estimate.complexity_class}")

    # Set and check baseline
    framework.set_baseline(PerformanceBaseline(
        name="sort_data_n1000",
        input_size=1000,
        mean_time=results[2].mean_time,  # Use actual result as baseline
        tolerance=0.3,
    ))

    # Re-run and check against baseline
    new_result = framework.benchmark(
        name="sort_data_n1000",
        function=sort_data,
        input_generator=list_generator,
        input_size=1000,
    )
    passed, message = framework.check_baseline(new_result)
    logger.info(f"Baseline check: {message}")

    print(framework.summary())


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demonstrate_property_testing()
    print()
    demonstrate_golden_testing()
    print()
    demonstrate_benchmarking()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Testing Framework - Solution"
    )
    parser.add_argument(
        "--demo",
        choices=["property", "golden", "benchmark", "all"],
        help="Run specific demonstration",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo == "property":
        demonstrate_property_testing()
    elif args.demo == "golden":
        demonstrate_golden_testing()
    elif args.demo == "benchmark":
        demonstrate_benchmarking()
    elif args.demo == "all":
        run_all_demos()
    else:
        run_all_demos()


if __name__ == "__main__":
    main()
