#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Algorithmic Complexity
Pytest Configuration and Shared Fixtures
═══════════════════════════════════════════════════════════════════════════════

This module provides shared pytest fixtures and configuration for testing the
Week 3 laboratory modules:
- BenchmarkSuite testing utilities
- ComplexityAnalyser testing utilities
- Sample data generators
- Performance testing helpers

USAGE
─────
Fixtures are automatically available to all test modules in this directory.

    def test_example(sample_array, benchmark_config):
        # Fixtures are injected automatically
        assert len(sample_array) == 100

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar("T")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests."""

    warmup_runs: int = 2
    measurement_runs: int = 5
    min_time_threshold: float = 1e-9
    max_time_threshold: float = 10.0
    tolerance_factor: float = 0.5


@dataclass
class TestDataSet:
    """Container for test data with metadata."""

    name: str
    data: list[Any]
    expected_complexity: str
    size: int = field(init=False)

    def __post_init__(self) -> None:
        """Calculate size after initialisation."""
        self.size = len(self.data)


@dataclass
class TimingResult:
    """Result of a timing operation."""

    elapsed_seconds: float
    iterations: int
    mean_time: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate mean time after initialisation."""
        self.mean_time = self.elapsed_seconds / max(self.iterations, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_array() -> list[int]:
    """Provide a sample array of 100 random integers."""
    random.seed(42)
    return [random.randint(1, 1000) for _ in range(100)]


@pytest.fixture
def sorted_array() -> list[int]:
    """Provide a sorted array of 100 integers."""
    return list(range(100))


@pytest.fixture
def reversed_array() -> list[int]:
    """Provide a reverse-sorted array of 100 integers."""
    return list(range(99, -1, -1))


@pytest.fixture
def small_array() -> list[int]:
    """Provide a small array for quick tests."""
    return [5, 2, 8, 1, 9, 3, 7, 4, 6]


@pytest.fixture
def large_array() -> list[int]:
    """Provide a larger array for performance tests."""
    random.seed(42)
    return [random.randint(1, 10000) for _ in range(1000)]


@pytest.fixture(params=[10, 50, 100, 500, 1000])
def variable_size_array(request: pytest.FixtureRequest) -> list[int]:
    """Provide arrays of various sizes for complexity testing."""
    size = request.param
    random.seed(42)
    return [random.randint(1, 10000) for _ in range(size)]


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CONFIGURATION FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Provide default benchmark configuration."""
    return BenchmarkConfig()


@pytest.fixture
def quick_benchmark_config() -> BenchmarkConfig:
    """Provide configuration for quick benchmark tests."""
    return BenchmarkConfig(
        warmup_runs=1,
        measurement_runs=3,
        max_time_threshold=5.0,
    )


@pytest.fixture
def thorough_benchmark_config() -> BenchmarkConfig:
    """Provide configuration for thorough benchmark tests."""
    return BenchmarkConfig(
        warmup_runs=5,
        measurement_runs=10,
        tolerance_factor=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TEST DATA SET FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def linear_test_data() -> TestDataSet:
    """Provide test data expected to have linear complexity."""
    random.seed(42)
    return TestDataSet(
        name="linear_search_data",
        data=[random.randint(1, 10000) for _ in range(500)],
        expected_complexity="O(n)",
    )


@pytest.fixture
def quadratic_test_data() -> TestDataSet:
    """Provide test data expected to have quadratic complexity."""
    random.seed(42)
    return TestDataSet(
        name="bubble_sort_data",
        data=[random.randint(1, 10000) for _ in range(200)],
        expected_complexity="O(n²)",
    )


@pytest.fixture
def logarithmic_test_data() -> TestDataSet:
    """Provide sorted test data for logarithmic algorithms."""
    return TestDataSet(
        name="binary_search_data",
        data=list(range(1000)),
        expected_complexity="O(log n)",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def timer() -> Callable[..., TimingResult]:
    """Provide a timing utility function."""

    def time_function(
        func: Callable[..., T],
        *args: Any,
        iterations: int = 1,
        **kwargs: Any,
    ) -> TimingResult:
        """Time a function's execution.

        Args:
            func: The function to time.
            *args: Positional arguments for the function.
            iterations: Number of iterations to run.
            **kwargs: Keyword arguments for the function.

        Returns:
            TimingResult with elapsed time and iteration count.
        """
        start = time.perf_counter()
        for _ in range(iterations):
            func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return TimingResult(elapsed_seconds=elapsed, iterations=iterations)

    return time_function


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE ALGORITHM FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sorting_algorithms() -> dict[str, Callable[[list[int]], list[int]]]:
    """Provide a collection of sorting algorithms for testing."""

    def bubble_sort(arr: list[int]) -> list[int]:
        """Simple bubble sort implementation."""
        result = arr.copy()
        n = len(result)
        for i in range(n):
            for j in range(0, n - i - 1):
                if result[j] > result[j + 1]:
                    result[j], result[j + 1] = result[j + 1], result[j]
        return result

    def insertion_sort(arr: list[int]) -> list[int]:
        """Simple insertion sort implementation."""
        result = arr.copy()
        for i in range(1, len(result)):
            key = result[i]
            j = i - 1
            while j >= 0 and result[j] > key:
                result[j + 1] = result[j]
                j -= 1
            result[j + 1] = key
        return result

    def selection_sort(arr: list[int]) -> list[int]:
        """Simple selection sort implementation."""
        result = arr.copy()
        n = len(result)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                if result[j] < result[min_idx]:
                    min_idx = j
            result[i], result[min_idx] = result[min_idx], result[i]
        return result

    def merge_sort(arr: list[int]) -> list[int]:
        """Simple merge sort implementation."""
        if len(arr) <= 1:
            return arr.copy()

        mid = len(arr) // 2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])

        result: list[int] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    return {
        "bubble_sort": bubble_sort,
        "insertion_sort": insertion_sort,
        "selection_sort": selection_sort,
        "merge_sort": merge_sort,
    }


@pytest.fixture
def search_algorithms() -> dict[str, Callable[[list[int], int], int]]:
    """Provide a collection of search algorithms for testing."""

    def linear_search(arr: list[int], target: int) -> int:
        """Simple linear search implementation."""
        for i, val in enumerate(arr):
            if val == target:
                return i
        return -1

    def binary_search(arr: list[int], target: int) -> int:
        """Simple binary search implementation (requires sorted array)."""
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    return {
        "linear_search": linear_search,
        "binary_search": binary_search,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT SIZE GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def input_sizes_small() -> list[int]:
    """Provide small input sizes for quick tests."""
    return [10, 20, 50, 100]


@pytest.fixture
def input_sizes_medium() -> list[int]:
    """Provide medium input sizes for standard tests."""
    return [100, 200, 500, 1000, 2000]


@pytest.fixture
def input_sizes_large() -> list[int]:
    """Provide large input sizes for performance tests."""
    return [1000, 2000, 5000, 10000, 20000]


@pytest.fixture
def input_sizes_exponential() -> list[int]:
    """Provide exponentially increasing input sizes."""
    return [2**i for i in range(4, 14)]  # 16 to 8192


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORARY FILE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "benchmark_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def temp_csv_file(tmp_path: Path) -> Path:
    """Provide a temporary CSV file path."""
    return tmp_path / "test_results.csv"


@pytest.fixture
def temp_json_file(tmp_path: Path) -> Path:
    """Provide a temporary JSON file path."""
    return tmp_path / "test_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# ASSERTION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def assert_complexity() -> Callable[[list[float], list[int], str], None]:
    """Provide a helper to assert expected complexity.

    This fixture returns a function that checks if measured times
    follow the expected complexity pattern within tolerance.
    """

    def _assert_complexity(
        times: list[float],
        sizes: list[int],
        expected: str,
        tolerance: float = 0.5,
    ) -> None:
        """Assert that times follow expected complexity.

        Args:
            times: Measured execution times.
            sizes: Corresponding input sizes.
            expected: Expected complexity class (e.g., "O(n)", "O(n²)").
            tolerance: Allowed deviation from expected ratio.
        """
        if len(times) < 2 or len(sizes) < 2:
            return

        # Calculate ratios between consecutive measurements
        ratios: list[float] = []
        for i in range(1, len(times)):
            if times[i - 1] > 0:
                ratios.append(times[i] / times[i - 1])

        # Calculate expected ratios based on complexity
        size_ratios: list[float] = []
        for i in range(1, len(sizes)):
            size_ratios.append(sizes[i] / sizes[i - 1])

        expected_ratios: list[float] = []
        for sr in size_ratios:
            if expected == "O(1)":
                expected_ratios.append(1.0)
            elif expected == "O(log n)":
                expected_ratios.append(1.0 + 1 / (2 * sr))  # Approximate
            elif expected == "O(n)":
                expected_ratios.append(sr)
            elif expected in ("O(n log n)", "O(n·log n)"):
                expected_ratios.append(sr * 1.1)  # Approximate
            elif expected in ("O(n²)", "O(n^2)"):
                expected_ratios.append(sr * sr)
            else:
                expected_ratios.append(sr)

        # Verify ratios are within tolerance
        for actual, expected_val in zip(ratios, expected_ratios):
            lower = expected_val * (1 - tolerance)
            upper = expected_val * (1 + tolerance)
            assert lower <= actual <= upper or actual < 0.001, (
                f"Complexity mismatch: expected ratio ~{expected_val:.2f}, "
                f"got {actual:.2f}"
            )

    return _assert_complexity


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "complexity: marks tests that verify complexity"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection based on markers."""
    # Skip slow tests unless explicitly requested
    if not config.getoption("-m", default=""):
        skip_slow = pytest.mark.skip(reason="Slow test - use -m slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def array_generator() -> Callable[[int, str], list[int]]:
    """Provide a function to generate arrays of different types."""

    def _generate(size: int, array_type: str = "random") -> list[int]:
        """Generate an array of the specified type.

        Args:
            size: Number of elements.
            array_type: One of 'random', 'sorted', 'reversed', 'nearly_sorted'.

        Returns:
            Generated array.
        """
        if array_type == "random":
            random.seed(42)
            return [random.randint(1, size * 10) for _ in range(size)]
        elif array_type == "sorted":
            return list(range(size))
        elif array_type == "reversed":
            return list(range(size - 1, -1, -1))
        elif array_type == "nearly_sorted":
            arr = list(range(size))
            # Swap ~10% of elements
            random.seed(42)
            for _ in range(size // 10):
                i, j = random.randint(0, size - 1), random.randint(0, size - 1)
                arr[i], arr[j] = arr[j], arr[i]
            return arr
        else:
            raise ValueError(f"Unknown array type: {array_type}")

    return _generate
