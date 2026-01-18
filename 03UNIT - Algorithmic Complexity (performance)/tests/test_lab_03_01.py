#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Tests for Lab 3.1 - Benchmark Suite
═══════════════════════════════════════════════════════════════════════════════

Thorough test suite for the benchmarking framework, covering:
- Timer functionality and accuracy
- BenchmarkResult data structures
- BenchmarkSuite orchestration
- Data generators
- Statistical analysis
- Export functionality

Coverage target: ≥80%

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_3_01_benchmark_suite import (
    Timer,
    BenchmarkResult,
    BenchmarkSuite,
    DataGenerator,
    SortingAlgorithms,
    SearchAlgorithms,
    StatisticalAnalyser,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TIMER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTimer:
    """Test suite for the Timer context manager."""

    def test_timer_measures_elapsed_time(self) -> None:
        """Timer should accurately measure elapsed time."""
        timer = Timer()
        with timer:
            time.sleep(0.05)  # Sleep for 50ms
        
        # Allow 20ms tolerance for system variance
        assert 0.04 <= timer.elapsed <= 0.1
    
    def test_timer_default_elapsed_is_zero(self) -> None:
        """Timer should have zero elapsed time before use."""
        timer = Timer()
        assert timer.elapsed == 0.0
    
    def test_timer_can_be_reused(self) -> None:
        """Timer should allow multiple measurements."""
        timer = Timer()
        
        with timer:
            time.sleep(0.02)
        first_elapsed = timer.elapsed
        
        with timer:
            time.sleep(0.03)
        second_elapsed = timer.elapsed
        
        assert second_elapsed != first_elapsed
        assert second_elapsed >= 0.02
    
    def test_timer_context_manager_protocol(self) -> None:
        """Timer should properly implement context manager protocol."""
        timer = Timer()
        result = timer.__enter__()
        assert result is timer
        
        timer.__exit__(None, None, None)
        assert timer.elapsed >= 0.0
    
    def test_timer_with_exception(self) -> None:
        """Timer should still record time even when exception occurs."""
        timer = Timer()
        
        with pytest.raises(ValueError):
            with timer:
                time.sleep(0.01)
                raise ValueError("Test exception")
        
        assert timer.elapsed >= 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BENCHMARK RESULT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkResult:
    """Test suite for BenchmarkResult dataclass."""

    def test_benchmark_result_creation(
        self,
        sample_benchmark_result: BenchmarkResult
    ) -> None:
        """BenchmarkResult should store all fields correctly."""
        result = sample_benchmark_result
        
        assert result.algorithm_name == "test_sort"
        assert result.input_size == 1000
        assert len(result.times) == 5
        assert result.mean_time > 0
        assert result.std_time >= 0
    
    def test_benchmark_result_statistics(self) -> None:
        """BenchmarkResult should calculate statistics correctly."""
        times = [0.1, 0.2, 0.15, 0.18, 0.12]
        result = BenchmarkResult(
            algorithm_name="test",
            input_size=100,
            times=times,
            mean_time=sum(times) / len(times),
            std_time=0.038,  # Approximate
            min_time=min(times),
            max_time=max(times),
            memory_bytes=1024,
        )
        
        assert result.mean_time == pytest.approx(0.15, rel=0.01)
        assert result.min_time == 0.1
        assert result.max_time == 0.2
    
    def test_benchmark_result_to_dict(
        self,
        sample_benchmark_result: BenchmarkResult
    ) -> None:
        """BenchmarkResult should convert to dictionary correctly."""
        result = sample_benchmark_result
        
        # Check required fields exist
        assert hasattr(result, 'algorithm_name')
        assert hasattr(result, 'input_size')
        assert hasattr(result, 'mean_time')
    
    def test_benchmark_result_with_metadata(self) -> None:
        """BenchmarkResult should handle optional metadata."""
        result = BenchmarkResult(
            algorithm_name="test",
            input_size=100,
            times=[0.1],
            mean_time=0.1,
            std_time=0.0,
            min_time=0.1,
            max_time=0.1,
            memory_bytes=512,
            comparisons=50,
            swaps=25,
        )
        
        assert result.comparisons == 50
        assert result.swaps == 25


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: DATA GENERATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataGenerator:
    """Test suite for DataGenerator class."""

    def test_generate_random_list(self) -> None:
        """DataGenerator should create random lists of specified size."""
        data = DataGenerator.random_list(100)
        
        assert len(data) == 100
        assert all(isinstance(x, int) for x in data)
    
    def test_generate_sorted_list(self) -> None:
        """DataGenerator should create sorted lists."""
        data = DataGenerator.sorted_list(100)
        
        assert len(data) == 100
        assert data == sorted(data)
    
    def test_generate_reversed_list(self) -> None:
        """DataGenerator should create reverse-sorted lists."""
        data = DataGenerator.reversed_list(100)
        
        assert len(data) == 100
        assert data == sorted(data, reverse=True)
    
    def test_generate_nearly_sorted_list(self) -> None:
        """DataGenerator should create nearly sorted lists."""
        data = DataGenerator.nearly_sorted_list(100, swaps=5)
        
        assert len(data) == 100
        # Should be mostly sorted (inversions count)
        inversions = sum(
            1 for i in range(len(data) - 1)
            if data[i] > data[i + 1]
        )
        assert inversions <= 10  # Allow some tolerance
    
    def test_generate_repeated_elements(self) -> None:
        """DataGenerator should create lists with repeated elements."""
        data = DataGenerator.repeated_elements(100, unique=5)
        
        assert len(data) == 100
        assert len(set(data)) == 5
    
    def test_generate_empty_list(self) -> None:
        """DataGenerator should handle size zero gracefully."""
        data = DataGenerator.random_list(0)
        assert len(data) == 0
    
    @pytest.mark.parametrize("size", [1, 10, 100, 1000])
    def test_generate_various_sizes(self, size: int) -> None:
        """DataGenerator should work for various sizes."""
        data = DataGenerator.random_list(size)
        assert len(data) == size


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SORTING ALGORITHMS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSortingAlgorithms:
    """Test suite for sorting algorithm implementations."""

    @pytest.fixture
    def unsorted_data(self) -> list[int]:
        """Provide unsorted test data."""
        return [64, 34, 25, 12, 22, 11, 90]
    
    @pytest.fixture
    def expected_sorted(self) -> list[int]:
        """Provide expected sorted result."""
        return [11, 12, 22, 25, 34, 64, 90]

    def test_bubble_sort(
        self,
        unsorted_data: list[int],
        expected_sorted: list[int]
    ) -> None:
        """Bubble sort should correctly sort data."""
        data = unsorted_data.copy()
        SortingAlgorithms.bubble_sort(data)
        assert data == expected_sorted
    
    def test_insertion_sort(
        self,
        unsorted_data: list[int],
        expected_sorted: list[int]
    ) -> None:
        """Insertion sort should correctly sort data."""
        data = unsorted_data.copy()
        SortingAlgorithms.insertion_sort(data)
        assert data == expected_sorted
    
    def test_selection_sort(
        self,
        unsorted_data: list[int],
        expected_sorted: list[int]
    ) -> None:
        """Selection sort should correctly sort data."""
        data = unsorted_data.copy()
        SortingAlgorithms.selection_sort(data)
        assert data == expected_sorted
    
    def test_merge_sort(
        self,
        unsorted_data: list[int],
        expected_sorted: list[int]
    ) -> None:
        """Merge sort should correctly sort data."""
        data = unsorted_data.copy()
        result = SortingAlgorithms.merge_sort(data)
        assert result == expected_sorted
    
    def test_quick_sort(
        self,
        unsorted_data: list[int],
        expected_sorted: list[int]
    ) -> None:
        """Quick sort should correctly sort data."""
        data = unsorted_data.copy()
        SortingAlgorithms.quick_sort(data)
        assert data == expected_sorted
    
    def test_heap_sort(
        self,
        unsorted_data: list[int],
        expected_sorted: list[int]
    ) -> None:
        """Heap sort should correctly sort data."""
        data = unsorted_data.copy()
        SortingAlgorithms.heap_sort(data)
        assert data == expected_sorted
    
    def test_sort_empty_list(self) -> None:
        """Sorting algorithms should handle empty lists."""
        data: list[int] = []
        SortingAlgorithms.bubble_sort(data)
        assert data == []
    
    def test_sort_single_element(self) -> None:
        """Sorting algorithms should handle single element lists."""
        data = [42]
        SortingAlgorithms.bubble_sort(data)
        assert data == [42]
    
    def test_sort_already_sorted(
        self,
        expected_sorted: list[int]
    ) -> None:
        """Sorting algorithms should handle already sorted data."""
        data = expected_sorted.copy()
        SortingAlgorithms.insertion_sort(data)
        assert data == expected_sorted
    
    def test_sort_duplicates(self) -> None:
        """Sorting algorithms should handle duplicate values."""
        data = [5, 2, 5, 1, 2, 5]
        SortingAlgorithms.merge_sort(data)
        assert sorted(data) == [1, 2, 2, 5, 5, 5]
    
    @pytest.mark.parametrize("algorithm", [
        SortingAlgorithms.bubble_sort,
        SortingAlgorithms.insertion_sort,
        SortingAlgorithms.selection_sort,
        SortingAlgorithms.quick_sort,
        SortingAlgorithms.heap_sort,
    ])
    def test_all_algorithms_produce_sorted_output(
        self,
        algorithm,
        random_test_data: list[int]
    ) -> None:
        """All sorting algorithms should produce correctly sorted output."""
        data = random_test_data.copy()
        algorithm(data)
        assert data == sorted(random_test_data)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SEARCH ALGORITHMS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchAlgorithms:
    """Test suite for search algorithm implementations."""

    @pytest.fixture
    def sorted_data(self) -> list[int]:
        """Provide sorted test data."""
        return [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

    def test_linear_search_found(self, sorted_data: list[int]) -> None:
        """Linear search should find existing element."""
        index = SearchAlgorithms.linear_search(sorted_data, 7)
        assert index == 3
    
    def test_linear_search_not_found(self, sorted_data: list[int]) -> None:
        """Linear search should return -1 for missing element."""
        index = SearchAlgorithms.linear_search(sorted_data, 8)
        assert index == -1
    
    def test_binary_search_found(self, sorted_data: list[int]) -> None:
        """Binary search should find existing element."""
        index = SearchAlgorithms.binary_search(sorted_data, 11)
        assert index == 5
    
    def test_binary_search_not_found(self, sorted_data: list[int]) -> None:
        """Binary search should return -1 for missing element."""
        index = SearchAlgorithms.binary_search(sorted_data, 6)
        assert index == -1
    
    def test_binary_search_first_element(self, sorted_data: list[int]) -> None:
        """Binary search should find first element."""
        index = SearchAlgorithms.binary_search(sorted_data, 1)
        assert index == 0
    
    def test_binary_search_last_element(self, sorted_data: list[int]) -> None:
        """Binary search should find last element."""
        index = SearchAlgorithms.binary_search(sorted_data, 19)
        assert index == 9
    
    def test_search_empty_list(self) -> None:
        """Search algorithms should handle empty lists."""
        assert SearchAlgorithms.linear_search([], 5) == -1
        assert SearchAlgorithms.binary_search([], 5) == -1
    
    def test_search_single_element_found(self) -> None:
        """Search algorithms should find element in single-element list."""
        assert SearchAlgorithms.linear_search([42], 42) == 0
        assert SearchAlgorithms.binary_search([42], 42) == 0
    
    def test_search_single_element_not_found(self) -> None:
        """Search algorithms should handle missing element in single-element list."""
        assert SearchAlgorithms.linear_search([42], 5) == -1
        assert SearchAlgorithms.binary_search([42], 5) == -1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: BENCHMARK SUITE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkSuite:
    """Test suite for BenchmarkSuite orchestration."""

    def test_suite_creation(self) -> None:
        """BenchmarkSuite should initialise correctly."""
        suite = BenchmarkSuite(
            name="test_suite",
            warmup_runs=2,
            benchmark_runs=3,
        )
        
        assert suite.name == "test_suite"
        assert suite.warmup_runs == 2
        assert suite.benchmark_runs == 3
    
    def test_register_algorithm(self) -> None:
        """BenchmarkSuite should register algorithms correctly."""
        suite = BenchmarkSuite(name="test")
        
        def dummy_sort(data: list[int]) -> None:
            data.sort()
        
        suite.register_algorithm("dummy_sort", dummy_sort)
        assert "dummy_sort" in suite.algorithms
    
    def test_register_data_generator(self) -> None:
        """BenchmarkSuite should register data generators correctly."""
        suite = BenchmarkSuite(name="test")
        
        def dummy_generator(size: int) -> list[int]:
            return list(range(size))
        
        suite.register_data_generator("sequential", dummy_generator)
        assert "sequential" in suite.data_generators
    
    def test_run_benchmark_single_algorithm(self) -> None:
        """BenchmarkSuite should benchmark a single algorithm."""
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=1,
            benchmark_runs=3,
        )
        
        def simple_sort(data: list[int]) -> None:
            data.sort()
        
        suite.register_algorithm("simple_sort", simple_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["simple_sort"],
            generators=["random"],
            sizes=[100],
        )
        
        assert len(results) == 1
        assert results[0].algorithm_name == "simple_sort"
        assert results[0].input_size == 100
        assert len(results[0].times) == 3
    
    def test_run_benchmark_multiple_sizes(self) -> None:
        """BenchmarkSuite should benchmark across multiple input sizes."""
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=1,
            benchmark_runs=2,
        )
        
        def simple_sort(data: list[int]) -> None:
            data.sort()
        
        suite.register_algorithm("simple_sort", simple_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["simple_sort"],
            generators=["random"],
            sizes=[50, 100, 200],
        )
        
        assert len(results) == 3
        sizes = [r.input_size for r in results]
        assert sizes == [50, 100, 200]
    
    def test_run_benchmark_multiple_algorithms(self) -> None:
        """BenchmarkSuite should benchmark multiple algorithms."""
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=1,
            benchmark_runs=2,
        )
        
        suite.register_algorithm("bubble", SortingAlgorithms.bubble_sort)
        suite.register_algorithm("insertion", SortingAlgorithms.insertion_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["bubble", "insertion"],
            generators=["random"],
            sizes=[50],
        )
        
        assert len(results) == 2
        names = {r.algorithm_name for r in results}
        assert names == {"bubble", "insertion"}
    
    def test_benchmark_warmup_effect(self) -> None:
        """BenchmarkSuite should perform warmup runs before timing."""
        call_count = 0
        
        def counting_sort(data: list[int]) -> None:
            nonlocal call_count
            call_count += 1
            data.sort()
        
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=3,
            benchmark_runs=5,
        )
        
        suite.register_algorithm("counting", counting_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        suite.run(
            algorithms=["counting"],
            generators=["random"],
            sizes=[50],
        )
        
        # Warmup + benchmark runs
        assert call_count == 8  # 3 warmup + 5 benchmark


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: STATISTICAL ANALYSER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStatisticalAnalyser:
    """Test suite for StatisticalAnalyser class."""

    def test_calculate_mean(self) -> None:
        """StatisticalAnalyser should calculate mean correctly."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean = StatisticalAnalyser.calculate_mean(times)
        assert mean == pytest.approx(3.0)
    
    def test_calculate_std(self) -> None:
        """StatisticalAnalyser should calculate standard deviation correctly."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        std = StatisticalAnalyser.calculate_std(times)
        # Population std dev of [1,2,3,4,5] is sqrt(2)
        assert std == pytest.approx(math.sqrt(2), rel=0.01)
    
    def test_calculate_median(self) -> None:
        """StatisticalAnalyser should calculate median correctly."""
        times_odd = [1.0, 3.0, 5.0, 7.0, 9.0]
        times_even = [1.0, 3.0, 5.0, 7.0]
        
        assert StatisticalAnalyser.calculate_median(times_odd) == 5.0
        assert StatisticalAnalyser.calculate_median(times_even) == 4.0
    
    def test_calculate_percentile(self) -> None:
        """StatisticalAnalyser should calculate percentiles correctly."""
        times = list(range(1, 101))  # 1 to 100
        
        p50 = StatisticalAnalyser.calculate_percentile(times, 50)
        p95 = StatisticalAnalyser.calculate_percentile(times, 95)
        
        assert p50 == pytest.approx(50, abs=1)
        assert p95 == pytest.approx(95, abs=1)
    
    def test_detect_outliers(self) -> None:
        """StatisticalAnalyser should detect outliers using IQR method."""
        times = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 100.0]  # 100.0 is outlier
        
        outliers = StatisticalAnalyser.detect_outliers(times)
        assert 100.0 in outliers
        assert len(outliers) == 1
    
    def test_confidence_interval(self) -> None:
        """StatisticalAnalyser should calculate confidence intervals."""
        times = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        ci_low, ci_high = StatisticalAnalyser.confidence_interval(
            times,
            confidence=0.95
        )
        
        assert ci_low < 3.0 < ci_high
        assert ci_low > 0
    
    def test_empty_data_handling(self) -> None:
        """StatisticalAnalyser should handle empty data gracefully."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            StatisticalAnalyser.calculate_mean([])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EXPORT FUNCTIONALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportFunctionality:
    """Test suite for result export functionality."""

    def test_export_to_csv(
        self,
        sample_benchmark_results: list[BenchmarkResult]
    ) -> None:
        """BenchmarkSuite should export results to CSV."""
        suite = BenchmarkSuite(name="test")
        suite.results = sample_benchmark_results
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            csv_path = Path(f.name)
        
        try:
            suite.export_csv(csv_path)
            
            assert csv_path.exists()
            content = csv_path.read_text()
            assert "algorithm_name" in content
            assert "input_size" in content
            assert "mean_time" in content
        finally:
            csv_path.unlink(missing_ok=True)
    
    def test_export_to_json(
        self,
        sample_benchmark_results: list[BenchmarkResult]
    ) -> None:
        """BenchmarkSuite should export results to JSON."""
        suite = BenchmarkSuite(name="test")
        suite.results = sample_benchmark_results
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json_path = Path(f.name)
        
        try:
            suite.export_json(json_path)
            
            assert json_path.exists()
            data = json.loads(json_path.read_text())
            assert isinstance(data, list)
            assert len(data) == len(sample_benchmark_results)
        finally:
            json_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for complete benchmark workflows."""

    def test_full_sorting_benchmark_workflow(self) -> None:
        """Complete sorting benchmark should run end-to-end."""
        suite = BenchmarkSuite(
            name="sorting_integration_test",
            warmup_runs=1,
            benchmark_runs=3,
        )
        
        # Register algorithms
        suite.register_algorithm("bubble", SortingAlgorithms.bubble_sort)
        suite.register_algorithm("merge", SortingAlgorithms.merge_sort)
        
        # Register generators
        suite.register_data_generator("random", DataGenerator.random_list)
        suite.register_data_generator("sorted", DataGenerator.sorted_list)
        
        # Run benchmarks
        results = suite.run(
            algorithms=["bubble", "merge"],
            generators=["random", "sorted"],
            sizes=[50, 100],
        )
        
        # Verify results structure
        assert len(results) == 8  # 2 algorithms × 2 generators × 2 sizes
        
        # Verify all results have required fields
        for result in results:
            assert result.algorithm_name in ["bubble", "merge"]
            assert result.input_size in [50, 100]
            assert result.mean_time > 0
            assert len(result.times) == 3
    
    def test_search_benchmark_workflow(self) -> None:
        """Complete search benchmark should run end-to-end."""
        suite = BenchmarkSuite(
            name="search_integration_test",
            warmup_runs=1,
            benchmark_runs=3,
        )
        
        # Search algorithms need sorted data
        def linear_wrapper(data: list[int]) -> None:
            SearchAlgorithms.linear_search(data, data[len(data) // 2])
        
        def binary_wrapper(data: list[int]) -> None:
            SearchAlgorithms.binary_search(data, data[len(data) // 2])
        
        suite.register_algorithm("linear", linear_wrapper)
        suite.register_algorithm("binary", binary_wrapper)
        suite.register_data_generator("sorted", DataGenerator.sorted_list)
        
        results = suite.run(
            algorithms=["linear", "binary"],
            generators=["sorted"],
            sizes=[1000, 10000],
        )
        
        assert len(results) == 4
    
    def test_benchmark_scaling_behaviour(self) -> None:
        """Benchmark should show expected scaling behaviour."""
        suite = BenchmarkSuite(
            name="scaling_test",
            warmup_runs=1,
            benchmark_runs=3,
        )
        
        suite.register_algorithm("bubble", SortingAlgorithms.bubble_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["bubble"],
            generators=["random"],
            sizes=[100, 200, 400],
        )
        
        # Bubble sort is O(n²), so doubling n should roughly quadruple time
        times = {r.input_size: r.mean_time for r in results}
        
        # Verify increasing trend (with tolerance for system variance)
        assert times[200] > times[100] * 1.5
        assert times[400] > times[200] * 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_benchmark_with_zero_warmup(self) -> None:
        """Benchmark should work with zero warmup runs."""
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=0,
            benchmark_runs=3,
        )
        
        suite.register_algorithm("sort", SortingAlgorithms.bubble_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["sort"],
            generators=["random"],
            sizes=[50],
        )
        
        assert len(results) == 1
        assert len(results[0].times) == 3
    
    def test_benchmark_with_single_run(self) -> None:
        """Benchmark should work with single benchmark run."""
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=1,
            benchmark_runs=1,
        )
        
        suite.register_algorithm("sort", SortingAlgorithms.bubble_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["sort"],
            generators=["random"],
            sizes=[50],
        )
        
        assert len(results[0].times) == 1
        assert results[0].std_time == 0.0  # No variance with single sample
    
    def test_unregistered_algorithm(self) -> None:
        """Benchmark should raise error for unregistered algorithm."""
        suite = BenchmarkSuite(name="test")
        suite.register_data_generator("random", DataGenerator.random_list)
        
        with pytest.raises(KeyError):
            suite.run(
                algorithms=["nonexistent"],
                generators=["random"],
                sizes=[50],
            )
    
    def test_unregistered_generator(self) -> None:
        """Benchmark should raise error for unregistered generator."""
        suite = BenchmarkSuite(name="test")
        suite.register_algorithm("sort", SortingAlgorithms.bubble_sort)
        
        with pytest.raises(KeyError):
            suite.run(
                algorithms=["sort"],
                generators=["nonexistent"],
                sizes=[50],
            )
    
    def test_very_small_input(self) -> None:
        """Benchmark should handle very small inputs."""
        suite = BenchmarkSuite(
            name="test",
            warmup_runs=1,
            benchmark_runs=3,
        )
        
        suite.register_algorithm("sort", SortingAlgorithms.bubble_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["sort"],
            generators=["random"],
            sizes=[1, 2, 3],
        )
        
        assert len(results) == 3
        for result in results:
            assert result.mean_time >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: PERFORMANCE TESTS (MARKED AS SLOW)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPerformance:
    """Performance tests for larger inputs (marked as slow)."""

    def test_large_input_benchmark(self) -> None:
        """Benchmark should handle larger inputs."""
        suite = BenchmarkSuite(
            name="large_test",
            warmup_runs=1,
            benchmark_runs=3,
        )
        
        suite.register_algorithm("merge", SortingAlgorithms.merge_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["merge"],
            generators=["random"],
            sizes=[10000, 50000],
        )
        
        assert len(results) == 2
        # Merge sort should complete in reasonable time
        assert all(r.mean_time < 10.0 for r in results)
    
    def test_many_runs_statistical_stability(self) -> None:
        """Many runs should produce statistically stable results."""
        suite = BenchmarkSuite(
            name="stats_test",
            warmup_runs=5,
            benchmark_runs=20,
        )
        
        suite.register_algorithm("insertion", SortingAlgorithms.insertion_sort)
        suite.register_data_generator("random", DataGenerator.random_list)
        
        results = suite.run(
            algorithms=["insertion"],
            generators=["random"],
            sizes=[100],
        )
        
        result = results[0]
        # Coefficient of variation should be reasonable (< 50%)
        if result.mean_time > 0:
            cv = result.std_time / result.mean_time
            assert cv < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
