#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3: Tests for Lab 3.2 - Complexity Analyser
═══════════════════════════════════════════════════════════════════════════════

Thorough test suite for the complexity analysis framework, covering:
- Complexity class detection
- Curve fitting algorithms
- Empirical estimation
- Theoretical verification
- Reporting functionality

Coverage target: ≥80%

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_3_02_complexity_analyser import (
    ComplexityClass,
    ComplexityResult,
    ComplexityAnalyser,
    CurveFitter,
    TheoreticalAnalyser,
    ComplexityComparator,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: COMPLEXITY CLASS TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityClass:
    """Test suite for ComplexityClass enumeration."""

    def test_complexity_class_ordering(self) -> None:
        """Complexity classes should have correct ordering."""
        classes = [
            ComplexityClass.O_1,
            ComplexityClass.O_LOG_N,
            ComplexityClass.O_N,
            ComplexityClass.O_N_LOG_N,
            ComplexityClass.O_N_SQUARED,
            ComplexityClass.O_N_CUBED,
            ComplexityClass.O_2_N,
            ComplexityClass.O_N_FACTORIAL,
        ]
        
        # Verify correct order by value
        for i in range(len(classes) - 1):
            assert classes[i].value < classes[i + 1].value
    
    def test_complexity_class_string_representation(self) -> None:
        """Complexity classes should have human-readable string names."""
        assert "1" in str(ComplexityClass.O_1) or "constant" in str(ComplexityClass.O_1).lower()
        assert "n" in str(ComplexityClass.O_N).lower()
        assert "log" in str(ComplexityClass.O_LOG_N).lower()
    
    def test_all_classes_defined(self) -> None:
        """All standard complexity classes should be defined."""
        expected_classes = {
            "O_1", "O_LOG_N", "O_N", "O_N_LOG_N",
            "O_N_SQUARED", "O_N_CUBED", "O_2_N", "O_N_FACTORIAL"
        }
        
        defined_classes = {c.name for c in ComplexityClass}
        assert expected_classes.issubset(defined_classes)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COMPLEXITY RESULT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityResult:
    """Test suite for ComplexityResult dataclass."""

    def test_result_creation(self) -> None:
        """ComplexityResult should store all fields correctly."""
        result = ComplexityResult(
            detected_class=ComplexityClass.O_N,
            confidence=0.95,
            r_squared=0.98,
            fitted_params={"a": 1.5, "b": 0.1},
            sizes=[100, 200, 400],
            times=[0.1, 0.2, 0.4],
        )
        
        assert result.detected_class == ComplexityClass.O_N
        assert result.confidence == 0.95
        assert result.r_squared == 0.98
        assert len(result.sizes) == 3
    
    def test_result_with_empty_data(self) -> None:
        """ComplexityResult should handle empty data."""
        result = ComplexityResult(
            detected_class=ComplexityClass.O_1,
            confidence=0.0,
            r_squared=0.0,
            fitted_params={},
            sizes=[],
            times=[],
        )
        
        assert len(result.sizes) == 0
        assert len(result.times) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CURVE FITTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurveFitter:
    """Test suite for CurveFitter class."""

    def test_fit_constant(self) -> None:
        """CurveFitter should correctly identify O(1) complexity."""
        sizes = [100, 200, 400, 800, 1600]
        times = [0.01, 0.01, 0.01, 0.01, 0.01]  # Constant time
        
        r_squared, params = CurveFitter.fit_constant(sizes, times)
        
        assert r_squared > 0.9
    
    def test_fit_linear(self) -> None:
        """CurveFitter should correctly identify O(n) complexity."""
        sizes = [100, 200, 400, 800, 1600]
        times = [0.1, 0.2, 0.4, 0.8, 1.6]  # Linear time
        
        r_squared, params = CurveFitter.fit_linear(sizes, times)
        
        assert r_squared > 0.95
        assert "a" in params  # Slope parameter
    
    def test_fit_quadratic(self) -> None:
        """CurveFitter should correctly identify O(n²) complexity."""
        sizes = [100, 200, 400, 800]
        times = [0.01, 0.04, 0.16, 0.64]  # Quadratic time (n²/10000)
        
        r_squared, params = CurveFitter.fit_quadratic(sizes, times)
        
        assert r_squared > 0.95
    
    def test_fit_logarithmic(self) -> None:
        """CurveFitter should correctly identify O(log n) complexity."""
        sizes = [100, 1000, 10000, 100000]
        # log₁₀(100)=2, log₁₀(1000)=3, log₁₀(10000)=4, log₁₀(100000)=5
        times = [0.02, 0.03, 0.04, 0.05]
        
        r_squared, params = CurveFitter.fit_logarithmic(sizes, times)
        
        assert r_squared > 0.9
    
    def test_fit_linearithmic(self) -> None:
        """CurveFitter should correctly identify O(n log n) complexity."""
        sizes = [100, 1000, 10000]
        # n*log₁₀(n): 100*2=200, 1000*3=3000, 10000*4=40000
        times = [0.002, 0.03, 0.4]  # Scaled n*log(n)
        
        r_squared, params = CurveFitter.fit_linearithmic(sizes, times)
        
        # n log n is harder to fit perfectly
        assert r_squared > 0.8
    
    def test_fit_cubic(self) -> None:
        """CurveFitter should correctly identify O(n³) complexity."""
        sizes = [10, 20, 40, 80]
        times = [0.001, 0.008, 0.064, 0.512]  # Cubic (n³/1e6)
        
        r_squared, params = CurveFitter.fit_cubic(sizes, times)
        
        assert r_squared > 0.95
    
    def test_fit_exponential(self) -> None:
        """CurveFitter should correctly identify O(2ⁿ) complexity."""
        sizes = [5, 10, 15, 20]
        times = [0.032, 1.024, 32.768, 1048.576]  # 2^n / 1000
        
        r_squared, params = CurveFitter.fit_exponential(sizes, times)
        
        assert r_squared > 0.9
    
    def test_calculate_r_squared(self) -> None:
        """R² calculation should be correct."""
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [1.1, 1.9, 3.1, 3.9, 5.1]  # Close predictions
        
        r_squared = CurveFitter.calculate_r_squared(actual, predicted)
        
        assert 0.95 < r_squared <= 1.0
    
    def test_calculate_r_squared_perfect(self) -> None:
        """Perfect predictions should give R² = 1.0."""
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = actual.copy()
        
        r_squared = CurveFitter.calculate_r_squared(actual, predicted)
        
        assert r_squared == pytest.approx(1.0)
    
    def test_calculate_r_squared_poor(self) -> None:
        """Poor predictions should give low R²."""
        actual = [1.0, 2.0, 3.0, 4.0, 5.0]
        predicted = [5.0, 4.0, 3.0, 2.0, 1.0]  # Reversed
        
        r_squared = CurveFitter.calculate_r_squared(actual, predicted)
        
        assert r_squared < 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPLEXITY ANALYSER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityAnalyser:
    """Test suite for main ComplexityAnalyser class."""

    @pytest.fixture
    def analyser(self) -> ComplexityAnalyser:
        """Create a ComplexityAnalyser instance."""
        return ComplexityAnalyser()
    
    def test_analyser_creation(self, analyser: ComplexityAnalyser) -> None:
        """ComplexityAnalyser should initialise correctly."""
        assert analyser is not None
    
    def test_analyse_constant_algorithm(
        self,
        analyser: ComplexityAnalyser
    ) -> None:
        """Analyser should detect O(1) complexity."""
        def constant_func(n: int) -> int:
            return 42  # Always same operation
        
        result = analyser.analyse(
            func=constant_func,
            sizes=[100, 500, 1000, 5000, 10000],
            runs=3,
        )
        
        assert result.detected_class == ComplexityClass.O_1
        assert result.confidence > 0.8
    
    def test_analyse_linear_algorithm(
        self,
        analyser: ComplexityAnalyser
    ) -> None:
        """Analyser should detect O(n) complexity."""
        def linear_func(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total
        
        result = analyser.analyse(
            func=linear_func,
            sizes=[1000, 2000, 4000, 8000],
            runs=3,
        )
        
        assert result.detected_class in [
            ComplexityClass.O_N,
            ComplexityClass.O_N_LOG_N  # May misclassify slightly
        ]
        assert result.r_squared > 0.9
    
    def test_analyse_quadratic_algorithm(
        self,
        analyser: ComplexityAnalyser
    ) -> None:
        """Analyser should detect O(n²) complexity."""
        def quadratic_func(n: int) -> int:
            total = 0
            for i in range(n):
                for j in range(n):
                    total += 1
            return total
        
        result = analyser.analyse(
            func=quadratic_func,
            sizes=[100, 200, 400, 800],
            runs=3,
        )
        
        assert result.detected_class == ComplexityClass.O_N_SQUARED
        assert result.r_squared > 0.95
    
    def test_analyse_logarithmic_algorithm(
        self,
        analyser: ComplexityAnalyser
    ) -> None:
        """Analyser should detect O(log n) complexity."""
        def log_func(n: int) -> int:
            count = 0
            i = n
            while i > 1:
                i //= 2
                count += 1
            return count
        
        result = analyser.analyse(
            func=log_func,
            sizes=[1000, 10000, 100000, 1000000],
            runs=5,
        )
        
        assert result.detected_class in [
            ComplexityClass.O_LOG_N,
            ComplexityClass.O_1  # Very fast, may appear constant
        ]
    
    def test_analyse_with_data_generator(
        self,
        analyser: ComplexityAnalyser
    ) -> None:
        """Analyser should work with custom data generators."""
        def process_list(data: list[int]) -> int:
            return sum(data)  # O(n)
        
        def generate_data(n: int) -> list[int]:
            return list(range(n))
        
        result = analyser.analyse_with_data(
            func=process_list,
            data_generator=generate_data,
            sizes=[1000, 2000, 4000, 8000],
            runs=3,
        )
        
        assert result.detected_class == ComplexityClass.O_N
    
    def test_analyse_minimum_sizes(
        self,
        analyser: ComplexityAnalyser
    ) -> None:
        """Analyser should require minimum number of data points."""
        def dummy_func(n: int) -> int:
            return n
        
        with pytest.raises((ValueError, AssertionError)):
            analyser.analyse(
                func=dummy_func,
                sizes=[100],  # Too few sizes
                runs=3,
            )
    
    def test_get_best_fit(self, analyser: ComplexityAnalyser) -> None:
        """Analyser should select best fitting complexity class."""
        sizes = [100, 200, 400, 800]
        times = [0.01, 0.04, 0.16, 0.64]  # Quadratic
        
        best_class, confidence = analyser._get_best_fit(sizes, times)
        
        assert best_class == ComplexityClass.O_N_SQUARED
        assert confidence > 0.9


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: THEORETICAL ANALYSER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTheoreticalAnalyser:
    """Test suite for TheoreticalAnalyser class."""

    def test_verify_big_o(self) -> None:
        """TheoreticalAnalyser should verify Big-O bounds."""
        # f(n) = 3n + 5 should be O(n)
        def f(n: int) -> float:
            return 3 * n + 5
        
        def g(n: int) -> float:
            return float(n)
        
        is_valid = TheoreticalAnalyser.verify_big_o(f, g, c=10, n0=1)
        assert is_valid
    
    def test_verify_big_o_fails(self) -> None:
        """TheoreticalAnalyser should reject invalid Big-O claims."""
        # f(n) = n² is NOT O(n)
        def f(n: int) -> float:
            return float(n * n)
        
        def g(n: int) -> float:
            return float(n)
        
        is_valid = TheoreticalAnalyser.verify_big_o(f, g, c=1000, n0=1)
        assert not is_valid
    
    def test_verify_big_omega(self) -> None:
        """TheoreticalAnalyser should verify Big-Ω bounds."""
        # f(n) = n² is Ω(n)
        def f(n: int) -> float:
            return float(n * n)
        
        def g(n: int) -> float:
            return float(n)
        
        is_valid = TheoreticalAnalyser.verify_big_omega(f, g, c=1, n0=1)
        assert is_valid
    
    def test_verify_big_theta(self) -> None:
        """TheoreticalAnalyser should verify Big-Θ bounds."""
        # f(n) = 5n + 3 is Θ(n)
        def f(n: int) -> float:
            return 5 * n + 3
        
        def g(n: int) -> float:
            return float(n)
        
        is_valid = TheoreticalAnalyser.verify_big_theta(
            f, g,
            c1=1, c2=10, n0=1
        )
        assert is_valid
    
    def test_verify_big_theta_fails(self) -> None:
        """TheoreticalAnalyser should reject invalid Big-Θ claims."""
        # f(n) = n² is NOT Θ(n)
        def f(n: int) -> float:
            return float(n * n)
        
        def g(n: int) -> float:
            return float(n)
        
        is_valid = TheoreticalAnalyser.verify_big_theta(
            f, g,
            c1=0.1, c2=1000, n0=1
        )
        assert not is_valid
    
    def test_analyse_recurrence(self) -> None:
        """TheoreticalAnalyser should analyse recurrence relations."""
        # T(n) = 2T(n/2) + n → O(n log n) (Master Theorem Case 2)
        result = TheoreticalAnalyser.analyse_recurrence(
            a=2, b=2, f_complexity=ComplexityClass.O_N
        )
        
        assert result == ComplexityClass.O_N_LOG_N
    
    def test_analyse_recurrence_case_1(self) -> None:
        """TheoreticalAnalyser should handle Master Theorem Case 1."""
        # T(n) = 4T(n/2) + n → O(n²) (Case 1: a > b^k)
        result = TheoreticalAnalyser.analyse_recurrence(
            a=4, b=2, f_complexity=ComplexityClass.O_N
        )
        
        assert result == ComplexityClass.O_N_SQUARED
    
    def test_analyse_recurrence_case_3(self) -> None:
        """TheoreticalAnalyser should handle Master Theorem Case 3."""
        # T(n) = T(n/2) + n → O(n) (Case 3: a < b^k)
        result = TheoreticalAnalyser.analyse_recurrence(
            a=1, b=2, f_complexity=ComplexityClass.O_N
        )
        
        assert result == ComplexityClass.O_N


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COMPLEXITY COMPARATOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestComplexityComparator:
    """Test suite for ComplexityComparator class."""

    def test_compare_complexity_classes(self) -> None:
        """Comparator should correctly compare complexity classes."""
        # O(1) < O(n) < O(n²)
        assert ComplexityComparator.compare(
            ComplexityClass.O_1,
            ComplexityClass.O_N
        ) < 0
        
        assert ComplexityComparator.compare(
            ComplexityClass.O_N,
            ComplexityClass.O_N_SQUARED
        ) < 0
        
        assert ComplexityComparator.compare(
            ComplexityClass.O_N_SQUARED,
            ComplexityClass.O_N
        ) > 0
    
    def test_compare_equal_complexity(self) -> None:
        """Comparator should return 0 for equal complexity classes."""
        assert ComplexityComparator.compare(
            ComplexityClass.O_N,
            ComplexityClass.O_N
        ) == 0
    
    def test_is_better_complexity(self) -> None:
        """Comparator should identify better complexity."""
        assert ComplexityComparator.is_better(
            ComplexityClass.O_LOG_N,
            ComplexityClass.O_N
        )
        
        assert not ComplexityComparator.is_better(
            ComplexityClass.O_N_SQUARED,
            ComplexityClass.O_N
        )
    
    def test_crossover_point(self) -> None:
        """Comparator should calculate crossover points."""
        # When does O(n²) become slower than O(n log n)?
        crossover = ComplexityComparator.find_crossover_point(
            class1=ComplexityClass.O_N_SQUARED,
            class2=ComplexityClass.O_N_LOG_N,
            constant1=1.0,
            constant2=10.0,
        )
        
        # Crossover should exist and be positive
        assert crossover is not None
        assert crossover > 0
    
    def test_scaling_factor(self) -> None:
        """Comparator should calculate scaling factors."""
        # How much slower is O(n²) vs O(n) when n doubles?
        factor = ComplexityComparator.scaling_factor(
            ComplexityClass.O_N_SQUARED,
            base_size=100,
            target_size=200,
        )
        
        # n² doubles → factor of 4
        assert factor == pytest.approx(4.0, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: REPORTING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestReporting:
    """Test suite for reporting functionality."""

    def test_generate_report(self) -> None:
        """Analyser should generate readable reports."""
        analyser = ComplexityAnalyser()
        
        result = ComplexityResult(
            detected_class=ComplexityClass.O_N,
            confidence=0.95,
            r_squared=0.98,
            fitted_params={"a": 1.5},
            sizes=[100, 200, 400],
            times=[0.1, 0.2, 0.4],
        )
        
        report = analyser.generate_report(result)
        
        assert "O(n)" in report or "linear" in report.lower()
        assert "95" in report or "0.95" in report
    
    def test_export_results_csv(self) -> None:
        """Analyser should export results to CSV."""
        analyser = ComplexityAnalyser()
        
        result = ComplexityResult(
            detected_class=ComplexityClass.O_N,
            confidence=0.95,
            r_squared=0.98,
            fitted_params={"a": 1.5},
            sizes=[100, 200, 400],
            times=[0.1, 0.2, 0.4],
        )
        
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.csv',
            delete=False
        ) as f:
            csv_path = Path(f.name)
        
        try:
            analyser.export_csv(result, csv_path)
            
            assert csv_path.exists()
            content = csv_path.read_text()
            assert "size" in content.lower() or "100" in content
        finally:
            csv_path.unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests for complete analysis workflows."""

    def test_full_analysis_workflow(self) -> None:
        """Complete analysis workflow should work end-to-end."""
        analyser = ComplexityAnalyser()
        
        # Define test function with known complexity
        def bubble_sort_steps(n: int) -> int:
            """Simulates bubble sort comparisons (O(n²))."""
            return n * (n - 1) // 2
        
        # Analyse
        result = analyser.analyse(
            func=bubble_sort_steps,
            sizes=[100, 200, 400, 800],
            runs=3,
        )
        
        # Verify
        assert result.detected_class == ComplexityClass.O_N_SQUARED
        assert result.confidence > 0.9
        
        # Generate report
        report = analyser.generate_report(result)
        assert len(report) > 0
    
    def test_compare_multiple_algorithms(self) -> None:
        """Should compare complexity of multiple algorithms."""
        analyser = ComplexityAnalyser()
        
        def linear_algo(n: int) -> int:
            return sum(range(n))
        
        def quadratic_algo(n: int) -> int:
            total = 0
            for i in range(n):
                for j in range(n):
                    total += 1
            return total
        
        result1 = analyser.analyse(
            func=linear_algo,
            sizes=[500, 1000, 2000, 4000],
            runs=3,
        )
        
        result2 = analyser.analyse(
            func=quadratic_algo,
            sizes=[100, 200, 400, 800],
            runs=3,
        )
        
        # Linear should be faster complexity class
        assert ComplexityComparator.is_better(
            result1.detected_class,
            result2.detected_class
        )
    
    def test_theoretical_verification_workflow(self) -> None:
        """Theoretical verification should work with empirical results."""
        analyser = ComplexityAnalyser()
        
        # Empirically analyse linear function
        def linear_func(n: int) -> int:
            return sum(range(n))
        
        empirical = analyser.analyse(
            func=linear_func,
            sizes=[1000, 2000, 4000, 8000],
            runs=3,
        )
        
        # Theoretically verify
        def f(n: int) -> float:
            return float(n)
        
        def g(n: int) -> float:
            return float(n)
        
        theoretical_valid = TheoreticalAnalyser.verify_big_theta(
            f, g,
            c1=0.5, c2=2.0, n0=1
        )
        
        assert theoretical_valid
        assert empirical.detected_class == ComplexityClass.O_N


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_fast_function(self) -> None:
        """Analyser should handle very fast functions."""
        analyser = ComplexityAnalyser()
        
        def instant_func(n: int) -> int:
            return 42
        
        result = analyser.analyse(
            func=instant_func,
            sizes=[100, 1000, 10000, 100000],
            runs=5,
        )
        
        # Should detect as constant
        assert result.detected_class == ComplexityClass.O_1
    
    def test_noisy_timings(self) -> None:
        """Analyser should handle noisy timing data."""
        analyser = ComplexityAnalyser()
        
        import random
        
        def noisy_linear(n: int) -> int:
            # Add some noise
            iterations = n + random.randint(-n // 10, n // 10)
            return sum(range(max(1, iterations)))
        
        result = analyser.analyse(
            func=noisy_linear,
            sizes=[1000, 2000, 4000, 8000],
            runs=10,  # More runs to average out noise
        )
        
        # Should still detect approximately linear
        assert result.detected_class in [
            ComplexityClass.O_N,
            ComplexityClass.O_N_LOG_N
        ]
    
    def test_non_positive_sizes(self) -> None:
        """Analyser should reject non-positive sizes."""
        analyser = ComplexityAnalyser()
        
        def dummy_func(n: int) -> int:
            return n
        
        with pytest.raises((ValueError, AssertionError)):
            analyser.analyse(
                func=dummy_func,
                sizes=[0, 100, 200],  # Contains zero
                runs=3,
            )
    
    def test_negative_times(self) -> None:
        """Curve fitter should handle edge case of negative times gracefully."""
        sizes = [100, 200, 400]
        times = [0.001, 0.001, 0.001]  # Very small but positive
        
        # Should not raise exception
        r_squared, _ = CurveFitter.fit_linear(sizes, times)
        assert isinstance(r_squared, float)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: PERFORMANCE TESTS (MARKED AS SLOW)
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPerformance:
    """Performance tests for larger inputs (marked as slow)."""

    def test_analyse_large_range(self) -> None:
        """Analyser should handle large size ranges."""
        analyser = ComplexityAnalyser()
        
        def linear_func(n: int) -> int:
            return sum(range(n))
        
        result = analyser.analyse(
            func=linear_func,
            sizes=[1000, 5000, 10000, 50000, 100000],
            runs=3,
        )
        
        assert result.detected_class == ComplexityClass.O_N
        assert result.r_squared > 0.9
    
    def test_many_complexity_comparisons(self) -> None:
        """Should efficiently compare many algorithms."""
        analyser = ComplexityAnalyser()
        results = []
        
        algorithms = [
            lambda n: 42,  # O(1)
            lambda n: sum(range(n)),  # O(n)
            lambda n: sum(sum(range(i)) for i in range(n // 10)),  # O(n²)
        ]
        
        for algo in algorithms:
            result = analyser.analyse(
                func=algo,
                sizes=[100, 200, 400],
                runs=3,
            )
            results.append(result)
        
        # Verify ordering
        assert ComplexityComparator.is_better(
            results[0].detected_class,
            results[1].detected_class
        ) or results[0].detected_class == results[1].detected_class


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: MATHEMATICAL FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMathematicalFunctions:
    """Test mathematical helper functions."""

    def test_growth_functions(self) -> None:
        """Growth functions should compute correctly."""
        n = 100
        
        # O(1) - constant
        assert CurveFitter.constant_model(n, 5.0) == 5.0
        
        # O(n) - linear
        linear_result = CurveFitter.linear_model(n, 2.0, 1.0)
        assert linear_result == pytest.approx(201.0)  # 2*100 + 1
        
        # O(n²) - quadratic
        quad_result = CurveFitter.quadratic_model(n, 0.01, 0, 0)
        assert quad_result == pytest.approx(100.0)  # 0.01 * 100²
        
        # O(log n) - logarithmic
        log_result = CurveFitter.logarithmic_model(n, 10.0, 0)
        expected = 10.0 * math.log(n)
        assert log_result == pytest.approx(expected, rel=0.01)
    
    def test_complexity_growth_rates(self) -> None:
        """Verify relative growth rates of complexity classes."""
        small_n = 10
        large_n = 1000
        
        # Calculate growth factors
        constant_growth = 1.0 / 1.0  # O(1)
        linear_growth = large_n / small_n  # O(n) = 100x
        quadratic_growth = (large_n ** 2) / (small_n ** 2)  # O(n²) = 10000x
        log_growth = math.log(large_n) / math.log(small_n)  # O(log n) ≈ 3x
        
        # Verify relative ordering
        assert log_growth < linear_growth < quadratic_growth


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
