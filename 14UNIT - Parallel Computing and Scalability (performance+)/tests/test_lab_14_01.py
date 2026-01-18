"""
Tests for 14UNIT Lab 01: Multiprocessing and Threading.

Run with: pytest tests/test_lab_14_01.py -v
"""

import pytest
import math
import time
from multiprocessing import cpu_count

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lab.lab_14_01_multiprocessing import (
    cpu_intensive_task,
    parallel_map,
    estimate_pi_sequential,
    estimate_pi_parallel,
    shared_counter_lock,
)


class TestCPUIntensiveTask:
    """Tests for cpu_intensive_task function."""
    
    def test_returns_correct_sum(self):
        """Verify sum of squares calculation."""
        assert cpu_intensive_task(5) == 0 + 1 + 4 + 9 + 16
        assert cpu_intensive_task(10) == sum(i*i for i in range(10))
    
    def test_zero_input(self):
        """Handle edge case of zero."""
        assert cpu_intensive_task(0) == 0


class TestParallelMap:
    """Tests for parallel_map function."""
    
    def test_squares_list(self, small_list, n_workers):
        """Map squaring function over list."""
        result = parallel_map(lambda x: x*x, small_list, n_workers)
        expected = [x*x for x in small_list]
        assert result == expected
    
    def test_preserves_order(self, n_workers):
        """Results should be in input order."""
        data = list(range(50))
        result = parallel_map(lambda x: x * 2, data, n_workers)
        assert result == [x * 2 for x in data]


class TestMonteCarlo:
    """Tests for Monte Carlo π estimation."""
    
    def test_sequential_accuracy(self):
        """Sequential estimate should be reasonably accurate."""
        pi_est = estimate_pi_sequential(100_000)
        assert abs(pi_est - math.pi) < 0.05
    
    def test_parallel_accuracy(self, n_workers):
        """Parallel estimate should be reasonably accurate."""
        pi_est, elapsed = estimate_pi_parallel(100_000, n_workers)
        assert abs(pi_est - math.pi) < 0.05
        assert elapsed > 0


class TestSynchronisation:
    """Tests for synchronisation primitives."""
    
    def test_safe_counter_correct(self):
        """Lock-protected counter should be exact."""
        result = shared_counter_lock()
        expected = 4 * 100_000
        assert result == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
