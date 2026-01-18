"""
Tests for 14UNIT Lab 02: Dask and Profiling.
Run with: pytest tests/test_lab_02.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

dask = pytest.importorskip("dask")

from lab.lab_14_02_dask_profiling import (
    sequential_pipeline,
    delayed_pipeline,
    create_dask_array,
    dask_array_operations,
    profile_function,
    ProfileResult,
)


class TestDaskDelayed:
    def test_sequential_pipeline(self):
        result = sequential_pipeline([1.0, 2.0, 3.0])
        assert result == 14.0  # 1 + 4 + 9
    
    def test_delayed_pipeline_computes(self):
        computation = delayed_pipeline([1.0, 2.0, 3.0])
        result = computation.compute()
        assert result == 14.0


class TestDaskArrays:
    def test_create_array(self):
        arr = create_dask_array((100, 100), chunks=(50, 50))
        assert arr.shape == (100, 100)
    
    def test_array_operations(self):
        arr = create_dask_array((100, 100), chunks=(50, 50))
        stats = dask_array_operations(arr)
        assert 'mean' in stats
        assert 0 < stats['mean'] < 1


class TestProfiling:
    def test_profile_execution(self):
        def sample(n):
            return sum(range(n))
        
        result, profile = profile_function(sample, 1000)
        assert result == sum(range(1000))
        assert isinstance(profile, ProfileResult)
        assert profile.total_time > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
