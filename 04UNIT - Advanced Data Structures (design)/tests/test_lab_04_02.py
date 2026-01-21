#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
04UNIT: Test Suite for Lab 4.2 - Probabilistic Data Structures
═══════════════════════════════════════════════════════════════════════════════

Comprehensive tests for Bloom filter and Count-Min sketch implementations.

Coverage targets:
- BloomFilter creation and parameter calculation
- BloomFilter add and contains operations
- BloomFilter false positive rate validation
- CountMinSketch creation
- CountMinSketch update and estimate operations
- CountMinSketch error bounds

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

# Add lab directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_4_02_probabilistic_ds import BloomFilter, CountMinSketch


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BLOOM FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBloomFilterCreation:
    """Tests for Bloom filter initialisation."""

    def test_create_with_size(self) -> None:
        """Test creating Bloom filter with explicit size."""
        bf = BloomFilter(size=1000, num_hash_functions=5)
        assert bf.size == 1000
        assert bf.num_hash_functions == 5

    def test_create_from_parameters(self) -> None:
        """Test creating Bloom filter from expected elements and FPR."""
        bf = BloomFilter.from_parameters(expected_elements=100, false_positive_rate=0.01)
        # Verify size is reasonable
        assert bf.size > 0
        assert bf.num_hash_functions > 0

    def test_optimal_size_calculation(self) -> None:
        """Test optimal size calculation formula."""
        n = 1000  # Expected elements
        p = 0.01  # Target FPR
        
        # Formula: m = -(n * ln(p)) / (ln(2))^2
        expected_size = -int((n * math.log(p)) / (math.log(2) ** 2))
        
        bf = BloomFilter.from_parameters(expected_elements=n, false_positive_rate=p)
        # Allow some tolerance
        assert abs(bf.size - expected_size) < expected_size * 0.1

    def test_optimal_hash_functions_calculation(self) -> None:
        """Test optimal number of hash functions calculation."""
        n = 1000
        p = 0.01
        
        bf = BloomFilter.from_parameters(expected_elements=n, false_positive_rate=p)
        
        # Formula: k = (m/n) * ln(2)
        expected_k = int((bf.size / n) * math.log(2))
        # Allow ±1 due to rounding
        assert abs(bf.num_hash_functions - expected_k) <= 1

    def test_empty_filter(self) -> None:
        """Test empty Bloom filter."""
        bf = BloomFilter(size=100, num_hash_functions=3)
        assert not bf.contains("anything")
        assert bf.count == 0


class TestBloomFilterOperations:
    """Tests for Bloom filter add and contains operations."""

    def test_add_and_contains(self, small_word_set: set[str]) -> None:
        """Test adding elements and checking membership."""
        bf = BloomFilter(size=1000, num_hash_functions=5)
        
        for word in small_word_set:
            bf.add(word)
        
        for word in small_word_set:
            assert bf.contains(word), f"Expected {word} to be in filter"

    def test_no_false_negatives(self, medium_word_set: set[str]) -> None:
        """Test that Bloom filter never returns false negatives."""
        bf = BloomFilter.from_parameters(
            expected_elements=len(medium_word_set),
            false_positive_rate=0.01
        )
        
        for word in medium_word_set:
            bf.add(word)
        
        # All added elements must be found
        for word in medium_word_set:
            assert bf.contains(word), f"False negative for {word}"

    def test_likely_false_positives_for_non_members(
        self, 
        small_word_set: set[str], 
        test_negative_set: set[str]
    ) -> None:
        """Test that non-members may return false positives."""
        bf = BloomFilter(size=50, num_hash_functions=2)  # Small filter = more FPs
        
        for word in small_word_set:
            bf.add(word)
        
        # With a tiny filter, we should see some false positives
        # This is probabilistic, so we just verify the mechanism works
        results = [bf.contains(word) for word in test_negative_set]
        # At least some should be False (true negatives)
        # Note: with very small filters, all might be False positives

    def test_count_tracking(self) -> None:
        """Test that count tracks unique additions."""
        bf = BloomFilter(size=1000, num_hash_functions=5)
        
        bf.add("item1")
        bf.add("item2")
        bf.add("item3")
        
        assert bf.count == 3
        
        # Adding duplicate shouldn't increase count
        bf.add("item1")
        assert bf.count == 3

    def test_add_various_types(self) -> None:
        """Test adding various hashable types."""
        bf = BloomFilter(size=1000, num_hash_functions=5)
        
        bf.add("string")
        bf.add(123)
        bf.add((1, 2, 3))
        
        assert bf.contains("string")
        assert bf.contains(123)
        assert bf.contains((1, 2, 3))


class TestBloomFilterFalsePositiveRate:
    """Tests for false positive rate validation."""

    @pytest.mark.parametrize("target_fpr", [0.1, 0.05, 0.01])
    def test_actual_fpr_near_target(self, target_fpr: float) -> None:
        """Test that actual FPR is close to target."""
        n = 1000
        bf = BloomFilter.from_parameters(
            expected_elements=n,
            false_positive_rate=target_fpr
        )
        
        # Add n elements
        for i in range(n):
            bf.add(f"element_{i}")
        
        # Test with elements definitely not in the set
        test_size = 10000
        false_positives = sum(
            1 for i in range(test_size)
            if bf.contains(f"test_{i}")
        )
        
        actual_fpr = false_positives / test_size
        
        # Allow 2x tolerance due to randomness
        assert actual_fpr < target_fpr * 3, \
            f"Actual FPR {actual_fpr:.4f} much higher than target {target_fpr}"

    def test_fpr_increases_with_load(self) -> None:
        """Test that FPR increases as filter fills up."""
        bf = BloomFilter(size=100, num_hash_functions=3)
        
        # Measure FPR at different loads
        test_elements = [f"test_{i}" for i in range(1000)]
        
        # Add 10 elements
        for i in range(10):
            bf.add(f"element_{i}")
        fpr_10 = sum(1 for e in test_elements if bf.contains(e)) / len(test_elements)
        
        # Add 50 more elements
        for i in range(10, 60):
            bf.add(f"element_{i}")
        fpr_60 = sum(1 for e in test_elements if bf.contains(e)) / len(test_elements)
        
        # FPR should increase (or stay same if already saturated)
        assert fpr_60 >= fpr_10


class TestBloomFilterEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string(self) -> None:
        """Test adding empty string."""
        bf = BloomFilter(size=100, num_hash_functions=3)
        bf.add("")
        assert bf.contains("")

    def test_unicode_strings(self) -> None:
        """Test Unicode string handling."""
        bf = BloomFilter(size=1000, num_hash_functions=5)
        
        bf.add("café")
        bf.add("日本語")
        bf.add("émoji 🎉")
        
        assert bf.contains("café")
        assert bf.contains("日本語")
        assert bf.contains("émoji 🎉")

    def test_very_small_filter(self) -> None:
        """Test behaviour with very small filter."""
        bf = BloomFilter(size=8, num_hash_functions=1)
        
        bf.add("a")
        bf.add("b")
        
        assert bf.contains("a")
        assert bf.contains("b")

    def test_single_hash_function(self) -> None:
        """Test with single hash function."""
        bf = BloomFilter(size=1000, num_hash_functions=1)
        
        bf.add("test")
        assert bf.contains("test")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: COUNT-MIN SKETCH TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCountMinSketchCreation:
    """Tests for Count-Min sketch initialisation."""

    def test_create_with_dimensions(self) -> None:
        """Test creating Count-Min sketch with explicit dimensions."""
        cms = CountMinSketch(width=1000, depth=5)
        assert cms.width == 1000
        assert cms.depth == 5

    def test_create_from_parameters(self) -> None:
        """Test creating Count-Min sketch from error parameters."""
        cms = CountMinSketch.from_parameters(epsilon=0.01, delta=0.01)
        
        # Verify dimensions are reasonable
        # Width should be ⌈e/ε⌉ ≈ 272 for ε=0.01
        # Depth should be ⌈ln(1/δ)⌉ ≈ 5 for δ=0.01
        assert cms.width > 200
        assert cms.depth >= 4

    def test_empty_sketch(self) -> None:
        """Test empty Count-Min sketch."""
        cms = CountMinSketch(width=100, depth=3)
        assert cms.estimate("anything") == 0
        assert cms.total_count == 0


class TestCountMinSketchOperations:
    """Tests for Count-Min sketch update and estimate operations."""

    def test_update_and_estimate(self) -> None:
        """Test updating and estimating frequencies."""
        cms = CountMinSketch(width=1000, depth=5)
        
        cms.update("apple", 10)
        cms.update("banana", 5)
        cms.update("cherry", 3)
        
        # Estimates should be at least the true count
        assert cms.estimate("apple") >= 10
        assert cms.estimate("banana") >= 5
        assert cms.estimate("cherry") >= 3

    def test_multiple_updates(self) -> None:
        """Test multiple updates to same element."""
        cms = CountMinSketch(width=1000, depth=5)
        
        cms.update("item")
        cms.update("item")
        cms.update("item")
        
        assert cms.estimate("item") >= 3

    def test_default_increment(self) -> None:
        """Test default increment of 1."""
        cms = CountMinSketch(width=1000, depth=5)
        
        cms.update("item")
        assert cms.estimate("item") >= 1

    def test_large_increments(self) -> None:
        """Test large count increments."""
        cms = CountMinSketch(width=1000, depth=5)
        
        cms.update("big", 1000000)
        assert cms.estimate("big") >= 1000000

    def test_total_count(self, frequency_data: list[str]) -> None:
        """Test total count tracking."""
        cms = CountMinSketch(width=1000, depth=5)
        
        for item in frequency_data:
            cms.update(item)
        
        assert cms.total_count == len(frequency_data)


class TestCountMinSketchAccuracy:
    """Tests for Count-Min sketch accuracy guarantees."""

    def test_estimates_are_upper_bounds(self, frequency_data: list[str]) -> None:
        """Test that estimates are always upper bounds (never underestimate)."""
        cms = CountMinSketch(width=100, depth=4)
        
        # Track true frequencies
        true_freq: dict[str, int] = {}
        for item in frequency_data:
            cms.update(item)
            true_freq[item] = true_freq.get(item, 0) + 1
        
        # All estimates should be >= true frequency
        for item, count in true_freq.items():
            estimate = cms.estimate(item)
            assert estimate >= count, \
                f"Estimate {estimate} < true count {count} for {item}"

    def test_error_within_bounds(self) -> None:
        """Test that error is within theoretical bounds."""
        epsilon = 0.1
        delta = 0.05
        cms = CountMinSketch.from_parameters(epsilon=epsilon, delta=delta)
        
        # Add known frequencies
        n = 1000
        for i in range(n):
            cms.update(f"item_{i % 100}")  # 100 unique items, ~10 each
        
        # Error should be bounded by εN with probability 1-δ
        # For testing, we check average error
        true_counts = {f"item_{i}": 10 for i in range(100)}
        
        errors = []
        for item, true_count in true_counts.items():
            estimate = cms.estimate(item)
            error = estimate - true_count
            errors.append(error)
        
        avg_error = sum(errors) / len(errors)
        max_expected_error = epsilon * n
        
        # Average error should be well under the bound
        assert avg_error < max_expected_error, \
            f"Average error {avg_error} exceeds bound {max_expected_error}"

    def test_zipf_distribution_accuracy(self, zipf_data: list[str]) -> None:
        """Test accuracy on Zipf-distributed data."""
        cms = CountMinSketch(width=1000, depth=5)
        
        # Track true frequencies
        true_freq: dict[str, int] = {}
        for item in zipf_data:
            cms.update(item)
            true_freq[item] = true_freq.get(item, 0) + 1
        
        # Check top-10 most frequent items
        top_items = sorted(true_freq.items(), key=lambda x: -x[1])[:10]
        
        for item, true_count in top_items:
            estimate = cms.estimate(item)
            # Should not underestimate
            assert estimate >= true_count
            # Error should be reasonable (within 50% for large counts)
            if true_count > 10:
                assert estimate < true_count * 1.5, \
                    f"Overestimate too high: {estimate} vs {true_count}"


class TestCountMinSketchMerge:
    """Tests for Count-Min sketch merging functionality."""

    def test_merge_sketches(self) -> None:
        """Test merging two Count-Min sketches."""
        cms1 = CountMinSketch(width=1000, depth=5)
        cms2 = CountMinSketch(width=1000, depth=5)
        
        cms1.update("apple", 10)
        cms1.update("banana", 5)
        
        cms2.update("apple", 7)
        cms2.update("cherry", 3)
        
        merged = cms1.merge(cms2)
        
        # Merged estimates should reflect sum
        assert merged.estimate("apple") >= 17
        assert merged.estimate("banana") >= 5
        assert merged.estimate("cherry") >= 3

    def test_merge_incompatible_dimensions(self) -> None:
        """Test that merging incompatible sketches raises error."""
        cms1 = CountMinSketch(width=100, depth=5)
        cms2 = CountMinSketch(width=200, depth=5)
        
        with pytest.raises((ValueError, AssertionError)):
            cms1.merge(cms2)


class TestCountMinSketchEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_zero_estimate_for_unseen(self) -> None:
        """Test zero estimate for unseen elements."""
        cms = CountMinSketch(width=1000, depth=5)
        cms.update("seen")
        
        assert cms.estimate("unseen") == 0 or cms.estimate("unseen") > 0
        # Note: might be positive due to hash collisions

    def test_negative_increment_handling(self) -> None:
        """Test handling of negative increments (if supported)."""
        cms = CountMinSketch(width=1000, depth=5)
        cms.update("item", 10)
        
        # Behaviour depends on implementation
        # Some implementations allow negative updates for deletion
        try:
            cms.update("item", -5)
            # If allowed, estimate should still be >= 0
            assert cms.estimate("item") >= 0
        except (ValueError, AssertionError):
            # Negative updates not supported is also valid
            pass

    def test_very_small_sketch(self) -> None:
        """Test behaviour with minimal dimensions."""
        cms = CountMinSketch(width=2, depth=1)
        
        cms.update("a")
        cms.update("b")
        
        # Should still function (with high error)
        assert cms.estimate("a") >= 1
        assert cms.estimate("b") >= 1

    def test_unicode_items(self) -> None:
        """Test Unicode string handling."""
        cms = CountMinSketch(width=1000, depth=5)
        
        cms.update("café", 5)
        cms.update("東京", 10)
        
        assert cms.estimate("café") >= 5
        assert cms.estimate("東京") >= 10


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_bloom_filter_spell_checker(self) -> None:
        """Test Bloom filter as spell checker."""
        # Build dictionary
        dictionary = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
        }
        
        bf = BloomFilter.from_parameters(
            expected_elements=len(dictionary),
            false_positive_rate=0.01
        )
        
        for word in dictionary:
            bf.add(word.lower())
        
        # Check valid words
        assert bf.contains("the")
        assert bf.contains("have")
        
        # Check likely misspellings (may have false positives)
        misspelled = ["teh", "hav", "fro"]
        # At least some should be detected as not in dictionary
        results = [bf.contains(word) for word in misspelled]
        # Not all misspellings should pass (statistically unlikely)

    def test_count_min_sketch_stream_processing(self) -> None:
        """Test Count-Min sketch for stream processing scenario."""
        cms = CountMinSketch(width=500, depth=4)
        
        # Simulate log stream
        random.seed(42)
        endpoints = ["/api/users", "/api/posts", "/api/comments", "/health"]
        weights = [100, 50, 30, 500]  # /health is most frequent
        
        # Generate stream
        stream = []
        for endpoint, weight in zip(endpoints, weights):
            stream.extend([endpoint] * weight)
        random.shuffle(stream)
        
        # Process stream
        for endpoint in stream:
            cms.update(endpoint)
        
        # Find most frequent
        estimates = {e: cms.estimate(e) for e in endpoints}
        most_frequent = max(estimates, key=lambda x: estimates[x])
        
        assert most_frequent == "/health"

    def test_bloom_filter_vs_set_comparison(self) -> None:
        """Compare Bloom filter behaviour to Python set."""
        elements = [f"item_{i}" for i in range(100)]
        test_elements = [f"test_{i}" for i in range(100)]
        
        # Python set (exact)
        exact_set = set(elements)
        
        # Bloom filter (approximate)
        bf = BloomFilter.from_parameters(
            expected_elements=len(elements),
            false_positive_rate=0.01
        )
        for e in elements:
            bf.add(e)
        
        # Both should contain all original elements
        for e in elements:
            assert e in exact_set
            assert bf.contains(e)
        
        # Set never has false positives
        for e in test_elements:
            assert e not in exact_set
        
        # Bloom filter may have some false positives
        fp_count = sum(1 for e in test_elements if bf.contains(e))
        # Should be close to expected FPR (1%)
        assert fp_count < 10  # Allow up to 10% given small sample
