"""
Exercise: Frequency Analysis (Medium)

Analyse word frequencies in text.

Duration: 20-25 minutes
Difficulty: ★★★☆☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations
from collections import Counter
import re


def word_frequencies(text: str) -> Counter[str]:
    """Count word frequencies (case-insensitive)."""
    # TODO: Implement
    pass


def top_n_words(text: str, n: int = 10) -> list[tuple[str, int]]:
    """Get the n most frequent words."""
    # TODO: Implement
    pass


def hapax_legomena(text: str) -> list[str]:
    """Find words that appear exactly once."""
    # TODO: Implement
    pass


def type_token_ratio(text: str) -> float:
    """Calculate type-token ratio (vocabulary / total words)."""
    # TODO: Implement
    pass


def vocabulary_richness(text: str) -> dict[str, float]:
    """
    Calculate multiple vocabulary richness metrics.
    
    Returns dict with: ttr, hapax_ratio, avg_word_length
    """
    # TODO: Implement
    pass


def run_tests() -> None:
    text = "the cat sat on the mat the cat was happy"
    freq = word_frequencies(text)
    assert freq["the"] == 3
    assert freq["cat"] == 2
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
