"""
Exercise: Text Normalisation (Medium)

Build a text preprocessing pipeline.

Duration: 20-25 minutes
Difficulty: ★★★☆☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations
import re
import unicodedata


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    # TODO: Implement
    pass


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text."""
    # TODO: Implement
    pass


def remove_numbers(text: str) -> str:
    """Remove all numeric characters."""
    # TODO: Implement
    pass


def normalise_unicode(text: str, form: str = "NFC") -> str:
    """Apply Unicode normalisation."""
    # TODO: Implement
    pass


def remove_accents(text: str) -> str:
    """Remove diacritical marks from characters."""
    # TODO: Implement using NFKD and combining check
    pass


def remove_stopwords(tokens: list[str], stopwords: set[str]) -> list[str]:
    """Remove stopwords from token list."""
    # TODO: Implement
    pass


def create_pipeline(*functions) -> callable:
    """
    Create a text processing pipeline from functions.
    
    Each function takes a string and returns a string.
    """
    def pipeline(text: str) -> str:
        result = text
        for func in functions:
            result = func(result)
        return result
    return pipeline


def run_tests() -> None:
    assert lowercase_text("HELLO") == "hello"
    assert "," not in remove_punctuation("Hello, World!")
    assert remove_accents("café") == "cafe"
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
