"""
Exercise: Tokenisation Introduction (Easy)

Practice basic tokenisation techniques.

Duration: 10-15 minutes
Difficulty: ★☆☆☆☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations


def whitespace_tokenise(text: str) -> list[str]:
    """Split text on whitespace."""
    # TODO: Implement
    pass


def simple_word_tokenise(text: str) -> list[str]:
    """Extract words using regex pattern \\w+."""
    import re
    # TODO: Implement
    pass


def sentence_split(text: str) -> list[str]:
    """Split text into sentences on . ! ?"""
    import re
    # TODO: Implement
    pass


def count_tokens(text: str) -> dict[str, int]:
    """Count token frequencies."""
    from collections import Counter
    # TODO: Implement
    pass


def run_tests() -> None:
    assert whitespace_tokenise("Hello World") == ["Hello", "World"]
    assert len(simple_word_tokenise("Hello, World!")) == 2
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
