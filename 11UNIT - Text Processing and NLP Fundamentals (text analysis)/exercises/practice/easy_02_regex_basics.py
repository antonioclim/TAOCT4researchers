"""
Exercise: Regex Basics (Easy)

Practice fundamental regular expression patterns.

Duration: 10-15 minutes
Difficulty: ★☆☆☆☆

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations
import re


def find_all_digits(text: str) -> list[str]:
    """
    Find all sequences of digits in text.
    
    Example:
        >>> find_all_digits("Order 123 and 456")
        ['123', '456']
    """
    # TODO: Implement using re.findall
    pass


def find_all_words(text: str) -> list[str]:
    """
    Find all words (sequences of word characters).
    
    Example:
        >>> find_all_words("Hello, World!")
        ['Hello', 'World']
    """
    # TODO: Implement using re.findall
    pass


def has_email(text: str) -> bool:
    """
    Check if text contains an email address.
    
    Example:
        >>> has_email("Contact: user@example.com")
        True
    """
    # TODO: Implement using re.search
    pass


def replace_digits(text: str, replacement: str = "#") -> str:
    """
    Replace all digits with a replacement character.
    
    Example:
        >>> replace_digits("Call 123-456-7890")
        'Call ###-###-####'
    """
    # TODO: Implement using re.sub
    pass


def split_on_punctuation(text: str) -> list[str]:
    """
    Split text on punctuation marks.
    
    Example:
        >>> split_on_punctuation("Hello! How are you?")
        ['Hello', ' How are you', '']
    """
    # TODO: Implement using re.split
    pass


def run_tests() -> None:
    """Run test cases."""
    assert find_all_digits("Order 123 and 456") == ["123", "456"]
    assert find_all_words("Hello, World!") == ["Hello", "World"]
    assert has_email("Contact: user@example.com") is True
    assert has_email("No email here") is False
    assert replace_digits("Call 123") == "Call ###"
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
