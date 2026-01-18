"""
Exercise: String Operations (Easy)

Practice basic string manipulation methods for text processing.

Duration: 10-15 minutes
Difficulty: ★☆☆☆☆

Learning Objectives:
- Apply string methods for case transformation
- Use split and join operations
- Perform string searching and counting

Instructions:
1. Complete each function according to its docstring
2. Run the tests to verify your solutions
3. Do not modify the function signatures

Author: Antonio Clim
Version: 1.0.0
"""

from __future__ import annotations


def normalise_whitespace(text: str) -> str:
    """
    Normalise whitespace in text.
    
    Remove leading/trailing whitespace and collapse multiple
    internal spaces to single spaces.
    
    Args:
        text: Input text with potentially irregular whitespace.
    
    Returns:
        Text with normalised whitespace.
    
    Example:
        >>> normalise_whitespace("  Hello   World  ")
        'Hello World'
    """
    # TODO: Implement this function
    pass


def count_words(text: str) -> int:
    """
    Count the number of words in text.
    
    Words are sequences of non-whitespace characters.
    
    Args:
        text: Input text.
    
    Returns:
        Number of words.
    
    Example:
        >>> count_words("The quick brown fox")
        4
    """
    # TODO: Implement this function
    pass


def reverse_words(text: str) -> str:
    """
    Reverse the order of words in text.
    
    Args:
        text: Input text.
    
    Returns:
        Text with words in reverse order.
    
    Example:
        >>> reverse_words("Hello World")
        'World Hello'
    """
    # TODO: Implement this function
    pass


def capitalise_words(text: str) -> str:
    """
    Capitalise the first letter of each word.
    
    Args:
        text: Input text.
    
    Returns:
        Text with capitalised words.
    
    Example:
        >>> capitalise_words("hello world")
        'Hello World'
    """
    # TODO: Implement this function
    pass


def find_longest_word(text: str) -> str:
    """
    Find the longest word in text.
    
    If multiple words have the same length, return the first one.
    
    Args:
        text: Input text.
    
    Returns:
        The longest word, or empty string if no words.
    
    Example:
        >>> find_longest_word("The quick brown fox")
        'quick'
    """
    # TODO: Implement this function
    pass


def replace_multiple(text: str, replacements: dict[str, str]) -> str:
    """
    Apply multiple string replacements.
    
    Args:
        text: Input text.
        replacements: Dictionary mapping old strings to new strings.
    
    Returns:
        Text with all replacements applied.
    
    Example:
        >>> replace_multiple("hello world", {"hello": "hi", "world": "there"})
        'hi there'
    """
    # TODO: Implement this function
    pass


def extract_initials(text: str) -> str:
    """
    Extract initials from a name.
    
    Args:
        text: A name (e.g., "John Smith").
    
    Returns:
        Initials in uppercase (e.g., "JS").
    
    Example:
        >>> extract_initials("John Smith")
        'JS'
        >>> extract_initials("Mary Jane Watson")
        'MJW'
    """
    # TODO: Implement this function
    pass


# =============================================================================
# TEST CASES
# =============================================================================

def run_tests() -> None:
    """Run all test cases."""
    # Test normalise_whitespace
    assert normalise_whitespace("  Hello   World  ") == "Hello World"
    assert normalise_whitespace("No  extra  spaces") == "No extra spaces"
    assert normalise_whitespace("   ") == ""
    
    # Test count_words
    assert count_words("The quick brown fox") == 4
    assert count_words("") == 0
    assert count_words("   ") == 0
    
    # Test reverse_words
    assert reverse_words("Hello World") == "World Hello"
    assert reverse_words("One") == "One"
    
    # Test capitalise_words
    assert capitalise_words("hello world") == "Hello World"
    assert capitalise_words("HELLO WORLD") == "Hello World"
    
    # Test find_longest_word
    assert find_longest_word("The quick brown fox") == "quick"
    assert find_longest_word("") == ""
    
    # Test replace_multiple
    assert replace_multiple("hello world", {"hello": "hi"}) == "hi world"
    
    # Test extract_initials
    assert extract_initials("John Smith") == "JS"
    assert extract_initials("Mary Jane Watson") == "MJW"
    
    print("All tests passed!")


if __name__ == "__main__":
    run_tests()
