#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Easy 01 - Basic File I/O
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
File I/O forms the foundation of data persistence. This exercise introduces
the essential patterns for reading and writing text files with proper resource
management and encoding specification.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Open and read text files using context managers
2. Write content to files with explicit encoding
3. Handle common file operation errors gracefully

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Read Entire File
# ═══════════════════════════════════════════════════════════════════════════════

def read_entire_file(filepath: Path) -> str:
    """
    Read and return the entire contents of a text file.

    Must use a context manager (with statement) for proper resource handling.
    Must specify UTF-8 encoding explicitly.

    Args:
        filepath: Path to the text file to read.

    Returns:
        Complete file contents as a string.

    Raises:
        FileNotFoundError: If the file does not exist.

    Example:
        >>> content = read_entire_file(Path('notes.txt'))
        >>> print(f"Read {len(content)} characters")
    """
    # TODO: Implement this function
    # Hint: Use 'with open(filepath, 'r', encoding='utf-8') as f:'
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Write Text File
# ═══════════════════════════════════════════════════════════════════════════════

def write_text_file(filepath: Path, content: str) -> int:
    """
    Write string content to a text file.

    Creates the file if it doesn't exist, overwrites if it does.
    Must use UTF-8 encoding.

    Args:
        filepath: Destination file path.
        content: String content to write.

    Returns:
        Number of characters written.

    Example:
        >>> chars = write_text_file(Path('output.txt'), 'Hello, World!')
        >>> print(f"Wrote {chars} characters")
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Append to File
# ═══════════════════════════════════════════════════════════════════════════════

def append_line(filepath: Path, line: str) -> None:
    """
    Append a line of text to a file.

    Adds a newline character after the content.
    Creates the file if it doesn't exist.

    Args:
        filepath: Target file path.
        line: Text to append (newline added automatically).

    Example:
        >>> append_line(Path('log.txt'), 'Entry 1')
        >>> append_line(Path('log.txt'), 'Entry 2')
    """
    # TODO: Implement this function
    # Hint: Use mode 'a' for append
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Count Lines
# ═══════════════════════════════════════════════════════════════════════════════

def count_lines(filepath: Path) -> int:
    """
    Count the number of lines in a text file.

    Empty files should return 0.

    Args:
        filepath: Path to the file.

    Returns:
        Number of lines in the file.

    Example:
        >>> count_lines(Path('data.txt'))
        42
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Safe File Read
# ═══════════════════════════════════════════════════════════════════════════════

def safe_read_file(filepath: Path, default: str = '') -> str:
    """
    Read file contents, returning default value if file doesn't exist.

    Unlike read_entire_file, this function doesn't raise an exception
    for missing files.

    Args:
        filepath: Path to the file.
        default: Value to return if file doesn't exist.

    Returns:
        File contents or default value.

    Example:
        >>> content = safe_read_file(Path('maybe.txt'), 'Not found')
    """
    # TODO: Implement this function
    # Hint: Use try/except to handle FileNotFoundError
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing Exercise 1: read_entire_file")
        test_file = test_dir / 'test.txt'
        test_file.write_text('Hello\nWorld\n', encoding='utf-8')
        content = read_entire_file(test_file)
        assert content == 'Hello\nWorld\n', "Content mismatch"
        print("  ✓ read_entire_file works correctly")

        print("\nTesting Exercise 2: write_text_file")
        output_file = test_dir / 'output.txt'
        chars = write_text_file(output_file, 'Test content')
        assert chars == 12, f"Expected 12 chars, got {chars}"
        assert output_file.read_text(encoding='utf-8') == 'Test content'
        print("  ✓ write_text_file works correctly")

        print("\nTesting Exercise 3: append_line")
        log_file = test_dir / 'log.txt'
        append_line(log_file, 'Line 1')
        append_line(log_file, 'Line 2')
        content = log_file.read_text(encoding='utf-8')
        assert content == 'Line 1\nLine 2\n', "Append content mismatch"
        print("  ✓ append_line works correctly")

        print("\nTesting Exercise 4: count_lines")
        lines = count_lines(test_file)
        assert lines == 2, f"Expected 2 lines, got {lines}"
        empty_file = test_dir / 'empty.txt'
        empty_file.touch()
        assert count_lines(empty_file) == 0, "Empty file should have 0 lines"
        print("  ✓ count_lines works correctly")

        print("\nTesting Exercise 5: safe_read_file")
        content = safe_read_file(test_file)
        assert content == 'Hello\nWorld\n'
        missing_content = safe_read_file(test_dir / 'missing.txt', 'DEFAULT')
        assert missing_content == 'DEFAULT', "Should return default for missing file"
        print("  ✓ safe_read_file works correctly")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
