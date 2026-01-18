#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Easy 01 - Basic File I/O
═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path


def read_entire_file(filepath: Path) -> str:
    """Read and return the entire contents of a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(filepath: Path, content: str) -> int:
    """Write string content to a text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        return f.write(content)


def append_line(filepath: Path, line: str) -> None:
    """Append a line of text to a file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def count_lines(filepath: Path) -> int:
    """Count the number of lines in a text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def safe_read_file(filepath: Path, default: str = '') -> str:
    """Read file contents, returning default value if file doesn't exist."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return default


def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing read_entire_file")
        test_file = test_dir / 'test.txt'
        test_file.write_text('Hello\nWorld\n', encoding='utf-8')
        content = read_entire_file(test_file)
        assert content == 'Hello\nWorld\n'
        print("  ✓ Passed")

        print("Testing write_text_file")
        output_file = test_dir / 'output.txt'
        chars = write_text_file(output_file, 'Test content')
        assert chars == 12
        print("  ✓ Passed")

        print("Testing append_line")
        log_file = test_dir / 'log.txt'
        append_line(log_file, 'Line 1')
        append_line(log_file, 'Line 2')
        content = log_file.read_text(encoding='utf-8')
        assert content == 'Line 1\nLine 2\n'
        print("  ✓ Passed")

        print("Testing count_lines")
        lines = count_lines(test_file)
        assert lines == 2
        print("  ✓ Passed")

        print("Testing safe_read_file")
        content = safe_read_file(test_file)
        assert content == 'Hello\nWorld\n'
        missing = safe_read_file(test_dir / 'missing.txt', 'DEFAULT')
        assert missing == 'DEFAULT'
        print("  ✓ Passed")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
