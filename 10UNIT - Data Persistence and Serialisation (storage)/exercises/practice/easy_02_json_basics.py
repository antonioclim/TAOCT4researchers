#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Practice Exercise: Easy 02 - JSON Basics
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
JSON (JavaScript Object Notation) provides human-readable serialisation for
configuration files, metadata records, and data interchange. This exercise
introduces fundamental JSON operations in Python.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Serialise Python dictionaries to JSON strings and files
2. Deserialise JSON data back to Python objects
3. Handle JSON formatting options for readability

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import json
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Dictionary to JSON String
# ═══════════════════════════════════════════════════════════════════════════════

def dict_to_json_string(data: dict[str, Any], pretty: bool = False) -> str:
    """
    Convert a dictionary to a JSON string.

    Args:
        data: Dictionary to serialise.
        pretty: If True, format with indentation for readability.

    Returns:
        JSON string representation.

    Example:
        >>> dict_to_json_string({'name': 'Alice', 'age': 30})
        '{"name": "Alice", "age": 30}'
        >>> dict_to_json_string({'name': 'Alice'}, pretty=True)
        '{\\n  "name": "Alice"\\n}'
    """
    # TODO: Implement this function
    # Hint: Use json.dumps() with indent parameter for pretty printing
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: JSON String to Dictionary
# ═══════════════════════════════════════════════════════════════════════════════

def json_string_to_dict(json_string: str) -> dict[str, Any]:
    """
    Parse a JSON string into a Python dictionary.

    Args:
        json_string: Valid JSON string.

    Returns:
        Parsed dictionary.

    Raises:
        json.JSONDecodeError: If the string is not valid JSON.

    Example:
        >>> json_string_to_dict('{"name": "Bob", "score": 95}')
        {'name': 'Bob', 'score': 95}
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Save Dictionary to JSON File
# ═══════════════════════════════════════════════════════════════════════════════

def save_dict_to_json(data: dict[str, Any], filepath: Path) -> None:
    """
    Save a dictionary to a JSON file with pretty formatting.

    Must use UTF-8 encoding and indent of 2 spaces.

    Args:
        data: Dictionary to save.
        filepath: Destination file path.

    Example:
        >>> save_dict_to_json({'experiment': 'trial_1'}, Path('config.json'))
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Load Dictionary from JSON File
# ═══════════════════════════════════════════════════════════════════════════════

def load_dict_from_json(filepath: Path) -> dict[str, Any]:
    """
    Load a dictionary from a JSON file.

    Args:
        filepath: Path to JSON file.

    Returns:
        Parsed dictionary.

    Example:
        >>> data = load_dict_from_json(Path('config.json'))
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 5: Merge JSON Files
# ═══════════════════════════════════════════════════════════════════════════════

def merge_json_files(
    file1: Path,
    file2: Path,
    output_path: Path
) -> dict[str, Any]:
    """
    Merge two JSON files into a single output file.

    Keys from file2 override keys from file1 if they conflict.

    Args:
        file1: First JSON file (base).
        file2: Second JSON file (overrides).
        output_path: Path for merged output.

    Returns:
        The merged dictionary.

    Example:
        >>> # file1: {"a": 1, "b": 2}
        >>> # file2: {"b": 3, "c": 4}
        >>> merged = merge_json_files(file1, file2, output)
        >>> merged
        {'a': 1, 'b': 3, 'c': 4}
    """
    # TODO: Implement this function
    # Hint: Load both files, use dict.update() or | operator
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing Exercise 1: dict_to_json_string")
        data = {'name': 'Alice', 'age': 30}
        json_str = dict_to_json_string(data)
        assert '"name"' in json_str and '"Alice"' in json_str
        pretty_str = dict_to_json_string(data, pretty=True)
        assert '\n' in pretty_str, "Pretty format should have newlines"
        print("  ✓ dict_to_json_string works correctly")

        print("\nTesting Exercise 2: json_string_to_dict")
        result = json_string_to_dict('{"score": 100, "passed": true}')
        assert result == {'score': 100, 'passed': True}
        print("  ✓ json_string_to_dict works correctly")

        print("\nTesting Exercise 3: save_dict_to_json")
        test_data = {'experiment': 'trial_1', 'parameters': [1, 2, 3]}
        json_file = test_dir / 'test.json'
        save_dict_to_json(test_data, json_file)
        assert json_file.exists(), "File should be created"
        content = json_file.read_text(encoding='utf-8')
        assert '"experiment"' in content
        print("  ✓ save_dict_to_json works correctly")

        print("\nTesting Exercise 4: load_dict_from_json")
        loaded = load_dict_from_json(json_file)
        assert loaded == test_data, "Loaded data should match original"
        print("  ✓ load_dict_from_json works correctly")

        print("\nTesting Exercise 5: merge_json_files")
        file1 = test_dir / 'base.json'
        file2 = test_dir / 'override.json'
        output = test_dir / 'merged.json'
        save_dict_to_json({'a': 1, 'b': 2}, file1)
        save_dict_to_json({'b': 3, 'c': 4}, file2)
        merged = merge_json_files(file1, file2, output)
        assert merged == {'a': 1, 'b': 3, 'c': 4}, "Merge should combine with override"
        assert output.exists(), "Output file should be created"
        print("  ✓ merge_json_files works correctly")

        print("\n" + "=" * 60)
        print("All tests passed! 🎉")
        print("=" * 60)


if __name__ == "__main__":
    run_tests()
