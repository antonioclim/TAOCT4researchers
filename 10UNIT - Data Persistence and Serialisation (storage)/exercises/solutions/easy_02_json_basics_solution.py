#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 10, Solution: Easy 02 - JSON Basics
═══════════════════════════════════════════════════════════════════════════════
"""

import json
from pathlib import Path
from typing import Any


def dict_to_json_string(data: dict[str, Any], pretty: bool = False) -> str:
    """Convert a dictionary to a JSON string."""
    if pretty:
        return json.dumps(data, indent=2)
    return json.dumps(data)


def json_string_to_dict(json_string: str) -> dict[str, Any]:
    """Parse a JSON string into a Python dictionary."""
    return json.loads(json_string)


def save_dict_to_json(data: dict[str, Any], filepath: Path) -> None:
    """Save a dictionary to a JSON file with pretty formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_dict_from_json(filepath: Path) -> dict[str, Any]:
    """Load a dictionary from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_json_files(
    file1: Path,
    file2: Path,
    output_path: Path
) -> dict[str, Any]:
    """Merge two JSON files into a single output file."""
    data1 = load_dict_from_json(file1)
    data2 = load_dict_from_json(file2)
    
    # Merge with file2 taking precedence
    merged = {**data1, **data2}
    
    save_dict_to_json(merged, output_path)
    return merged


def run_tests() -> None:
    """Run basic tests for the exercises."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)

        print("Testing dict_to_json_string")
        data = {'name': 'Alice', 'age': 30}
        json_str = dict_to_json_string(data)
        assert '"name"' in json_str
        pretty_str = dict_to_json_string(data, pretty=True)
        assert '\n' in pretty_str
        print("  ✓ Passed")

        print("Testing json_string_to_dict")
        result = json_string_to_dict('{"score": 100, "passed": true}')
        assert result == {'score': 100, 'passed': True}
        print("  ✓ Passed")

        print("Testing save_dict_to_json")
        test_data = {'experiment': 'trial_1', 'parameters': [1, 2, 3]}
        json_file = test_dir / 'test.json'
        save_dict_to_json(test_data, json_file)
        assert json_file.exists()
        print("  ✓ Passed")

        print("Testing load_dict_from_json")
        loaded = load_dict_from_json(json_file)
        assert loaded == test_data
        print("  ✓ Passed")

        print("Testing merge_json_files")
        file1 = test_dir / 'base.json'
        file2 = test_dir / 'override.json'
        output = test_dir / 'merged.json'
        save_dict_to_json({'a': 1, 'b': 2}, file1)
        save_dict_to_json({'b': 3, 'c': 4}, file2)
        merged = merge_json_files(file1, file2, output)
        assert merged == {'a': 1, 'b': 3, 'c': 4}
        print("  ✓ Passed")

        print("\nAll tests passed! 🎉")


if __name__ == "__main__":
    run_tests()
