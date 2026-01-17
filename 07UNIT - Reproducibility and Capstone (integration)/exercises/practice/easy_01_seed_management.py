#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 7, Practice Exercise: Easy 01 - Seed Management Basics
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Reproducibility begins with controlling randomness. This exercise introduces
the fundamental concept of setting random seeds to ensure experiments can be
replicated exactly. You will implement basic seed management for Python's
built-in random module and NumPy.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Set random seeds for Python's random module
2. Set random seeds for NumPy
3. Verify that seeding produces consistent results

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 20 minutes

═══════════════════════════════════════════════════════════════════════════════
"""

import random
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Basic Seed Setting
# ═══════════════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    """
    Set the random seed for both Python's random module and NumPy.

    This function should set seeds for:
    1. Python's built-in random module
    2. NumPy's random number generator

    Args:
        seed: Integer seed value for reproducibility.

    Example:
        >>> set_global_seed(42)
        >>> random.random()  # Should always return same value
        0.6394267984578837
        >>> np.random.random()  # Should always return same value
        0.3745401188473625
    """
    # TODO: Implement this function
    # Hint: Use random.seed() and np.random.seed()
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Verify Reproducibility
# ═══════════════════════════════════════════════════════════════════════════════

def generate_random_list(seed: int, length: int) -> list[float]:
    """
    Generate a list of random floats with a specific seed.

    This function should:
    1. Set the seed using Python's random module
    2. Generate 'length' random floats between 0 and 1
    3. Return the list

    Args:
        seed: Integer seed for reproducibility.
        length: Number of random floats to generate.

    Returns:
        List of random floats.

    Example:
        >>> result1 = generate_random_list(42, 5)
        >>> result2 = generate_random_list(42, 5)
        >>> result1 == result2
        True
    """
    # TODO: Implement this function
    pass


def generate_random_array(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    """
    Generate a NumPy array of random floats with a specific seed.

    This function should:
    1. Set the NumPy seed
    2. Generate an array of the given shape with random floats
    3. Return the array

    Args:
        seed: Integer seed for reproducibility.
        shape: Shape of the array to generate.

    Returns:
        NumPy array of random floats.

    Example:
        >>> arr1 = generate_random_array(42, (3, 3))
        >>> arr2 = generate_random_array(42, (3, 3))
        >>> np.array_equal(arr1, arr2)
        True
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Seed State Capture
# ═══════════════════════════════════════════════════════════════════════════════

def capture_random_state() -> tuple[Any, dict[str, Any]]:
    """
    Capture the current state of both random generators.

    This function should capture and return the internal states of:
    1. Python's random module
    2. NumPy's random generator

    Returns:
        Tuple of (python_state, numpy_state).

    Example:
        >>> set_global_seed(42)
        >>> state = capture_random_state()
        >>> random.random()  # Advances state
        >>> # State can be used to restore later
    """
    # TODO: Implement this function
    # Hint: Use random.getstate() and np.random.get_state()
    pass


def restore_random_state(
    python_state: Any,
    numpy_state: dict[str, Any],
) -> None:
    """
    Restore the random generator states.

    Args:
        python_state: State from random.getstate().
        numpy_state: State from np.random.get_state().

    Example:
        >>> set_global_seed(42)
        >>> state = capture_random_state()
        >>> val1 = random.random()
        >>> restore_random_state(*state)
        >>> val2 = random.random()
        >>> val1 == val2
        True
    """
    # TODO: Implement this function
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR SOLUTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run basic tests for the exercises."""
    print("Testing Exercise 1: set_global_seed")
    set_global_seed(42)
    val1 = random.random()
    set_global_seed(42)
    val2 = random.random()
    assert val1 == val2, "Python random not reproducible"
    print("  ✓ Python random is reproducible")

    set_global_seed(42)
    arr1 = np.random.random(5)
    set_global_seed(42)
    arr2 = np.random.random(5)
    assert np.array_equal(arr1, arr2), "NumPy random not reproducible"
    print("  ✓ NumPy random is reproducible")

    print("\nTesting Exercise 2: generate_random_list")
    list1 = generate_random_list(123, 10)
    list2 = generate_random_list(123, 10)
    assert list1 == list2, "Lists not equal"
    assert len(list1) == 10, "Wrong length"
    print("  ✓ generate_random_list works correctly")

    print("\nTesting Exercise 2: generate_random_array")
    arr1 = generate_random_array(456, (5, 5))
    arr2 = generate_random_array(456, (5, 5))
    assert np.array_equal(arr1, arr2), "Arrays not equal"
    assert arr1.shape == (5, 5), "Wrong shape"
    print("  ✓ generate_random_array works correctly")

    print("\nTesting Exercise 3: capture and restore state")
    set_global_seed(789)
    state = capture_random_state()
    val1 = random.random()
    np_val1 = np.random.random()
    restore_random_state(*state)
    val2 = random.random()
    np_val2 = np.random.random()
    assert val1 == val2, "Python state not restored"
    assert np_val1 == np_val2, "NumPy state not restored"
    print("  ✓ State capture and restore works correctly")

    print("\n" + "=" * 60)
    print("All tests passed! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
