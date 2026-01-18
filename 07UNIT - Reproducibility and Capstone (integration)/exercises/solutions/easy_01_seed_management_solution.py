#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
SOLUTION: Easy Exercise 1 — Seed Management
═══════════════════════════════════════════════════════════════════════════════

Complete solutions for reproducible random number generation exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: GLOBAL SEED SETTING — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int) -> None:
    """
    Set the global random seed for both Python's random and NumPy.

    This function ensures reproducibility across the entire application by
    synchronising the random number generators of both libraries.

    Args:
        seed: The integer seed value to use for both generators.

    Example:
        >>> set_global_seed(42)
        >>> random.random()  # Will always produce the same value
        0.6394267984578837
        >>> np.random.random()  # Will always produce the same value
        0.3745401188473625
    """
    random.seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: REPRODUCIBLE GENERATION — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_random_list(
    size: int,
    min_val: float = 0.0,
    max_val: float = 1.0,
    seed: int | None = None
) -> list[float]:
    """
    Generate a list of random floats with optional seed for reproducibility.

    Uses Python's built-in random module to generate uniform random values
    within the specified range. When a seed is provided, the function
    produces deterministic output.

    Args:
        size: Number of random values to generate.
        min_val: Minimum value (inclusive). Defaults to 0.0.
        max_val: Maximum value (exclusive). Defaults to 1.0.
        seed: Optional seed for reproducibility. If None, uses current
            random state.

    Returns:
        A list of random floats within [min_val, max_val).

    Raises:
        ValueError: If size is negative or min_val >= max_val.

    Example:
        >>> generate_random_list(3, seed=42)
        [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]
    """
    if size < 0:
        raise ValueError("Size must be non-negative")
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")

    if seed is not None:
        random.seed(seed)

    return [random.uniform(min_val, max_val) for _ in range(size)]


def generate_random_array(
    shape: tuple[int, ...],
    seed: int | None = None
) -> np.ndarray:
    """
    Generate a NumPy array of random floats with optional seed.

    Uses NumPy's random number generator to create an array of the
    specified shape filled with uniform random values in [0, 1).

    Args:
        shape: Tuple specifying the dimensions of the output array.
        seed: Optional seed for reproducibility.

    Returns:
        NumPy array of the specified shape with random values.

    Raises:
        ValueError: If any dimension is negative.

    Example:
        >>> generate_random_array((2, 3), seed=42)
        array([[0.37454012, 0.95071431, 0.73199394],
               [0.59865848, 0.15601864, 0.15599452]])
    """
    if any(dim < 0 for dim in shape):
        raise ValueError("All dimensions must be non-negative")

    if seed is not None:
        np.random.seed(seed)

    return np.random.random(shape)


def verify_reproducibility(
    generator_func: callable,
    seed: int,
    *args: Any,
    **kwargs: Any
) -> bool:
    """
    Verify that a generator function produces reproducible results.

    Calls the generator function twice with the same seed and arguments,
    then compares the outputs to ensure they are identical.

    Args:
        generator_func: The function to test for reproducibility.
        seed: The seed value to use for both calls.
        *args: Positional arguments to pass to the generator.
        **kwargs: Keyword arguments to pass to the generator.

    Returns:
        True if both calls produce identical results, False otherwise.

    Example:
        >>> verify_reproducibility(generate_random_list, 42, 5)
        True
    """
    # First generation
    result1 = generator_func(*args, seed=seed, **kwargs)

    # Second generation with same seed
    result2 = generator_func(*args, seed=seed, **kwargs)

    # Compare results
    if isinstance(result1, np.ndarray):
        return np.array_equal(result1, result2)
    return result1 == result2


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: STATE CAPTURE AND RESTORATION — SOLUTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RandomState:
    """
    Container for captured random states from both Python and NumPy.

    Attributes:
        python_state: The state tuple from random.getstate().
        numpy_state: The state dictionary from np.random.get_state().
    """

    python_state: tuple
    numpy_state: dict


def capture_random_state() -> RandomState:
    """
    Capture the current state of both Python and NumPy random generators.

    This function snapshots the internal state of both random number
    generators, allowing the exact sequence to be reproduced later.

    Returns:
        A RandomState object containing both states.

    Example:
        >>> random.seed(42)
        >>> np.random.seed(42)
        >>> state = capture_random_state()
        >>> random.random()  # Advances the state
        0.6394267984578837
        >>> restore_random_state(state)
        >>> random.random()  # Same value again
        0.6394267984578837
    """
    return RandomState(
        python_state=random.getstate(),
        numpy_state=np.random.get_state()
    )


def restore_random_state(state: RandomState) -> None:
    """
    Restore both Python and NumPy random generators to a captured state.

    This function restores the internal state of both random number
    generators from a previously captured RandomState object.

    Args:
        state: A RandomState object containing the states to restore.

    Example:
        >>> state = capture_random_state()
        >>> values_before = [random.random() for _ in range(3)]
        >>> restore_random_state(state)
        >>> values_after = [random.random() for _ in range(3)]
        >>> values_before == values_after
        True
    """
    random.setstate(state.python_state)
    np.random.set_state(state.numpy_state)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION AND DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Run all validation tests for the exercises."""
    print("=" * 70)
    print("SOLUTION VALIDATION: Easy Exercise 1 — Seed Management")
    print("=" * 70)

    # Test Exercise 1: Global seed setting
    print("\n--- Exercise 1: Global Seed Setting ---")
    set_global_seed(42)
    py_val1 = random.random()
    np_val1 = np.random.random()

    set_global_seed(42)
    py_val2 = random.random()
    np_val2 = np.random.random()

    assert py_val1 == py_val2, "Python random not reproducible"
    assert np_val1 == np_val2, "NumPy random not reproducible"
    print("✓ Global seed setting works correctly")

    # Test Exercise 2: Reproducible generation
    print("\n--- Exercise 2: Reproducible Generation ---")
    list1 = generate_random_list(10, seed=123)
    list2 = generate_random_list(10, seed=123)
    assert list1 == list2, "List generation not reproducible"
    print("✓ List generation is reproducible")

    arr1 = generate_random_array((3, 4), seed=456)
    arr2 = generate_random_array((3, 4), seed=456)
    assert np.array_equal(arr1, arr2), "Array generation not reproducible"
    print("✓ Array generation is reproducible")

    assert verify_reproducibility(generate_random_list, 789, 5)
    print("✓ Reproducibility verification works")

    # Test Exercise 3: State capture and restoration
    print("\n--- Exercise 3: State Capture and Restoration ---")
    set_global_seed(999)
    state = capture_random_state()

    # Generate some values
    original_py = [random.random() for _ in range(5)]
    original_np = [np.random.random() for _ in range(5)]

    # Restore and regenerate
    restore_random_state(state)
    restored_py = [random.random() for _ in range(5)]
    restored_np = [np.random.random() for _ in range(5)]

    assert original_py == restored_py, "Python state not restored"
    assert np.allclose(original_np, restored_np), "NumPy state not restored"
    print("✓ State capture and restoration works correctly")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
