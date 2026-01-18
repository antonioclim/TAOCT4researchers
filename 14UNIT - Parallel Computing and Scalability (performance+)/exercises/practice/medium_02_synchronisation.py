#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
14UNIT Exercise: Synchronisation Patterns (Medium)
═══════════════════════════════════════════════════════════════════════════════

DIFFICULTY: ★★★☆☆ (Medium)
ESTIMATED TIME: 20 minutes
PREREQUISITES: Multiprocessing basics, shared state concepts

LEARNING OBJECTIVES
───────────────────
- LO2: Implement synchronisation primitives correctly
- LO3: Recognise and prevent race conditions

PROBLEM DESCRIPTION
───────────────────
Implement thread-safe counters and demonstrate the difference between
synchronised and unsynchronised access to shared state.

TASKS
─────
1. Implement `unsafe_counter` - demonstrate race condition (educational)
2. Implement `safe_counter_lock` - counter protected by Lock
3. Implement `safe_counter_value` - counter using Value with get_lock()

HINTS
─────
- Hint 1: multiprocessing.Value provides shared memory with a lock
- Hint 2: Use 'with lock:' for automatic acquire/release
- Hint 3: Value.get_lock() returns the associated lock

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from multiprocessing import Lock, Process, Value
from typing import Any


def unsafe_counter(n_processes: int, increments_per_process: int) -> int:
    """
    Demonstrate race condition with unsynchronised counter.
    
    WARNING: This function intentionally contains a bug for educational
    purposes. The returned value will typically be less than expected.
    
    Args:
        n_processes: Number of concurrent processes.
        increments_per_process: Increments each process performs.
        
    Returns:
        Final counter value (typically less than n_processes * increments_per_process).
        
    Example:
        >>> result = unsafe_counter(4, 10000)
        >>> expected = 4 * 10000
        >>> result < expected  # Race condition causes lost increments
        True
    """
    # TODO: Implement this function
    # 1. Create a shared Value('i', 0) - integer initialised to 0
    # 2. Define a worker function that increments counter.value in a loop
    #    WITHOUT using a lock (intentionally unsafe)
    # 3. Create and start n_processes processes
    # 4. Join all processes
    # 5. Return the final counter value
    raise NotImplementedError("Implement unsafe_counter")


def safe_counter_lock(
    n_processes: int,
    increments_per_process: int
) -> int:
    """
    Thread-safe counter using explicit Lock.
    
    Args:
        n_processes: Number of concurrent processes.
        increments_per_process: Increments each process performs.
        
    Returns:
        Final counter value (exactly n_processes * increments_per_process).
        
    Example:
        >>> result = safe_counter_lock(4, 10000)
        >>> result == 4 * 10000
        True
    """
    # TODO: Implement this function
    # 1. Create a shared Value and a Lock
    # 2. Define a worker that acquires the lock before each increment
    # 3. Create, start, and join processes
    # 4. Return the final value
    raise NotImplementedError("Implement safe_counter_lock")


def safe_counter_value(
    n_processes: int,
    increments_per_process: int
) -> int:
    """
    Thread-safe counter using Value's built-in lock.
    
    Value objects have a get_lock() method that returns
    their associated lock for synchronisation.
    
    Args:
        n_processes: Number of concurrent processes.
        increments_per_process: Increments each process performs.
        
    Returns:
        Final counter value (exactly n_processes * increments_per_process).
        
    Example:
        >>> result = safe_counter_value(4, 10000)
        >>> result == 4 * 10000
        True
    """
    # TODO: Implement this function
    # Use counter.get_lock() instead of a separate Lock object
    raise NotImplementedError("Implement safe_counter_value")


def demonstrate_race_condition() -> dict[str, Any]:
    """
    Compare unsafe vs safe counter implementations.
    
    Returns:
        Dictionary with results from each implementation.
    """
    n_processes = 4
    increments = 50_000
    expected = n_processes * increments
    
    results = {
        'expected': expected,
        'unsafe': unsafe_counter(n_processes, increments),
        'safe_lock': safe_counter_lock(n_processes, increments),
        'safe_value': safe_counter_value(n_processes, increments),
    }
    
    results['unsafe_loss'] = expected - results['unsafe']
    results['unsafe_loss_pct'] = 100 * results['unsafe_loss'] / expected
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_unsafe_counter() -> None:
    """Test that unsafe counter demonstrates race condition."""
    result = unsafe_counter(4, 10_000)
    expected = 40_000
    # Should lose some increments due to race condition
    # (In rare cases might pass by luck, but usually fails)
    print(f"  Unsafe: {result} / {expected} ({100*result/expected:.1f}%)")
    # We don't assert failure - just demonstrate


def test_safe_counter_lock() -> None:
    """Test that lock-based counter is correct."""
    result = safe_counter_lock(4, 10_000)
    expected = 40_000
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ safe_counter_lock: {result} == {expected}")


def test_safe_counter_value() -> None:
    """Test that Value-based counter is correct."""
    result = safe_counter_value(4, 10_000)
    expected = 40_000
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ safe_counter_value: {result} == {expected}")


def main() -> None:
    """Run all tests."""
    print("Running medium_02_synchronisation tests...")
    print("-" * 50)
    
    try:
        test_unsafe_counter()
        test_safe_counter_lock()
        test_safe_counter_value()
        print("-" * 50)
        print("All tests passed! ✓")
    except NotImplementedError as e:
        print(f"Not yet implemented: {e}")
    except AssertionError as e:
        print(f"Test failed: {e}")


if __name__ == '__main__':
    main()
