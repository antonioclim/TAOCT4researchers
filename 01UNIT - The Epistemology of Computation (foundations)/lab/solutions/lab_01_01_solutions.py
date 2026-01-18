"""
Week 1: Lab 1 Solutions — Turing Machine Exercises.

This module contains complete solutions for the Turing machine exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
These solutions are provided for educational purposes only.
"""

from typing import Dict
import sys
sys.path.insert(0, '..')

from lab_1_01_turing_machine import (
    TuringMachine, TuringSimulator, Transition, Direction
)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Binary Increment Machine
# ═══════════════════════════════════════════════════════════════════════════════

def create_binary_increment_machine() -> TuringMachine:
    """Create a Turing machine that increments a binary number.
    
    Algorithm:
        1. Move right to the end of the number
        2. Move left, flipping 1s to 0s (carry propagation)
        3. When a 0 is found, flip to 1 and halt
        4. If blank is reached, write 1 (overflow)
    
    Examples:
        "0" → "1"
        "1" → "10"
        "101" → "110"
        "111" → "1000"
    
    Returns:
        A TuringMachine that performs binary increment.
    """
    transitions: Dict[tuple[str, str], Transition] = {
        # State q_seek_end: Move right to find end of number
        ("q_seek_end", "0"): Transition("q_seek_end", "0", Direction.RIGHT),
        ("q_seek_end", "1"): Transition("q_seek_end", "1", Direction.RIGHT),
        ("q_seek_end", "□"): Transition("q_increment", "□", Direction.LEFT),
        
        # State q_increment: Process bits from right to left
        ("q_increment", "0"): Transition("accept", "1", Direction.STAY),  # No carry
        ("q_increment", "1"): Transition("q_increment", "0", Direction.LEFT),  # Carry
        ("q_increment", "□"): Transition("accept", "1", Direction.STAY),  # Overflow
    }
    
    return TuringMachine(
        transitions=transitions,
        initial_state="q_seek_end",
        accept_state="accept",
        reject_state="reject",
        blank_symbol="□",
    )


def test_binary_increment():
    """Test the binary increment machine."""
    machine = create_binary_increment_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("0", "1"),       # 0 → 1
        ("1", "10"),      # 1 → 2
        ("10", "11"),     # 2 → 3
        ("11", "100"),    # 3 → 4
        ("111", "1000"),  # 7 → 8
        ("1011", "1100"), # 11 → 12
        ("1111", "10000"),# 15 → 16
    ]
    
    print("Binary Increment Machine Tests")
    print("=" * 50)
    
    all_passed = True
    for input_str, expected in test_cases:
        simulator.load(input_str)
        simulator.run()
        output = simulator.get_output()
        
        passed = output == expected
        status = "✓" if passed else "✗"
        print(f"  {status} Input: {input_str:8s} → Output: {output:8s} (expected: {expected})")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Balanced Parentheses Machine
# ═══════════════════════════════════════════════════════════════════════════════

def create_balanced_parentheses_machine() -> TuringMachine:
    """Create a Turing machine that checks balanced parentheses.
    
    Algorithm:
        1. Find the rightmost '(' that hasn't been matched
        2. Find the first ')' to its right
        3. Mark both as matched (X)
        4. Repeat until no unmatched parentheses remain
        5. Accept if all matched, reject if mismatch found
    
    Examples:
        "" → Accept
        "()" → Accept
        "(())" → Accept
        "(()" → Reject
        ")(" → Reject
    
    Returns:
        A TuringMachine that verifies balanced parentheses.
    """
    transitions: Dict[tuple[str, str], Transition] = {
        # State q_scan: Scan for the rightmost unmatched '('
        ("q_scan", "("): Transition("q_scan", "(", Direction.RIGHT),
        ("q_scan", ")"): Transition("q_scan", ")", Direction.RIGHT),
        ("q_scan", "X"): Transition("q_scan", "X", Direction.RIGHT),
        ("q_scan", "□"): Transition("q_back", "□", Direction.LEFT),
        
        # State q_back: Go back to find rightmost '('
        ("q_back", ")"): Transition("q_back", ")", Direction.LEFT),
        ("q_back", "X"): Transition("q_back", "X", Direction.LEFT),
        ("q_back", "("): Transition("q_find_close", "X", Direction.RIGHT),
        ("q_back", "□"): Transition("q_verify", "□", Direction.RIGHT),
        
        # State q_find_close: Find matching ')' to the right
        ("q_find_close", "X"): Transition("q_find_close", "X", Direction.RIGHT),
        ("q_find_close", "("): Transition("reject", "(", Direction.STAY),  # Nested unmatched
        ("q_find_close", ")"): Transition("q_scan", "X", Direction.LEFT),
        ("q_find_close", "□"): Transition("reject", "□", Direction.STAY),  # No match
        
        # State q_verify: Verify all parentheses are matched
        ("q_verify", "X"): Transition("q_verify", "X", Direction.RIGHT),
        ("q_verify", "□"): Transition("accept", "□", Direction.STAY),
        ("q_verify", "("): Transition("reject", "(", Direction.STAY),
        ("q_verify", ")"): Transition("reject", ")", Direction.STAY),
    }
    
    return TuringMachine(
        transitions=transitions,
        initial_state="q_scan",
        accept_state="accept",
        reject_state="reject",
        blank_symbol="□",
    )


def test_balanced_parentheses():
    """Test the balanced parentheses machine."""
    machine = create_balanced_parentheses_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("", True),           # Empty string is balanced
        ("()", True),         # Single pair
        ("(())", True),       # Nested
        ("()()", True),       # Sequential
        ("((()))", True),     # Deeply nested
        ("(()())", True),     # Mixed
        ("(", False),         # Unmatched open
        (")", False),         # Unmatched close
        ("(()", False),       # Missing close
        ("())", False),       # Extra close
        (")(", False),        # Wrong order
    ]
    
    print("Balanced Parentheses Machine Tests")
    print("=" * 50)
    
    all_passed = True
    for input_str, expected in test_cases:
        # Handle empty string
        if input_str:
            simulator.load(input_str)
        else:
            simulator.load("□")
        
        simulator.run()
        result = simulator.accepted
        
        passed = result == expected
        status = "✓" if passed else "✗"
        expected_str = "Accept" if expected else "Reject"
        result_str = "Accept" if result else "Reject"
        print(f"  {status} Input: {repr(input_str):12s} → {result_str:8s} (expected: {expected_str})")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: String Reversal Machine
# ═══════════════════════════════════════════════════════════════════════════════

def create_string_reversal_machine() -> TuringMachine:
    """Create a Turing machine that reverses a binary string.
    
    Algorithm (multi-pass):
        1. Mark the leftmost unmarked symbol and remember it
        2. Move to the right end and find space for output
        3. Write the remembered symbol
        4. Return left and repeat
        5. Accept when all input symbols are marked
    
    This is a challenging exercise that requires careful state management.
    
    Returns:
        A TuringMachine that reverses binary strings.
    """
    # This is a complex machine requiring many states.
    # We use markers: X for processed, Y for the separator
    
    transitions: Dict[tuple[str, str], Transition] = {
        # Initial state: Find first unmarked symbol
        ("q0", "0"): Transition("carry0_right", "X", Direction.RIGHT),
        ("q0", "1"): Transition("carry1_right", "X", Direction.RIGHT),
        ("q0", "X"): Transition("q0", "X", Direction.RIGHT),
        ("q0", "Y"): Transition("cleanup", "Y", Direction.RIGHT),  # Done
        ("q0", "□"): Transition("cleanup", "Y", Direction.LEFT),   # Add separator
        
        # Carrying 0 to the right
        ("carry0_right", "0"): Transition("carry0_right", "0", Direction.RIGHT),
        ("carry0_right", "1"): Transition("carry0_right", "1", Direction.RIGHT),
        ("carry0_right", "X"): Transition("carry0_right", "X", Direction.RIGHT),
        ("carry0_right", "Y"): Transition("carry0_right", "Y", Direction.RIGHT),
        ("carry0_right", "□"): Transition("return_left", "0", Direction.LEFT),
        
        # Carrying 1 to the right
        ("carry1_right", "0"): Transition("carry1_right", "0", Direction.RIGHT),
        ("carry1_right", "1"): Transition("carry1_right", "1", Direction.RIGHT),
        ("carry1_right", "X"): Transition("carry1_right", "X", Direction.RIGHT),
        ("carry1_right", "Y"): Transition("carry1_right", "Y", Direction.RIGHT),
        ("carry1_right", "□"): Transition("return_left", "1", Direction.LEFT),
        
        # Return to the left to process next symbol
        ("return_left", "0"): Transition("return_left", "0", Direction.LEFT),
        ("return_left", "1"): Transition("return_left", "1", Direction.LEFT),
        ("return_left", "X"): Transition("return_left", "X", Direction.LEFT),
        ("return_left", "Y"): Transition("return_left", "Y", Direction.LEFT),
        ("return_left", "□"): Transition("q0", "□", Direction.RIGHT),
        
        # Cleanup: Remove markers and separator
        ("cleanup", "X"): Transition("cleanup", "□", Direction.RIGHT),
        ("cleanup", "Y"): Transition("cleanup", "□", Direction.RIGHT),
        ("cleanup", "0"): Transition("cleanup", "0", Direction.RIGHT),
        ("cleanup", "1"): Transition("cleanup", "1", Direction.RIGHT),
        ("cleanup", "□"): Transition("accept", "□", Direction.STAY),
    }
    
    return TuringMachine(
        transitions=transitions,
        initial_state="q0",
        accept_state="accept",
        reject_state="reject",
        blank_symbol="□",
    )


def test_string_reversal():
    """Test the string reversal machine."""
    machine = create_string_reversal_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("0", "0"),
        ("1", "1"),
        ("01", "10"),
        ("10", "01"),
        ("110", "011"),
        ("1010", "0101"),
        ("11100", "00111"),
    ]
    
    print("String Reversal Machine Tests")
    print("=" * 50)
    
    all_passed = True
    for input_str, expected in test_cases:
        simulator.load(input_str)
        simulator.run(max_steps=1000)
        output = simulator.get_output()
        
        passed = output == expected
        status = "✓" if passed else "✗"
        print(f"  {status} Input: {input_str:8s} → Output: {output:8s} (expected: {expected})")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("═" * 60)
    print("WEEK 1: LAB 1 SOLUTIONS — TURING MACHINE EXERCISES")
    print("═" * 60)
    print()
    
    results = []
    
    results.append(("Binary Increment", test_binary_increment()))
    results.append(("Balanced Parentheses", test_balanced_parentheses()))
    results.append(("String Reversal", test_string_reversal()))
    
    print("═" * 60)
    print("SUMMARY")
    print("═" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:25s} {status}")
    
    all_passed = all(p for _, p in results)
    print()
    print(f"Overall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
