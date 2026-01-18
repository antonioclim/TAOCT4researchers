# Practice Exercise: Medium 01

## Palindrome Turing Machine Design

**Difficulty:** Medium  
**Estimated Time:** 30 minutes  
**Topics:** Turing machine design, state diagrams

---

## Problem

The provided palindrome checker works for binary strings (0s and 1s). Your task is to extend it to work with a **ternary alphabet** (0, 1, 2).

**Input:** A string of 0s, 1s and 2s  
**Output:** Accept if palindrome, reject otherwise

**Examples:**
- "012210" → Accept (is a palindrome)
- "0120" → Reject (not a palindrome)
- "22" → Accept
- "212" → Accept
- "" → Accept (empty string)

---

## Requirements

1. **State Diagram:** Draw a state transition diagram
2. **Transition Table:** Write out all transitions
3. **Implementation:** Create the `TuringMachine` in Python
4. **Testing:** Verify with at least 5 test cases

---

## Starter Code

```python
def create_ternary_palindrome_checker() -> TuringMachine:
    """
    Create a Turing machine that checks if a ternary string is a palindrome.
    
    Algorithm:
    1. Read and erase the first character, remember it
    2. Move to the end, check if last character matches
    3. Erase the last character
    4. Return to the beginning and repeat
    5. Accept if string becomes empty
    """
    return TuringMachine(
        transitions={
            # TODO: Implement transitions
            # You will need states for:
            # - q0: initial state, read first character
            # - q_seek_end_for_0: looking for end after reading 0
            # - q_seek_end_for_1: looking for end after reading 1
            # - q_seek_end_for_2: looking for end after reading 2
            # - q_check_0, q_check_1, q_check_2: verify last character
            # - q_return: go back to the beginning
        },
        initial_state="q0",
        accept_state="accept",
        reject_state="reject"
    )
```

---

## Hints

<details>
<summary>Hint 1: State count</summary>

You will need approximately 8-10 states:
- 1 initial state
- 3 "seek end" states (one for each remembered symbol)
- 3 "check" states (one for each expected symbol)
- 1 "return" state
- 2 terminal states (accept/reject)

</details>

<details>
<summary>Hint 2: Transition pattern</summary>

For each symbol X ∈ {0, 1, 2}:
```
(q0, X) → (q_seek_end_for_X, □, R)  # Read and erase first
(q_seek_end_for_X, Y) → (q_seek_end_for_X, Y, R) for Y ∈ {0,1,2}
(q_seek_end_for_X, □) → (q_check_X, □, L)  # Found end
(q_check_X, X) → (q_return, □, L)  # Match! Erase and return
(q_check_X, Y) → (reject, Y, S) for Y ≠ X  # Mismatch
```

</details>

---

## Testing Template

```python
def test_ternary_palindrome():
    machine = create_ternary_palindrome_checker()
    sim = TuringSimulator(machine)
    
    test_cases = [
        ("", True),
        ("0", True),
        ("1", True),
        ("2", True),
        ("00", True),
        ("01", False),
        ("010", True),
        ("012", False),
        ("012210", True),
        ("0120210", True),
        ("01onal", False),  # Invalid input
    ]
    
    for input_str, expected in test_cases:
        sim.load(input_str if input_str else "□")
        result = sim.run()
        status = "✓" if result == expected else "✗"
        print(f"{status} '{input_str}' → {result} (expected {expected})")
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
def create_ternary_palindrome_checker() -> TuringMachine:
    return TuringMachine(
        transitions={
            # Initial: read first character
            ("q0", "0"): Transition("q_seek_end_for_0", "□", Direction.RIGHT),
            ("q0", "1"): Transition("q_seek_end_for_1", "□", Direction.RIGHT),
            ("q0", "2"): Transition("q_seek_end_for_2", "□", Direction.RIGHT),
            ("q0", "□"): Transition("accept", "□", Direction.STAY),
            
            # Seek end for 0
            ("q_seek_end_for_0", "0"): Transition("q_seek_end_for_0", "0", Direction.RIGHT),
            ("q_seek_end_for_0", "1"): Transition("q_seek_end_for_0", "1", Direction.RIGHT),
            ("q_seek_end_for_0", "2"): Transition("q_seek_end_for_0", "2", Direction.RIGHT),
            ("q_seek_end_for_0", "□"): Transition("q_check_0", "□", Direction.LEFT),
            
            # Seek end for 1
            ("q_seek_end_for_1", "0"): Transition("q_seek_end_for_1", "0", Direction.RIGHT),
            ("q_seek_end_for_1", "1"): Transition("q_seek_end_for_1", "1", Direction.RIGHT),
            ("q_seek_end_for_1", "2"): Transition("q_seek_end_for_1", "2", Direction.RIGHT),
            ("q_seek_end_for_1", "□"): Transition("q_check_1", "□", Direction.LEFT),
            
            # Seek end for 2
            ("q_seek_end_for_2", "0"): Transition("q_seek_end_for_2", "0", Direction.RIGHT),
            ("q_seek_end_for_2", "1"): Transition("q_seek_end_for_2", "1", Direction.RIGHT),
            ("q_seek_end_for_2", "2"): Transition("q_seek_end_for_2", "2", Direction.RIGHT),
            ("q_seek_end_for_2", "□"): Transition("q_check_2", "□", Direction.LEFT),
            
            # Check for 0
            ("q_check_0", "0"): Transition("q_return", "□", Direction.LEFT),
            ("q_check_0", "1"): Transition("reject", "1", Direction.STAY),
            ("q_check_0", "2"): Transition("reject", "2", Direction.STAY),
            ("q_check_0", "□"): Transition("accept", "□", Direction.STAY),
            
            # Check for 1
            ("q_check_1", "1"): Transition("q_return", "□", Direction.LEFT),
            ("q_check_1", "0"): Transition("reject", "0", Direction.STAY),
            ("q_check_1", "2"): Transition("reject", "2", Direction.STAY),
            ("q_check_1", "□"): Transition("accept", "□", Direction.STAY),
            
            # Check for 2
            ("q_check_2", "2"): Transition("q_return", "□", Direction.LEFT),
            ("q_check_2", "0"): Transition("reject", "0", Direction.STAY),
            ("q_check_2", "1"): Transition("reject", "1", Direction.STAY),
            ("q_check_2", "□"): Transition("accept", "□", Direction.STAY),
            
            # Return to beginning
            ("q_return", "0"): Transition("q_return", "0", Direction.LEFT),
            ("q_return", "1"): Transition("q_return", "1", Direction.LEFT),
            ("q_return", "2"): Transition("q_return", "2", Direction.LEFT),
            ("q_return", "□"): Transition("q0", "□", Direction.RIGHT),
        },
        initial_state="q0",
        accept_state="accept",
        reject_state="reject"
    )
```

</details>

---

© 2025 Antonio Clim. All rights reserved.
