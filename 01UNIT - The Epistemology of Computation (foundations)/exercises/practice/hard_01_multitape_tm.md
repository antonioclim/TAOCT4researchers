# Practice Exercise: Hard 01

## Multi-Tape Turing Machine Simulation

**Difficulty:** Hard  
**Estimated Time:** 45 minutes  
**Prerequisites:** Turing machine implementation, Python proficiency  

---

## Background

A k-tape Turing machine has k independent tapes, each with its own head. In a single step, the machine can read all k symbols under the heads, write k symbols, move each head independently and transition to a new state.

Multi-tape machines are equivalent in power to single-tape machines but can be exponentially faster for certain problems.

---

## Problem

Implement a 2-tape Turing machine simulator and use it to implement binary addition.

### Specification

**Input:** Two binary numbers on tape 1, separated by '+' (e.g., "101+11")  
**Output:** The sum on tape 2 (e.g., "1000" for 5+3=8)

### Requirements

1. Extend the existing `TuringMachine` class to support multiple tapes
2. Implement a `MultiTapeSimulator` class
3. Create a 2-tape binary addition machine
4. All code must have type hints and docstrings

---

## Starter Code

```python
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple, List

class Direction(Enum):
    LEFT = auto()
    RIGHT = auto()
    STAY = auto()

@dataclass(frozen=True)
class MultiTapeTransition:
    """Transition for a k-tape Turing machine.
    
    Attributes:
        next_state: The state to transition to.
        write_symbols: Tuple of symbols to write (one per tape).
        directions: Tuple of directions to move (one per tape).
    """
    next_state: str
    write_symbols: Tuple[str, ...]
    directions: Tuple[Direction, ...]


class MultiTapeTuringMachine:
    """A Turing machine with multiple tapes.
    
    TODO: Implement this class.
    """
    pass


class MultiTapeSimulator:
    """Simulator for multi-tape Turing machines.
    
    TODO: Implement this class.
    """
    pass


def create_binary_addition_machine() -> MultiTapeTuringMachine:
    """Create a 2-tape machine for binary addition.
    
    Algorithm:
    1. Copy the first number from tape 1 to tape 2
    2. Position tape 1 head after the '+'
    3. Add the second number to tape 2, bit by bit
    4. Handle carry propagation
    
    TODO: Implement this function.
    """
    pass
```

---

## Algorithm Hints

<details>
<summary>💡 Hint 1: Addition Strategy</summary>

For binary addition with two tapes:
1. Copy first operand to tape 2 (result tape)
2. Align both numbers at their least significant bits
3. Add bit by bit from right to left
4. Track carry in the state

</details>

<details>
<summary>💡 Hint 2: State Design</summary>

Consider states like:
- `copy_first`: Copy first number to tape 2
- `seek_second`: Find the second operand
- `align`: Position heads at rightmost digits
- `add_nc`: Adding without carry
- `add_c`: Adding with carry
- `propagate`: Propagate final carry

</details>

<details>
<summary>💡 Hint 3: Transition Structure</summary>

Each transition reads from both tapes:
```python
# Key: (state, symbol_tape1, symbol_tape2)
# Value: MultiTapeTransition(next_state, (write1, write2), (dir1, dir2))

transitions = {
    ("copy_first", "1", "□"): MultiTapeTransition(
        "copy_first",
        ("1", "1"),  # Write to both tapes
        (Direction.RIGHT, Direction.RIGHT)
    ),
    ...
}
```

</details>

---

## Test Cases

```python
def test_binary_addition():
    machine = create_binary_addition_machine()
    simulator = MultiTapeSimulator(machine)
    
    test_cases = [
        ("0+0", "0"),
        ("1+0", "1"),
        ("0+1", "1"),
        ("1+1", "10"),
        ("10+1", "11"),
        ("11+1", "100"),
        ("101+11", "1000"),    # 5 + 3 = 8
        ("1111+1", "10000"),   # 15 + 1 = 16
        ("1010+101", "1111"),  # 10 + 5 = 15
    ]
    
    for input_str, expected in test_cases:
        simulator.load(input_str)
        simulator.run()
        assert simulator.get_output(tape=1) == expected
```

---

## Solution Outline

<details>
<summary>Click to reveal solution structure</summary>

```python
class MultiTapeTuringMachine:
    def __init__(
        self,
        num_tapes: int,
        transitions: Dict[Tuple[str, ...], MultiTapeTransition],
        initial_state: str,
        accept_state: str = "accept",
        reject_state: str = "reject",
        blank_symbol: str = "□"
    ):
        self.num_tapes = num_tapes
        self.transitions = transitions
        self.initial_state = initial_state
        self.accept_state = accept_state
        self.reject_state = reject_state
        self.blank_symbol = blank_symbol


class MultiTapeSimulator:
    def __init__(self, machine: MultiTapeTuringMachine):
        self.machine = machine
        self.tapes: List[Dict[int, str]] = []
        self.heads: List[int] = []
        self.state: str = ""
        self.step_count: int = 0
    
    def load(self, input_string: str, tape: int = 0):
        # Initialize tapes
        self.tapes = [{} for _ in range(self.machine.num_tapes)]
        self.heads = [0] * self.machine.num_tapes
        self.state = self.machine.initial_state
        self.step_count = 0
        
        # Load input onto specified tape
        for i, c in enumerate(input_string):
            self.tapes[tape][i] = c
    
    def step(self) -> bool:
        # Read symbols from all tapes
        symbols = tuple(
            self.tapes[t].get(self.heads[t], self.machine.blank_symbol)
            for t in range(self.machine.num_tapes)
        )
        
        # Look up transition
        key = (self.state,) + symbols
        if key not in self.machine.transitions:
            self.state = self.machine.reject_state
            return False
        
        trans = self.machine.transitions[key]
        
        # Apply transition
        for t in range(self.machine.num_tapes):
            self.tapes[t][self.heads[t]] = trans.write_symbols[t]
            if trans.directions[t] == Direction.LEFT:
                self.heads[t] -= 1
            elif trans.directions[t] == Direction.RIGHT:
                self.heads[t] += 1
        
        self.state = trans.next_state
        self.step_count += 1
        
        return self.state not in (
            self.machine.accept_state,
            self.machine.reject_state
        )
```

The full binary addition machine requires approximately 20-30 transitions covering all combinations of bits and carry states.

</details>

---

## Extension Challenges

1. Implement binary subtraction
2. Implement binary multiplication using repeated addition
3. Compare the step counts with single-tape implementations

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
