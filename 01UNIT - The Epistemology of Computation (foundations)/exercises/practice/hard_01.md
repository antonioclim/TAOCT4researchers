# Practice Exercise: Hard 01

## Universal Turing Machine Simulator

**Difficulty:** Hard  
**Estimated Time:** 60 minutes  
**Topics:** Universal computation, Turing machine encoding, meta-programming

---

## Problem

A **Universal Turing Machine** (UTM) is a Turing machine that can simulate any other Turing machine. It takes as input:
1. An encoding of a Turing machine M
2. An input string w

And simulates M running on w.

Your task is to implement a simplified UTM simulator in Python that:
1. Accepts Turing machines in a textual encoding format
2. Simulates the encoded machine on a given input
3. Returns the same result as running the machine directly

---

## Encoding Format

Use the following format to encode Turing machines:

```
# Machine definition file format
STATES: q0, q1, q2, accept, reject
INITIAL: q0
ACCEPT: accept
REJECT: reject
ALPHABET: 0, 1, □

# Transitions: (state, symbol) -> (new_state, write, direction)
TRANSITIONS:
q0, 0 -> q1, X, R
q0, 1 -> q0, 1, R
q1, 0 -> q1, 0, R
q1, □ -> accept, □, S
```

---

## Requirements

### Part 1: Parser

Implement a parser that reads the encoding format:

```python
@dataclass
class EncodedMachine:
    states: set[str]
    initial_state: str
    accept_state: str
    reject_state: str
    alphabet: set[str]
    transitions: dict[tuple[str, str], tuple[str, str, str]]

def parse_machine_encoding(text: str) -> EncodedMachine:
    """Parse a machine encoding from text format."""
    # TODO: Implement
    pass
```

### Part 2: Universal Simulator

Implement the UTM that uses the parsed encoding:

```python
class UniversalTuringMachine:
    """
    A Universal Turing Machine simulator.
    
    This simulates any Turing machine given its encoding.
    """
    
    def __init__(self):
        self.loaded_machine: EncodedMachine | None = None
        self.tape: dict[int, str] = {}
        self.head: int = 0
        self.state: str = ""
    
    def load_machine(self, encoding: str) -> None:
        """Load a machine from its textual encoding."""
        self.loaded_machine = parse_machine_encoding(encoding)
    
    def load_input(self, input_string: str) -> None:
        """Load input onto the tape."""
        # TODO: Implement
        pass
    
    def step(self) -> bool:
        """Execute one step. Returns False if halted."""
        # TODO: Implement
        pass
    
    def run(self, max_steps: int = 10000) -> tuple[bool, str]:
        """
        Run the machine.
        
        Returns:
            (accepted, output) tuple
        """
        # TODO: Implement
        pass
```

### Part 3: Verification

Create test cases that verify the UTM produces the same results as direct simulation:

```python
def test_utm_equivalence():
    # Define a machine directly
    direct_machine = create_palindrome_checker()
    
    # Encode the same machine
    encoding = """
    STATES: q0, q_seek_end_for_0, q_seek_end_for_1, q_check_0, q_check_1, q_return, accept, reject
    INITIAL: q0
    ACCEPT: accept
    REJECT: reject
    ALPHABET: 0, 1, □
    
    TRANSITIONS:
    q0, 0 -> q_seek_end_for_0, □, R
    q0, 1 -> q_seek_end_for_1, □, R
    q0, □ -> accept, □, S
    # ... rest of transitions ...
    """
    
    utm = UniversalTuringMachine()
    utm.load_machine(encoding)
    
    test_inputs = ["", "0", "1", "010", "0110", "0101"]
    
    for input_str in test_inputs:
        # Direct simulation
        direct_sim = TuringSimulator(direct_machine)
        direct_sim.load(input_str or "□")
        direct_result = direct_sim.run()
        
        # UTM simulation
        utm.load_input(input_str)
        utm_result, _ = utm.run()
        
        assert direct_result == utm_result, f"Mismatch on '{input_str}'"
```

---

## Theoretical Background

The existence of a Universal Turing Machine proves several important facts:

1. **Universality:** A single machine can compute anything any Turing machine can compute
2. **Programmability:** The "program" (machine encoding) is just data
3. **Self-reference:** A UTM can simulate itself (leading to undecidability proofs)

The UTM is the theoretical foundation for:
- Stored-program computers (von Neumann architecture)
- Interpreters and virtual machines
- The concept of software

---

## Hints

<details>
<summary>Hint 1: Parsing transitions</summary>

Use regular expressions or simple string splitting:

```python
import re

def parse_transition(line: str):
    # "q0, 0 -> q1, X, R"
    match = re.match(r'(\w+),\s*(.+?)\s*->\s*(\w+),\s*(.+?),\s*([LRS])', line)
    if match:
        state, symbol, new_state, write, direction = match.groups()
        return (state, symbol), (new_state, write, direction)
```

</details>

<details>
<summary>Hint 2: Handling the blank symbol</summary>

The blank symbol (□) may appear in the encoding as either "□" or "blank" or "_". Normalise it:

```python
def normalise_symbol(s: str) -> str:
    if s.strip() in ('□', 'blank', '_', 'B'):
        return '□'
    return s.strip()
```

</details>

---

## Extension Challenge

Implement a **self-interpreter**: encode your UTM as a Turing machine and run it on the UTM to simulate a third machine. This demonstrates the power of universal computation!

---

## Solution

<details>
<summary>Click to reveal solution</summary>

See `exercises/solutions/hard_01_solution.py` for the complete implementation.

Key insights:
1. The parser needs to handle comments and whitespace gracefully
2. Direction mapping: 'L' → -1, 'R' → +1, 'S' → 0
3. The tape should use a dictionary for sparse representation
4. Error handling for undefined transitions (implicit reject)

</details>

---

© 2025 Antonio Clim. All rights reserved.
