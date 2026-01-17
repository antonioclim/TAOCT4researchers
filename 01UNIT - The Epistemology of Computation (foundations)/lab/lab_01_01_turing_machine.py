#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 1, Lab 1: Turing Machine Simulator
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
In 1936, Alan Turing published "On Computable Numbers, with an Application to
the Entscheidungsproblem". This seminal paper introduced the concept of an
"automatic machine" (now called a Turing machine) as an abstract model of
computation. Ironically, Turing never built a physical Turing machine; the
model was purely mathematical — a thought experiment that laid the foundation
for the entire field of computer science.

PREREQUISITES
─────────────
- Week 0: Basic Python knowledge
- Python: Intermediate (classes, dataclasses, pattern matching)
- Libraries: None (standard library only)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Define the formal components of a Turing machine
2. Implement a working Turing machine simulator in Python
3. Trace the execution of a Turing machine step by step
4. Design Turing machines for simple computational tasks

ESTIMATED TIME
──────────────
- Reading: 30 minutes
- Coding: 90 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
Python 3.12+ (for pattern matching syntax)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Direction(Enum):
    """Direction of head movement on the tape.
    
    The standard Turing machine model uses LEFT and RIGHT movements.
    STAY is an extension that simplifies certain machine designs.
    """
    LEFT = auto()
    RIGHT = auto()
    STAY = auto()  # Extension to the original model


@dataclass(frozen=True)
class Transition:
    """A transition in the Turing machine.
    
    Represents the transition function δ: Q × Γ → Q × Γ × {L, R}
    
    A transition specifies what the machine should do when it reads
    a particular symbol in a particular state: which state to enter,
    what symbol to write, and which direction to move.
    
    Attributes:
        next_state: The state to transition to.
        write_symbol: The symbol to write to the tape.
        direction: The direction to move the head.
    
    Example:
        >>> t = Transition("q1", "X", Direction.RIGHT)
        >>> t.next_state
        'q1'
    """
    next_state: str
    write_symbol: str
    direction: Direction


@dataclass
class TuringMachine:
    """Complete implementation of a Turing Machine.
    
    Formally, M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject) where:
    
    - Q: Finite set of states (implicit from transitions.keys())
    - Σ: Input alphabet (subset of Γ, excludes blank)
    - Γ: Tape alphabet (symbols from transitions)
    - δ: Transition function (the transitions dict)
    - q₀: Initial state (initial_state)
    - q_accept: Accepting state (accept_state)
    - q_reject: Rejecting state (reject_state)
    
    Attributes:
        transitions: Dictionary mapping (state, symbol) to Transition.
        initial_state: The starting state of the machine.
        accept_state: The accepting (halt with success) state.
        reject_state: The rejecting (halt with failure) state.
        blank_symbol: Symbol representing empty tape cells.
    
    Raises:
        ValueError: If no transitions are defined for the initial state.
    
    Example:
        >>> tm = TuringMachine(
        ...     transitions={("q0", "1"): Transition("q0", "1", Direction.RIGHT)},
        ...     initial_state="q0"
        ... )
    """
    transitions: dict[tuple[str, str], Transition]
    initial_state: str
    accept_state: str = "accept"
    reject_state: str = "reject"
    blank_symbol: str = "□"
    
    def __post_init__(self) -> None:
        """Validate the machine definition after initialisation."""
        initial_transitions = [
            k for k in self.transitions if k[0] == self.initial_state
        ]
        if not initial_transitions:
            raise ValueError(
                f"No transitions defined for initial state '{self.initial_state}'"
            )
        logger.debug(
            "Created Turing machine with %d transitions",
            len(self.transitions)
        )


@dataclass
class Configuration:
    """The instantaneous configuration of a Turing machine.
    
    A configuration captures the complete state of computation at any moment:
    the tape contents, head position and current state. This is sometimes
    called an "instantaneous description" (ID) in textbooks.
    
    Attributes:
        tape: Dictionary mapping positions to symbols (sparse representation).
        head_position: Current position of the read/write head.
        current_state: The machine's current state.
        step_count: Number of steps executed so far.
    
    Note:
        Using a dictionary for the tape allows efficient representation of
        an infinite tape; only non-blank cells are stored.
    """
    tape: dict[int, str] = field(default_factory=dict)
    head_position: int = 0
    current_state: str = ""
    step_count: int = 0
    
    def read(self, blank: str = "□") -> str:
        """Read the symbol under the head.
        
        Args:
            blank: The blank symbol to return for empty cells.
        
        Returns:
            The symbol at the current head position.
        """
        return self.tape.get(self.head_position, blank)
    
    def write(self, symbol: str) -> None:
        """Write a symbol at the current head position.
        
        Args:
            symbol: The symbol to write to the tape.
        """
        self.tape[self.head_position] = symbol
    
    def move(self, direction: Direction) -> None:
        """Move the head in the specified direction.
        
        Args:
            direction: The direction to move (LEFT, RIGHT or STAY).
        """
        match direction:
            case Direction.LEFT:
                self.head_position -= 1
            case Direction.RIGHT:
                self.head_position += 1
            case Direction.STAY:
                pass
    
    def to_string(self, context: int = 10) -> str:
        """Create a visual representation of the configuration.
        
        Args:
            context: Number of cells to display around the head.
        
        Returns:
            A multi-line string showing the tape, head position and state.
        
        Example:
            >>> config = Configuration(
            ...     tape={0: '1', 1: '1', 2: '1'},
            ...     head_position=1,
            ...     current_state="q0",
            ...     step_count=5
            ... )
            >>> print(config.to_string(context=3))  # doctest: +SKIP
        """
        # Determine the range to display
        positions = list(self.tape.keys()) + [self.head_position]
        if positions:
            min_pos = min(min(positions), self.head_position - context)
            max_pos = max(max(positions), self.head_position + context)
        else:
            min_pos = -context
            max_pos = context
        
        # Build the visual representation
        cells = []
        markers = []
        for pos in range(min_pos, max_pos + 1):
            symbol = self.tape.get(pos, "□")
            cells.append(f" {symbol} ")
            if pos == self.head_position:
                markers.append(" ▲ ")
            else:
                markers.append("   ")
        
        tape_str = "│" + "│".join(cells) + "│"
        marker_str = " " + " ".join(markers)
        
        return (
            f"Step {self.step_count} | State: {self.current_state}\n"
            f"{tape_str}\n"
            f"{marker_str}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TuringSimulator:
    """Simulator for Turing Machines with execution control.
    
    This simulator provides:
    - Step-by-step execution with full control
    - Batch execution with configurable step limits
    - State visualisation at any point
    - Complete execution history
    
    Attributes:
        machine: The Turing machine being simulated.
        config: The current configuration (None until input is loaded).
        history: List of all configurations during execution.
    
    Example:
        >>> machine = create_unary_increment_machine()
        >>> sim = TuringSimulator(machine)
        >>> sim.load("111")
        >>> sim.run()
        True
        >>> sim.get_output()
        '1111'
    """
    
    def __init__(self, machine: TuringMachine) -> None:
        """Initialise the simulator with a Turing machine.
        
        Args:
            machine: The Turing machine to simulate.
        """
        self.machine = machine
        self.config: Configuration | None = None
        self.history: list[Configuration] = []
        logger.debug("Simulator initialised for machine")
    
    def load(self, input_string: str) -> None:
        """Load an input string onto the tape.
        
        This resets the machine to its initial state with the input
        string written to the tape starting at position 0.
        
        Args:
            input_string: The string to process.
        """
        self.config = Configuration(
            tape={i: c for i, c in enumerate(input_string)},
            head_position=0,
            current_state=self.machine.initial_state,
            step_count=0
        )
        self.history = [self._copy_config()]
        logger.debug("Loaded input: '%s'", input_string)
    
    def _copy_config(self) -> Configuration:
        """Create a deep copy of the current configuration."""
        assert self.config is not None
        return Configuration(
            tape=dict(self.config.tape),
            head_position=self.config.head_position,
            current_state=self.config.current_state,
            step_count=self.config.step_count
        )
    
    def step(self) -> bool:
        """Execute a single step of the machine.
        
        Returns:
            True if the machine can continue, False if it has halted.
        
        Raises:
            RuntimeError: If no input has been loaded.
        """
        if self.config is None:
            raise RuntimeError("No input loaded. Call load() first.")
        
        # Check for terminal states
        if self.config.current_state in (
            self.machine.accept_state, 
            self.machine.reject_state
        ):
            return False
        
        # Read the current symbol
        symbol = self.config.read(self.machine.blank_symbol)
        
        # Look up the transition
        key = (self.config.current_state, symbol)
        if key not in self.machine.transitions:
            # No transition = implicit rejection
            logger.debug(
                "No transition for (%s, %s) - rejecting",
                self.config.current_state, symbol
            )
            self.config.current_state = self.machine.reject_state
            return False
        
        transition = self.machine.transitions[key]
        
        # Execute the transition
        self.config.write(transition.write_symbol)
        self.config.move(transition.direction)
        self.config.current_state = transition.next_state
        self.config.step_count += 1
        
        # Save to history
        self.history.append(self._copy_config())
        
        logger.debug(
            "Step %d: (%s, %s) → (%s, %s, %s)",
            self.config.step_count,
            key[0], key[1],
            transition.next_state,
            transition.write_symbol,
            transition.direction.name
        )
        
        return True
    
    def run(self, max_steps: int = 10000, verbose: bool = False) -> bool:
        """Run the machine until it halts or reaches the step limit.
        
        Args:
            max_steps: Maximum number of steps to execute.
            verbose: Whether to print each configuration.
        
        Returns:
            True if the machine accepted, False if it rejected.
        
        Raises:
            RuntimeError: If no input has been loaded.
        """
        if self.config is None:
            raise RuntimeError("No input loaded. Call load() first.")
        
        while self.config.step_count < max_steps:
            if verbose:
                print(self.config.to_string())
                print()
            
            if not self.step():
                break
        
        if verbose:
            print(self.config.to_string())
            print(f"\n{'='*50}")
            print(f"Result: {'ACCEPTED' if self.accepted else 'REJECTED'}")
            print(f"Total steps: {self.config.step_count}")
        
        logger.info(
            "Execution complete: %s in %d steps",
            "ACCEPTED" if self.accepted else "REJECTED",
            self.config.step_count
        )
        
        return self.accepted
    
    def run_generator(
        self, max_steps: int = 10000
    ) -> Iterator[Configuration]:
        """Generate configurations during execution.
        
        This generator yields each configuration as the machine runs,
        allowing for custom visualisation or analysis.
        
        Args:
            max_steps: Maximum number of steps to execute.
        
        Yields:
            Configuration objects for each step.
        
        Raises:
            RuntimeError: If no input has been loaded.
        """
        if self.config is None:
            raise RuntimeError("No input loaded. Call load() first.")
        
        yield self._copy_config()
        
        while self.config.step_count < max_steps:
            if not self.step():
                break
            yield self._copy_config()
    
    @property
    def accepted(self) -> bool:
        """Check whether the machine accepted the input."""
        return (
            self.config is not None and 
            self.config.current_state == self.machine.accept_state
        )
    
    @property
    def rejected(self) -> bool:
        """Check whether the machine rejected the input."""
        return (
            self.config is not None and 
            self.config.current_state == self.machine.reject_state
        )
    
    @property
    def halted(self) -> bool:
        """Check whether the machine has halted."""
        return self.accepted or self.rejected
    
    def get_output(self) -> str:
        """Return the tape contents as a string (without edge blanks).
        
        Returns:
            The non-blank portion of the tape as a string.
        """
        if self.config is None:
            return ""
        
        if not self.config.tape:
            return ""
        
        min_pos = min(self.config.tape.keys())
        max_pos = max(self.config.tape.keys())
        
        result = []
        for pos in range(min_pos, max_pos + 1):
            symbol = self.config.tape.get(pos, self.machine.blank_symbol)
            if symbol != self.machine.blank_symbol:
                result.append(symbol)
        
        return "".join(result)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EXAMPLE MACHINES
# ═══════════════════════════════════════════════════════════════════════════════

def create_unary_increment_machine() -> TuringMachine:
    """Create a Turing machine that increments a number in unary representation.
    
    Unary representation: n is represented as n occurrences of '1'.
    For example: 0 = "", 1 = "1", 3 = "111", 5 = "11111"
    
    Operation: Add one '1' to the end of the number.
    
    State diagram::
    
        ┌─────┐  1/1,R   ┌─────┐  □/1,S   ┌────────┐
        │ q0  │ ───────▶ │ q0  │ ───────▶ │ accept │
        └─────┘          └─────┘          └────────┘
             ▲              │
             └──────────────┘
    
    Returns:
        A TuringMachine that increments unary numbers.
    
    Example:
        >>> machine = create_unary_increment_machine()
        >>> sim = TuringSimulator(machine)
        >>> sim.load("111")
        >>> sim.run()
        True
        >>> sim.get_output()
        '1111'
    """
    return TuringMachine(
        transitions={
            # q0: Move right over all 1s
            ("q0", "1"): Transition("q0", "1", Direction.RIGHT),
            # q0: When blank found, write 1 and accept
            ("q0", "□"): Transition("accept", "1", Direction.STAY),
        },
        initial_state="q0",
        accept_state="accept"
    )


def create_unary_addition_machine() -> TuringMachine:
    """Create a Turing machine that adds two numbers in unary representation.
    
    Input format: "111+11" (representing 3 + 2)
    Output format: "11111" (representing 5)
    
    Algorithm:
    1. Find the '+' symbol
    2. Replace '+' with blank
    3. Move left and erase one '1' (compensating for removed separator)
    4. Move to the result
    
    Returns:
        A TuringMachine that performs unary addition.
    
    Example:
        >>> machine = create_unary_addition_machine()
        >>> sim = TuringSimulator(machine)
        >>> sim.load("111+11")  # 3 + 2
        >>> sim.run()
        True
        >>> len(sim.get_output())  # Result should be 5
        5
    """
    return TuringMachine(
        transitions={
            # q_find_plus: Search for the '+' symbol
            ("q_find_plus", "1"): Transition(
                "q_find_plus", "1", Direction.RIGHT
            ),
            ("q_find_plus", "+"): Transition(
                "q_erase_left", "□", Direction.LEFT
            ),
            
            # q_erase_left: Move left and erase one '1'
            ("q_erase_left", "1"): Transition(
                "q_go_right", "□", Direction.RIGHT
            ),
            
            # q_go_right: Move right over blanks to find remaining '1's
            ("q_go_right", "□"): Transition(
                "q_go_right", "□", Direction.RIGHT
            ),
            ("q_go_right", "1"): Transition(
                "q_compact", "1", Direction.LEFT
            ),
            
            # q_compact: Compact the result
            ("q_compact", "□"): Transition(
                "q_find_one", "□", Direction.LEFT
            ),
            
            # q_find_one: Find the next '1' to move
            ("q_find_one", "□"): Transition(
                "q_find_one", "□", Direction.LEFT
            ),
            ("q_find_one", "1"): Transition(
                "accept", "1", Direction.STAY
            ),
        },
        initial_state="q_find_plus",
        accept_state="accept"
    )


def create_palindrome_checker() -> TuringMachine:
    """Create a Turing machine that checks if a binary string is a palindrome.
    
    Input: A string of '0's and '1's
    Output: Accept if palindrome, reject otherwise
    
    Algorithm:
    1. Read and erase the first character
    2. Move to the last character
    3. Check if they match
    4. Erase the last character
    5. Repeat until the string is empty or one character remains
    
    State diagram (simplified)::
    
        ┌────┐ 0/□  ┌─────────┐ □/□  ┌────────┐
        │ q0 │─────▶│ q_find0 │─────▶│ accept │
        └────┘      └─────────┘      └────────┘
           │ 1/□        │ 0/□
           ▼            ▼
        ┌─────────┐  ┌────────┐
        │ q_find1 │  │ reject │
        └─────────┘  └────────┘
    
    Returns:
        A TuringMachine that checks for palindromes.
    
    Example:
        >>> machine = create_palindrome_checker()
        >>> sim = TuringSimulator(machine)
        >>> sim.load("10101")
        >>> sim.run()
        True
        >>> sim.load("10100")
        >>> sim.run()
        False
    """
    return TuringMachine(
        transitions={
            # q0: Read the first character
            ("q0", "0"): Transition(
                "q_seek_end_for_0", "□", Direction.RIGHT
            ),
            ("q0", "1"): Transition(
                "q_seek_end_for_1", "□", Direction.RIGHT
            ),
            ("q0", "□"): Transition(
                "accept", "□", Direction.STAY
            ),  # Empty string = palindrome
            
            # q_seek_end_for_0: Read '0', seek the last character
            ("q_seek_end_for_0", "0"): Transition(
                "q_seek_end_for_0", "0", Direction.RIGHT
            ),
            ("q_seek_end_for_0", "1"): Transition(
                "q_seek_end_for_0", "1", Direction.RIGHT
            ),
            ("q_seek_end_for_0", "□"): Transition(
                "q_check_0", "□", Direction.LEFT
            ),
            
            # q_check_0: Verify the last character is '0'
            ("q_check_0", "0"): Transition(
                "q_return", "□", Direction.LEFT
            ),
            ("q_check_0", "1"): Transition(
                "reject", "1", Direction.STAY
            ),
            ("q_check_0", "□"): Transition(
                "accept", "□", Direction.STAY
            ),  # Single character
            
            # q_seek_end_for_1: Read '1', seek the last character
            ("q_seek_end_for_1", "0"): Transition(
                "q_seek_end_for_1", "0", Direction.RIGHT
            ),
            ("q_seek_end_for_1", "1"): Transition(
                "q_seek_end_for_1", "1", Direction.RIGHT
            ),
            ("q_seek_end_for_1", "□"): Transition(
                "q_check_1", "□", Direction.LEFT
            ),
            
            # q_check_1: Verify the last character is '1'
            ("q_check_1", "1"): Transition(
                "q_return", "□", Direction.LEFT
            ),
            ("q_check_1", "0"): Transition(
                "reject", "0", Direction.STAY
            ),
            ("q_check_1", "□"): Transition(
                "accept", "□", Direction.STAY
            ),
            
            # q_return: Return to the beginning of the remaining string
            ("q_return", "0"): Transition(
                "q_return", "0", Direction.LEFT
            ),
            ("q_return", "1"): Transition(
                "q_return", "1", Direction.LEFT
            ),
            ("q_return", "□"): Transition(
                "q0", "□", Direction.RIGHT
            ),
        },
        initial_state="q0",
        accept_state="accept",
        reject_state="reject"
    )


def create_binary_successor_machine() -> TuringMachine:
    """Create a Turing machine that computes the successor of a binary number.
    
    Input: A binary number (e.g., "101" for 5)
    Output: The successor (e.g., "110" for 6)
    
    Algorithm:
    1. Move to the rightmost digit
    2. If '0', change to '1' and accept
    3. If '1', change to '0' and propagate carry left
    4. If blank with carry, write '1'
    
    Returns:
        A TuringMachine that computes binary successors.
    
    Example:
        >>> machine = create_binary_successor_machine()
        >>> sim = TuringSimulator(machine)
        >>> sim.load("101")  # 5 in binary
        >>> sim.run()
        True
        >>> sim.get_output()  # Should be 6 = "110"
        '110'
    """
    return TuringMachine(
        transitions={
            # q_seek_end: Move to the rightmost digit
            ("q_seek_end", "0"): Transition(
                "q_seek_end", "0", Direction.RIGHT
            ),
            ("q_seek_end", "1"): Transition(
                "q_seek_end", "1", Direction.RIGHT
            ),
            ("q_seek_end", "□"): Transition(
                "q_increment", "□", Direction.LEFT
            ),
            
            # q_increment: Process digits from right to left
            ("q_increment", "0"): Transition(
                "accept", "1", Direction.STAY
            ),  # No carry needed
            ("q_increment", "1"): Transition(
                "q_increment", "0", Direction.LEFT
            ),  # Carry propagates
            ("q_increment", "□"): Transition(
                "accept", "1", Direction.STAY
            ),  # Overflow: add new digit
        },
        initial_state="q_seek_end",
        accept_state="accept"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_increment() -> None:
    """Demonstrate unary increment operation."""
    logger.info("Running unary increment demonstration")
    print("=" * 60)
    print("DEMO: Increment in Unary Representation")
    print("=" * 60)
    
    machine = create_unary_increment_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = ["", "1", "111", "11111"]
    
    for input_str in test_cases:
        simulator.load(input_str if input_str else "□")
        simulator.run()
        output = simulator.get_output()
        
        input_val = len(input_str)
        output_val = len(output)
        
        print(f"  {input_val} ({input_str or '∅'}) + 1 = {output_val} ({output})")
    
    print()


def demo_addition() -> None:
    """Demonstrate unary addition operation."""
    logger.info("Running unary addition demonstration")
    print("=" * 60)
    print("DEMO: Addition in Unary Representation")
    print("=" * 60)
    
    machine = create_unary_addition_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("1+1", 2),      # 1 + 1 = 2
        ("11+1", 3),     # 2 + 1 = 3
        ("111+11", 5),   # 3 + 2 = 5
    ]
    
    for input_str, expected in test_cases:
        simulator.load(input_str)
        simulator.run()
        output = simulator.get_output()
        actual = len(output)
        
        status = "✓" if actual == expected else "✗"
        parts = input_str.split("+")
        
        print(
            f"  {status} {len(parts[0])} + {len(parts[1])} = {actual} "
            f"(expected {expected})"
        )
    
    print()


def demo_palindrome() -> None:
    """Demonstrate palindrome checking."""
    logger.info("Running palindrome checking demonstration")
    print("=" * 60)
    print("DEMO: Palindrome Verification")
    print("=" * 60)
    
    machine = create_palindrome_checker()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("", True),
        ("0", True),
        ("1", True),
        ("00", True),
        ("11", True),
        ("01", False),
        ("10", False),
        ("010", True),
        ("0110", True),
        ("0101", False),
        ("10101", True),
        ("110011", True),
    ]
    
    for input_str, expected in test_cases:
        simulator.load(input_str if input_str else "□")
        result = simulator.run()
        
        status = "✓" if result == expected else "✗"
        result_str = "palindrome" if result else "not palindrome"
        
        print(f"  {status} '{input_str or '∅'}' → {result_str}")
    
    print()


def demo_binary_successor() -> None:
    """Demonstrate binary successor operation."""
    logger.info("Running binary successor demonstration")
    print("=" * 60)
    print("DEMO: Binary Successor (n → n+1)")
    print("=" * 60)
    
    machine = create_binary_successor_machine()
    simulator = TuringSimulator(machine)
    
    test_cases = [
        ("0", "1"),       # 0 → 1
        ("1", "10"),      # 1 → 2
        ("10", "11"),     # 2 → 3
        ("11", "100"),    # 3 → 4
        ("101", "110"),   # 5 → 6
        ("111", "1000"),  # 7 → 8
        ("1011", "1100"), # 11 → 12
    ]
    
    for input_str, expected in test_cases:
        simulator.load(input_str)
        simulator.run()
        output = simulator.get_output()
        
        status = "✓" if output == expected else "✗"
        input_val = int(input_str, 2)
        output_val = int(output, 2) if output else 0
        
        print(
            f"  {status} {input_str} ({input_val}) → "
            f"{output} ({output_val})"
        )
    
    print()


def demo_step_by_step() -> None:
    """Demonstrate step-by-step execution."""
    logger.info("Running step-by-step demonstration")
    print("=" * 60)
    print("DEMO: Step-by-Step Execution - Palindrome Check '010'")
    print("=" * 60)
    print()
    
    machine = create_palindrome_checker()
    simulator = TuringSimulator(machine)
    simulator.load("010")
    simulator.run(verbose=True)
    
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 1: Binary Increment                                                  ║
║                                                                               ║
║ Implement a Turing machine that increments a binary number.                   ║
║                                                                               ║
║ Examples:                                                                     ║
║   "0"    → "1"                                                               ║
║   "1"    → "10"                                                              ║
║   "10"   → "11"                                                              ║
║   "11"   → "100"                                                             ║
║   "111"  → "1000"                                                            ║
║   "1011" → "1100"                                                            ║
║                                                                               ║
║ Hint: Start from the right, propagate the carry to the left.                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def create_binary_increment_machine() -> TuringMachine:
    """EXERCISE: Complete the implementation.
    
    Suggested algorithm:
    1. Move to the end of the number
    2. If '0', replace with '1' and accept
    3. If '1', replace with '0' and move left (carry)
    4. Repeat until no carry or reach the beginning of the tape
    5. If blank with carry, write '1'
    
    Returns:
        A TuringMachine that increments binary numbers.
    """
    # TODO: Implement transitions
    return TuringMachine(
        transitions={
            # Complete here...
            ("q_start", "□"): Transition(
                "accept", "□", Direction.STAY
            ),  # Placeholder
        },
        initial_state="q_start",
        accept_state="accept"
    )


"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 2: Balanced Parentheses                                              ║
║                                                                               ║
║ Implement a Turing machine that checks if parentheses are balanced.           ║
║                                                                               ║
║ Examples:                                                                     ║
║   "()"     → accept                                                          ║
║   "(())"   → accept                                                          ║
║   "()()"   → accept                                                          ║
║   "(()"    → reject                                                          ║
║   "())"    → reject                                                          ║
║   "(()())" → accept                                                          ║
║                                                                               ║
║ Hint: Find the first ')', replace with 'X', find the matching '(',           ║
║       replace with 'X', repeat. Accept if all are 'X'.                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def create_balanced_parentheses_machine() -> TuringMachine:
    """EXERCISE: Complete the implementation.
    
    Returns:
        A TuringMachine that checks for balanced parentheses.
    """
    # TODO: Implement transitions
    return TuringMachine(
        transitions={
            ("q_start", "□"): Transition("accept", "□", Direction.STAY),
        },
        initial_state="q_start",
        accept_state="accept",
        reject_state="reject"
    )


"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 3: String Reversal                                                   ║
║                                                                               ║
║ Implement a Turing machine that reverses a binary string.                     ║
║                                                                               ║
║ Examples:                                                                     ║
║   "01"   → "10"                                                              ║
║   "110"  → "011"                                                             ║
║   "1010" → "0101"                                                            ║
║                                                                               ║
║ Hint: Use a marker symbol to separate input from output. Copy characters     ║
║       from right to left one at a time.                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def create_string_reversal_machine() -> TuringMachine:
    """EXERCISE: Complete the implementation.
    
    Returns:
        A TuringMachine that reverses binary strings.
    """
    # TODO: Implement transitions
    return TuringMachine(
        transitions={
            ("q_start", "□"): Transition("accept", "□", Direction.STAY),
        },
        initial_state="q_start",
        accept_state="accept"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def visualise_machine(machine: TuringMachine) -> str:
    """Create a text visualisation of a Turing machine's transition table.
    
    Args:
        machine: The Turing machine to visualise.
    
    Returns:
        A formatted string showing the transition table.
    """
    lines = ["Transition Table:", "=" * 60]
    lines.append(
        f"{'State':<15} {'Symbol':<8} → "
        f"{'Next':<15} {'Write':<8} {'Move':<6}"
    )
    lines.append("-" * 60)
    
    for (state, symbol), trans in sorted(machine.transitions.items()):
        lines.append(
            f"{state:<15} {symbol:<8} → "
            f"{trans.next_state:<15} {trans.write_symbol:<8} "
            f"{trans.direction.name:<6}"
        )
    
    lines.append("=" * 60)
    lines.append(f"Initial state: {machine.initial_state}")
    lines.append(f"Accept state: {machine.accept_state}")
    lines.append(f"Reject state: {machine.reject_state}")
    
    return "\n".join(lines)


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_increment()
    demo_addition()
    demo_palindrome()
    demo_binary_successor()
    demo_step_by_step()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Turing Machine Simulator - Week 1, Lab 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_1_01_turing_machine.py --demo
  python lab_1_01_turing_machine.py --run palindrome "10101"
  python lab_1_01_turing_machine.py --run increment "111"
  python lab_1_01_turing_machine.py --run successor "101" -v

Available machines: increment, addition, palindrome, successor
        """
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations"
    )
    parser.add_argument(
        "--run",
        nargs=2,
        metavar=("MACHINE", "INPUT"),
        help="Run a specific machine on input"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show step-by-step execution"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        print("\n" + "═" * 60)
        print("  WEEK 1, LAB 1: TURING MACHINE SIMULATOR")
        print("═" * 60 + "\n")
        run_all_demos()
        print("=" * 60)
        print("Exercises to complete in code:")
        print("  1. create_binary_increment_machine()")
        print("  2. create_balanced_parentheses_machine()")
        print("  3. create_string_reversal_machine()")
        print("=" * 60)
    elif args.run:
        machine_name, input_str = args.run
        
        machines = {
            "increment": create_unary_increment_machine,
            "addition": create_unary_addition_machine,
            "palindrome": create_palindrome_checker,
            "successor": create_binary_successor_machine,
        }
        
        if machine_name not in machines:
            print(f"Unknown machine: {machine_name}")
            print(f"Available: {', '.join(machines.keys())}")
            return
        
        machine = machines[machine_name]()
        simulator = TuringSimulator(machine)
        simulator.load(input_str)
        
        print(f"\nRunning {machine_name} on input '{input_str}'")
        print("-" * 40)
        
        result = simulator.run(verbose=args.verbose)
        
        print("-" * 40)
        print(f"Result: {'ACCEPTED' if result else 'REJECTED'}")
        print(f"Output: '{simulator.get_output()}'")
        print(f"Steps: {simulator.config.step_count if simulator.config else 0}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
