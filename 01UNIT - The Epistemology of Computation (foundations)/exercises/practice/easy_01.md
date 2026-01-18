# Practice Exercise: Easy 01

## Turing Machine Trace

**Difficulty:** Easy  
**Estimated Time:** 15 minutes  
**Topics:** Turing machines, execution tracing

---

## Problem

Given the following Turing machine for unary increment:

```
M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject) where:
- Q = {q0, accept}
- Σ = {1}
- Γ = {1, □}
- q₀ = q0
- q_accept = accept

δ (transition function):
  (q0, 1) → (q0, 1, R)
  (q0, □) → (accept, 1, S)
```

**Task:** Trace the execution of this machine on input "111" (representing 3).

---

## Requirements

1. Write out each configuration (tape contents, head position, state)
2. Show the final tape contents
3. Determine if the machine accepts or rejects
4. State what the output represents

---

## Template

Fill in the following trace:

```
Step 0: State = _____, Tape = [1][1][1], Head at position ___
Step 1: State = _____, Tape = [_][_][_], Head at position ___
Step 2: State = _____, Tape = [_][_][_], Head at position ___
Step 3: State = _____, Tape = [_][_][_][_], Head at position ___
Step 4: State = _____, Tape = [_][_][_][_], Head at position ___

Final result: _________
Output value: ___
```

---

## Verification

Run your trace against the simulator:

```python
from lab.lab_1_01_turing_machine import (
    create_unary_increment_machine,
    TuringSimulator
)

machine = create_unary_increment_machine()
sim = TuringSimulator(machine)
sim.load("111")
sim.run(verbose=True)
print(f"Output: {sim.get_output()}")
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

```
Step 0: State = q0, Tape = [1][1][1], Head at position 0
Step 1: State = q0, Tape = [1][1][1], Head at position 1
Step 2: State = q0, Tape = [1][1][1], Head at position 2
Step 3: State = q0, Tape = [1][1][1], Head at position 3
Step 4: State = accept, Tape = [1][1][1][1], Head at position 3

Final result: ACCEPTED
Output value: 4 (represented as "1111")
```

The machine reads each '1', moves right, and when it hits the blank, writes a '1' and accepts. This effectively increments the unary number by 1.

</details>

---

© 2025 Antonio Clim. All rights reserved.
