# Practice Exercise: Easy 01

## Turing Machine Trace

**Difficulty:** Easy  
**Estimated Time:** 15 minutes  
**Prerequisites:** Understanding of Turing machine components  

---

## Problem

Given the following Turing machine for unary increment:

**States:** Q = {q0, accept}  
**Tape Alphabet:** Γ = {1, □}  
**Initial State:** q0  
**Accept State:** accept  

**Transitions:**
- δ(q0, 1) = (q0, 1, R)
- δ(q0, □) = (accept, 1, S)

**Input:** "111" (representing 3 in unary)

### Task

Trace the execution of this machine step by step. For each step, record:
1. Current state
2. Tape contents (with head position marked by underlining)
3. The transition applied

### Template

```
Step 0: State = q0, Tape = [1̲][1][1][ ]
        Read: 1, Apply: δ(q0, 1) = (q0, 1, R)

Step 1: State = ___, Tape = _______________
        Read: ___, Apply: _______________

(continue for all steps...)
```

---

## Expected Output

After the machine halts:
- Final tape contents: "1111" (representing 4)
- Final state: accept
- Total steps: 4

---

## Solution

<details>
<summary>Click to reveal solution</summary>

```
Step 0: State = q0, Tape = [1̲][1][1][□]
        Read: 1, Apply: δ(q0, 1) = (q0, 1, R)

Step 1: State = q0, Tape = [1][1̲][1][□]
        Read: 1, Apply: δ(q0, 1) = (q0, 1, R)

Step 2: State = q0, Tape = [1][1][1̲][□]
        Read: 1, Apply: δ(q0, 1) = (q0, 1, R)

Step 3: State = q0, Tape = [1][1][1][□̲]
        Read: □, Apply: δ(q0, □) = (accept, 1, S)

Step 4: State = accept, Tape = [1][1][1][1̲]
        HALT - Accept state reached

Final result: "1111" (4 in unary)
Total steps: 4
```

The machine scans right over all 1s, then writes a new 1 at the first blank position, incrementing the number by one.

</details>

---

## Learning Points

1. Turing machines execute one transition per step
2. The head moves based on the transition's direction component
3. The machine halts when it enters an accepting or rejecting state
4. Tracing helps visualise the computation process

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
