# Practice Exercise: Medium 01

## Designing a Turing Machine

**Difficulty:** Medium  
**Estimated Time:** 25 minutes  
**Prerequisites:** Turing machine implementation lab  

---

## Problem

Design a Turing machine that checks whether a binary string contains an equal number of 0s and 1s.

### Specification

**Input:** A string over {0, 1}  
**Output:** Accept if #0s = #1s, reject otherwise

### Examples

| Input | Expected | Reason |
|-------|----------|--------|
| "" | Accept | 0 zeros, 0 ones |
| "01" | Accept | 1 zero, 1 one |
| "10" | Accept | 1 zero, 1 one |
| "0011" | Accept | 2 zeros, 2 ones |
| "1100" | Accept | 2 zeros, 2 ones |
| "0101" | Accept | 2 zeros, 2 ones |
| "0" | Reject | 1 zero, 0 ones |
| "1" | Reject | 0 zeros, 1 one |
| "001" | Reject | 2 zeros, 1 one |
| "110" | Reject | 1 zero, 2 ones |

---

## Task

1. Define the state set Q
2. Define the tape alphabet Γ
3. Write out all transitions in δ
4. Explain your algorithm in plain English

### Template

```
States: Q = {___}

Tape Alphabet: Γ = {___}

Initial State: q₀ = ___

Accept State: q_accept = ___

Reject State: q_reject = ___

Transitions:
δ(state, symbol) = (next_state, write_symbol, direction)

___

Algorithm Description:
___
```

---

## Hints

<details>
<summary>💡 Hint 1: Matching Strategy</summary>

One approach: repeatedly find one 0 and one 1, mark them as "used" (e.g., with X), until either:
- All symbols are marked → Accept
- Only 0s or only 1s remain unmarked → Reject

</details>

<details>
<summary>💡 Hint 2: State Design</summary>

Consider these states:
- `q_find_0`: Looking for an unmarked 0
- `q_find_1`: Found a 0, now looking for an unmarked 1
- `q_return`: Return to the start to find the next pair
- `q_verify`: Check that all symbols are marked

</details>

---

## Solution

<details>
<summary>Click to reveal solution</summary>

**States:** Q = {q_find_0, q_find_1, q_return, q_verify, accept, reject}

**Tape Alphabet:** Γ = {0, 1, X, □}

**Initial State:** q₀ = q_find_0

**Transitions:**

```
# Looking for a 0 to mark
δ(q_find_0, 0) = (q_find_1, X, R)    # Found 0, mark it, look for 1
δ(q_find_0, 1) = (q_find_0, 1, R)    # Skip 1s
δ(q_find_0, X) = (q_find_0, X, R)    # Skip marked
δ(q_find_0, □) = (q_verify, □, L)    # No more 0s, verify

# Looking for a 1 to match the 0
δ(q_find_1, 0) = (q_find_1, 0, R)    # Skip 0s
δ(q_find_1, 1) = (q_return, X, L)    # Found 1, mark it, return
δ(q_find_1, X) = (q_find_1, X, R)    # Skip marked
δ(q_find_1, □) = (reject, □, S)     # No 1 found, unmatched

# Return to start
δ(q_return, 0) = (q_return, 0, L)
δ(q_return, 1) = (q_return, 1, L)
δ(q_return, X) = (q_return, X, L)
δ(q_return, □) = (q_find_0, □, R)

# Verify all symbols are marked
δ(q_verify, X) = (q_verify, X, L)    # All marked so far
δ(q_verify, 0) = (reject, 0, S)     # Unmarked 0 remains
δ(q_verify, 1) = (reject, 1, S)     # Unmarked 1 remains
δ(q_verify, □) = (accept, □, S)     # All matched
```

**Algorithm Description:**

1. Scan right to find an unmarked 0
2. Mark it with X
3. Continue scanning right to find an unmarked 1
4. Mark it with X
5. Return to the beginning
6. Repeat until no unmarked 0s remain
7. Verify no unmarked symbols remain
8. Accept if all marked, reject if any unmarked

Time complexity: O(n²) where n is input length.

</details>

---

## Extension Challenge

Can you design a more efficient algorithm? Consider whether it is possible to achieve O(n) time complexity on a single-tape Turing machine.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
