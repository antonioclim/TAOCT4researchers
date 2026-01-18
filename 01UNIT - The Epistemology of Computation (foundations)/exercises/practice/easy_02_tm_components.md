# Practice Exercise: Easy 02

## Identifying Turing Machine Components

**Difficulty:** Easy  
**Estimated Time:** 10 minutes  
**Prerequisites:** Lecture notes on Turing machine definition  

---

## Problem

For each of the following descriptions, identify which component of a Turing machine (Q, Σ, Γ, δ, q₀, q_accept, q_reject) is being described.

### Questions

1. The set {0, 1} representing valid input characters for a binary number checker.

2. The rule that says "if in state q2 reading symbol 'a', write 'X', move right, and go to state q3."

3. The state where computation begins before any symbols are read.

4. The set {q_start, q_scan, q_match, q_done, q_fail} in a parenthesis matcher.

5. The set {0, 1, X, □} including the blank symbol and a marker.

6. The state that, when entered, causes the machine to halt and report "yes."

7. The state that, when entered, causes the machine to halt and report "no."

---

## Template Answer

```
1. Component: ___
   Explanation: ___

2. Component: ___
   Explanation: ___

(continue for all questions...)
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

1. **Component:** Σ (Sigma - Input Alphabet)  
   **Explanation:** The input alphabet contains symbols that can appear in the input string. It does not include the blank symbol.

2. **Component:** δ (Delta - Transition Function)  
   **Explanation:** The transition function δ: Q × Γ → Q × Γ × {L, R} defines the machine's behaviour.

3. **Component:** q₀ (Initial State)  
   **Explanation:** The initial state is where the machine begins execution.

4. **Component:** Q (State Set)  
   **Explanation:** Q is the finite set of all states the machine can be in.

5. **Component:** Γ (Gamma - Tape Alphabet)  
   **Explanation:** The tape alphabet includes all symbols that can appear on the tape, including the input alphabet Σ, the blank symbol and any auxiliary markers.

6. **Component:** q_accept (Accept State)  
   **Explanation:** Entering the accept state causes the machine to halt and accept the input.

7. **Component:** q_reject (Reject State)  
   **Explanation:** Entering the reject state causes the machine to halt and reject the input.

</details>

---

## Learning Points

1. Σ ⊂ Γ (the input alphabet is a subset of the tape alphabet)
2. The blank symbol □ ∈ Γ but □ ∉ Σ
3. q_accept ≠ q_reject (they must be distinct states)
4. The transition function δ is the "program" of the machine

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
