# Week 1: Knowledge Check Quiz

## The Epistemology of Computation

**Time Limit:** 20 minutes  
**Total Points:** 50  
**Format:** 6 multiple choice, 4 short answer  

---

## Multiple Choice Questions (30 points)

*Select the single best answer for each question. Each question is worth 5 points.*

---

### Question 1

Which of the following is NOT a component of the formal definition of a Turing machine?

A) A finite set of states Q  
B) A tape alphabet Γ  
C) A stack for storing intermediate values  
D) A transition function δ  

---

### Question 2

In lambda calculus, the expression `(λx.λy.x) a b` reduces to:

A) `b`  
B) `a`  
C) `λy.a`  
D) `a b`  

---

### Question 3

The Church-Turing thesis states that:

A) Turing machines are faster than lambda calculus  
B) Any effectively computable function can be computed by a Turing machine  
C) All programming languages are equivalent  
D) The halting problem is decidable  

---

### Question 4

Which statement about the Halting Problem is correct?

A) It can be solved for any specific program with enough computing power  
B) It is undecidable in general but decidable for simple programs  
C) No algorithm can correctly determine halting for all program-input pairs  
D) Modern AI systems have solved the Halting Problem  

---

### Question 5

In an Abstract Syntax Tree for the expression `2 + 3 * 4`, which operation appears at the root?

A) Multiplication (*)  
B) Addition (+)  
C) The number 2  
D) The number 4  

---

### Question 6

The Church numeral `λf.λx.f (f (f x))` represents which number?

A) 0  
B) 1  
C) 2  
D) 3  

---

## Short Answer Questions (20 points)

*Answer each question in 2-4 sentences. Each question is worth 5 points.*

---

### Question 7

Explain why the blank symbol (□) must be part of the tape alphabet Γ but not part of the input alphabet Σ.

**Your Answer:**

---

### Question 8

Describe what happens during beta reduction in lambda calculus. Give a simple example.

**Your Answer:**

---

### Question 9

What is a closure in the context of programming language implementation? Why is it necessary for implementing functions with free variables?

**Your Answer:**

---

### Question 10

Name one problem in your research domain that could potentially be modelled using a finite state machine. Briefly explain why.

**Your Answer:**

---

## Answer Key

<details>
<summary>Click to reveal answers (for self-assessment only)</summary>

### Multiple Choice

1. **C) A stack for storing intermediate values**  
   Turing machines have an infinite tape but no stack. Pushdown automata have stacks but are less powerful than Turing machines.

2. **B) a**  
   First, (λx.λy.x) a → λy.a (K combinator). Then (λy.a) b → a (y is not used).

3. **B) Any effectively computable function can be computed by a Turing machine**  
   The thesis equates informal "computability" with formal Turing-computability.

4. **C) No algorithm can correctly determine halting for all program-input pairs**  
   Turing proved this using a diagonalisation argument.

5. **B) Addition (+)**  
   Due to operator precedence, multiplication binds tighter, making addition the root operation: +(2, *(3, 4)).

6. **D) 3**  
   The numeral applies f three times to x.

### Short Answer

7. The blank symbol represents empty tape cells and is used by the machine to detect boundaries and write intermediate results. It cannot be in the input alphabet because the input is finite and placed on otherwise blank tape. If blank were an input symbol, we could not distinguish between input and empty tape.

8. Beta reduction substitutes the argument into the function body. For (λx.M) N, we replace all free occurrences of x in M with N. Example: (λx.x + 1) 5 → 5 + 1 → 6.

9. A closure is a function bundled with its lexical environment (the variable bindings from its definition context). It is necessary because when a function references variables from an outer scope, those values must be preserved even after the outer function returns.

10. Answers will vary. Examples: DNA sequence pattern matching (bioinformatics), protocol state tracking (networking), experimental procedure validation (laboratory sciences), user interaction flows (HCI).

</details>

---

## Grading Scale

| Score | Grade | Description |
|-------|-------|-------------|
| 45-50 | A | Excellent understanding |
| 40-44 | B | Good understanding |
| 35-39 | C | Satisfactory understanding |
| 30-34 | D | Needs improvement |
| <30 | F | Requires review of material |

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
