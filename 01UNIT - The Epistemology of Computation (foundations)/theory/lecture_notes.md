# Week 1: Lecture Notes

## The Epistemology of Computation

> *"We may compare a man in the process of computing a real number to a machine which is only capable of a finite number of conditions."*  
> — Alan Turing, 1936

---

## Introduction: What Does It Mean to Compute?

The question of what constitutes computation might seem purely academic, yet it lies at the heart of every algorithm we write, every simulation we run and every data analysis we perform. Before we can think computationally, we must understand what computation actually is.

This week, we trace the intellectual history of computation from its philosophical origins through to the practical tools we use today. We shall see how abstract mathematical models from the 1930s directly inform the structure of modern programming languages and, by extension, how we approach research problems computationally.

---

## 1. Historical Context: The Crisis in Mathematics

### 1.1 Hilbert's Programme

In 1900, David Hilbert presented 23 unsolved problems that would shape twentieth-century mathematics. His programme sought to formalise all of mathematics on a solid axiomatic foundation. Three key properties were sought:

1. **Completeness**: Every true mathematical statement can be proved
2. **Consistency**: No contradictions can be derived
3. **Decidability**: An algorithm exists to determine the truth of any statement

The third property, known as the *Entscheidungsproblem* (decision problem), asked whether there exists a mechanical procedure to determine if any given mathematical statement is provable.

### 1.2 Gödel's Incompleteness Theorems

In 1931, Kurt Gödel shattered hopes for completeness by proving that any sufficiently powerful formal system contains true statements that cannot be proved within that system. This was a philosophical earthquake, but it left the decidability question open.

### 1.3 Enter Turing

In 1936, Alan Turing, then a 24-year-old Cambridge graduate student, published "On Computable Numbers, with an Application to the Entscheidungsproblem." His approach was revolutionary: rather than working within existing mathematical formalisms, he asked what it means for a human to compute.

Turing imagined a human "computer" (the original meaning of the word) working with pencil and paper, following explicit instructions. He then abstracted this process into a mathematical model — the Turing machine.

---

## 2. The Turing Machine

### 2.1 Informal Description

A Turing machine consists of:

- An infinitely long tape divided into cells, each containing a symbol
- A head that can read and write symbols, moving left or right
- A finite set of states that govern behaviour
- A transition function that determines actions based on current state and symbol

The machine operates by reading the symbol under the head, consulting the transition function to determine what to write, which direction to move and which state to enter next. This process continues until the machine enters a halting state.

### 2.2 Formal Definition

Formally, a Turing machine is a 7-tuple M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject) where:

- **Q** is a finite set of states
- **Σ** is the input alphabet (not containing the blank symbol)
- **Γ** is the tape alphabet (Σ ⊂ Γ, includes blank symbol □)
- **δ: Q × Γ → Q × Γ × {L, R}** is the transition function
- **q₀ ∈ Q** is the initial state
- **q_accept ∈ Q** is the accepting state
- **q_reject ∈ Q** is the rejecting state (q_accept ≠ q_reject)

### 2.3 Configuration and Computation

A configuration captures the complete state of a computation at any moment: the tape contents, head position and current state. A computation is a sequence of configurations, each following from the previous via the transition function.

We say a machine **accepts** an input if it eventually reaches q_accept, **rejects** if it reaches q_reject, and **loops** if it never halts. The language recognised by a machine is the set of all inputs it accepts.

### 2.4 Example: Unary Addition

Consider adding two unary numbers separated by a plus sign: "111+11" should produce "11111" (3 + 2 = 5).

The algorithm:
1. Scan right to find the "+"
2. Replace "+" with "1"
3. Scan right to find the rightmost "1"
4. Erase it (replace with blank)
5. Halt

This simple example illustrates how complex operations emerge from elementary read-write-move operations.

---

## 3. Lambda Calculus: An Alternative Foundation

### 3.1 Church's Approach

Simultaneously with Turing, Alonzo Church developed lambda calculus, a formal system based on function abstraction and application. Where Turing's model is imperative (do this, then that), Church's is functional (apply this function to that argument).

### 3.2 Core Concepts

Lambda calculus has only three constructs:

1. **Variables**: x, y, z, ...
2. **Abstraction**: λx.M (a function with parameter x and body M)
3. **Application**: M N (applying function M to argument N)

Despite this minimalism, lambda calculus can express any computable function. Numbers, booleans and data structures all emerge from function application patterns.

### 3.3 Church Numerals

Natural numbers are encoded as:

- 0 = λf.λx.x (apply f zero times to x)
- 1 = λf.λx.f x (apply f once)
- 2 = λf.λx.f (f x) (apply f twice)
- n = λf.λx.fⁿ x (apply f n times)

Addition becomes: λm.λn.λf.λx.m f (n f x)

This encoding demonstrates how computation can arise from pure function manipulation, without explicit state or memory.

---

## 4. The Church-Turing Thesis

### 4.1 Equivalence of Models

A remarkable fact: Turing machines and lambda calculus compute exactly the same class of functions. Any function computable by one can be computed by the other. This equivalence extends to every other reasonable model of computation ever proposed: recursive functions, Post systems, register machines and cellular automata.

### 4.2 The Thesis

The Church-Turing thesis states that any function computable by an "effective procedure" — an algorithm in the intuitive sense — is computable by a Turing machine. This is not a theorem (it cannot be proved because "effective procedure" is informal) but a definition: we define computability as Turing-computability.

### 4.3 Implications for Research

The thesis has profound implications:

1. **Universality**: Any general-purpose computer can simulate any other
2. **Limits**: Problems uncomputable on Turing machines are uncomputable everywhere
3. **Abstraction**: We can reason about computability without specifying implementation

For researchers, this means the language or platform you choose does not affect what is computable — only how efficiently it can be computed.

---

## 5. The Halting Problem and Limits of Computation

### 5.1 The Problem

Given a Turing machine M and input w, determine whether M halts on w.

### 5.2 Undecidability

Turing proved that no algorithm can solve the Halting Problem in general. The proof uses diagonalisation:

Suppose H(M, w) decides whether M halts on w. Construct D that, given M:
1. Runs H(M, M)
2. If H says "halts", loop forever
3. If H says "loops", halt

What does D(D) do? If it halts, H(D, D) said "loops", contradiction. If it loops, H(D, D) said "halts", contradiction.

### 5.3 Practical Implications

The undecidability of halting means we cannot:

- Write a perfect virus scanner (detecting all malicious code)
- Create a universal debugger (finding all bugs automatically)
- Build a complete verification system (proving all programs correct)

This does not mean these tasks are hopeless — it means complete automation is impossible, and human judgment remains essential.

---

## 6. From Theory to Practice: Interpreters and Compilers

### 6.1 The Connection

Modern programming languages are practical implementations of these theoretical models. Imperative languages (C, Python) descend from Turing's stateful computation; functional languages (Haskell, ML) from Church's lambda calculus.

### 6.2 Abstract Syntax Trees

When we write code, we express computation as text. The first step of execution is parsing this text into an abstract syntax tree (AST), a hierarchical representation of the program's structure.

For the expression "3 + 4 * 2":

```
    BinOp(+)
   /        \
Num(3)    BinOp(*)
          /      \
       Num(4)   Num(2)
```

The tree captures precedence (multiplication before addition) and structure without the surface syntax.

### 6.3 Evaluation

An interpreter walks the AST, computing values recursively:

```python
def evaluate(node):
    match node:
        case Num(n):
            return n
        case BinOp(left, "+", right):
            return evaluate(left) + evaluate(right)
        case BinOp(left, "*", right):
            return evaluate(left) * evaluate(right)
```

This pattern — recursive descent over a tree structure — is fundamental to language implementation and appears throughout computer science.

### 6.4 Variables and Environments

To handle variables, we introduce environments: mappings from names to values.

```python
def evaluate(node, env):
    match node:
        case Var(name):
            return env[name]
        case Let(name, value, body):
            new_env = env | {name: evaluate(value, env)}
            return evaluate(body, new_env)
```

This mirrors lambda calculus: variable binding creates a new environment.

---

## 7. Connecting to Research

### 7.1 Finite State Machines in Bioinformatics

DNA sequence analysis frequently uses finite automata (simplified Turing machines with no tape). Pattern matching algorithms like those in BLAST use automata to efficiently search genomic databases.

### 7.2 Parsers in Computational Linguistics

Natural language processing relies heavily on formal grammar theory. Context-free grammars, pushdown automata and parsing algorithms from theoretical computer science underpin tools like spaCy and Stanford NLP.

### 7.3 Cellular Automata in Physics

Stephen Wolfram's "A New Kind of Science" explores cellular automata as models of physical processes. These are essentially Turing machines with parallel, spatially distributed computation.

### 7.4 Neural Computation

The question of whether neural networks are Turing-complete — capable of universal computation — connects deep learning to foundational theory. Understanding computational limits helps us reason about what AI can and cannot achieve.

---

## 8. Key Takeaways

1. **Computation has a precise definition**: Turing machines formalise what it means to compute
2. **Multiple equivalent models exist**: Turing machines and lambda calculus compute the same functions
3. **Some problems are unsolvable**: The Halting Problem sets fundamental limits
4. **Theory informs practice**: AST interpreters directly implement theoretical concepts
5. **Research applications abound**: From bioinformatics to physics, computation theory applies everywhere

---

## 9. Looking Ahead: Week 2

Next week, we move from foundational theory to practical software design. The concepts introduced here — state, transitions, hierarchical structure — will reappear as design patterns: State, Strategy and Composite. Understanding computation's foundations prepares us to build well-structured research software.

---

## References

1. Turing, A. M. (1936). On Computable Numbers, with an Application to the Entscheidungsproblem. *Proceedings of the London Mathematical Society*, 42(2), 230-265.

2. Church, A. (1936). An Unsolvable Problem of Elementary Number Theory. *American Journal of Mathematics*, 58(2), 345-363.

3. Sipser, M. (2012). *Introduction to the Theory of Computation* (3rd ed.). Cengage Learning.

4. Abelson, H., & Sussman, G. J. (1996). *Structure and Interpretation of Computer Programs* (2nd ed.). MIT Press.

5. Hopcroft, J. E., Motwani, R., & Ullman, J. D. (2006). *Introduction to Automata Theory, Languages, and Computation* (3rd ed.). Pearson.

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
