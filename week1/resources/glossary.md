# Week 1: Glossary

## The Epistemology of Computation — Key Terms

---

## A

### Abstract Syntax Tree (AST)
A tree representation of the hierarchical structure of source code. Each node represents a construct (e.g., operation, literal, variable). ASTs eliminate syntactic details like parentheses, making the logical structure explicit.

### Abstraction (Lambda)
In lambda calculus, an expression of the form λx.M that defines a function with parameter x and body M. The lambda symbol λ is read as "lambda" or "function of."

### Accept State
A designated state in a Turing machine that, when entered, causes the machine to halt and indicate that the input is accepted (member of the language).

### Alpha Conversion
Renaming bound variables in a lambda expression to avoid name conflicts during substitution. For example, λx.x can be alpha-converted to λy.y without changing its meaning.

### Application
In lambda calculus, the operation of applying a function to an argument, written as M N (function M applied to argument N).

---

## B

### Beta Reduction
The fundamental computation rule in lambda calculus: (λx.M) N → M[x := N]. Substitutes the argument N for all free occurrences of x in M.

### Blank Symbol
A special symbol (often written □ or B) in a Turing machine's tape alphabet representing an empty cell. Not part of the input alphabet.

### Bound Variable
A variable that appears within the scope of a lambda abstraction that binds it. In λx.x + y, x is bound while y is free.

---

## C

### Church Encoding
A method of representing data (numbers, booleans, pairs) as lambda expressions. Named after Alonzo Church.

### Church Numeral
A representation of natural numbers in lambda calculus where n is encoded as λf.λx.fⁿ(x) — a function that applies f n times to x.

### Church-Turing Thesis
The hypothesis that any function computable by an "effective procedure" (intuitive algorithm) is computable by a Turing machine. Not a theorem but a definition of computability.

### Closure
A function bundled with its lexical environment (the variable bindings from where it was defined). Necessary for functions that reference variables from outer scopes.

### Combinator
A lambda expression with no free variables. Examples include I = λx.x, K = λx.λy.x, and S = λx.λy.λz.x z (y z).

### Computability
The property of a function or problem that can be solved by an algorithm (equivalently, by a Turing machine).

### Configuration
The complete instantaneous description of a Turing machine at a point in time: the tape contents, head position and current state.

---

## D

### Decidable
A language L is decidable if there exists a Turing machine that halts on all inputs, accepting those in L and rejecting those not in L.

### Diagonalisation
A proof technique (used by Cantor and Turing) that constructs a counterexample by differing from every element of an enumeration in at least one position.

---

## E

### Entscheidungsproblem
German for "decision problem." The question of whether an algorithm exists to determine if any mathematical statement is provable. Turing proved it unsolvable.

### Environment
In interpreter implementation, a mapping from variable names to their values. Updated when entering new scopes (let bindings, function calls).

### Evaluation
The process of computing the value of an expression by recursively processing its AST.

---

## F

### Fixed Point
A value x such that f(x) = x. In lambda calculus, Y f is a fixed point of f, meaning Y f = f (Y f).

### Free Variable
A variable that is not bound by any enclosing lambda. In λx.x + y, y is free.

---

## G

### Grammar
A formal specification of the syntax of a language, defining which strings are valid. Context-free grammars are commonly used for programming languages.

---

## H

### Halting Problem
The problem of determining, given a program P and input I, whether P halts on I. Proven undecidable by Turing in 1936.

### Head (Turing Machine)
The component of a Turing machine that reads and writes symbols on the tape. Moves left or right (or stays) after each transition.

---

## I

### Input Alphabet
The set Σ of symbols that can appear in the input string to a Turing machine. Does not include the blank symbol.

### Interpreter
A program that executes source code directly by traversing and evaluating its AST, rather than compiling to machine code.

---

## K

### K Combinator
The combinator K = λx.λy.x that returns its first argument, ignoring the second. Equivalent to Church TRUE.

---

## L

### Lambda Calculus
A formal system developed by Alonzo Church based on function abstraction and application. Equivalent in power to Turing machines.

### Language (Formal)
A set of strings over an alphabet. A Turing machine "recognises" a language by accepting exactly the strings in that set.

### Lexer (Lexical Analyser)
The component of an interpreter or compiler that converts a stream of characters into a stream of tokens.

---

## N

### Normal Form
A lambda expression that cannot be further reduced (contains no redexes). Not all expressions have normal forms.

---

## O

### Omega Combinator
The expression Ω = (λx.x x)(λx.x x) that has no normal form — beta reduction loops forever.

---

## P

### Parser
The component of an interpreter that converts a stream of tokens into an AST according to the language grammar.

### Precedence
Rules determining which operators bind more tightly. Higher precedence operators are evaluated first (appear deeper in AST).

---

## R

### Recognisable (Turing-Recognisable)
A language L is recognisable if a Turing machine exists that accepts all strings in L (may loop on strings not in L).

### Recursive Descent
A top-down parsing technique where each grammar rule is implemented as a function that may call other rule functions.

### Redex
A reducible expression — a subterm of the form (λx.M) N that can be beta-reduced.

### Reject State
A designated state in a Turing machine that, when entered, causes the machine to halt and indicate that the input is rejected.

---

## S

### S Combinator
The combinator S = λx.λy.λz.x z (y z). Together with K, can express any lambda term.

### State
One of the finite number of conditions a Turing machine can be in, determining its behaviour when reading a symbol.

### Substitution
The operation M[x := N] that replaces all free occurrences of x in M with N, taking care to avoid variable capture.

---

## T

### Tape Alphabet
The set Γ of all symbols that can appear on a Turing machine's tape, including the input alphabet Σ and the blank symbol.

### Token
A categorised unit of text produced by a lexer: a number, identifier, operator, keyword, etc.

### Transition Function
The function δ: Q × Γ → Q × Γ × {L, R} that defines a Turing machine's behaviour: given state and symbol, specifies next state, symbol to write and direction.

### Turing Complete
A system capable of simulating any Turing machine. Modern programming languages are Turing complete.

### Turing Machine
An abstract mathematical model of computation defined by Turing in 1936. Consists of an infinite tape, a head and a finite state machine.

---

## U

### Undecidable
A problem for which no algorithm exists that always halts with the correct answer. The Halting Problem is the canonical example.

### Unary Representation
Encoding of a number n as n repeated symbols (e.g., 3 = "111").

---

## V

### Variable (Lambda)
A symbol representing a value in lambda calculus. Can be free or bound.

---

## Y

### Y Combinator
A fixed-point combinator Y = λf.(λx.f(x x))(λx.f(x x)) that enables recursion in lambda calculus: Y f = f (Y f).

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*

© 2025 Antonio Clim. All rights reserved.
