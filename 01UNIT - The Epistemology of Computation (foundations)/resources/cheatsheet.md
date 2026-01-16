# Week 1: Cheatsheet

## The Epistemology of Computation — Quick Reference

---

## Turing Machine Definition

**M = (Q, Σ, Γ, δ, q₀, q_accept, q_reject)**

| Symbol | Name | Description |
|--------|------|-------------|
| Q | States | Finite set of states |
| Σ | Input alphabet | Symbols in input (no blank) |
| Γ | Tape alphabet | All symbols including blank, Σ ⊂ Γ |
| δ | Transition function | δ: Q × Γ → Q × Γ × {L, R} |
| q₀ | Initial state | Starting state |
| q_accept | Accept state | Halts and accepts |
| q_reject | Reject state | Halts and rejects |

**Transition notation:** δ(state, read) = (next_state, write, direction)

---

## Lambda Calculus Syntax

```
<expr> ::= <var>                    Variable
         | λ<var>.<expr>            Abstraction
         | <expr> <expr>            Application
```

**Conventions:**
- Application is left-associative: `x y z` = `((x y) z)`
- Abstraction extends right: `λx.M N` = `λx.(M N)`

---

## Beta Reduction

**(λx.M) N → M[x := N]**

Substitute N for all free occurrences of x in M.

**Example:**
```
(λx.x + 1) 5 → 5 + 1 → 6
```

**Free vs Bound:**
- In `λx.x y`, x is bound, y is free
- Only substitute for free variables

---

## Church Encodings

### Booleans

| Name | Encoding | Behaviour |
|------|----------|-----------|
| TRUE | λx.λy.x | Select first |
| FALSE | λx.λy.y | Select second |
| NOT | λp.p FALSE TRUE | Flip |
| AND | λp.λq.p q FALSE | Both true |
| OR | λp.λq.p TRUE q | Either true |
| IF | λc.λt.λe.c t e | Conditional |

### Numerals

| n | Encoding | Pattern |
|---|----------|---------|
| 0 | λf.λx.x | f⁰(x) |
| 1 | λf.λx.f x | f¹(x) |
| 2 | λf.λx.f (f x) | f²(x) |
| n | λf.λx.fⁿ(x) | Apply f n times |

### Arithmetic

| Operation | Encoding |
|-----------|----------|
| SUCC | λn.λf.λx.f (n f x) |
| ADD | λm.λn.λf.λx.m f (n f x) |
| MULT | λm.λn.λf.m (n f) |
| ISZERO | λn.n (λx.FALSE) TRUE |

---

## Famous Combinators

| Name | Definition | Purpose |
|------|------------|---------|
| I | λx.x | Identity |
| K | λx.λy.x | Constant (= TRUE) |
| S | λx.λy.λz.x z (y z) | Substitution |
| Y | λf.(λx.f(x x))(λx.f(x x)) | Fixed point |
| Ω | (λx.x x)(λx.x x) | Divergence |

**S K K = I** (derivation of identity)

---

## AST Node Types

```python
Num(value: float)           # Literal: 42
Var(name: str)              # Variable: x
BinOp(left, op, right)      # Binary: x + y
UnaryOp(op, operand)        # Unary: -x
FuncCall(name, args)        # Call: sin(x)
Let(name, value, body)      # Binding: let x = 5 in x + 1
Lambda(param, body)         # Function: fun x -> x * x
IfExpr(cond, then, else)    # Conditional: if ... then ... else ...
```

---

## Operator Precedence (High to Low)

1. **Parentheses** `()`
2. **Function call** `f(x)`
3. **Unary** `-x`
4. **Power** `^` (right-associative)
5. **Multiplicative** `*`, `/`, `%`
6. **Additive** `+`, `-`
7. **Comparison** `<`, `>`, `<=`, `>=`, `==`, `!=`

---

## Halting Problem

**Statement:** No algorithm can determine, for all programs P and inputs I, whether P halts on I.

**Proof sketch (diagonalisation):**
1. Assume H(P, I) decides halting
2. Build D that runs H(M, M) and does opposite
3. D(D) leads to contradiction
4. Therefore H cannot exist

**Implications:**
- No perfect debugger
- No complete program verification
- No universal virus scanner

---

## Python Quick Reference

### Type Hints

```python
def func(x: int) -> str:          # Simple types
def func(items: list[int]) -> None:  # Generic
def func(x: int | None) -> str:   # Union (3.10+)
```

### Pattern Matching (3.10+)

```python
match expr:
    case Num(value):
        return value
    case BinOp(left, '+', right):
        return eval(left) + eval(right)
    case _:
        raise ValueError()
```

### Dataclasses

```python
@dataclass(frozen=True)  # Immutable
class Node:
    value: int
    children: tuple[Node, ...]
```

---

## Key Formulas

**Church-Turing Thesis:** Effective computability = Turing computability

**Turing Machine Time:** O(steps) where each step is one transition

**Multi-tape speedup:** k-tape TM can simulate single-tape with O(t²) → O(t)

---

*Print this page for quick reference during exercises.*

© 2025 Antonio Clim

