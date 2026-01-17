# Practice Exercise: Medium 02

## Beta Reduction Practice

**Difficulty:** Medium  
**Estimated Time:** 20 minutes  
**Prerequisites:** Lambda calculus lab  

---

## Problem

Perform beta reduction on each of the following expressions. Show every step clearly, using the notation:

```
Expression → (by reducing redex) → Next expression
```

---

## Expression 1: K Combinator Application

Reduce: `(λx.λy.x) a b`

The K combinator takes two arguments and returns the first.

---

## Expression 2: S Combinator (Partial)

Reduce: `(λx.λy.λz.x z (y z)) f g`

Note: This will not fully reduce since f and g are free variables.

---

## Expression 3: Self-Application

Reduce: `(λx.x x) (λy.y)`

What is the final result?

---

## Expression 4: Church Numeral Successor

Given:
- `SUCC = λn.λf.λx.f (n f x)`
- `1 = λf.λx.f x`

Reduce: `SUCC 1`

Show that the result equals `2 = λf.λx.f (f x)`.

---

## Expression 5: Boolean AND

Given:
- `TRUE = λx.λy.x`
- `FALSE = λx.λy.y`
- `AND = λp.λq.p q FALSE`

Reduce: `AND TRUE FALSE`

---

## Template Answer

```
Expression 1:
  (λx.λy.x) a b
→ (substituting a for x)
→ ___
→ ___
→ Final: ___

Expression 2:
  ___

(continue for all expressions...)
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

### Expression 1: K Combinator

```
  (λx.λy.x) a b
→ ((λx.λy.x) a) b              [explicit association]
→ (λy.a) b                      [β: x ↦ a]
→ a                             [β: y ↦ b, but y not in body]

Final: a
```

The K combinator returns its first argument, discarding the second.

---

### Expression 2: S Combinator (Partial)

```
  (λx.λy.λz.x z (y z)) f g
→ ((λx.λy.λz.x z (y z)) f) g   [explicit association]
→ (λy.λz.f z (y z)) g          [β: x ↦ f]
→ λz.f z (g z)                  [β: y ↦ g]

Final: λz.f z (g z)
```

This is S f g, a function waiting for argument z.

---

### Expression 3: Self-Application

```
  (λx.x x) (λy.y)
→ (λy.y) (λy.y)                 [β: x ↦ (λy.y)]
→ λy.y                          [β: y ↦ (λy.y), but y not free]

Final: λy.y (the identity function)
```

---

### Expression 4: Church Successor

```
  SUCC 1
= (λn.λf.λx.f (n f x)) (λf.λx.f x)
→ λf.λx.f ((λf.λx.f x) f x)     [β: n ↦ 1]
→ λf.λx.f ((λx.f x) x)          [β: f ↦ f in inner term]
→ λf.λx.f (f x)                 [β: x ↦ x in inner term]

Final: λf.λx.f (f x) = 2 ✓
```

---

### Expression 5: Boolean AND

```
  AND TRUE FALSE
= (λp.λq.p q FALSE) TRUE FALSE
→ (λq.TRUE q FALSE) FALSE       [β: p ↦ TRUE]
→ TRUE FALSE FALSE              [β: q ↦ FALSE]
= (λx.λy.x) FALSE FALSE
→ (λy.FALSE) FALSE              [β: x ↦ FALSE]
→ FALSE                         [β: y ↦ FALSE, y not in body]

Final: FALSE ✓
```

TRUE AND FALSE = FALSE, as expected.

</details>

---

## Common Mistakes

1. **Forgetting parentheses:** Application is left-associative
2. **Variable capture:** Be careful when substituting into abstractions
3. **Stopping too early:** Continue until no more redexes exist
4. **Wrong substitution:** Only substitute for free occurrences

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
