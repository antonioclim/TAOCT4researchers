# Practice Exercise: Hard 02

## Church Numerals: Predecessor Function

**Difficulty:** Hard  
**Estimated Time:** 40 minutes  
**Prerequisites:** Lambda calculus lab, Church encodings  

---

## Background

While successor, addition and multiplication are relatively direct with Church numerals, the predecessor function is surprisingly tricky. Alonzo Church himself initially thought subtraction might be impossible in lambda calculus!

The key insight is to use **pairs** to build up the result.

---

## Problem

Implement the predecessor function for Church numerals in pure lambda calculus.

### Specification

```
PRED 0 = 0
PRED n = n - 1  (for n > 0)
```

### Requirements

1. Define Church pairs (PAIR, FST, SND)
2. Use pairs to implement PRED
3. Verify your implementation reduces correctly
4. Implement in both lambda calculus notation and Python

---

## Part 1: Church Pairs

First, implement pairs in lambda calculus.

### Definition

A pair (a, b) should satisfy:
- FST (PAIR a b) = a
- SND (PAIR a b) = b

### Template

```
PAIR = λx.λy.λf.___
FST  = λp.p ___
SND  = λp.p ___
```

---

## Part 2: The Predecessor Algorithm

The key insight: to compute PRED n, we build pairs (0,0), (1,0), (2,1), ..., (n, n-1) by applying a transformation n times, then take the second element.

### Transformation Function

```
SHIFT = λp.PAIR (SUCC (FST p)) (FST p)
```

This transforms (k, k-1) into (k+1, k).

### Predecessor Definition

```
PRED = λn.SND (n SHIFT (PAIR 0 0))
```

---

## Tasks

### Task 1: Verify SHIFT

Show that `SHIFT (PAIR 2 1)` reduces to `PAIR 3 2`.

### Task 2: Trace PRED 3

Show the complete reduction of `PRED 3` step by step.

### Task 3: Python Implementation

Implement all functions in Python and verify they work correctly.

```python
# Template
def church_pair(x, y):
    """Create a Church pair."""
    # TODO
    pass

def church_fst(p):
    """Extract first element of a Church pair."""
    # TODO
    pass

def church_snd(p):
    """Extract second element of a Church pair."""
    # TODO
    pass

def church_shift(p):
    """Transform (k, k-1) to (k+1, k)."""
    # TODO
    pass

def church_pred(n):
    """Compute predecessor of Church numeral n."""
    # TODO
    pass
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

### Part 1: Church Pairs

```
PAIR = λx.λy.λf.f x y
FST  = λp.p (λx.λy.x)    = λp.p TRUE
SND  = λp.p (λx.λy.y)    = λp.p FALSE
```

The pair stores x and y, then when given a selector function f, applies f to both.

---

### Task 1: Verify SHIFT

```
SHIFT (PAIR 2 1)
= (λp.PAIR (SUCC (FST p)) (FST p)) (PAIR 2 1)
→ PAIR (SUCC (FST (PAIR 2 1))) (FST (PAIR 2 1))
→ PAIR (SUCC 2) 2
→ PAIR 3 2 ✓
```

---

### Task 2: Trace PRED 3

```
PRED 3
= SND (3 SHIFT (PAIR 0 0))
= SND ((λf.λx.f(f(f x))) SHIFT (PAIR 0 0))
→ SND (SHIFT (SHIFT (SHIFT (PAIR 0 0))))

Let's trace the inner applications:
  PAIR 0 0                    = (0, 0)
  SHIFT (PAIR 0 0)            = PAIR 1 0 = (1, 0)
  SHIFT (PAIR 1 0)            = PAIR 2 1 = (2, 1)
  SHIFT (PAIR 2 1)            = PAIR 3 2 = (3, 2)

So:
  SND (PAIR 3 2) = 2 ✓
```

---

### Task 3: Python Implementation

```python
def church_numeral(n):
    """Create Church numeral for n."""
    def numeral(f):
        def apply_n(x):
            result = x
            for _ in range(n):
                result = f(result)
            return result
        return apply_n
    return numeral

def church_to_int(n):
    """Convert Church numeral to Python int."""
    return n(lambda x: x + 1)(0)

def church_pair(x, y):
    """Create a Church pair (x, y)."""
    return lambda f: f(x)(y)

def church_fst(p):
    """Extract first element: FST (x, y) = x."""
    return p(lambda x: lambda y: x)

def church_snd(p):
    """Extract second element: SND (x, y) = y."""
    return p(lambda x: lambda y: y)

def church_succ(n):
    """Successor: SUCC n = n + 1."""
    return lambda f: lambda x: f(n(f)(x))

def church_shift(p):
    """Shift: (k, prev) -> (k+1, k)."""
    k = church_fst(p)
    return church_pair(church_succ(k), k)

def church_pred(n):
    """Predecessor: PRED n = max(0, n-1)."""
    zero_pair = church_pair(church_numeral(0), church_numeral(0))
    result_pair = n(church_shift)(zero_pair)
    return church_snd(result_pair)

# Verification
for i in range(10):
    n = church_numeral(i)
    pred_n = church_pred(n)
    expected = max(0, i - 1)
    actual = church_to_int(pred_n)
    print(f"PRED {i} = {actual} (expected {expected})")
    assert actual == expected
```

Output:
```
PRED 0 = 0 (expected 0)
PRED 1 = 0 (expected 0)
PRED 2 = 1 (expected 1)
PRED 3 = 2 (expected 2)
...
```

</details>

---

## Why Is This Hard?

1. **No decrement operation:** Lambda calculus has no built-in subtraction
2. **Structural recursion:** Church numerals only support iteration, not deconstruction
3. **Information loss:** A numeral n only knows "apply f n times," not "what is n-1"
4. **The trick:** Build up pairs to maintain "memory" of the previous value

---

## Extension: Subtraction

Using PRED, implement subtraction:

```
SUB = λm.λn.n PRED m
```

This applies PRED to m, n times, giving m - n (or 0 if n > m).

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
