# Practice Exercise: Medium 02

## Implementing Church Arithmetic

**Difficulty:** Medium  
**Estimated Time:** 30 minutes  
**Topics:** Lambda calculus, Church encodings, Python implementation

---

## Problem

Implement Church encoding arithmetic operations as both lambda calculus expressions and Python functions.

---

## Part A: Lambda Expressions

Complete the following Church encoding definitions:

### 1. Predecessor (PRED)

The predecessor function returns n-1 (or 0 if n=0).

```
PRED = λn.λf.λx.???
```

**Hint:** Use pairs. Define:
- PAIR = λx.λy.λf.f x y
- FIRST = λp.p (λx.λy.x)
- SECOND = λp.p (λx.λy.y)

Then build (n, n-1) by starting with (0, 0) and applying a transformation n times.

### 2. Subtraction (SUB)

```
SUB = λm.λn.???
```

**Hint:** SUB m n = apply PRED to m exactly n times.

### 3. Less Than or Equal (LEQ)

```
LEQ = λm.λn.???
```

**Hint:** m ≤ n iff m - n = 0. Use IS_ZERO.

---

## Part B: Python Implementation

Implement these operations as Python functions:

```python
from typing import Callable

# Type alias for Church-style functions
ChurchNum = Callable[[Callable], Callable]

def church_pred(n: ChurchNum) -> ChurchNum:
    """
    Predecessor function for Church numerals.
    
    PRED(0) = 0
    PRED(n) = n - 1
    
    Args:
        n: A Church numeral
        
    Returns:
        The predecessor as a Church numeral
    """
    # TODO: Implement using the pair technique
    pass


def church_sub(m: ChurchNum, n: ChurchNum) -> ChurchNum:
    """
    Subtraction for Church numerals.
    
    SUB(m, n) = m - n (or 0 if m < n)
    """
    # TODO: Implement
    pass


def church_leq(m: ChurchNum, n: ChurchNum) -> ChurchNum:
    """
    Less than or equal comparison.
    
    Returns Church TRUE if m <= n, Church FALSE otherwise.
    """
    # TODO: Implement
    pass


def church_to_int(n: ChurchNum) -> int:
    """Convert Church numeral to Python int."""
    return n(lambda x: x + 1)(0)


def int_to_church(n: int) -> ChurchNum:
    """Convert Python int to Church numeral."""
    def numeral(f):
        def apply_n_times(x):
            result = x
            for _ in range(n):
                result = f(result)
            return result
        return apply_n_times
    return numeral
```

---

## Test Cases

```python
def test_church_arithmetic():
    # Test predecessor
    for i in range(5):
        n = int_to_church(i)
        pred_n = church_pred(n)
        expected = max(0, i - 1)
        assert church_to_int(pred_n) == expected, f"PRED({i}) failed"
    
    # Test subtraction
    test_cases = [
        (5, 3, 2),
        (3, 5, 0),  # Clamped to 0
        (4, 4, 0),
        (10, 3, 7),
    ]
    for m, n, expected in test_cases:
        result = church_to_int(church_sub(int_to_church(m), int_to_church(n)))
        assert result == expected, f"SUB({m}, {n}) = {result}, expected {expected}"
    
    # Test LEQ
    true_cases = [(0, 0), (0, 5), (3, 3), (2, 5)]
    false_cases = [(5, 0), (3, 2), (10, 5)]
    
    TRUE = int_to_church(1)  # Simplified check
    FALSE = int_to_church(0)
    
    print("All tests passed!")
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

### Lambda Expressions

**Predecessor using pairs:**
```
PAIR   = λx.λy.λf.f x y
FIRST  = λp.p TRUE      where TRUE = λx.λy.x
SECOND = λp.p FALSE     where FALSE = λx.λy.y

SHIFT  = λp.PAIR (SECOND p) (SUCC (SECOND p))
       = λp.λf.f (p FALSE) (SUCC (p FALSE))

PRED   = λn.FIRST (n SHIFT (PAIR 0 0))
```

**Subtraction:**
```
SUB = λm.λn.n PRED m
```

**Less than or equal:**
```
LEQ = λm.λn.IS_ZERO (SUB m n)
```

### Python Implementation

```python
def church_pred(n: ChurchNum) -> ChurchNum:
    """Predecessor using the pair technique."""
    # Pair constructors
    def pair(x, y):
        return lambda f: f(x)(y)
    
    def first(p):
        return p(lambda x: lambda y: x)
    
    def second(p):
        return p(lambda x: lambda y: y)
    
    # Shift function: (a, b) -> (b, b+1)
    def shift(p):
        b = second(p)
        return pair(b, lambda f: lambda x: f(b(f)(x)))
    
    # Start with (0, 0), apply shift n times, take first
    zero = lambda f: lambda x: x
    initial = pair(zero, zero)
    
    result_pair = n(shift)(initial)
    return first(result_pair)


def church_sub(m: ChurchNum, n: ChurchNum) -> ChurchNum:
    """Subtraction: apply pred n times to m."""
    return n(church_pred)(m)


def church_leq(m: ChurchNum, n: ChurchNum) -> ChurchNum:
    """m <= n iff m - n = 0."""
    diff = church_sub(m, n)
    
    # IS_ZERO: returns TRUE if n is 0
    def is_zero(num):
        # Apply num to (λx.FALSE) and TRUE
        # If num is 0, returns TRUE (no applications)
        # If num > 0, returns FALSE (at least one application)
        church_true = lambda x: lambda y: x
        church_false = lambda x: lambda y: y
        return num(lambda _: church_false)(church_true)
    
    return is_zero(diff)
```

</details>

---

© 2025 Antonio Clim. All rights reserved.
