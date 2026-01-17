# Practice Exercise: Easy 03

## Church Numeral Identification

**Difficulty:** Easy  
**Estimated Time:** 15 minutes  
**Topics:** Lambda calculus, Church encodings

---

## Problem

Church numerals encode natural numbers as functions. The pattern is:

```
0 = λf.λx.x           (apply f zero times)
1 = λf.λx.f x         (apply f once)
2 = λf.λx.f (f x)     (apply f twice)
n = λf.λx.fⁿ(x)       (apply f n times)
```

**Task:** Identify what number each of the following lambda expressions represents.

---

## Expressions

1. `λf.λx.f (f (f x))`

2. `λf.λx.f (f (f (f (f x))))`

3. `λf.λx.x`

4. `λf.λx.f x`

---

## Template

| Expression | Number | Reasoning |
|------------|--------|-----------|
| 1 | ___ | f is applied ___ times |
| 2 | ___ | f is applied ___ times |
| 3 | ___ | f is applied ___ times |
| 4 | ___ | f is applied ___ times |

---

## Verification

Use the lambda calculus module:

```python
from lab.lab_1_02_lambda_calculus import (
    church_numeral,
    church_to_int
)

# Create Church numerals and verify
for n in range(6):
    numeral = church_numeral(n)
    print(f"{n} = {numeral}")
```

---

## Bonus Challenge

What is the result of applying the successor function to expression 1?

```
SUCC = λn.λf.λx.f (n f x)
```

Compute: SUCC (λf.λx.f (f (f x))) = ?

---

## Solution

<details>
<summary>Click to reveal solution</summary>

| Expression | Number | Reasoning |
|------------|--------|-----------|
| 1 | 3 | f is applied 3 times: f(f(f(x))) |
| 2 | 5 | f is applied 5 times: f(f(f(f(f(x))))) |
| 3 | 0 | f is applied 0 times: just x |
| 4 | 1 | f is applied 1 time: f(x) |

**Bonus:**
```
SUCC 3 = λf.λx.f (3 f x)
       = λf.λx.f (f (f (f x)))
       = 4
```

The successor adds one more application of f.

</details>

---

© 2025 Antonio Clim. All rights reserved.
