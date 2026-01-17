# Practice Exercise: Hard 02

## Y Combinator and Recursion

**Difficulty:** Hard  
**Estimated Time:** 60 minutes  
**Topics:** Fixed-point combinators, recursion without named functions, lambda calculus

---

## Problem

Lambda calculus has no built-in support for recursion — functions cannot refer to themselves by name. The **Y combinator** solves this by finding the fixed point of a function.

Your task is to:
1. Understand the Y combinator mathematically
2. Implement recursive functions using the Y combinator
3. Prove that Y F = F (Y F) for any F

---

## Background

### The Y Combinator

```
Y = λf.(λx.f (x x)) (λx.f (x x))
```

For any function F, `Y F` is a fixed point: `Y F = F (Y F)`.

This means if we have a "template" for a recursive function that takes itself as a parameter, Y can "tie the knot" and produce the actual recursive function.

### Example: Factorial

Without recursion, we cannot write `fact n = if n == 0 then 1 else n * fact(n-1)` because `fact` cannot refer to itself.

Instead, we write a **template** that takes the recursive function as a parameter:

```
fact_template = λself.λn.if n == 0 then 1 else n * self(n - 1)
```

Then: `factorial = Y fact_template`

---

## Part 1: Mathematical Proof

Prove that `Y F = F (Y F)` by beta reduction.

Starting with:
```
Y F = (λf.(λx.f (x x)) (λx.f (x x))) F
```

Show the reduction steps:

```
Step 1: _______________
Step 2: _______________
Step 3: _______________
...
Final: F (Y F)
```

---

## Part 2: Implementation in Python

Implement the Y combinator and use it for recursive functions.

**Note:** Python uses strict (eager) evaluation, so the standard Y combinator causes infinite recursion. Use the **Z combinator** instead:

```
Z = λf.(λx.f (λy.x x y)) (λx.f (λy.x x y))
```

```python
def z_combinator(f):
    """
    The Z combinator for strict evaluation.
    
    Z = λf.(λx.f (λy.x x y)) (λx.f (λy.x x y))
    
    Usage:
        factorial = z_combinator(
            lambda self: lambda n: 1 if n == 0 else n * self(n - 1)
        )
    """
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))
```

### Task: Implement These Functions Using Z

```python
# 1. Factorial
factorial = z_combinator(
    lambda self: lambda n: # TODO
)

# 2. Fibonacci
fibonacci = z_combinator(
    lambda self: lambda n: # TODO
)

# 3. Sum of list
sum_list = z_combinator(
    lambda self: lambda lst: # TODO
)

# 4. Length of list
length = z_combinator(
    lambda self: lambda lst: # TODO
)

# 5. Map function
map_func = z_combinator(
    lambda self: lambda f: lambda lst: # TODO
)
```

---

## Part 3: Church Numeral Recursion

Implement factorial using Church numerals and the Y combinator (in the AST representation).

```python
from lab.lab_1_02_lambda_calculus import *

def church_factorial_template() -> LambdaExpr:
    """
    Create the factorial template for use with Y.
    
    Template = λself.λn.IS_ZERO n 1 (MULT n (self (PRED n)))
    """
    # TODO: Implement using the LambdaExpr AST
    pass

def compute_factorial_church(n: int) -> int:
    """
    Compute n! using Church numerals and Y combinator.
    
    Note: This will be slow for large n due to beta reduction.
    """
    y = y_combinator()
    template = church_factorial_template()
    factorial = App(y, template)
    
    n_church = church_numeral(n)
    result = App(factorial, n_church)
    
    reduced = beta_reduce(result, max_steps=10000)
    return church_to_int(reduced)
```

---

## Part 4: Analysis

Answer the following questions:

1. **Why does the standard Y combinator cause infinite loops in strict languages?**

2. **What is the difference between the Y and Z combinators?**

3. **Can you implement mutual recursion (two functions that call each other) using combinators?**

4. **What is the relationship between fixed points and recursive definitions?**

---

## Test Cases

```python
def test_z_combinator():
    # Factorial
    factorial = z_combinator(
        lambda self: lambda n: 1 if n == 0 else n * self(n - 1)
    )
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    assert factorial(10) == 3628800
    
    # Fibonacci
    fibonacci = z_combinator(
        lambda self: lambda n: n if n <= 1 else self(n - 1) + self(n - 2)
    )
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
    
    # Sum of list
    sum_list = z_combinator(
        lambda self: lambda lst: 0 if not lst else lst[0] + self(lst[1:])
    )
    assert sum_list([]) == 0
    assert sum_list([1, 2, 3]) == 6
    
    print("All Z combinator tests passed!")
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

### Part 1: Proof

```
Y F = (λf.(λx.f (x x)) (λx.f (x x))) F

Step 1: Apply outer abstraction (substitute f = F)
      = (λx.F (x x)) (λx.F (x x))

Step 2: Apply the function (substitute x = (λx.F (x x)))
      = F ((λx.F (x x)) (λx.F (x x)))

Step 3: Recognise that the argument is Y F
      = F (Y F)

QED
```

### Part 2: Python Implementation

```python
def z_combinator(f):
    return (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))

# 1. Factorial
factorial = z_combinator(
    lambda self: lambda n: 1 if n == 0 else n * self(n - 1)
)

# 2. Fibonacci
fibonacci = z_combinator(
    lambda self: lambda n: n if n <= 1 else self(n - 1) + self(n - 2)
)

# 3. Sum of list
sum_list = z_combinator(
    lambda self: lambda lst: 0 if not lst else lst[0] + self(lst[1:])
)

# 4. Length
length = z_combinator(
    lambda self: lambda lst: 0 if not lst else 1 + self(lst[1:])
)

# 5. Map
map_func = z_combinator(
    lambda self: lambda f: lambda lst: [] if not lst else [f(lst[0])] + self(f)(lst[1:])
)
```

### Part 4: Analysis Answers

1. **Y causes infinite loops** because strict evaluation evaluates `x x` immediately before `f` can "guard" it with a conditional.

2. **Z wraps the self-application** in a lambda (`λy.x x y`), delaying evaluation until the function is actually called.

3. **Mutual recursion** requires a tuple-based approach: define a pair of functions and use projection to select which one to call.

4. **Fixed points are recursive definitions** because `Y F = F (Y F)` means the result equals applying F to itself — exactly what recursion is.

</details>

---

© 2025 Antonio Clim. All rights reserved.
