# Practice Exercise: Hard 03

## Extending the Mini-Interpreter

**Difficulty:** Hard  
**Estimated Time:** 60 minutes  
**Prerequisites:** AST interpreter lab, parser implementation  

---

## Problem

Extend the mini-interpreter with **recursive function definitions** using the `letrec` construct.

### Motivation

The current `let` binding does not allow a function to refer to itself. For example, this does not work:

```
let fact = fun n -> if n == 0 then 1 else n * fact(n - 1)
in fact(5)
```

The `fact` inside the body cannot see itself because the binding is not yet complete when the body is defined.

### Solution: letrec

The `letrec` construct creates a recursive binding:

```
letrec fact = fun n -> if n == 0 then 1 else n * fact(n - 1)
in fact(5)
```

---

## Tasks

### Task 1: Add the AST Node

Create a new `LetRec` dataclass:

```python
@dataclass(frozen=True)
class LetRec:
    """Recursive let binding.
    
    Syntax: letrec name = value in body
    
    The value expression can reference 'name' recursively.
    """
    name: str
    value: 'Expr'
    body: 'Expr'
```

### Task 2: Update the Lexer

Add the `letrec` keyword to the lexer.

### Task 3: Update the Parser

Add parsing support for `letrec` expressions.

### Task 4: Update the Evaluator

Implement evaluation of `LetRec`. The key is to create a closure that contains a reference to itself.

---

## Implementation Hints

<details>
<summary>💡 Hint 1: Self-Referential Closure</summary>

The trick is to create the closure first with a placeholder environment, then update the environment to include the closure itself:

```python
case LetRec(name, value, body):
    # Create a mutable environment for self-reference
    rec_env = dict(env)
    
    # Evaluate the value (should be a lambda)
    closure = self.evaluate(value, rec_env)
    
    # Now add the closure to its own environment
    if isinstance(closure, Closure):
        closure.env[name] = closure
    
    # Add to environment for body
    rec_env[name] = closure
    
    return self.evaluate(body, rec_env)
```

Note: This requires making `Closure.env` mutable.

</details>

<details>
<summary>💡 Hint 2: Alternative - Fixed Point</summary>

Another approach uses a fixed-point combinator. Define:

```python
def make_recursive(f, env, name, body):
    """Create a recursive closure using a fixed point."""
    # Create closure with a wrapper that provides self-reference
    def rec_apply(arg):
        # Evaluate body with name bound to this function
        ...
```

</details>

---

## Test Cases

```python
def test_letrec():
    # Factorial
    assert evaluate("""
        letrec fact = fun n ->
            if n == 0 then 1
            else n * fact(n - 1)
        in fact(5)
    """) == 120
    
    # Fibonacci
    assert evaluate("""
        letrec fib = fun n ->
            if n <= 1 then n
            else fib(n - 1) + fib(n - 2)
        in fib(10)
    """) == 55
    
    # Sum from 1 to n
    assert evaluate("""
        letrec sum = fun n ->
            if n == 0 then 0
            else n + sum(n - 1)
        in sum(10)
    """) == 55
    
    # GCD
    assert evaluate("""
        letrec gcd = fun a ->
            fun b ->
                if b == 0 then a
                else gcd(b)(a % b)
        in gcd(48)(18)
    """) == 6
```

---

## Starter Code

```python
# Add to TokenType enum
LETREC = auto()

# Add to Lexer.KEYWORDS
'letrec': TokenType.LETREC,

# Add AST node
@dataclass(frozen=True)
class LetRec:
    name: str
    value: Expr
    body: Expr

# Update Expr type alias
Expr = Union[Num, Var, BinOp, UnaryOp, FuncCall, Let, LetRec, Lambda, IfExpr]

# Add to Parser.expr()
if self.current_token.type == TokenType.LETREC:
    return self.letrec_expr()

# Add Parser.letrec_expr()
def letrec_expr(self) -> LetRec:
    self.eat(TokenType.LETREC)
    name_token = self.eat(TokenType.IDENTIFIER)
    name = str(name_token.value)
    self.eat(TokenType.EQUALS)
    value = self.expr()
    self.eat(TokenType.IN)
    body = self.expr()
    return LetRec(name, value, body)

# Update Closure to allow mutable env
@dataclass
class Closure:  # Note: removed frozen=True
    param: str
    body: Expr
    env: dict[str, Value]

# Add to Evaluator.evaluate()
case LetRec(name, value, body):
    # TODO: Implement recursive binding
    pass
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

```python
# In Evaluator.evaluate():
case LetRec(name, value, body):
    # Create environment for recursive evaluation
    rec_env = dict(env)
    
    # Evaluate the value expression
    # It should be a lambda for recursion to make sense
    val = self.evaluate(value, rec_env)
    
    if isinstance(val, Closure):
        # Add self-reference to the closure's environment
        # This is the key to making recursion work
        val.env[name] = val
        rec_env[name] = val
    else:
        # Non-function letrec (unusual but allowed)
        rec_env[name] = val
    
    # Evaluate the body with the recursive binding
    return self.evaluate(body, rec_env)
```

The full implementation requires:

1. **Lexer update:** Add `LETREC` token type and keyword mapping
2. **AST update:** Add `LetRec` dataclass
3. **Parser update:** Add `letrec_expr` method and case in `expr`
4. **Closure update:** Make `env` mutable (remove `frozen=True`)
5. **Evaluator update:** Add `LetRec` case with self-referential binding

</details>

---

## Extension Challenges

1. **Mutual recursion:** Implement `letrec* f = ... and g = ... in ...` for mutually recursive functions

2. **Tail call optimisation:** Detect and optimise tail-recursive calls to prevent stack overflow

3. **Memoisation:** Add automatic memoisation for recursive functions

---

## Reflection Questions

1. Why does the standard `let` not support recursion?

2. How does the Y combinator relate to `letrec`?

3. What are the trade-offs between mutable and immutable environments?

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
