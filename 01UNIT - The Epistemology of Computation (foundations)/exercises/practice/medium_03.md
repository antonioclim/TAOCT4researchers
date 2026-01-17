# Practice Exercise: Medium 03

## Extending the Interpreter with Power Operator

**Difficulty:** Medium  
**Estimated Time:** 30 minutes  
**Topics:** Lexer extension, parser modification, operator precedence

---

## Problem

The mini-interpreter already supports the `^` operator for exponentiation. Your task is to add support for the `**` operator (Python-style) as an **alias** for exponentiation, while maintaining correct right-associativity.

Additionally, add support for the **factorial** postfix operator `!`.

---

## Requirements

### 1. Power Operator Alias

Both `^` and `**` should work for exponentiation:

```
2 ^ 3       # → 8
2 ** 3      # → 8 (same as above)
2 ** 3 ** 2 # → 512 (right-associative: 2^(3^2) = 2^9)
```

### 2. Factorial Operator

Add the `!` postfix operator:

```
5!          # → 120
3! + 2!     # → 8 (6 + 2)
(2 + 3)!    # → 120
```

**Precedence:** Factorial should bind tighter than exponentiation.

```
2 ^ 3!      # → 2^6 = 64 (factorial first)
```

---

## Implementation Guide

### Step 1: Modify the Lexer

Add a new token type for `**`:

```python
class TokenType(Enum):
    # ... existing tokens ...
    DOUBLE_STAR = auto()  # **
    BANG = auto()         # !
```

Update `get_next_token()`:

```python
# Check for ** before single *
if self.current_char == '*' and self.peek() == '*':
    pos = self.pos
    self.advance()
    self.advance()
    return Token(TokenType.DOUBLE_STAR, '**', pos)
```

### Step 2: Add AST Node for Factorial

```python
@dataclass(frozen=True)
class Factorial:
    """Factorial operation (postfix)."""
    operand: Expr
```

### Step 3: Modify the Parser

Update the grammar:

```
power      → postfix ('^' power | '**' power)?
postfix    → unary ('!')*
unary      → '-' unary | postfix
```

```python
def power(self) -> Expr:
    base = self.postfix()  # Changed from unary
    
    if self.current_token.type in (TokenType.CARET, TokenType.DOUBLE_STAR):
        self.eat(self.current_token.type)
        exponent = self.power()
        return BinOp(base, '^', exponent)
    
    return base

def postfix(self) -> Expr:
    expr = self.unary()
    
    while self.current_token.type == TokenType.BANG:
        self.eat(TokenType.BANG)
        expr = Factorial(expr)
    
    return expr
```

### Step 4: Extend the Evaluator

```python
case Factorial(operand):
    val = self.evaluate(operand, env)
    if not isinstance(val, (int, float)):
        raise TypeError("Factorial requires a number")
    if val < 0 or val != int(val):
        raise ValueError("Factorial requires non-negative integer")
    return float(math.factorial(int(val)))
```

---

## Test Cases

```python
def test_extended_operators():
    tests = [
        # Power operator alias
        ("2 ** 3", 8.0),
        ("2 ^ 3", 8.0),
        ("2 ** 3 ** 2", 512.0),  # Right-associative
        
        # Factorial
        ("5!", 120.0),
        ("0!", 1.0),
        ("1!", 1.0),
        ("3! + 2!", 8.0),
        
        # Combined
        ("2 ^ 3!", 64.0),  # 2^6
        ("(2 + 1)!", 6.0),
        ("2 ** 2!", 16.0),  # 2^2
        
        # Chained factorial (5!! is not double factorial, it's (5!)!)
        # This would be very large, so skip or handle specially
    ]
    
    for expr, expected in tests:
        result = evaluate(expr)
        assert abs(result - expected) < 1e-10, f"{expr} = {result}, expected {expected}"
```

---

## Starter Template

```python
# Add to TokenType enum
DOUBLE_STAR = auto()
BANG = auto()

# Add to Lexer.get_next_token()
# ... handle ** and ! tokens ...

# Add AST node
@dataclass(frozen=True)
class Factorial:
    operand: 'Expr'

# Update Expr union type
Expr = Union[Num, Var, BinOp, UnaryOp, FuncCall, Let, Lambda, IfExpr, Factorial]

# Update Parser with postfix method

# Update Evaluator match statement
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

See the complete implementation in `exercises/solutions/medium_03_solution.py`.

Key changes:

1. **Lexer:** Added `DOUBLE_STAR` and `BANG` token detection
2. **AST:** Added `Factorial` dataclass
3. **Parser:** Added `postfix()` method between `power()` and `unary()`
4. **Evaluator:** Added `Factorial` case using `math.factorial()`

The tricky part is getting the precedence right. The grammar ensures:
- Factorial binds tightest (postfix)
- Then exponentiation (right-associative)
- Then unary minus
- Then multiplication/division
- Then addition/subtraction
- Then comparison

</details>

---

© 2025 Antonio Clim. All rights reserved.
