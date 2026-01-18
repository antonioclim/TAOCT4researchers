# Practice Exercise: Easy 02

## AST Construction

**Difficulty:** Easy  
**Estimated Time:** 15 minutes  
**Topics:** Abstract syntax trees, expression parsing

---

## Problem

Given the arithmetic expression:

```
(2 + 3) * 4 - 1
```

**Task:** Draw the abstract syntax tree (AST) for this expression.

---

## Requirements

1. Identify the root node (outermost operation)
2. Build the tree respecting operator precedence
3. Label each node with its type (Num, BinOp)
4. Verify by evaluating the tree bottom-up

---

## Template

Complete the following AST structure:

```
           _______
          /       \
      _______     ___
     /       \
   ___       ___
  /   \
___   ___
```

Fill in the blanks:
- Root node: _______
- Left child of root: _______
- Right child of root: _______

---

## Verification

Use the interpreter to check your AST:

```python
from lab.lab_1_03_ast_interpreter import parse

expr = "(2 + 3) * 4 - 1"
ast = parse(expr)
print(ast)
```

---

## Expected Evaluation

Work through the tree bottom-up:
1. Evaluate (2 + 3) = ___
2. Evaluate ___ * 4 = ___
3. Evaluate ___ - 1 = ___

---

## Solution

<details>
<summary>Click to reveal solution</summary>

```
          BinOp(-)
          /       \
     BinOp(*)     Num(1)
     /       \
 BinOp(+)   Num(4)
  /   \
Num(2) Num(3)
```

**AST representation:**
```python
BinOp(
    BinOp(
        BinOp(Num(2), '+', Num(3)),
        '*',
        Num(4)
    ),
    '-',
    Num(1)
)
```

**Evaluation:**
1. (2 + 3) = 5
2. 5 * 4 = 20
3. 20 - 1 = 19

The key insight is that subtraction is the outermost (last) operation due to precedence: multiplication binds tighter than subtraction.

</details>

---

© 2025 Antonio Clim. All rights reserved.
