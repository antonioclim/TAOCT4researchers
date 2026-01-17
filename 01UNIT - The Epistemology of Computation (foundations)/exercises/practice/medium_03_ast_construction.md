# Practice Exercise: Medium 03

## AST Construction

**Difficulty:** Medium  
**Estimated Time:** 20 minutes  
**Prerequisites:** AST interpreter lab  

---

## Problem

For each expression, construct the Abstract Syntax Tree (AST) using the node types from the interpreter lab. Draw the tree structure and write the equivalent Python dataclass representation.

---

## Expression 1

`3 + 4 * 2`

### Tree Structure

Draw the AST showing operator precedence.

### Python Representation

Write using `Num`, `BinOp` constructors.

---

## Expression 2

`let x = 5 in x * x + 1`

### Tree Structure

Draw the complete AST.

### Python Representation

Write using `Let`, `Var`, `Num`, `BinOp` constructors.

---

## Expression 3

`if a > b then a else b`

### Tree Structure

Draw the conditional expression tree.

### Python Representation

Write using `IfExpr`, `Var`, `BinOp` constructors.

---

## Expression 4

`fun x -> fun y -> x + y`

### Tree Structure

Draw the nested lambda expression.

### Python Representation

Write using `Lambda`, `Var`, `BinOp` constructors.

---

## Expression 5

`let double = fun x -> x * 2 in double(5)`

### Tree Structure

Draw the complete AST including the function call.

### Python Representation

Write using `Let`, `Lambda`, `Var`, `Num`, `BinOp`, `FuncCall` constructors.

---

## Template Answer

```
Expression 1: 3 + 4 * 2

Tree:
       ___
      /   \
    ___   ___
   /       \
  ___     ___

Python:
BinOp(___, ___, ___)
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

### Expression 1: `3 + 4 * 2`

**Tree:**
```
       BinOp(+)
      /        \
   Num(3)    BinOp(*)
            /        \
         Num(4)    Num(2)
```

**Python:**
```python
BinOp(
    Num(3),
    '+',
    BinOp(Num(4), '*', Num(2))
)
```

Note: Multiplication has higher precedence, so it is deeper in the tree.

---

### Expression 2: `let x = 5 in x * x + 1`

**Tree:**
```
            Let
           / | \
         'x' |  \
             |   \
          Num(5)  BinOp(+)
                 /        \
            BinOp(*)     Num(1)
           /        \
        Var(x)    Var(x)
```

**Python:**
```python
Let(
    'x',
    Num(5),
    BinOp(
        BinOp(Var('x'), '*', Var('x')),
        '+',
        Num(1)
    )
)
```

---

### Expression 3: `if a > b then a else b`

**Tree:**
```
           IfExpr
         /   |    \
        /    |     \
   BinOp(>)  |      \
   /    \    |       \
Var(a) Var(b) Var(a)  Var(b)
```

**Python:**
```python
IfExpr(
    BinOp(Var('a'), '>', Var('b')),
    Var('a'),
    Var('b')
)
```

---

### Expression 4: `fun x -> fun y -> x + y`

**Tree:**
```
        Lambda
       /      \
     'x'    Lambda
           /      \
         'y'    BinOp(+)
               /        \
           Var(x)     Var(y)
```

**Python:**
```python
Lambda(
    'x',
    Lambda(
        'y',
        BinOp(Var('x'), '+', Var('y'))
    )
)
```

This is a curried addition function.

---

### Expression 5: `let double = fun x -> x * 2 in double(5)`

**Tree:**
```
              Let
            /  |  \
      'double' |   \
               |    \
            Lambda   FuncCall
           /      \   /      \
         'x'     BinOp(*)  'double'  Num(5)
                /        \
             Var(x)    Num(2)
```

**Python:**
```python
Let(
    'double',
    Lambda(
        'x',
        BinOp(Var('x'), '*', Num(2))
    ),
    FuncCall('double', (Num(5),))
)
```

</details>

---

## Key Insights

1. **Precedence determines depth:** Higher-precedence operators appear deeper in the tree
2. **Left-associativity:** For equal precedence, left operand is evaluated first
3. **Scope is structural:** Let and Lambda bodies are subtrees
4. **ASTs eliminate ambiguity:** The tree structure makes evaluation order explicit

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
