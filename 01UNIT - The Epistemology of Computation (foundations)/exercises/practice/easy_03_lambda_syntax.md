# Practice Exercise: Easy 03

## Lambda Calculus Syntax

**Difficulty:** Easy  
**Estimated Time:** 10 minutes  
**Prerequisites:** Lambda calculus introduction  

---

## Problem

Identify whether each of the following is a valid lambda calculus expression. If valid, identify its type (variable, abstraction or application). If invalid, explain why.

### Expressions

1. `x`
2. `λx.x`
3. `λx`
4. `f x`
5. `(λx.x) y`
6. `λx.λy.x`
7. `.x`
8. `λ.x`
9. `(λx.x x) (λx.x x)`
10. `x y z`

---

## Template Answer

```
1. Valid/Invalid: ___
   Type/Reason: ___

2. Valid/Invalid: ___
   Type/Reason: ___

(continue for all expressions...)
```

---

## Solution

<details>
<summary>Click to reveal solution</summary>

1. **Valid**  
   **Type:** Variable  
   A single variable is the simplest lambda expression.

2. **Valid**  
   **Type:** Abstraction  
   The identity function: takes x and returns x.

3. **Invalid**  
   **Reason:** Missing body after the parameter. An abstraction must have the form λx.M where M is an expression.

4. **Valid**  
   **Type:** Application  
   Function f applied to argument x.

5. **Valid**  
   **Type:** Application  
   The identity function applied to y. Reduces to y.

6. **Valid**  
   **Type:** Abstraction (nested)  
   The K combinator: takes x, returns a function that ignores its argument and returns x.

7. **Invalid**  
   **Reason:** Missing λ and parameter. Cannot start with a dot.

8. **Invalid**  
   **Reason:** Missing parameter name between λ and dot.

9. **Valid**  
   **Type:** Application  
   The omega combinator Ω. This expression has no normal form (infinite reduction).

10. **Valid**  
    **Type:** Application (nested)  
    Parsed as ((x y) z) by left-associativity. x applied to y, then result applied to z.

</details>

---

## Syntax Rules Summary

1. **Variables:** Any identifier (x, y, foo, etc.)
2. **Abstraction:** λ followed by parameter, dot, then body: `λx.M`
3. **Application:** Two expressions side by side: `M N`
4. **Associativity:** Application is left-associative: `x y z` = `((x y) z)`
5. **Abstraction extends right:** `λx.M N` = `λx.(M N)` not `(λx.M) N`

---

*THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS — Week 1*
