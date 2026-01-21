# Practice Exercise: Power Function (Easy)

**Difficulty**: ★☆☆☆☆  
**Estimated Time**: 15 minutes  
**Topic**: Linear and Logarithmic Recursion

---

## Problem Statement

Implement a recursive function to compute x raised to the power n.

## Function Signature

```python
def power(x: float, n: int) -> float:
    """
    Compute x^n recursively.
    
    Args:
        x: Base (any real number)
        n: Exponent (non-negative integer)
    
    Returns:
        x raised to the power n
    
    Examples:
        >>> power(2, 10)
        1024
        >>> power(3, 0)
        1
        >>> power(2.5, 2)
        6.25
    """
    pass
```

## Challenge

After implementing the O(n) solution, implement an O(log n) version using the property:
- x^n = (x^(n/2))^2 if n is even
- x^n = x × (x^(n/2))^2 if n is odd

## Test Cases

```python
assert power(2, 0) == 1
assert power(2, 1) == 2
assert power(2, 10) == 1024
assert power(3, 4) == 81
assert abs(power(2.5, 2) - 6.25) < 0.001
```

## Expected Complexity

- **Naive**: Time O(n), Space O(n)
- **Optimised**: Time O(log n), Space O(log n)

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

**Naive O(n):**
```
power(x, n) = x * power(x, n-1)
power(x, 0) = 1
```

**Optimised O(log n):**
```
power(x, n) = power(x, n//2)^2        if n is even
            = x * power(x, n//2)^2    if n is odd
power(x, 0) = 1
```

</details>
