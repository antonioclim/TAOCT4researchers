# Practice Exercise: Sum of Digits (Easy)

**Difficulty**: ★☆☆☆☆  
**Estimated Time**: 15 minutes  
**Topic**: Linear Recursion

---

## Problem Statement

Write a recursive function that computes the sum of digits of a non-negative integer.

## Function Signature

```python
def sum_digits(n: int) -> int:
    """
    Compute sum of digits recursively.
    
    Args:
        n: Non-negative integer
    
    Returns:
        Sum of all digits in n
    
    Examples:
        >>> sum_digits(123)
        6  # 1 + 2 + 3
        >>> sum_digits(9999)
        36
        >>> sum_digits(0)
        0
    """
    pass
```

## Hints

1. What is the base case? (Think: when does a number have only one digit?)
2. How can you extract the last digit of a number?
3. How can you remove the last digit from a number?

## Test Cases

```python
assert sum_digits(0) == 0
assert sum_digits(5) == 5
assert sum_digits(123) == 6
assert sum_digits(9999) == 36
assert sum_digits(10000) == 1
```

## Expected Complexity

- **Time**: O(d) where d is the number of digits
- **Space**: O(d) for recursion stack

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

1. Base case: `n < 10` (single digit)
2. Last digit: `n % 10`
3. Remove last digit: `n // 10`
4. Recurrence: `sum_digits(n) = (n % 10) + sum_digits(n // 10)`

</details>
