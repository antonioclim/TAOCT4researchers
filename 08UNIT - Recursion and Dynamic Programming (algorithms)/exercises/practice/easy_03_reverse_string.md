# Practice Exercise: Reverse String (Easy)

**Difficulty**: ★☆☆☆☆  
**Estimated Time**: 10 minutes  
**Topic**: Linear Recursion

---

## Problem Statement

Write a recursive function that reverses a string.

## Function Signature

```python
def reverse_string(s: str) -> str:
    """
    Reverse a string recursively.
    
    Args:
        s: Input string
    
    Returns:
        Reversed string
    
    Examples:
        >>> reverse_string("hello")
        "olleh"
        >>> reverse_string("")
        ""
        >>> reverse_string("a")
        "a"
    """
    pass
```

## Test Cases

```python
assert reverse_string("") == ""
assert reverse_string("a") == "a"
assert reverse_string("ab") == "ba"
assert reverse_string("hello") == "olleh"
assert reverse_string("Python") == "nohtyP"
```

## Expected Complexity

- **Time**: O(n)
- **Space**: O(n) for recursion stack

---

## Solution Approach

<details>
<summary>Click to reveal hints</summary>

Base case: empty string or single character  
Recursive case: reverse(s) = s[-1] + reverse(s[:-1])

Alternative: reverse(s) = reverse(s[1:]) + s[0]

</details>
