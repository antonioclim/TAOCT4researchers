#!/usr/bin/env python3
"""
Solution: Reverse String (Easy)
"""


def reverse_string(s: str) -> str:
    """
    Reverse a string recursively.
    
    Time: O(n)
    Space: O(n) for recursion stack
    """
    # Base case
    if len(s) <= 1:
        return s
    
    # Recursive case: last char + reverse of rest
    return s[-1] + reverse_string(s[:-1])


def reverse_string_alt(s: str) -> str:
    """
    Alternative: first char goes to end.
    
    Time: O(n)
    Space: O(n)
    """
    if len(s) <= 1:
        return s
    return reverse_string_alt(s[1:]) + s[0]


def reverse_string_iterative(s: str) -> str:
    """
    Iterative version using two pointers.
    
    Time: O(n)
    Space: O(n) for result
    """
    chars = list(s)
    left, right = 0, len(chars) - 1
    
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left += 1
        right -= 1
    
    return "".join(chars)


if __name__ == "__main__":
    # Test cases
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
    assert reverse_string("ab") == "ba"
    assert reverse_string("hello") == "olleh"
    assert reverse_string("Python") == "nohtyP"
    
    print("All tests passed!")
