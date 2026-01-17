#!/usr/bin/env python3
import logging

logger = logging.getLogger(__name__)

"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 3, Practice Exercise: Easy 03 — Loop Complexity Analysis
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Analysing loops is fundamental to complexity analysis. This exercise builds
intuition for recognising common patterns and their complexities.

LEARNING OBJECTIVES
───────────────────
After completing this exercise, you will be able to:
1. Analyse single and nested loops
2. Apply sum and product rules
3. Recognise logarithmic patterns

ESTIMATED TIME
──────────────
20 minutes

DIFFICULTY
──────────
⭐ Easy (1/3)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Analyse These Functions
# ═══════════════════════════════════════════════════════════════════════════════

def func_a(n: int) -> int:
    """
    Analyse the time complexity of this function.

    TODO: What is the time complexity? O(?)
    """
    count = 0
    for i in range(n):
        count += 1
    return count


def func_b(n: int) -> int:
    """
    Analyse the time complexity of this function.

    TODO: What is the time complexity? O(?)
    """
    count = 0
    for i in range(n):
        for j in range(n):
            count += 1
    return count


def func_c(n: int) -> int:
    """
    Analyse the time complexity of this function.

    TODO: What is the time complexity? O(?)
    """
    count = 0
    for i in range(n):
        for j in range(i):
            count += 1
    return count


def func_d(n: int) -> int:
    """
    Analyse the time complexity of this function.

    TODO: What is the time complexity? O(?)
    """
    count = 0
    i = n
    while i > 0:
        count += 1
        i = i // 2
    return count


def func_e(n: int) -> int:
    """
    Analyse the time complexity of this function.

    TODO: What is the time complexity? O(?)
    """
    count = 0
    for i in range(n):
        j = 1
        while j < n:
            count += 1
            j *= 2
    return count


def func_f(n: int) -> int:
    """
    Analyse the time complexity of this function.

    TODO: What is the time complexity? O(?)
    """
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Your Complexity Answers
# ═══════════════════════════════════════════════════════════════════════════════

def get_complexity_answers() -> dict[str, str]:
    """
    Return your complexity analysis for each function.

    Returns:
        Dictionary mapping function name to complexity string.

    Example:
        >>> answers = get_complexity_answers()
        >>> answers['func_a']
        'O(n)'  # or whatever you determine
    """
    # TODO: Fill in your answers
    return {
        "func_a": "O(?)",
        "func_b": "O(?)",
        "func_c": "O(?)",
        "func_d": "O(?)",
        "func_e": "O(?)",
        "func_f": "O(?)",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Count Operations
# ═══════════════════════════════════════════════════════════════════════════════

def count_operations_formula(func_name: str, n: int) -> int:
    """
    Return the exact number of operations (count increments) for the function.

    Derive a formula for each function and implement it.

    Args:
        func_name: Name of function ('func_a', 'func_b', etc.)
        n: Input size.

    Returns:
        Exact number of count increments.

    Example:
        >>> count_operations_formula('func_a', 10)
        10  # Linear function runs exactly n times
        >>> count_operations_formula('func_b', 10)
        100  # n * n = 100
    """
    # TODO: Implement formulas for each function
    # For func_c, think about: 0 + 1 + 2 + ... + (n-1) = ?
    # For func_d, how many times can you halve n before reaching 0?
    # For func_e, think about: n * floor(log2(n))

    if func_name == "func_a":
        raise NotImplementedError("Implement formula for func_a")
    elif func_name == "func_b":
        raise NotImplementedError("Implement formula for func_b")
    elif func_name == "func_c":
        raise NotImplementedError("Implement formula for func_c")
    elif func_name == "func_d":
        raise NotImplementedError("Implement formula for func_d")
    elif func_name == "func_e":
        raise NotImplementedError("Implement formula for func_e")
    elif func_name == "func_f":
        raise NotImplementedError("Implement formula for func_f")
    else:
        raise ValueError(f"Unknown function: {func_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 4: Verify Empirically
# ═══════════════════════════════════════════════════════════════════════════════

def verify_formula(func_name: str, n_values: list[int]) -> list[tuple[int, int, int]]:
    """
    Verify your formula against actual function execution.

    Args:
        func_name: Name of the function to verify.
        n_values: List of input sizes to test.

    Returns:
        List of tuples (n, actual_count, formula_count).
        actual_count should equal formula_count if your formula is correct.
    """
    # Map names to functions
    functions = {
        "func_a": func_a,
        "func_b": func_b,
        "func_c": func_c,
        "func_d": func_d,
        "func_e": func_e,
        "func_f": func_f,
    }

    # TODO: For each n, run the function and compare to formula
    results = []
    for n in n_values:
        actual = functions[func_name](n)
        try:
            formula = count_operations_formula(func_name, n)
        except NotImplementedError:
            formula = -1
        results.append((n, actual, formula))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _test_implementations() -> None:
    """Test all implementations."""
    logger.info("Testing loop analysis...\n")

    # Test Exercise 2
    logger.info("Exercise 2: Your complexity answers")
    answers = get_complexity_answers()
    for func, complexity in answers.items():
        logger.info(f"  {func}: {complexity}")

    # Correct answers for reference
    correct = {
        "func_a": "O(n)",
        "func_b": "O(n^2)",
        "func_c": "O(n^2)",  # Sum 0 to n-1 = n(n-1)/2
        "func_d": "O(log n)",
        "func_e": "O(n log n)",
        "func_f": "O(n^3)",
    }

    logger.info("\n  Correct answers:")
    for func, complexity in correct.items():
        logger.info(f"    {func}: {complexity}")

    # Test Exercise 3 & 4
    logger.info("\nExercise 3 & 4: Formula verification")
    test_values = [5, 10, 20]

    for func_name in ["func_a", "func_b", "func_c", "func_d", "func_e", "func_f"]:
        logger.info(f"\n  {func_name}:")
        results = verify_formula(func_name, test_values)
        all_match = True
        for n, actual, formula in results:
            match = "✓" if actual == formula else "✗"
            if actual != formula:
                all_match = False
            logger.info(f"    n={n:2d}: actual={actual:5d}, formula={formula:5d} {match}")

        if all_match:
            logger.info(f"    All formulas correct!")
        else:
            logger.info(f"    Some formulas need fixing")

    logger.info("\n" + "=" * 60)
    logger.info("Implement count_operations_formula to verify your analysis!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _test_implementations()
