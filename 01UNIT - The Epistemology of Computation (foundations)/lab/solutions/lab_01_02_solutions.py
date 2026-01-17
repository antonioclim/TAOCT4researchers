"""
Week 1: Lab 2 Solutions — Lambda Calculus Exercises.

This module contains complete solutions for the lambda calculus exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
These solutions are provided for educational purposes only.
"""

import sys
sys.path.insert(0, '..')

from lab_1_02_lambda_calculus import (
    LambdaExpr, Var, Abs, App,
    beta_reduce, trace_reduction, church_to_int, church_to_bool,
    TRUE, FALSE, church_numeral, church_succ, church_add,
    python_church_numeral
)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Church Predecessor Function
# ═══════════════════════════════════════════════════════════════════════════════

def church_pair() -> LambdaExpr:
    """Create the Church pair constructor.
    
    PAIR = λx.λy.λf.f x y
    
    A pair stores two values and applies a selector function to them.
    
    Returns:
        The PAIR combinator as a lambda expression.
    """
    x = Var("x")
    y = Var("y")
    f = Var("f")
    return Abs("x", Abs("y", Abs("f", App(App(f, x), y))))


def church_fst() -> LambdaExpr:
    """Create the first element selector.
    
    FST = λp.p TRUE = λp.p (λx.λy.x)
    
    Returns:
        The FST combinator as a lambda expression.
    """
    p = Var("p")
    return Abs("p", App(p, TRUE))


def church_snd() -> LambdaExpr:
    """Create the second element selector.
    
    SND = λp.p FALSE = λp.p (λx.λy.y)
    
    Returns:
        The SND combinator as a lambda expression.
    """
    p = Var("p")
    return Abs("p", App(p, FALSE))


def church_shift() -> LambdaExpr:
    """Create the shift function for predecessor.
    
    SHIFT = λp.PAIR (SUCC (FST p)) (FST p)
    
    Transforms (k, prev) into (k+1, k).
    This is the key insight for implementing predecessor.
    
    Returns:
        The SHIFT combinator as a lambda expression.
    """
    p = Var("p")
    fst_p = App(church_fst(), p)
    succ_fst_p = App(church_succ(), fst_p)
    pair = church_pair()
    return Abs("p", App(App(pair, succ_fst_p), fst_p))


def church_pred() -> LambdaExpr:
    """Create the predecessor function for Church numerals.
    
    PRED = λn.SND (n SHIFT (PAIR 0 0))
    
    Algorithm:
        1. Start with pair (0, 0)
        2. Apply SHIFT n times: (0,0) → (1,0) → (2,1) → ... → (n, n-1)
        3. Take the second element to get n-1
    
    Returns:
        The PRED combinator as a lambda expression.
    
    Example:
        PRED 0 = 0
        PRED 3 = 2
    """
    n = Var("n")
    zero = church_numeral(0)
    pair = church_pair()
    zero_pair = App(App(pair, zero), zero)
    shift = church_shift()
    snd = church_snd()
    
    # n SHIFT (PAIR 0 0) applies SHIFT n times to the initial pair
    n_shifted = App(App(n, shift), zero_pair)
    
    return Abs("n", App(snd, n_shifted))


def test_church_pred():
    """Test the Church predecessor function."""
    print("Church Predecessor Tests")
    print("=" * 50)
    
    pred = church_pred()
    
    test_cases = [(0, 0), (1, 0), (2, 1), (3, 2), (5, 4), (10, 9)]
    
    all_passed = True
    for n, expected in test_cases:
        numeral = church_numeral(n)
        pred_n = App(pred, numeral)
        result = church_to_int(pred_n)
        
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} PRED {n} = {result} (expected: {expected})")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: Church Subtraction
# ═══════════════════════════════════════════════════════════════════════════════

def church_sub() -> LambdaExpr:
    """Create the subtraction function for Church numerals.
    
    SUB = λm.λn.n PRED m
    
    Applies PRED to m exactly n times.
    Result is m - n, or 0 if n > m (natural subtraction).
    
    Returns:
        The SUB combinator as a lambda expression.
    """
    m = Var("m")
    n = Var("n")
    pred = church_pred()
    
    # n PRED m applies PRED to m, n times
    return Abs("m", Abs("n", App(App(n, pred), m)))


def test_church_sub():
    """Test the Church subtraction function."""
    print("Church Subtraction Tests")
    print("=" * 50)
    
    sub = church_sub()
    
    test_cases = [
        (5, 3, 2),
        (10, 4, 6),
        (3, 3, 0),
        (2, 5, 0),  # 2 - 5 = 0 (natural subtraction)
        (7, 0, 7),
        (0, 0, 0),
    ]
    
    all_passed = True
    for m, n, expected in test_cases:
        m_num = church_numeral(m)
        n_num = church_numeral(n)
        diff = App(App(sub, m_num), n_num)
        result = church_to_int(diff)
        
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} {m} - {n} = {result} (expected: {expected})")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: Church Less-Than-Or-Equal (LEQ)
# ═══════════════════════════════════════════════════════════════════════════════

def church_iszero() -> LambdaExpr:
    """Create the IS_ZERO predicate.
    
    ISZERO = λn.n (λx.FALSE) TRUE
    
    Returns TRUE if n is 0, FALSE otherwise.
    """
    n = Var("n")
    x = Var("x")
    always_false = Abs("x", FALSE)
    return Abs("n", App(App(n, always_false), TRUE))


def church_leq() -> LambdaExpr:
    """Create the less-than-or-equal predicate.
    
    LEQ = λm.λn.ISZERO (SUB m n)
    
    m ≤ n if and only if m - n = 0
    
    Returns:
        The LEQ predicate as a lambda expression.
    """
    m = Var("m")
    n = Var("n")
    sub = church_sub()
    iszero = church_iszero()
    
    diff = App(App(sub, m), n)
    return Abs("m", Abs("n", App(iszero, diff)))


def test_church_leq():
    """Test the Church less-than-or-equal predicate."""
    print("Church LEQ Tests")
    print("=" * 50)
    
    leq = church_leq()
    
    test_cases = [
        (0, 0, True),
        (0, 5, True),
        (3, 5, True),
        (5, 5, True),
        (5, 3, False),
        (10, 5, False),
    ]
    
    all_passed = True
    for m, n, expected in test_cases:
        m_num = church_numeral(m)
        n_num = church_numeral(n)
        result_expr = App(App(leq, m_num), n_num)
        result = church_to_bool(result_expr)
        
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} {m} ≤ {n} = {result} (expected: {expected})")
        
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON IMPLEMENTATIONS (for comparison and testing)
# ═══════════════════════════════════════════════════════════════════════════════

def python_church_pair(x, y):
    """Python implementation of Church pair."""
    return lambda f: f(x)(y)


def python_church_fst(p):
    """Python implementation of FST."""
    return p(lambda x: lambda y: x)


def python_church_snd(p):
    """Python implementation of SND."""
    return p(lambda x: lambda y: y)


def python_church_succ(n):
    """Python implementation of SUCC."""
    return lambda f: lambda x: f(n(f)(x))


def python_church_shift(p):
    """Python implementation of SHIFT."""
    k = python_church_fst(p)
    return python_church_pair(python_church_succ(k), k)


def python_church_pred(n):
    """Python implementation of PRED."""
    zero = python_church_numeral(0)
    zero_pair = python_church_pair(zero, zero)
    result_pair = n(python_church_shift)(zero_pair)
    return python_church_snd(result_pair)


def python_church_to_int(n):
    """Convert Python Church numeral to integer."""
    return n(lambda x: x + 1)(0)


def test_python_implementations():
    """Test the Python implementations."""
    print("Python Church Encoding Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Test predecessor
    for i in range(10):
        n = python_church_numeral(i)
        pred_n = python_church_pred(n)
        result = python_church_to_int(pred_n)
        expected = max(0, i - 1)
        
        passed = result == expected
        if not passed:
            all_passed = False
            print(f"  ✗ PRED {i} = {result} (expected: {expected})")
    
    if all_passed:
        print("  ✓ All predecessor tests passed")
    
    # Test pairs
    for a, b in [(1, 2), (3, 4), (0, 5)]:
        pair = python_church_pair(
            python_church_numeral(a),
            python_church_numeral(b)
        )
        fst_result = python_church_to_int(python_church_fst(pair))
        snd_result = python_church_to_int(python_church_snd(pair))
        
        if fst_result != a or snd_result != b:
            all_passed = False
            print(f"  ✗ Pair ({a}, {b}): FST={fst_result}, SND={snd_result}")
    
    print("  ✓ All pair tests passed")
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("═" * 60)
    print("WEEK 1: LAB 2 SOLUTIONS — LAMBDA CALCULUS EXERCISES")
    print("═" * 60)
    print()
    
    results = []
    
    results.append(("Church Predecessor", test_church_pred()))
    results.append(("Church Subtraction", test_church_sub()))
    results.append(("Church LEQ", test_church_leq()))
    results.append(("Python Implementations", test_python_implementations()))
    
    print("═" * 60)
    print("SUMMARY")
    print("═" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:25s} {status}")
    
    all_passed = all(p for _, p in results)
    print()
    print(f"Overall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
