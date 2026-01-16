#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 1, Lab 2: Lambda Calculus Basics
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Lambda calculus, developed by Alonzo Church in the 1930s, provides an
alternative foundation for computation based on function abstraction and
application. While Turing machines model computation imperatively (do this,
then that), lambda calculus models it functionally (apply this function to
that argument). The remarkable Church-Turing thesis states that both models
compute exactly the same class of functions.

This lab explores the core concepts of lambda calculus and demonstrates how
to implement them in Python, bridging the gap between abstract theory and
practical functional programming.

PREREQUISITES
─────────────
- Week 1, Lab 1: Understanding of computational models
- Python: Intermediate (closures, recursion, first-class functions)
- Libraries: None (standard library only)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Explain the three constructs of lambda calculus (variables, abstraction,
   application)
2. Implement Church encodings for booleans and natural numbers
3. Perform beta reduction by hand and programmatically
4. Recognise the connection between lambda calculus and functional programming

ESTIMATED TIME
──────────────
- Reading: 25 minutes
- Coding: 65 minutes
- Total: 90 minutes

DEPENDENCIES
────────────
Python 3.12+ (for pattern matching and type hints)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LAMBDA CALCULUS AST
# ═══════════════════════════════════════════════════════════════════════════════

class LambdaExpr(ABC):
    """Abstract base class for lambda calculus expressions.
    
    Lambda calculus has exactly three constructs:
    1. Variables: x, y, z, ...
    2. Abstractions: λx.M (a function with parameter x and body M)
    3. Applications: M N (applying function M to argument N)
    
    All computation in lambda calculus arises from these three constructs
    through the process of beta reduction: (λx.M) N → M[x := N]
    """
    
    @abstractmethod
    def __str__(self) -> str:
        """Return a string representation of the expression."""
        pass
    
    @abstractmethod
    def free_variables(self) -> set[str]:
        """Return the set of free variables in this expression.
        
        A variable is free if it is not bound by an enclosing lambda.
        For example, in λx.x y, x is bound but y is free.
        """
        pass
    
    @abstractmethod
    def substitute(self, var: str, replacement: LambdaExpr) -> LambdaExpr:
        """Substitute replacement for all free occurrences of var.
        
        This implements the substitution M[x := N] used in beta reduction.
        Care must be taken to avoid variable capture.
        
        Args:
            var: The variable name to replace.
            replacement: The expression to substitute.
        
        Returns:
            A new expression with the substitution applied.
        """
        pass


@dataclass(frozen=True)
class Var(LambdaExpr):
    """A variable in lambda calculus.
    
    Variables are the atomic building blocks of lambda expressions.
    They represent placeholders that may be substituted during reduction.
    
    Attributes:
        name: The variable's identifier.
    
    Example:
        >>> x = Var("x")
        >>> str(x)
        'x'
        >>> x.free_variables()
        {'x'}
    """
    name: str
    
    def __str__(self) -> str:
        return self.name
    
    def free_variables(self) -> set[str]:
        return {self.name}
    
    def substitute(self, var: str, replacement: LambdaExpr) -> LambdaExpr:
        if self.name == var:
            return replacement
        return self


@dataclass(frozen=True)
class Abs(LambdaExpr):
    """A lambda abstraction (function definition).
    
    An abstraction λx.M creates a function with parameter x and body M.
    When applied to an argument N, it produces M[x := N] (beta reduction).
    
    Attributes:
        param: The parameter name.
        body: The function body (a LambdaExpr).
    
    Example:
        >>> identity = Abs("x", Var("x"))
        >>> str(identity)
        'λx.x'
        >>> identity.free_variables()
        set()
    """
    param: str
    body: LambdaExpr
    
    def __str__(self) -> str:
        return f"λ{self.param}.{self.body}"
    
    def free_variables(self) -> set[str]:
        return self.body.free_variables() - {self.param}
    
    def substitute(self, var: str, replacement: LambdaExpr) -> LambdaExpr:
        if var == self.param:
            # The variable is bound here; no substitution in body
            return self
        
        if self.param in replacement.free_variables():
            # Would cause variable capture; need alpha conversion
            # For simplicity, we append a prime to the parameter
            new_param = self.param + "'"
            new_body = self.body.substitute(self.param, Var(new_param))
            return Abs(new_param, new_body.substitute(var, replacement))
        
        return Abs(self.param, self.body.substitute(var, replacement))


@dataclass(frozen=True)
class App(LambdaExpr):
    """A function application.
    
    An application M N represents applying function M to argument N.
    If M is an abstraction λx.B, this can be beta-reduced to B[x := N].
    
    Attributes:
        func: The function expression.
        arg: The argument expression.
    
    Example:
        >>> app = App(Abs("x", Var("x")), Var("y"))
        >>> str(app)
        '(λx.x) y'
    """
    func: LambdaExpr
    arg: LambdaExpr
    
    def __str__(self) -> str:
        func_str = f"({self.func})" if isinstance(self.func, Abs) else str(self.func)
        arg_str = f"({self.arg})" if isinstance(self.arg, (Abs, App)) else str(self.arg)
        return f"{func_str} {arg_str}"
    
    def free_variables(self) -> set[str]:
        return self.func.free_variables() | self.arg.free_variables()
    
    def substitute(self, var: str, replacement: LambdaExpr) -> LambdaExpr:
        return App(
            self.func.substitute(var, replacement),
            self.arg.substitute(var, replacement)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BETA REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def beta_reduce_once(expr: LambdaExpr) -> tuple[LambdaExpr, bool]:
    """Perform a single beta reduction step (leftmost-outermost).
    
    Beta reduction is the fundamental computation rule:
    (λx.M) N → M[x := N]
    
    This function finds the leftmost redex (reducible expression) and
    reduces it. This strategy is called "normal order" reduction.
    
    Args:
        expr: The expression to reduce.
    
    Returns:
        A tuple (reduced_expr, was_reduced) where was_reduced indicates
        whether a reduction was performed.
    
    Example:
        >>> expr = App(Abs("x", Var("x")), Var("y"))
        >>> reduced, did_reduce = beta_reduce_once(expr)
        >>> str(reduced)
        'y'
        >>> did_reduce
        True
    """
    match expr:
        case Var(_):
            return expr, False
        
        case App(Abs(param, body), arg):
            # Found a redex! Perform beta reduction
            logger.debug("Beta reducing: (%s) %s", Abs(param, body), arg)
            return body.substitute(param, arg), True
        
        case App(func, arg):
            # Try to reduce the function first
            reduced_func, did_reduce = beta_reduce_once(func)
            if did_reduce:
                return App(reduced_func, arg), True
            
            # Then try the argument
            reduced_arg, did_reduce = beta_reduce_once(arg)
            return App(func, reduced_arg), did_reduce
        
        case Abs(param, body):
            # Reduce inside the abstraction body
            reduced_body, did_reduce = beta_reduce_once(body)
            return Abs(param, reduced_body), did_reduce
        
        case _:
            return expr, False


def beta_reduce(expr: LambdaExpr, max_steps: int = 100) -> LambdaExpr:
    """Fully beta-reduce an expression to normal form.
    
    Repeatedly applies beta reduction until no more reductions are possible
    or the step limit is reached.
    
    Args:
        expr: The expression to reduce.
        max_steps: Maximum number of reduction steps.
    
    Returns:
        The fully reduced expression (normal form if reached).
    
    Raises:
        RuntimeError: If reduction does not terminate within max_steps.
    
    Example:
        >>> expr = App(Abs("x", App(Var("x"), Var("x"))), Var("y"))
        >>> str(beta_reduce(expr))
        'y y'
    """
    logger.debug("Starting reduction of: %s", expr)
    
    for step in range(max_steps):
        reduced, did_reduce = beta_reduce_once(expr)
        if not did_reduce:
            logger.debug("Reached normal form in %d steps", step)
            return reduced
        expr = reduced
        logger.debug("Step %d: %s", step + 1, expr)
    
    raise RuntimeError(
        f"Reduction did not terminate within {max_steps} steps. "
        f"Expression may diverge."
    )


def trace_reduction(expr: LambdaExpr, max_steps: int = 100) -> list[LambdaExpr]:
    """Trace all steps of beta reduction.
    
    Args:
        expr: The expression to reduce.
        max_steps: Maximum number of reduction steps.
    
    Returns:
        A list of expressions showing each reduction step.
    """
    trace = [expr]
    
    for _ in range(max_steps):
        reduced, did_reduce = beta_reduce_once(expr)
        if not did_reduce:
            break
        trace.append(reduced)
        expr = reduced
    
    return trace


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CHURCH ENCODINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Church Booleans
# ───────────────
# TRUE  = λx.λy.x  (selects first argument)
# FALSE = λx.λy.y  (selects second argument)

TRUE = Abs("x", Abs("y", Var("x")))
FALSE = Abs("x", Abs("y", Var("y")))


def church_not() -> LambdaExpr:
    """Return the Church encoding of NOT.
    
    NOT = λp.p FALSE TRUE
    
    If p is TRUE, NOT p reduces to FALSE.
    If p is FALSE, NOT p reduces to TRUE.
    
    Returns:
        The NOT combinator as a LambdaExpr.
    """
    p = Var("p")
    return Abs("p", App(App(p, FALSE), TRUE))


def church_and() -> LambdaExpr:
    """Return the Church encoding of AND.
    
    AND = λp.λq.p q FALSE
    
    If p is FALSE, return FALSE immediately.
    If p is TRUE, return q.
    
    Returns:
        The AND combinator as a LambdaExpr.
    """
    p, q = Var("p"), Var("q")
    return Abs("p", Abs("q", App(App(p, q), FALSE)))


def church_or() -> LambdaExpr:
    """Return the Church encoding of OR.
    
    OR = λp.λq.p TRUE q
    
    If p is TRUE, return TRUE immediately.
    If p is FALSE, return q.
    
    Returns:
        The OR combinator as a LambdaExpr.
    """
    p, q = Var("p"), Var("q")
    return Abs("p", Abs("q", App(App(p, TRUE), q)))


def church_if() -> LambdaExpr:
    """Return the Church encoding of IF-THEN-ELSE.
    
    IF = λc.λt.λe.c t e
    
    If c is TRUE, return t (then branch).
    If c is FALSE, return e (else branch).
    
    Returns:
        The IF combinator as a LambdaExpr.
    """
    c, t, e = Var("c"), Var("t"), Var("e")
    return Abs("c", Abs("t", Abs("e", App(App(c, t), e))))


# Church Numerals
# ───────────────
# n = λf.λx.f^n(x)  (apply f n times to x)
#
# 0 = λf.λx.x
# 1 = λf.λx.f x
# 2 = λf.λx.f (f x)
# 3 = λf.λx.f (f (f x))

def church_numeral(n: int) -> LambdaExpr:
    """Create the Church encoding of a natural number.
    
    The Church numeral n represents the number as a function that
    applies its first argument n times to its second argument.
    
    Args:
        n: A non-negative integer.
    
    Returns:
        The Church numeral for n.
    
    Raises:
        ValueError: If n is negative.
    
    Example:
        >>> str(church_numeral(0))
        'λf.λx.x'
        >>> str(church_numeral(2))
        'λf.λx.f (f x)'
    """
    if n < 0:
        raise ValueError("Church numerals only represent non-negative integers")
    
    f, x = Var("f"), Var("x")
    
    # Build f^n(x)
    body: LambdaExpr = x
    for _ in range(n):
        body = App(f, body)
    
    return Abs("f", Abs("x", body))


def church_succ() -> LambdaExpr:
    """Return the Church encoding of the successor function.
    
    SUCC = λn.λf.λx.f (n f x)
    
    SUCC n applies f one more time than n does.
    
    Returns:
        The SUCC combinator as a LambdaExpr.
    """
    n, f, x = Var("n"), Var("f"), Var("x")
    return Abs("n", Abs("f", Abs("x", App(f, App(App(n, f), x)))))


def church_add() -> LambdaExpr:
    """Return the Church encoding of addition.
    
    ADD = λm.λn.λf.λx.m f (n f x)
    
    m + n applies f m times, then n times.
    
    Returns:
        The ADD combinator as a LambdaExpr.
    """
    m, n, f, x = Var("m"), Var("n"), Var("f"), Var("x")
    return Abs("m", Abs("n", Abs("f", Abs("x", 
        App(App(m, f), App(App(n, f), x))
    ))))


def church_mult() -> LambdaExpr:
    """Return the Church encoding of multiplication.
    
    MULT = λm.λn.λf.m (n f)
    
    m × n applies (n f) m times, which applies f m×n times.
    
    Returns:
        The MULT combinator as a LambdaExpr.
    """
    m, n, f = Var("m"), Var("n"), Var("f")
    return Abs("m", Abs("n", Abs("f", App(m, App(n, f)))))


def church_exp() -> LambdaExpr:
    """Return the Church encoding of exponentiation.
    
    EXP = λm.λn.n m
    
    m^n applies m n times (function composition).
    
    Returns:
        The EXP combinator as a LambdaExpr.
    """
    m, n = Var("m"), Var("n")
    return Abs("m", Abs("n", App(n, m)))


def church_iszero() -> LambdaExpr:
    """Return the Church encoding of IS_ZERO.
    
    IS_ZERO = λn.n (λx.FALSE) TRUE
    
    If n is zero, TRUE is returned unchanged.
    If n > 0, f is applied at least once, returning FALSE.
    
    Returns:
        The IS_ZERO combinator as a LambdaExpr.
    """
    n, x = Var("n"), Var("x")
    const_false = Abs("x", FALSE)
    return Abs("n", App(App(n, const_false), TRUE))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FIXED-POINT COMBINATORS
# ═══════════════════════════════════════════════════════════════════════════════

def y_combinator() -> LambdaExpr:
    """Return the Y combinator (fixed-point combinator).
    
    Y = λf.(λx.f (x x)) (λx.f (x x))
    
    The Y combinator allows defining recursive functions in lambda calculus.
    For any function g, Y g is a fixed point: Y g = g (Y g).
    
    Note: This combinator does not terminate under strict evaluation.
    Use the Z combinator for strict languages.
    
    Returns:
        The Y combinator as a LambdaExpr.
    """
    f, x = Var("f"), Var("x")
    inner = Abs("x", App(f, App(x, x)))
    return Abs("f", App(inner, inner))


def omega_combinator() -> LambdaExpr:
    """Return the Omega combinator (divergent expression).
    
    Ω = (λx.x x) (λx.x x)
    
    This expression has no normal form; beta reduction loops forever.
    It demonstrates that not all lambda expressions terminate.
    
    Returns:
        The Omega combinator as a LambdaExpr.
    """
    x = Var("x")
    self_app = Abs("x", App(x, x))
    return App(self_app, self_app)


def s_combinator() -> LambdaExpr:
    """Return the S combinator.
    
    S = λx.λy.λz.x z (y z)
    
    S is one of the combinators in combinatory logic. Together with K,
    it can express any lambda term.
    
    Returns:
        The S combinator as a LambdaExpr.
    """
    x, y, z = Var("x"), Var("y"), Var("z")
    return Abs("x", Abs("y", Abs("z", App(App(x, z), App(y, z)))))


def k_combinator() -> LambdaExpr:
    """Return the K combinator.
    
    K = λx.λy.x
    
    K returns its first argument and discards its second.
    This is the same as Church TRUE.
    
    Returns:
        The K combinator as a LambdaExpr.
    """
    return Abs("x", Abs("y", Var("x")))


def i_combinator() -> LambdaExpr:
    """Return the I combinator (identity).
    
    I = λx.x
    
    I returns its argument unchanged. It can be derived from S and K:
    I = S K K
    
    Returns:
        The I combinator as a LambdaExpr.
    """
    return Abs("x", Var("x"))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CONVERSION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def church_to_int(expr: LambdaExpr, max_steps: int = 100) -> int:
    """Convert a Church numeral to a Python integer.
    
    This function applies the Church numeral to increment and 0,
    then counts how many times increment was called.
    
    Args:
        expr: A Church numeral expression.
        max_steps: Maximum reduction steps.
    
    Returns:
        The integer value of the Church numeral.
    
    Example:
        >>> church_to_int(church_numeral(5))
        5
    """
    # We'll reduce (n SUCC 0) and count the SUCCs
    # For simplicity, use a different approach: reduce and parse
    
    # Create a counter using unique variable names
    f = Var("__inc__")
    x = Var("__zero__")
    
    # Apply the numeral to f and x
    applied = App(App(expr, f), x)
    reduced = beta_reduce(applied, max_steps)
    
    # Count nested applications of f
    count = 0
    current = reduced
    while isinstance(current, App) and str(current.func) == "__inc__":
        count += 1
        current = current.arg
    
    return count


def church_to_bool(expr: LambdaExpr, max_steps: int = 100) -> bool:
    """Convert a Church boolean to a Python boolean.
    
    This function applies the Church boolean to True and False markers,
    then checks which was selected.
    
    Args:
        expr: A Church boolean expression.
        max_steps: Maximum reduction steps.
    
    Returns:
        True if expr is Church TRUE, False if Church FALSE.
    
    Example:
        >>> church_to_bool(TRUE)
        True
        >>> church_to_bool(FALSE)
        False
    """
    true_marker = Var("__TRUE__")
    false_marker = Var("__FALSE__")
    
    applied = App(App(expr, true_marker), false_marker)
    reduced = beta_reduce(applied, max_steps)
    
    if str(reduced) == "__TRUE__":
        return True
    elif str(reduced) == "__FALSE__":
        return False
    else:
        raise ValueError(f"Expression is not a Church boolean: {reduced}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: PYTHON INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Type alias for Church-style functions
ChurchFunc = Callable[[Callable], Callable]


def python_church_numeral(n: int) -> ChurchFunc:
    """Create a Church numeral as a Python function.
    
    This demonstrates that Church encodings work in any language with
    first-class functions.
    
    Args:
        n: A non-negative integer.
    
    Returns:
        A Python function representing the Church numeral.
    
    Example:
        >>> three = python_church_numeral(3)
        >>> three(lambda x: x + 1)(0)
        3
    """
    def numeral(f: Callable) -> Callable:
        def apply_n_times(x):
            result = x
            for _ in range(n):
                result = f(result)
            return result
        return apply_n_times
    return numeral


def python_church_true(x: object) -> Callable:
    """Church TRUE as a Python function."""
    return lambda y: x


def python_church_false(x: object) -> Callable:
    """Church FALSE as a Python function."""
    return lambda y: y


def python_church_succ(n: ChurchFunc) -> ChurchFunc:
    """Church SUCC as a Python function."""
    def succ(f: Callable) -> Callable:
        return lambda x: f(n(f)(x))
    return succ


def python_church_add(m: ChurchFunc, n: ChurchFunc) -> ChurchFunc:
    """Church ADD as a Python function."""
    def add(f: Callable) -> Callable:
        return lambda x: m(f)(n(f)(x))
    return add


def python_church_mult(m: ChurchFunc, n: ChurchFunc) -> ChurchFunc:
    """Church MULT as a Python function."""
    def mult(f: Callable) -> Callable:
        return m(n(f))
    return mult


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def demo_basic_reduction() -> None:
    """Demonstrate basic beta reduction."""
    logger.info("Running basic reduction demonstration")
    print("=" * 60)
    print("DEMO: Basic Beta Reduction")
    print("=" * 60)
    
    # Identity applied to y
    identity = Abs("x", Var("x"))
    expr = App(identity, Var("y"))
    print(f"\n1. Identity application:")
    print(f"   (λx.x) y")
    print(f"   → {beta_reduce(expr)}")
    
    # Nested reduction
    k = Abs("x", Abs("y", Var("x")))
    expr2 = App(App(k, Var("a")), Var("b"))
    print(f"\n2. K combinator:")
    print(f"   K a b = (λx.λy.x) a b")
    trace = trace_reduction(expr2)
    for i, step in enumerate(trace):
        prefix = "   " if i == 0 else "   → "
        print(f"{prefix}{step}")
    
    print()


def demo_church_booleans() -> None:
    """Demonstrate Church boolean operations."""
    logger.info("Running Church booleans demonstration")
    print("=" * 60)
    print("DEMO: Church Booleans")
    print("=" * 60)
    
    print(f"\nTRUE  = {TRUE}")
    print(f"FALSE = {FALSE}")
    print(f"NOT   = {church_not()}")
    print(f"AND   = {church_and()}")
    print(f"OR    = {church_or()}")
    
    # Test NOT
    not_true = App(church_not(), TRUE)
    not_false = App(church_not(), FALSE)
    
    print(f"\nNOT TRUE  → {church_to_bool(not_true)}")
    print(f"NOT FALSE → {church_to_bool(not_false)}")
    
    # Test AND
    and_expr = church_and()
    cases = [
        (TRUE, TRUE, "TRUE AND TRUE"),
        (TRUE, FALSE, "TRUE AND FALSE"),
        (FALSE, TRUE, "FALSE AND TRUE"),
        (FALSE, FALSE, "FALSE AND FALSE"),
    ]
    
    print()
    for p, q, label in cases:
        result = App(App(and_expr, p), q)
        print(f"{label} → {church_to_bool(result)}")
    
    print()


def demo_church_numerals() -> None:
    """Demonstrate Church numeral arithmetic."""
    logger.info("Running Church numerals demonstration")
    print("=" * 60)
    print("DEMO: Church Numerals")
    print("=" * 60)
    
    print("\nChurch numerals:")
    for i in range(5):
        print(f"  {i} = {church_numeral(i)}")
    
    print(f"\nSUCC = {church_succ()}")
    print(f"ADD  = {church_add()}")
    print(f"MULT = {church_mult()}")
    
    # Test successor
    two = church_numeral(2)
    succ_two = App(church_succ(), two)
    print(f"\nSUCC 2 = {church_to_int(succ_two)}")
    
    # Test addition
    three = church_numeral(3)
    add_expr = App(App(church_add(), two), three)
    print(f"2 + 3 = {church_to_int(add_expr)}")
    
    # Test multiplication
    mult_expr = App(App(church_mult(), two), three)
    print(f"2 × 3 = {church_to_int(mult_expr)}")
    
    # Test IS_ZERO
    print(f"\nIS_ZERO 0 → {church_to_bool(App(church_iszero(), church_numeral(0)))}")
    print(f"IS_ZERO 1 → {church_to_bool(App(church_iszero(), church_numeral(1)))}")
    
    print()


def demo_python_church() -> None:
    """Demonstrate Church encodings as Python functions."""
    logger.info("Running Python Church encoding demonstration")
    print("=" * 60)
    print("DEMO: Church Encodings in Python")
    print("=" * 60)
    
    inc = lambda x: x + 1
    
    print("\nChurch numerals as Python functions:")
    for i in range(5):
        numeral = python_church_numeral(i)
        value = numeral(inc)(0)
        print(f"  {i} → applies inc {value} times")
    
    # Arithmetic
    two = python_church_numeral(2)
    three = python_church_numeral(3)
    
    sum_result = python_church_add(two, three)(inc)(0)
    prod_result = python_church_mult(two, three)(inc)(0)
    
    print(f"\n2 + 3 = {sum_result}")
    print(f"2 × 3 = {prod_result}")
    
    # Booleans
    print("\nChurch booleans as Python functions:")
    print(f"  TRUE(1)(0)  = {python_church_true(1)(0)}")
    print(f"  FALSE(1)(0) = {python_church_false(1)(0)}")
    
    print()


def demo_combinators() -> None:
    """Demonstrate famous combinators."""
    logger.info("Running combinators demonstration")
    print("=" * 60)
    print("DEMO: Famous Combinators")
    print("=" * 60)
    
    print(f"\nI (Identity)    = {i_combinator()}")
    print(f"K (Constant)    = {k_combinator()}")
    print(f"S (Substitution) = {s_combinator()}")
    print(f"Y (Fixed-point)  = {y_combinator()}")
    print(f"Ω (Omega)       = {omega_combinator()}")
    
    # Verify I = S K K
    s, k = s_combinator(), k_combinator()
    skk = App(App(s, k), k)
    skk_applied = App(skk, Var("z"))
    reduced = beta_reduce(skk_applied)
    
    print(f"\nVerifying I = S K K:")
    print(f"  S K K z → {reduced}")
    print(f"  I z     → {beta_reduce(App(i_combinator(), Var('z')))}")
    
    print()


def run_all_demos() -> None:
    """Run all demonstration functions."""
    demo_basic_reduction()
    demo_church_booleans()
    demo_church_numerals()
    demo_python_church()
    demo_combinators()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 1: Church Predecessor                                                ║
║                                                                               ║
║ Implement the predecessor function for Church numerals.                       ║
║                                                                               ║
║ PRED 0 = 0                                                                   ║
║ PRED n = n - 1 (for n > 0)                                                   ║
║                                                                               ║
║ Hint: Use pairs. Create a function that transforms (n, n-1) to (n+1, n).     ║
║       Starting from (0, 0), apply n times to get (n, n-1).                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def church_pred() -> LambdaExpr:
    """EXERCISE: Implement the predecessor function.
    
    Returns:
        The PRED combinator as a LambdaExpr.
    """
    # TODO: Implement
    # Hint: You'll need Church pairs first
    return Abs("n", Var("n"))  # Placeholder


"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 2: Church Subtraction                                                ║
║                                                                               ║
║ Using PRED, implement subtraction: SUB m n = m - n (or 0 if m < n).          ║
║                                                                               ║
║ Hint: SUB = λm.λn.n PRED m                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def church_sub() -> LambdaExpr:
    """EXERCISE: Implement subtraction using PRED.
    
    Returns:
        The SUB combinator as a LambdaExpr.
    """
    # TODO: Implement
    return Abs("m", Abs("n", Var("m")))  # Placeholder


"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 3: Less Than or Equal                                                ║
║                                                                               ║
║ Implement LEQ: LEQ m n returns TRUE if m ≤ n, FALSE otherwise.               ║
║                                                                               ║
║ Hint: m ≤ n iff m - n = 0.                                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


def church_leq() -> LambdaExpr:
    """EXERCISE: Implement less-than-or-equal comparison.
    
    Returns:
        The LEQ combinator as a LambdaExpr.
    """
    # TODO: Implement
    return Abs("m", Abs("n", TRUE))  # Placeholder


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Lambda Calculus Basics - Week 1, Lab 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_1_02_lambda_calculus.py --demo
  python lab_1_02_lambda_calculus.py --reduce "((λx.x) y)"
  python lab_1_02_lambda_calculus.py --church 5

Note: The --reduce option uses a simple parser for basic expressions.
        """
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations"
    )
    parser.add_argument(
        "--church",
        type=int,
        metavar="N",
        help="Display Church numeral for N"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        print("\n" + "═" * 60)
        print("  WEEK 1, LAB 2: LAMBDA CALCULUS BASICS")
        print("═" * 60 + "\n")
        run_all_demos()
        print("=" * 60)
        print("Exercises to complete in code:")
        print("  1. church_pred() - Predecessor function")
        print("  2. church_sub() - Subtraction")
        print("  3. church_leq() - Less than or equal")
        print("=" * 60)
    elif args.church is not None:
        numeral = church_numeral(args.church)
        print(f"{args.church} = {numeral}")
        
        if args.verbose:
            print(f"\nAs Python: applies f {args.church} times to x")
            py_numeral = python_church_numeral(args.church)
            result = py_numeral(lambda x: x + 1)(0)
            print(f"Verification: {result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
