# ruff: noqa

"""
Week 1: Lab 3 Solutions — AST Interpreter Exercises.

This module contains complete solutions for the interpreter exercises.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
These solutions are provided for educational purposes only.
"""

import sys
sys.path.insert(0, '..')

from dataclasses import dataclass
from lab_1_03_ast_interpreter import (
    Expr, Num, BinOp, UnaryOp, Lexer, TokenType, Evaluator
)


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1: Logical Operators (AND, OR, NOT)
# ═══════════════════════════════════════════════════════════════════════════════

# To implement logical operators, we need to:
# 1. Add token types: AND, OR, NOT
# 2. Update the lexer to recognise these keywords
# 3. Update the parser to handle them at appropriate precedence
# 4. Update the evaluator to compute them

# For this solution, we show how to extend the existing implementation.

class ExtendedTokenType(TokenType):
    """Extended token types including logical operators."""
    AND = 'AND'
    OR = 'OR'
    NOT = 'NOT'


def create_extended_lexer(text: str) -> Lexer:
    """Create a lexer that recognises logical operators.
    
    Args:
        text: The source code to tokenise.
    
    Returns:
        A Lexer with extended keyword support.
    """
    lexer = Lexer(text)
    # Add logical keywords
    lexer.KEYWORDS['and'] = ExtendedTokenType.AND
    lexer.KEYWORDS['or'] = ExtendedTokenType.OR
    lexer.KEYWORDS['not'] = ExtendedTokenType.NOT
    return lexer


# The evaluator needs to handle AND, OR, NOT in BinOp and UnaryOp cases.
# Here's a demonstration of how to extend the evaluator:

def evaluate_with_logic(expr: Expr, env: dict | None = None) -> float:
    """Evaluate expressions with logical operator support.
    
    Logical semantics:
        - 0 is falsy, non-zero is truthy
        - and: returns 1 if both operands are truthy, else 0
        - or: returns 1 if either operand is truthy, else 0
        - not: returns 1 if operand is falsy, else 0
    
    Args:
        expr: The expression to evaluate.
        env: Variable environment.
    
    Returns:
        The computed value.
    """
    if env is None:
        env = {}
    
    evaluator = Evaluator()
    
    # Store original evaluate method
    original_evaluate = evaluator.evaluate
    
    def extended_evaluate(e: Expr, environment: dict) -> float:
        """Extended evaluation with logical operators."""
        match e:
            case BinOp(left, 'and', right):
                left_val = extended_evaluate(left, environment)
                # Short-circuit evaluation
                if left_val == 0:
                    return 0.0
                right_val = extended_evaluate(right, environment)
                return 1.0 if right_val != 0 else 0.0
            
            case BinOp(left, 'or', right):
                left_val = extended_evaluate(left, environment)
                # Short-circuit evaluation
                if left_val != 0:
                    return 1.0
                right_val = extended_evaluate(right, environment)
                return 1.0 if right_val != 0 else 0.0
            
            case UnaryOp('not', operand):
                val = extended_evaluate(operand, environment)
                return 1.0 if val == 0 else 0.0
            
            case _:
                return original_evaluate(e, environment)
    
    evaluator.evaluate = extended_evaluate
    return evaluator.evaluate(expr, env)


def test_logical_operators():
    """Test the logical operator implementation."""
    print("Logical Operators Tests")
    print("=" * 50)
    
    # Since we can't easily modify the parser, we test with AST directly
    test_cases = [
        # AND tests
        (BinOp(Num(1), 'and', Num(1)), 1.0, "1 and 1"),
        (BinOp(Num(1), 'and', Num(0)), 0.0, "1 and 0"),
        (BinOp(Num(0), 'and', Num(1)), 0.0, "0 and 1"),
        (BinOp(Num(0), 'and', Num(0)), 0.0, "0 and 0"),
        
        # OR tests
        (BinOp(Num(1), 'or', Num(1)), 1.0, "1 or 1"),
        (BinOp(Num(1), 'or', Num(0)), 1.0, "1 or 0"),
        (BinOp(Num(0), 'or', Num(1)), 1.0, "0 or 1"),
        (BinOp(Num(0), 'or', Num(0)), 0.0, "0 or 0"),
        
        # NOT tests
        (UnaryOp('not', Num(1)), 0.0, "not 1"),
        (UnaryOp('not', Num(0)), 1.0, "not 0"),
        (UnaryOp('not', Num(5)), 0.0, "not 5"),
        
        # Complex expressions
        (BinOp(BinOp(Num(1), 'and', Num(1)), 'or', Num(0)), 1.0, "(1 and 1) or 0"),
        (UnaryOp('not', BinOp(Num(0), 'or', Num(0))), 1.0, "not (0 or 0)"),
    ]
    
    all_passed = True
    for expr, expected, description in test_cases:
        result = evaluate_with_logic(expr)
        passed = abs(result - expected) < 1e-10
        status = "✓" if passed else "✗"
        print(f"  {status} {description} = {result} (expected: {expected})")
        if not passed:
            all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2: List Operations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ListExpr:
    """List literal expression."""
    elements: tuple[Expr, ...]


@dataclass(frozen=True)
class ListOp:
    """List operation expression.
    
    Supported operations:
        - head: First element
        - tail: All but first
        - cons: Add element to front
        - length: Number of elements
        - empty: Check if empty
    """
    op: str
    args: tuple[Expr, ...]


# For list values at runtime:
@dataclass
class ListValue:
    """Runtime list value."""
    elements: list


def evaluate_with_lists(expr: Expr, env: dict | None = None):
    """Evaluate expressions with list support.
    
    Args:
        expr: The expression to evaluate.
        env: Variable environment.
    
    Returns:
        The computed value (may be a ListValue).
    """
    if env is None:
        env = {}
    
    evaluator = Evaluator()
    original_evaluate = evaluator.evaluate
    
    def extended_evaluate(e: Expr, environment: dict):
        match e:
            case ListExpr(elements):
                return ListValue([extended_evaluate(el, environment) for el in elements])
            
            case ListOp('head', (lst_expr,)):
                lst = extended_evaluate(lst_expr, environment)
                if not isinstance(lst, ListValue) or not lst.elements:
                    raise ValueError("Cannot take head of empty or non-list")
                return lst.elements[0]
            
            case ListOp('tail', (lst_expr,)):
                lst = extended_evaluate(lst_expr, environment)
                if not isinstance(lst, ListValue) or not lst.elements:
                    raise ValueError("Cannot take tail of empty or non-list")
                return ListValue(lst.elements[1:])
            
            case ListOp('cons', (elem_expr, lst_expr)):
                elem = extended_evaluate(elem_expr, environment)
                lst = extended_evaluate(lst_expr, environment)
                if not isinstance(lst, ListValue):
                    raise ValueError("cons requires a list")
                return ListValue([elem] + lst.elements)
            
            case ListOp('length', (lst_expr,)):
                lst = extended_evaluate(lst_expr, environment)
                if not isinstance(lst, ListValue):
                    raise ValueError("length requires a list")
                return float(len(lst.elements))
            
            case ListOp('empty', (lst_expr,)):
                lst = extended_evaluate(lst_expr, environment)
                if not isinstance(lst, ListValue):
                    raise ValueError("empty requires a list")
                return 1.0 if not lst.elements else 0.0
            
            case _:
                return original_evaluate(e, environment)
    
    evaluator.evaluate = extended_evaluate
    return evaluator.evaluate(expr, env)


def test_list_operations():
    """Test the list operations implementation."""
    print("List Operations Tests")
    print("=" * 50)
    
    # Create test lists using AST
    list_123 = ListExpr((Num(1), Num(2), Num(3)))
    empty_list = ListExpr(())
    
    test_cases = [
        # Length tests
        (ListOp('length', (list_123,)), 3.0, "length [1,2,3]"),
        (ListOp('length', (empty_list,)), 0.0, "length []"),
        
        # Empty tests
        (ListOp('empty', (empty_list,)), 1.0, "empty []"),
        (ListOp('empty', (list_123,)), 0.0, "empty [1,2,3]"),
        
        # Head tests
        (ListOp('head', (list_123,)), 1.0, "head [1,2,3]"),
        
        # Cons tests
        (ListOp('length', (ListOp('cons', (Num(0), list_123)),)), 4.0, 
         "length (cons 0 [1,2,3])"),
    ]
    
    all_passed = True
    for expr, expected, description in test_cases:
        result = evaluate_with_lists(expr)
        
        if isinstance(expected, float):
            passed = abs(result - expected) < 1e-10
        else:
            passed = result == expected
        
        status = "✓" if passed else "✗"
        print(f"  {status} {description} = {result} (expected: {expected})")
        if not passed:
            all_passed = False
    
    # Test tail separately (returns list)
    tail_result = evaluate_with_lists(ListOp('tail', (list_123,)))
    if isinstance(tail_result, ListValue) and tail_result.elements == [2.0, 3.0]:
        print("  ✓ tail [1,2,3] = [2, 3]")
    else:
        print(f"  ✗ tail [1,2,3] = {tail_result}")
        all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3: String Operations
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StringExpr:
    """String literal expression."""
    value: str


@dataclass(frozen=True)
class StringOp:
    """String operation expression.
    
    Supported operations:
        - concat: Concatenate two strings
        - strlen: Length of string
        - substr: Substring extraction
        - charat: Character at index
    """
    op: str
    args: tuple[Expr, ...]


@dataclass
class StringValue:
    """Runtime string value."""
    value: str


def evaluate_with_strings(expr: Expr, env: dict | None = None):
    """Evaluate expressions with string support.
    
    Args:
        expr: The expression to evaluate.
        env: Variable environment.
    
    Returns:
        The computed value (may be a StringValue).
    """
    if env is None:
        env = {}
    
    evaluator = Evaluator()
    original_evaluate = evaluator.evaluate
    
    def extended_evaluate(e: Expr, environment: dict):
        match e:
            case StringExpr(value):
                return StringValue(value)
            
            case StringOp('concat', (s1_expr, s2_expr)):
                s1 = extended_evaluate(s1_expr, environment)
                s2 = extended_evaluate(s2_expr, environment)
                if not isinstance(s1, StringValue) or not isinstance(s2, StringValue):
                    raise ValueError("concat requires strings")
                return StringValue(s1.value + s2.value)
            
            case StringOp('strlen', (s_expr,)):
                s = extended_evaluate(s_expr, environment)
                if not isinstance(s, StringValue):
                    raise ValueError("strlen requires a string")
                return float(len(s.value))
            
            case StringOp('substr', (s_expr, start_expr, length_expr)):
                s = extended_evaluate(s_expr, environment)
                start = int(extended_evaluate(start_expr, environment))
                length = int(extended_evaluate(length_expr, environment))
                if not isinstance(s, StringValue):
                    raise ValueError("substr requires a string")
                return StringValue(s.value[start:start + length])
            
            case StringOp('charat', (s_expr, idx_expr)):
                s = extended_evaluate(s_expr, environment)
                idx = int(extended_evaluate(idx_expr, environment))
                if not isinstance(s, StringValue):
                    raise ValueError("charat requires a string")
                return StringValue(s.value[idx] if 0 <= idx < len(s.value) else "")
            
            case _:
                return original_evaluate(e, environment)
    
    evaluator.evaluate = extended_evaluate
    return evaluator.evaluate(expr, env)


def test_string_operations():
    """Test the string operations implementation."""
    print("String Operations Tests")
    print("=" * 50)
    
    hello = StringExpr("Hello")
    world = StringExpr("World")
    
    test_cases = [
        # Length tests
        (StringOp('strlen', (hello,)), 5.0, 'strlen "Hello"'),
        (StringOp('strlen', (StringExpr(""),)), 0.0, 'strlen ""'),
        
        # Concat tests
        (StringOp('strlen', (StringOp('concat', (hello, world)),)), 10.0, 
         'strlen (concat "Hello" "World")'),
    ]
    
    all_passed = True
    for expr, expected, description in test_cases:
        result = evaluate_with_strings(expr)
        
        if isinstance(expected, float):
            passed = abs(result - expected) < 1e-10
        else:
            passed = result == expected
        
        status = "✓" if passed else "✗"
        print(f"  {status} {description} = {result} (expected: {expected})")
        if not passed:
            all_passed = False
    
    # Test concat separately
    concat_result = evaluate_with_strings(StringOp('concat', (hello, world)))
    if isinstance(concat_result, StringValue) and concat_result.value == "HelloWorld":
        print('  ✓ concat "Hello" "World" = "HelloWorld"')
    else:
        print(f'  ✗ concat "Hello" "World" = {concat_result}')
        all_passed = False
    
    # Test charat
    charat_result = evaluate_with_strings(StringOp('charat', (hello, Num(1))))
    if isinstance(charat_result, StringValue) and charat_result.value == "e":
        print('  ✓ charat "Hello" 1 = "e"')
    else:
        print(f'  ✗ charat "Hello" 1 = {charat_result}')
        all_passed = False
    
    print()
    return all_passed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("═" * 60)
    print("WEEK 1: LAB 3 SOLUTIONS — INTERPRETER EXERCISES")
    print("═" * 60)
    print()
    
    results = []
    
    results.append(("Logical Operators", test_logical_operators()))
    results.append(("List Operations", test_list_operations()))
    results.append(("String Operations", test_string_operations()))
    
    print("═" * 60)
    print("SUMMARY")
    print("═" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name:25s} {status}")
    
    all_passed = all(p for _, p in results)
    print()
    print(f"Overall: {'All tests passed!' if all_passed else 'Some tests failed.'}")
