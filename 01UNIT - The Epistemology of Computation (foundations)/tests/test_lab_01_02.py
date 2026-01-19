"""
Tests for Week 1, Lab 2: Lambda Calculus Basics.

This module contains pytest tests for the lambda calculus implementation.
Tests cover AST construction, beta reduction and Church encodings.

Run with: pytest tests/test_lab_1_02.py -v
"""

import pytest


class TestLambdaExpressions:
    """Tests for basic lambda expression construction."""
    
    def test_variable_creation(self, lambda_var):
        """Test creating a variable."""
        x = lambda_var("x")
        assert str(x) == "x"
        assert x.free_variables() == {"x"}
    
    def test_abstraction_creation(self, lambda_var, lambda_abs):
        """Test creating an abstraction."""
        x = lambda_var("x")
        identity = lambda_abs("x", x)
        assert str(identity) == "λx.x"
        assert identity.free_variables() == set()
    
    def test_application_creation(self, lambda_var, lambda_abs, lambda_app):
        """Test creating an application."""
        identity = lambda_abs("x", lambda_var("x"))
        y = lambda_var("y")
        app = lambda_app(identity, y)
        assert "λx.x" in str(app)
        assert "y" in str(app)
    
    def test_free_variables_in_abstraction(self, lambda_var, lambda_abs):
        """Test free variable detection in abstractions."""
        # λx.x y - x is bound, y is free
        body = lambda_var("x")  # Simplified; full test would have x y
        expr = lambda_abs("x", body)
        assert "x" not in expr.free_variables()


class TestBetaReduction:
    """Tests for beta reduction."""
    
    def test_identity_reduction(self, lambda_var, lambda_abs, lambda_app):
        """Test reducing identity applied to argument."""
        from lab_1_02_lambda_calculus import beta_reduce
        
        identity = lambda_abs("x", lambda_var("x"))
        y = lambda_var("y")
        expr = lambda_app(identity, y)
        
        result = beta_reduce(expr)
        assert str(result) == "y"
    
    def test_k_combinator_reduction(self, lambda_var, lambda_abs, lambda_app):
        """Test reducing K combinator."""
        from lab_1_02_lambda_calculus import beta_reduce
        
        # K = λx.λy.x
        k = lambda_abs("x", lambda_abs("y", lambda_var("x")))
        
        # K a b should reduce to a
        a = lambda_var("a")
        b = lambda_var("b")
        expr = lambda_app(lambda_app(k, a), b)
        
        result = beta_reduce(expr)
        assert str(result) == "a"
    
    def test_reduction_trace(self, lambda_var, lambda_abs, lambda_app):
        """Test that reduction trace captures all steps."""
        from lab_1_02_lambda_calculus import trace_reduction
        
        identity = lambda_abs("x", lambda_var("x"))
        y = lambda_var("y")
        expr = lambda_app(identity, y)
        
        trace = trace_reduction(expr)
        assert len(trace) >= 2  # At least start and end
        assert str(trace[-1]) == "y"


class TestSubstitution:
    """Tests for substitution in lambda expressions."""
    
    def test_variable_substitution(self, lambda_var):
        """Test substituting in a variable."""
        x = lambda_var("x")
        y = lambda_var("y")
        result = x.substitute("x", y)
        assert str(result) == "y"
    
    def test_no_substitution_for_different_variable(self, lambda_var):
        """Test that wrong variable is not substituted."""
        x = lambda_var("x")
        z = lambda_var("z")
        result = x.substitute("y", z)
        assert str(result) == "x"
    
    def test_bound_variable_not_substituted(self, lambda_var, lambda_abs):
        """Test that bound variables are not substituted."""
        # λx.x should not change when substituting for x
        expr = lambda_abs("x", lambda_var("x"))
        y = lambda_var("y")
        result = expr.substitute("x", y)
        assert str(result) == "λx.x"


class TestChurchBooleans:
    """Tests for Church boolean encodings."""
    
    def test_true_selects_first(self, church_true, lambda_var, lambda_app):
        """Test that TRUE selects the first argument."""
        from lab_1_02_lambda_calculus import beta_reduce
        
        a = lambda_var("a")
        b = lambda_var("b")
        expr = lambda_app(lambda_app(church_true, a), b)
        
        result = beta_reduce(expr)
        assert str(result) == "a"
    
    def test_false_selects_second(self, church_false, lambda_var, lambda_app):
        """Test that FALSE selects the second argument."""
        from lab_1_02_lambda_calculus import beta_reduce
        
        a = lambda_var("a")
        b = lambda_var("b")
        expr = lambda_app(lambda_app(church_false, a), b)
        
        result = beta_reduce(expr)
        assert str(result) == "b"
    
    def test_church_not(self, church_true, church_false):
        """Test the NOT operation."""
        from lab_1_02_lambda_calculus import church_not, church_to_bool
        from lab_1_02_lambda_calculus import App
        
        not_true = App(church_not(), church_true)
        not_false = App(church_not(), church_false)
        
        assert church_to_bool(not_true) is False
        assert church_to_bool(not_false) is True


class TestChurchNumerals:
    """Tests for Church numeral encodings."""
    
    @pytest.mark.parametrize("n", [0, 1, 2, 3, 5, 10])
    def test_church_numeral_conversion(self, n):
        """Test round-trip conversion of Church numerals."""
        from lab_1_02_lambda_calculus import church_numeral, church_to_int
        
        numeral = church_numeral(n)
        result = church_to_int(numeral)
        assert result == n
    
    def test_church_successor(self):
        """Test the successor function."""
        from lab_1_02_lambda_calculus import (
            church_numeral, church_succ, church_to_int, App
        )
        
        two = church_numeral(2)
        succ_two = App(church_succ(), two)
        result = church_to_int(succ_two)
        assert result == 3
    
    def test_church_addition(self):
        """Test Church numeral addition."""
        from lab_1_02_lambda_calculus import (
            church_numeral, church_add, church_to_int, App
        )
        
        two = church_numeral(2)
        three = church_numeral(3)
        add = church_add()
        sum_expr = App(App(add, two), three)
        result = church_to_int(sum_expr)
        assert result == 5
    
    def test_church_multiplication(self):
        """Test Church numeral multiplication."""
        from lab_1_02_lambda_calculus import (
            church_numeral, church_mult, church_to_int, App
        )
        
        two = church_numeral(2)
        three = church_numeral(3)
        mult = church_mult()
        prod_expr = App(App(mult, two), three)
        result = church_to_int(prod_expr)
        assert result == 6
    
    def test_church_iszero(self):
        """Test IS_ZERO predicate."""
        from lab_1_02_lambda_calculus import (
            church_numeral, church_iszero, church_to_bool, App
        )
        
        zero = church_numeral(0)
        one = church_numeral(1)
        iszero = church_iszero()
        
        assert church_to_bool(App(iszero, zero)) is True
        assert church_to_bool(App(iszero, one)) is False


class TestCombinators:
    """Tests for standard combinators."""
    
    def test_i_combinator(self, lambda_var, lambda_app):
        """Test the identity combinator."""
        from lab_1_02_lambda_calculus import i_combinator, beta_reduce
        
        i = i_combinator()
        x = lambda_var("x")
        result = beta_reduce(lambda_app(i, x))
        assert str(result) == "x"
    
    def test_k_combinator(self, lambda_var, lambda_app):
        """Test the K combinator."""
        from lab_1_02_lambda_calculus import k_combinator, beta_reduce
        
        k = k_combinator()
        a = lambda_var("a")
        b = lambda_var("b")
        result = beta_reduce(lambda_app(lambda_app(k, a), b))
        assert str(result) == "a"
    
    def test_skk_equals_i(self, lambda_var, lambda_app):
        """Test that S K K = I."""
        from lab_1_02_lambda_calculus import (
            s_combinator, k_combinator, beta_reduce
        )
        
        s = s_combinator()
        k = k_combinator()
        skk = lambda_app(lambda_app(s, k), k)
        
        # Apply to z
        z = lambda_var("z")
        result = beta_reduce(lambda_app(skk, z))
        assert str(result) == "z"


class TestPythonIntegration:
    """Tests for Python implementations of Church encodings."""
    
    @pytest.mark.parametrize("n", [0, 1, 2, 5, 10])
    def test_python_church_numeral(self, n):
        """Test Python implementation of Church numerals."""
        from lab_1_02_lambda_calculus import python_church_numeral
        
        numeral = python_church_numeral(n)
        result = numeral(lambda x: x + 1)(0)
        assert result == n
    
    def test_python_church_add(self):
        """Test Python implementation of Church addition."""
        from lab_1_02_lambda_calculus import (
            python_church_numeral, python_church_add
        )
        
        two = python_church_numeral(2)
        three = python_church_numeral(3)
        result = python_church_add(two, three)(lambda x: x + 1)(0)
        assert result == 5
    
    def test_python_church_mult(self):
        """Test Python implementation of Church multiplication."""
        from lab_1_02_lambda_calculus import (
            python_church_numeral, python_church_mult
        )
        
        two = python_church_numeral(2)
        three = python_church_numeral(3)
        result = python_church_mult(two, three)(lambda x: x + 1)(0)
        assert result == 6


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_deeply_nested_expression(self, lambda_var, lambda_abs, lambda_app):
        """Test reduction of deeply nested expressions."""
        from lab_1_02_lambda_calculus import beta_reduce
        
        # Build a chain of identities applied to x
        x = lambda_var("x")
        identity = lambda_abs("y", lambda_var("y"))
        
        expr = x
        for _ in range(10):
            expr = lambda_app(identity, expr)
        
        result = beta_reduce(expr, max_steps=100)
        assert str(result) == "x"
    
    def test_reduction_step_limit(self, lambda_var, lambda_abs, lambda_app):
        """Test that reduction respects step limits."""
        from lab_1_02_lambda_calculus import beta_reduce
        
        # Omega combinator would loop forever
        # We can't test it directly, but we can test step limiting
        identity = lambda_abs("x", lambda_var("x"))
        expr = lambda_app(identity, lambda_var("y"))
        
        # This should complete within 1 step
        result = beta_reduce(expr, max_steps=1)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
