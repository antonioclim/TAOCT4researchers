"""
Tests for Week 1, Lab 3: Mini-Interpreter with AST.

This module contains pytest tests for the expression interpreter.
Tests cover lexing, parsing, evaluation and error handling.

Run with: pytest tests/test_lab_1_03.py -v
"""

import pytest


class TestLexer:
    """Tests for the lexer (tokeniser)."""
    
    def test_number_tokenisation(self):
        """Test tokenising numbers."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        lexer = Lexer("42")
        token = lexer.get_next_token()
        assert token.type == TokenType.NUMBER
        assert token.value == 42.0
    
    def test_float_tokenisation(self):
        """Test tokenising floating-point numbers."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        lexer = Lexer("3.14159")
        token = lexer.get_next_token()
        assert token.type == TokenType.NUMBER
        assert abs(token.value - 3.14159) < 1e-10
    
    def test_scientific_notation(self):
        """Test tokenising scientific notation."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        lexer = Lexer("1.5e-10")
        token = lexer.get_next_token()
        assert token.type == TokenType.NUMBER
        assert abs(token.value - 1.5e-10) < 1e-20
    
    def test_identifier_tokenisation(self):
        """Test tokenising identifiers."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        lexer = Lexer("variable_name")
        token = lexer.get_next_token()
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "variable_name"
    
    def test_keyword_tokenisation(self):
        """Test tokenising keywords."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        keywords = [
            ("let", TokenType.LET),
            ("in", TokenType.IN),
            ("fun", TokenType.FUN),
            ("if", TokenType.IF),
            ("then", TokenType.THEN),
            ("else", TokenType.ELSE),
        ]
        
        for text, expected_type in keywords:
            lexer = Lexer(text)
            token = lexer.get_next_token()
            assert token.type == expected_type
    
    def test_operator_tokenisation(self):
        """Test tokenising operators."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        operators = [
            ("+", TokenType.PLUS),
            ("-", TokenType.MINUS),
            ("*", TokenType.STAR),
            ("/", TokenType.SLASH),
            ("^", TokenType.CARET),
            ("<", TokenType.LT),
            (">", TokenType.GT),
            ("<=", TokenType.LE),
            (">=", TokenType.GE),
            ("==", TokenType.EQ),
            ("!=", TokenType.NE),
            ("->", TokenType.ARROW),
        ]
        
        for text, expected_type in operators:
            lexer = Lexer(text)
            token = lexer.get_next_token()
            assert token.type == expected_type
    
    def test_whitespace_skipping(self):
        """Test that whitespace is skipped."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        lexer = Lexer("  42  ")
        token = lexer.get_next_token()
        assert token.type == TokenType.NUMBER
    
    def test_comment_skipping(self):
        """Test that comments are skipped."""
        from lab_1_03_ast_interpreter import Lexer, TokenType
        
        lexer = Lexer("# this is a comment\n42")
        token = lexer.get_next_token()
        assert token.type == TokenType.NUMBER
        assert token.value == 42.0


class TestParser:
    """Tests for the parser."""
    
    def test_parse_number(self, parser, ast_nodes):
        """Test parsing a number."""
        ast = parser("42")
        assert isinstance(ast, ast_nodes['Num'])
        assert ast.value == 42.0
    
    def test_parse_variable(self, parser, ast_nodes):
        """Test parsing a variable."""
        ast = parser("x")
        assert isinstance(ast, ast_nodes['Var'])
        assert ast.name == "x"
    
    def test_parse_addition(self, parser, ast_nodes):
        """Test parsing addition."""
        ast = parser("1 + 2")
        assert isinstance(ast, ast_nodes['BinOp'])
        assert ast.op == "+"
    
    def test_parse_precedence(self, parser, ast_nodes):
        """Test that precedence is respected."""
        ast = parser("1 + 2 * 3")
        # Root should be addition
        assert isinstance(ast, ast_nodes['BinOp'])
        assert ast.op == "+"
        # Right child should be multiplication
        assert isinstance(ast.right, ast_nodes['BinOp'])
        assert ast.right.op == "*"
    
    def test_parse_parentheses(self, parser, ast_nodes):
        """Test parsing parenthesised expressions."""
        ast = parser("(1 + 2) * 3")
        # Root should be multiplication
        assert isinstance(ast, ast_nodes['BinOp'])
        assert ast.op == "*"
        # Left child should be addition
        assert isinstance(ast.left, ast_nodes['BinOp'])
        assert ast.left.op == "+"
    
    def test_parse_unary(self, parser, ast_nodes):
        """Test parsing unary operators."""
        ast = parser("-5")
        assert isinstance(ast, ast_nodes['UnaryOp'])
        assert ast.op == "-"
    
    def test_parse_let(self, parser, ast_nodes):
        """Test parsing let expressions."""
        ast = parser("let x = 5 in x + 1")
        assert isinstance(ast, ast_nodes['Let'])
        assert ast.name == "x"
    
    def test_parse_lambda(self, parser, ast_nodes):
        """Test parsing lambda expressions."""
        ast = parser("fun x -> x + 1")
        assert isinstance(ast, ast_nodes['Lambda'])
        assert ast.param == "x"
    
    def test_parse_if(self, parser, ast_nodes):
        """Test parsing if expressions."""
        ast = parser("if x > 0 then x else 0")
        assert isinstance(ast, ast_nodes['IfExpr'])
    
    def test_parse_function_call(self, parser, ast_nodes):
        """Test parsing function calls."""
        ast = parser("sin(x)")
        assert isinstance(ast, ast_nodes['FuncCall'])
        assert ast.name == "sin"
        assert len(ast.args) == 1
    
    def test_parse_function_call_multiple_args(self, parser, ast_nodes):
        """Test parsing function calls with multiple arguments."""
        ast = parser("max(a, b)")
        assert isinstance(ast, ast_nodes['FuncCall'])
        assert ast.name == "max"
        assert len(ast.args) == 2


class TestEvaluator:
    """Tests for the evaluator."""
    
    @pytest.mark.parametrize("expr,expected", [
        ("2 + 3", 5.0),
        ("10 - 4", 6.0),
        ("3 * 4", 12.0),
        ("15 / 3", 5.0),
        ("2 ^ 3", 8.0),
        ("10 % 3", 1.0),
    ])
    def test_arithmetic(self, evaluator, expr, expected):
        """Test basic arithmetic operations."""
        result = evaluator(expr)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.parametrize("expr,expected", [
        ("2 + 3 * 4", 14.0),
        ("(2 + 3) * 4", 20.0),
        ("2 ^ 3 ^ 2", 512.0),  # Right-associative
        ("10 - 2 - 3", 5.0),   # Left-associative
    ])
    def test_precedence_and_associativity(self, evaluator, expr, expected):
        """Test operator precedence and associativity."""
        result = evaluator(expr)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.parametrize("expr,expected", [
        ("-5", -5.0),
        ("--5", 5.0),
        ("3 + -2", 1.0),
        ("-(2 + 3)", -5.0),
    ])
    def test_unary_operators(self, evaluator, expr, expected):
        """Test unary operators."""
        result = evaluator(expr)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.parametrize("expr,expected", [
        ("let x = 5 in x", 5.0),
        ("let x = 5 in x + 1", 6.0),
        ("let x = 2 in let y = 3 in x + y", 5.0),
        ("let x = 5 in let x = 10 in x", 10.0),  # Shadowing
    ])
    def test_let_bindings(self, evaluator, expr, expected):
        """Test let bindings."""
        result = evaluator(expr)
        assert abs(result - expected) < 1e-10
    
    @pytest.mark.parametrize("expr,expected", [
        ("abs(-5)", 5.0),
        ("max(3, 7)", 7.0),
        ("min(3, 7)", 3.0),
        ("floor(3.7)", 3.0),
        ("ceil(3.2)", 4.0),
        ("sqrt(16)", 4.0),
    ])
    def test_builtin_functions(self, evaluator, expr, expected):
        """Test built-in functions."""
        result = evaluator(expr)
        assert abs(result - expected) < 1e-10
    
    def test_sin_cos(self, evaluator):
        """Test trigonometric functions."""
        assert abs(evaluator("sin(0)") - 0.0) < 1e-10
        assert abs(evaluator("cos(0)") - 1.0) < 1e-10
    
    @pytest.mark.parametrize("expr,expected", [
        ("1 < 2", 1.0),
        ("2 < 1", 0.0),
        ("5 == 5", 1.0),
        ("5 != 5", 0.0),
        ("3 <= 3", 1.0),
        ("3 >= 4", 0.0),
    ])
    def test_comparisons(self, evaluator, expr, expected):
        """Test comparison operators."""
        result = evaluator(expr)
        assert result == expected
    
    @pytest.mark.parametrize("expr,expected", [
        ("if 1 > 0 then 10 else 20", 10.0),
        ("if 0 > 1 then 10 else 20", 20.0),
        ("if 1 == 1 then 100 else 200", 100.0),
    ])
    def test_conditionals(self, evaluator, expr, expected):
        """Test conditional expressions."""
        result = evaluator(expr)
        assert result == expected
    
    @pytest.mark.parametrize("expr,expected", [
        ("let square = fun x -> x * x in square(5)", 25.0),
        ("let add1 = fun x -> x + 1 in add1(10)", 11.0),
        ("let f = fun x -> x * 2 in f(f(3))", 12.0),
    ])
    def test_lambdas(self, evaluator, expr, expected):
        """Test lambda expressions."""
        result = evaluator(expr)
        assert abs(result - expected) < 1e-10
    
    def test_complex_expression(self, evaluator):
        """Test a complex nested expression."""
        expr = """
            let a = 2 in
            let b = 3 in
            let c = 4 in
            a * b + c
        """
        result = evaluator(expr)
        assert result == 10.0
    
    def test_nested_functions(self, evaluator):
        """Test nested function applications."""
        expr = """
            let f = fun x -> x * x in
            let g = fun y -> y + 1 in
            f(g(3))
        """
        result = evaluator(expr)
        assert result == 16.0  # (3+1)^2 = 16


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_undefined_variable(self, evaluator):
        """Test that undefined variables raise NameError."""
        with pytest.raises(NameError, match="Undefined variable"):
            evaluator("x")
    
    def test_unknown_function(self, evaluator):
        """Test that unknown functions raise NameError."""
        with pytest.raises(NameError, match="Unknown function"):
            evaluator("unknown(1)")
    
    def test_division_by_zero(self, evaluator):
        """Test that division by zero raises ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            evaluator("1 / 0")
    
    def test_syntax_error(self, parser):
        """Test that invalid syntax raises SyntaxError."""
        with pytest.raises(SyntaxError):
            parser("let x =")
    
    def test_unexpected_token(self, parser):
        """Test that unexpected tokens raise SyntaxError."""
        with pytest.raises(SyntaxError):
            parser("1 + + 2")


class TestClosure:
    """Tests for closure behaviour."""
    
    def test_closure_captures_environment(self, evaluator):
        """Test that closures capture their defining environment."""
        expr = """
            let x = 10 in
            let f = fun y -> x + y in
            let x = 20 in
            f(5)
        """
        result = evaluator(expr)
        # f should use x = 10 from its definition, not x = 20
        assert result == 15.0
    
    def test_closure_with_let(self, evaluator):
        """Test closures created in let bindings."""
        expr = """
            let make_adder = fun n -> fun x -> x + n in
            let add5 = make_adder(5) in
            add5(10)
        """
        result = evaluator(expr)
        assert result == 15.0


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_quadratic_formula(self, evaluator):
        """Test implementing quadratic formula."""
        expr = """
            let a = 1 in
            let b = -5 in
            let c = 6 in
            let discriminant = b * b - 4 * a * c in
            let x1 = (-b + sqrt(discriminant)) / (2 * a) in
            x1
        """
        result = evaluator(expr)
        assert abs(result - 3.0) < 1e-10
    
    def test_max_of_three(self, evaluator):
        """Test finding maximum of three numbers."""
        expr = """
            let max3 = fun a -> fun b -> fun c ->
                if a > b then
                    if a > c then a else c
                else
                    if b > c then b else c
            in max3(5)(9)(3)
        """
        result = evaluator(expr)
        assert result == 9.0
    
    def test_absolute_value(self, evaluator):
        """Test implementing absolute value."""
        expr = """
            let my_abs = fun x -> if x < 0 then -x else x in
            my_abs(-5) + my_abs(3)
        """
        result = evaluator(expr)
        assert result == 8.0


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_very_large_number(self, evaluator):
        """Test handling very large numbers."""
        result = evaluator("10 ^ 100")
        assert result == 10 ** 100
    
    def test_very_small_number(self, evaluator):
        """Test handling very small numbers."""
        result = evaluator("1e-100")
        assert abs(result - 1e-100) < 1e-110
    
    def test_deeply_nested_parentheses(self, evaluator):
        """Test deeply nested parentheses."""
        expr = "((((((1 + 2))))))"
        result = evaluator(expr)
        assert result == 3.0
    
    def test_long_chain_of_operations(self, evaluator):
        """Test long chain of operations."""
        expr = "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10"
        result = evaluator(expr)
        assert result == 55.0
    
    def test_complex_boolean_expression(self, evaluator):
        """Test complex boolean-like expressions."""
        expr = """
            let x = 5 in
            let y = 10 in
            if x < y then
                if x > 0 then 1 else 0
            else
                0
        """
        result = evaluator(expr)
        assert result == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
