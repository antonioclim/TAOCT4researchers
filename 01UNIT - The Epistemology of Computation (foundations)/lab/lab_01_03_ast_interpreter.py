# mypy: ignore-errors
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 1, Lab 3: Mini-Interpreter with AST and Pattern Matching
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Every programming language, from Python to MATLAB, from R to Julia, operates
on the same fundamental principle:

    Text → Tokens → AST → Evaluation/Compilation

Understanding this flow enables you to:
1. Build your own DSLs (Domain Specific Languages) for research
2. Understand how tools like linters, formatters and transpilers work
3. Think more clearly about the logical structure of computation

REAL-WORLD EXAMPLE
──────────────────
Jupyter Notebooks process each cell as follows:
1. The Python text is parsed into an AST
2. The AST is compiled to bytecode
3. The bytecode is executed by the Python VM
4. The result is captured and displayed

When you write `2 + 2` in Jupyter, Python does:
- Lexer: ['2', '+', '2']
- Parser: BinOp(Num(2), Add(), Num(2))
- Compiler: LOAD_CONST 2, LOAD_CONST 2, BINARY_ADD
- VM: Stack [2] → Stack [2, 2] → Stack [4]
- Output: 4

PREREQUISITES
─────────────
- Week 1, Labs 1-2: Understanding of computation models
- Python: Intermediate (pattern matching, recursion, classes)
- Libraries: None (standard library only)

LEARNING OBJECTIVES
───────────────────
After completing this lab, you will be able to:
1. Implement a simple lexer (tokeniser)
2. Implement a recursive-descent parser
3. Implement an evaluator using pattern matching
4. Extend the interpreter with variables and functions

ESTIMATED TIME
──────────────
- Reading: 40 minutes
- Coding: 80 minutes
- Total: 120 minutes

DEPENDENCIES
────────────
Python 3.12+ (for pattern matching syntax)

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ABSTRACT SYNTAX TREE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Num:
    """Node for numeric literals.
    
    In the AST, this node represents a constant value. It is 'frozen'
    because ASTs should be immutable (facilitating caching, comparison
    and debugging).
    
    Attributes:
        value: The numeric value.
    
    Example:
        >>> Num(42)
        Num(42)
        >>> Num(3.14)
        Num(3.14)
    """
    value: float
    
    def __repr__(self) -> str:
        # Display integers without decimals
        if self.value == int(self.value):
            return f"Num({int(self.value)})"
        return f"Num({self.value})"


@dataclass(frozen=True)
class Var:
    """Node for variables (references to named values).
    
    Attributes:
        name: The variable identifier.
    
    Example:
        >>> Var("x")
        Var('x')
    """
    name: str
    
    def __repr__(self) -> str:
        return f"Var({self.name!r})"


@dataclass(frozen=True)
class BinOp:
    """Node for binary operations (two operands).
    
    The operator is stored as a string for simplicity. In a real compiler,
    it would be an enum or a separate node type.
    
    Attributes:
        left: The left operand expression.
        op: The operator symbol.
        right: The right operand expression.
    
    Example:
        >>> BinOp(Num(2), '+', Num(3))
        BinOp(Num(2), '+', Num(3))
    """
    left: 'Expr'
    op: str
    right: 'Expr'
    
    def __repr__(self) -> str:
        return f"BinOp({self.left!r}, {self.op!r}, {self.right!r})"


@dataclass(frozen=True)
class UnaryOp:
    """Node for unary operations (single operand).
    
    Attributes:
        op: The operator symbol.
        operand: The operand expression.
    
    Example:
        >>> UnaryOp('-', Num(5))
        UnaryOp('-', Num(5))
    """
    op: str
    operand: 'Expr'
    
    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand!r})"


@dataclass(frozen=True)
class FuncCall:
    """Node for function calls.
    
    Attributes:
        name: The function name.
        args: Tuple of argument expressions.
    
    Example:
        >>> FuncCall('sin', (Var('x'),))
        FuncCall('sin', (Var('x'),))
    """
    name: str
    args: tuple['Expr', ...]
    
    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"FuncCall({self.name!r}, ({args_str}))"


@dataclass(frozen=True)
class Call:
    """Node for calling the result of an expression.

    This node supports application chaining such as ``f(1)(2)``. The first call
    ``f(1)`` can still be represented as :class:`FuncCall` when the callee is a
    bare identifier, while subsequent calls are expressed as :class:`Call`.

    Attributes:
        callee: Expression evaluating to a callable value.
        args: Tuple of argument expressions.
    """

    callee: 'Expr'
    args: tuple['Expr', ...]

    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"Call({self.callee!r}, ({args_str}))"


@dataclass(frozen=True)
class Let:
    """Node for variable bindings (let-binding).
    
    Syntax: let x = expr1 in expr2
    Semantics: evaluate expr2 in a context where x = value of expr1
    
    Attributes:
        name: The variable name to bind.
        value: The expression whose value is bound.
        body: The expression to evaluate with the binding.
    
    Example:
        >>> Let('x', Num(5), BinOp(Var('x'), '+', Num(1)))
        Let('x', Num(5), BinOp(Var('x'), '+', Num(1)))
    """
    name: str
    value: 'Expr'
    body: 'Expr'
    
    def __repr__(self) -> str:
        return f"Let({self.name!r}, {self.value!r}, {self.body!r})"


@dataclass(frozen=True)
class Lambda:
    """Node for anonymous functions (lambda expressions).
    
    Syntax: fun x -> expr
    
    Attributes:
        param: The parameter name.
        body: The function body expression.
    
    Example:
        >>> Lambda('x', BinOp(Var('x'), '*', Var('x')))
        Lambda('x', BinOp(Var('x'), '*', Var('x')))
    """
    param: str
    body: 'Expr'
    
    def __repr__(self) -> str:
        return f"Lambda({self.param!r}, {self.body!r})"


@dataclass(frozen=True)
class IfExpr:
    """Node for conditional expressions.
    
    Syntax: if cond then expr1 else expr2
    
    Unlike statements, conditional expressions return a value.
    
    Attributes:
        condition: The condition expression.
        then_branch: Expression evaluated if condition is true.
        else_branch: Expression evaluated if condition is false.
    
    Example:
        >>> IfExpr(BinOp(Var('x'), '>', Num(0)), Var('x'), UnaryOp('-', Var('x')))
        IfExpr(...)
    """
    condition: 'Expr'
    then_branch: 'Expr'
    else_branch: 'Expr'
    
    def __repr__(self) -> str:
        return f"IfExpr({self.condition!r}, {self.then_branch!r}, {self.else_branch!r})"


# Union type for all expressions
Expr = Union[Num, Var, BinOp, UnaryOp, FuncCall, Call, Let, Lambda, IfExpr, Call]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: LEXER (Tokeniser)
# ═══════════════════════════════════════════════════════════════════════════════

class TokenType(Enum):
    """Token types recognised by the lexer."""
    NUMBER = auto()
    IDENTIFIER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    CARET = auto()      # ^
    PERCENT = auto()    # %
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    EQUALS = auto()     # =
    ARROW = auto()      # ->
    LT = auto()         # <
    GT = auto()         # >
    LE = auto()         # <=
    GE = auto()         # >=
    EQ = auto()         # ==
    NE = auto()         # !=
    # Keywords
    LET = auto()
    IN = auto()
    FUN = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    # Special
    EOF = auto()


@dataclass
class Token:
    """An individual token produced by the lexer.
    
    Attributes:
        type: The token type.
        value: The token value (string or number).
        position: Position in the original text (for error reporting).
    """
    type: TokenType
    value: str | float
    position: int
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


class Lexer:
    """Lexical analyser (tokeniser).
    
    Transforms a string of characters into a sequence of tokens.
    
    Example:
        >>> lexer = Lexer("let x = 5 in x + 1")
        >>> lexer.get_next_token()
        Token(LET, 'let')
    """
    
    # Reserved keywords
    KEYWORDS = {
        'let': TokenType.LET,
        'in': TokenType.IN,
        'fun': TokenType.FUN,
        'if': TokenType.IF,
        'then': TokenType.THEN,
        'else': TokenType.ELSE,
    }
    
    def __init__(self, text: str) -> None:
        """Initialise the lexer with input text.
        
        Args:
            text: The source code to tokenise.
        """
        self.text = text
        self.pos = 0
        self.current_char: str | None = text[0] if text else None
        logger.debug("Lexer initialised with %d characters", len(text))
    
    def error(self, message: str) -> None:
        """Raise an exception with context."""
        context = self.text[max(0, self.pos - 10):self.pos + 10]
        raise SyntaxError(f"{message} at position {self.pos}: ...{context}...")
    
    def advance(self) -> None:
        """Advance to the next character."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
    
    def peek(self) -> str | None:
        """Look at the next character without advancing."""
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None
    
    def skip_whitespace(self) -> None:
        """Skip over whitespace characters."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self) -> None:
        """Skip comments (# to end of line)."""
        while self.current_char is not None and self.current_char != '\n':
            self.advance()
    
    def number(self) -> Token:
        """Parse a number (integer or floating-point).
        
        Supports:
        - Integers: 42
        - Decimals: 3.14
        - Scientific notation: 1.5e-10
        
        Returns:
            A NUMBER token.
        """
        start_pos = self.pos
        result = ''
        
        # Integer part
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        
        # Decimal part
        peek_char = self.peek()
        if self.current_char == '.' and peek_char is not None and peek_char.isdigit():
            result += self.current_char
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
        
        # Scientific notation (optional)
        if self.current_char in ('e', 'E'):
            result += self.current_char
            self.advance()
            if self.current_char in ('+', '-'):
                result += self.current_char
                self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
        
        return Token(TokenType.NUMBER, float(result), start_pos)
    
    def identifier(self) -> Token:
        """Parse an identifier or keyword.
        
        Returns:
            An IDENTIFIER or keyword token.
        """
        start_pos = self.pos
        result = ''
        
        while (
            self.current_char is not None and 
            (self.current_char.isalnum() or self.current_char == '_')
        ):
            result += self.current_char
            self.advance()
        
        # Check if it is a keyword
        token_type = self.KEYWORDS.get(result, TokenType.IDENTIFIER)
        return Token(token_type, result, start_pos)
    
    def get_next_token(self) -> Token:
        """Return the next token from the input.
        
        Returns:
            The next token.
        
        Raises:
            SyntaxError: If an unexpected character is encountered.
        """
        while self.current_char is not None:
            # Whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Comment
            if self.current_char == '#':
                self.skip_comment()
                continue
            
            # Number
            if self.current_char.isdigit():
                return self.number()
            
            # Identifier or keyword
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            
            # Two-character operators
            if self.current_char == '-' and self.peek() == '>':
                pos = self.pos
                self.advance()
                self.advance()
                return Token(TokenType.ARROW, '->', pos)
            
            if self.current_char == '<' and self.peek() == '=':
                pos = self.pos
                self.advance()
                self.advance()
                return Token(TokenType.LE, '<=', pos)
            
            if self.current_char == '>' and self.peek() == '=':
                pos = self.pos
                self.advance()
                self.advance()
                return Token(TokenType.GE, '>=', pos)
            
            if self.current_char == '=' and self.peek() == '=':
                pos = self.pos
                self.advance()
                self.advance()
                return Token(TokenType.EQ, '==', pos)
            
            if self.current_char == '!' and self.peek() == '=':
                pos = self.pos
                self.advance()
                self.advance()
                return Token(TokenType.NE, '!=', pos)
            
            # Single-character operators
            single_char_tokens = {
                '+': TokenType.PLUS,
                '-': TokenType.MINUS,
                '*': TokenType.STAR,
                '/': TokenType.SLASH,
                '^': TokenType.CARET,
                '%': TokenType.PERCENT,
                '(': TokenType.LPAREN,
                ')': TokenType.RPAREN,
                ',': TokenType.COMMA,
                '=': TokenType.EQUALS,
                '<': TokenType.LT,
                '>': TokenType.GT,
            }
            
            if self.current_char in single_char_tokens:
                char = self.current_char
                pos = self.pos
                self.advance()
                return Token(single_char_tokens[char], char, pos)
            
            self.error(f"Unexpected character: {self.current_char!r}")
        
        return Token(TokenType.EOF, '', self.pos)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """Recursive-descent parser for expressions.
    
    Grammar (in order of increasing precedence):
    
    expr       → let | if | lambda | comparison
    comparison → additive (('<' | '>' | '<=' | '>=' | '==' | '!=') additive)*
    additive   → multiplicative (('+' | '-') multiplicative)*
    multiplicative → power (('*' | '/' | '%') power)*
    power      → unary ('^' power)?
    unary      → '-' unary | primary
    primary    → NUMBER | IDENTIFIER | IDENTIFIER '(' args ')' | '(' expr ')'
    
    The parser builds an AST from the token stream using recursive descent,
    respecting operator precedence and associativity.
    """
    
    def __init__(self, lexer: Lexer) -> None:
        """Initialise the parser with a lexer.
        
        Args:
            lexer: The lexer providing tokens.
        """
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
        logger.debug("Parser initialised, first token: %s", self.current_token)
    
    def error(self, message: str) -> None:
        """Raise a syntax error with context."""
        raise SyntaxError(
            f"{message}. Got {self.current_token} at position "
            f"{self.current_token.position}"
        )
    
    def eat(self, token_type: TokenType) -> Token:
        """Consume the current token if it matches the expected type.
        
        Args:
            token_type: The expected token type.
        
        Returns:
            The consumed token.
        
        Raises:
            SyntaxError: If the token type does not match.
        """
        if self.current_token.type == token_type:
            token = self.current_token
            self.current_token = self.lexer.get_next_token()
            return token

        self.error(f"Expected {token_type}, got {self.current_token.type}")
        raise AssertionError("Unreachable")


        self.error(f"Expected {token_type}, got {self.current_token.type}")
        raise AssertionError("Unreachable")
        
        self.error(f"Expected {token_type.name}")
    
    def parse(self) -> Expr:
        """Parse the entire expression.
        
        Returns:
            The parsed AST.
        """
        result = self.expr()
        if self.current_token.type != TokenType.EOF:
            self.error("Unexpected token after expression")
        return result
    
    def expr(self) -> Expr:
        """Parse an expression (lowest precedence)."""
        # let-binding
        if self.current_token.type == TokenType.LET:
            return self.let_expr()
        
        # if-expression
        if self.current_token.type == TokenType.IF:
            return self.if_expr()
        
        # lambda expression
        if self.current_token.type == TokenType.FUN:
            return self.lambda_expr()
        
        return self.comparison()
    
    def let_expr(self) -> Let:
        """Parse a let-binding: let x = expr1 in expr2."""
        self.eat(TokenType.LET)
        name_token = self.eat(TokenType.IDENTIFIER)
        name = str(name_token.value)
        self.eat(TokenType.EQUALS)
        value = self.expr()
        self.eat(TokenType.IN)
        body = self.expr()
        return Let(name, value, body)
    
    def if_expr(self) -> IfExpr:
        """Parse a conditional: if cond then expr1 else expr2."""
        self.eat(TokenType.IF)
        condition = self.expr()
        self.eat(TokenType.THEN)
        then_branch = self.expr()
        self.eat(TokenType.ELSE)
        else_branch = self.expr()
        return IfExpr(condition, then_branch, else_branch)
    
    def lambda_expr(self) -> Lambda:
        """Parse a lambda: fun x -> expr."""
        self.eat(TokenType.FUN)
        param_token = self.eat(TokenType.IDENTIFIER)
        param = str(param_token.value)
        self.eat(TokenType.ARROW)
        body = self.expr()
        return Lambda(param, body)
    
    def comparison(self) -> Expr:
        """Parse comparison operators (lowest arithmetic precedence)."""
        left = self.additive()
        
        while self.current_token.type in (
            TokenType.LT, TokenType.GT, 
            TokenType.LE, TokenType.GE,
            TokenType.EQ, TokenType.NE
        ):
            op = str(self.current_token.value)
            self.eat(self.current_token.type)
            right = self.additive()
            left = BinOp(left, op, right)
        
        return left
    
    def additive(self) -> Expr:
        """Parse addition and subtraction."""
        left = self.multiplicative()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op = str(self.current_token.value)
            self.eat(self.current_token.type)
            right = self.multiplicative()
            left = BinOp(left, op, right)
        
        return left
    
    def multiplicative(self) -> Expr:
        """Parse multiplication, division and modulo."""
        left = self.power()
        
        while self.current_token.type in (
            TokenType.STAR, TokenType.SLASH, TokenType.PERCENT
        ):
            op = str(self.current_token.value)
            self.eat(self.current_token.type)
            right = self.power()
            left = BinOp(left, op, right)
        
        return left
    
    def power(self) -> Expr:
        """Parse exponentiation (right-associative)."""
        base = self.unary()
        
        if self.current_token.type == TokenType.CARET:
            self.eat(TokenType.CARET)
            # Right-associative: 2^3^2 = 2^(3^2) = 512
            exponent = self.power()
            return BinOp(base, '^', exponent)
        
        return base
    
    def unary(self) -> Expr:
        """Parse unary operators."""
        if self.current_token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            operand = self.unary()
            return UnaryOp('-', operand)
        
        return self.postfix()
    
    def postfix(self) -> Expr:
        """Parse postfix call syntax.

        After a primary expression is parsed, a sequence of argument lists may
        follow. This supports application chaining such as ``f(1)(2)``.
        """
        expr = self.primary()

        while self.current_token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            args: list[Expr] = []

            if self.current_token.type != TokenType.RPAREN:
                args.append(self.expr())
                while self.current_token.type == TokenType.COMMA:
                    self.eat(TokenType.COMMA)
                    args.append(self.expr())

            self.eat(TokenType.RPAREN)
            expr = Call(expr, tuple(args))

        return expr


    def primary(self) -> Expr:
        """Parse primary expressions (highest precedence)."""
        token = self.current_token
        
        # Number literal
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Num(float(token.value))
        
        # Identifier or function call
        if token.type == TokenType.IDENTIFIER:
            name = str(token.value)
            self.eat(TokenType.IDENTIFIER)
            
            # Check for function call
            if self.current_token.type == TokenType.LPAREN:
                self.eat(TokenType.LPAREN)
                args: list[Expr] = []
                
                if self.current_token.type != TokenType.RPAREN:
                    args.append(self.expr())
                    
                    while self.current_token.type == TokenType.COMMA:
                        self.eat(TokenType.COMMA)
                        args.append(self.expr())
                
                self.eat(TokenType.RPAREN)
                return FuncCall(name, tuple(args))
            
            return Var(name)
        
        # Parenthesised expression
        if token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            result = self.expr()
            self.eat(TokenType.RPAREN)
            return result
        
        self.error(f"Unexpected token: {token}")

        raise AssertionError("Unreachable")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Type for runtime values
Value = Union[int, float, 'Closure']


@dataclass
class Closure:
    """A closure represents a function together with its lexical environment.
    
    When we define a function with `fun x -> body`, we capture the current
    environment so that `body` can access variables from that point.
    
    Attributes:
        param: The parameter name.
        body: The function body expression.
        env: The captured environment.
    """
    param: str
    body: Expr
    env: dict[str, Value]
    
    def __repr__(self) -> str:
        return f"<closure: fun {self.param} -> ...>"


# Execution environment: mapping from names to values
Environment = dict[str, Value]


# Built-in functions
BUILTINS: dict[str, Callable[..., float]] = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'sqrt': math.sqrt,
    'abs': abs,
    'floor': math.floor,
    'ceil': math.ceil,
    'log': math.log,
    'log10': math.log10,
    'exp': math.exp,
    'min': min,
    'max': max,
    'pow': pow,
    'round': round,
}


class Evaluator:
    """Expression evaluator with support for:
    
    - Arithmetic operations
    - Variables and let-bindings
    - Anonymous functions (lambdas)
    - Conditional expressions
    - Built-in mathematical functions
    
    The evaluator uses pattern matching to dispatch on expression types
    and maintains an environment for variable bindings.
    """
    
    def __init__(self) -> None:
        """Initialise the evaluator."""
        self.builtins = BUILTINS
    
    def evaluate(self, expr: Expr, env: Environment | None = None) -> Value:
        """Evaluate an expression in a given environment.
        
        Args:
            expr: The expression to evaluate.
            env: The execution environment (variable → value mapping).
        
        Returns:
            The value of the expression (float or Closure).
        
        Raises:
            NameError: If a variable is undefined.
            TypeError: If operations are applied to wrong types.
            ZeroDivisionError: If division by zero occurs.
        """
        if env is None:
            env = {}
        
        match expr:
            # Numeric literal
            case Num(value):
                return value
            
            # Variable reference
            case Var(name):
                if name in env:
                    return env[name]
                raise NameError(f"Undefined variable: {name}")
            
            # Binary operation
            case BinOp(left, op, right):
                left_val = self.evaluate(left, env)
                right_val = self.evaluate(right, env)
                
                # Verify both are numbers
                if not isinstance(left_val, (int, float)) or not isinstance(right_val, (int, float)):
                    raise TypeError(f"Cannot apply {op} to non-numeric values")
                
                match op:
                    case '+':
                        return left_val + right_val
                    case '-':
                        return left_val - right_val
                    case '*':
                        return left_val * right_val
                    case '/':
                        if right_val == 0:
                            raise ZeroDivisionError("Division by zero")
                        return left_val / right_val
                    case '^':
                        base_int: int | None = None
                        exp_int: int | None = None
                        if isinstance(left_val, int):
                            base_int = left_val
                        elif isinstance(left_val, float) and left_val.is_integer():
                            base_int = int(left_val)

                        if isinstance(right_val, int):
                            exp_int = right_val
                        elif isinstance(right_val, float) and right_val.is_integer():
                            exp_int = int(right_val)

                        if base_int is not None and exp_int is not None and exp_int >= 0:
                            return pow(base_int, exp_int)
                        return left_val ** right_val
                    case '%':
                        return left_val % right_val
                    case '<':
                        return 1.0 if left_val < right_val else 0.0
                    case '>':
                        return 1.0 if left_val > right_val else 0.0
                    case '<=':
                        return 1.0 if left_val <= right_val else 0.0
                    case '>=':
                        return 1.0 if left_val >= right_val else 0.0
                    case '==':
                        return 1.0 if left_val == right_val else 0.0
                    case '!=':
                        return 1.0 if left_val != right_val else 0.0
                    case _:

                        raise ValueError(f"Unknown operator: {op}")
            
            # Unary operation
            case UnaryOp(op, operand):
                val = self.evaluate(operand, env)
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Cannot apply unary {op} to non-numeric value")

                match op:
                    case '-':
                        return -val
                    case _:
                        raise ValueError(f"Unknown unary operator: {op}")
            
            # Function call
            case FuncCall(name, args):
                # Built-in function
                if name in self.builtins:
                    evaluated_args = [self.evaluate(arg, env) for arg in args]
                    return self.builtins[name](*evaluated_args)
                
                # User-defined function (closure)
                if name in env:
                    func = env[name]
                    if isinstance(func, Closure):
                        if len(args) != 1:
                            raise TypeError(
                                f"Function {name} expects 1 argument, got {len(args)}"
                            )
                        arg_val = self.evaluate(args[0], env)
                        new_env = {**func.env, func.param: arg_val}
                        return self.evaluate(func.body, new_env)
                
                raise NameError(f"Unknown function: {name}")
            
            # Call (application chaining)
            case Call(callee, args):
                func_val = self.evaluate(callee, env)
                if not isinstance(func_val, Closure):
                    raise TypeError("Attempted to call a non-function value")
                if len(args) != 1:
                    raise TypeError("Functions take exactly one argument in this language")
                arg_val = self.evaluate(args[0], env)
                new_env = {**func_val.env, func_val.param: arg_val}
                return self.evaluate(func_val.body, new_env)

            # Let binding
            case Let(name, value, body):
                val = self.evaluate(value, env)
                new_env = {**env, name: val}
                return self.evaluate(body, new_env)
            
            # Lambda
            case Lambda(param, body):
                return Closure(param, body, env.copy())
            
            # If expression
            case IfExpr(condition, then_branch, else_branch):
                cond_val = self.evaluate(condition, env)
                if not isinstance(cond_val, (int, float)):
                    raise TypeError("Condition must evaluate to a number")
                
                # 0.0 is false, anything else is true
                if cond_val != 0.0:
                    return self.evaluate(then_branch, env)
                else:
                    return self.evaluate(else_branch, env)
            
            case _:
                raise ValueError(f"Unknown expression type: {expr}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: INTERFACE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse(text: str) -> Expr:
    """Parse source code into an AST.
    
    Args:
        text: The source code to parse.
    
    Returns:
        The parsed expression.
    """
    lexer = Lexer(text)
    parser = Parser(lexer)
    return parser.parse()


def evaluate(text: str, env: Environment | None = None) -> Value:
    """Parse and evaluate source code.
    
    Args:
        text: The source code to evaluate.
        env: Optional environment with predefined variables.
    
    Returns:
        The result of evaluation.
    """
    expr = parse(text)
    evaluator = Evaluator()
    return evaluator.evaluate(expr, env)


def repl() -> None:
    """Run an interactive Read-Eval-Print Loop.
    
    The REPL maintains an environment across evaluations, allowing
    you to define variables and functions that persist.
    
    Commands:
        :quit or :q  - Exit the REPL
        :env         - Show current environment
        :ast <expr>  - Show AST for expression
        :help        - Show help
    """
    logger.info("Starting interactive REPL")
    print("=" * 60)
    print("Mini-Interpreter REPL")
    print("Type :help for commands, :quit to exit")
    print("=" * 60)
    
    evaluator = Evaluator()
    env: Environment = {}
    
    while True:
        try:
            line = input("\n>>> ").strip()
            
            if not line:
                continue
            
            # Commands
            if line.lower() in (':quit', ':q'):
                break
            
            if line.lower() == ':help':
                print("""
Commands:
  :quit, :q     - Exit the REPL
  :env          - Show current environment
  :ast <expr>   - Show AST for expression
  :help         - Show this help

Syntax:
  let x = 5 in x + 1          - Variable binding
  fun x -> x * x              - Lambda function
  if a > b then a else b      - Conditional
  sin(x), cos(x), sqrt(x)     - Built-in functions
""")
                continue
            
            if line.lower() == ':env':
                if env:
                    for name, value in env.items():
                        print(f"  {name} = {value}")
                else:
                    print("  (empty)")
                continue
            
            if line.lower().startswith(':ast '):
                expr_text = line[5:].strip()
                try:
                    ast = parse(expr_text)
                    print(f"  {ast}")
                except Exception as e:
                    print(f"  Error: {e}")
                continue
            
            # Evaluate expression
            result = evaluator.evaluate(parse(line), env)
            print(f"= {result}")
            
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: TESTS AND DEMONSTRATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> tuple[int, int]:
    """Run the test suite for validation.
    
    Returns:
        A tuple (passed, failed) with test counts.
    """
    logger.info("Running test suite")
    print("=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    tests = [
        # Basic arithmetic
        ("2 + 3", 5.0),
        ("10 - 4", 6.0),
        ("3 * 4", 12.0),
        ("15 / 3", 5.0),
        ("2 ^ 3", 8.0),
        ("10 % 3", 1.0),
        
        # Precedence
        ("2 + 3 * 4", 14.0),
        ("(2 + 3) * 4", 20.0),
        ("2 ^ 3 ^ 2", 512.0),  # 2^(3^2) = 2^9 = 512
        
        # Unary
        ("-5", -5.0),
        ("--5", 5.0),
        ("3 + -2", 1.0),
        
        # Let bindings
        ("let x = 5 in x", 5.0),
        ("let x = 5 in x + 1", 6.0),
        ("let x = 2 in let y = 3 in x + y", 5.0),
        ("let x = 5 in let x = 10 in x", 10.0),  # shadowing
        
        # Built-in functions
        ("abs(-5)", 5.0),
        ("max(3, 7)", 7.0),
        ("min(3, 7)", 3.0),
        ("floor(3.7)", 3.0),
        ("ceil(3.2)", 4.0),
        
        # Comparisons
        ("1 < 2", 1.0),
        ("2 < 1", 0.0),
        ("5 == 5", 1.0),
        ("5 != 5", 0.0),
        
        # If expressions
        ("if 1 > 0 then 10 else 20", 10.0),
        ("if 0 > 1 then 10 else 20", 20.0),
        ("if 1 == 1 then 100 else 200", 100.0),
        
        # Lambda and application
        ("let square = fun x -> x * x in square(5)", 25.0),
        ("let add1 = fun x -> x + 1 in add1(10)", 11.0),
        
        # Complex expressions
        ("let a = 2 in let b = 3 in let c = 4 in a * b + c", 10.0),
        ("let f = fun x -> x * x in let g = fun y -> y + 1 in f(g(3))", 16.0),
    ]
    
    passed = 0
    failed = 0
    
    for expr_text, expected in tests:
        try:
            result = evaluate(expr_text)
            if not isinstance(result, (int, float)):
                raise TypeError("Expected a numeric result")
            if not isinstance(result, (int, float)):
                raise TypeError("Expected a numeric result")
            if abs(float(result) - expected) < 1e-10:
                print(f"  ✓ {expr_text} = {result}")
                passed += 1
            else:
                print(f"  ✗ {expr_text} = {result} (expected {expected})")
                failed += 1
        except Exception as e:
            print(f"  ✗ {expr_text} raised {type(e).__name__}: {e}")
            failed += 1
    
    print()
    print(f"Results: {passed} passed, {failed} failed")
    print()
    
    return passed, failed


def demo_ast_printing() -> None:
    """Demonstrate AST structure visualisation."""
    logger.info("Running AST demonstration")
    print("=" * 60)
    print("AST STRUCTURE DEMONSTRATION")
    print("=" * 60)
    
    examples = [
        "2 + 3 * 4",
        "let x = 5 in x + 1",
        "fun x -> x * x",
        "if a > b then a else b",
    ]
    
    for text in examples:
        print(f"\nExpression: {text}")
        ast = parse(text)
        print(f"AST: {ast}")
    
    print()


def demo_step_by_step() -> None:
    """Demonstrate step-by-step evaluation."""
    logger.info("Running step-by-step demonstration")
    print("=" * 60)
    print("STEP-BY-STEP EVALUATION")
    print("=" * 60)
    
    expressions = [
        ("Basic: 2 + 3 * 4", "2 + 3 * 4"),
        ("Let: let x = 5 in x * 2", "let x = 5 in x * 2"),
        ("Lambda: let f = fun x -> x + 1 in f(10)", "let f = fun x -> x + 1 in f(10)"),
        ("Conditional: if 5 > 3 then 100 else 200", "if 5 > 3 then 100 else 200"),
    ]
    
    for label, expr_text in expressions:
        print(f"\n{label}")
        print(f"  Expression: {expr_text}")
        print(f"  AST: {parse(expr_text)}")
        print(f"  Result: {evaluate(expr_text)}")
    
    print()


def run_all_demos() -> None:
    """Run all demonstration functions."""
    run_tests()
    demo_ast_printing()
    demo_step_by_step()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: EXERCISES
# ═══════════════════════════════════════════════════════════════════════════════

"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 1: Logical Operators                                                 ║
║                                                                               ║
║ Add support for logical operators: and, or, not                               ║
║                                                                               ║
║ Requirements:                                                                 ║
║ - and, or should use short-circuit evaluation                                 ║
║ - not should be a unary operator                                              ║
║                                                                               ║
║ Examples:                                                                     ║
║   1 and 0       → 0 (false)                                                  ║
║   1 or 0        → 1 (true)                                                   ║
║   not 1         → 0 (false)                                                  ║
║   1 > 0 and 2 > 1  → 1 (true)                                                ║
║                                                                               ║
║ Hint: Add new token types and extend the parser and evaluator.               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 2: Lists                                                             ║
║                                                                               ║
║ Add support for lists:                                                        ║
║ - Syntax: [1, 2, 3]                                                          ║
║ - Operations: head(list), tail(list), cons(elem, list),                      ║
║               length(list), empty(list)                                       ║
║ - Concatenation: list1 ++ list2                                              ║
║                                                                               ║
║ Examples:                                                                     ║
║   head([1, 2, 3])         → 1                                                ║
║   tail([1, 2, 3])         → [2, 3]                                           ║
║   cons(0, [1, 2])         → [0, 1, 2]                                        ║
║   [1, 2] ++ [3, 4]        → [1, 2, 3, 4]                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════════╗
║ EXERCISE 3: String Support                                                    ║
║                                                                               ║
║ Add support for string literals and operations:                               ║
║ - Syntax: "hello world"                                                      ║
║ - Concatenation: str1 ++ str2                                                ║
║ - Length: length(str)                                                        ║
║ - Comparison: str1 == str2                                                   ║
║                                                                               ║
║ This requires modifying the Value type to include strings.                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="Mini-Interpreter with AST - Week 1, Lab 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lab_1_03_ast_interpreter.py --demo
  python lab_1_03_ast_interpreter.py --repl
  python lab_1_03_ast_interpreter.py --eval "let x = 5 in x * 2"
  python lab_1_03_ast_interpreter.py --ast "2 + 3 * 4"

The interpreter supports:
  - Arithmetic: +, -, *, /, ^, %
  - Comparison: <, >, <=, >=, ==, !=
  - Let bindings: let x = expr in body
  - Lambdas: fun x -> body
  - Conditionals: if cond then expr1 else expr2
  - Built-ins: sin, cos, sqrt, abs, min, max, etc.
        """
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all demonstrations"
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Start interactive REPL"
    )
    parser.add_argument(
        "--eval", "-e",
        metavar="EXPR",
        help="Evaluate an expression"
    )
    parser.add_argument(
        "--ast",
        metavar="EXPR",
        help="Show AST for an expression"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test suite"
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
        print("  WEEK 1, LAB 3: MINI-INTERPRETER WITH AST")
        print("═" * 60 + "\n")
        run_all_demos()
        print("=" * 60)
        print("Exercises to complete in code:")
        print("  1. Add logical operators (and, or, not)")
        print("  2. Add list support")
        print("  3. Add string support")
        print("=" * 60)
    elif args.repl:
        repl()
    elif args.eval:
        try:
            result = evaluate(args.eval)
            print(result)
        except Exception as e:
            print(f"Error: {e}")
    elif args.ast:
        try:
            ast = parse(args.ast)
            print(ast)
        except Exception as e:
            print(f"Error: {e}")
    elif args.test:
        passed, failed = run_tests()
        exit(0 if failed == 0 else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
