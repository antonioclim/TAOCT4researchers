#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 1, Lab 3: Mini-Interpreter cu AST și Pattern Matching
═══════════════════════════════════════════════════════════════════════════════

MOTIVAȚIE
─────────
Orice limbaj de programare, de la Python la MATLAB, de la R la Julia, 
funcționează pe același principiu:

    Text → Tokens → AST → Evaluare/Compilare

Înțelegerea acestui flux vă permite să:
1. Construiți propriile DSL-uri (Domain Specific Languages) pentru cercetare
2. Înțelegeți cum funcționează tools ca linters, formatters, transpilers
3. Gândiți mai clar despre structura logică a computației

EXEMPLU DIN LUMEA REALĂ
───────────────────────
Jupyter Notebooks procesează fiecare celulă astfel:
1. Textul Python e parsat într-un AST
2. AST-ul e compilat în bytecode
3. Bytecode-ul e executat de Python VM
4. Rezultatul e capturat și afișat

Când scrieți `2 + 2` în Jupyter, Python face:
- Lexer: ['2', '+', '2'] 
- Parser: BinOp(Num(2), Add(), Num(2))
- Compiler: LOAD_CONST 2, LOAD_CONST 2, BINARY_ADD
- VM: Stack [2] → Stack [2, 2] → Stack [4]
- Output: 4

OBIECTIVE
─────────
1. Implementarea unui lexer simplu
2. Implementarea unui parser recursive-descent
3. Implementarea unui evaluator cu pattern matching
4. Extinderea cu variabile și funcții

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, Callable
from enum import Enum, auto
import re
import math


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: DEFINIREA AST-ULUI
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Num:
    """
    Nod pentru literale numerice.
    
    În AST, acest nod reprezintă o valoare constantă.
    E "frozen" pentru că AST-urile trebuie să fie imutabile
    (facilitează caching, comparații, debugging).
    """
    value: float
    
    def __repr__(self) -> str:
        # Afișăm întregi fără zecimale
        if self.value == int(self.value):
            return f"Num({int(self.value)})"
        return f"Num({self.value})"


@dataclass(frozen=True)
class Var:
    """
    Nod pentru variabile (referințe la valori denumite).
    
    Exemplu: în expresia "x + 1", "x" devine Var("x")
    """
    name: str
    
    def __repr__(self) -> str:
        return f"Var({self.name!r})"


@dataclass(frozen=True)
class BinOp:
    """
    Nod pentru operații binare (două operanzi).
    
    Operatorul e stocat ca string pentru simplitate.
    Într-un compilator real, ar fi un enum sau un nod separat.
    """
    left: 'Expr'
    op: str
    right: 'Expr'
    
    def __repr__(self) -> str:
        return f"BinOp({self.left!r}, {self.op!r}, {self.right!r})"


@dataclass(frozen=True)
class UnaryOp:
    """
    Nod pentru operații unare (un singur operand).
    
    Exemple: negație (-x), not logic (!x)
    """
    op: str
    operand: 'Expr'
    
    def __repr__(self) -> str:
        return f"UnaryOp({self.op!r}, {self.operand!r})"


@dataclass(frozen=True)
class FuncCall:
    """
    Nod pentru apeluri de funcții.
    
    Exemplu: sin(x), max(a, b)
    """
    name: str
    args: tuple['Expr', ...]
    
    def __repr__(self) -> str:
        args_str = ", ".join(repr(a) for a in self.args)
        return f"FuncCall({self.name!r}, ({args_str}))"


@dataclass(frozen=True)
class Let:
    """
    Nod pentru legarea variabilelor (let-binding).
    
    Sintaxă: let x = expr1 in expr2
    
    Semantică: evaluează expr2 într-un context unde x = valoarea lui expr1
    
    Exemplu: let x = 5 in x + 1  →  6
    """
    name: str
    value: 'Expr'
    body: 'Expr'
    
    def __repr__(self) -> str:
        return f"Let({self.name!r}, {self.value!r}, {self.body!r})"


@dataclass(frozen=True)
class Lambda:
    """
    Nod pentru funcții anonime (lambda expressions).
    
    Sintaxă: fun x -> expr
    
    Exemplu: fun x -> x * x  (funcția care ridică la pătrat)
    """
    param: str
    body: 'Expr'
    
    def __repr__(self) -> str:
        return f"Lambda({self.param!r}, {self.body!r})"


@dataclass(frozen=True)
class IfExpr:
    """
    Nod pentru expresii condiționale.
    
    Sintaxă: if cond then expr1 else expr2
    
    Spre deosebire de statement-uri, expresiile condiționale
    returnează o valoare.
    """
    condition: 'Expr'
    then_branch: 'Expr'
    else_branch: 'Expr'
    
    def __repr__(self) -> str:
        return f"IfExpr({self.condition!r}, {self.then_branch!r}, {self.else_branch!r})"


# Tipul uniune pentru toate expresiile
Expr = Union[Num, Var, BinOp, UnaryOp, FuncCall, Let, Lambda, IfExpr]


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: LEXER (Tokenizer)
# ═══════════════════════════════════════════════════════════════════════════════

class TokenType(Enum):
    """Tipuri de tokeni recunoscuți de lexer."""
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
    """Un token individual produs de lexer."""
    type: TokenType
    value: str | float
    position: int  # Poziția în textul original (pentru error reporting)
    
    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r})"


class Lexer:
    """
    Analizor lexical (tokenizer).
    
    Transformă un șir de caractere într-o secvență de tokeni.
    
    Exemplu:
        "let x = 5 in x + 1"
        →
        [LET, IDENTIFIER(x), EQUALS, NUMBER(5), IN, IDENTIFIER(x), PLUS, NUMBER(1), EOF]
    """
    
    # Cuvinte cheie rezervate
    KEYWORDS = {
        'let': TokenType.LET,
        'in': TokenType.IN,
        'fun': TokenType.FUN,
        'if': TokenType.IF,
        'then': TokenType.THEN,
        'else': TokenType.ELSE,
    }
    
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0
        self.current_char: str | None = text[0] if text else None
    
    def error(self, message: str) -> None:
        """Aruncă o excepție cu context."""
        context = self.text[max(0, self.pos - 10):self.pos + 10]
        raise SyntaxError(f"{message} at position {self.pos}: ...{context}...")
    
    def advance(self) -> None:
        """Avansează la următorul caracter."""
        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None
    
    def peek(self) -> str | None:
        """Privește următorul caracter fără a avansa."""
        peek_pos = self.pos + 1
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None
    
    def skip_whitespace(self) -> None:
        """Sare peste spații."""
        while self.current_char is not None and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self) -> None:
        """Sare peste comentarii (# până la newline)."""
        while self.current_char is not None and self.current_char != '\n':
            self.advance()
    
    def number(self) -> Token:
        """Parsează un număr (întreg sau cu virgulă)."""
        start_pos = self.pos
        result = ''
        
        # Partea întreagă
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        
        # Partea zecimală
        if self.current_char == '.' and self.peek() and self.peek().isdigit():
            result += self.current_char
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                result += self.current_char
                self.advance()
        
        # Notație științifică (opțional)
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
        """Parsează un identificator sau cuvânt cheie."""
        start_pos = self.pos
        result = ''
        
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            result += self.current_char
            self.advance()
        
        # Verificăm dacă e keyword
        token_type = self.KEYWORDS.get(result, TokenType.IDENTIFIER)
        return Token(token_type, result, start_pos)
    
    def get_next_token(self) -> Token:
        """Returnează următorul token."""
        while self.current_char is not None:
            # Whitespace
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            # Comentariu
            if self.current_char == '#':
                self.skip_comment()
                continue
            
            # Număr
            if self.current_char.isdigit():
                return self.number()
            
            # Identificator sau keyword
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            
            # Operatori multi-caracter
            if self.current_char == '-' and self.peek() == '>':
                token = Token(TokenType.ARROW, '->', self.pos)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '<' and self.peek() == '=':
                token = Token(TokenType.LE, '<=', self.pos)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '>' and self.peek() == '=':
                token = Token(TokenType.GE, '>=', self.pos)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '=' and self.peek() == '=':
                token = Token(TokenType.EQ, '==', self.pos)
                self.advance()
                self.advance()
                return token
            
            if self.current_char == '!' and self.peek() == '=':
                token = Token(TokenType.NE, '!=', self.pos)
                self.advance()
                self.advance()
                return token
            
            # Operatori single-caracter
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
                token = Token(single_char_tokens[self.current_char], self.current_char, self.pos)
                self.advance()
                return token
            
            self.error(f"Unknown character: {self.current_char!r}")
        
        return Token(TokenType.EOF, '', self.pos)
    
    def tokenize(self) -> list[Token]:
        """Returnează lista completă de tokeni."""
        tokens = []
        while True:
            token = self.get_next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """
    Parser recursive-descent cu precedență de operatori.
    
    Gramatica (simplificată):
    
        expr      ::= let_expr | lambda_expr | if_expr | or_expr
        let_expr  ::= 'let' IDENT '=' expr 'in' expr
        lambda    ::= 'fun' IDENT '->' expr
        if_expr   ::= 'if' expr 'then' expr 'else' expr
        or_expr   ::= and_expr (('||') and_expr)*
        and_expr  ::= cmp_expr (('&&') cmp_expr)*
        cmp_expr  ::= add_expr (('<'|'>'|'<='|'>='|'=='|'!=') add_expr)*
        add_expr  ::= mul_expr (('+'|'-') mul_expr)*
        mul_expr  ::= pow_expr (('*'|'/'|'%') pow_expr)*
        pow_expr  ::= unary ('^' pow_expr)?    # right-associative
        unary     ::= '-' unary | primary
        primary   ::= NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
        args      ::= expr (',' expr)*
    """
    
    def __init__(self, lexer: Lexer) -> None:
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()
    
    def error(self, message: str) -> None:
        """Aruncă o excepție de parsare."""
        raise SyntaxError(f"Parse error: {message}. Got {self.current_token}")
    
    def eat(self, token_type: TokenType) -> Token:
        """Consumă un token de tipul specificat."""
        if self.current_token.type == token_type:
            token = self.current_token
            self.current_token = self.lexer.get_next_token()
            return token
        self.error(f"Expected {token_type.name}")
    
    def parse(self) -> Expr:
        """Punct de intrare principal."""
        result = self.expr()
        if self.current_token.type != TokenType.EOF:
            self.error("Unexpected token after expression")
        return result
    
    def expr(self) -> Expr:
        """Parsează o expresie completă."""
        # Let expression
        if self.current_token.type == TokenType.LET:
            return self.let_expr()
        
        # Lambda expression
        if self.current_token.type == TokenType.FUN:
            return self.lambda_expr()
        
        # If expression
        if self.current_token.type == TokenType.IF:
            return self.if_expr()
        
        return self.comparison()
    
    def let_expr(self) -> Expr:
        """Parsează: let x = value in body"""
        self.eat(TokenType.LET)
        name_token = self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.EQUALS)
        value = self.expr()
        self.eat(TokenType.IN)
        body = self.expr()
        return Let(str(name_token.value), value, body)
    
    def lambda_expr(self) -> Expr:
        """Parsează: fun x -> body"""
        self.eat(TokenType.FUN)
        param_token = self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.ARROW)
        body = self.expr()
        return Lambda(str(param_token.value), body)
    
    def if_expr(self) -> Expr:
        """Parsează: if cond then expr1 else expr2"""
        self.eat(TokenType.IF)
        condition = self.expr()
        self.eat(TokenType.THEN)
        then_branch = self.expr()
        self.eat(TokenType.ELSE)
        else_branch = self.expr()
        return IfExpr(condition, then_branch, else_branch)
    
    def comparison(self) -> Expr:
        """Parsează operatori de comparație."""
        left = self.additive()
        
        while self.current_token.type in (TokenType.LT, TokenType.GT, 
                                          TokenType.LE, TokenType.GE,
                                          TokenType.EQ, TokenType.NE):
            op_token = self.current_token
            self.eat(op_token.type)
            right = self.additive()
            left = BinOp(left, str(op_token.value), right)
        
        return left
    
    def additive(self) -> Expr:
        """Parsează + și -."""
        left = self.multiplicative()
        
        while self.current_token.type in (TokenType.PLUS, TokenType.MINUS):
            op_token = self.current_token
            self.eat(op_token.type)
            right = self.multiplicative()
            left = BinOp(left, str(op_token.value), right)
        
        return left
    
    def multiplicative(self) -> Expr:
        """Parsează *, / și %."""
        left = self.power()
        
        while self.current_token.type in (TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op_token = self.current_token
            self.eat(op_token.type)
            right = self.power()
            left = BinOp(left, str(op_token.value), right)
        
        return left
    
    def power(self) -> Expr:
        """Parsează ^ (right-associative)."""
        base = self.unary()
        
        if self.current_token.type == TokenType.CARET:
            self.eat(TokenType.CARET)
            # Recursiv pentru right-associativity: 2^3^4 = 2^(3^4)
            exponent = self.power()
            return BinOp(base, '^', exponent)
        
        return base
    
    def unary(self) -> Expr:
        """Parsează operatori unari."""
        if self.current_token.type == TokenType.MINUS:
            self.eat(TokenType.MINUS)
            operand = self.unary()
            return UnaryOp('-', operand)
        
        return self.primary()
    
    def primary(self) -> Expr:
        """Parsează expresii primare: numere, variabile, apeluri de funcții, paranteze."""
        token = self.current_token
        
        # Număr
        if token.type == TokenType.NUMBER:
            self.eat(TokenType.NUMBER)
            return Num(float(token.value))
        
        # Identificator (variabilă sau apel de funcție)
        if token.type == TokenType.IDENTIFIER:
            self.eat(TokenType.IDENTIFIER)
            name = str(token.value)
            
            # Verificăm dacă e apel de funcție
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
        
        # Expresie între paranteze
        if token.type == TokenType.LPAREN:
            self.eat(TokenType.LPAREN)
            result = self.expr()
            self.eat(TokenType.RPAREN)
            return result
        
        self.error(f"Unexpected token: {token}")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

# Tip pentru valorile din runtime
Value = Union[float, 'Closure']


@dataclass
class Closure:
    """
    O închidere (closure) reprezintă o funcție împreună cu mediul său lexical.
    
    Când definim o funcție cu `fun x -> body`, captăm mediul curent
    pentru ca `body` să poată accesa variabilele din acel punct.
    """
    param: str
    body: Expr
    env: dict[str, Value]
    
    def __repr__(self) -> str:
        return f"<closure: fun {self.param} -> ...>"


# Mediul de execuție: mapare nume → valoare
Environment = dict[str, Value]


# Funcții built-in
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
    """
    Evaluator de expresii cu suport pentru:
    - Operații aritmetice
    - Variabile și let-bindings
    - Funcții anonime (lambda)
    - Expresii condiționale
    - Funcții built-in matematice
    """
    
    def __init__(self) -> None:
        self.builtins = BUILTINS
    
    def evaluate(self, expr: Expr, env: Environment | None = None) -> Value:
        """
        Evaluează o expresie în contextul unui mediu.
        
        Args:
            expr: Expresia de evaluat
            env: Mediul de execuție (mapping variabilă → valoare)
            
        Returns:
            Valoarea expresiei (float sau Closure)
        """
        if env is None:
            env = {}
        
        match expr:
            # Literal numeric
            case Num(value):
                return value
            
            # Variabilă
            case Var(name):
                if name in env:
                    return env[name]
                raise NameError(f"Undefined variable: {name}")
            
            # Operație binară
            case BinOp(left, op, right):
                l = self.evaluate(left, env)
                r = self.evaluate(right, env)
                
                # Verificăm că sunt numere
                if not isinstance(l, (int, float)) or not isinstance(r, (int, float)):
                    raise TypeError(f"Cannot apply {op} to non-numeric values")
                
                match op:
                    case '+': return l + r
                    case '-': return l - r
                    case '*': return l * r
                    case '/':
                        if r == 0:
                            raise ZeroDivisionError("Division by zero")
                        return l / r
                    case '^': return l ** r
                    case '%': return l % r
                    case '<': return 1.0 if l < r else 0.0
                    case '>': return 1.0 if l > r else 0.0
                    case '<=': return 1.0 if l <= r else 0.0
                    case '>=': return 1.0 if l >= r else 0.0
                    case '==': return 1.0 if l == r else 0.0
                    case '!=': return 1.0 if l != r else 0.0
                    case _:
                        raise ValueError(f"Unknown operator: {op}")
            
            # Operație unară
            case UnaryOp(op, operand):
                val = self.evaluate(operand, env)
                if not isinstance(val, (int, float)):
                    raise TypeError(f"Cannot apply unary {op} to non-numeric value")
                
                match op:
                    case '-': return -val
                    case _:
                        raise ValueError(f"Unknown unary operator: {op}")
            
            # Apel de funcție
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
                            raise TypeError(f"Function {name} expects 1 argument, got {len(args)}")
                        arg_val = self.evaluate(args[0], env)
                        new_env = {**func.env, func.param: arg_val}
                        return self.evaluate(func.body, new_env)
                
                raise NameError(f"Unknown function: {name}")
            
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
                
                # 0.0 e false, orice altceva e true
                if cond_val != 0.0:
                    return self.evaluate(then_branch, env)
                else:
                    return self.evaluate(else_branch, env)
            
            case _:
                raise ValueError(f"Unknown expression type: {expr}")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: INTERFAȚĂ
# ═══════════════════════════════════════════════════════════════════════════════

def parse(text: str) -> Expr:
    """Helper pentru parsare."""
    lexer = Lexer(text)
    parser = Parser(lexer)
    return parser.parse()


def evaluate(text: str, env: Environment | None = None) -> Value:
    """Helper pentru evaluare."""
    expr = parse(text)
    evaluator = Evaluator()
    return evaluator.evaluate(expr, env)


def repl() -> None:
    """Read-Eval-Print Loop interactiv."""
    print("Mini-Interpreter REPL")
    print("Type 'quit' to exit, 'help' for examples")
    print()
    
    env: Environment = {}
    evaluator = Evaluator()
    
    while True:
        try:
            text = input(">>> ").strip()
            
            if not text:
                continue
            
            if text.lower() == 'quit':
                break
            
            if text.lower() == 'help':
                print("""
Examples:
  2 + 3 * 4           # Basic arithmetic
  let x = 5 in x + 1  # Let bindings
  sin(3.14159 / 2)    # Built-in functions
  fun x -> x * x      # Lambda functions
  if 1 > 0 then 1 else 0  # Conditionals
  2 ^ 3 ^ 2           # Power (right-associative)
                """)
                continue
            
            expr = parse(text)
            print(f"AST: {expr}")
            
            result = evaluator.evaluate(expr, env)
            print(f"Result: {result}")
            
        except (SyntaxError, NameError, TypeError, ZeroDivisionError, ValueError) as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted")
            break
        except EOFError:
            break
    
    print("Goodbye!")


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA VI: TESTE ȘI DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def run_tests() -> None:
    """Rulează suite de teste pentru validare."""
    print("=" * 60)
    print("RUNNING TESTS")
    print("=" * 60)
    
    tests = [
        # Aritmetică de bază
        ("2 + 3", 5.0),
        ("10 - 4", 6.0),
        ("3 * 4", 12.0),
        ("15 / 3", 5.0),
        ("2 ^ 3", 8.0),
        ("10 % 3", 1.0),
        
        # Precedență
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
        
        # Funcții built-in
        ("abs(-5)", 5.0),
        ("max(3, 7)", 7.0),
        ("min(3, 7)", 3.0),
        ("floor(3.7)", 3.0),
        ("ceil(3.2)", 4.0),
        
        # Comparații
        ("1 < 2", 1.0),
        ("2 < 1", 0.0),
        ("5 == 5", 1.0),
        ("5 != 5", 0.0),
        
        # If expressions
        ("if 1 > 0 then 10 else 20", 10.0),
        ("if 0 > 1 then 10 else 20", 20.0),
        ("if 1 == 1 then 100 else 200", 100.0),
        
        # Lambda și aplicare
        ("let square = fun x -> x * x in square(5)", 25.0),
        ("let add1 = fun x -> x + 1 in add1(10)", 11.0),
        
        # Expresii complexe
        ("let a = 2 in let b = 3 in let c = 4 in a * b + c", 10.0),
        ("let f = fun x -> x * x in let g = fun y -> y + 1 in f(g(3))", 16.0),
    ]
    
    passed = 0
    failed = 0
    
    for expr_text, expected in tests:
        try:
            result = evaluate(expr_text)
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


def demo_ast_printing() -> None:
    """Demonstrație: afișarea AST-urilor."""
    print("=" * 60)
    print("AST STRUCTURE DEMO")
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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--repl":
        repl()
    else:
        run_tests()
        demo_ast_printing()
        
        print("=" * 60)
        print("Run with --repl flag for interactive mode")
        print("=" * 60)
