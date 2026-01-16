"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 1: The Epistemology of Computation — Laboratory Package
═══════════════════════════════════════════════════════════════════════════════

This package contains three laboratory modules:

1. lab_1_01_turing_machine: Turing machine simulator
2. lab_1_02_lambda_calculus: Lambda calculus basics
3. lab_1_03_ast_interpreter: Expression interpreter

Each module can be run independently with the --demo flag for demonstrations
or --help for usage information.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from .lab_1_01_turing_machine import (
    TuringMachine,
    TuringSimulator,
    Configuration,
    Transition,
    Direction,
    create_unary_increment_machine,
    create_unary_addition_machine,
    create_palindrome_checker,
    create_binary_successor_machine,
)

from .lab_1_02_lambda_calculus import (
    LambdaExpr,
    Var as LambdaVar,
    Abs,
    App,
    beta_reduce,
    beta_reduce_once,
    trace_reduction,
    church_numeral,
    church_to_int,
    TRUE,
    FALSE,
)

__all__ = [
    # Turing machine
    "TuringMachine",
    "TuringSimulator",
    "Configuration",
    "Transition",
    "Direction",
    "create_unary_increment_machine",
    "create_unary_addition_machine",
    "create_palindrome_checker",
    "create_binary_successor_machine",
    # Lambda calculus
    "LambdaExpr",
    "LambdaVar",
    "Abs",
    "App",
    "beta_reduce",
    "beta_reduce_once",
    "trace_reduction",
    "church_numeral",
    "church_to_int",
    "TRUE",
    "FALSE",
]

__version__ = "1.0.0"
__author__ = "Antonio Clim"
