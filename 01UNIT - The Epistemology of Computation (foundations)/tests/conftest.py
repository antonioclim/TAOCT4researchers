"""
Pytest configuration and fixtures for Week 1 tests.

This module provides shared fixtures and configuration for all Week 1 tests.
"""

import sys
from pathlib import Path

import pytest

# Add the lab directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))


# ═══════════════════════════════════════════════════════════════════════════════
# TURING MACHINE FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def increment_machine():
    """Fixture providing the unary increment Turing machine."""
    from lab_1_01_turing_machine import create_unary_increment_machine
    return create_unary_increment_machine()


@pytest.fixture
def addition_machine():
    """Fixture providing the unary addition Turing machine."""
    from lab_1_01_turing_machine import create_unary_addition_machine
    return create_unary_addition_machine()


@pytest.fixture
def palindrome_machine():
    """Fixture providing the palindrome checker Turing machine."""
    from lab_1_01_turing_machine import create_palindrome_checker
    return create_palindrome_checker()


@pytest.fixture
def successor_machine():
    """Fixture providing the binary successor Turing machine."""
    from lab_1_01_turing_machine import create_binary_successor_machine
    return create_binary_successor_machine()


@pytest.fixture
def simulator():
    """Fixture providing a fresh TuringSimulator factory."""
    from lab_1_01_turing_machine import TuringSimulator
    
    def _make_simulator(machine):
        return TuringSimulator(machine)
    
    return _make_simulator


# ═══════════════════════════════════════════════════════════════════════════════
# LAMBDA CALCULUS FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def lambda_var():
    """Fixture providing the Var constructor."""
    from lab_1_02_lambda_calculus import Var
    return Var


@pytest.fixture
def lambda_abs():
    """Fixture providing the Abs constructor."""
    from lab_1_02_lambda_calculus import Abs
    return Abs


@pytest.fixture
def lambda_app():
    """Fixture providing the App constructor."""
    from lab_1_02_lambda_calculus import App
    return App


@pytest.fixture
def church_true():
    """Fixture providing Church TRUE."""
    from lab_1_02_lambda_calculus import TRUE
    return TRUE


@pytest.fixture
def church_false():
    """Fixture providing Church FALSE."""
    from lab_1_02_lambda_calculus import FALSE
    return FALSE


# ═══════════════════════════════════════════════════════════════════════════════
# AST INTERPRETER FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def parser():
    """Fixture providing the parse function."""
    from lab_1_03_ast_interpreter import parse
    return parse


@pytest.fixture
def evaluator():
    """Fixture providing the evaluate function."""
    from lab_1_03_ast_interpreter import evaluate
    return evaluate


@pytest.fixture
def ast_nodes():
    """Fixture providing all AST node constructors."""
    from lab_1_03_ast_interpreter import (
        Num, Var, BinOp, UnaryOp, FuncCall, Let, Lambda, IfExpr
    )
    return {
        'Num': Num,
        'Var': Var,
        'BinOp': BinOp,
        'UnaryOp': UnaryOp,
        'FuncCall': FuncCall,
        'Let': Let,
        'Lambda': Lambda,
        'IfExpr': IfExpr,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
