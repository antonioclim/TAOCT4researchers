"""Compatibility wrapper for the unit test suite.

The canonical implementation is in ``lab_01_03_ast_interpreter.py``. Some
external materials use the shorter import name ``lab_1_03_ast_interpreter``.
This wrapper preserves that import path without duplicating logic.
"""

from __future__ import annotations

from lab_01_03_ast_interpreter import *  # noqa: F403
