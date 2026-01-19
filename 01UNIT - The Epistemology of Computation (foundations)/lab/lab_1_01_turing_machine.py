"""Compatibility wrapper for the unit test suite.

The canonical implementation is in ``lab_01_01_turing_machine.py``. Some
external materials use the shorter import name ``lab_1_01_turing_machine``.
This wrapper preserves that import path without duplicating logic.
"""

from __future__ import annotations

from lab_01_01_turing_machine import *  # noqa: F403
