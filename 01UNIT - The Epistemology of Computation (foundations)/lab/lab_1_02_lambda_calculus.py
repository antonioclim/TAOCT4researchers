"""Compatibility wrapper for the unit test suite.

The canonical implementation is in ``lab_01_02_lambda_calculus.py``. Some
external materials use the shorter import name ``lab_1_02_lambda_calculus``.
This wrapper preserves that import path without duplicating logic.
"""

from __future__ import annotations

from lab_01_02_lambda_calculus import *  # noqa: F403
