"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT: Abstraction and Encapsulation — Laboratory Package
═══════════════════════════════════════════════════════════════════════════════

This package contains the laboratory materials for 02UNIT.

Modules:
    lab_02_01_simulation_framework: OOP simulation framework with Protocols
    lab_02_02_design_patterns: Design patterns catalogue for research software

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from .lab_02_01_simulation_framework import (
    Simulable,
    SimulationRunner,
    SimulationResult,
    SIRState,
    SIRSimulation,
    NBodyState,
    NBodySimulation,
    Body,
)

from .lab_02_02_design_patterns import (
    IntegrationStrategy,
    RectangleRule,
    TrapezoidRule,
    SimpsonRule,
    NumericalIntegrator,
    Observable,
    Observer,
    AgentFactory,
    AgentPopulation,
    Command,
    CommandHistory,
)

__all__ = [
    # Lab 1: Simulation Framework
    'Simulable',
    'SimulationRunner',
    'SimulationResult',
    'SIRState',
    'SIRSimulation',
    'NBodyState',
    'NBodySimulation',
    'Body',
    # Lab 2: Design Patterns
    'IntegrationStrategy',
    'RectangleRule',
    'TrapezoidRule',
    'SimpsonRule',
    'NumericalIntegrator',
    'Observable',
    'Observer',
    'AgentFactory',
    'AgentPopulation',
    'Command',
    'CommandHistory',
]

__version__ = '2.0.0'
__author__ = 'Antonio Clim'
