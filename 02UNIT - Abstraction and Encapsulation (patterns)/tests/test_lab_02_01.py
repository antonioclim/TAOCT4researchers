#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
02UNIT: Tests for Lab 2.01 — Simulation Framework
═══════════════════════════════════════════════════════════════════════════════

Test suite for the OOP simulation framework.

Coverage targets:
- SIRSimulation class
- NBodySimulation class
- SimulationRunner class
- Simulable protocol compliance

© 2025 Antonio Clim. All rights reserved.
═══════════════════════════════════════════════════════════════════════════════
"""

import pytest
import numpy as np
from typing import Any
import sys
from pathlib import Path

# Add lab directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "lab"))

from lab_02_01_simulation_framework import (
    SIRSimulation,
    SIRState,
    NBodySimulation,
    NBodyState,
    SimulationRunner,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SIR SIMULATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSIRSimulation:
    """Tests for SIR epidemic model simulation."""
    
    def test_initial_state(self, sir_initial_state, sir_parameters):
        """Test that initial state is set correctly."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        state = sim.state()
        assert abs(state.susceptible - 0.99) < 1e-9
        assert abs(state.infected - 0.01) < 1e-9
        assert abs(state.recovered - 0.0) < 1e-9
    
    def test_conservation(self, sir_initial_state, sir_parameters):
        """Test that S + I + R = 1 is conserved."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        # Run 100 steps
        for _ in range(100):
            sim.step(0.1)
            state = sim.state()
            total = state.susceptible + state.infected + state.recovered
            assert abs(total - 1.0) < 1e-6, f"Conservation violated: S+I+R = {total}"
    
    def test_epidemic_growth(self, sir_initial_state, sir_parameters):
        """Test that infection grows initially when R0 > 1."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        initial_infected = sim.state().infected
        
        # Run a few steps
        for _ in range(10):
            sim.step(0.1)
        
        # With R0 = beta/gamma = 0.3/0.1 = 3 > 1, infections should grow
        assert sim.state().infected > initial_infected
    
    def test_recovery_only(self):
        """Test that with no susceptibles, only recovery occurs."""
        sim = SIRSimulation(
            susceptible=0.0,
            infected=1.0,
            recovered=0.0,
            beta=0.3,
            gamma=0.1,
        )
        
        for _ in range(100):
            sim.step(0.1)
        
        state = sim.state()
        # Most should have recovered
        assert state.recovered > 0.9
        assert state.infected < 0.1
    
    def test_negative_values_prevented(self):
        """Test that state values don't go negative."""
        sim = SIRSimulation(
            susceptible=0.001,
            infected=0.999,
            recovered=0.0,
            beta=0.3,
            gamma=0.5,  # High recovery rate
        )
        
        for _ in range(200):
            sim.step(0.1)
            state = sim.state()
            assert state.susceptible >= 0
            assert state.infected >= 0
            assert state.recovered >= 0
    
    def test_is_done_condition(self):
        """Test that simulation ends when infection is negligible."""
        sim = SIRSimulation(
            susceptible=0.5,
            infected=0.5,
            recovered=0.0,
            beta=0.1,
            gamma=0.5,  # Very fast recovery
        )
        
        max_steps = 1000
        steps = 0
        while not sim.is_done() and steps < max_steps:
            sim.step(0.1)
            steps += 1
        
        assert sim.is_done(), "Simulation should have completed"
        assert sim.state().infected < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# N-BODY SIMULATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestNBodySimulation:
    """Tests for N-body gravitational simulation."""
    
    def test_initial_state(
        self,
        nbody_initial_positions,
        nbody_initial_velocities,
        nbody_masses,
    ):
        """Test that initial state is set correctly."""
        sim = NBodySimulation(
            positions=nbody_initial_positions.copy(),
            velocities=nbody_initial_velocities.copy(),
            masses=nbody_masses.copy(),
        )
        
        state = sim.state()
        np.testing.assert_array_almost_equal(
            state.positions, nbody_initial_positions
        )
        np.testing.assert_array_almost_equal(
            state.velocities, nbody_initial_velocities
        )
    
    def test_energy_conservation(
        self,
        nbody_initial_positions,
        nbody_initial_velocities,
        nbody_masses,
    ):
        """Test that total energy is approximately conserved."""
        sim = NBodySimulation(
            positions=nbody_initial_positions.copy(),
            velocities=nbody_initial_velocities.copy(),
            masses=nbody_masses.copy(),
            G=1.0,
        )
        
        def compute_energy(state: NBodyState, masses: np.ndarray, G: float) -> float:
            """Compute total energy (kinetic + potential)."""
            # Kinetic energy
            ke = 0.5 * np.sum(masses[:, np.newaxis] * state.velocities**2)
            
            # Potential energy
            pe = 0.0
            n = len(masses)
            for i in range(n):
                for j in range(i + 1, n):
                    r = np.linalg.norm(state.positions[i] - state.positions[j])
                    if r > 1e-10:
                        pe -= G * masses[i] * masses[j] / r
            
            return ke + pe
        
        initial_energy = compute_energy(sim.state(), nbody_masses, 1.0)
        
        # Run simulation
        for _ in range(100):
            sim.step(0.01)
        
        final_energy = compute_energy(sim.state(), nbody_masses, 1.0)
        
        # Energy should be conserved within reasonable tolerance
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        assert relative_error < 0.1, f"Energy not conserved: {relative_error:.2%} error"
    
    def test_two_body_orbit(self):
        """Test that two-body system produces stable orbit."""
        # Sun-like body at centre, planet in circular orbit
        positions = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
        ])
        
        # Circular orbit velocity: v = sqrt(GM/r)
        v_circular = 1.0  # Simplified
        velocities = np.array([
            [0.0, 0.0],
            [0.0, v_circular],
        ])
        
        masses = np.array([1.0, 0.001])
        
        sim = NBodySimulation(
            positions=positions,
            velocities=velocities,
            masses=masses,
            G=1.0,
        )
        
        initial_distance = np.linalg.norm(sim.state().positions[1])
        
        # Run for a while
        for _ in range(100):
            sim.step(0.01)
        
        final_distance = np.linalg.norm(sim.state().positions[1])
        
        # Distance from centre should remain roughly constant
        assert abs(final_distance - initial_distance) < 0.5
    
    def test_time_tracking(
        self,
        nbody_initial_positions,
        nbody_initial_velocities,
        nbody_masses,
    ):
        """Test that simulation time is tracked correctly."""
        sim = NBodySimulation(
            positions=nbody_initial_positions.copy(),
            velocities=nbody_initial_velocities.copy(),
            masses=nbody_masses.copy(),
        )
        
        assert sim.state().time == 0.0
        
        sim.step(0.1)
        assert abs(sim.state().time - 0.1) < 1e-9
        
        sim.step(0.1)
        assert abs(sim.state().time - 0.2) < 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION RUNNER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulationRunner:
    """Tests for the generic SimulationRunner."""
    
    def test_run_sir_simulation(self, sir_initial_state, sir_parameters):
        """Test running SIR simulation through runner."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        runner = SimulationRunner(sim)
        history = runner.run(dt=0.1, max_steps=50)
        
        assert len(history) > 0
        assert len(history) <= 51  # Initial + up to 50 steps
    
    def test_run_nbody_simulation(
        self,
        nbody_initial_positions,
        nbody_initial_velocities,
        nbody_masses,
    ):
        """Test running N-body simulation through runner."""
        sim = NBodySimulation(
            positions=nbody_initial_positions.copy(),
            velocities=nbody_initial_velocities.copy(),
            masses=nbody_masses.copy(),
        )
        
        runner = SimulationRunner(sim)
        history = runner.run(dt=0.01, max_steps=100)
        
        assert len(history) == 101  # Initial + 100 steps
    
    def test_history_recording(self, sir_initial_state, sir_parameters):
        """Test that history is recorded correctly."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        runner = SimulationRunner(sim)
        history = runner.run(dt=0.1, max_steps=10)
        
        # First state should be initial
        assert abs(history[0].susceptible - 0.99) < 1e-9
        
        # Last state should be different
        assert history[-1].susceptible != history[0].susceptible
    
    def test_early_termination(self):
        """Test that runner stops when is_done returns True."""
        sim = SIRSimulation(
            susceptible=0.0,
            infected=0.001,
            recovered=0.999,
            beta=0.3,
            gamma=0.5,
        )
        
        runner = SimulationRunner(sim)
        history = runner.run(dt=0.1, max_steps=1000)
        
        # Should terminate early
        assert len(history) < 1000


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL COMPLIANCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulableProtocol:
    """Tests for Simulable protocol compliance."""
    
    def test_sir_has_required_methods(self):
        """Test that SIRSimulation has all protocol methods."""
        sim = SIRSimulation()
        
        assert hasattr(sim, 'step')
        assert hasattr(sim, 'state')
        assert hasattr(sim, 'is_done')
        assert callable(sim.step)
        assert callable(sim.state)
        assert callable(sim.is_done)
    
    def test_nbody_has_required_methods(
        self,
        nbody_initial_positions,
        nbody_initial_velocities,
        nbody_masses,
    ):
        """Test that NBodySimulation has all protocol methods."""
        sim = NBodySimulation(
            positions=nbody_initial_positions,
            velocities=nbody_initial_velocities,
            masses=nbody_masses,
        )
        
        assert hasattr(sim, 'step')
        assert hasattr(sim, 'state')
        assert hasattr(sim, 'is_done')
        assert callable(sim.step)
        assert callable(sim.state)
        assert callable(sim.is_done)
    
    def test_state_returns_correct_type(self, sir_initial_state, sir_parameters):
        """Test that state() returns the expected type."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        state = sim.state()
        assert isinstance(state, SIRState)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_sir_all_susceptible(self):
        """Test SIR with 100% susceptible (no epidemic)."""
        sim = SIRSimulation(
            susceptible=1.0,
            infected=0.0,
            recovered=0.0,
        )
        
        sim.step(0.1)
        state = sim.state()
        
        # Nothing should change
        assert state.susceptible == 1.0
        assert state.infected == 0.0
    
    def test_sir_all_recovered(self):
        """Test SIR with 100% recovered (immune population)."""
        sim = SIRSimulation(
            susceptible=0.0,
            infected=0.0,
            recovered=1.0,
        )
        
        sim.step(0.1)
        state = sim.state()
        
        # Nothing should change
        assert state.recovered == 1.0
    
    def test_nbody_single_particle(self):
        """Test N-body with single particle (no forces)."""
        positions = np.array([[1.0, 2.0]])
        velocities = np.array([[0.5, -0.5]])
        masses = np.array([1.0])
        
        sim = NBodySimulation(
            positions=positions,
            velocities=velocities,
            masses=masses,
        )
        
        sim.step(1.0)
        state = sim.state()
        
        # Should move in straight line
        expected_pos = np.array([[1.5, 1.5]])
        np.testing.assert_array_almost_equal(state.positions, expected_pos)
    
    def test_very_small_timestep(self, sir_initial_state, sir_parameters):
        """Test simulation with very small timestep."""
        sim = SIRSimulation(
            susceptible=sir_initial_state["susceptible"],
            infected=sir_initial_state["infected"],
            recovered=sir_initial_state["recovered"],
            beta=sir_parameters["beta"],
            gamma=sir_parameters["gamma"],
        )
        
        # Should not cause numerical issues
        for _ in range(100):
            sim.step(1e-6)
        
        state = sim.state()
        total = state.susceptible + state.infected + state.recovered
        assert abs(total - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
