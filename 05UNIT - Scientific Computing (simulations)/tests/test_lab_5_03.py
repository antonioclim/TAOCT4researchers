#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 5: Scientific Computing — Agent-Based Modelling Tests
═══════════════════════════════════════════════════════════════════════════════

Test suite for lab_5_03_agent_based_modelling.py

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import numpy as np
import pytest

from lab.lab_5_03_agent_based_modelling import (
    Boid,
    BoidsSimulation,
    SchellingAgent,
    SchellingModel,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SCHELLING MODEL BASIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSchellingModelBasic:
    """Basic tests for Schelling model."""
    
    def test_model_creation(self, rng: np.random.Generator) -> None:
        """Model should initialise correctly."""
        model = SchellingModel(grid_size=20, empty_ratio=0.2, threshold=0.3, rng=rng)
        
        assert model.grid_size == 20
        assert model.threshold == 0.3
        assert model.grid.shape == (20, 20)
    
    def test_grid_has_correct_empty_ratio(self, rng: np.random.Generator) -> None:
        """Grid should have approximately correct empty ratio."""
        model = SchellingModel(grid_size=30, empty_ratio=0.2, rng=rng)
        
        n_empty = np.sum(model.grid == 0)
        total = model.grid_size ** 2
        
        actual_ratio = n_empty / total
        assert abs(actual_ratio - 0.2) < 0.01
    
    def test_grid_has_two_agent_types(self, rng: np.random.Generator) -> None:
        """Grid should have exactly two agent types plus empty."""
        model = SchellingModel(grid_size=20, rng=rng)
        
        unique_values = set(model.grid.flatten())
        assert unique_values == {0, 1, 2}
    
    def test_type_ratio_respected(self, rng: np.random.Generator) -> None:
        """Type ratio should be approximately correct."""
        model = SchellingModel(
            grid_size=30,
            empty_ratio=0.2,
            type_ratio=0.6,  # 60% type 1
            rng=rng,
        )
        
        n_type1 = np.sum(model.grid == 1)
        n_type2 = np.sum(model.grid == 2)
        total_agents = n_type1 + n_type2
        
        actual_ratio = n_type1 / total_agents
        assert abs(actual_ratio - 0.6) < 0.05


class TestSchellingModelNeighbours:
    """Tests for neighbour finding."""
    
    def test_centre_cell_has_8_neighbours(self, rng: np.random.Generator) -> None:
        """Centre cell should have up to 8 neighbours."""
        model = SchellingModel(grid_size=10, empty_ratio=0.0, rng=rng)
        
        # All cells occupied, so centre should have 8 neighbours
        neighbours = model.get_neighbours(5, 5)
        assert len(neighbours) == 8
    
    def test_corner_cell_has_3_neighbours(self, rng: np.random.Generator) -> None:
        """Corner cell should have up to 3 neighbours."""
        model = SchellingModel(grid_size=10, empty_ratio=0.0, rng=rng)
        
        # Corner (0, 0) has neighbours at (0,1), (1,0), (1,1)
        neighbours = model.get_neighbours(0, 0)
        assert len(neighbours) == 3
    
    def test_edge_cell_has_5_neighbours(self, rng: np.random.Generator) -> None:
        """Edge cell should have up to 5 neighbours."""
        model = SchellingModel(grid_size=10, empty_ratio=0.0, rng=rng)
        
        # Edge cell (0, 5) has 5 neighbours
        neighbours = model.get_neighbours(0, 5)
        assert len(neighbours) == 5
    
    def test_empty_neighbour_cells_not_counted(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Empty cells should not be counted as neighbours."""
        model = SchellingModel(grid_size=10, empty_ratio=0.5, rng=rng)
        
        # With 50% empty, neighbours list will have fewer than max
        neighbours = model.get_neighbours(5, 5)
        
        # No zeros should be in neighbours (empty cells excluded)
        assert 0 not in neighbours


class TestSchellingModelHappiness:
    """Tests for happiness calculation."""
    
    def test_all_same_type_neighbours_happy(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Agent with all same-type neighbours should be happy."""
        model = SchellingModel(grid_size=10, threshold=0.3, rng=rng)
        
        # Create a homogeneous 3x3 block
        model.grid[4:7, 4:7] = 1
        
        assert model.is_happy(5, 5)
    
    def test_no_same_type_neighbours_unhappy(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Agent with no same-type neighbours should be unhappy."""
        model = SchellingModel(grid_size=10, threshold=0.3, rng=rng)
        
        # Create opposite block around one agent
        model.grid[4:7, 4:7] = 2
        model.grid[5, 5] = 1  # Lone type 1 agent
        
        assert not model.is_happy(5, 5)
    
    def test_threshold_boundary(self, rng: np.random.Generator) -> None:
        """Agent exactly at threshold should be happy."""
        model = SchellingModel(grid_size=10, threshold=0.3, rng=rng)
        
        # Create 3 same-type out of 8 neighbours (37.5% > 30%)
        model.grid[4:7, 4:7] = 2  # Fill with type 2
        model.grid[5, 5] = 1      # Centre is type 1
        model.grid[4, 4] = 1      # Add same-type neighbours
        model.grid[4, 5] = 1
        model.grid[4, 6] = 1      # Now 3/8 = 37.5% same-type
        
        assert model.is_happy(5, 5)
    
    def test_empty_cell_is_happy(self, rng: np.random.Generator) -> None:
        """Empty cells should always be considered happy."""
        model = SchellingModel(grid_size=10, rng=rng)
        
        # Find an empty cell
        empty_positions = np.argwhere(model.grid == 0)
        if len(empty_positions) > 0:
            row, col = empty_positions[0]
            assert model.is_happy(row, col)


class TestSchellingModelDynamics:
    """Tests for simulation dynamics."""
    
    def test_step_returns_metrics(self, rng: np.random.Generator) -> None:
        """Step should return metrics dictionary."""
        model = SchellingModel(grid_size=20, rng=rng)
        
        metrics = model.step()
        
        assert "step" in metrics
        assert "moved" in metrics
        assert "happy_fraction" in metrics
        assert "segregation" in metrics
    
    def test_step_increments_counter(self, rng: np.random.Generator) -> None:
        """Step count should increment."""
        model = SchellingModel(grid_size=20, rng=rng)
        
        assert model.step_count == 0
        model.step()
        assert model.step_count == 1
        model.step()
        assert model.step_count == 2
    
    def test_happy_agents_dont_move(self, rng: np.random.Generator) -> None:
        """All-happy configuration should have no movement."""
        model = SchellingModel(grid_size=20, threshold=0.0, rng=rng)
        
        # With threshold=0, everyone is happy
        metrics = model.step()
        assert metrics["moved"] == 0
    
    def test_segregation_increases(self, rng: np.random.Generator) -> None:
        """Segregation should generally increase over time."""
        model = SchellingModel(
            grid_size=30,
            empty_ratio=0.2,
            threshold=0.3,
            rng=rng,
        )
        
        initial_seg = model.segregation_index()
        model.run(max_steps=50)
        final_seg = model.segregation_index()
        
        # Segregation typically increases
        assert final_seg >= initial_seg
    
    def test_happy_fraction_increases(self, rng: np.random.Generator) -> None:
        """Happy fraction should increase over time."""
        model = SchellingModel(
            grid_size=30,
            empty_ratio=0.2,
            threshold=0.3,
            rng=rng,
        )
        
        initial_happy = model.happy_fraction()
        model.run(max_steps=50)
        final_happy = model.happy_fraction()
        
        # Happy fraction should increase
        assert final_happy >= initial_happy
    
    def test_run_returns_history(self, rng: np.random.Generator) -> None:
        """Run should return list of metrics."""
        model = SchellingModel(grid_size=20, rng=rng)
        
        history = model.run(max_steps=10)
        
        assert len(history) <= 10
        assert all("step" in h for h in history)


class TestSchellingMetrics:
    """Tests for metric calculations."""
    
    def test_segregation_index_range(self, rng: np.random.Generator) -> None:
        """Segregation index should be in [0, 1]."""
        model = SchellingModel(grid_size=20, rng=rng)
        
        seg = model.segregation_index()
        assert 0.0 <= seg <= 1.0
    
    def test_happy_fraction_range(self, rng: np.random.Generator) -> None:
        """Happy fraction should be in [0, 1]."""
        model = SchellingModel(grid_size=20, rng=rng)
        
        happy = model.happy_fraction()
        assert 0.0 <= happy <= 1.0
    
    def test_completely_segregated_has_high_index(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Completely segregated grid should have high index."""
        model = SchellingModel(grid_size=20, empty_ratio=0.0, rng=rng)
        
        # Create perfectly segregated grid
        model.grid[:10, :] = 1
        model.grid[10:, :] = 2
        
        seg = model.segregation_index()
        assert seg > 0.8


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BOIDS MODEL BASIC TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBoidBasic:
    """Basic tests for Boid class."""
    
    def test_boid_creation(self) -> None:
        """Boid should initialise correctly."""
        pos = np.array([100.0, 200.0])
        vel = np.array([1.0, 2.0])
        
        boid = Boid(position=pos, velocity=vel)
        
        assert np.allclose(boid.position, pos)
        assert np.allclose(boid.velocity, vel)
    
    def test_boid_speed(self) -> None:
        """Speed should be velocity magnitude."""
        vel = np.array([3.0, 4.0])
        boid = Boid(position=np.array([0.0, 0.0]), velocity=vel)
        
        assert boid.speed == 5.0
    
    def test_boid_distance(self) -> None:
        """Distance calculation should be correct."""
        boid1 = Boid(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]))
        boid2 = Boid(position=np.array([3.0, 4.0]), velocity=np.array([0.0, 0.0]))
        
        assert boid1.distance_to(boid2) == 5.0
    
    def test_boid_unique_ids(self) -> None:
        """Boids should have unique IDs."""
        boid1 = Boid(position=np.array([0.0, 0.0]), velocity=np.array([0.0, 0.0]))
        boid2 = Boid(position=np.array([1.0, 1.0]), velocity=np.array([0.0, 0.0]))
        
        assert boid1.boid_id != boid2.boid_id


class TestBoidsSimulationBasic:
    """Basic tests for BoidsSimulation."""
    
    def test_simulation_creation(self, rng: np.random.Generator) -> None:
        """Simulation should initialise correctly."""
        sim = BoidsSimulation(
            width=800,
            height=600,
            n_boids=50,
            rng=rng,
        )
        
        assert sim.width == 800
        assert sim.height == 600
        assert len(sim.boids) == 50
    
    def test_boids_start_in_bounds(self, rng: np.random.Generator) -> None:
        """All boids should start within bounds."""
        sim = BoidsSimulation(width=800, height=600, n_boids=100, rng=rng)
        
        for boid in sim.boids:
            assert 0 <= boid.position[0] <= 800
            assert 0 <= boid.position[1] <= 600
    
    def test_boids_have_velocity(self, rng: np.random.Generator) -> None:
        """Boids should have non-zero initial velocity."""
        sim = BoidsSimulation(width=800, height=600, n_boids=50, rng=rng)
        
        # At least some boids should have non-zero velocity
        speeds = [b.speed for b in sim.boids]
        assert np.mean(speeds) > 0


class TestBoidsNeighbours:
    """Tests for neighbour finding in boids."""
    
    def test_nearby_boids_found(self, rng: np.random.Generator) -> None:
        """Nearby boids should be detected."""
        sim = BoidsSimulation(
            width=800,
            height=600,
            n_boids=2,
            visual_range=100,
            rng=rng,
        )
        
        # Place boids close together
        sim.boids[0].position = np.array([400.0, 300.0])
        sim.boids[1].position = np.array([420.0, 300.0])  # 20 units away
        
        neighbours = sim.get_neighbours(sim.boids[0])
        
        assert len(neighbours) == 1
        assert neighbours[0] == sim.boids[1]
    
    def test_far_boids_not_neighbours(self, rng: np.random.Generator) -> None:
        """Far boids should not be neighbours."""
        sim = BoidsSimulation(
            width=800,
            height=600,
            n_boids=2,
            visual_range=50,
            rng=rng,
        )
        
        # Place boids far apart
        sim.boids[0].position = np.array([100.0, 100.0])
        sim.boids[1].position = np.array([700.0, 500.0])
        
        neighbours = sim.get_neighbours(sim.boids[0])
        
        assert len(neighbours) == 0
    
    def test_self_not_neighbour(self, rng: np.random.Generator) -> None:
        """Boid should not be its own neighbour."""
        sim = BoidsSimulation(width=800, height=600, n_boids=5, rng=rng)
        
        for boid in sim.boids:
            neighbours = sim.get_neighbours(boid)
            assert boid not in neighbours


class TestBoidsSteeringRules:
    """Tests for steering rules."""
    
    def test_separation_pushes_away(self, rng: np.random.Generator) -> None:
        """Separation should push boids apart."""
        sim = BoidsSimulation(
            width=800,
            height=600,
            n_boids=2,
            min_distance=50,
            rng=rng,
        )
        
        # Place boids very close
        sim.boids[0].position = np.array([400.0, 300.0])
        sim.boids[1].position = np.array([410.0, 300.0])  # 10 < 50
        
        neighbours = sim.get_neighbours(sim.boids[0])
        sep = sim.separation(sim.boids[0], neighbours)
        
        # Should steer away from neighbour (negative x direction)
        assert sep[0] < 0
    
    def test_alignment_matches_velocity(self, rng: np.random.Generator) -> None:
        """Alignment should steer toward average velocity."""
        sim = BoidsSimulation(width=800, height=600, n_boids=3, rng=rng)
        
        # Set velocities
        sim.boids[0].velocity = np.array([0.0, 0.0])
        sim.boids[1].velocity = np.array([5.0, 0.0])
        sim.boids[2].velocity = np.array([5.0, 0.0])
        
        # Place close together
        for i, boid in enumerate(sim.boids):
            boid.position = np.array([400.0 + i*10, 300.0])
        
        neighbours = sim.get_neighbours(sim.boids[0])
        if len(neighbours) > 0:
            ali = sim.alignment(sim.boids[0], neighbours)
            
            # Should steer toward average (positive x)
            assert ali[0] > 0
    
    def test_cohesion_steers_to_centre(self, rng: np.random.Generator) -> None:
        """Cohesion should steer toward centre of mass."""
        sim = BoidsSimulation(width=800, height=600, n_boids=3, rng=rng)
        
        # Place boid 0 at origin, others to the right
        sim.boids[0].position = np.array([100.0, 300.0])
        sim.boids[1].position = np.array([200.0, 300.0])
        sim.boids[2].position = np.array([200.0, 300.0])
        
        # Ensure they're within visual range
        sim.visual_range = 200
        
        neighbours = sim.get_neighbours(sim.boids[0])
        if len(neighbours) > 0:
            coh = sim.cohesion(sim.boids[0], neighbours)
            
            # Should steer right (toward centre of neighbours)
            assert coh[0] > 0


class TestBoidsSimulationDynamics:
    """Tests for simulation dynamics."""
    
    def test_step_updates_positions(self, rng: np.random.Generator) -> None:
        """Step should update boid positions."""
        sim = BoidsSimulation(width=800, height=600, n_boids=10, rng=rng)
        
        initial_positions = [b.position.copy() for b in sim.boids]
        sim.step()
        final_positions = [b.position.copy() for b in sim.boids]
        
        # At least some positions should change
        changed = sum(
            not np.allclose(i, f)
            for i, f in zip(initial_positions, final_positions)
        )
        assert changed > 0
    
    def test_step_returns_metrics(self, rng: np.random.Generator) -> None:
        """Step should return metrics dictionary."""
        sim = BoidsSimulation(width=800, height=600, n_boids=20, rng=rng)
        
        metrics = sim.step()
        
        assert "step" in metrics
        assert "polarisation" in metrics
        assert "avg_neighbours" in metrics
    
    def test_run_returns_history(self, rng: np.random.Generator) -> None:
        """Run should return list of metrics."""
        sim = BoidsSimulation(width=800, height=600, n_boids=20, rng=rng)
        
        history = sim.run(n_steps=10)
        
        assert len(history) == 10
        assert all("polarisation" in h for h in history)
    
    def test_speed_limited(self, rng: np.random.Generator) -> None:
        """Boid speed should be limited to max_speed."""
        sim = BoidsSimulation(
            width=800,
            height=600,
            n_boids=50,
            max_speed=5.0,
            rng=rng,
        )
        
        sim.run(n_steps=20)
        
        for boid in sim.boids:
            assert boid.speed <= 5.0 * 1.01  # Small tolerance


class TestBoidsMetrics:
    """Tests for boids metrics."""
    
    def test_polarisation_range(self, rng: np.random.Generator) -> None:
        """Polarisation should be in [0, 1]."""
        sim = BoidsSimulation(width=800, height=600, n_boids=50, rng=rng)
        
        sim.run(n_steps=10)
        pol = sim.polarisation()
        
        assert 0.0 <= pol <= 1.0
    
    def test_aligned_boids_high_polarisation(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Perfectly aligned boids should have polarisation = 1."""
        sim = BoidsSimulation(width=800, height=600, n_boids=10, rng=rng)
        
        # Set all velocities the same
        for boid in sim.boids:
            boid.velocity = np.array([1.0, 0.0])
        
        pol = sim.polarisation()
        assert pol > 0.99
    
    def test_average_neighbours_positive(self, rng: np.random.Generator) -> None:
        """Average neighbours should be non-negative."""
        sim = BoidsSimulation(
            width=400,
            height=300,
            n_boids=50,
            visual_range=100,
            rng=rng,
        )
        
        avg = sim.average_neighbours()
        assert avg >= 0
    
    def test_polarisation_increases_with_time(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Polarisation typically increases as boids align."""
        sim = BoidsSimulation(
            width=400,
            height=300,
            n_boids=50,
            visual_range=100,
            alignment_weight=2.0,
            rng=rng,
        )
        
        initial_pol = sim.polarisation()
        sim.run(n_steps=100)
        final_pol = sim.polarisation()
        
        # Polarisation should generally increase
        assert final_pol >= initial_pol * 0.9  # Allow some variance


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_boid_simulation(self, rng: np.random.Generator) -> None:
        """Simulation should work with single boid."""
        sim = BoidsSimulation(width=800, height=600, n_boids=1, rng=rng)
        
        history = sim.run(n_steps=10)
        
        assert len(history) == 10
        assert all(h["avg_neighbours"] == 0 for h in history)
    
    def test_small_grid_schelling(self, rng: np.random.Generator) -> None:
        """Schelling should work on small grid."""
        model = SchellingModel(grid_size=5, rng=rng)
        
        history = model.run(max_steps=10)
        
        assert len(history) <= 10
    
    def test_high_threshold_schelling(self, rng: np.random.Generator) -> None:
        """High threshold should cause lots of movement."""
        model = SchellingModel(
            grid_size=20,
            threshold=0.9,  # Very demanding
            rng=rng,
        )
        
        # Most agents will be unhappy initially
        initial_happy = model.happy_fraction()
        assert initial_happy < 0.5  # Most should be unhappy
    
    def test_zero_threshold_schelling(self, rng: np.random.Generator) -> None:
        """Zero threshold should make everyone happy."""
        model = SchellingModel(
            grid_size=20,
            threshold=0.0,
            rng=rng,
        )
        
        assert model.happy_fraction() == 1.0
        
        metrics = model.step()
        assert metrics["moved"] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRISED TESTS
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("threshold", [0.1, 0.3, 0.5, 0.7])
def test_schelling_varying_threshold(
    threshold: float,
    rng: np.random.Generator,
) -> None:
    """Model should work with varying thresholds."""
    model = SchellingModel(grid_size=20, threshold=threshold, rng=rng)
    
    history = model.run(max_steps=30)
    
    assert len(history) > 0
    # Higher thresholds typically lead to more segregation
    assert 0 <= model.segregation_index() <= 1


@pytest.mark.parametrize("n_boids", [10, 50, 100])
def test_boids_varying_population(
    n_boids: int,
    rng: np.random.Generator,
) -> None:
    """Boids should work with varying populations."""
    sim = BoidsSimulation(width=800, height=600, n_boids=n_boids, rng=rng)
    
    history = sim.run(n_steps=20)
    
    assert len(history) == 20
    assert len(sim.boids) == n_boids


@pytest.mark.parametrize(
    "sep,ali,coh",
    [
        (1.0, 1.0, 1.0),
        (3.0, 1.0, 1.0),
        (1.0, 3.0, 1.0),
        (1.0, 1.0, 3.0),
    ],
)
def test_boids_varying_weights(
    sep: float,
    ali: float,
    coh: float,
    rng: np.random.Generator,
) -> None:
    """Boids should work with varying rule weights."""
    sim = BoidsSimulation(
        width=800,
        height=600,
        n_boids=30,
        separation_weight=sep,
        alignment_weight=ali,
        cohesion_weight=coh,
        rng=rng,
    )
    
    history = sim.run(n_steps=20)
    
    assert len(history) == 20
