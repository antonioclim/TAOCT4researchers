#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Hard 03 — Animated Visualisation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create an animated visualisation showing the evolution of an agent-based model.

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 60 minutes
BLOOM LEVEL: Create

TASK
────
Complete the `AnimatedSimulation` class that:
1. Simulates a simple agent-based model (e.g., random walk or flocking)
2. Creates an animated matplotlib visualisation
3. Saves the animation as GIF or MP4
4. Includes frame counter and simulation statistics

HINTS
─────
- Use matplotlib.animation.FuncAnimation
- Set blit=True for better performance
- Use fig.savefig() with writer='pillow' for GIF
- Keep agent positions in numpy arrays for speed

DEPENDENCIES
────────────
pip install pillow  # For GIF export

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable


@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    n_agents: int = 100
    world_size: float = 10.0
    speed: float = 0.1
    n_frames: int = 200
    fps: int = 30
    seed: int = 42


class AgentSimulation:
    """Simple agent-based simulation with random walk."""
    
    def __init__(self, config: SimulationConfig):
        """Initialise the simulation.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        np.random.seed(config.seed)
        
        # Initialise agent positions randomly
        self.positions = np.random.uniform(
            0, config.world_size,
            size=(config.n_agents, 2)
        )
        
        # Initialise velocities
        angles = np.random.uniform(0, 2 * np.pi, config.n_agents)
        self.velocities = config.speed * np.column_stack([
            np.cos(angles),
            np.sin(angles)
        ])
        
        self.frame = 0
        self.history = []
    
    def step(self) -> np.ndarray:
        """Advance simulation by one step.
        
        Returns:
            Updated positions array
        """
        # TODO: Implement the simulation step
        # 1. Update positions based on velocities
        # 2. Handle boundary conditions (wrap around or bounce)
        # 3. Optionally: Add random perturbation to velocities
        # 4. Store frame in history
        # 5. Increment frame counter
        
        # YOUR CODE HERE
        
        return self.positions
    
    def get_statistics(self) -> dict:
        """Calculate current simulation statistics.
        
        Returns:
            Dictionary with statistics
        """
        # TODO: Calculate statistics
        # - Mean position
        # - Position variance (spread)
        # - Mean velocity magnitude
        
        # YOUR CODE HERE
        return {}


class AnimatedSimulation:
    """Animated visualisation of agent simulation."""
    
    def __init__(
        self,
        simulation: AgentSimulation,
        figsize: tuple[float, float] = (8, 8)
    ):
        """Initialise the animation.
        
        Args:
            simulation: AgentSimulation instance
            figsize: Figure size in inches
        """
        self.sim = simulation
        self.config = simulation.config
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self._setup_plot()
        
        # Animation elements (to be updated)
        self.scatter = None
        self.frame_text = None
        self.stats_text = None
    
    def _setup_plot(self) -> None:
        """Configure the plot appearance."""
        self.ax.set_xlim(0, self.config.world_size)
        self.ax.set_ylim(0, self.config.world_size)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1a1a2e')
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Remove spines
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.ax.set_title('Agent-Based Model Animation', color='#58a6ff', fontsize=14)
    
    def init_animation(self) -> tuple:
        """Initialise animation elements.
        
        Returns:
            Tuple of artists to be animated
        """
        # TODO: Create initial plot elements
        # 1. Create scatter plot for agents
        # 2. Create text elements for frame counter and statistics
        # 3. Return tuple of all elements that will be animated
        
        # YOUR CODE HERE
        
        return ()  # Replace with actual elements
    
    def update_frame(self, frame: int) -> tuple:
        """Update animation for a single frame.
        
        Args:
            frame: Frame number
            
        Returns:
            Tuple of updated artists
        """
        # TODO: Update animation elements
        # 1. Advance simulation by one step
        # 2. Update scatter plot positions
        # 3. Update frame counter text
        # 4. Update statistics text
        # 5. Return tuple of updated elements
        
        # YOUR CODE HERE
        
        return ()  # Replace with actual elements
    
    def create_animation(self) -> FuncAnimation:
        """Create the animation object.
        
        Returns:
            FuncAnimation instance
        """
        # TODO: Create FuncAnimation
        # Use self.init_animation as init_func
        # Use self.update_frame as func
        # Set frames, interval, blit
        
        # YOUR CODE HERE
        anim = None  # Replace with FuncAnimation
        
        return anim
    
    def save(
        self,
        path: Path | str,
        format: str = 'gif',
        dpi: int = 100
    ) -> None:
        """Save animation to file.
        
        Args:
            path: Output file path
            format: 'gif' or 'mp4'
            dpi: Resolution
        """
        anim = self.create_animation()
        
        if anim is None:
            print("Animation not created")
            return
        
        path = Path(path)
        
        if format == 'gif':
            writer = 'pillow'
        elif format == 'mp4':
            writer = 'ffmpeg'
        else:
            raise ValueError(f"Unknown format: {format}")
        
        anim.save(
            path,
            writer=writer,
            fps=self.config.fps,
            dpi=dpi
        )
        
        print(f"Saved animation to: {path}")
    
    def show(self) -> None:
        """Display the animation."""
        anim = self.create_animation()
        if anim is not None:
            plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE: BOIDS FLOCKING (Advanced)
# ═══════════════════════════════════════════════════════════════════════════════

class BoidsSimulation(AgentSimulation):
    """Boids flocking simulation with separation, alignment, and cohesion."""
    
    def __init__(
        self,
        config: SimulationConfig,
        separation_radius: float = 0.5,
        alignment_radius: float = 1.0,
        cohesion_radius: float = 2.0
    ):
        super().__init__(config)
        self.sep_r = separation_radius
        self.ali_r = alignment_radius
        self.coh_r = cohesion_radius
    
    def step(self) -> np.ndarray:
        """Advance boids simulation with flocking rules."""
        # This is provided as a reference implementation
        new_velocities = self.velocities.copy()
        
        for i in range(self.config.n_agents):
            pos_i = self.positions[i]
            
            # Calculate distances to all other agents
            diffs = self.positions - pos_i
            dists = np.linalg.norm(diffs, axis=1)
            
            # Separation: avoid crowding
            sep_mask = (dists > 0) & (dists < self.sep_r)
            if sep_mask.any():
                sep_force = -diffs[sep_mask].sum(axis=0)
                new_velocities[i] += 0.05 * sep_force
            
            # Alignment: steer towards average heading
            ali_mask = (dists > 0) & (dists < self.ali_r)
            if ali_mask.any():
                avg_vel = self.velocities[ali_mask].mean(axis=0)
                new_velocities[i] += 0.02 * (avg_vel - self.velocities[i])
            
            # Cohesion: steer towards centre of mass
            coh_mask = (dists > 0) & (dists < self.coh_r)
            if coh_mask.any():
                centre = self.positions[coh_mask].mean(axis=0)
                new_velocities[i] += 0.01 * (centre - pos_i)
        
        # Normalise velocities
        speeds = np.linalg.norm(new_velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        self.velocities = self.config.speed * new_velocities / speeds
        
        # Update positions
        self.positions += self.velocities
        
        # Wrap around boundaries
        self.positions = self.positions % self.config.world_size
        
        self.frame += 1
        return self.positions


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Create simulation
    config = SimulationConfig(n_agents=50, n_frames=100, fps=20)
    sim = AgentSimulation(config)
    
    # Test simulation step
    initial_pos = sim.positions.copy()
    sim.step()
    
    if np.array_equal(sim.positions, initial_pos):
        print("⚠ Positions did not change after step (implement step())")
    else:
        print("✓ Simulation step working")
    
    # Test animation
    anim_vis = AnimatedSimulation(sim)
    
    init_elements = anim_vis.init_animation()
    if not init_elements:
        print("⚠ init_animation() returned empty tuple")
    else:
        print("✓ Animation initialised")
    
    update_elements = anim_vis.update_frame(0)
    if not update_elements:
        print("⚠ update_frame() returned empty tuple")
    else:
        print("✓ Frame update working")
    
    # Create animation object
    anim = anim_vis.create_animation()
    
    if anim is None:
        print("⚠ create_animation() returned None")
    else:
        print("✓ Animation created")
    
    print("\nRunning demonstration with Boids...")
    
    # Demonstrate with Boids (reference implementation)
    boids_config = SimulationConfig(n_agents=100, n_frames=200, fps=30)
    boids_sim = BoidsSimulation(boids_config)
    boids_anim = AnimatedSimulation(boids_sim)
    
    # Show animation (this will block until window is closed)
    print("Close the animation window to complete the test.")
    boids_anim.show()
    
    print("✓ All tests completed!")
    return True


if __name__ == "__main__":
    test_implementation()
