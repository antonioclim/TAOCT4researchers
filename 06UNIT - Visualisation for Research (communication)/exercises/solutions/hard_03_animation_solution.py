#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Hard 03 — Animated Visualisation (SOLUTION)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create an animated visualisation showing the evolution of an agent-based model.

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 60 minutes
BLOOM LEVEL: Create

This file contains the complete solution with detailed explanations.

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimulationConfig:
    """Configuration for the simulation.
    
    Attributes:
        n_agents: Number of agents in the simulation
        world_size: Size of the square world (world_size x world_size)
        speed: Base movement speed of agents
        n_frames: Total number of animation frames
        fps: Frames per second for output
        seed: Random seed for reproducibility
    """
    n_agents: int = 100
    world_size: float = 10.0
    speed: float = 0.1
    n_frames: int = 200
    fps: int = 30
    seed: int = 42


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: AGENT SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentSimulation:
    """Simple agent-based simulation with random walk.
    
    This class implements a basic random walk model where agents move
    in a bounded 2D space with periodic boundary conditions.
    
    Attributes:
        config: Simulation configuration
        positions: Agent positions as (n_agents, 2) array
        velocities: Agent velocities as (n_agents, 2) array
        frame: Current frame number
        history: List of position snapshots
    """
    
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
        
        # Initialise velocities with random directions
        angles = np.random.uniform(0, 2 * np.pi, config.n_agents)
        self.velocities = config.speed * np.column_stack([
            np.cos(angles),
            np.sin(angles)
        ])
        
        self.frame = 0
        self.history: list[np.ndarray] = []
        
        logger.debug(f"Initialised simulation with {config.n_agents} agents")
    
    def step(self) -> np.ndarray:
        """Advance simulation by one step.
        
        The simulation step:
        1. Updates positions based on velocities
        2. Applies periodic boundary conditions (wrap around)
        3. Adds random perturbation to velocities
        4. Stores frame in history
        5. Increments frame counter
        
        Returns:
            Updated positions array (n_agents, 2)
        """
        # SOLUTION: Update positions based on velocities
        self.positions += self.velocities
        
        # SOLUTION: Periodic boundary conditions (wrap around)
        self.positions = self.positions % self.config.world_size
        
        # SOLUTION: Add random perturbation to velocities (Brownian motion)
        perturbation = np.random.normal(0, 0.02, self.velocities.shape)
        self.velocities += perturbation
        
        # Normalise velocities to maintain constant speed
        speeds = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)  # Avoid division by zero
        self.velocities = self.config.speed * self.velocities / speeds
        
        # SOLUTION: Store frame in history
        self.history.append(self.positions.copy())
        
        # SOLUTION: Increment frame counter
        self.frame += 1
        
        return self.positions
    
    def get_statistics(self) -> dict:
        """Calculate current simulation statistics.
        
        Returns:
            Dictionary containing:
            - mean_x: Mean x position
            - mean_y: Mean y position
            - spread: Standard deviation of positions (measure of dispersion)
            - mean_speed: Mean velocity magnitude
        """
        # SOLUTION: Calculate statistics
        mean_pos = np.mean(self.positions, axis=0)
        spread = np.std(self.positions)
        mean_speed = np.mean(np.linalg.norm(self.velocities, axis=1))
        
        return {
            'mean_x': mean_pos[0],
            'mean_y': mean_pos[1],
            'spread': spread,
            'mean_speed': mean_speed,
            'frame': self.frame
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: ANIMATED VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

class AnimatedSimulation:
    """Animated visualisation of agent simulation.
    
    Creates a matplotlib animation showing agents moving in real-time
    with frame counter and statistics overlay.
    
    Attributes:
        sim: AgentSimulation instance
        config: Simulation configuration
        fig: Matplotlib figure
        ax: Matplotlib axes
        scatter: Scatter plot artist
        frame_text: Text artist for frame counter
        stats_text: Text artist for statistics
    """
    
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
        
        logger.debug("Initialised animation visualisation")
    
    def _setup_plot(self) -> None:
        """Configure the plot appearance with dark theme."""
        self.ax.set_xlim(0, self.config.world_size)
        self.ax.set_ylim(0, self.config.world_size)
        self.ax.set_aspect('equal')
        
        # Dark theme colours
        self.ax.set_facecolor('#1a1a2e')
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Remove spines for clean look
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        
        # Remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Title
        self.ax.set_title(
            'Agent-Based Model Animation',
            color='#58a6ff',
            fontsize=14,
            fontweight='bold'
        )
    
    def init_animation(self) -> tuple:
        """Initialise animation elements.
        
        Creates:
        - Scatter plot for agent positions
        - Frame counter text
        - Statistics text overlay
        
        Returns:
            Tuple of artists to be animated (scatter, frame_text, stats_text)
        """
        # SOLUTION: Create scatter plot for agents
        self.scatter = self.ax.scatter(
            self.sim.positions[:, 0],
            self.sim.positions[:, 1],
            c=np.linspace(0, 1, self.config.n_agents),
            cmap='plasma',
            s=30,
            alpha=0.8,
            edgecolors='none'
        )
        
        # SOLUTION: Create frame counter text
        self.frame_text = self.ax.text(
            0.02, 0.98,
            'Frame: 0',
            transform=self.ax.transAxes,
            fontsize=12,
            fontfamily='monospace',
            color='#58a6ff',
            verticalalignment='top',
            horizontalalignment='left'
        )
        
        # SOLUTION: Create statistics text
        self.stats_text = self.ax.text(
            0.98, 0.98,
            '',
            transform=self.ax.transAxes,
            fontsize=10,
            fontfamily='monospace',
            color='#8b949e',
            verticalalignment='top',
            horizontalalignment='right'
        )
        
        return (self.scatter, self.frame_text, self.stats_text)
    
    def update_frame(self, frame: int) -> tuple:
        """Update animation for a single frame.
        
        Args:
            frame: Frame number (passed by FuncAnimation)
            
        Returns:
            Tuple of updated artists
        """
        # SOLUTION: Advance simulation by one step
        positions = self.sim.step()
        
        # SOLUTION: Update scatter plot positions
        self.scatter.set_offsets(positions)
        
        # SOLUTION: Update frame counter text
        self.frame_text.set_text(f'Frame: {self.sim.frame}')
        
        # SOLUTION: Update statistics text
        stats = self.sim.get_statistics()
        stats_str = (
            f"Spread: {stats['spread']:.2f}\n"
            f"Mean: ({stats['mean_x']:.1f}, {stats['mean_y']:.1f})"
        )
        self.stats_text.set_text(stats_str)
        
        return (self.scatter, self.frame_text, self.stats_text)
    
    def create_animation(self) -> FuncAnimation:
        """Create the animation object.
        
        Returns:
            FuncAnimation instance ready for display or saving
        """
        # SOLUTION: Create FuncAnimation
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            init_func=self.init_animation,
            frames=self.config.n_frames,
            interval=1000 // self.config.fps,  # Convert FPS to milliseconds
            blit=True,  # Optimisation: only redraw changed elements
            repeat=True
        )
        
        logger.info(f"Created animation with {self.config.n_frames} frames")
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
            dpi: Resolution (dots per inch)
        """
        anim = self.create_animation()
        path = Path(path)
        
        # Select appropriate writer
        if format == 'gif':
            writer = 'pillow'
        elif format == 'mp4':
            writer = 'ffmpeg'
        else:
            raise ValueError(f"Unknown format: {format}. Use 'gif' or 'mp4'.")
        
        logger.info(f"Saving animation to {path} (this may take a moment)...")
        
        anim.save(
            path,
            writer=writer,
            fps=self.config.fps,
            dpi=dpi,
            progress_callback=lambda i, n: logger.debug(f"Frame {i}/{n}")
        )
        
        logger.info(f"Saved animation to: {path}")
    
    def show(self) -> None:
        """Display the animation interactively."""
        anim = self.create_animation()
        plt.tight_layout()
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BOIDS FLOCKING SIMULATION (Advanced)
# ═══════════════════════════════════════════════════════════════════════════════

class BoidsSimulation(AgentSimulation):
    """Boids flocking simulation with separation, alignment and cohesion.
    
    Implements Craig Reynolds' Boids algorithm (1987) with three rules:
    1. Separation: Avoid crowding neighbours
    2. Alignment: Steer towards average heading of neighbours
    3. Cohesion: Steer towards centre of mass of neighbours
    
    Attributes:
        sep_r: Separation radius (avoid agents within this distance)
        ali_r: Alignment radius (align with agents within this distance)
        coh_r: Cohesion radius (move towards centre of agents within)
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        separation_radius: float = 0.5,
        alignment_radius: float = 1.0,
        cohesion_radius: float = 2.0,
        separation_weight: float = 0.05,
        alignment_weight: float = 0.02,
        cohesion_weight: float = 0.01
    ):
        """Initialise boids simulation.
        
        Args:
            config: Simulation configuration
            separation_radius: Distance for separation behaviour
            alignment_radius: Distance for alignment behaviour
            cohesion_radius: Distance for cohesion behaviour
            separation_weight: Strength of separation force
            alignment_weight: Strength of alignment force
            cohesion_weight: Strength of cohesion force
        """
        super().__init__(config)
        self.sep_r = separation_radius
        self.ali_r = alignment_radius
        self.coh_r = cohesion_radius
        self.sep_w = separation_weight
        self.ali_w = alignment_weight
        self.coh_w = cohesion_weight
        
        logger.debug(f"Initialised Boids with radii: sep={sep_r}, ali={ali_r}, coh={coh_r}")
    
    def step(self) -> np.ndarray:
        """Advance boids simulation with flocking rules.
        
        Returns:
            Updated positions array
        """
        new_velocities = self.velocities.copy()
        
        for i in range(self.config.n_agents):
            pos_i = self.positions[i]
            
            # Calculate distances to all other agents
            diffs = self.positions - pos_i
            dists = np.linalg.norm(diffs, axis=1)
            
            # Rule 1: Separation — avoid crowding
            sep_mask = (dists > 0) & (dists < self.sep_r)
            if sep_mask.any():
                sep_force = -diffs[sep_mask].sum(axis=0)
                new_velocities[i] += self.sep_w * sep_force
            
            # Rule 2: Alignment — steer towards average heading
            ali_mask = (dists > 0) & (dists < self.ali_r)
            if ali_mask.any():
                avg_vel = self.velocities[ali_mask].mean(axis=0)
                new_velocities[i] += self.ali_w * (avg_vel - self.velocities[i])
            
            # Rule 3: Cohesion — steer towards centre of mass
            coh_mask = (dists > 0) & (dists < self.coh_r)
            if coh_mask.any():
                centre = self.positions[coh_mask].mean(axis=0)
                new_velocities[i] += self.coh_w * (centre - pos_i)
        
        # Normalise velocities to maintain constant speed
        speeds = np.linalg.norm(new_velocities, axis=1, keepdims=True)
        speeds = np.maximum(speeds, 1e-6)
        self.velocities = self.config.speed * new_velocities / speeds
        
        # Update positions
        self.positions += self.velocities
        
        # Periodic boundary conditions
        self.positions = self.positions % self.config.world_size
        
        # Record history
        self.history.append(self.positions.copy())
        self.frame += 1
        
        return self.positions


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation() -> bool:
    """Test the complete implementation.
    
    Returns:
        True if all tests pass
    """
    logger.info("Running implementation tests...")
    all_passed = True
    
    # Test 1: Simulation step
    config = SimulationConfig(n_agents=50, n_frames=100, fps=20, seed=42)
    sim = AgentSimulation(config)
    initial_pos = sim.positions.copy()
    sim.step()
    
    if np.array_equal(sim.positions, initial_pos):
        logger.error("✗ Positions did not change after step")
        all_passed = False
    else:
        logger.info("✓ Simulation step working")
    
    # Test 2: Frame counter
    if sim.frame != 1:
        logger.error(f"✗ Frame counter incorrect: expected 1, got {sim.frame}")
        all_passed = False
    else:
        logger.info("✓ Frame counter working")
    
    # Test 3: Statistics
    stats = sim.get_statistics()
    required_keys = {'mean_x', 'mean_y', 'spread', 'mean_speed'}
    if not required_keys.issubset(stats.keys()):
        logger.error(f"✗ Statistics missing keys: {required_keys - stats.keys()}")
        all_passed = False
    else:
        logger.info("✓ Statistics working")
    
    # Test 4: Animation initialisation
    anim_vis = AnimatedSimulation(sim)
    init_elements = anim_vis.init_animation()
    
    if len(init_elements) < 3:
        logger.error(f"✗ init_animation returned {len(init_elements)} elements, expected 3")
        all_passed = False
    else:
        logger.info("✓ Animation initialised")
    
    # Test 5: Frame update
    update_elements = anim_vis.update_frame(0)
    
    if len(update_elements) < 3:
        logger.error(f"✗ update_frame returned {len(update_elements)} elements, expected 3")
        all_passed = False
    else:
        logger.info("✓ Frame update working")
    
    # Test 6: Animation creation
    anim = anim_vis.create_animation()
    
    if anim is None:
        logger.error("✗ create_animation returned None")
        all_passed = False
    else:
        logger.info("✓ Animation created")
    
    # Test 7: Boids simulation
    boids_config = SimulationConfig(n_agents=30, n_frames=50, seed=42)
    boids = BoidsSimulation(boids_config)
    
    for _ in range(10):
        boids.step()
    
    if boids.frame != 10:
        logger.error(f"✗ Boids frame counter incorrect: expected 10, got {boids.frame}")
        all_passed = False
    else:
        logger.info("✓ Boids simulation working")
    
    plt.close('all')
    
    if all_passed:
        logger.info("═" * 50)
        logger.info("✓ All tests passed!")
        logger.info("═" * 50)
    else:
        logger.error("═" * 50)
        logger.error("✗ Some tests failed")
        logger.error("═" * 50)
    
    return all_passed


def demonstrate_animation() -> None:
    """Demonstrate the animation with Boids flocking."""
    logger.info("Creating Boids flocking demonstration...")
    
    config = SimulationConfig(
        n_agents=100,
        world_size=10.0,
        speed=0.15,
        n_frames=300,
        fps=30,
        seed=42
    )
    
    sim = BoidsSimulation(
        config,
        separation_radius=0.5,
        alignment_radius=1.5,
        cohesion_radius=2.5
    )
    
    anim_vis = AnimatedSimulation(sim, figsize=(10, 10))
    
    logger.info("Displaying animation (close window to exit)...")
    anim_vis.show()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Animated agent-based model visualisation"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demonstration"
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="PATH",
        help="Save animation to file (gif or mp4)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.test:
        success = test_implementation()
        exit(0 if success else 1)
    
    if args.save:
        logger.info(f"Saving animation to {args.save}...")
        config = SimulationConfig(n_agents=100, n_frames=200, fps=30)
        sim = BoidsSimulation(config)
        anim_vis = AnimatedSimulation(sim)
        
        fmt = 'mp4' if args.save.endswith('.mp4') else 'gif'
        anim_vis.save(args.save, format=fmt)
    elif args.demo:
        demonstrate_animation()
    else:
        # Default: run tests then demo
        if test_implementation():
            demonstrate_animation()


if __name__ == "__main__":
    main()
