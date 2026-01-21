#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Easy 01 — Basic Line Plot
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create a simple line plot showing the convergence of a Monte Carlo estimator.

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 15 minutes
BLOOM LEVEL: Apply

TASK
────
Complete the function `create_convergence_plot()` that:
1. Generates Monte Carlo estimates of π using random sampling
2. Creates a line plot showing how the estimate converges to π
3. Adds a horizontal reference line at y = π
4. Labels axes appropriately

HINTS
─────
- Use np.random.uniform() to generate random points in [0, 1] × [0, 1]
- A point is inside the quarter circle if x² + y² ≤ 1
- π ≈ 4 × (points inside circle) / (total points)
- Use ax.axhline() for the reference line

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def estimate_pi_monte_carlo(n_samples: int, seed: int = 42) -> list[float]:
    """Generate running Monte Carlo estimates of π.
    
    Args:
        n_samples: Total number of random samples
        seed: Random seed for reproducibility
        
    Returns:
        List of cumulative π estimates after each sample
    """
    np.random.seed(seed)
    estimates = []
    inside_count = 0
    
    # TODO: Implement Monte Carlo estimation
    # For each sample:
    #   1. Generate random x, y in [0, 1]
    #   2. Check if point is inside quarter circle (x² + y² ≤ 1)
    #   3. Update inside_count if inside
    #   4. Calculate current estimate: π ≈ 4 × inside_count / current_sample_number
    #   5. Append estimate to list
    
    # YOUR CODE HERE
    pass
    
    return estimates


def create_convergence_plot(
    estimates: list[float],
    output_path: Path | None = None
) -> plt.Figure:
    """Create a line plot showing Monte Carlo convergence.
    
    Args:
        estimates: List of π estimates at each iteration
        output_path: Optional path to save the figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # TODO: Complete the plot
    # 1. Plot the estimates as a line
    # 2. Add a horizontal line at y = π (use np.pi)
    # 3. Set appropriate axis labels
    # 4. Add a title
    # 5. Add a legend
    
    # YOUR CODE HERE
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Test Monte Carlo estimation
    estimates = estimate_pi_monte_carlo(1000)
    
    if estimates is None or len(estimates) == 0:
        print("❌ estimate_pi_monte_carlo() returned empty or None")
        return False
    
    if len(estimates) != 1000:
        print(f"❌ Expected 1000 estimates, got {len(estimates)}")
        return False
    
    final_estimate = estimates[-1]
    if abs(final_estimate - np.pi) > 0.2:
        print(f"❌ Final estimate {final_estimate:.4f} too far from π")
        return False
    
    print(f"✓ Monte Carlo estimation working (final estimate: {final_estimate:.4f})")
    
    # Test plotting
    fig = create_convergence_plot(estimates)
    if fig is None:
        print("❌ create_convergence_plot() returned None")
        return False
    
    ax = fig.axes[0]
    if len(ax.lines) < 2:
        print("❌ Plot should have at least 2 lines (data + reference)")
        return False
    
    print("✓ Plot created successfully")
    print("✓ All tests passed!")
    
    plt.show()
    return True


if __name__ == "__main__":
    test_implementation()
