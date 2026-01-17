#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Easy 02 — Scatter Plot with Regression
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create a scatter plot with a linear regression line overlaid.

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 15 minutes
BLOOM LEVEL: Apply

TASK
────
Complete the function `create_scatter_regression()` that:
1. Creates a scatter plot of x vs y data
2. Fits a linear regression line
3. Displays the R² value and equation on the plot
4. Uses appropriate styling

HINTS
─────
- Use np.polyfit(x, y, 1) for linear regression
- R² = 1 - (SS_res / SS_tot)
- Use ax.text() to add the equation annotation

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_sample_data(
    n_points: int = 50,
    noise_level: float = 0.3,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data with linear trend and noise.
    
    Args:
        n_points: Number of data points
        noise_level: Standard deviation of noise
        seed: Random seed
        
    Returns:
        Tuple of (x, y) arrays
    """
    np.random.seed(seed)
    x = np.linspace(0, 10, n_points)
    y = 2.5 * x + 3 + np.random.normal(0, noise_level * 5, n_points)
    return x, y


def calculate_r_squared(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate the coefficient of determination (R²).
    
    Args:
        y_actual: Actual y values
        y_predicted: Predicted y values from regression
        
    Returns:
        R² value between 0 and 1
    """
    # TODO: Implement R² calculation
    # R² = 1 - (SS_res / SS_tot)
    # SS_res = sum((y_actual - y_predicted)²)
    # SS_tot = sum((y_actual - mean(y_actual))²)
    
    # YOUR CODE HERE
    pass


def create_scatter_regression(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path | None = None
) -> plt.Figure:
    """Create scatter plot with regression line.
    
    Args:
        x: X-axis values
        y: Y-axis values
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # TODO: Complete the implementation
    # 1. Create scatter plot of data points
    # 2. Fit linear regression using np.polyfit(x, y, 1)
    # 3. Calculate predicted y values
    # 4. Plot regression line
    # 5. Calculate R² using calculate_r_squared()
    # 6. Add text annotation with equation and R²
    # 7. Add axis labels and title
    
    # YOUR CODE HERE
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Test R² calculation
    y_actual = np.array([1, 2, 3, 4, 5])
    y_predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    r2 = calculate_r_squared(y_actual, y_predicted)
    
    if r2 is None:
        print("❌ calculate_r_squared() returned None")
        return False
    
    if not 0.9 < r2 <= 1.0:
        print(f"❌ R² = {r2:.4f} seems incorrect for near-perfect fit")
        return False
    
    print(f"✓ R² calculation working ({r2:.4f})")
    
    # Test scatter plot
    x, y = generate_sample_data()
    fig = create_scatter_regression(x, y)
    
    if fig is None:
        print("❌ create_scatter_regression() returned None")
        return False
    
    ax = fig.axes[0]
    if len(ax.collections) < 1:
        print("❌ No scatter points found")
        return False
    
    if len(ax.lines) < 1:
        print("❌ No regression line found")
        return False
    
    print("✓ Scatter plot created successfully")
    print("✓ All tests passed!")
    
    plt.show()
    return True


if __name__ == "__main__":
    test_implementation()
