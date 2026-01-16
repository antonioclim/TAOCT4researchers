#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Easy 03 — Bar Chart with Error Bars
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create a bar chart showing mean values with standard error bars.

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 15 minutes
BLOOM LEVEL: Apply

TASK
────
Complete the function `create_bar_chart()` that:
1. Creates a bar chart of mean values for different categories
2. Adds error bars showing standard error of the mean
3. Uses a colourblind-friendly palette
4. Follows Tufte's principles (remove unnecessary elements)

HINTS
─────
- Standard error = std / sqrt(n)
- Use ax.bar() with yerr parameter
- Remove top and right spines for cleaner appearance

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# Colourblind-friendly palette (Wong, 2011)
COLOURBLIND_PALETTE = [
    '#0072B2',  # Blue
    '#E69F00',  # Orange
    '#009E73',  # Green
    '#CC79A7',  # Pink
    '#F0E442',  # Yellow
    '#56B4E9',  # Light blue
    '#D55E00',  # Red-orange
]


def generate_experiment_data(seed: int = 42) -> dict[str, np.ndarray]:
    """Generate simulated experimental data for multiple conditions.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary mapping condition names to measurement arrays
    """
    np.random.seed(seed)
    return {
        'Control': np.random.normal(50, 8, 30),
        'Treatment A': np.random.normal(65, 10, 30),
        'Treatment B': np.random.normal(72, 12, 30),
        'Treatment C': np.random.normal(58, 9, 30),
    }


def calculate_statistics(data: dict[str, np.ndarray]) -> tuple[list, list, list]:
    """Calculate means and standard errors for each condition.
    
    Args:
        data: Dictionary of condition names to measurement arrays
        
    Returns:
        Tuple of (categories, means, standard_errors)
    """
    # TODO: Calculate statistics for each condition
    # Mean = np.mean(values)
    # Standard Error = np.std(values) / np.sqrt(len(values))
    
    categories = []
    means = []
    standard_errors = []
    
    # YOUR CODE HERE
    
    return categories, means, standard_errors


def create_bar_chart(
    categories: list[str],
    means: list[float],
    errors: list[float],
    output_path: Path | None = None
) -> plt.Figure:
    """Create bar chart with error bars.
    
    Args:
        categories: List of category names
        means: List of mean values
        errors: List of standard error values
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # TODO: Complete the bar chart
    # 1. Create bars using ax.bar() with yerr for error bars
    # 2. Use COLOURBLIND_PALETTE for bar colours
    # 3. Add axis labels and title
    # 4. Remove top and right spines
    # 5. Add value labels on top of each bar
    
    # YOUR CODE HERE
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Generate test data
    data = generate_experiment_data()
    
    # Test statistics calculation
    categories, means, errors = calculate_statistics(data)
    
    if not categories or not means or not errors:
        print("❌ calculate_statistics() returned empty lists")
        return False
    
    if len(categories) != 4:
        print(f"❌ Expected 4 categories, got {len(categories)}")
        return False
    
    if not all(isinstance(m, (int, float)) for m in means):
        print("❌ Means should be numeric values")
        return False
    
    print(f"✓ Statistics calculated for {len(categories)} conditions")
    
    # Test bar chart creation
    fig = create_bar_chart(categories, means, errors)
    
    if fig is None:
        print("❌ create_bar_chart() returned None")
        return False
    
    ax = fig.axes[0]
    if len(ax.patches) < 4:
        print("❌ Bar chart should have at least 4 bars")
        return False
    
    print("✓ Bar chart created successfully")
    print("✓ All tests passed!")
    
    plt.show()
    return True


if __name__ == "__main__":
    test_implementation()
