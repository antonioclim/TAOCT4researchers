#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Medium 02 — Small Multiples (Facet Grid)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create a small multiples display comparing trends across categories.

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 30 minutes
BLOOM LEVEL: Apply/Analyse

TASK
────
Complete the function `create_small_multiples()` that:
1. Creates a grid of subplots (one per category)
2. Maintains consistent axes across all panels
3. Adds shared axis labels
4. Uses Tufte's principles for clean design

HINTS
─────
- Use plt.subplots(nrows, ncols, sharex=True, sharey=True)
- fig.supxlabel() and fig.supylabel() for shared labels
- Keep styling minimal and consistent

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_timeseries_data(
    n_categories: int = 6,
    n_timepoints: int = 50,
    seed: int = 42
) -> dict[str, dict[str, np.ndarray]]:
    """Generate time series data for multiple categories.
    
    Args:
        n_categories: Number of different categories
        n_timepoints: Number of time points
        seed: Random seed
        
    Returns:
        Dictionary mapping category names to {'time': array, 'value': array}
    """
    np.random.seed(seed)
    
    categories = ['Region A', 'Region B', 'Region C', 
                  'Region D', 'Region E', 'Region F'][:n_categories]
    
    data = {}
    time = np.arange(n_timepoints)
    
    for i, cat in enumerate(categories):
        # Generate trend with different patterns
        trend = 10 + i * 5 + 0.5 * time
        seasonal = 5 * np.sin(2 * np.pi * time / 12 + i)
        noise = np.random.normal(0, 2, n_timepoints)
        
        data[cat] = {
            'time': time,
            'value': trend + seasonal + noise
        }
    
    return data


def create_small_multiples(
    data: dict[str, dict[str, np.ndarray]],
    ncols: int = 3,
    output_path: Path | None = None
) -> plt.Figure:
    """Create small multiples grid of time series plots.
    
    Args:
        data: Dictionary of category -> {'time': array, 'value': array}
        ncols: Number of columns in the grid
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    categories = list(data.keys())
    n_categories = len(categories)
    nrows = (n_categories + ncols - 1) // ncols  # Ceiling division
    
    # TODO: Complete the small multiples display
    # 1. Create figure with subplots grid (shared x and y axes)
    # 2. Loop through categories and plot each in its panel
    # 3. Add category name as panel title
    # 4. Remove redundant tick labels (only left column has y labels, bottom row has x labels)
    # 5. Add shared axis labels using fig.supxlabel() and fig.supylabel()
    # 6. Remove top and right spines from each subplot
    # 7. Add overall figure title
    
    # YOUR CODE HERE
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), 
                             sharex=True, sharey=True)
    
    # ... implement plotting ...
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    
    return fig


def calculate_grid_layout(n_items: int, ncols: int) -> tuple[int, int]:
    """Calculate grid dimensions for given number of items.
    
    Args:
        n_items: Number of items to arrange
        ncols: Desired number of columns
        
    Returns:
        Tuple of (nrows, ncols)
    """
    # TODO: Implement grid calculation
    # nrows should be ceiling of n_items / ncols
    
    # YOUR CODE HERE
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Test grid layout calculation
    nrows, ncols = calculate_grid_layout(6, 3) or (0, 0)
    
    if nrows != 2 or ncols != 3:
        print(f"❌ Grid for 6 items, 3 cols should be (2, 3), got ({nrows}, {ncols})")
        if nrows == 0:
            print("   calculate_grid_layout() returned None")
        return False
    
    print("✓ Grid layout calculation correct")
    
    # Generate test data
    data = generate_timeseries_data(n_categories=6)
    
    if len(data) != 6:
        print("❌ Data generation failed")
        return False
    
    print("✓ Test data generated")
    
    # Test small multiples creation
    fig = create_small_multiples(data, ncols=3)
    
    if fig is None:
        print("❌ create_small_multiples() returned None")
        return False
    
    # Check number of subplots
    if len(fig.axes) < 6:
        print(f"❌ Expected at least 6 subplots, got {len(fig.axes)}")
        return False
    
    print("✓ Small multiples created successfully")
    print("✓ All tests passed!")
    
    plt.show()
    return True


if __name__ == "__main__":
    test_implementation()
