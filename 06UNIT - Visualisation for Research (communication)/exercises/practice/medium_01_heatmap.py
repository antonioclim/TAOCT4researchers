#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Medium 01 — Heatmap with Annotations
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create a correlation heatmap with cell annotations and proper colour mapping.

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 25 minutes
BLOOM LEVEL: Apply/Analyse

TASK
────
Complete the function `create_correlation_heatmap()` that:
1. Calculates correlation matrix from data
2. Creates a heatmap with diverging colour scale
3. Annotates each cell with correlation value
4. Masks the upper triangle (optional but professional)
5. Includes a colour bar with proper label

HINTS
─────
- Use np.corrcoef() or pandas .corr() for correlation
- Use ax.imshow() or seaborn.heatmap()
- Diverging colormaps: 'RdBu_r', 'coolwarm'
- Use np.triu_indices() for upper triangle mask

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def generate_correlated_data(
    n_samples: int = 100,
    n_features: int = 6,
    seed: int = 42
) -> tuple[np.ndarray, list[str]]:
    """Generate sample data with some correlated features.
    
    Args:
        n_samples: Number of observations
        n_features: Number of variables
        seed: Random seed
        
    Returns:
        Tuple of (data array, feature names)
    """
    np.random.seed(seed)
    
    # Create base random data
    base = np.random.randn(n_samples, 3)
    
    # Create correlated features
    data = np.column_stack([
        base[:, 0],                                    # Feature A
        base[:, 0] * 0.8 + np.random.randn(n_samples) * 0.3,  # B correlates with A
        base[:, 1],                                    # Feature C
        base[:, 1] * -0.6 + np.random.randn(n_samples) * 0.4, # D anti-correlates with C
        base[:, 2],                                    # Feature E
        np.random.randn(n_samples),                    # F is independent
    ])
    
    feature_names = ['Var A', 'Var B', 'Var C', 'Var D', 'Var E', 'Var F']
    
    return data, feature_names


def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """Calculate Pearson correlation matrix.
    
    Args:
        data: 2D array of shape (n_samples, n_features)
        
    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    # TODO: Calculate correlation matrix
    # Hint: np.corrcoef() expects features in rows, so transpose may be needed
    
    # YOUR CODE HERE
    pass


def create_correlation_heatmap(
    corr_matrix: np.ndarray,
    labels: list[str],
    mask_upper: bool = True,
    output_path: Path | None = None
) -> plt.Figure:
    """Create annotated correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix
        labels: Feature labels
        mask_upper: Whether to mask upper triangle
        output_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    n = len(labels)
    
    # TODO: Complete the heatmap
    # 1. Create mask for upper triangle if mask_upper is True
    # 2. Use ax.imshow() with diverging colormap (vmin=-1, vmax=1)
    # 3. Add colour bar
    # 4. Set tick labels
    # 5. Add text annotations for each cell
    # 6. Add title
    
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
    data, labels = generate_correlated_data()
    
    # Test correlation calculation
    corr = calculate_correlation_matrix(data)
    
    if corr is None:
        print("❌ calculate_correlation_matrix() returned None")
        return False
    
    if corr.shape != (6, 6):
        print(f"❌ Expected (6, 6) matrix, got {corr.shape}")
        return False
    
    # Check diagonal is 1
    if not np.allclose(np.diag(corr), 1.0):
        print("❌ Diagonal should be 1.0 (self-correlation)")
        return False
    
    # Check symmetry
    if not np.allclose(corr, corr.T):
        print("❌ Correlation matrix should be symmetric")
        return False
    
    print("✓ Correlation matrix calculated correctly")
    
    # Test heatmap creation
    fig = create_correlation_heatmap(corr, labels)
    
    if fig is None:
        print("❌ create_correlation_heatmap() returned None")
        return False
    
    print("✓ Heatmap created successfully")
    print("✓ All tests passed!")
    
    plt.show()
    return True


if __name__ == "__main__":
    test_implementation()
