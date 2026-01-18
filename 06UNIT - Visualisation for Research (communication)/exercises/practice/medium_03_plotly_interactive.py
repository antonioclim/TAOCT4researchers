#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Medium 03 — Interactive Plotly Visualisation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Create an interactive scatter plot with hover information using Plotly.

DIFFICULTY: ⭐⭐ Medium
ESTIMATED TIME: 30 minutes
BLOOM LEVEL: Apply/Create

TASK
────
Complete the function `create_interactive_scatter()` that:
1. Creates a Plotly scatter plot with colour encoding by category
2. Adds informative hover tooltips
3. Includes dropdown or slider for filtering
4. Exports to standalone HTML

HINTS
─────
- Use plotly.express for quick plots or plotly.graph_objects for control
- hover_data parameter adds fields to tooltips
- fig.update_layout() for styling
- fig.write_html() for export

DEPENDENCIES
────────────
pip install plotly

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
import numpy as np

# Check for plotly availability
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not installed. Run: pip install plotly")


def generate_simulation_results(
    n_samples: int = 200,
    seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate simulated experimental results.
    
    Args:
        n_samples: Number of data points
        seed: Random seed
        
    Returns:
        Dictionary with arrays for each variable
    """
    np.random.seed(seed)
    
    methods = np.random.choice(['Monte Carlo', 'RK4', 'Euler', 'Adaptive'], n_samples)
    
    # Generate data with method-dependent patterns
    base_accuracy = {'Monte Carlo': 0.85, 'RK4': 0.95, 'Euler': 0.75, 'Adaptive': 0.92}
    base_time = {'Monte Carlo': 100, 'RK4': 50, 'Euler': 20, 'Adaptive': 80}
    
    accuracy = np.array([
        base_accuracy[m] + np.random.normal(0, 0.05) for m in methods
    ])
    
    computation_time = np.array([
        base_time[m] * (1 + np.random.exponential(0.3)) for m in methods
    ])
    
    sample_size = np.random.randint(100, 10000, n_samples)
    
    return {
        'method': methods,
        'accuracy': np.clip(accuracy, 0, 1),
        'computation_time': computation_time,
        'sample_size': sample_size,
        'experiment_id': np.arange(n_samples),
    }


def create_interactive_scatter(
    data: dict[str, np.ndarray],
    output_path: Path | None = None
) -> 'go.Figure | None':
    """Create interactive Plotly scatter plot.
    
    Args:
        data: Dictionary with 'method', 'accuracy', 'computation_time', etc.
        output_path: Optional path to save HTML
        
    Returns:
        Plotly Figure object (or None if plotly not available)
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None
    
    # TODO: Complete the interactive visualisation
    # 1. Create scatter plot with px.scatter()
    #    - x = computation_time
    #    - y = accuracy
    #    - color = method
    #    - size = sample_size (optional)
    #    - hover_data = include experiment_id and sample_size
    # 2. Update layout with title, axis labels, template
    # 3. Add dropdown menu to filter by method (advanced)
    # 4. Save to HTML if output_path provided
    
    # YOUR CODE HERE
    fig = None  # Replace with your implementation
    
    if output_path and fig is not None:
        fig.write_html(str(output_path))
    
    return fig


def create_dashboard_layout(
    data: dict[str, np.ndarray],
    output_path: Path | None = None
) -> 'go.Figure | None':
    """Create a multi-panel dashboard with linked views.
    
    Args:
        data: Dictionary with simulation results
        output_path: Optional path to save HTML
        
    Returns:
        Plotly Figure object
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    # TODO: Create a 2x2 dashboard layout
    # 1. Top-left: Scatter plot (accuracy vs time)
    # 2. Top-right: Box plot (accuracy by method)
    # 3. Bottom-left: Histogram (computation time distribution)
    # 4. Bottom-right: Bar chart (mean accuracy by method)
    
    # Use make_subplots(rows=2, cols=2, ...)
    
    # YOUR CODE HERE
    fig = None  # Replace with your implementation
    
    if output_path and fig is not None:
        fig.write_html(str(output_path))
    
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    if not PLOTLY_AVAILABLE:
        print("❌ Cannot test: plotly not installed")
        print("   Install with: pip install plotly")
        return False
    
    # Generate test data
    data = generate_simulation_results(n_samples=100)
    
    if len(data['method']) != 100:
        print("❌ Data generation failed")
        return False
    
    print("✓ Test data generated")
    
    # Test interactive scatter
    fig = create_interactive_scatter(data)
    
    if fig is None:
        print("❌ create_interactive_scatter() returned None")
        return False
    
    print("✓ Interactive scatter created")
    
    # Test dashboard (optional)
    dashboard = create_dashboard_layout(data)
    
    if dashboard is not None:
        print("✓ Dashboard layout created")
    else:
        print("⚠ Dashboard not implemented (optional)")
    
    print("✓ All tests passed!")
    
    # Show the figure
    fig.show()
    
    return True


if __name__ == "__main__":
    test_implementation()
