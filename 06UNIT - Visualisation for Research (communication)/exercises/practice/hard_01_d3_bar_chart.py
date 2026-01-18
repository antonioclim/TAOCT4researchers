#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Practice Exercise: Hard 01 — D3.js Bar Chart Generator
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE
─────────
Generate a standalone HTML file containing an animated D3.js bar chart.

DIFFICULTY: ⭐⭐⭐ Hard
ESTIMATED TIME: 45 minutes
BLOOM LEVEL: Create

TASK
────
Complete the function `generate_d3_bar_chart()` that:
1. Takes data as Python dictionaries
2. Generates a complete HTML file with embedded D3.js
3. Implements the enter-update-exit pattern
4. Includes animated transitions
5. Adds hover tooltips

HINTS
─────
- Use f-strings or Template for HTML generation
- D3 v7 can be loaded from CDN
- json.dumps() converts Python data to JavaScript-compatible format
- Include inline CSS for styling

LICENCE
───────
© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
═══════════════════════════════════════════════════════════════════════════════
"""

import json
from pathlib import Path


D3_BAR_CHART_TEMPLATE = '''<!DOCTYPE html>
<html lang="en-GB">
<head>
    <meta charset="utf-8">
    <title>D3.js Bar Chart — Week 6 Exercise</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #e0e0e0;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px;
        }}
        h1 {{
            color: #58a6ff;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: #16213e;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        }}
        .bar {{
            transition: opacity 0.2s;
        }}
        .bar:hover {{
            opacity: 0.8;
        }}
        .tooltip {{
            position: absolute;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px 15px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            font-size: 14px;
        }}
        .axis text {{
            fill: #8b949e;
            font-size: 12px;
        }}
        .axis path, .axis line {{
            stroke: #30363d;
        }}
        .controls {{
            margin-top: 20px;
        }}
        button {{
            background: #238636;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            margin: 0 5px;
        }}
        button:hover {{
            background: #2ea043;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="chart-container">
        <svg id="chart"></svg>
    </div>
    <div class="controls">
        <button onclick="updateData()">Randomise Data</button>
        <button onclick="sortBars()">Sort Bars</button>
    </div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Data from Python
        const initialData = {data_json};
        
        // Chart dimensions
        const margin = {{top: 20, right: 20, bottom: 40, left: 50}};
        const width = {width} - margin.left - margin.right;
        const height = {height} - margin.top - margin.bottom;
        
        // Colour palette (colourblind-friendly)
        const colours = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'];
        
        // Create SVG
        const svg = d3.select('#chart')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
        
        // Scales
        const xScale = d3.scaleBand()
            .range([0, width])
            .padding(0.2);
        
        const yScale = d3.scaleLinear()
            .range([height, 0]);
        
        // Axes groups
        const xAxisGroup = svg.append('g')
            .attr('class', 'axis x-axis')
            .attr('transform', `translate(0,${{height}})`);
        
        const yAxisGroup = svg.append('g')
            .attr('class', 'axis y-axis');
        
        // Tooltip reference
        const tooltip = d3.select('#tooltip');
        
        // Current data
        let currentData = [...initialData];
        
        // TODO: Complete the update function
        // This function should implement the enter-update-exit pattern
        function update(data) {{
            // 1. Update scales
            xScale.domain(data.map(d => d.category));
            yScale.domain([0, d3.max(data, d => d.value) * 1.1]);
            
            // 2. Update axes with transition
            xAxisGroup.transition().duration(500).call(d3.axisBottom(xScale));
            yAxisGroup.transition().duration(500).call(d3.axisLeft(yScale));
            
            // 3. Data join
            const bars = svg.selectAll('.bar')
                .data(data, d => d.category);
            
            // 4. EXIT: Remove old bars
            // YOUR CODE HERE: Animate bars shrinking before removal
            
            // 5. ENTER: Add new bars
            // YOUR CODE HERE: New bars should start with height 0 and animate up
            
            // 6. UPDATE: Transition existing bars
            // YOUR CODE HERE: Animate position and height changes
            
            // 7. Add tooltip interactions
            // YOUR CODE HERE: Show tooltip on mouseover, hide on mouseout
        }}
        
        // Randomise data
        function updateData() {{
            currentData = currentData.map(d => ({{
                ...d,
                value: Math.random() * 100
            }}));
            update(currentData);
        }}
        
        // Sort bars by value
        function sortBars() {{
            currentData.sort((a, b) => b.value - a.value);
            update(currentData);
        }}
        
        // Initial render
        update(currentData);
    </script>
</body>
</html>
'''


def generate_sample_data() -> list[dict]:
    """Generate sample data for the bar chart.
    
    Returns:
        List of dictionaries with 'category' and 'value' keys
    """
    import random
    random.seed(42)
    
    categories = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
    return [
        {'category': cat, 'value': random.uniform(20, 100)}
        for cat in categories
    ]


def generate_d3_bar_chart(
    data: list[dict],
    title: str = 'Interactive Bar Chart',
    width: int = 600,
    height: int = 400,
    output_path: Path | None = None
) -> str:
    """Generate HTML file with D3.js bar chart.
    
    Args:
        data: List of {'category': str, 'value': float} dictionaries
        title: Chart title
        width: Chart width in pixels
        height: Chart height in pixels
        output_path: Optional path to save HTML file
        
    Returns:
        HTML string
    """
    # TODO: Complete the HTML generation
    # 1. Convert data to JSON string
    # 2. Format the template with data, title, width, height
    # 3. Save to file if output_path provided
    # 4. Return the HTML string
    
    # YOUR CODE HERE
    html = None  # Replace with your implementation
    
    if output_path and html:
        output_path.write_text(html)
        print(f"Saved to: {output_path}")
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# SOLUTION TEMPLATE (Implement the D3 update function)
# ═══════════════════════════════════════════════════════════════════════════════

D3_UPDATE_SOLUTION = '''
// Complete update function with enter-update-exit pattern
function update(data) {
    // Update scales
    xScale.domain(data.map(d => d.category));
    yScale.domain([0, d3.max(data, d => d.value) * 1.1]);
    
    // Update axes
    xAxisGroup.transition().duration(500).call(d3.axisBottom(xScale));
    yAxisGroup.transition().duration(500).call(d3.axisLeft(yScale));
    
    // Data join
    const bars = svg.selectAll('.bar')
        .data(data, d => d.category);
    
    // EXIT
    bars.exit()
        .transition()
        .duration(300)
        .attr('y', height)
        .attr('height', 0)
        .remove();
    
    // ENTER
    const barsEnter = bars.enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => xScale(d.category))
        .attr('width', xScale.bandwidth())
        .attr('y', height)
        .attr('height', 0)
        .attr('fill', (d, i) => colours[i % colours.length]);
    
    // UPDATE + ENTER
    bars.merge(barsEnter)
        .transition()
        .duration(500)
        .attr('x', d => xScale(d.category))
        .attr('width', xScale.bandwidth())
        .attr('y', d => yScale(d.value))
        .attr('height', d => height - yScale(d.value));
    
    // Tooltips
    svg.selectAll('.bar')
        .on('mouseover', function(event, d) {
            tooltip
                .style('opacity', 1)
                .html(`<strong>${d.category}</strong><br>Value: ${d.value.toFixed(1)}`)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 20) + 'px');
        })
        .on('mouseout', function() {
            tooltip.style('opacity', 0);
        });
}
'''


# ═══════════════════════════════════════════════════════════════════════════════
# TEST YOUR IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def test_implementation():
    """Test your implementation."""
    # Generate test data
    data = generate_sample_data()
    
    if not data or len(data) != 6:
        print("❌ Data generation failed")
        return False
    
    print("✓ Test data generated")
    
    # Test HTML generation
    html = generate_d3_bar_chart(data, title='Test Chart')
    
    if html is None:
        print("❌ generate_d3_bar_chart() returned None")
        return False
    
    if '<!DOCTYPE html>' not in html:
        print("❌ HTML doesn't contain DOCTYPE")
        return False
    
    if 'd3.v7.min.js' not in html:
        print("❌ HTML doesn't include D3.js")
        return False
    
    print("✓ HTML generated successfully")
    
    # Save to file for manual inspection
    output_path = Path('test_d3_chart.html')
    generate_d3_bar_chart(data, output_path=output_path)
    
    print(f"✓ Saved to {output_path}")
    print("  Open in browser to test interactivity")
    print("✓ All tests passed!")
    
    return True


if __name__ == "__main__":
    test_implementation()
