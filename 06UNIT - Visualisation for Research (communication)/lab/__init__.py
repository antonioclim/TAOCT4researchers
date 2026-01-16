"""
Week 6 Laboratory Package: Visualisation for Research

This package provides tools and utilities for creating publication-ready
static figures and interactive dashboards for research data visualisation.

Modules:
    lab_6_01_static_plots: Publication-quality static figure creation
    lab_6_02_interactive_viz: Interactive dashboard development

Example:
    >>> from lab import lab_6_01_static_plots as viz
    >>> style = viz.PlotStyle.for_journal('nature')
    >>> style.apply()
    >>> fig, ax = viz.create_figure()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> viz.save_publication_figure(fig, 'output/figure_1')

© 2025 Antonio Clim. All rights reserved.
See README.md for full licence terms.
"""

from .lab_6_01_static_plots import (
    PlotStyle,
    PALETTES,
    JOURNAL_STYLES,
    create_figure,
    save_publication_figure,
    add_annotation,
    add_significance_bar,
    plot_with_error_band,
    scatter_with_regression,
    create_heatmap,
    generate_d3_line_chart_html,
    export_to_d3_json,
)

from .lab_6_02_interactive_viz import (
    DataPoint,
    Dataset,
    MetricCard,
    FilterControl,
    DashboardConfig,
    generate_dashboard_html,
    generate_bar_chart_svg,
    generate_line_chart_svg,
    generate_pie_chart_svg,
    STREAMLIT_TEMPLATE,
)

__all__ = [
    # Static plots
    'PlotStyle',
    'PALETTES',
    'JOURNAL_STYLES',
    'create_figure',
    'save_publication_figure',
    'add_annotation',
    'add_significance_bar',
    'plot_with_error_band',
    'scatter_with_regression',
    'create_heatmap',
    'generate_d3_line_chart_html',
    'export_to_d3_json',
    # Interactive dashboards
    'DataPoint',
    'Dataset',
    'MetricCard',
    'FilterControl',
    'DashboardConfig',
    'generate_dashboard_html',
    'generate_bar_chart_svg',
    'generate_line_chart_svg',
    'generate_pie_chart_svg',
    'STREAMLIT_TEMPLATE',
]

__version__ = '1.0.0'
__author__ = 'Antonio Clim'
