#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
THE ART OF COMPUTATIONAL THINKING FOR RESEARCHERS
Week 6, Lab: Visualization Toolkit
═══════════════════════════════════════════════════════════════════════════════

CONTEXT
───────
Vizualizarea eficientă a datelor este o competență critică în cercetare.
Acest laborator oferă un toolkit pentru crearea de vizualizări:
- Publication-ready (format academic)
- Interactive (explorare)
- Reproducible (scriptable)

PRINCIPII TUFTE
───────────────
1. Data-ink ratio: maximizați informația per pixel
2. Chartjunk: eliminați elementele decorative inutile
3. Lie factor: reprezentați proporțiile corect
4. Small multiples: comparații prin repetare

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Any
import json

# Optional imports
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA I: CONFIGURARE STILURI
# ═══════════════════════════════════════════════════════════════════════════════

# Palete de culori (colorblind-friendly)
PALETTES = {
    'default': ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860'],
    'colorblind': ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9'],
    'grayscale': ['#000000', '#404040', '#808080', '#BFBFBF', '#D9D9D9', '#F2F2F2'],
    'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4'],
    'viridis': ['#440154', '#31688E', '#35B779', '#FDE725', '#21918C', '#5DC863'],
}

# Configurații pentru jurnale academice
JOURNAL_STYLES = {
    'nature': {
        'figure.figsize': (3.5, 2.5),  # Single column
        'font.size': 8,
        'font.family': 'sans-serif',
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'savefig.dpi': 300,
    },
    'science': {
        'figure.figsize': (3.5, 2.625),
        'font.size': 7,
        'font.family': 'sans-serif',
        'axes.linewidth': 0.5,
        'lines.linewidth': 0.75,
        'savefig.dpi': 300,
    },
    'ieee': {
        'figure.figsize': (3.5, 2.5),
        'font.size': 8,
        'font.family': 'serif',
        'axes.linewidth': 0.5,
        'lines.linewidth': 1.0,
        'savefig.dpi': 600,
    },
    'thesis': {
        'figure.figsize': (6, 4),
        'font.size': 11,
        'font.family': 'serif',
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'savefig.dpi': 300,
    },
    'presentation': {
        'figure.figsize': (10, 6),
        'font.size': 14,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.0,
        'lines.linewidth': 2.0,
        'savefig.dpi': 150,
    },
}


@dataclass
class PlotStyle:
    """Configurație pentru stilul ploturilor."""
    
    # Dimensiuni și DPI
    figsize: tuple[float, float] = (6, 4)
    dpi: int = 300
    
    # Fonturi
    font_family: str = 'serif'
    font_size: float = 10
    title_size: float = 12
    label_size: float = 10
    tick_size: float = 9
    legend_size: float = 9
    
    # Linii și markere
    line_width: float = 1.5
    marker_size: float = 5
    axes_linewidth: float = 0.8
    
    # Culori
    palette: str = 'colorblind'
    background: str = 'white'
    grid_color: str = '#CCCCCC'
    grid_alpha: float = 0.5
    
    # Grid
    show_grid: bool = True
    grid_style: str = '--'
    
    # Spine visibility
    spine_top: bool = False
    spine_right: bool = False
    
    def apply(self) -> None:
        """Aplică stilul la matplotlib."""
        if not HAS_MATPLOTLIB:
            return
        
        mpl.rcParams.update({
            'figure.figsize': self.figsize,
            'figure.dpi': self.dpi,
            'savefig.dpi': self.dpi,
            
            'font.family': self.font_family,
            'font.size': self.font_size,
            
            'axes.titlesize': self.title_size,
            'axes.labelsize': self.label_size,
            'axes.linewidth': self.axes_linewidth,
            
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            
            'legend.fontsize': self.legend_size,
            
            'lines.linewidth': self.line_width,
            'lines.markersize': self.marker_size,
            
            'axes.facecolor': self.background,
            'figure.facecolor': self.background,
            
            'axes.grid': self.show_grid,
            'grid.color': self.grid_color,
            'grid.alpha': self.grid_alpha,
            'grid.linestyle': self.grid_style,
            
            'axes.spines.top': self.spine_top,
            'axes.spines.right': self.spine_right,
            
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
        })
    
    def get_colors(self, n: int | None = None) -> list[str]:
        """Returnează culorile din paletă."""
        colors = PALETTES.get(self.palette, PALETTES['default'])
        if n is not None:
            # Cycle colors if needed
            return [colors[i % len(colors)] for i in range(n)]
        return colors
    
    @classmethod
    def for_journal(cls, journal: str) -> 'PlotStyle':
        """Creează stil pentru un jurnal specific."""
        if journal not in JOURNAL_STYLES:
            raise ValueError(f"Unknown journal: {journal}. Available: {list(JOURNAL_STYLES.keys())}")
        
        config = JOURNAL_STYLES[journal]
        return cls(
            figsize=config['figure.figsize'],
            dpi=config['savefig.dpi'],
            font_family=config['font.family'],
            font_size=config['font.size'],
            line_width=config['lines.linewidth'],
            axes_linewidth=config['axes.linewidth'],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA II: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_figure(
    nrows: int = 1,
    ncols: int = 1,
    style: PlotStyle | None = None,
    **kwargs
) -> tuple['Figure', 'Axes | list[Axes]']:
    """
    Creează o figură cu stil aplicat.
    
    Returns:
        (fig, ax) sau (fig, axes) pentru subplots
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for create_figure")
    
    if style is not None:
        style.apply()
    
    fig, axes = plt.subplots(nrows, ncols, **kwargs)
    return fig, axes


def save_publication_figure(
    fig: 'Figure',
    filename: str,
    formats: list[str] = ['pdf', 'png'],
    dpi: int = 300
) -> list[str]:
    """
    Salvează figura în multiple formate pentru publicare.
    
    Args:
        fig: Figura matplotlib
        filename: Nume bază (fără extensie)
        formats: Lista de formate ['pdf', 'png', 'svg', 'eps']
        dpi: Rezoluție pentru formate raster
        
    Returns:
        Lista căilor fișierelor salvate
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required")
    
    saved = []
    for fmt in formats:
        path = f"{filename}.{fmt}"
        fig.savefig(
            path,
            format=fmt,
            dpi=dpi if fmt in ['png', 'jpg', 'tiff'] else None,
            bbox_inches='tight',
            pad_inches=0.05,
            facecolor=fig.get_facecolor(),
            edgecolor='none'
        )
        saved.append(path)
    
    return saved


def add_annotation(
    ax: 'Axes',
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float] | None = None,
    fontsize: float = 9,
    **kwargs
) -> None:
    """Adaugă o anotare cu stil consistent."""
    defaults = {
        'fontsize': fontsize,
        'ha': 'center',
        'va': 'bottom',
        'arrowprops': {'arrowstyle': '->', 'color': 'gray', 'lw': 0.5}
    }
    defaults.update(kwargs)
    
    if xytext is None:
        ax.annotate(text, xy, **{k: v for k, v in defaults.items() if k != 'arrowprops'})
    else:
        ax.annotate(text, xy, xytext, **defaults)


def add_significance_bars(
    ax: 'Axes',
    x1: float,
    x2: float,
    y: float,
    significance: str = '*',
    height: float = 0.02
) -> None:
    """
    Adaugă bare de semnificație statistică între două grupuri.
    
    Args:
        ax: Axes matplotlib
        x1, x2: Poziții x ale celor două grupuri
        y: Înălțimea bării
        significance: Simbol ('*', '**', '***', 'ns')
        height: Înălțimea relativă a barei
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    bar_height = y_range * height
    
    ax.plot([x1, x1, x2, x2], [y, y + bar_height, y + bar_height, y], 
            lw=0.8, c='black')
    ax.text((x1 + x2) / 2, y + bar_height, significance,
            ha='center', va='bottom', fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA III: PLOT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

def line_plot_with_error(
    ax: 'Axes',
    x: list[float],
    y: list[float],
    yerr: list[float] | None = None,
    label: str | None = None,
    color: str | None = None,
    fill_alpha: float = 0.2,
    **kwargs
) -> None:
    """
    Line plot cu bandă de eroare (util pentru experimente repetate).
    """
    line, = ax.plot(x, y, label=label, color=color, **kwargs)
    
    if yerr is not None:
        y_arr = [yi for yi in y]
        yerr_arr = [ei for ei in yerr]
        upper = [y_arr[i] + yerr_arr[i] for i in range(len(y_arr))]
        lower = [y_arr[i] - yerr_arr[i] for i in range(len(y_arr))]
        ax.fill_between(x, lower, upper, alpha=fill_alpha, color=line.get_color())


def bar_plot_grouped(
    ax: 'Axes',
    categories: list[str],
    groups: dict[str, list[float]],
    errors: dict[str, list[float]] | None = None,
    colors: list[str] | None = None,
    bar_width: float = 0.8
) -> None:
    """
    Bar plot grupat pentru comparații între condiții.
    
    Args:
        categories: Numele categoriilor pe axa x
        groups: {group_name: [values]} pentru fiecare grup
        errors: {group_name: [error_values]} opțional
        colors: Lista de culori pentru grupuri
    """
    n_groups = len(groups)
    n_categories = len(categories)
    
    width = bar_width / n_groups
    x = list(range(n_categories))
    
    if colors is None:
        colors = PALETTES['colorblind'][:n_groups]
    
    for i, (group_name, values) in enumerate(groups.items()):
        offset = (i - n_groups/2 + 0.5) * width
        positions = [xi + offset for xi in x]
        
        err = errors.get(group_name) if errors else None
        ax.bar(positions, values, width, label=group_name, color=colors[i],
               yerr=err, capsize=3, error_kw={'linewidth': 0.8})
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()


def scatter_with_regression(
    ax: 'Axes',
    x: list[float],
    y: list[float],
    color: str = '#4C72B0',
    show_equation: bool = True,
    show_r2: bool = True
) -> tuple[float, float, float]:
    """
    Scatter plot cu linia de regresie liniară.
    
    Returns:
        (slope, intercept, r_squared)
    """
    n = len(x)
    
    # Calculează regresie liniară
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    ss_xy = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    ss_xx = sum((x[i] - mean_x) ** 2 for i in range(n))
    ss_yy = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    slope = ss_xy / ss_xx if ss_xx != 0 else 0
    intercept = mean_y - slope * mean_x
    
    # R²
    ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_yy) if ss_yy != 0 else 0
    
    # Plot
    ax.scatter(x, y, c=color, alpha=0.6, s=30, edgecolors='white', linewidth=0.5)
    
    x_line = [min(x), max(x)]
    y_line = [slope * xi + intercept for xi in x_line]
    ax.plot(x_line, y_line, c=color, linestyle='--', linewidth=1.5)
    
    # Annotation
    if show_equation or show_r2:
        text_parts = []
        if show_equation:
            sign = '+' if intercept >= 0 else '-'
            text_parts.append(f'y = {slope:.3f}x {sign} {abs(intercept):.3f}')
        if show_r2:
            text_parts.append(f'R² = {r_squared:.3f}')
        
        ax.text(0.05, 0.95, '\n'.join(text_parts),
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return slope, intercept, r_squared


def heatmap(
    ax: 'Axes',
    data: list[list[float]],
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    cmap: str = 'viridis',
    annotate: bool = True,
    fmt: str = '.2f',
    cbar_label: str | None = None
) -> None:
    """
    Heatmap cu anotări.
    """
    if HAS_NUMPY:
        data_arr = np.array(data)
    else:
        data_arr = data
    
    n_rows = len(data)
    n_cols = len(data[0]) if data else 0
    
    im = ax.imshow(data_arr, cmap=cmap, aspect='auto')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va='bottom')
    
    # Labels
    if row_labels:
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels)
    if col_labels:
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=45, ha='right')
    
    # Annotations
    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                value = data[i][j]
                color = 'white' if value > (max(max(row) for row in data) + min(min(row) for row in data)) / 2 else 'black'
                ax.text(j, i, f'{value:{fmt}}', ha='center', va='center', color=color, fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA IV: D3.JS EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_d3_json(
    data: dict[str, Any],
    filename: str
) -> None:
    """
    Exportă date în format JSON pentru vizualizare D3.js.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def generate_d3_line_chart_html(
    data: list[dict[str, float]],
    x_key: str,
    y_keys: list[str],
    title: str = "Line Chart",
    width: int = 800,
    height: int = 400
) -> str:
    """
    Generează cod HTML pentru un line chart D3.js.
    
    Args:
        data: Lista de dict-uri [{x: val, y1: val, y2: val}, ...]
        x_key: Cheia pentru axa x
        y_keys: Lista de chei pentru linii
        title: Titlul graficului
        
    Returns:
        String HTML complet
    """
    colors = PALETTES['colorblind'][:len(y_keys)]
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: sans-serif; }}
        .line {{ fill: none; stroke-width: 2; }}
        .axis-label {{ font-size: 12px; }}
        .title {{ font-size: 16px; font-weight: bold; }}
        .legend {{ font-size: 11px; }}
    </style>
</head>
<body>
    <div id="chart"></div>
    <script>
        const data = {json.dumps(data)};
        const margin = {{top: 40, right: 100, bottom: 50, left: 60}};
        const width = {width} - margin.left - margin.right;
        const height = {height} - margin.top - margin.bottom;
        
        const svg = d3.select("#chart")
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${{margin.left}},${{margin.top}})`);
        
        // Scales
        const x = d3.scaleLinear()
            .domain(d3.extent(data, d => d["{x_key}"]))
            .range([0, width]);
        
        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => Math.max({", ".join(f'd["{k}"]' for k in y_keys)}))])
            .range([height, 0]);
        
        // Axes
        svg.append("g")
            .attr("transform", `translate(0,${{height}})`)
            .call(d3.axisBottom(x));
        
        svg.append("g")
            .call(d3.axisLeft(y));
        
        // Title
        svg.append("text")
            .attr("class", "title")
            .attr("x", width / 2)
            .attr("y", -15)
            .attr("text-anchor", "middle")
            .text("{title}");
        
        // Lines
        const colors = {json.dumps(colors)};
        const yKeys = {json.dumps(y_keys)};
        
        yKeys.forEach((key, i) => {{
            const line = d3.line()
                .x(d => x(d["{x_key}"]))
                .y(d => y(d[key]));
            
            svg.append("path")
                .datum(data)
                .attr("class", "line")
                .attr("stroke", colors[i])
                .attr("d", line);
            
            // Legend
            svg.append("text")
                .attr("class", "legend")
                .attr("x", width + 10)
                .attr("y", 20 + i * 20)
                .attr("fill", colors[i])
                .text(key);
        }});
    </script>
</body>
</html>'''
    
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# PARTEA V: DEMONSTRAȚII
# ═══════════════════════════════════════════════════════════════════════════════

def demo_styles() -> None:
    """Demonstrație: diferite stiluri de publicare."""
    print("=" * 60)
    print("DEMO: Publication Styles")
    print("=" * 60)
    print()
    
    for journal in ['nature', 'science', 'ieee', 'thesis', 'presentation']:
        style = PlotStyle.for_journal(journal)
        print(f"{journal.upper():15} - figsize: {style.figsize}, "
              f"font: {style.font_size}pt {style.font_family}, "
              f"dpi: {style.dpi}")
    
    print()


def demo_color_palettes() -> None:
    """Demonstrație: palete de culori disponibile."""
    print("=" * 60)
    print("DEMO: Color Palettes")
    print("=" * 60)
    print()
    
    for name, colors in PALETTES.items():
        color_str = ' '.join(f'[{c}]' for c in colors[:4])
        print(f"{name:15} {color_str} ...")
    
    print()
    print("Recomandat pentru accesibilitate: 'colorblind'")
    print()


def demo_d3_export() -> None:
    """Demonstrație: export pentru D3.js."""
    print("=" * 60)
    print("DEMO: D3.js Export")
    print("=" * 60)
    print()
    
    # Date exemplu
    data = [
        {'time': i, 'model_a': math.sin(i/5) + 1, 'model_b': math.cos(i/5) + 1}
        for i in range(50)
    ]
    
    html = generate_d3_line_chart_html(
        data, 
        x_key='time', 
        y_keys=['model_a', 'model_b'],
        title='Model Comparison'
    )
    
    print("HTML generat pentru D3.js line chart:")
    print(f"  - Lungime: {len(html)} caractere")
    print(f"  - Date: {len(data)} puncte")
    print()
    print("Pentru a vizualiza, salvați HTML-ul și deschideți în browser.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  WEEK 6 LAB: VISUALIZATION TOOLKIT")
    print("═" * 60 + "\n")
    
    demo_styles()
    demo_color_palettes()
    demo_d3_export()
    
    print("=" * 60)
    print("Următorii pași:")
    print("  1. Instalați matplotlib: pip install matplotlib")
    print("  2. Aplicați PlotStyle.for_journal('nature').apply()")
    print("  3. Creați figuri cu create_figure()")
    print("  4. Salvați cu save_publication_figure()")
    print("=" * 60)
