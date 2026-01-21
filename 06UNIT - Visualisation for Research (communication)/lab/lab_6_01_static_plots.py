#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06UNIT - Visualisation for Research
Lab 06_01: Static Visualisation Toolkit (test-facing API)

This module provides a compact, test-facing API for generating publication-ready static figures.
The course kit also contains a longer pedagogical implementation in ``lab_06_01_static_plots.py``;
however, the unit tests bundled with the kit import the non-zero-padded module name.

The implementation below is intentionally self-contained and focuses on correctness and reproducible
outputs (consistent file creation, deterministic regression calculations and standards-compliant JSON).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Palettes and journal styles
# ─────────────────────────────────────────────────────────────────────────────

PALETTES: dict[str, list[str]] = {
    "colorblind": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#F0E442",
        "#56B4E9",
        "#E69F00",
        "#000000",
    ],
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    "grayscale": ["#000000", "#444444", "#888888", "#CCCCCC"],
    "nature": ["#386641", "#6A994E", "#A7C957", "#BC4749", "#F2E8CF"],
    "viridis": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
}

JOURNAL_STYLES: dict[str, dict[str, Any]] = {
    "nature": {
        "figure.figsize": (3.35, 2.5),
        "font.size": 7,
        "font.family": "sans-serif",
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.dpi": 300,
    },
    "science": {
        "figure.figsize": (3.35, 2.5),
        "font.size": 7,
        "font.family": "sans-serif",
        "axes.linewidth": 0.5,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "figure.dpi": 300,
    },
    "ieee": {
        "figure.figsize": (3.5, 2.5),
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.linewidth": 0.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "figure.dpi": 300,
    },
    "plos": {
        "figure.figsize": (6.0, 4.0),
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.linewidth": 0.8,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
    },
}


@dataclass(frozen=True)
class PlotStyle:
    """Encapsulates a visual style for reproducible matplotlib figures."""

    palette: str = "colorblind"
    font_size: int = 10
    figure_width: float = 6.0
    figure_height: float = 4.0
    dpi: int = 150

    def get_colours(self, n: int | None) -> list[str]:
        colours = PALETTES.get(self.palette, PALETTES["default"])
        if n is None:
            return list(colours)
        if n <= 0:
            return []
        return [colours[i % len(colours)] for i in range(n)]

    def apply(self) -> None:
        plt.rcParams["font.size"] = self.font_size
        plt.rcParams["figure.dpi"] = self.dpi

    @classmethod
    def for_journal(cls, journal: str) -> "PlotStyle":
        if journal not in JOURNAL_STYLES:
            raise ValueError(f"Unknown journal style: {journal}")
        st = JOURNAL_STYLES[journal]
        w, h = st.get("figure.figsize", (cls.figure_width, cls.figure_height))
        return cls(
            palette="colorblind",
            font_size=int(st.get("font.size", cls.font_size)),
            figure_width=float(w),
            figure_height=float(h),
            dpi=int(st.get("figure.dpi", cls.dpi)),
        )


def create_figure(
    *,
    width: float | None = None,
    height: float | None = None,
    dpi: int | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a single-axes figure with explicit dimensions."""

    w = 6.0 if width is None else float(width)
    h = 4.0 if height is None else float(height)
    d = 150 if dpi is None else int(dpi)
    fig, ax = plt.subplots(figsize=(w, h), dpi=d)
    return fig, ax


def save_publication_figure(
    fig: plt.Figure,
    filename: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf", "svg"),
    dpi: int | None = None,
) -> dict[str, Path]:
    """Save a figure in one or more formats and return the written paths."""

    base = Path(filename)
    if base.suffix:
        base = base.with_suffix("")
    out: dict[str, Path] = {}
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        out[fmt] = path
    return out


def add_annotation(ax: plt.Axes, text: str, *, x: float, y: float, **kwargs: Any) -> Any:
    """Add an annotation and return the created artist."""

    return ax.annotate(text, (x, y), **kwargs)


def _p_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_significance_bar(
    ax: plt.Axes,
    *,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.05,
    linewidth: float = 1.0,
) -> None:
    """Draw a simple significance bar with star label."""

    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="black", lw=linewidth)
    ax.text((x1 + x2) / 2.0, y + height, _p_to_stars(p_value), ha="center", va="bottom")


def plot_with_error_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    errors: np.ndarray,
    *,
    label: str | None = None,
    colour: str | None = None,
    alpha: float = 0.2,
) -> Any:
    line = ax.plot(x, y, label=label, color=colour)[0]
    ax.fill_between(x, y - errors, y + errors, color=line.get_color(), alpha=alpha)
    return line


def scatter_with_regression(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    colour: str | None = None,
) -> tuple[Any, Any, float, float, float]:
    """Scatter plot with ordinary least squares regression line."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")

    slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
    y_pred = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - y_pred) ** 2))
    ss_tot = float(np.sum((y_arr - float(np.mean(y_arr))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    scatter = ax.scatter(x_arr, y_arr, color=colour)
    line = ax.plot([float(np.min(x_arr)), float(np.max(x_arr))],
                   [slope * float(np.min(x_arr)) + intercept, slope * float(np.max(x_arr)) + intercept],
                   color=colour or "black")[0]
    return scatter, line, float(slope), float(intercept), float(r2)


def create_heatmap(
    data: pd.DataFrame | np.ndarray,
    *,
    ax: plt.Axes | None = None,
    annotate: bool = False,
    cmap: str = "viridis",
) -> Any:
    if ax is None:
        _, ax = create_figure()
    arr = data.values if isinstance(data, pd.DataFrame) else np.asarray(data)
    img = ax.imshow(arr, cmap=cmap, aspect="auto")
    if annotate:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.2g}", ha="center", va="center", fontsize=8)
    return img


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def export_to_d3_json(data: pd.DataFrame | list[dict[str, Any]], filename: str | Path) -> Path:
    """Export tabular data to a JSON file suitable for D3."""

    records: list[dict[str, Any]]
    if isinstance(data, pd.DataFrame):
        records = [
            {k: _coerce_jsonable(v) for k, v in row.items()} for row in data.to_dict(orient="records")
        ]
    else:
        records = [{k: _coerce_jsonable(v) for k, v in row.items()} for row in data]

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return path


def generate_d3_line_chart_html(
    data: pd.DataFrame,
    *,
    x_key: str,
    y_keys: list[str],
    title: str = "Line chart",
    output_path: str | Path | None = None,
) -> str:
    """Generate a minimal self-contained D3 line chart page."""

    payload = data[[x_key, *y_keys]].to_dict(orient="records")
    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://d3js.org/d3.v7.min.js\"></script>
</head>
<body style=\"margin:0;background:#111;color:#eee;font-family:system-ui\">
  <div style=\"padding:16px\"><h1 style=\"margin:0 0 8px 0\;font-size:20px\">{title}</h1>
  <svg id=\"chart\" width=\"900\" height=\"500\" role=\"img\" aria-label=\"Line chart\"></svg></div>
<script>
const data = {json.dumps(payload)};
const xKey = {json.dumps(x_key)};
const yKeys = {json.dumps(y_keys)};
const svg = d3.select('#chart');
const width = +svg.attr('width'), height = +svg.attr('height');
const margin = {{top: 20, right: 20, bottom: 40, left: 50}};
const innerW = width - margin.left - margin.right;
const innerH = height - margin.top - margin.bottom;
const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
const x = d3.scaleLinear().domain(d3.extent(data, d => +d[xKey])).range([0, innerW]);
const y = d3.scaleLinear().domain([d3.min(data, d => d3.min(yKeys, k => +d[k])), d3.max(data, d => d3.max(yKeys, k => +d[k]))]).nice().range([innerH, 0]);
g.append('g').attr('transform', `translate(0,${innerH})`).call(d3.axisBottom(x));
g.append('g').call(d3.axisLeft(y));
const colours = d3.schemeTableau10;
yKeys.forEach((k, i) => {{
  const line = d3.line().x(d => x(+d[xKey])).y(d => y(+d[k]));
  g.append('path').datum(data).attr('fill','none').attr('stroke', colours[i % colours.length]).attr('stroke-width',2).attr('d', line);
}});
</script>
</body>
</html>"""

    if output_path is not None:
        Path(output_path).write_text(html, encoding="utf-8")
    return html

