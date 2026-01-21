#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""06UNIT - Visualisation for Research
Lab 06_02: Interactive Visualisation Toolkit (test-facing API)

This module implements a small, deterministic subset of the unit's interactive
visualisation tooling. The focus is on...
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Literal


@dataclass(frozen=True)
class DataPoint:
    """A single labelled scalar observation.

    Args:
        label: Human-readable label for the observation.
        value: Numeric value.
        category: Optional categorical group.
        colour: Optional colour hint as a hex string (e.g., "#4a9eff").
        metadata: Optional dictionary of extra fields.
    """

    label: str
    value: float
    category: str = ""
    colour: str | None = None
    metadata: dict[str, str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "label": self.label,
            "value": float(self.value),
            "category": self.category,
            "colour": self.colour,
            "metadata": dict(self.metadata) if self.metadata else {},
        }


@dataclass
class Dataset:
    """A collection of :class:`DataPoint` objects."""

    points: list[DataPoint] = field(default_factory=list)
    name: str = ""

    def __len__(self) -> int:
        return len(self.points)

    def categories(self) -> list[str]:
        cats = sorted({p.category for p in self.points if p.category})
        return cats

    def filter_by_category(self, category: str) -> "Dataset":
        return Dataset(points=[p for p in self.points if p.category == category], name=self.name)

    def filter_by_value_range(self, *, min_value: float | None = None, max_value: float | None = None) -> "Dataset":
        def ok(v: float) -> bool:
            if min_value is not None and v < min_value:
                return False
            if max_value is not None and v > max_value:
                return False
            return True

        return Dataset(points=[p for p in self.points if ok(p.value)], name=self.name)

    def aggregate_by_category(self) -> dict[str, float]:
        by: dict[str, list[float]] = {}
        for p in self.points:
            by.setdefault(p.category or "(none)", []).append(p.value)
        return {k: float(mean(vs)) for k, vs in by.items()}

    def get_statistics(self) -> dict[str, float]:
        if not self.points:
            return {"count": 0.0, "min": 0.0, "max": 0.0, "mean": 0.0}
        values = [p.value for p in self.points]
        return {
            "count": float(len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
            "mean": float(sum(values) / len(values)),
        }

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "points": [p.to_dict() for p in self.points]})


@dataclass(frozen=True)
class MetricCard:
    """A simple summary card for dashboards."""

    title: str
    value: str
    unit: str = ""
    change: str | None = None
    change_positive: bool | None = None

    def to_html(self) -> str:
        title = html.escape(self.title)
        value = html.escape(self.value)
        unit = html.escape(self.unit)
        change_html = ""
        if self.change is not None:
            cls = "neutral"
            if self.change_positive is True:
                cls = "positive"
            elif self.change_positive is False:
                cls = "negative"
            change_html = f"<div class=\"metric-change {cls}\">{html.escape(self.change)}</div>"
        return (
            f"<div class=\"metric-card\">"
            f"<div class=\"metric-title\">{title}</div>"
            f"<div class=\"metric-value\">{value}<span class=\"metric-unit\">{unit}</span></div>"
            f"{change_html}"
            f"</div>"
        )


@dataclass(frozen=True)
class FilterControl:
    """A UI control that filters a dataset."""

    name: str
    label: str
    control_type: Literal["select", "range"]
    options: list[str] | None = None
    min_value: float | None = None
    max_value: float | None = None

    @property
    def element_id(self) -> str:
        safe = "".join(ch if ch.isalnum() else "-" for ch in self.name.lower())
        return f"filter-{safe}"

    def to_html(self) -> str:
        label = html.escape(self.label)
        eid = html.escape(self.element_id)
        if self.control_type == "select":
            opts = self.options or []
            options_html = "\n".join(
                f"<option value=\"{html.escape(o)}\">{html.escape(o)}</option>" for o in opts
            )
            return (
                f"<label for=\"{eid}\">{label}</label>"
                f"<select id=\"{eid}\" name=\"{html.escape(self.name)}\">{options_html}</select>"
            )
        # range
        mn = 0.0 if self.min_value is None else float(self.min_value)
        mx = 1.0 if self.max_value is None else float(self.max_value)
        return (
            f"<label for=\"{eid}\">{label}</label>"
            f"<input id=\"{eid}\" type=\"range\" min=\"{mn}\" max=\"{mx}\" step=\"0.01\">"
        )


@dataclass(frozen=True)
class DashboardConfig:
    """Configuration for dashboard generation."""

    title: str
    dataset: Dataset
    filters: list[FilterControl] = field(default_factory=list)
    theme: Literal["dark", "light"] = "dark"


def _svg_root(*, width: int, height: int) -> list[str]:
    return [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width} {height}\" width=\"{width}\" height=\"{height}\">",
        "<title>Chart</title>",
    ]


def generate_bar_chart_svg(
    data: dict[str, float],
    *,
    width: int = 640,
    height: int = 360,
    title: str | None = None,
    colours: dict[str, str] | None = None,
) -> str:
    if not data:
        return "".join(_svg_root(width=width, height=height) + ["</svg>"])

    items = list(data.items())
    max_val = max(abs(v) for _, v in items) or 1.0
    bar_w = (width - 40) / len(items)

    parts = _svg_root(width=width, height=height)
    if title:
        parts.append(f"<text x=\"20\" y=\"24\" fill=\"#eaeaea\">{html.escape(title)}</text>")

    for i, (k, v) in enumerate(items):
        x = 20 + i * bar_w
        h = (height - 60) * (abs(v) / max_val)
        y = (height - 30) - h
        colour = (colours or {}).get(k) or "#4a9eff"
        parts.append(
            f"<rect x=\"{x:.2f}\" y=\"{y:.2f}\" width=\"{bar_w*0.8:.2f}\" height=\"{h:.2f}\" fill=\"{html.escape(colour)}\" />"
        )
        parts.append(
            f"<text x=\"{x:.2f}\" y=\"{height-10}\" font-size=\"12\" fill=\"#eaeaea\">{html.escape(k)}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


def generate_line_chart_svg(
    data: list[tuple[float, float]],
    *,
    width: int = 640,
    height: int = 360,
    title: str | None = None,
    colour: str = "#4a9eff",
) -> str:
    parts = _svg_root(width=width, height=height)
    if title:
        parts.append(f"<text x=\"20\" y=\"24\" fill=\"#eaeaea\">{html.escape(title)}</text>")

    if not data:
        parts.append("</svg>")
        return "".join(parts)

    xs = [x for x, _ in data]
    ys = [y for _, y in data]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(x: float) -> float:
        return 20 + (width - 40) * ((x - x_min) / (x_max - x_min))

    def sy(y: float) -> float:
        return (height - 30) - (height - 60) * ((y - y_min) / (y_max - y_min))

    d = "M " + " L ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in data)
    parts.append(
        f"<path d=\"{d}\" fill=\"none\" stroke=\"{html.escape(colour)}\" stroke-width=\"2\" />"
    )
    parts.append("</svg>")
    return "".join(parts)


def generate_pie_chart_svg(
    data: dict[str, float],
    *,
    width: int = 420,
    height: int = 420,
    title: str | None = None,
    colours: dict[str, str] | None = None,
) -> str:
    parts = _svg_root(width=width, height=height)
    if title:
        parts.append(f"<text x=\"20\" y=\"24\" fill=\"#eaeaea\">{html.escape(title)}</text>")

    total = sum(abs(v) for v in data.values())
    if total <= 0:
        parts.append("</svg>")
        return "".join(parts)

    # Simple pie using SVG arcs
    import math

    cx, cy = width / 2, height / 2 + 10
    r = min(width, height) * 0.35
    start = 0.0
    items = list(data.items())

    for key, val in items:
        frac = abs(val) / total
        end = start + frac * 2 * math.pi
        x1, y1 = cx + r * math.cos(start), cy + r * math.sin(start)
        x2, y2 = cx + r * math.cos(end), cy + r * math.sin(end)
        large = 1 if end - start > math.pi else 0
        colour = (colours or {}).get(key) or "#4a9eff"
        path = (
            f"M {cx:.2f},{cy:.2f} L {x1:.2f},{y1:.2f} "
            f"A {r:.2f},{r:.2f} 0 {large} 1 {x2:.2f},{y2:.2f} Z"
        )
        parts.append(f"<path d=\"{path}\" fill=\"{html.escape(colour)}\" />")
        parts.append(
            f"<text x=\"20\" y=\"{40 + 16*items.index((key,val))}\" fill=\"#eaeaea\">{html.escape(key)}</text>"
        )
        start = end

    parts.append("</svg>")
    return "".join(parts)


def generate_dashboard_html(config: DashboardConfig) -> str:
    """Generate a self-contained HTML dashboard."""

    bg = "#1a1a2e" if config.theme == "dark" else "#ffffff"
    fg = "#eaeaea" if config.theme == "dark" else "#111111"

    filters_html = "\n".join(f.to_html() for f in config.filters)
    stats = config.dataset.get_statistics()
    metric = MetricCard(title="Count", value=str(int(stats["count"])), unit="", change=None)

    bar_svg = generate_bar_chart_svg(config.dataset.aggregate_by_category(), title="Mean by category")

    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{html.escape(config.title)}</title>
<style>
body{{background:{bg};color:{fg};font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:24px;}}
.metric-card{{border:1px solid rgba(255,255,255,.15);border-radius:12px;padding:12px;max-width:260px;}}
.metric-title{{font-size:12px;opacity:.85;}}
.metric-value{{font-size:28px;font-weight:600;}}
.metric-unit{{font-size:14px;margin-left:6px;opacity:.8;}}
.metric-change.positive{{color:#2ecc71;}}
.metric-change.negative{{color:#e74c3c;}}
.controls label{{display:block;margin-top:10px;margin-bottom:4px;}}
.controls select,.controls input{{width:280px;}}
</style>
</head>
<body>
<h1>{html.escape(config.title)}</h1>
<div class=\"controls\">{filters_html}</div>
<div style=\"margin-top:16px\">{metric.to_html()}</div>
<div style=\"margin-top:18px\">{bar_svg}</div>
<script type=\"application/json\" id=\"dataset\">{html.escape(config.dataset.to_json())}</script>
</body>
</html>"""

