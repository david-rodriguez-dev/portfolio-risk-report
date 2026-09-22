"""Small inline-SVG charts, no dependencies.

Just enough for a risk report: multi-line time series, a filled drawdown
area, horizontal bars, and a correlation heatmap. Everything returns a
string; the report embeds it directly, so the HTML file stands alone.
"""
from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd

PALETTE = ["#1f5fbf", "#d1495b", "#2a9d8f", "#e9a03b", "#7b5ea7", "#6c757d", "#17becf"]
FONT = "font-family='ui-sans-serif, system-ui, sans-serif'"


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi == lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    first = np.ceil(lo / step) * step
    return list(np.arange(first, hi + step / 2, step))


def _year_ticks(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    years = sorted(set(index.year))
    return [index[index.year == y][0] for y in years if y != index[0].year or index[0].month == 1]


class _Frame:
    """Plot area geometry and value-to-pixel mapping."""

    def __init__(self, width, height, x0, x1, y0, y1, margin=(16, 20, 40, 64)):
        top, right, bottom, left = margin
        self.w, self.h = width, height
        self.left, self.top = left, top
        self.pw, self.ph = width - left - right, height - top - bottom
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1

    def x(self, v) -> float:
        return self.left + (v - self.x0) / (self.x1 - self.x0) * self.pw if self.x1 != self.x0 else self.left

    def y(self, v) -> float:
        return self.top + (1 - (v - self.y0) / (self.y1 - self.y0)) * self.ph if self.y1 != self.y0 else self.top

    def axes(self, index: pd.DatetimeIndex, y_fmt, zero_line: bool) -> str:
        parts = [f"<rect x='{self.left}' y='{self.top}' width='{self.pw}' height='{self.ph}' fill='none' stroke='#ccc'/>"]
        for t in _ticks(self.y0, self.y1):
            y = self.y(t)
            parts.append(f"<line x1='{self.left}' x2='{self.left + self.pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='#eee'/>")
            parts.append(f"<text x='{self.left - 6}' y='{y + 4:.1f}' text-anchor='end' font-size='11' fill='#555' {FONT}>{escape(y_fmt(t))}</text>")
        for d in _year_ticks(index):
            x = self.x(d.value)
            parts.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='{self.top}' y2='{self.top + self.ph}' stroke='#eee'/>")
            parts.append(f"<text x='{x:.1f}' y='{self.top + self.ph + 16}' text-anchor='middle' font-size='11' fill='#555' {FONT}>{d.year}</text>")
        if zero_line and self.y0 < 0 < self.y1:
            y = self.y(0)
            parts.append(f"<line x1='{self.left}' x2='{self.left + self.pw}' y1='{y:.1f}' y2='{y:.1f}' stroke='#999' stroke-dasharray='3,3'/>")
        return "".join(parts)


def _path(frame: _Frame, s: pd.Series) -> str:
    pts = [f"{frame.x(i.value):.1f},{frame.y(v):.1f}" for i, v in s.items() if not np.isnan(v)]
    return "M" + " L".join(pts)


def _title(text: str, width: int) -> str:
    return f"<text x='{width / 2}' y='14' text-anchor='middle' font-size='13' font-weight='600' fill='#222' {FONT}>{escape(text)}</text>"


def line_chart(series: dict[str, pd.Series], title: str, *, width=880, height=300, y_fmt=lambda v: f"{v:.2f}",
               zero_line=False, log_scale=False) -> str:
    frame_df = pd.concat(series.values(), axis=1)
    frame_df.columns = list(series)
    values = np.log(frame_df) if log_scale else frame_df
    lo, hi = float(np.nanmin(values.to_numpy())), float(np.nanmax(values.to_numpy()))
    pad = (hi - lo) * 0.05 or 1.0
    fr = _Frame(width, height, frame_df.index[0].value, frame_df.index[-1].value, lo - pad, hi + pad, margin=(24, 20, 40, 64))
    fmt = (lambda v: y_fmt(float(np.exp(v)))) if log_scale else y_fmt
    out = [f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{escape(title)}'>", _title(title, width),
           fr.axes(frame_df.index, fmt, zero_line)]
    for k, (name, col) in enumerate(values.items()):
        out.append(f"<path d='{_path(fr, col)}' fill='none' stroke='{PALETTE[k % len(PALETTE)]}' stroke-width='1.6'/>")
    lx = fr.left + 8
    for k, name in enumerate(series):
        out.append(f"<rect x='{lx}' y='{fr.top + 6 + 16 * k}' width='12' height='3' fill='{PALETTE[k % len(PALETTE)]}'/>")
        out.append(f"<text x='{lx + 16}' y='{fr.top + 10 + 16 * k}' font-size='11' fill='#333' {FONT}>{escape(name)}</text>")
    out.append("</svg>")
    return "".join(out)


def area_chart(s: pd.Series, title: str, *, width=880, height=220, color="#d1495b", y_fmt=lambda v: f"{v:.0%}") -> str:
    lo = float(min(s.min(), 0.0)) * 1.05 or -0.01
    fr = _Frame(width, height, s.index[0].value, s.index[-1].value, lo, 0.0, margin=(24, 20, 40, 64))
    base = fr.y(0)
    pts = [f"{fr.x(i.value):.1f},{fr.y(v):.1f}" for i, v in s.items()]
    d = f"M{fr.x(s.index[0].value):.1f},{base:.1f} L" + " L".join(pts) + f" L{fr.x(s.index[-1].value):.1f},{base:.1f} Z"
    return (f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{escape(title)}'>{_title(title, width)}"
            f"{fr.axes(s.index, y_fmt, False)}<path d='{d}' fill='{color}' fill-opacity='0.35' stroke='{color}' stroke-width='1'/></svg>")


def bar_chart(values: pd.Series, title: str, *, width=880, fmt=lambda v: f"{v:.1%}", color="#1f5fbf") -> str:
    n = len(values)
    row_h, left, top = 24, 220, 28
    height = top + n * row_h + 12
    vmax = float(max(values.abs().max(), 1e-9))
    scale = (width - left - 90) / vmax
    out = [f"<svg viewBox='0 0 {width} {height}' width='100%' role='img' aria-label='{escape(title)}'>", _title(title, width)]
    for k, (name, v) in enumerate(values.items()):
        y = top + k * row_h
        w = abs(float(v)) * scale
        x = left if v >= 0 else left - w
        fill = color if v >= 0 else "#d1495b"
        out.append(f"<text x='{left - 8}' y='{y + 16}' text-anchor='end' font-size='12' fill='#333' {FONT}>{escape(str(name))}</text>")
        out.append(f"<rect x='{x:.1f}' y='{y + 4}' width='{w:.1f}' height='{row_h - 8}' fill='{fill}' fill-opacity='0.85'/>")
        out.append(f"<text x='{left + w + 6:.1f}' y='{y + 16}' font-size='12' fill='#333' {FONT}>{escape(fmt(float(v)))}</text>")
    out.append("</svg>")
    return "".join(out)


def _diverging(v: float) -> str:
    """-1 -> blue, 0 -> white, +1 -> red."""
    v = max(-1.0, min(1.0, v))
    if v >= 0:
        r, g, b = 255, int(255 * (1 - v * 0.75)), int(255 * (1 - v * 0.75))
    else:
        r, g, b = int(255 * (1 + v * 0.75)), int(255 * (1 + v * 0.75)), 255
    return f"rgb({r},{g},{b})"


def heatmap(df: pd.DataFrame, title: str, *, cell=56, label_w=170) -> str:
    n = len(df)
    width, height = label_w + n * cell + 20, 28 + label_w // 2 + n * cell + 10
    top = 28 + label_w // 2
    out = [f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-width:{width}px' role='img' aria-label='{escape(title)}'>", _title(title, width)]
    for j, col in enumerate(df.columns):
        x = label_w + j * cell + cell / 2
        out.append(f"<text transform='translate({x},{top - 6}) rotate(-45)' font-size='11' fill='#333' {FONT}>{escape(str(col))}</text>")
    for i, row in enumerate(df.index):
        y = top + i * cell
        out.append(f"<text x='{label_w - 8}' y='{y + cell / 2 + 4}' text-anchor='end' font-size='11' fill='#333' {FONT}>{escape(str(row))}</text>")
        for j, col in enumerate(df.columns):
            v = float(df.iloc[i, j])
            x = label_w + j * cell
            out.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' fill='{_diverging(v)}' stroke='#fff'/>")
            out.append(f"<text x='{x + cell / 2}' y='{y + cell / 2 + 4}' text-anchor='middle' font-size='11' fill='#222' {FONT}>{v:.2f}</text>")
    out.append("</svg>")
    return "".join(out)
