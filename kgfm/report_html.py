"""HTML rendering for `kgfm report`.

Produces a single self-contained ``report.html`` per run: the same comparison
table as ``table.md``, plus the training curves recovered from the per-cell
logs and the run's metadata.

Charts are hand-built inline SVG rather than matplotlib. kgfm's install is
torch + numpy (transformers optional), and a plotting stack is a lot of
dependency for a handful of line charts — this way the report also opens
anywhere with no assets to resolve.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Point = Tuple[float, float]

# Colours are picked to stay legible on both light and dark backgrounds, so a
# single palette works under either theme.
_SERIES_COLORS = ["#2f6fd0", "#c2410c", "#0f766e", "#7c3aed", "#b91c1c", "#0369a1"]

_CSS = """
:root {
  --bg: #ffffff; --fg: #1a1a1a; --muted: #5f6b7a; --line: #e2e6ec;
  --card: #f7f9fb; --accent: #2f6fd0; --warn: #b91c1c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c; --fg: #e6e9ee; --muted: #9aa5b4; --line: #2b313a;
    --card: #1b1f26; --accent: #6ea8fe; --warn: #ff8080;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
        "Hiragino Sans", "Noto Sans JP", sans-serif;
}
main { max-width: 1040px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .75rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--line); }
h3 { font-size: .95rem; margin: 1.5rem 0 .5rem; font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 1.5rem; font-size: .9rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
        gap: .5rem 1.25rem; margin: 0; }
.grid div { display: flex; gap: .5rem; font-size: .875rem; min-width: 0; }
.grid dt { color: var(--muted); flex: 0 0 auto; }
.grid dd { margin: 0; word-break: break-all; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .875rem; }
th, td { padding: .45rem .7rem; border-bottom: 1px solid var(--line);
         text-align: right; white-space: nowrap; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
th:nth-child(3), td:nth-child(3) { text-align: left; }
thead th { color: var(--muted); font-weight: 600; border-bottom: 2px solid var(--line); }
tbody tr:hover { background: var(--card); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
        padding: .9rem 1.1rem; margin: .75rem 0; }
.card h3 { margin-top: 0; }
.failed { color: var(--warn); }
.chart { display: block; width: 100%; height: auto; max-width: 720px; }
.charts { display: flex; flex-wrap: wrap; gap: 1.5rem; }
.charts figure { margin: 0; flex: 1 1 320px; min-width: 0; }
.charts.wide figure { flex: 1 1 100%; }
.mpl svg { display: block; width: 100%; height: auto; max-width: 760px; }
.js-plotly-plot { width: 100% !important; }
figcaption { color: var(--muted); font-size: .8rem; margin-bottom: .25rem; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em; }
.note { color: var(--muted); font-size: .85rem; }
details { margin: .75rem 0; }
summary { cursor: pointer; font-size: .9rem; font-weight: 600; }
ol.cmds { margin: .5rem 0; padding-left: 1.4rem; }
ol.cmds li { margin: .35rem 0; }
ol.cmds code { display: inline-block; background: var(--card); padding: .15rem .4rem;
               border-radius: 4px; word-break: break-all; }
footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
         border-top: 1px solid var(--line); padding-top: .75rem; }
"""


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------


def _nice_ticks(lo: float, hi: float, count: int = 5) -> List[float]:
    """Evenly spaced tick values across [lo, hi], collapsing a flat range."""
    if hi <= lo:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + step * i for i in range(count)]


def _fmt_num(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 0.1:
        return f"{v:.3f}"
    return f"{v:.2e}"


def line_chart(
    series: Sequence[Tuple[str, Sequence[Point]]],
    *,
    x_label: str,
    y_label: str,
    width: int = 700,
    height: int = 240,
    y_fmt: Callable[[float], str] = _fmt_num,
) -> str:
    """Render one or more (label, points) series as a standalone inline SVG."""
    series = [(name, list(pts)) for name, pts in series if pts]
    if not series:
        return '<p class="note">no data</p>'

    left, right, top, bottom = 62, 14, 14, 40
    pw, ph = width - left - right, height - top - bottom

    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 1, x_hi + 1
    if y_hi == y_lo:
        pad = abs(y_hi) * 0.05 or 0.5
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        pad = (y_hi - y_lo) * 0.08
        y_lo, y_hi = y_lo - pad, y_hi + pad

    def px(x: float) -> float:
        return left + (x - x_lo) / (x_hi - x_lo) * pw

    def py(y: float) -> float:
        return top + ph - (y - y_lo) / (y_hi - y_lo) * ph

    out: List[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{html.escape(y_label)} vs {html.escape(x_label)}">',
        '<g stroke="currentColor" stroke-opacity=".18" stroke-width="1">',
    ]
    for ty in _nice_ticks(y_lo, y_hi):
        y = py(ty)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + pw}" y2="{y:.1f}"/>')
    out.append("</g>")

    # tick labels
    out.append('<g font-size="10" fill="currentColor" fill-opacity=".6">')
    for ty in _nice_ticks(y_lo, y_hi):
        out.append(
            f'<text x="{left - 8}" y="{py(ty) + 3:.1f}" text-anchor="end">'
            f"{html.escape(y_fmt(ty))}</text>"
        )
    for tx in _nice_ticks(x_lo, x_hi):
        out.append(
            f'<text x="{px(tx):.1f}" y="{top + ph + 16}" text-anchor="middle">'
            f"{html.escape(_fmt_num(tx))}</text>"
        )
    out.append("</g>")

    # axis titles
    out.append(
        f'<text x="{left + pw / 2:.0f}" y="{height - 4}" text-anchor="middle" '
        f'font-size="11" fill="currentColor" fill-opacity=".75">'
        f"{html.escape(x_label)}</text>"
    )
    out.append(
        f'<text x="14" y="{top + ph / 2:.0f}" font-size="11" fill="currentColor" '
        f'fill-opacity=".75" text-anchor="middle" '
        f'transform="rotate(-90 14 {top + ph / 2:.0f})">{html.escape(y_label)}</text>'
    )

    # axes
    out.append(
        f'<g stroke="currentColor" stroke-opacity=".45"><line x1="{left}" y1="{top}" '
        f'x2="{left}" y2="{top + ph}"/><line x1="{left}" y1="{top + ph}" '
        f'x2="{left + pw}" y2="{top + ph}"/></g>'
    )

    for i, (name, pts) in enumerate(series):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        out.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{coords}"/>'
        )
        if len(pts) <= 40:
            for x, y in pts:
                out.append(
                    f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2.5" fill="{color}">'
                    f"<title>{html.escape(_fmt_num(x))}, "
                    f"{html.escape(y_fmt(y))}</title></circle>"
                )
        if len(series) > 1:
            out.append(
                f'<text x="{left + 8 + i * 120}" y="{top + 12}" font-size="10" '
                f'fill="{color}">{html.escape(name)}</text>'
            )
    out.append("</svg>")
    return "".join(out)


def bar_chart_svg(
    categories: Sequence[str],
    groups: Sequence[Tuple[str, Sequence[Optional[float]]]],
    *,
    y_label: str,
    width: int = 700,
    height: int = 300,
) -> str:
    """Dependency-free grouped bars, matching `line_chart`'s look."""
    groups = [(n, list(v)) for n, v in groups if any(x is not None for x in v)]
    if not categories or not groups:
        return '<p class="note">no data</p>'

    left, right, top, bottom = 62, 14, 26, 64
    pw, ph = width - left - right, height - top - bottom
    values = [v for _, vals in groups for v in vals if v is not None]
    y_hi = max(values + [0.0]) * 1.12 or 1.0
    y_lo = min(values + [0.0])
    y_lo = min(y_lo, 0.0)

    def py(v: float) -> float:
        return top + ph - (v - y_lo) / (y_hi - y_lo) * ph

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{html.escape(y_label)} by method">',
        '<g stroke="currentColor" stroke-opacity=".18">',
    ]
    for tv in _nice_ticks(y_lo, y_hi):
        out.append(f'<line x1="{left}" y1="{py(tv):.1f}" x2="{left + pw}" '
                   f'y2="{py(tv):.1f}"/>')
    out.append("</g>")
    out.append('<g font-size="10" fill="currentColor" fill-opacity=".6">')
    for tv in _nice_ticks(y_lo, y_hi):
        out.append(f'<text x="{left - 8}" y="{py(tv) + 3:.1f}" text-anchor="end">'
                   f"{html.escape(_fmt_num(tv))}</text>")
    out.append("</g>")

    slot = pw / max(1, len(categories))
    bw = slot * 0.8 / len(groups)
    for ci, cat in enumerate(categories):
        base = left + slot * ci + slot * 0.1
        for gi, (name, vals) in enumerate(groups):
            v = vals[ci] if ci < len(vals) else None
            if v is None:
                continue
            x = base + gi * bw
            y = py(max(v, 0.0))
            h = abs(py(v) - py(0.0))
            color = _SERIES_COLORS[gi % len(_SERIES_COLORS)]
            out.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" '
                f'height="{max(h, 1):.1f}" fill="{color}" rx="2">'
                f"<title>{html.escape(name)} — {html.escape(cat)}: "
                f"{html.escape(_fmt_num(v))}</title></rect>"
            )
        out.append(
            f'<text x="{left + slot * ci + slot / 2:.1f}" y="{top + ph + 16}" '
            f'text-anchor="middle" font-size="9" fill="currentColor" '
            f'fill-opacity=".7">{html.escape(cat[:22])}</text>'
        )
    for gi, (name, _) in enumerate(groups):
        out.append(
            f'<text x="{left + gi * 110}" y="14" font-size="10" '
            f'fill="{_SERIES_COLORS[gi % len(_SERIES_COLORS)]}">'
            f"{html.escape(name)}</text>"
        )
    out.append("</svg>")
    return "".join(out)


def scatter_chart_svg(
    groups: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    *,
    x_label: str,
    y_label: str,
    width: int = 700,
    height: int = 420,
) -> str:
    """Dependency-free point cloud, matching `line_chart`'s look."""
    groups = [(n, list(p)) for n, p in groups if p]
    if not groups:
        return '<p class="note">no data</p>'

    left, right, top, bottom = 56, 14, 30, 42
    pw, ph = width - left - right, height - top - bottom
    xs = [x for _, pts in groups for x, _ in pts]
    ys = [y for _, pts in groups for _, y in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 1, x_hi + 1
    if y_hi == y_lo:
        y_lo, y_hi = y_lo - 1, y_hi + 1

    def px(x: float) -> float:
        return left + (x - x_lo) / (x_hi - x_lo) * pw

    def py(y: float) -> float:
        return top + ph - (y - y_lo) / (y_hi - y_lo) * ph

    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="{html.escape(y_label)} vs {html.escape(x_label)}">',
        f'<g stroke="currentColor" stroke-opacity=".45">'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + ph}"/>'
        f'<line x1="{left}" y1="{top + ph}" x2="{left + pw}" y2="{top + ph}"/></g>',
    ]
    for i, (name, pts) in enumerate(groups):
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        # One <g> per group keeps the file far smaller than per-point fills.
        out.append(f'<g fill="{color}" fill-opacity=".55">')
        out += [f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="2"/>' for x, y in pts]
        out.append("</g>")
        out.append(
            f'<text x="{left + 6 + i * 118}" y="18" font-size="10" fill="{color}">'
            f"{html.escape(f'{name} ({len(pts)})')}</text>"
        )
    out.append(
        f'<text x="{left + pw / 2:.0f}" y="{height - 6}" text-anchor="middle" '
        f'font-size="11" fill="currentColor" fill-opacity=".75">'
        f"{html.escape(x_label)}</text>"
    )
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------


def _kv_grid(pairs: Sequence[Tuple[str, Any]]) -> str:
    items = "".join(
        f"<div><dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd></div>"
        for k, v in pairs
        if v not in (None, "")
    )
    return f'<dl class="grid">{items}</dl>'


def _table(rows: List[Dict[str, str]]) -> str:
    """The same rows as table.md, rendered as HTML.

    Deliberately no "best value" highlighting. The rows are not mutually
    comparable — kgfm cells may be pooled or filtered, and even within
    `filtered` kgfm ranks tails only while ULTRA / MOTIF average head and tail
    over a different n_eval. Bolding a column maximum would invite exactly the
    cross-protocol reading benchmarks/README.md spends a section warning about.
    """
    if not rows:
        return '<p class="note">no results found</p>'
    headers = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = "".join(
            f"<td>{html.escape(str(r.get(h, '—')))}</td>" for h in headers
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="scroll"><table><thead><tr>' + head + "</tr></thead><tbody>"
        + "".join(body) + "</tbody></table></div>"
    )


def _cell_charts(curve: Any, backend: str) -> str:
    """Every training chart for one cell, with its summary numbers."""
    from . import charts as charts_mod

    figures: List[Tuple[str, str, str]] = []

    # The headline chart: train and valid loss together. Both are the same
    # in-batch-negative objective at the same batch size, so they share an
    # axis and the gap between them is meaningful.
    loss_series = [("train loss", list(zip(curve.steps, curve.losses)))]
    if curve.valid_losses:
        loss_series.append(("valid loss", curve.valid_losses))
        caption, note = "loss — train vs valid", ""
    else:
        caption = "loss — train"
        note = ('<p class="note">No valid loss logged. Run with '
                "<code>--eval-every</code> set (and "
                "<code>--valid-loss-batches &gt; 0</code>) to overlay it.</p>")
    figures.append((caption, charts_mod.chart(
        loss_series, x_label="optimizer step", y_label="loss", backend=backend,
    ), note))

    metric_series = [
        (name, pts) for name, pts in sorted(curve.valid_metrics.items()) if pts
    ]
    if metric_series:
        figures.append(("validation metrics (in-loop eval is always pooled)",
                        charts_mod.chart(metric_series, x_label="optimizer step",
                                         y_label="score", backend=backend), ""))

    if curve.valid_losses and curve.gap():
        figures.append((
            "generalisation gap (valid − train loss)",
            charts_mod.chart([("gap", curve.gap())], x_label="optimizer step",
                             y_label="valid − train loss", backend=backend),
            '<p class="note">Rising = the model is starting to fit the training '
            "stream rather than the task.</p>",
        ))

    if curve.grad_norms:
        figures.append((
            "gradient norm (pre-clip)",
            charts_mod.chart([("‖g‖", curve.grad_norms)], x_label="optimizer step",
                             y_label="gradient norm", backend=backend),
            '<p class="note">Measured before clipping, so a flat line at the '
            "clip threshold means <code>--grad-clip</code> is binding on every "
            "step.</p>",
        ))

    if curve.rates:
        figures.append(("throughput", charts_mod.chart(
            [("examples/s", list(zip(curve.steps, curve.rates)))],
            x_label="optimizer step", y_label="examples / s", backend=backend,
        ), ""))

    stats: List[Tuple[str, Any]] = [("steps logged", len(curve.steps))]
    if curve.losses:
        stats += [("train loss", f"{curve.losses[0]:.4f} → {curve.losses[-1]:.4f}"),
                  ("Δ train", f"{curve.losses[-1] - curve.losses[0]:+.4f}")]
    if curve.valid_losses:
        vs = [v for _, v in curve.valid_losses]
        stats += [("valid loss", f"{vs[0]:.4f} → {vs[-1]:.4f}"),
                  ("Δ valid", f"{vs[-1] - vs[0]:+.4f}"),
                  ("final gap (valid − train)", f"{vs[-1] - curve.losses[-1]:+.4f}")]
    if curve.encoder_info:
        stats.append(("model", curve.encoder_info))

    figs = "".join(
        f"<figure><figcaption>{html.escape(cap)}</figcaption>{body}{note}</figure>"
        for cap, body, note in figures
    )
    return _kv_grid(stats) + f'<div class="charts wide">{figs}</div>'


def _record_block(record: Dict[str, Any]) -> str:
    """Final metrics + settings for one (method, protocol) result."""
    metrics = record.get("metrics") or {}
    scores = [(k, f"{v:.4f}" if isinstance(v, float) and abs(v) < 1 else v)
              for k, v in metrics.items()
              if isinstance(v, (int, float)) and not isinstance(v, bool)]
    settings: List[Tuple[str, Any]] = []
    for key, value in (record.get("protocol") or {}).items():
        if value is not None and key != "type":
            settings.append((f"protocol.{key}", value))
    for key in ("train_seconds", "elapsed_seconds", "world_size", "batch_size",
                "proj_dim", "conda_env", "gpus", "ckpt"):
        if record.get(key) is not None:
            settings.append((key, record[key]))

    err = ""
    if record.get("error"):
        hint = record.get("hint")
        err = (f'<p class="failed">{html.escape(str(record["error"]))}'
               + (f'<br><span class="note">{html.escape(str(hint))}</span>'
                  if hint else "") + "</p>")
    return (err + "<p class=\"note\">final metrics</p>" + _kv_grid(scores)
            + "<p class=\"note\">settings</p>" + _kv_grid(settings))


def _cmd_list(entries: Sequence[Dict[str, Any]]) -> str:
    """Commands as a copy-pasteable ordered list."""
    items = []
    for e in entries:
        cmd = html.escape(str(e.get("command", "")))
        bits = []
        if e.get("ts"):
            bits.append(html.escape(str(e["ts"])))
        if e.get("note"):
            bits.append(html.escape(str(e["note"])))
        meta = f'<span class="note"> — {" · ".join(bits)}</span>' if bits else ""
        parent = ""
        if e.get("parent"):
            parent = (f'<div class="note">via <code>'
                      f'{html.escape(str(e["parent"]))}</code></div>')
        items.append(f"<li><code>{cmd}</code>{meta}{parent}</li>")
    return f'<ol class="cmds">{"".join(items)}</ol>'


def _commands_section(commands: Sequence[Dict[str, Any]]) -> str:
    """Top-level invocations that produced this run."""
    invocations = [c for c in commands if c.get("kind") != "step"]
    if not invocations:
        return ""
    return (
        "<details open><summary>commands that produced this run "
        f'<span class="note">({len(invocations)})</span></summary>'
        + _cmd_list(invocations)
        + '<p class="note">Child processes each command spawned are listed in '
          "the section they belong to. The full log is "
          "<code>commands.jsonl</code> in this directory.</p></details>"
    )


def _method_commands(commands: Sequence[Dict[str, Any]], tag: str,
                     names: Sequence[str]) -> str:
    """The child commands attributable to one method."""
    wanted = {tag, *names}
    steps = [c for c in commands
             if c.get("kind") == "step" and c.get("tag") in wanted]
    if not steps:
        return ""
    return ("<details><summary>commands "
            f'<span class="note">({len(steps)})</span></summary>'
            + _cmd_list(steps) + "</details>")


# Short, readable legend entries: the relation labels are full IRIs.
def _short_label(value: str, limit: int = 26) -> str:
    tail = value.rsplit("#", 1)[-1].rsplit("/", 1)[-1] or value
    return tail if len(tail) <= limit else tail[: limit - 1] + "…"


def _embedding_section(record: Dict[str, Any], backend: str) -> str:
    """Scatter plots of a checkpoint's h / t embedding space."""
    from . import charts as charts_mod

    pts = record.get("points") or {}
    xs, ys = pts.get("x") or [], pts.get("y") or []
    if not xs:
        return ""

    figures = []
    for key, caption in (
        ("role", "coloured by role (head vs tail)"),
        ("relation", "coloured by relation of the sampled triple"),
        ("type", "coloured by RDF node type"),
    ):
        labels = pts.get(key)
        if not labels:
            continue
        distinct = set(labels)
        # A single-valued label paints one flat colour and says nothing.
        if len(distinct) < 2:
            continue
        groups: Dict[str, List[Tuple[float, float]]] = {}
        for x, y, lab in zip(xs, ys, labels):
            groups.setdefault(_short_label(str(lab)), []).append((x, y))
        figures.append((
            caption,
            charts_mod.scatter_chart(
                sorted(groups.items()), x_label="component 1",
                y_label="component 2", backend=backend,
            ),
            "",
        ))
    if not figures:
        return ""

    reducer = record.get("reducer", "?")
    stats: List[Tuple[str, Any]] = [
        ("reducer", reducer),
        ("points", record.get("n_points")),
        ("embedding dim", record.get("dim")),
    ]
    ev = record.get("explained_variance_ratio")
    if ev:
        stats.append(("explained variance",
                      " + ".join(f"{v:.1%}" for v in ev)
                      + f" = {sum(ev):.1%}"))
    note = (
        '<p class="note">Entity strings sampled from the test split, '
        "deduplicated by text and split evenly between head and tail, then "
        f"encoded with this checkpoint and projected with {html.escape(reducer)}. "
        "The colour labels are never seen during training — kgfm only reads the "
        "text fields — so any grouping here was learned.</p>"
    )
    figs = "".join(
        f"<figure><figcaption>{html.escape(cap)}</figcaption>{body}{extra}</figure>"
        for cap, body, extra in figures
    )
    return note + _kv_grid(stats) + f'<div class="charts wide">{figs}</div>'


def _method_key(record: Dict[str, Any]) -> Tuple[str, str]:
    """(sort key, display name) identifying the method a record belongs to."""
    method = record.get("method", "?")
    encoder = record.get("encoder")
    if not encoder:
        # Baselines sort after kgfm, in the same order as the table
        # (ULTRA then MOTIF) rather than alphabetically.
        order = {"ULTRA": "2", "MOTIF": "3"}.get(method, "4")
        return (f"{order}_{method}", method)
    # The key has to name every axis the sweep varies, or two different cells
    # share a section: their training curves would be interleaved under one
    # heading and the head axis would be invisible. "auto" is omitted so runs
    # that sweep a single head keep the section titles they have always had.
    head = record.get("head")
    parts = [encoder] + ([head] if head and head != "auto" else [])
    if record.get("freeze_encoder"):
        parts.append("frozen")
    suffix = "".join(f", {p}" for p in parts[1:])
    return (f"1_{'_'.join(parts)}", f"{method} ({encoder}{suffix})")


def _method_sections(
    records: List[Dict[str, Any]],
    curves: List[Any],
    backend: str,
    embeddings: Optional[Dict[str, Dict[str, Any]]] = None,
    commands: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    """One section per method, with a subsection per protocol.

    Grouped by method rather than by plot type: a cell trains once and is then
    re-scored under each protocol, so the training curves belong to the method
    and only the metrics differ per protocol. Showing the curves once under the
    method avoids duplicating them across protocol subsections.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    names: Dict[str, str] = {}
    for rec in records:
        key, name = _method_key(rec)
        grouped.setdefault(key, []).append(rec)
        names[key] = name

    # Curves are keyed by cell tag; `cell` is "<protocol>_<tag>".
    curve_by_tag: Dict[str, Any] = {}
    for c in curves:
        if not c.steps:
            continue
        _, _, tag = c.cell.partition("_")
        curve_by_tag.setdefault(tag, c)

    out = []
    for key in sorted(grouped):
        recs = grouped[key]
        tag = key.split("_", 1)[1]        # strip the sort prefix
        curve = curve_by_tag.get(tag)
        out.append(f"<h2>{html.escape(names[key])}</h2>")
        cmds = _method_commands(commands or [], tag, [names[key], tag])
        if cmds:
            out.append(cmds)

        for rec in sorted(recs, key=lambda r: (r.get("protocol") or {}).get("type", "")):
            proto = (rec.get("protocol") or {}).get("type")
            if not proto:
                proto = ("filtered"
                         if str(rec.get("method", "")).lower() in ("ultra", "motif")
                         else "?")
            mode = rec.get("mode", "trained")
            out.append(
                f'<h3>protocol: {html.escape(proto)} '
                f'<span class="note">({html.escape(str(mode))})</span></h3>'
                f'<div class="card">{_record_block(rec)}</div>'
            )

        emb = (embeddings or {}).get(tag)
        if emb:
            out.append("<h3>embedding space</h3>")
            out.append(f'<div class="card">{_embedding_section(emb, backend)}</div>')

        if curve is not None:
            out.append("<h3>training</h3>")
            out.append(
                '<p class="note">The cell trains once; each protocol above '
                "re-scores that same checkpoint, so these curves cover all of "
                "them.</p>"
            )
            out.append(f'<div class="card">{_cell_charts(curve, backend)}</div>')
        elif key.startswith("1_"):
            out.append(
                '<h3>training</h3><p class="note">No step log found for this '
                "cell.</p>"
            )
    return "".join(out)


def _cross_cell_section(curves: List[Any], backend: str = "auto") -> str:
    """All cells on shared axes, so encoders can be compared directly."""
    from . import charts as charts_mod

    trained = [c for c in curves if c.steps]
    if len(trained) < 2:
        return ""

    figures = []
    # x = examples seen, not steps: cells run at different batch sizes, so the
    # same step count means very different amounts of data.
    figures.append((
        "train loss by examples seen",
        charts_mod.chart(
            [(c.cell, list(zip(c.examples(c.steps), c.losses))) for c in trained],
            x_label="examples seen", y_label="train loss", backend=backend,
        ),
        '<p class="note">Plotted against examples rather than steps: cells with '
        "different batch sizes see very different amounts of data per step.</p>",
    ))
    valid_loss = [(c.cell, c.valid_losses) for c in trained if c.valid_losses]
    if valid_loss:
        figures.append(("valid loss by step", charts_mod.chart(
            valid_loss, x_label="optimizer step", y_label="valid loss",
            backend=backend), ""))
    mrr = [(c.cell, c.valid_metrics.get("MRR", [])) for c in trained]
    if any(pts for _, pts in mrr):
        figures.append(("validation MRR by step", charts_mod.chart(
            mrr, x_label="optimizer step", y_label="MRR", backend=backend), ""))

    figs = "".join(
        f"<figure><figcaption>{html.escape(cap)}</figcaption>{body}{note}</figure>"
        for cap, body, note in figures
    )
    return f'<div class="card"><div class="charts wide">{figs}</div></div>'


def _metric_profile_section(
    records: List[Dict[str, Any]], backend: str = "auto"
) -> str:
    """Final metrics as grouped bars, one chart per protocol.

    Split by protocol on purpose: a single chart mixing pooled and filtered
    rows would invite the cross-protocol comparison the table deliberately
    avoids. Within a protocol the bars are at least measuring the same thing —
    though kgfm still ranks tails only while ULTRA / MOTIF average head+tail.
    """
    from . import charts as charts_mod

    metric_keys = ["mrr", "hits@1", "hits@3", "hits@10", "ndcg"]
    by_protocol: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}
    for rec in records:
        if rec.get("error"):
            continue
        proto = (rec.get("protocol") or {}).get("type")
        if not proto:
            proto = ("filtered"
                     if str(rec.get("method", "")).lower() in ("ultra", "motif")
                     else "?")
        _, name = _method_key(rec)
        flat = {}
        for k, v in (rec.get("metrics") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[k.lower().replace("hit@", "hits@")] = float(v)
        if flat:
            by_protocol.setdefault(proto, []).append((name, flat))

    blocks = []
    for proto, entries in sorted(by_protocol.items()):
        present = [k for k in metric_keys if any(k in m for _, m in entries)]
        if not present:
            continue
        blocks.append(
            f'<h3>final metrics — {html.escape(proto)} protocol</h3>'
            f'<div class="card">'
            + charts_mod.bar_chart(
                [name for name, _ in entries],
                [(key, [m.get(key) for _, m in entries]) for key in present],
                y_label="score", backend=backend,
            ) + "</div>"
        )
    if not blocks:
        return ""
    return (
        '<p class="note">One chart per protocol — bars are only meaningful '
        "within a protocol, and even then kgfm ranks tails only while "
        "ULTRA / MOTIF average head and tail.</p>" + "".join(blocks)
    )


def render(
    run_dir: Path,
    rows: List[Dict[str, str]],
    records: List[Dict[str, Any]],
    curves: List[Any],
    meta: Optional[Dict[str, Any]] = None,
    kg_stats: Optional[Dict[str, Any]] = None,
    backend: str = "auto",
    embeddings: Optional[Dict[str, Dict[str, Any]]] = None,
    commands: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    from . import charts as charts_mod

    charts_mod.reset_plotly_state()
    meta = meta or {}
    title = f"kgfm benchmark — {run_dir.name}"

    meta_pairs = [
        ("run", run_dir.name),
        ("timestamp (UTC)", meta.get("timestamp_utc")),
        ("host", meta.get("host")),
        ("conda env", meta.get("conda_env")),
        ("python", meta.get("python")),
        ("torch", meta.get("torch")),
        ("gpu", meta.get("gpu")),
        ("git rev", (meta.get("git_rev") or "")[:12] or None),
        ("dirty files", meta.get("git_dirty_files")),
        ("charts", charts_mod.describe(backend)),
    ]

    sections = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="sub">{html.escape(str(run_dir))}</p>',
        _kv_grid(meta_pairs),
        _commands_section(commands or []),
        '<p class="note">Each method gets its own section below, with one '
        "subsection per evaluation protocol. Everything that compares methods "
        "against each other is collected at the end.</p>",
        # --- per method ---
        _method_sections(records, curves, backend, embeddings, commands)
        or '<p class="note">No results found in this run directory.</p>',
        # --- cross-method, last ---
        "<h2>Cross-method comparison</h2>",
        '<p class="note"><strong>Rows are not directly comparable.</strong> '
        "kgfm rows may be <em>pooled</em> (ranked against a sampled candidate "
        "pool) or <em>filtered</em> (ranked against the full tail vocabulary); "
        "ULTRA / MOTIF always report filtered ranking <em>averaged over head "
        "and tail</em>, over a different <code>n_eval</code>. No column "
        "maximum is highlighted for that reason — see "
        "<code>benchmarks/README.md</code> before drawing conclusions across "
        "methods.</p>",
        _table(rows),
        _metric_profile_section(records, backend),
    ]

    cross = _cross_cell_section(curves, backend)
    if cross:
        sections += ["<h3>training curves across cells</h3>", cross]

    if kg_stats:
        sections += ["<h2>Prepared KG</h2>", _kv_grid(list(kg_stats.items()))]
    params = meta.get("params") or {}
    if params:
        sections += ["<h2>Run parameters</h2>", _kv_grid(list(params.items()))]

    sections.append(
        "<footer>Generated by <code>kgfm report</code> on "
        f"{time.strftime('%Y-%m-%d %H:%M:%S')}. "
        "Curves are parsed from the per-cell logs in this directory.</footer>"
    )

    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n"
        "</head>\n<body>\n<main>\n" + "\n".join(sections) + "\n</main>\n</body>\n</html>\n"
    )
