"""Chart backends for `kgfm report`.

Three renderers, all returning an HTML fragment to drop into the report:

- ``plotly``     — interactive (hover, zoom, legend toggling). Best for
  reading a training curve closely. plotly.js is inlined once per page so
  the report stays a single self-contained file.
- ``matplotlib`` — static, rendered to inline SVG (vector, so it stays crisp
  and prints well) and embedded directly.
- ``svg``        — the hand-built fallback in `report_html`, used when neither
  library is installed. Keeps the report working on a bare install, since
  matplotlib / plotly are an optional extra (``pip install -e '.[report]'``).

``auto`` picks the first available in that order.
"""

from __future__ import annotations

import html
from typing import Dict, List, Optional, Sequence, Tuple

Series = Tuple[str, Sequence[Tuple[float, float]]]

BACKENDS = ("auto", "plotly", "matplotlib", "svg")

# Shared palette so a curve keeps its colour across backends.
COLORS = ["#2f6fd0", "#c2410c", "#0f766e", "#7c3aed", "#b91c1c", "#0369a1"]

# Scatter needs its own palette: COLORS has six entries, which is fine for the
# line charts (they never carry more than a handful of series) but made a
# 13-entry relation legend assign every colour twice — two unrelated relations
# painted identically, which is worse than folding them away. Ten
# reasonably-separable hues, ordered so the first few stay distinct under
# common colour-vision deficiencies (Okabe-Ito for the first six).
SCATTER_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
    "#56B4E9", "#7C3AED", "#B91C1C", "#0F766E", "#A16207",
]

# The folded-together remainder in a scatter legend. Kept out of the palette
# and painted neutral grey on purpose: it is not a category, so giving it a hue
# would read as one more class competing with the real ones.
OTHER_LABEL = "other"
OTHER_COLOR = "#94a3b8"


def _scatter_colors(groups) -> List[str]:
    """One colour per group, with the `other` bucket held out of the palette.

    Held out rather than simply indexed past, so folding a long tail away does
    not shift every real category's colour by one.
    """
    out, i = [], 0
    for name, _ in groups:
        if name == OTHER_LABEL:
            out.append(OTHER_COLOR)
            continue
        out.append(SCATTER_COLORS[i % len(SCATTER_COLORS)])
        i += 1
    return out


def available(backend: str = "auto") -> str:
    """Resolve ``backend`` to one that can actually render here."""
    if backend != "auto":
        return backend
    try:
        import plotly.graph_objects  # noqa: F401

        return "plotly"
    except ImportError:
        pass
    try:
        import matplotlib  # noqa: F401

        return "matplotlib"
    except ImportError:
        return "svg"


def describe(backend: str) -> str:
    resolved = available(backend)
    try:
        if resolved == "plotly":
            import plotly

            return f"plotly {plotly.__version__} (interactive)"
        if resolved == "matplotlib":
            import matplotlib

            return f"matplotlib {matplotlib.__version__} (static SVG)"
    except ImportError:
        pass
    return "built-in SVG (no plotting library installed)"


# ---------------------------------------------------------------------------
# plotly
# ---------------------------------------------------------------------------

_PLOTLY_EMITTED = False


def _pick(colors, i, fallback_index):
    """Explicit colour when one was supplied for this series, else the palette."""
    if colors and i < len(colors) and colors[i]:
        return colors[i]
    return COLORS[fallback_index % len(COLORS)]


def _plotly_chart(
    series: List[Series],
    *,
    x_label: str,
    y_label: str,
    y2_series: Optional[List[Series]] = None,
    y2_label: str = "",
    height: int = 320,
    colors: Optional[Sequence[str]] = None,
) -> str:
    global _PLOTLY_EMITTED
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, pts) in enumerate(series):
        if not pts:
            continue
        xs, ys = zip(*pts)
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), name=name, mode="lines+markers",
            line=dict(color=_pick(colors, i, i), width=2),
            marker=dict(size=5),
            hovertemplate=f"{name}<br>step %{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))
    offset = len(series)
    for j, (name, pts) in enumerate(y2_series or []):
        if not pts:
            continue
        xs, ys = zip(*pts)
        fig.add_trace(go.Scatter(
            x=list(xs), y=list(ys), name=name, mode="lines+markers", yaxis="y2",
            line=dict(color=_pick(colors, offset + j, offset + j), width=2,
                      dash="dot"),
            marker=dict(size=5, symbol="diamond"),
            hovertemplate=f"{name}<br>step %{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))

    layout = dict(
        height=height,
        margin=dict(l=60, r=60, t=10, b=45),
        xaxis=dict(title=x_label, zeroline=False),
        yaxis=dict(title=y_label, zeroline=False),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        # Transparent so the page's light/dark background shows through.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    if y2_series:
        layout["yaxis2"] = dict(
            title=y2_label, overlaying="y", side="right", zeroline=False,
        )
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.2)")

    # plotly.js is ~3MB; inline it once and reference it from later figures so
    # a multi-chart report doesn't multiply that by the number of charts.
    include = "inline" if not _PLOTLY_EMITTED else False
    _PLOTLY_EMITTED = True
    return fig.to_html(
        include_plotlyjs=include, full_html=False,
        config={"displaylogo": False, "responsive": True},
    )


def reset_plotly_state() -> None:
    """Force the next plotly figure to re-inline the library (new page)."""
    global _PLOTLY_EMITTED
    _PLOTLY_EMITTED = False


# ---------------------------------------------------------------------------
# matplotlib
# ---------------------------------------------------------------------------


def _matplotlib_chart(
    series: List[Series],
    *,
    x_label: str,
    y_label: str,
    y2_series: Optional[List[Series]] = None,
    y2_label: str = "",
    height: int = 320,
    colors: Optional[Sequence[str]] = None,
) -> str:
    import io

    import matplotlib

    matplotlib.use("Agg")           # no display on a compute node
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, height / 100), dpi=100)
    for i, (name, pts) in enumerate(series):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=4, lw=2,
                color=_pick(colors, i, i), label=name)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=.25)
    handles, labels = ax.get_legend_handles_labels()

    if y2_series:
        ax2 = ax.twinx()
        offset = len(series)
        for j, (name, pts) in enumerate(y2_series):
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, marker="D", ms=4, lw=2, ls="--",
                     color=_pick(colors, offset + j, offset + j), label=name)
        ax2.set_ylabel(y2_label)
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    if handles:
        ax.legend(handles, labels, fontsize=8, framealpha=.6)

    # Transparent background + neutral text so one image reads on either theme.
    for spine in ax.spines.values():
        spine.set_alpha(.4)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    buf = io.StringIO()
    fig.tight_layout()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    # Drop the XML prolog / doctype so the fragment can be inlined in HTML.
    start = svg.find("<svg")
    return f'<div class="mpl">{svg[start:] if start >= 0 else svg}</div>'


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def chart(
    series: List[Series],
    *,
    x_label: str,
    y_label: str,
    backend: str = "auto",
    y2_series: Optional[List[Series]] = None,
    y2_label: str = "",
    height: int = 320,
    colors: Optional[Sequence[str]] = None,
) -> str:
    """Render one chart, optionally with a second y-axis, as an HTML fragment.

    ``colors`` overrides the categorical palette with one explicit colour per
    series, which is how the scaling plots make hue mean *model size* rather
    than series order. It covers ``y2_series`` too, indexed after the primary
    ones — a figure that puts train loss on y1 and valid loss on y2 must give
    the same cell the same hue on both axes or the encoding is a lie.
    """
    series = [(n, list(p)) for n, p in series if p]
    y2_series = [(n, list(p)) for n, p in (y2_series or []) if p]
    if not series and not y2_series:
        return '<p class="note">no data</p>'

    resolved = available(backend)
    try:
        if resolved == "plotly":
            return _plotly_chart(
                series, x_label=x_label, y_label=y_label,
                y2_series=y2_series, y2_label=y2_label, height=height,
                colors=colors,
            )
        if resolved == "matplotlib":
            return _matplotlib_chart(
                series, x_label=x_label, y_label=y_label,
                y2_series=y2_series, y2_label=y2_label, height=height,
                colors=colors,
            )
    except Exception as exc:                                # noqa: BLE001
        # A plotting failure must not cost you the rest of the report.
        note = (f'<p class="note">chart backend {html.escape(resolved)} failed '
                f"({html.escape(type(exc).__name__)}); using built-in SVG</p>")
        return note + _fallback(series + y2_series, x_label, y_label, colors)
    return _fallback(series + y2_series, x_label, y_label, colors)


def _fallback(series: List[Series], x_label: str, y_label: str,
              colors: Optional[Sequence[str]] = None) -> str:
    from .report_html import line_chart

    return line_chart(series, x_label=x_label, y_label=y_label, colors=colors)


# ---------------------------------------------------------------------------
# grouped bars (final metrics per method)
# ---------------------------------------------------------------------------


def bar_chart(
    categories: Sequence[str],
    groups: Sequence[Tuple[str, Sequence[Optional[float]]]],
    *,
    y_label: str,
    backend: str = "auto",
    height: int = 320,
) -> str:
    """Grouped bars: one group per series, one bar per category."""
    groups = [(n, list(v)) for n, v in groups if any(x is not None for x in v)]
    if not categories or not groups:
        return '<p class="note">no data</p>'
    resolved = available(backend)
    try:
        if resolved == "plotly":
            return _plotly_bars(categories, groups, y_label=y_label, height=height)
        if resolved == "matplotlib":
            return _matplotlib_bars(categories, groups, y_label=y_label, height=height)
    except Exception as exc:                                # noqa: BLE001
        note = (f'<p class="note">chart backend {html.escape(resolved)} failed '
                f"({html.escape(type(exc).__name__)}); using built-in SVG</p>")
        return note + _bar_fallback(categories, groups, y_label)
    return _bar_fallback(categories, groups, y_label)


def _plotly_bars(categories, groups, *, y_label: str, height: int) -> str:
    global _PLOTLY_EMITTED
    import plotly.graph_objects as go

    fig = go.Figure()
    for i, (name, values) in enumerate(groups):
        fig.add_trace(go.Bar(
            x=list(categories), y=list(values), name=name,
            marker_color=COLORS[i % len(COLORS)],
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=height,
        margin=dict(l=60, r=20, t=10, b=60),
        yaxis=dict(title=y_label, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.2)")
    include = "inline" if not _PLOTLY_EMITTED else False
    _PLOTLY_EMITTED = True
    return fig.to_html(include_plotlyjs=include, full_html=False,
                       config={"displaylogo": False, "responsive": True})


def _matplotlib_bars(categories, groups, *, y_label: str, height: int) -> str:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_groups = len(groups)
    width = 0.8 / n_groups
    fig, ax = plt.subplots(figsize=(7.2, height / 100), dpi=100)
    for i, (name, values) in enumerate(groups):
        xs = [j + (i - (n_groups - 1) / 2) * width for j in range(len(categories))]
        ys = [0 if v is None else v for v in values]
        ax.bar(xs, ys, width=width, label=name, color=COLORS[i % len(COLORS)])
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(y_label)
    ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=8, framealpha=.6)
    for spine in ax.spines.values():
        spine.set_alpha(.4)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    buf = io.StringIO()
    fig.tight_layout()
    fig.savefig(buf, format="svg", transparent=True)
    plt.close(fig)
    svg = buf.getvalue()
    start = svg.find("<svg")
    return f'<div class="mpl">{svg[start:] if start >= 0 else svg}</div>'


def _bar_fallback(categories, groups, y_label: str) -> str:
    from .report_html import bar_chart_svg

    return bar_chart_svg(categories, groups, y_label=y_label)


# ---------------------------------------------------------------------------
# scatter (2-D embedding projections)
# ---------------------------------------------------------------------------


def scatter_chart(
    groups: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    *,
    x_label: str,
    y_label: str,
    backend: str = "auto",
    height: int = 560,
    max_legend: int = 10,
) -> str:
    """Point cloud, one colour per group (role or node type)."""
    groups = [(n, list(p)) for n, p in groups if p]
    if not groups:
        return '<p class="note">no data</p>'
    # Largest group FIRST, because draw order is paint order in both backends
    # and these classes overlap almost completely in projection: h vs t and
    # Literal vs URIRef sit on top of each other, so whichever went last used
    # to hide the other entirely and the panel said nothing about the split it
    # was colouring (measured on the scaling run: 't' 2000 pts covering 'h'
    # 2000, and URIRef 3227 covering Literal 773). Painting big to small keeps
    # the minority class visible.
    groups.sort(key=lambda g: len(g[1]), reverse=True)
    # Too many categories makes the legend useless and the colours ambiguous;
    # keep the largest and fold the rest into one bucket. The bucket goes to
    # the front so it stays background: it is a leftover, not a category, and
    # it is the widest-spread group (45 relations fold to 12 + other here).
    if len(groups) > max_legend:
        rest = [pt for _, pts in groups[max_legend:] for pt in pts]
        groups = ([(OTHER_LABEL, rest)] if rest else []) + groups[:max_legend]

    resolved = available(backend)
    try:
        if resolved == "plotly":
            return _plotly_scatter(groups, x_label=x_label, y_label=y_label,
                                   height=height)
        if resolved == "matplotlib":
            return _matplotlib_scatter(groups, x_label=x_label, y_label=y_label,
                                       height=height)
    except Exception as exc:                                # noqa: BLE001
        note = (f'<p class="note">chart backend {html.escape(resolved)} failed '
                f"({html.escape(type(exc).__name__)}); using built-in SVG</p>")
        return note + _scatter_fallback(groups, x_label, y_label)
    return _scatter_fallback(groups, x_label, y_label)


def _plotly_scatter(groups, *, x_label: str, y_label: str, height: int) -> str:
    global _PLOTLY_EMITTED
    import plotly.graph_objects as go

    colors = _scatter_colors(groups)
    fig = go.Figure()
    for (name, pts), color in zip(groups, colors):
        xs, ys = zip(*pts)
        fig.add_trace(go.Scattergl(          # gl handles thousands of points
            x=list(xs), y=list(ys), name=f"{name} ({len(pts)})", mode="markers",
            marker=dict(size=5, opacity=.45, color=color),
            hovertemplate=f"{name}<br>%{{x:.3f}}, %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        # Right margin is legend space. The legend used to be horizontal at
        # y=1.0 with only 10px of top margin, so a 13-entry relation legend
        # was drawn straight over the top of the point cloud.
        height=height, margin=dict(l=55, r=185, t=10, b=45),
        xaxis=dict(title=x_label, zeroline=False),
        yaxis=dict(title=y_label, zeroline=False, scaleanchor="x", scaleratio=1),
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.02,
            # Legend swatches follow marker size otherwise, and size-5 dots at
            # 60% opacity are not identifiable at legend scale.
            itemsizing="constant", font=dict(size=10),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.2)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.2)")
    include = "inline" if not _PLOTLY_EMITTED else False
    _PLOTLY_EMITTED = True
    return fig.to_html(include_plotlyjs=include, full_html=False,
                       config={"displaylogo": False, "responsive": True})


def _matplotlib_scatter(groups, *, x_label: str, y_label: str, height: int) -> str:
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Square plot area plus a fixed legend column on the right, matching the
    # plotly path: `adjustable="datalim"` honours equal aspect by widening the
    # data range to the box, so a wide box leaves the cloud in a narrow strip
    # with empty gutters either side.
    side = height / 100
    fig, ax = plt.subplots(figsize=(side + 1.9, side), dpi=100)
    for i, ((name, pts), color) in enumerate(zip(groups, _scatter_colors(groups))):
        xs, ys = zip(*pts)
        # Explicit zorder rather than relying on call order, so the paint
        # order the caller established survives any later reordering here.
        ax.scatter(xs, ys, s=7, alpha=.45, linewidths=0, zorder=2 + i,
                   color=color, label=f"{name} ({len(pts)})")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=.25)
    ax.legend(fontsize=7, framealpha=.6, markerscale=2.2,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    for spine in ax.spines.values():
        spine.set_alpha(.4)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    buf = io.StringIO()
    fig.tight_layout()
    # Thousands of markers as SVG paths bloat the file; rasterize the points
    # and keep the axes as vector.
    for coll in ax.collections:
        coll.set_rasterized(True)
    fig.savefig(buf, format="svg", transparent=True, dpi=110)
    plt.close(fig)
    svg = buf.getvalue()
    start = svg.find("<svg")
    return f'<div class="mpl">{svg[start:] if start >= 0 else svg}</div>'


def _scatter_fallback(groups, x_label: str, y_label: str) -> str:
    from .report_html import scatter_chart_svg

    return scatter_chart_svg(groups, x_label=x_label, y_label=y_label)
