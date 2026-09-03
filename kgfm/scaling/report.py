"""`kgfm scaling report` — scaling-law plots for a benchmark run directory.

Reads the same `cell_*.log` files `kgfm report` does and re-expresses each
cell's validation history in scaling coordinates: compute on the x-axis,
validation loss on the y-axis, one line per model size. Writes
``scaling_table.md`` and ``scaling_report.html`` next to the ordinary report,
so a run directory can carry both.

The axes are log10 rather than log-scaled: `kgfm/charts.py` has three backends
behind one call and no log-axis support, and transforming the data keeps all
three working identically. Ticks therefore read ``-2`` for 0.01 PF-days.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import charts as charts_mod
from ..report import DEFAULT_RESULTS_ROOT, _markdown_table
from ..report_html import _CSS, _kv_grid
from ..runs import RunLogger, resolve_run_dir
from .compute import PF_DAY, format_pf_days
from .palette import by_size, viridis
from .points import (MIN_SATURATING_POINTS, SaturatingFit, ScalingSeries,
                     Y_AXES, by_family, collect, family_of, fit_power_law,
                     fit_saturating, frontier, frontier_on, point_y)

# Compute budgets (PF-days) at which the "loss vs model size" slices are cut.
# Each slice answers "at this budget, which size was ahead?", which is the
# question a scaling study is ultimately for.
DEFAULT_SLICES = (0.01, 0.1, 1.0, 10.0)


def _log10(values: Sequence[float]) -> List[float]:
    return [math.log10(v) for v in values if v > 0]


def _axis_vs_compute(series: Sequence[ScalingSeries], backend: str,
                     axis: str = "loss") -> str:
    """The headline plot: a y-axis against compute, coloured by size.

    Cells are ordered by parameter count so the palette runs small -> large,
    which is what makes the colour readable as "model size" at a glance.
    """
    plot, sizes = [], []
    for s in series:
        if not s.comparable:
            continue
        pts = []
        for p in s.points:
            y = point_y(p, axis)
            if p.pf_days > 0 and y is not None:
                pts.append((math.log10(p.pf_days), y))
        if pts:
            plot.append((s.label, pts))
            sizes.append(s.params)
    if not plot:
        return ('<p class="note">No cell is on the compute axis yet. This '
                "needs logs written after the trainer started recording "
                "<code>tokens=</code>, and a transformer encoder.</p>")
    return charts_mod.chart(
        plot, x_label="log10(compute, PF-days)", y_label=Y_AXES[axis][0],
        backend=backend, height=380, colors=by_size(sizes),
    )


def _loss_vs_step(series: Sequence[ScalingSeries], backend: str) -> str:
    """Train (solid, left axis) and validation (dotted, right) loss vs step.

    This is the one plot in the report on a *step* axis rather than a compute
    axis, and it is here to answer a specific operational question: can a cell
    be stopped when its validation loss starts rising? On this task the answer
    is no, and the shape is why — see `early_stopping_note`.

    Two axes because the two losses are not on the same scale (train reaches
    ~0.7 while validation sits near 3.0); one axis would flatten the validation
    curves into a line and hide the very feature the plot exists to show.
    """
    train, valid, colors = [], [], []
    sizes = [s.params for s in series]
    ramp = by_size(sizes)
    for i, s in enumerate(series):
        if s.train_losses:
            train.append((f"{s.label} train", list(s.train_losses)))
            colors.append(ramp[i])
    for i, s in enumerate(series):
        pts = [(p.step, p.valid_loss) for p in s.points]
        if pts:
            valid.append((f"{s.label} valid", pts))
            colors.append(ramp[i])
    if not train and not valid:
        return '<p class="note">no loss history in these logs</p>'
    return charts_mod.chart(
        train, x_label="training step", y_label="training loss",
        y2_series=valid, y2_label="validation loss",
        backend=backend, height=420, colors=colors,
    )


def early_stopping_note(series: Sequence[ScalingSeries]) -> str:
    """Measured verdict on early stopping, computed from these very curves.

    Reports, per cell, where validation loss first bottoms out and how much
    better it later gets. A cell whose late minimum beats its early one by
    more than noise is a cell that patience-based early stopping would have
    truncated at the wrong place.
    """
    rows = []
    for s in series:
        losses = [(p.step, p.valid_loss) for p in s.points]
        if len(losses) < 8:
            continue
        cut = len(losses) // 4
        e_step, e_loss = min(losses[:cut], key=lambda t: t[1])
        l_step, l_loss = min(losses[cut:], key=lambda t: t[1])
        rows.append({
            "Cell": s.cell,
            "Params": f"{s.params:,}",
            "Best in first 25%": f"{e_loss:.4f} @ {e_step:,}",
            "Best after that": f"{l_loss:.4f} @ {l_step:,}",
            "Later is better by": f"{e_loss - l_loss:+.4f}",
            "Early stop would": ("LOSE %.4f" % (e_loss - l_loss)
                                 if e_loss - l_loss > 0.02 else "be safe"),
        })
    if not rows:
        return ""
    harmed = [r["Cell"] for r in rows if r["Early stop would"].startswith("LOSE")]
    verdict = (
        "<p><strong>Early stopping is not safe on this task.</strong> "
        + ", ".join(f"<code>{html.escape(c)}</code>" for c in harmed)
        + " reach their best validation loss only <em>after</em> a sustained "
          "mid-run rise, so any patience that fires during that rise truncates "
          "them before their real optimum.</p>"
        if harmed else
        "<p>No cell here improves materially after its early minimum, so early "
        "stopping would have been safe on this particular run.</p>"
    )
    return (verdict + '<div class="scroll">' + _table_html(rows) + "</div>")


def _loss_vs_examples(series: Sequence[ScalingSeries], backend: str) -> str:
    plot, sizes = [], []
    for s in series:
        pts = [(math.log10(p.examples), p.valid_loss)
               for p in s.points if p.examples > 0]
        if pts:
            plot.append((s.label, pts))
            sizes.append(s.params)
    # Same colour = same model size on every chart in the report.
    return charts_mod.chart(
        plot, x_label="log10(training triples seen)",
        y_label="validation loss", backend=backend, height=340,
        colors=by_size(sizes),
    ) if plot else '<p class="note">no data</p>'


def _frontier_plot(series: Sequence[ScalingSeries], backend: str,
                   axis: str = "loss"
                   ) -> Tuple[str, Optional[Tuple[float, float, float]],
                              Optional[SaturatingFit]]:
    """Compute-efficient frontier in log-log, with both fitted laws."""
    env = frontier_on(series, axis)
    label = Y_AXES[axis][0]
    if not env:
        return '<p class="note">no comparable cells</p>', None, None

    # Every cell's own trajectory goes in first, coloured by size, so the
    # frontier is visible as the lower envelope *of something* rather than a
    # line with no context — which model is on the frontier at which budget is
    # the whole point, and a bare envelope hides it.
    plot: List[Tuple[str, List[Tuple[float, float]]]] = []
    colors: List[str] = []
    sizes = [s.params for s in series if s.comparable]
    ramp = by_size(sizes)
    k = 0
    for s in series:
        if not s.comparable:
            continue
        cloud = []
        for p in s.points:
            y = point_y(p, axis)
            if p.pf_days > 0 and y is not None and y > 0:
                cloud.append((math.log10(p.pf_days), math.log10(y)))
        if cloud:
            plot.append((s.label, cloud))
            colors.append(ramp[k])
        k += 1

    pts = [(math.log10(c), math.log10(l)) for c, l, _ in env if c > 0 and l > 0]
    plot.append(("frontier", pts))
    colors.append("#b91c1c")          # deliberately outside viridis
    raw = [(c, l) for c, l, _ in env]
    fit = fit_power_law(raw)
    xs = [x for x, _ in pts]
    lo, hi = (min(xs), max(xs)) if xs else (0.0, 1.0)
    if fit:
        a, b, _ = fit
        plot.append((f"fit  L = {a:.3g}\u00b7C^{b:.3f}",
                     [(x, math.log10(a) + b * x) for x in (lo, hi)]))
        colors.append("#111111")
    # The two-parameter law cannot bend. If the frontier is flattening — which
    # is what "no scaling law" usually turns out to be — only the saturating
    # form can say so, and it is drawn on the same axes so the difference is
    # visible rather than a number in a caption.
    sat = fit_saturating(raw)
    if sat and sat.a > 0:
        curve = []
        for i in range(41):
            x = lo + (hi - lo) * i / 40
            y = sat.l_inf + sat.a * (10.0 ** x) ** sat.b
            if y > 0:
                curve.append((x, math.log10(y)))
        if len(curve) > 1:
            plot.append((f"fit  {sat.label}", curve))
            colors.append("#7c3aed")
    return charts_mod.chart(
        plot, x_label="log10(compute, PF-days)",
        y_label=f"log10({label})",
        backend=backend, height=360, colors=colors,
    ), fit, sat


def _family_rows(series: Sequence[ScalingSeries], axis: str = "loss"
                 ) -> List[Dict[str, str]]:
    """One fit per size-family, plus the pooled fit, as table rows.

    A pooled frontier mixes recipes: `bert-medium` and `bge-large` differ in
    size *and* in what they were pretrained on, so an exponent fitted across
    them is not a capacity measurement. Fitting inside a family holds the
    recipe fixed, which is the only version of the number worth quoting.
    """
    groups = dict(by_family(series))
    groups["all cells (pooled)"] = [s for s in series if s.comparable]
    rows = []
    for name, cells in groups.items():
        env = frontier_on(cells, axis)
        raw = [(c, l) for c, l, _ in env]
        fit = fit_power_law(raw)
        sat = fit_saturating(raw)
        rows.append({
            "Family": name,
            "Sizes": str(len({c.params for c in cells})),
            "Frontier points": str(len(raw)),
            "Exponent b": f"{fit[1]:.4f}" if fit else "—",
            "R\u00b2": f"{fit[2]:.3f}" if fit else "—",
            "Floor L\u221e": (
                f"{sat.l_inf:.4g}" + (" (unconstrained)" if sat.at_boundary else "")
                if sat else f"needs {MIN_SATURATING_POINTS} pts"),
            "b with floor": f"{sat.b:.4f}" if sat else "—",
            "R\u00b2 with floor": f"{sat.r2:.3f}" if sat else "—",
        })
    return rows


def _family_plot(series: Sequence[ScalingSeries], backend: str,
                 axis: str = "loss") -> str:
    """Each family's own frontier, so the fits above can be eyeballed."""
    plot = []
    for name, cells in by_family(series).items():
        env = frontier_on(cells, axis)
        pts = [(math.log10(c), math.log10(l)) for c, l, _ in env
               if c > 0 and l > 0]
        if len(pts) >= 2:
            plot.append((name, pts))
    if not plot:
        return ('<p class="note">No family here has more than one model size. '
                "A within-family fit needs at least two sizes sharing a "
                "pretraining recipe — see FAMILIES in kgfm/scaling/points.py.</p>")
    return charts_mod.chart(
        plot, x_label="log10(compute, PF-days)",
        y_label=f"log10({Y_AXES[axis][0]})", backend=backend, height=360,
    )


def _loss_vs_params(series: Sequence[ScalingSeries], backend: str,
                    slices: Sequence[float] = DEFAULT_SLICES) -> str:
    """Loss against model size, sliced at fixed compute budgets.

    For each budget, each cell contributes the best loss it had reached by
    then — i.e. "if you stop everything at C, who is ahead?".
    """
    plot = []
    for budget in slices:
        pts = []
        for s in series:
            if not s.comparable:
                continue
            reached = [p.valid_loss for p in s.points if 0 < p.pf_days <= budget]
            if reached and s.params > 0:
                pts.append((math.log10(s.params), min(reached)))
        if len(pts) >= 2:
            plot.append((f"C <= {format_pf_days(budget)} PF-days", sorted(pts)))
    return charts_mod.chart(
        plot, x_label="log10(trainable parameters)", y_label="best validation loss",
        backend=backend, height=340,
    ) if plot else ('<p class="note">Not enough cells reached these budgets. '
                    "Widen `slices` or train longer.</p>")


def _frontier_params(series: Sequence[ScalingSeries], backend: str) -> str:
    """Which model size sits on the frontier at each compute budget."""
    env = frontier(series)
    by_cell = {s.cell: s for s in series}
    pts = [(math.log10(c), math.log10(by_cell[cell].params))
           for c, _, cell in env if c > 0 and cell in by_cell
           and by_cell[cell].params > 0]
    if len(pts) < 2:
        return '<p class="note">not enough frontier points</p>'
    return charts_mod.chart(
        [("frontier model size", pts)], x_label="log10(compute, PF-days)",
        y_label="log10(trainable parameters)", backend=backend, height=320,
    )


def _epoch_examples(train_list: str) -> Optional[int]:
    """Rows in one pass over the training list, or None if it cannot be read.

    Counted, not estimated: the ChEMBL files are not uniform (1.81 GB down to
    39 KB), so scaling one file's count by the file count is 26% off. The
    result is cached per (path, size, mtime), so this is a few seconds after
    the first call and must never be allowed to fail a report.
    """
    try:
        from ..data import epoch_examples, read_file_list
        files = read_file_list(train_list)
        return epoch_examples(files) or None
    except Exception:                                       # noqa: BLE001
        return None


def _settings_section(out_dir: Path, series: Sequence[ScalingSeries]) -> str:
    """What was actually run: the settings, and what the step counts mean.

    A step count is not interpretable on its own — 48,000 steps is a large
    number and a tiny fraction of this corpus at the same time. The per-cell
    table therefore carries examples seen and the equivalent in epochs beside
    the step count, computed from the *measured* row count of the training
    list rather than from the config.
    """
    params: Dict[str, Any] = {}
    meta = out_dir / "meta.json"
    if meta.is_file():
        try:
            params = json.loads(meta.read_text()).get("params", {}) or {}
        except (OSError, ValueError):
            params = {}

    train_list = params.get("train_list", "")
    total = _epoch_examples(train_list) if train_list else None

    rows = []
    for s in series:
        last = s.points[-1] if s.points else None
        steps = last.step if last else 0
        examples = last.examples if last else 0
        rows.append({
            "Cell": s.cell,
            "Params": f"{s.params:,}",
            "Steps": f"{steps:,}",
            "Examples seen": f"{examples:,}",
            "Epochs": (f"{examples / total:.4f}" if total and examples else "—"),
            "Tokens (encoder)": f"{last.tokens:,}" if last and last.tokens else "—",
            "PF-days": format_pf_days(last.pf_days) if last else "—",
        })

    def _g(key: str, default: str = "—") -> str:
        v = params.get(key)
        return default if v in (None, "", []) else str(v)

    grid = [
        ("config", _g("config_file")),
        ("train list", _g("train_list")),
        ("valid / test list", f"{_g('valid_list')} / {_g('test_list')}"),
        ("rows in one epoch", f"{total:,}" if total else "not counted"),
        ("batch size (global)", _g("batch_size")),
        ("GPUs per cell", _g("nproc", "1")),
        ("grad accumulation", _g("gradient_accumulation_steps", "1")),
        ("proj_dim / head", f"{_g('proj_dim')} / {', '.join(params.get('heads') or ['—'])}"),
        ("loss", _g("loss", "contrastive (default)")),
        ("encoder / head weight decay",
         f"{_g('encoder_weight_decay')} / {_g('head_weight_decay')}"),
        ("eval every", f"{_g('eval_every')} steps"),
        ("valid-loss batches", _g("valid_loss_batches")),
        ("eval protocol", ", ".join(params.get("protocols") or ["—"])),
        ("candidate pool / eval triples",
         f"{_g('pool_size')} / {_g('n_eval_triples')}"),
        ("seed", _g("seed", "0")),
    ]
    note = (
        "<p class=\"note\">Epochs are computed from the <em>measured</em> row "
        "count of the training list, not from the config — the ChEMBL files "
        "are far from uniform, so scaling one file's count by the file count "
        "is 26% off. Note how small the epoch figures are: this study trains "
        "many steps on a corpus large enough that it never approaches one "
        "pass, so there is no per-row repetition and the train/valid gap is "
        "population shift between files, not memorisation.</p>"
    )
    return (f"<h2>Experiment settings</h2>{_kv_grid(grid)}"
            f'<div class="scroll">{_table_html(rows)}</div>{note}')


def build_rows(series: Sequence[ScalingSeries]) -> List[Dict[str, str]]:
    rows = []
    for s in series:
        best = s.best()
        last = s.points[-1] if s.points else None
        rows.append({
            "Cell": s.cell,
            "Encoder": s.encoder,
            "Mode": "frozen" if s.frozen else "fine-tune",
            "Params": f"{s.params:,}",
            "Best valid loss": f"{best.valid_loss:.4f}" if best else "—",
            "PF-days @ best": format_pf_days(best.pf_days) if best else "—",
            "PF-days total": format_pf_days(last.pf_days) if last else "—",
            "Tokens": f"{last.tokens:,}" if last and last.tokens else "—",
            "Triples": f"{last.examples:,}" if last else "—",
            "On compute axis": "yes" if s.comparable else "no",
        })
    return rows


def render(series: Sequence[ScalingSeries], out_dir: Path,
           backend: str = "auto") -> str:
    rows = build_rows(series)
    incomparable = [s.label for s in series if not s.comparable]
    family_rows = _family_rows(series)
    # Only offer the second y-axis if it was actually measured. In-loop
    # validation logs MRR alongside the loss, but a run configured with
    # `eval_n_triples: 0` has loss and nothing else.
    has_mrr = any(point_y(p, "mrr_error") is not None
                  for s in series for p in s.points)

    # The plotly backend inlines its ~3MB library into whichever figure is
    # rendered FIRST and has every later figure reference it. So the figures
    # must be *rendered* in the same order they are *placed*, or the earlier
    # ones call Plotly.newPlot before the library exists and silently show
    # nothing. Hence the thunks: the list below is document order, and
    # rendering walks it in that order rather than whatever order the values
    # happened to be computed in.
    fit: Optional[Tuple[float, float, float]] = None
    sat: Optional[SaturatingFit] = None

    def frontier_fig() -> str:
        nonlocal fit, sat
        body, fit, sat = _frontier_plot(series, backend, "loss")
        return body

    mrr_fit: Optional[Tuple[float, float, float]] = None
    mrr_sat: Optional[SaturatingFit] = None

    def mrr_frontier_fig() -> str:
        nonlocal mrr_fit, mrr_sat
        body, mrr_fit, mrr_sat = _frontier_plot(series, backend, "mrr_error")
        return body

    plan: List[Tuple[str, Any, str]] = [
        ("validation loss vs compute, coloured by model size",
         lambda: _axis_vs_compute(series, backend, "loss"),
         "<p class=\"note\">One line per cell, traced along its own training "
         "run, coloured by log(parameters) on a viridis ramp — dark = small, "
         "yellow = large. This is the plot the study exists for: where the "
         "lines cross, the larger model has become worth its cost.</p>"),
        ("compute-efficient frontier (log-log) with power-law fit",
         frontier_fig, ""),
        ("frontier per size-family, fitted separately",
         lambda: _family_plot(series, backend, "loss"),
         "<p class=\"note\">A pooled fit across every cell varies size and "
         "pretraining recipe at once, so its exponent is not a statement "
         "about capacity. These hold the recipe fixed. The table below "
         "carries both.</p>"),
        ("best validation loss vs model size, at fixed compute budgets",
         lambda: _loss_vs_params(series, backend), ""),
        ("model size on the frontier vs compute",
         lambda: _frontier_params(series, backend),
         "<p class=\"note\">Rising = bigger models earn their place as the "
         "budget grows; flat = the budget is better spent on data.</p>"),
        ("train and validation loss vs step — why early stopping fails here",
         lambda: _loss_vs_step(series, backend), "__EARLY_STOP__"),
        ("validation loss vs training data seen",
         lambda: _loss_vs_examples(series, backend),
         "<p class=\"note\">The data axis, for separating \"more compute\" "
         "from \"more data\" — they move together unless model size varies.</p>"),
    ]
    if has_mrr:
        # A second y-axis, because the first one has a ceiling built into it.
        plan += [
            ("ranking error (1 - MRR) vs compute, coloured by model size",
             lambda: _axis_vs_compute(series, backend, "mrr_error"),
             f"<p class=\"note\">{html.escape(Y_AXES['mrr_error'][1])} "
             "Same cells, same colours, same x-axis as the first plot — only "
             "the y-axis differs, so a slope that appears here and not there "
             "is the loss's ceiling talking, not the model's.</p>"),
            ("ranking-error frontier (log-log) with power-law fit",
             mrr_frontier_fig, ""),
        ]
    rendered = [(cap, body_fn(), note) for cap, body_fn, note in plan]

    def _fit_note(f: Optional[Tuple[float, float, float]],
                  sf: Optional[SaturatingFit], symbol: str) -> str:
        if not f:
            return ("<p class=\"note\">Not enough frontier points to fit a "
                    "power law (three are needed). Add model sizes or train "
                    "longer.</p>")
        out = (f"<p><strong>{symbol} = {f[0]:.4g} · C<sup>{f[1]:.4f}</sup>"
               f"</strong> (R² = {f[2]:.3f} in log-log, C in PF-days). A more "
               "negative exponent means the metric falls faster per decade of "
               "compute.</p>")
        if sf:
            out += (f"<p><strong>{symbol} = {sf.l_inf:.4g} + {sf.a:.3g} · "
                    f"C<sup>{sf.b:.4f}</sup></strong> (R² = {sf.r2:.3f} in "
                    f"linear {symbol}, {sf.n} points). The three-parameter "
                    "form separates an irreducible floor from the part that "
                    "still moves with compute; the two-parameter fit cannot "
                    "bend, so on a flattening curve it reports a shallow "
                    "exponent that is really the floor.")
            if sf.at_boundary:
                out += (" <strong>The floor sat at the edge of the scan</strong> "
                        "— the data does not constrain it, so read |b| as a "
                        "lower bound rather than an estimate.")
            out += "</p>"
        else:
            out += (f"<p class=\"note\">No floor fitted: that needs "
                    f"{MIN_SATURATING_POINTS} frontier points, below which "
                    "L∞ and the coefficient are not separately "
                    "identifiable.</p>")
        return out

    notes = {
        "train and validation loss vs step — why early stopping fails here":
            ("<p class=\"note\">Solid = training loss (left axis), dotted = "
             "validation loss (right axis); same hue is the same cell. The "
             "axes differ because the two losses do not share a scale, and "
             "one axis would flatten the validation curves.</p>"
             + early_stopping_note(series)),
        "compute-efficient frontier (log-log) with power-law fit":
            _fit_note(fit, sat, "L"),
        "ranking-error frontier (log-log) with power-law fit":
            _fit_note(mrr_fit, mrr_sat, "1−MRR"),
    }
    rendered = [(cap, body, notes.get(cap, note)) for cap, body, note in rendered]
    figs = "".join(
        f"<figure><figcaption>{html.escape(cap)}</figcaption>{body}{note}</figure>"
        for cap, body, note in rendered
    )

    caveats = [
        "Each line is one training run sampled at its validation points, so "
        "points within a line share a model and differ only in how long it "
        "trained.",
        "Colour is viridis over log10(trainable parameters) and means the same "
        "thing on every chart here.",
        "Compute is 6·N·T for a fine-tuned encoder and 2·N·T for a frozen one "
        "(no backward through the LM), plus 6·N·T for the head, with T the "
        "<em>measured</em> padded token count.",
        "The pooled exponent mixes model size with pretraining recipe. Quote "
        "a within-family exponent instead; the family table says which is "
        "which.",
        "Validation loss is a softmax over in-batch negatives, so it is "
        "bounded above by ln(B) and has a floor set by how often a tail "
        "repeats inside a batch. Both compress the range a size axis can "
        "move it through — which is why 1−MRR, ranked against a "
        "fixed-size candidate pool, is plotted alongside it.",
    ]
    if incomparable:
        caveats.append(
            "Excluded from every compute axis: "
            + ", ".join(html.escape(x) for x in incomparable)
            + " — an EmbeddingBag lookup does no work proportional to its "
            "parameter count, so 6ND would rank it as the most expensive "
            "model in the sweep when it is the cheapest."
        )

    body = f"""<main>
<h1>kgfm scaling study</h1>
<p class="sub">{html.escape(str(out_dir))}</p>
{_settings_section(out_dir, series)}
<h2>Cells</h2>
<div class="scroll">{_table_html(rows)}</div>
<h2>Fits</h2>
<div class="scroll">{_table_html(family_rows)}</div>
<h2>Scaling plots</h2>
<div class="card"><div class="charts wide">{figs}</div></div>
<h2>How to read this</h2>
<ul>{''.join(f'<li>{c}</li>' for c in caveats)}</ul>
<h2>Run</h2>
{_kv_grid([("cells", str(len(series))),
           ("on compute axis", str(sum(1 for s in series if s.comparable))),
           ("1 PF-day", f"{PF_DAY:.3g} FLOPs")])}
<footer>kgfm scaling report</footer>
</main>"""
    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>kgfm scaling — {html.escape(out_dir.name)}</title>"
            f"<style>{_CSS}</style></head><body>{body}</body></html>")


def _table_html(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return '<p class="note">no cells</p>'
    heads = list(rows[0])
    out = ["<table><thead><tr>"]
    out += [f"<th>{html.escape(h)}</th>" for h in heads]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(
            f"<td>{html.escape(str(r.get(h, '—')))}</td>" for h in heads) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def build(out_dir: Path, backend: str = "auto") -> Dict[str, Any]:
    series = collect(out_dir)
    if not series:
        raise SystemExit(
            f"No usable training logs in {out_dir}.\n"
            "A scaling report needs cell_*.log files with validation losses "
            "(`--valid-loss-batches > 0`)."
        )
    rows = build_rows(series)
    table = _markdown_table(rows)
    (out_dir / "scaling_table.md").write_text(table)
    (out_dir / "scaling_report.html").write_text(render(series, out_dir, backend))

    def _axis_summary(axis: str) -> Dict[str, Any]:
        env = frontier_on(series, axis)
        raw = [(c, l) for c, l, _ in env]
        fit = fit_power_law(raw)
        sat = fit_saturating(raw)
        return {
            "frontier": [{"pf_days": c, "value": l, "cell": cell}
                         for c, l, cell in env],
            "power_law": ({"a": fit[0], "exponent": fit[1], "r2": fit[2]}
                          if fit else None),
            # Three-parameter form. `at_boundary` means the floor is not
            # identified by this data — the exponent beside it is then a lower
            # bound on |b|, not an estimate.
            "saturating": ({"l_inf": sat.l_inf, "a": sat.a, "exponent": sat.b,
                            "r2": sat.r2, "points": sat.n,
                            "at_boundary": sat.at_boundary} if sat else None),
            "families": [
                {"family": name,
                 "sizes": sorted({c.params for c in cells}),
                 "power_law": (lambda f: {"a": f[0], "exponent": f[1],
                                          "r2": f[2]} if f else None)(
                     fit_power_law([(c, l) for c, l, _ in
                                    frontier_on(cells, axis)])),
                 "saturating": (lambda sf: {"l_inf": sf.l_inf, "a": sf.a,
                                            "exponent": sf.b, "r2": sf.r2,
                                            "points": sf.n,
                                            "at_boundary": sf.at_boundary}
                                if sf else None)(
                     fit_saturating([(c, l) for c, l, _ in
                                     frontier_on(cells, axis)]))}
                for name, cells in by_family(series).items()
            ],
        }

    axes = {axis: _axis_summary(axis) for axis in Y_AXES}
    loss = axes["loss"]
    fit = fit_power_law([(c, l) for c, l, _ in frontier_on(series, "loss")])
    summary = {
        "cells": [
            {"cell": s.cell, "encoder": s.encoder, "family": family_of(s.encoder),
             "params": s.params,
             "frozen": s.frozen, "comparable": s.comparable,
             "points": len(s.points),
             "best_valid_loss": (s.best().valid_loss if s.best() else None),
             "pf_days_total": (s.points[-1].pf_days if s.points else None)}
            for s in series
        ],
        # Kept at the top level under their historical names so anything
        # already reading this file still finds them; `axes` is the general
        # form and carries the same numbers for every y-axis.
        "frontier": [{"pf_days": e["pf_days"], "valid_loss": e["value"],
                      "cell": e["cell"]} for e in loss["frontier"]],
        "power_law": loss["power_law"],
        "saturating": loss["saturating"],
        "families": loss["families"],
        "axes": axes,
    }
    (out_dir / "scaling_points.json").write_text(json.dumps(summary, indent=2))
    print(table)
    print(_markdown_table(_family_rows(series)))
    if fit:
        print(f"[scaling] frontier fit: L = {fit[0]:.4g} * C^{fit[1]:.4f}  "
              f"(R^2={fit[2]:.3f})")
    else:
        print("[scaling] frontier fit: not enough points (need 3)")
    sat = loss["saturating"]
    if sat:
        print(f"[scaling] with floor:   L = {sat['l_inf']:.4g} + "
              f"{sat['a']:.4g} * C^{sat['exponent']:.4f}  "
              f"(R^2={sat['r2']:.3f}, linear L"
              + (", floor unconstrained" if sat["at_boundary"] else "") + ")")
    for fam in loss["families"]:
        pl = fam["power_law"]
        if pl:
            print(f"[scaling] family {fam['family']}: b={pl['exponent']:.4f} "
                  f"(R^2={pl['r2']:.3f}, {len(fam['sizes'])} sizes)")
    for name in ("scaling_table.md", "scaling_report.html", "scaling_points.json"):
        print(f"[scaling] wrote {out_dir / name}")
    return summary


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", default="latest",
                   help="Run directory to analyse ('latest', a timestamp, or "
                        "a path).")
    p.add_argument("--results-root", default="benchmark_scaling/results/chembl",
                   help="Where scaling run directories live.")
    p.add_argument("--chart-backend", default="auto",
                   choices=["auto", "plotly", "matplotlib", "svg"])


def run_from_args(args: argparse.Namespace) -> None:
    out_dir = resolve_run_dir(args.results_root, args.out_dir)
    build(out_dir, backend=args.chart_backend)
    RunLogger(out_dir).record_command(note="scaling report")
