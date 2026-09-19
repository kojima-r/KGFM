#!/usr/bin/env python
"""Dataset-size scaling report: how loss falls as UNIQUE training data grows.

`kgfm scaling` plots loss against *compute* with one line per model size. This
plots loss against the number of *unique training rows*, with the model held
fixed — a different question that needs a different plot, which is why it is
its own script rather than a flag on that one.

    python benchmark_scaling_data/report_data.py --out-dir latest
    python benchmark_scaling_data/report_data.py --out-dir latest --charts matplotlib

WHY THE SHAPE OF THE PLOT IS DIFFERENT
In the model-size study each cell contributes a whole *curve* (its training
trajectory is a sweep over compute) and the law is fitted to the lower
envelope. Here D is a property of the cell, not of the point: every validation
measurement in a cell shares one D. So the law is fitted to **one point per
cell** — its best validation loss — and the trajectory is used for something
else: to show that small-D cells turn upward while large-D cells do not.

    L(D) = L_inf + a * D^b

`best valid loss` is the right estimator for L(D) precisely because a
finite-data run overfits: the final loss measures how long it was left running
past its optimum, whereas the minimum measures what that much data can buy.
This is the opposite of the model-size study, where taking a minimum over a
two-phase curve is the thing that goes wrong — there the curve rises and comes
back down, so a *stopping rule* truncates it; here we are not stopping
anything, just reading the best point off a finished run.

Nothing here re-runs training: it reads the same `cell_*.log` files
`kgfm report` parses, so the report can be regenerated for a finished run.
D itself comes from the `[init] data cap:` line plus `kgfm.data.count_rows`,
which is cached on disk — so resolving D is instant on a corpus already
counted and never guessed from the file count.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgfm import charts as charts_mod                      # noqa: E402
from kgfm.data import count_rows, read_file_list           # noqa: E402
from kgfm.encoders import is_transformer                   # noqa: E402
from kgfm.report import TrainingCurve, discover_curves     # noqa: E402
from kgfm.report_html import _CSS, _kv_grid                # noqa: E402
from kgfm.runs import RunLogger, resolve_run_dir           # noqa: E402
from kgfm.scaling.palette import by_size                   # noqa: E402
from kgfm.scaling.points import (MIN_SATURATING_POINTS,    # noqa: E402
                                 SaturatingFit, fit_power_law,
                                 fit_saturating)

DEFAULT_RESULTS_ROOT = "benchmark_scaling_data/results/chembl"

# Lower is better on both, which is what lets one fitting routine serve both.
Y_AXES: Dict[str, Tuple[str, str]] = {
    "loss": (
        "best validation loss",
        "The training objective on held-out batches, at the training batch "
        "size. Bounded above by ln(B) and floored by repeated tails, so it "
        "has a limited range to move in.",
    ),
    "mrr_error": (
        "1 - validation MRR",
        "Ranking error against a fixed-size candidate pool. No ceiling from "
        "the objective, so on the model-size axis it revealed an exponent "
        "8x steeper than the loss axis did. Reported here for the same "
        "reason.",
    ),
}


# The x-axis. `rows` is the natural unit for this knob (`max_rows_per_file`
# counts rows); `tokens` is the unit scaling-law papers plot against, so both
# are reported. They are NOT simply proportional: rows differ in length, and
# because the cap takes each file's *first* D rows the cells see different
# text, so measured tokens/example ranges 240-303 across cells here.
X_AXES: Dict[str, Tuple[str, str]] = {
    "rows": (
        "unique training rows (D)",
        "The knob itself: sum over train files of min(rows, "
        "max_rows_per_file).",
    ),
    "tokens": (
        "unique training tokens",
        "The same pool measured in tokens — the convention in the scaling-law "
        "literature. Counted, not assumed: the trainer logs cumulative padded "
        "tokens through the encoder, and unique tokens is that rate times the "
        "pool size.",
    ),
}


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

@dataclass
class DataCell:
    """One training run, identified by how much unique data it could reach."""

    tag: str
    encoder: str
    params: Optional[int]
    unique_rows: Optional[int]        # D — the x-axis
    max_rows_per_file: Optional[int]  # the knob that produced D
    train_files: Optional[int]
    global_batch_size: int
    steps_run: int
    # (step, cumulative padded tokens through the encoder, all ranks). The
    # trainer logs this per rank; `collect` multiplies by world_size.
    tokens: List[Tuple[int, int]] = field(default_factory=list)
    # True when a "token" means a subword token. False for ngram, whose
    # counter records n-gram lookups instead — a different unit that must not
    # share an axis with transformer tokens (see encoders.py).
    subword_tokens: bool = True
    # (step, value) histories straight from the log.
    train_losses: List[Tuple[int, float]] = field(default_factory=list)
    valid_losses: List[Tuple[int, float]] = field(default_factory=list)
    valid_mrr: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def tokens_per_example(self) -> Optional[float]:
        """Padded tokens the encoder sees per training triple.

        Measured from the run rather than derived from `3 * max_length`: the
        tokenizer pads to the longest sequence in the batch, so the real rate
        is well below the cap and differs between cells that read different
        parts of the corpus.
        """
        if not self.tokens or not self.global_batch_size:
            return None
        step, tok = self.tokens[-1]
        return tok / (step * self.global_batch_size) if step else None

    @property
    def unique_tokens(self) -> Optional[int]:
        """The x-axis in token units: pool size times the measured rate."""
        rate = self.tokens_per_example
        if rate is None or not self.unique_rows:
            return None
        return int(self.unique_rows * rate)

    @property
    def tokens_processed(self) -> Optional[int]:
        """Cumulative tokens the run actually pushed, repetition included.

        Distinct from `unique_tokens`: a cell that revisits its pool 100 times
        processes 100x its dataset. Plotting against this is the Kaplan-style
        view; plotting against `unique_tokens` is the data-scaling view.
        """
        return self.tokens[-1][1] if self.tokens else None

    def x_value(self, kind: str) -> Optional[float]:
        v = self.unique_rows if kind == "rows" else self.unique_tokens
        return None if not v else float(v)

    def tokens_at(self, step: int) -> Optional[int]:
        """Cumulative tokens at (or just before) ``step``.

        Nearest earlier sample rather than interpolation: the counter is
        monotone and logged often, so this is honest about the resolution
        actually available.
        """
        prior = [t for s, t in self.tokens if s <= step]
        if prior:
            return prior[-1]
        return self.tokens[0][1] if self.tokens else None

    @property
    def examples_seen(self) -> int:
        return self.steps_run * self.global_batch_size

    @property
    def epochs(self) -> Optional[float]:
        """Passes over the unique pool by the end of the run.

        The number that says whether a cell was in the repetition regime at
        all: below 1 it never saw a row twice, and no finite-data effect can
        have shown up yet whatever D says.
        """
        if not self.unique_rows:
            return None
        return self.examples_seen / self.unique_rows

    def best(self, axis: str) -> Optional[Tuple[int, float]]:
        """(step, value) of the best point on ``axis``. Lower is better."""
        pts = self.series(axis)
        return min(pts, key=lambda p: p[1]) if pts else None

    def series(self, axis: str) -> List[Tuple[int, float]]:
        if axis == "loss":
            return self.valid_losses
        if axis == "mrr_error":
            # Clamped away from 0 because the axis is fitted in log space.
            return [(s, max(1e-6, 1.0 - v)) for s, v in self.valid_mrr]
        return []

    def epochs_at(self, step: int) -> Optional[float]:
        if not self.unique_rows:
            return None
        return step * self.global_batch_size / self.unique_rows

    @property
    def label(self) -> str:
        return f"D={_rows(self.unique_rows)}"


def _rows(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.3g}{suffix}"
    return str(n)


def _params(n: Optional[int]) -> str:
    if not n:
        return "—"
    return (f"{n / 1e9:.1f}B" if n >= 1e9 else
            f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K")


# --------------------------------------------------------------------------
# collecting
# --------------------------------------------------------------------------

def _unique_rows(curve: TrainingCurve, train_list: str,
                 cache: Dict[Tuple[str, Optional[int]], Optional[int]]
                 ) -> Optional[int]:
    """D for one cell: sum over train files of min(rows(f), cap).

    Not `files x cap`: that is only right when every file is larger than the
    cap, which is true of ChEMBL and not true in general (biomodels files are
    ~1k rows). Counting is exact and cached on disk by `data.count_rows`, so
    the honest version costs nothing after the first run.
    """
    if curve.train_files is None:
        # Log predates the `[init] data cap:` line, so the cap is unknown —
        # and "unknown" is not "uncapped": reporting the full corpus for a run
        # that was actually capped would put the point at the wrong end of the
        # axis. Say nothing instead.
        return None
    key = (train_list, curve.max_rows_per_file)
    if key in cache:
        return cache[key]
    files = read_file_list(train_list) if Path(train_list).is_file() else []
    if not files:
        cache[key] = None
        return None
    counts = count_rows(files)
    cap = curve.max_rows_per_file
    total = sum(counts if cap is None else [min(c, cap) for c in counts])
    if curve.row_keep_prob and curve.row_keep_prob < 1.0:
        total = int(total * curve.row_keep_prob)
    cache[key] = total
    return total


def collect(out_dir: Path, train_list: str) -> List[DataCell]:
    """Every cell in a run directory, ordered by unique data size.

    A cell appears once even though it has one log per protocol: the protocols
    re-score the same checkpoint and share a training trajectory.
    """
    cache: Dict[Tuple[str, Optional[int]], Optional[int]] = {}
    seen: Dict[str, DataCell] = {}
    for curve in discover_curves(out_dir):
        if not curve.steps or not curve.valid_losses:
            continue
        _, _, tag = curve.cell.partition("_")   # cell_<protocol>_<tag>
        if tag in seen:
            continue
        seen[tag] = DataCell(
            tag=tag,
            encoder=curve.encoder_name or tag,
            params=curve.params_trainable or curve.params_total,
            unique_rows=_unique_rows(curve, train_list, cache),
            max_rows_per_file=curve.max_rows_per_file,
            train_files=curve.train_files,
            global_batch_size=curve.global_batch_size or 1,
            steps_run=max(curve.steps),
            # The trainer logs tokens per rank; the axis wants the total.
            tokens=[(st, tk * (curve.world_size or 1))
                    for st, tk in curve.tokens],
            subword_tokens=is_transformer(curve.encoder_name or ""),
            train_losses=list(zip(curve.steps, curve.losses)),
            valid_losses=list(curve.valid_losses),
            valid_mrr=list(curve.valid_metrics.get("MRR", [])),
        )
    cells = list(seen.values())
    # Unknown-D cells sort last; they cannot be plotted on the data axis but
    # still belong in the table so the run is fully accounted for.
    return sorted(cells, key=lambda c: (c.unique_rows is None,
                                        c.unique_rows or 0))


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

@dataclass
class DataFit:
    axis: str
    power: Optional[Tuple[float, float, float]]   # (a, b, r2), every cell
    saturating: Optional[SaturatingFit]
    points: List[Tuple[float, float, str]]        # (x, best value, tag)
    # The same plain fit over only the cells where D is the binding
    # constraint, i.e. those that completed at least one pass over their pool.
    power_bound: Optional[Tuple[float, float, float]] = None
    points_bound: List[Tuple[float, float, str]] = field(default_factory=list)
    # "rows" or "tokens". Only the coefficient differs between them when the
    # two are proportional; they are not exactly proportional here, so the
    # exponents differ slightly too.
    x_kind: str = "rows"
    # False when the x unit is ngram lookups rather than subword tokens.
    x_comparable: bool = True


def fit_axis(cells: Sequence[DataCell], axis: str,
             x_kind: str = "rows") -> DataFit:
    """Fit L(D) to one best point per cell.

    One point per cell, not one per validation measurement: within a cell every
    point shares the same D, so including them all would weight cells by how
    often they were validated and fit a vertical smear rather than a law.

    Three fits, because the honest exponent depends on which cells are
    measuring data at all:

    * ``power`` over every cell — comparable to whatever else quotes a plain
      ``a*D^b``, and the number a flat tail spoils.
    * ``saturating`` over every cell — the floored form, which is what a
      bending frontier needs; its ``at_boundary`` flag says when the floor is
      pinned to ``min(L)`` and ``b`` is only a lower bound.
    * ``power_bound`` over cells with at least one pass — the **primary**
      number. A cell that never finished a pass had its step budget bind
      instead of D, so it sits at whatever the budget buys and drags the
      exponent flat. Measured on the two production sweeps, dropping those
      cells roughly *doubled* |b| and took R² from 0.58/0.71 to 0.98/0.96.
    """
    pts: List[Tuple[float, float, str]] = []
    bound: List[Tuple[float, float, str]] = []
    for c in cells:
        best = c.best(axis)
        x = c.x_value(x_kind)
        if best and x:
            pts.append((x, best[1], c.tag))
            if (c.epochs or 0) >= 1.0:
                bound.append((x, best[1], c.tag))
    xy = [(d, v) for d, v, _ in pts]
    return DataFit(
        axis=axis,
        power=fit_power_law(xy),
        saturating=fit_saturating(xy),
        points=pts,
        power_bound=fit_power_law([(d, v) for d, v, _ in bound]),
        points_bound=bound,
        x_kind=x_kind,
        x_comparable=(x_kind != "tokens"
                      or all(c.subword_tokens for c in cells)),
    )


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------

def _colors(cells: Sequence[DataCell]) -> List[str]:
    """Viridis by log10(D) — hue means data size, in every figure."""
    return by_size([float(c.unique_rows or 0) for c in cells])


def _fit_curve(fit: DataFit, n: int = 60
               ) -> List[Tuple[float, float]]:
    """The fitted L(D) sampled for drawing, in log10-x coordinates."""
    if not fit.points:
        return []
    xs = [d for d, _, _ in fit.points]
    lo, hi = math.log10(min(xs)), math.log10(max(xs))
    if hi <= lo:
        return []
    out = []
    for i in range(n + 1):
        lx = lo + (hi - lo) * i / n
        d = 10.0 ** lx
        if fit.saturating is not None:
            s = fit.saturating
            y = s.l_inf + s.a * d ** s.b
        elif fit.power is not None:
            a, b, _ = fit.power
            y = a * d ** b
        else:
            return []
        if y > 0:
            out.append((lx, math.log10(y)))
    return out


def plot_law(cells: Sequence[DataCell], fit: DataFit, backend: str) -> str:
    """The headline figure: best loss vs unique data, log-log, with the fit."""
    if not fit.points:
        return '<p class="note">no cell has a known unique-data size</p>'
    measured = [(math.log10(d), math.log10(v)) for d, v, _ in fit.points
                if v > 0]
    series = [("measured (best per cell)", measured)]
    colors = ["#2563eb"]
    curve = _fit_curve(fit)
    if curve:
        label = "fit"
        if fit.saturating is not None:
            label = f"fit  L∞ + a·D^{fit.saturating.b:.3f}"
        elif fit.power is not None:
            label = f"fit  a·D^{fit.power[1]:.3f}"
        series.append((label, curve))
        colors.append("#111827")
    label, _ = Y_AXES[fit.axis]
    xlabel, _ = X_AXES[fit.x_kind]
    return charts_mod.chart(
        series,
        x_label=f"log10 {xlabel}",
        y_label=f"log10 {label}",
        backend=backend,
        colors=colors,
        height=380,
    )


def plot_trajectories(cells: Sequence[DataCell], backend: str,
                      axis: str = "loss") -> str:
    """Validation trajectories vs step, coloured by D.

    This is the figure that shows *why* L(D) looks the way it does: a cell with
    too little data bottoms out early and then climbs, and the climb is
    steeper the smaller D is. On the data axis that upturn is the signal, not
    an artifact to be stopped out.
    """
    usable = [c for c in cells if c.series(axis)]
    if not usable:
        return '<p class="note">no validation history</p>'
    series = [(c.label, [(float(s), v) for s, v in c.series(axis)])
              for c in usable]
    label, _ = Y_AXES[axis]
    return charts_mod.chart(
        series, x_label="optimizer step", y_label=label,
        backend=backend, colors=_colors(usable), height=380,
    )


def plot_vs_epochs(cells: Sequence[DataCell], backend: str) -> str:
    """The same trajectories against *passes over the pool*.

    Step and epoch are the same axis rescaled per cell, and the rescaling is
    the point: if the cells' minima line up here but not on the step axis, the
    optimum is set by how many times the data was reused rather than by how
    long the run was.
    """
    usable = [c for c in cells if c.valid_losses and c.unique_rows]
    if not usable:
        return '<p class="note">no validation history with a known D</p>'
    series = []
    for c in usable:
        pts = [(c.epochs_at(s), v) for s, v in c.valid_losses]
        pts = [(math.log10(e), v) for e, v in pts if e and e > 0]
        if pts:
            series.append((c.label, pts))
    if not series:
        return '<p class="note">no positive epoch values</p>'
    return charts_mod.chart(
        series, x_label="log10 passes over the unique pool",
        y_label="validation loss", backend=backend,
        colors=_colors(usable), height=380,
    )


def plot_vs_tokens_processed(cells: Sequence[DataCell], backend: str,
                             axis: str = "loss") -> str:
    """Validation curve against cumulative tokens processed, coloured by D.

    The Kaplan-style figure, and the one that shows what a *data-constrained*
    run does to it: every cell pushes the same number of tokens, so the lines
    end at the same x, and they separate purely by how much unique data those
    tokens came from. A small-D cell spends most of its budget re-reading the
    same tokens and turns upward; a large-D cell is still seeing new ones.
    """
    usable = [c for c in cells if c.series(axis) and c.tokens]
    if not usable:
        return '<p class="note">no token counts logged</p>'
    series = []
    for c in usable:
        pts = []
        for st, v in c.series(axis):
            tk = c.tokens_at(st)
            if tk and tk > 0:
                pts.append((math.log10(tk), v))
        if pts:
            series.append((c.label, pts))
    if not series:
        return '<p class="note">no positive token counts</p>'
    label, _ = Y_AXES[axis]
    return charts_mod.chart(
        series, x_label="log10 training tokens processed (cumulative)",
        y_label=label, backend=backend,
        colors=_colors(usable), height=380,
    )


def plot_gap(cells: Sequence[DataCell], backend: str) -> str:
    """Generalisation gap (valid - train) at the end of each run, vs D.

    The mechanism behind L(D) stated directly: less unique data, more of the
    loss explained by memorising it.
    """
    pts = []
    for c in cells:
        if not (c.unique_rows and c.valid_losses and c.train_losses):
            continue
        vstep, vloss = c.valid_losses[-1]
        prior = [l for s, l in c.train_losses if s <= vstep]
        if prior:
            pts.append((math.log10(c.unique_rows), vloss - prior[-1]))
    if not pts:
        return '<p class="note">not enough history</p>'
    return charts_mod.chart(
        [("valid - train, final eval", sorted(pts))],
        x_label="log10 unique training rows (D)",
        y_label="generalisation gap (nats)",
        backend=backend, colors=["#dc2626"], height=320,
    )


# --------------------------------------------------------------------------
# tables / text
# --------------------------------------------------------------------------

def build_rows(cells: Sequence[DataCell]) -> List[Dict[str, str]]:
    rows = []
    for c in cells:
        best_loss = c.best("loss")
        best_err = c.best("mrr_error")
        ep = c.epochs
        rows.append({
            "Cell": c.tag,
            "max_rows_per_file": ("all" if c.max_rows_per_file is None
                                  else f"{c.max_rows_per_file:,}"),
            "Unique rows (D)": (f"{c.unique_rows:,}" if c.unique_rows
                                else "unknown"),
            "Unique tokens": (f"{c.unique_tokens:,}" if c.unique_tokens
                              else "—"),
            "Tok/example": (f"{c.tokens_per_example:.1f}"
                            if c.tokens_per_example else "—"),
            "Tokens processed": (f"{c.tokens_processed:,}"
                                 if c.tokens_processed else "—"),
            "Examples seen": f"{c.examples_seen:,}",
            "Passes over D": f"{ep:.2f}" if ep else "—",
            "Best valid loss": f"{best_loss[1]:.4f}" if best_loss else "—",
            "@step": f"{best_loss[0]:,}" if best_loss else "—",
            "Best MRR": (f"{1.0 - best_err[1]:.4f}" if best_err else "—"),
        })
    return rows


def _table_html(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return '<p class="note">no cells</p>'
    heads = list(rows[0])
    out = ["<table><thead><tr>"]
    out += [f"<th>{html.escape(h)}</th>" for h in heads]
    out.append("</tr></thead><tbody>")
    for r in rows:
        out.append("<tr>" + "".join(
            f"<td>{html.escape(str(r.get(h, '—')))}</td>" for h in heads)
            + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def table_md(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_no cells_\n"
    heads = list(rows[0])
    lines = ["| " + " | ".join(heads) + " |",
             "|" + "|".join("---" for _ in heads) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "—")) for h in heads) + " |")
    return "\n".join(lines) + "\n"


def fit_html(fit: DataFit) -> str:
    label, why = Y_AXES[fit.axis]
    xlabel, xwhy = X_AXES[fit.x_kind]
    sym = "D" if fit.x_kind == "rows" else "T"
    out = [f"<p class=\"note\">{html.escape(why)}</p>",
           f"<p class=\"note\">x = {html.escape(xlabel)}. "
           f"{html.escape(xwhy)}</p>"]
    if not fit.x_comparable:
        out.append(
            '<p class="note"><strong>Unit warning:</strong> this run uses the '
            'ngram encoder, whose token counter records n-gram lookups, not '
            'subword tokens. The exponent is meaningful within this run, but '
            'the x values are not comparable with a transformer run\u2019s.</p>')
    if fit.power_bound and len(fit.points_bound) < len(fit.points):
        a, b, r2 = fit.power_bound
        out.append(
            f"<p><strong>{html.escape(label)} = {a:.4g} · "
            f"{sym}<sup>{b:.4f}</sup></strong> (R² = {r2:.3f}) — over the "
            f"{len(fit.points_bound)} cells where <strong>D actually "
            f"binds</strong> (at least one pass over the pool). "
            f"<strong>This is the exponent to quote.</strong></p>")
    if fit.power:
        a, b, r2 = fit.power
        extra = (" — including cells whose step budget bound instead of D, "
                 "which flattens it"
                 if fit.power_bound and len(fit.points_bound) < len(fit.points)
                 else "")
        out.append(f"<p>{html.escape(label)} = "
                   f"{a:.4g} · {sym}<sup>{b:.4f}</sup> "
                   f"(R² = {r2:.3f}, all {len(fit.points)} cells){extra}</p>")
    else:
        out.append('<p class="note">plain power law needs 3 cells with a '
                   'known D.</p>')
    if fit.saturating:
        sf = fit.saturating
        bound = " <em>(unconstrained — read |b| as a lower bound)</em>" \
            if sf.at_boundary else ""
        out.append(f"<p><strong>{html.escape(label)} = {sf.l_inf:.4g} + "
                   f"{sf.a:.3g} · {sym}<sup>{sf.b:.4f}</sup></strong> "
                   f"(R² = {sf.r2:.3f}){bound}</p>")
    else:
        out.append(f'<p class="note">the floored form '
                   f'L∞ + a·{sym}^b needs {MIN_SATURATING_POINTS} cells to be '
                   f'identifiable ({len(fit.points)} here), so it is not '
                   f'fitted — with fewer points L∞ trades off against a '
                   f'almost exactly.</p>')
    return "".join(out)


def regime_note(cells: Sequence[DataCell]) -> str:
    """State plainly which cells actually repeated their data.

    A data-scaling law only means something over cells that reached the
    repetition regime; below one pass, D is not binding and the cell is
    measuring the compute budget instead. Recomputed from the run rather than
    asserted, because it depends on max_steps and batch size.
    """
    known = [c for c in cells if c.epochs is not None]
    if not known:
        return ('<p class="note">No cell has a known unique-data size, so the '
                'regime cannot be determined. Logs written before the '
                '<code>[init] data cap:</code> line lack it.</p>')
    repeated = [c for c in known if (c.epochs or 0) >= 1.0]
    single = [c for c in known if (c.epochs or 0) < 1.0]
    parts = []
    if repeated:
        worst = max(repeated, key=lambda c: c.epochs or 0)
        parts.append(
            f"<p><strong>{len(repeated)} of {len(known)} cells reached the "
            f"repetition regime</strong> (at least one full pass over their "
            f"pool); the most-repeated saw its data "
            f"{worst.epochs:.1f}x ({worst.label}).</p>")
    if single:
        biggest = max(single, key=lambda c: c.unique_rows or 0)
        parts.append(
            f"<p><strong>{len(single)} cell(s) never completed one pass</strong> "
            f"(smallest fraction at {biggest.label}, "
            f"{biggest.epochs:.3f} passes). For those, D is not the binding "
            f"constraint — the step budget is — so they measure compute, not "
            f"data, and a law fitted through them flattens. Either raise "
            f"<code>max_steps</code> or lower those "
            f"<code>data_sizes</code> entries.</p>")
    if not single:
        parts.append('<p class="note">Every cell repeated its data at least '
                     'once, so D is binding throughout and the fit is a '
                     'statement about data rather than about the budget.</p>')
    return "".join(parts)


def overfit_note(cells: Sequence[DataCell]) -> str:
    """Where the best point landed, per cell — interior vs at the end.

    The data-axis analogue of the model study's budget-binding caveat. A best
    loss at the last measurement is a lower bound, not an optimum.
    """
    rows = []
    for c in cells:
        best = c.best("loss")
        if not best or not c.valid_losses:
            continue
        last_step = c.valid_losses[-1][0]
        interior = best[0] < last_step
        final = c.valid_losses[-1][1]
        rows.append({
            "Cell": c.tag,
            "D": _rows(c.unique_rows),
            "Best loss": f"{best[1]:.4f}",
            "@step": f"{best[0]:,}",
            "Final loss": f"{final:.4f}",
            "Rise after best": f"{final - best[1]:+.4f}",
            "Minimum": "interior" if interior else "at the end (lower bound)",
        })
    if not rows:
        return ""
    at_end = sum(1 for r in rows if r["Minimum"].startswith("at the end"))
    note = (f'<p class="note">{at_end} of {len(rows)} cells put their best '
            f'validation loss at the last measurement, so those numbers are '
            f'lower bounds — the run was still improving when the budget ran '
            f'out. A cell with an interior minimum and a positive rise after '
            f'it is one that ran out of *data*, not of steps, and that is the '
            f'effect this study measures.</p>')
    return _table_html(rows) + note


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------

def settings_html(out_dir: Path, cells: Sequence[DataCell],
                  train_list: str) -> str:
    meta = {}
    meta_path = out_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
    params = meta.get("params", {}) or {}
    first = cells[0] if cells else None
    grid = [
        ("run", out_dir.name),
        ("host / git", f"{meta.get('host', '?')} / "
                       f"{str(meta.get('git_rev', '?'))[:8]}"),
        ("config", params.get("config_file", "?")),
        ("train list", train_list),
        ("encoder (held fixed)",
         f"{first.encoder} ({_params(first.params)})" if first else "?"),
        ("data_sizes (the axis)",
         ", ".join(str(d) for d in (params.get("data_sizes") or [])) or "?"),
        ("max_steps", params.get("max_steps", "?")),
        ("batch size", first.global_batch_size if first else "?"),
        ("examples per cell", f"{first.examples_seen:,}" if first else "?"),
        ("eval every", f"{params.get('eval_every', '?')} steps"),
        ("valid-loss batches", params.get("valid_loss_batches", "?")),
        ("candidate pool / eval triples",
         f"{params.get('pool_size', '?')} / "
         f"{params.get('n_eval_triples', '?')}"),
    ]
    note = (
        '<p class="note">Every cell is the <em>same model</em> trained for the '
        '<em>same number of steps</em> on the <em>same number of files</em>. '
        'The only thing that varies is <code>max_rows_per_file</code>, which '
        'caps each file independently — so the number of entity populations '
        'in the stream is constant and only the rows per population change. '
        'Unique rows are counted, not derived from files x cap: that product '
        'is only right when every file is larger than the cap.</p>'
    )
    return (f"<h2>Experiment settings</h2>{_kv_grid(grid)}{note}")


def render(out_dir: Path, cells: Sequence[DataCell], train_list: str,
           backend: str) -> Tuple[str, Dict[str, DataFit]]:
    fits = {axis: fit_axis(cells, axis) for axis in Y_AXES}
    tok_fits = {axis: fit_axis(cells, axis, "tokens") for axis in Y_AXES}
    has_mrr = bool(fits["mrr_error"].points)
    has_tokens = bool(tok_fits["loss"].points)

    # The plotly backend inlines its library into whichever figure renders
    # FIRST and has later figures reference it, so figures must be rendered in
    # document order. Hence the thunks.
    blocks: List[Tuple[str, object]] = [
        ("<h2>The law: best loss vs unique data</h2>", None),
        ("", lambda: plot_law(cells, fits["loss"], backend)),
        (fit_html(fits["loss"]), None),
        (regime_note(cells), None),
    ]
    if has_mrr:
        blocks += [
            ("<h2>Same cells, ranking-error axis</h2>", None),
            ("", lambda: plot_law(cells, fits["mrr_error"], backend)),
            (fit_html(fits["mrr_error"]), None),
        ]
    if has_tokens:
        blocks += [
            ("<h2>The law in token units</h2>", None),
            ('<p class="note">The same cells against unique training '
             '<em>tokens</em>, which is how scaling laws are conventionally '
             'plotted. Not a relabelling of the rows axis: rows differ in '
             'length and each cell reads a different part of the corpus, so '
             'the measured tokens-per-example is not constant across '
             'cells.</p>', None),
            ("", lambda: plot_law(cells, tok_fits["loss"], backend)),
            (fit_html(tok_fits["loss"]), None),
        ]
        if has_mrr:
            blocks += [
                ("<h3>Ranking error vs unique tokens</h3>", None),
                ("", lambda: plot_law(cells, tok_fits["mrr_error"], backend)),
                (fit_html(tok_fits["mrr_error"]), None),
            ]
        blocks += [
            ("<h3>Validation loss vs training tokens processed</h3>", None),
            ('<p class="note">Cumulative tokens actually pushed, repetition '
             'included. Every cell pushes the same total, so the lines end at '
             'the same x and separate only by how much unique data those '
             'tokens came from — which is the data-constrained version of the '
             'usual loss-vs-tokens figure.</p>', None),
            ("", lambda: plot_vs_tokens_processed(cells, backend, "loss")),
        ]
    blocks += [
        ("<h2>Trajectories — why the law bends</h2>", None),
        ('<p class="note">Validation loss against step, coloured by D '
         '(viridis, darker = less data). A cell with too little data bottoms '
         'out and then climbs; the climb is the finite-data effect.</p>', None),
        ("", lambda: plot_trajectories(cells, backend, "loss")),
        ('<p class="note">The same runs against passes over their own pool. If '
         'the minima line up on this axis but not on the step axis, the '
         'optimum is set by reuse rather than by run length.</p>', None),
        ("", lambda: plot_vs_epochs(cells, backend)),
        ("<h2>Generalisation gap vs unique data</h2>", None),
        ("", lambda: plot_gap(cells, backend)),
        ("<h2>Per cell</h2>", None),
        (f'<div class="scroll">{_table_html(build_rows(cells))}</div>', None),
        ("<h3>Where each minimum landed</h3>", None),
        (f'<div class="scroll">{overfit_note(cells)}</div>', None),
    ]

    body = [settings_html(out_dir, cells, train_list)]
    for static, thunk in blocks:
        if static:
            body.append(static)
        if thunk is not None:
            body.append(f'<div class="charts wide">{thunk()}</div>')

    fits.update({f"{k}__tokens": v for k, v in tok_fits.items()})
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>kgfm data scaling — {html.escape(out_dir.name)}</title>'
        f"<style>{_CSS}</style></head><body>"
        f"<h1>Dataset-size scaling</h1>{''.join(body)}</body></html>"
    )
    return page, fits


def fits_json(cells: Sequence[DataCell],
              fits: Dict[str, DataFit]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "cells": [{
            "tag": c.tag,
            "encoder": c.encoder,
            "params": c.params,
            "max_rows_per_file": c.max_rows_per_file,
            "unique_rows": c.unique_rows,
            "unique_tokens": c.unique_tokens,
            "tokens_per_example": c.tokens_per_example,
            "tokens_processed": c.tokens_processed,
            "subword_tokens": c.subword_tokens,
            "train_files": c.train_files,
            "global_batch_size": c.global_batch_size,
            "steps_run": c.steps_run,
            "examples_seen": c.examples_seen,
            "passes_over_pool": c.epochs,
            "best_valid_loss": (c.best("loss") or (None, None))[1],
            "best_valid_loss_step": (c.best("loss") or (None, None))[0],
            "best_mrr": (None if not c.best("mrr_error")
                         else 1.0 - c.best("mrr_error")[1]),
            "valid_losses": c.valid_losses,
            "valid_mrr": c.valid_mrr,
        } for c in cells],
        "axes": {},
    }
    for axis, fit in fits.items():
        entry: Dict[str, object] = {
            "x_kind": fit.x_kind,
            "x_comparable": fit.x_comparable,
            "points": [{"x": d, "value": v, "cell": t}
                       for d, v, t in fit.points],
            "cells_where_d_binds": [t for _, _, t in fit.points_bound],
        }
        if fit.power:
            a, b, r2 = fit.power
            entry["power_law"] = {"a": a, "b": b, "r2": r2}
        if fit.power_bound:
            a, b, r2 = fit.power_bound
            entry["power_law_d_binds"] = {"a": a, "b": b, "r2": r2,
                                          "n": len(fit.points_bound)}
        if fit.saturating:
            s = fit.saturating
            entry["saturating"] = {"l_inf": s.l_inf, "a": s.a, "b": s.b,
                                   "r2": s.r2, "at_boundary": s.at_boundary}
        out["axes"][axis] = entry            # type: ignore[index]
    return out


def build(out_dir: Path, train_list: str, backend: str = "auto"
          ) -> Dict[str, object]:
    cells = collect(out_dir, train_list)
    if not cells:
        raise SystemExit(
            f"No usable training logs in {out_dir}.\n"
            "This report needs cell_*.log files with validation losses — run "
            "the sweep with valid_loss_batches > 0."
        )
    page, fits = render(out_dir, cells, train_list, backend)
    rows = build_rows(cells)

    (out_dir / "data_scaling_report.html").write_text(page, encoding="utf-8")
    (out_dir / "data_scaling_table.md").write_text(table_md(rows),
                                                   encoding="utf-8")
    payload = fits_json(cells, fits)
    (out_dir / "data_scaling_points.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    print(table_md(rows))
    for axis, fit in fits.items():
        if not fit.points:
            continue
        # `axis` is the dict key, which carries a `__tokens` suffix for the
        # token-axis fits; the y-axis name lives on the fit itself.
        label, _ = Y_AXES[fit.axis]
        sym = "D" if fit.x_kind == "rows" else "T"
        label = f"{label} [x={fit.x_kind}]"
        if fit.power_bound:
            a, b, r2 = fit.power_bound
            print(f"{label}: {a:.4g} * {sym}^{b:.4f}  (R2={r2:.3f}, "
                  f"{len(fit.points_bound)} cells where D binds)  <-- quote this")
        if fit.power:
            a, b, r2 = fit.power
            print(f"{label}: {a:.4g} * {sym}^{b:.4f}  (R2={r2:.3f}, "
                  f"all {len(fit.points)} cells)")
        if fit.saturating:
            sf = fit.saturating
            edge = "  (unconstrained)" if sf.at_boundary else ""
            print(f"{label}: {sf.l_inf:.4g} + {sf.a:.3g} * {sym}^{sf.b:.4f}  "
                  f"(R2={sf.r2:.3f}){edge}")
    print(f"\nwrote {out_dir}/data_scaling_{{report.html,table.md,points.json}}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="report_data.py",
        description="Dataset-size scaling report for a benchmark_scaling_data "
                    "run.",
    )
    p.add_argument("--out-dir", default="latest",
                   help="Run directory, a timestamp, or 'latest'.")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                   help=f"Where runs live (default {DEFAULT_RESULTS_ROOT}).")
    p.add_argument("--train-list", default="list_chembl/train.txt",
                   help="Train list the run used; its row counts turn "
                        "max_rows_per_file into a unique-row count.")
    p.add_argument("--charts", default="auto",
                   choices=["auto", "plotly", "matplotlib", "svg"],
                   help="Chart backend (default auto).")
    args = p.parse_args(argv)

    out_dir = resolve_run_dir(args.results_root, args.out_dir)
    # Recorded like every other command that writes into a run directory, so
    # the run's provenance stays complete.
    try:
        RunLogger(out_dir).record_command(tag="data-scaling")
    except Exception:                                   # pragma: no cover
        pass
    build(out_dir, args.train_list, args.charts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
