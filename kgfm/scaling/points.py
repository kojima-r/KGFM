"""Turning per-cell training logs into scaling-law data points.

One point is *one validation measurement*: a (compute, loss) pair for a given
model size. A whole training run therefore contributes a whole curve, which is
exactly the shape a Kaplan-style plot wants — each line is one model, traced
along its own training trajectory, and the frontier is the lower envelope over
all of them.

Nothing here re-runs anything: it reads the same `cell_*.log` files that
`kgfm report` parses, so a scaling report can be produced for a run that
finished weeks ago, and adding a model size to the study is just another cell
in the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..encoders import is_transformer
from ..report import TrainingCurve, discover_curves
from .compute import CellCompute, encoder_params, head_params, vectors_per_step


@dataclass
class ScalingPoint:
    """One validation measurement, in scaling-law coordinates."""

    cell: str
    step: int
    pf_days: float
    valid_loss: float
    params: int              # trainable parameters — what "model size" means here
    tokens: int              # cumulative encoder tokens, all ranks
    examples: int            # cumulative training triples seen
    comparable: bool         # False when FLOPs are not meaningful (ngram)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingSeries:
    """Every point for one cell, plus what identifies it."""

    cell: str
    encoder: str
    params: int
    frozen: bool
    comparable: bool
    points: List[ScalingPoint] = field(default_factory=list)
    # (step, training loss) straight from the log. Carried alongside the
    # validation points because the *contrast* between the two is what shows
    # that a mid-run rise in validation loss is not the end of training.
    train_losses: List[Tuple[int, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        n = self.params
        size = (f"{n/1e9:.1f}B" if n >= 1e9 else
                f"{n/1e6:.0f}M" if n >= 1e6 else f"{n/1e3:.0f}K")
        suffix = ", frozen" if self.frozen else ""
        return f"{self.encoder} ({size}{suffix})"

    def best(self) -> Optional[ScalingPoint]:
        """The lowest validation loss this cell reached."""
        return min(self.points, key=lambda p: p.valid_loss) if self.points else None


def _tokens_at(curve: TrainingCurve, step: int) -> Optional[int]:
    """Cumulative tokens at (or just before) ``step``, per rank.

    Interpolation is deliberately avoided: the counter is monotone and logged
    often, so the nearest earlier sample is both correct-in-spirit and honest
    about the resolution actually available.
    """
    prior = [(s, t) for s, t in curve.tokens if s <= step]
    if prior:
        return prior[-1][1]
    return curve.tokens[0][1] if curve.tokens else None


def series_from_curve(curve: TrainingCurve, frozen: bool) -> Optional[ScalingSeries]:
    """Build a cell's scaling series, or None if the log lacks what it needs."""
    if not curve.valid_losses or curve.params_total is None:
        return None
    world = curve.world_size or 1
    global_bs = curve.global_batch_size or 0
    if not global_bs:
        return None

    head = head_params(curve.params_total, curve.params_trainable or 0, frozen)
    profile = CellCompute(
        encoder=curve.encoder_name or curve.cell,
        encoder_params=encoder_params(curve.params_total, head),
        head_params=head,
        frozen=frozen,
        is_transformer=is_transformer(curve.encoder_name or ""),
    )

    series = ScalingSeries(
        cell=curve.cell,
        encoder=curve.encoder_name or curve.cell,
        params=profile.trainable_params,
        frozen=frozen,
        comparable=profile.comparable,
        train_losses=list(zip(curve.steps, curve.losses)),
    )
    by_step: Dict[int, Dict[str, float]] = {}
    for name, pts in curve.valid_metrics.items():
        for st, v in pts:
            by_step.setdefault(st, {})[name] = v

    for step, vloss in curve.valid_losses:
        tokens_rank = _tokens_at(curve, step)
        if tokens_rank is None:
            # Logs written before the trainer counted tokens. Fall back to the
            # examples axis and mark the cell incomparable rather than invent a
            # sequence length — a guessed T silently rescales the whole x-axis.
            tokens_total = 0
            comparable = False
        else:
            tokens_total = tokens_rank * world
            comparable = profile.comparable
        examples = step * global_bs
        vectors = step * vectors_per_step(global_bs)
        series.points.append(ScalingPoint(
            cell=curve.cell,
            step=step,
            pf_days=profile.pf_days(tokens_total, vectors),
            valid_loss=vloss,
            params=profile.trainable_params,
            tokens=tokens_total,
            examples=examples,
            comparable=comparable,
            metrics=by_step.get(step, {}),
        ))
    # A log written before the trainer counted tokens has no compute axis at
    # all, whatever its architecture. Say so once, at the series level, rather
    # than emitting a column of zeroes that reads like "free".
    if not any(p.tokens for p in series.points):
        series.comparable = False
    return series


def collect(out_dir: Path, frozen_tags: Optional[Sequence[str]] = None
            ) -> List[ScalingSeries]:
    """Every cell's scaling series in a run directory.

    A cell appears once even though it has one log per protocol, because the
    protocols re-score the same checkpoint and share a training trajectory.
    """
    frozen_tags = set(frozen_tags or ())
    seen: Dict[str, ScalingSeries] = {}
    for curve in discover_curves(out_dir):
        if not curve.steps:
            continue
        _, _, tag = curve.cell.partition("_")
        if tag in seen:
            continue
        frozen = tag.endswith("_frozen") or tag in frozen_tags
        series = series_from_curve(curve, frozen)
        if series and series.points:
            # The curve is named "<protocol>_<tag>"; the cell is the tag. Retag
            # the points too, or anything that joins a frontier entry back to
            # its series by name silently finds nothing.
            series.cell = tag
            for point in series.points:
                point.cell = tag
            seen[tag] = series
    return [seen[k] for k in sorted(seen, key=lambda t: seen[t].params)]


def frontier(series: Sequence[ScalingSeries], bins: int = 24
             ) -> List[Tuple[float, float, str]]:
    """The compute-efficient frontier: best loss achieved per compute budget.

    Log-spaced bins over PF-days, each keeping the single best (lowest-loss)
    point. This is the envelope the ``L = (C/Cc)^a`` fit is made against —
    fitting all points instead would measure "how long everything trained",
    not "what the best model at this budget achieves".
    """
    import math

    pts = [p for s in series if s.comparable for p in s.points if p.pf_days > 0]
    if not pts:
        return []
    lo = math.log10(min(p.pf_days for p in pts))
    hi = math.log10(max(p.pf_days for p in pts))
    if hi <= lo:
        best = min(pts, key=lambda p: p.valid_loss)
        return [(best.pf_days, best.valid_loss, best.cell)]
    width = (hi - lo) / bins
    buckets: Dict[int, ScalingPoint] = {}
    for p in pts:
        idx = min(bins - 1, int((math.log10(p.pf_days) - lo) / width))
        cur = buckets.get(idx)
        if cur is None or p.valid_loss < cur.valid_loss:
            buckets[idx] = p
    ordered = [buckets[i] for i in sorted(buckets)]
    return [(p.pf_days, p.valid_loss, p.cell) for p in ordered]


def fit_power_law(points: Sequence[Tuple[float, float]]
                  ) -> Optional[Tuple[float, float, float]]:
    """Least-squares fit of ``L = a * C^b`` in log-log space.

    Returns ``(a, b, r2)``. ``b`` is the scaling exponent — the number the
    whole study exists to estimate. Needs at least three points to mean
    anything, and points with non-positive loss are dropped because the fit is
    on ``log L``.
    """
    import math

    usable = [(c, l) for c, l in points if c > 0 and l > 0]
    if len(usable) < 3:
        return None
    xs = [math.log10(c) for c, _ in usable]
    ys = [math.log10(l) for _, l in usable]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    log_a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (log_a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return 10.0 ** log_a, b, r2


# --- families -------------------------------------------------------------
# A "family" is a set of encoders that vary in size but share a recipe, so a
# fit within one is a statement about capacity rather than about which
# pretraining corpus each model happened to get. The pooled frontier mixes
# recipes and therefore measures both at once; these do not.
FAMILIES: Dict[str, str] = {
    "bert-tiny": "BERT (Turc et al., pretrained)",
    "bert-mini": "BERT (Turc et al., pretrained)",
    "bert-small": "BERT (Turc et al., pretrained)",
    "bert-medium": "BERT (Turc et al., pretrained)",
    "scratch-tiny": "BERT (random init)",
    "scratch-mini": "BERT (random init)",
    "scratch-small": "BERT (random init)",
    "scratch-medium": "BERT (random init)",
    "scratch-base": "BERT (random init)",
    "scratch-xl": "BERT (random init)",
    "scratch-large": "BERT (random init)",
    "bge-large": "retrieval encoders",
    "e5-large": "retrieval encoders",
    "gte-large": "retrieval encoders",
    "mpnet": "retrieval encoders",
    "e5-mistral-7b": "retrieval encoders",
}


def family_of(encoder: str) -> str:
    """Which size-family an encoder belongs to; its own name if it is alone."""
    return FAMILIES.get(encoder, encoder)


def by_family(series: Sequence[ScalingSeries]) -> Dict[str, List[ScalingSeries]]:
    """Group comparable cells by family, keeping only families with >1 size."""
    groups: Dict[str, List[ScalingSeries]] = {}
    for s in series:
        if s.comparable:
            groups.setdefault(family_of(s.encoder), []).append(s)
    return {name: cells for name, cells in sorted(groups.items())
            if len({c.params for c in cells}) > 1}


# --- y-axes ---------------------------------------------------------------
# The in-batch contrastive loss is not the only defensible y-axis, and it is a
# poor one for a scaling law: it is bounded above by ln(B) and below by the
# near-duplicate structure of the batch, so a whole decade of model size can
# only move it within that band. A ranking metric against a candidate pool of
# the same size in every cell has no such ceiling built into the objective.
# `error = 1 - MRR` is used rather than MRR itself because a power law
# describes a quantity that decays toward zero, not one that rises toward one.


def _metric(metrics: Dict[str, float], name: str) -> Optional[float]:
    """Look a metric up by name, case-insensitively.

    `kgfm/eval.py` names them `MRR`, `Hit@10`, `nDCG` — display casing, not
    identifiers — while `loss` is lowercase. Matching exactly meant asking for
    "mrr" and silently getting None, which reads downstream as "this run never
    measured it" rather than "the key is spelled differently".
    """
    if name in metrics:
        return float(metrics[name])
    lowered = name.lower()
    for key, value in metrics.items():
        if key.lower() == lowered:
            return float(value)
    return None


def point_y(point: "ScalingPoint", axis: str) -> Optional[float]:
    """Value of ``point`` on the named y-axis, or None if it was not measured."""
    if axis == "loss":
        return point.valid_loss
    if axis == "mrr_error":
        mrr = _metric(point.metrics, "MRR")
        # Clamped away from 0 because the axis is plotted and fitted in log
        # space; a perfect MRR of exactly 1 would otherwise drop the point.
        return None if mrr is None else max(1e-6, 1.0 - mrr)
    return _metric(point.metrics, axis)


Y_AXES: Dict[str, Tuple[str, str]] = {
    # name -> (axis label, why it is here)
    "loss": ("validation loss",
             "The training objective itself, measured on held-out batches at "
             "the training batch size."),
    "mrr_error": ("1 - validation MRR",
                  "Ranking error against a fixed-size candidate pool. Unlike "
                  "the in-batch loss it is not bounded by ln(B) and does not "
                  "have a floor set by repeated tails, so a decade of model "
                  "size has room to show."),
}


def frontier_on(series: Sequence[ScalingSeries], axis: str = "loss",
                bins: int = 24) -> List[Tuple[float, float, str]]:
    """`frontier`, on any y-axis in ``Y_AXES``. Lower is better on all of them."""
    import math

    pts: List[Tuple[ScalingPoint, float]] = []
    for s in series:
        if not s.comparable:
            continue
        for p in s.points:
            y = point_y(p, axis)
            if p.pf_days > 0 and y is not None and y > 0:
                pts.append((p, y))
    if not pts:
        return []
    lo = math.log10(min(p.pf_days for p, _ in pts))
    hi = math.log10(max(p.pf_days for p, _ in pts))
    if hi <= lo:
        p, y = min(pts, key=lambda t: t[1])
        return [(p.pf_days, y, p.cell)]
    width = (hi - lo) / bins
    buckets: Dict[int, Tuple[ScalingPoint, float]] = {}
    for p, y in pts:
        idx = min(bins - 1, int((math.log10(p.pf_days) - lo) / width))
        cur = buckets.get(idx)
        if cur is None or y < cur[1]:
            buckets[idx] = (p, y)
    return [(buckets[i][0].pf_days, buckets[i][1], buckets[i][0].cell)
            for i in sorted(buckets)]


# --- saturating fit -------------------------------------------------------

@dataclass
class SaturatingFit:
    """``L = L_inf + a * C^b`` — a power law with an irreducible floor."""

    l_inf: float
    a: float
    b: float
    r2: float                # in linear L, not log L: comparable across fits
    n: int
    at_boundary: bool        # L_inf pinned against min(L); see below

    @property
    def label(self) -> str:
        return f"L = {self.l_inf:.4g} + {self.a:.3g}·C^{self.b:.3f}"


# Below this many points the three parameters are not separately identifiable:
# L_inf trades off against `a` almost exactly, so the fit reports whatever the
# grid search wandered into. Four points produced an L_inf sitting on the
# boundary every time it was tried here.
MIN_SATURATING_POINTS = 6


def fit_saturating(points: Sequence[Tuple[float, float]]
                   ) -> Optional[SaturatingFit]:
    """Fit ``L = L_inf + a * C^b`` without scipy.

    For a fixed ``L_inf`` the model is linear in log-log on ``L - L_inf``, so
    the whole fit is a 1-D scan over ``L_inf`` with an OLS solve inside. The
    scan is scored by residuals in **linear** L rather than log L: scoring in
    log space rewards pushing ``L_inf`` up against ``min(L)``, because that
    inflates the dynamic range of ``log(L - L_inf)`` and flatters R² while
    fitting nothing.

    Returns None when there are too few points to identify three parameters.
    ``at_boundary`` says the optimum sat at the top of the scan, which means
    the data does not constrain the floor — read the exponent as a lower bound
    on |b|, not as an estimate.
    """
    import math

    usable = [(c, l) for c, l in points if c > 0 and l > 0]
    if len(usable) < MIN_SATURATING_POINTS:
        return None
    ys = [l for _, l in usable]
    y_min, y_mean = min(ys), sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)

    best: Optional[SaturatingFit] = None
    steps = 200
    # 0 reproduces the plain power law; the top of the range approaches the
    # best loss observed, which is the largest floor the data can support.
    for i in range(steps + 1):
        l_inf = y_min * (i / steps) * 0.999
        xs = [math.log10(c) for c, _ in usable]
        zs = [math.log10(l - l_inf) for _, l in usable]
        n = len(xs)
        mx, mz = sum(xs) / n, sum(zs) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx <= 0:
            continue
        b = sum((x - mx) * (z - mz) for x, z in zip(xs, zs)) / sxx
        a = 10.0 ** (mz - b * mx)
        ss_res = sum((l - (l_inf + a * c ** b)) ** 2 for c, l in usable)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        if best is None or r2 > best.r2:
            best = SaturatingFit(l_inf=l_inf, a=a, b=b, r2=r2, n=n,
                                 at_boundary=(i >= steps - 1))
    return best
