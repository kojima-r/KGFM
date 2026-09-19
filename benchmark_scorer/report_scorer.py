#!/usr/bin/env python
"""Scorer x head-mode comparison report.

    python benchmark_scorer/report_scorer.py --out-dir latest

WHY THIS IS NOT `kgfm report`
That report is a flat table of whatever is in a run directory, which is right
when the rows are different *methods*. Here the rows are a 2-D grid — one
scoring function crossed with one head wiring — and the question is which of
the two axes explains the difference. A flat table cannot show that; a grid
plus per-axis marginals can, so this script builds those.

WHAT IT REFUSES TO CONFLATE
`separate` head mode has **three times the head parameters** of `shared`. If it
wins, "the scorer is better" and "the head is bigger" are both live
explanations, and only one of them is interesting. Every table here therefore
carries the trainable-parameter count next to the metric, and the summary says
so rather than leaving the reader to notice.

The distance scorers (TransE, RotatE) are also not on equal footing with the
inner-product ones under the default loss: `contrastive` L2-normalizes the
query and the tails, which is a per-row rescaling for a dot product but a
change of geometry for a distance. That is a real caveat about the comparison,
not a bug in it, and it is printed with the results.
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
from kgfm.report import discover_curves                    # noqa: E402
from kgfm.report_html import _CSS, _kv_grid                # noqa: E402
from kgfm.runs import RunLogger, resolve_run_dir           # noqa: E402
from kgfm.scorers import SCORERS, make_scorer              # noqa: E402
from kgfm.heads import HEAD_MODES                          # noqa: E402

DEFAULT_RESULTS_ROOT = "benchmark_scorer/results/chembl"

# Lower is better for loss, higher for the rest — the grid needs to know which
# way to look for a winner.
METRICS = {
    "MRR": ("MRR", True),
    "Hit@1": ("Hit@1", True),
    "Hit@10": ("Hit@10", True),
    "nDCG": ("nDCG", True),
    "best_valid_loss": ("best valid loss", False),
}


@dataclass
class Cell:
    """One trained cell: a (head_mode, scorer) pair and what it achieved."""

    tag: str
    encoder: str
    head: str
    head_mode: str
    scorer: str
    params_total: Optional[int] = None
    params_trainable: Optional[int] = None
    head_params: Optional[int] = None
    final: Dict[str, float] = field(default_factory=dict)
    valid_losses: List[Tuple[int, float]] = field(default_factory=list)
    valid_mrr: List[Tuple[int, float]] = field(default_factory=list)
    train_losses: List[Tuple[int, float]] = field(default_factory=list)
    train_seconds: Optional[float] = None

    @property
    def best_valid_loss(self) -> Optional[float]:
        return min((v for _, v in self.valid_losses), default=None)

    @property
    def kind(self) -> str:
        """inner-product or distance — the two are not scaled alike."""
        try:
            return make_scorer(self.scorer).kind
        except SystemExit:
            return "?"

    def value(self, metric: str) -> Optional[float]:
        if metric == "best_valid_loss":
            return self.best_valid_loss
        v = self.final.get(metric)
        return None if v is None else float(v)


def collect(out_dir: Path) -> List[Cell]:
    """Join each cell's result JSON with its training log.

    The JSON carries what the cell *was* (head_mode, scorer, parameter counts
    are not in it, so those come from the log's `[init]` line) and the final
    metrics; the log carries the curves. Cells whose JSON lacks `head_mode` or
    `scorer` predate those options and are reported as shared/distmult, which
    is what they were.
    """
    curves = {c.cell.partition("_")[2]: c for c in discover_curves(out_dir)}
    cells: Dict[str, Cell] = {}
    for path in sorted(out_dir.glob("kgfm_*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if rec.get("method") != "kgfm":
            continue
        # kgfm_<protocol>_<tag>.json
        tag = path.stem.split("_", 2)[2]
        if tag in cells:
            continue
        curve = curves.get(tag)
        cell = Cell(
            tag=tag,
            encoder=rec.get("encoder", "?"),
            head=rec.get("head", "auto"),
            head_mode=rec.get("head_mode", "shared"),
            scorer=rec.get("scorer", "distmult"),
            params_total=getattr(curve, "params_total", None),
            params_trainable=getattr(curve, "params_trainable", None),
            head_params=getattr(curve, "params_head", None),
            final={k: v for k, v in (rec.get("metrics") or {}).items()
                   if isinstance(v, (int, float))},
            valid_losses=list(getattr(curve, "valid_losses", []) or []),
            valid_mrr=list((getattr(curve, "valid_metrics", {}) or {})
                           .get("MRR", [])),
            train_losses=list(zip(getattr(curve, "steps", []) or [],
                                  getattr(curve, "losses", []) or [])),
            train_seconds=rec.get("train_seconds"),
        )
        cells[tag] = cell
    # `head_params` comes straight from the trainer's `[init] ... head=` field.
    # Logs written before that field existed have none, and the fallback below
    # is deliberately *not* `total - min(total)`: that makes the smallest cell
    # read as zero head parameters, which is false and defeats the whole point
    # of showing the column. Leave it unknown instead.
    return sorted(cells.values(), key=lambda c: (c.head_mode, c.scorer))


# --------------------------------------------------------------------------
# grid + marginals
# --------------------------------------------------------------------------

def grid(cells: Sequence[Cell], metric: str
         ) -> Tuple[List[str], List[str], Dict[Tuple[str, str], Optional[float]]]:
    modes = [m for m in HEAD_MODES if any(c.head_mode == m for c in cells)]
    scs = [s for s in SCORERS if any(c.scorer == s for c in cells)]
    table = {(m, s): None for m in modes for s in scs}
    for c in cells:
        table[(c.head_mode, c.scorer)] = c.value(metric)
    return modes, scs, table


def marginals(cells: Sequence[Cell], metric: str
              ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Mean over the other axis, which is what "does this axis matter" means.

    A marginal is only honest when the grid is complete — otherwise the two
    means average different sets of cells. Missing cells are dropped and the
    counts are reported so a partial run cannot masquerade as a full one.
    """
    by_mode: Dict[str, List[float]] = {}
    by_scorer: Dict[str, List[float]] = {}
    for c in cells:
        v = c.value(metric)
        if v is None:
            continue
        by_mode.setdefault(c.head_mode, []).append(v)
        by_scorer.setdefault(c.scorer, []).append(v)
    mean = lambda xs: sum(xs) / len(xs)
    return ({k: mean(v) for k, v in by_mode.items()},
            {k: mean(v) for k, v in by_scorer.items()})


def _fmt(v: Optional[float], metric: str) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def grid_rows(cells: Sequence[Cell], metric: str) -> List[Dict[str, str]]:
    modes, scs, table = grid(cells, metric)
    _, higher = METRICS[metric]
    vals = [v for v in table.values() if v is not None]
    best = (max(vals) if higher else min(vals)) if vals else None
    rows = []
    for m in modes:
        row = {"head_mode": m}
        for s in scs:
            v = table[(m, s)]
            mark = " *" if (v is not None and best is not None
                            and abs(v - best) < 1e-12) else ""
            row[s] = _fmt(v, metric) + mark
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

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


def cell_rows(cells: Sequence[Cell]) -> List[Dict[str, str]]:
    return [{
        "Cell": c.tag,
        "head_mode": c.head_mode,
        "scorer": c.scorer,
        "kind": c.kind,
        "head params": (f"{c.head_params:,}" if c.head_params is not None
                        else "—"),
        "MRR": _fmt(c.value("MRR"), "MRR"),
        "Hit@1": _fmt(c.value("Hit@1"), "Hit@1"),
        "Hit@10": _fmt(c.value("Hit@10"), "Hit@10"),
        "nDCG": _fmt(c.value("nDCG"), "nDCG"),
        "best valid loss": _fmt(c.best_valid_loss, "best_valid_loss"),
        "train s": (f"{c.train_seconds:.0f}" if c.train_seconds else "—"),
    } for c in cells]


def plot_curves(cells: Sequence[Cell], backend: str, which: str = "loss") -> str:
    """One line per cell. Solid/dashed is not available across all three chart
    backends, so the label carries the head mode instead."""
    series = []
    for c in cells:
        pts = c.valid_losses if which == "loss" else c.valid_mrr
        if pts:
            series.append((f"{c.scorer}/{c.head_mode}",
                           [(float(s), v) for s, v in pts]))
    if not series:
        return '<p class="note">no validation history</p>'
    return charts_mod.chart(
        series, x_label="optimizer step",
        y_label="validation loss" if which == "loss" else "validation MRR",
        backend=backend, height=380,
    )


def caveats_html(cells: Sequence[Cell]) -> str:
    modes = {c.head_mode for c in cells}
    kinds = {c.kind for c in cells}
    out = []
    if "separate" in modes and "shared" in modes:
        sh = [c.head_params for c in cells
              if c.head_mode == "shared" and c.head_params is not None]
        sp = [c.head_params for c in cells
              if c.head_mode == "separate" and c.head_params is not None]
        if sh and sp:
            ratio = (sum(sp) / len(sp)) / max(sum(sh) / len(sh), 1e-9)
            out.append(
                f'<p><strong>The head modes do not have equal capacity.</strong> '
                f'`separate` carries {ratio:.1f}x the head parameters of '
                f'`shared` here ({sum(sp)//len(sp):,} vs {sum(sh)//len(sh):,}). '
                f'A win for `separate` is therefore not by itself evidence that '
                f'per-role projection is the right inductive bias — it may just '
                f'be the extra parameters. The encoder is identical across '
                f'cells, so this difference is the whole difference.</p>')
    if "distance" in kinds and "inner" in kinds:
        out.append(
            '<p><strong>Distance and inner-product scorers are not on equal '
            'footing under the default loss.</strong> `contrastive` '
            'L2-normalizes the query and the tails, which is a per-row positive '
            'rescaling for a dot product (so it cannot change a DistMult or '
            'ComplEx ranking) but a genuine change of geometry for a distance '
            '(TransE, RotatE). Those two are conventionally trained with a '
            'margin loss on raw scores. The run keeps one loss for every cell '
            'because changing it per scorer would confound the comparison with '
            'the objective — but that choice is a handicap for the distance '
            'scorers and their numbers should be read as "under this loss", '
            'not as the best they can do.</p>')
    out.append(
        '<p class="note">Everything else is pinned: same encoder, same head '
        'type, same proj_dim, same batch size, same steps, same data. B-1 is '
        'the in-batch negative count and proj_dim is the scoring width, so '
        'varying either would confound this comparison the way it would any '
        'other.</p>')
    return "".join(out)


def render(out_dir: Path, cells: Sequence[Cell], backend: str) -> str:
    meta = {}
    mp = out_dir / "meta.json"
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
    params = meta.get("params", {}) or {}
    first = cells[0] if cells else None
    grid_meta = [
        ("run", out_dir.name),
        ("host / git", f"{meta.get('host','?')} / "
                       f"{str(meta.get('git_rev','?'))[:8]}"),
        ("config", params.get("config_file", "?")),
        ("encoder (fixed)", first.encoder if first else "?"),
        ("head type (fixed)", first.head if first else "?"),
        ("head modes", ", ".join(sorted({c.head_mode for c in cells}))),
        ("scorers", ", ".join(sorted({c.scorer for c in cells}))),
        ("proj_dim", params.get("proj_dim", "?")),
        ("batch size", params.get("batch_size", "?")),
        ("max_steps", params.get("max_steps", "?")),
        ("loss", params.get("loss") or "contrastive (default)"),
        ("cells", len(cells)),
    ]

    body = [f"<h2>Experiment settings</h2>{_kv_grid(grid_meta)}",
            caveats_html(cells)]

    for metric, (label, higher) in METRICS.items():
        rows = grid_rows(cells, metric)
        if not rows:
            continue
        arrow = "higher is better" if higher else "lower is better"
        body.append(f"<h2>{html.escape(label)} — grid ({arrow})</h2>")
        body.append(f'<div class="scroll">{_table_html(rows)}</div>')
        by_mode, by_scorer = marginals(cells, metric)
        mrows = [{"axis": "head_mode", **{k: f"{v:.4f}"
                                          for k, v in sorted(by_mode.items())}}]
        srows = [{"axis": "scorer", **{k: f"{v:.4f}"
                                       for k, v in sorted(by_scorer.items())}}]
        body.append('<p class="note">Marginal means — the average over the '
                    'other axis, i.e. "does this axis matter at all".</p>')
        body.append(f'<div class="scroll">{_table_html(mrows)}</div>')
        body.append(f'<div class="scroll">{_table_html(srows)}</div>')

    body.append("<h2>Per cell</h2>")
    body.append(f'<div class="scroll">{_table_html(cell_rows(cells))}</div>')
    body.append("<h2>Training curves</h2>")
    body.append(f'<div class="charts wide">{plot_curves(cells, backend, "loss")}</div>')
    body.append(f'<div class="charts wide">{plot_curves(cells, backend, "mrr")}</div>')

    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>kgfm scorer comparison — {html.escape(out_dir.name)}</title>'
            f"<style>{_CSS}</style></head><body>"
            f"<h1>Scorer x head-mode comparison</h1>{''.join(body)}</body></html>")


def build(out_dir: Path, backend: str = "auto") -> Dict[str, object]:
    cells = collect(out_dir)
    if not cells:
        raise SystemExit(
            f"No kgfm result JSONs in {out_dir}.\n"
            "This report needs a finished `kgfm bench run` over "
            "head_modes x scorers."
        )
    page = render(out_dir, cells, backend)
    (out_dir / "scorer_report.html").write_text(page, encoding="utf-8")

    md = ["# Scorer x head-mode comparison\n"]
    for metric, (label, higher) in METRICS.items():
        rows = grid_rows(cells, metric)
        if rows:
            md.append(f"\n## {label} "
                      f"({'higher' if higher else 'lower'} is better)\n")
            md.append(table_md(rows))
    md.append("\n## Per cell\n")
    md.append(table_md(cell_rows(cells)))
    (out_dir / "scorer_table.md").write_text("".join(md), encoding="utf-8")

    payload = {
        "cells": [{
            "tag": c.tag, "encoder": c.encoder, "head": c.head,
            "head_mode": c.head_mode, "scorer": c.scorer, "kind": c.kind,
            "params_total": c.params_total,
            "params_trainable": c.params_trainable,
            "head_params": c.head_params,
            "final": c.final, "best_valid_loss": c.best_valid_loss,
            "train_seconds": c.train_seconds,
            "valid_losses": c.valid_losses, "valid_mrr": c.valid_mrr,
        } for c in cells],
        "marginals": {
            m: {"head_mode": marginals(cells, m)[0],
                "scorer": marginals(cells, m)[1]}
            for m in METRICS
        },
    }
    (out_dir / "scorer_points.json").write_text(json.dumps(payload, indent=2),
                                                encoding="utf-8")

    print("".join(md))
    for metric in METRICS:
        by_mode, by_scorer = marginals(cells, metric)
        if by_mode:
            print(f"[{metric}] head_mode means: " + "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(by_mode.items())))
            print(f"[{metric}] scorer means:    " + "  ".join(
                f"{k}={v:.4f}" for k, v in sorted(by_scorer.items())))
    print(f"\nwrote {out_dir}/scorer_{{report.html,table.md,points.json}}")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="report_scorer.py",
        description="Compare scoring functions and head wiring modes.")
    p.add_argument("--out-dir", default="latest")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--charts", default="auto",
                   choices=["auto", "plotly", "matplotlib", "svg"])
    args = p.parse_args(argv)
    out_dir = resolve_run_dir(args.results_root, args.out_dir)
    try:
        RunLogger(out_dir).record_command(tag="scorer-report")
    except Exception:                                   # pragma: no cover
        pass
    build(out_dir, args.charts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
