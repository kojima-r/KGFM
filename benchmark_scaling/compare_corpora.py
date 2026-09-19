#!/usr/bin/env python
"""Compare model-size scaling exponents measured on two different corpora.

    python benchmark_scaling/compare_corpora.py \
        --run benchmark_scaling/results/chembl/20260827T040036Z_scaling_scratch \
        --run benchmark_scaling/results/random/<ts>_scaling_random

WHY A SEPARATE SCRIPT
`kgfm scaling` reports one run. The question here spans two, and answering it
honestly needs the two to be put on the *same* footing first — which they are
not as they stand:

* the ChEMBL study swept seven sizes, this one five, and an exponent fitted
  over a wider range of N is not the same estimate;
* the ChEMBL study ran 48,000 steps, this one 24,000, and a scaling frontier
  keeps moving as the runs get longer.

Both are fixable without re-training, because a cell's whole trajectory is in
its `cell_*.log`: restrict to the sizes both runs have, and drop every
validation point past the shorter run's step budget. That is what `--max-step`
and `--only-common-sizes` (both on by default) do, and it is the difference
between comparing two numbers and comparing two experiments.

The untruncated fits are printed too, so the effect of the matching is visible
rather than assumed.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgfm.scaling.points import (ScalingSeries, Y_AXES, collect,  # noqa: E402
                                 fit_power_law, fit_saturating, frontier_on)


def truncate(series: Sequence[ScalingSeries], max_step: Optional[int],
             keep: Optional[set]) -> List[ScalingSeries]:
    """Copies of ``series`` restricted to ``keep`` sizes and ``max_step``."""
    out: List[ScalingSeries] = []
    for s in series:
        if keep is not None and s.cell not in keep:
            continue
        t = copy.copy(s)
        t.points = [p for p in s.points
                    if max_step is None or p.step <= max_step]
        if t.points:
            out.append(t)
    return out


def fits(series: Sequence[ScalingSeries], axis: str) -> Dict[str, object]:
    fr = frontier_on(series, axis)
    xy = [(c, v) for c, v, _ in fr]
    power = fit_power_law(xy)
    sat = fit_saturating(xy)
    return {"n_frontier": len(fr), "power": power, "sat": sat,
            "cells": sorted({s.cell for s in series}),
            "steps": max((p.step for s in series for p in s.points),
                         default=0)}


def fmt(f: Dict[str, object]) -> str:
    power = f["power"]
    sat = f["sat"]
    left = ("b=n/a" if not power
            else f"b={power[1]:+.4f} R2={power[2]:.3f}")
    if sat is None:
        right = "sat=n/a"
    else:
        edge = " (edge)" if sat.at_boundary else ""
        right = f"sat b={sat.b:+.4f} L_inf={sat.l_inf:.3f} R2={sat.r2:.3f}{edge}"
    return f"{left:<24}{right}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="compare_corpora.py",
        description="Put two model-size scaling runs on the same footing and "
                    "compare their exponents.")
    p.add_argument("--run", action="append", required=True, metavar="DIR",
                   help="A scaling run directory. Repeatable; give two or more.")
    p.add_argument("--max-step", type=int, default=None,
                   help="Drop validation points past this step. Default: the "
                        "smallest final step across the runs, so the "
                        "comparison is at a matched budget.")
    p.add_argument("--all-sizes", action="store_true",
                   help="Do not restrict to the sizes every run has. Off by "
                        "default because a wider N range is a different "
                        "estimate, not a better one.")
    args = p.parse_args(argv)

    if len(args.run) < 2:
        raise SystemExit("Give --run at least twice.")

    runs: Dict[str, List[ScalingSeries]] = {}
    for d in args.run:
        path = Path(d)
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {d}")
        series = [s for s in collect(path) if s.points]
        if not series:
            raise SystemExit(f"No usable cell logs in {d}")
        runs[path.name] = series

    common = set.intersection(*[{s.cell for s in v} for v in runs.values()])
    budget = min(max(p.step for s in v for p in s.points)
                 for v in runs.values())
    max_step = args.max_step if args.max_step is not None else budget
    keep = None if args.all_sizes else common

    print(f"runs: {len(runs)}")
    for name, series in runs.items():
        sizes = sorted({s.cell for s in series})
        last = max(p.step for s in series for p in s.points)
        print(f"  {name}\n    {len(sizes)} sizes, {last:,} steps: "
              f"{', '.join(sizes)}")
    print(f"\ncommon sizes: {len(common)} ({', '.join(sorted(common))})")
    print(f"matched budget: step <= {max_step:,}"
          + ("" if args.max_step is not None else " (the shorter run)"))

    for axis in Y_AXES:
        label, _ = Y_AXES[axis]
        print(f"\n=== {label} ===")
        for name, series in runs.items():
            raw = fits(series, axis)
            print(f"  {name}")
            print(f"    as-run   ({len(raw['cells'])} sizes, "
                  f"{raw['steps']:,} steps, {raw['n_frontier']} frontier pts)"
                  f"  {fmt(raw)}")
            m = fits(truncate(series, max_step, keep), axis)
            print(f"    matched  ({len(m['cells'])} sizes, "
                  f"{m['steps']:,} steps, {m['n_frontier']} frontier pts)"
                  f"  {fmt(m)}")

    print("\nRead the `matched` rows against each other; `as-run` is there to "
          "show how much the matching moved things.\nA difference in b between "
          "two corpora is only a corpus effect if the learning rates were "
          "probed\non each corpus separately — otherwise it is confounded with "
          "how well one corpus's\nrates happened to transfer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
