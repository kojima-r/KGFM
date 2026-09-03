#!/usr/bin/env python
"""Merge several `lr_probe.py` output directories into one table.

The probe is run in phases (see run_lr_probe.sh: concurrency is capped by GPU
memory only at the large end), and each phase writes its own directory. This
joins them so the winners are chosen from one table rather than three.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lr_probe import summarize  # noqa: E402


def main(argv: list) -> int:
    if not argv:
        print("usage: summarize_lr_probe.py <probe-dir> [<probe-dir> ...]")
        return 2
    results = []
    for d in argv:
        path = Path(d) / "lr_probe.json"
        if not path.is_file():
            print(f"[warn] no lr_probe.json in {d}, skipping")
            continue
        results.extend(json.loads(path.read_text()))
    if not results:
        print("no results found")
        return 1
    # Preserve the order the cells were requested in rather than sorting by
    # name: "scratch-base" would otherwise sort before "scratch-tiny" and the
    # table would not read small -> large.
    order = ["scratch-tiny", "scratch-mini", "scratch-small",
             "scratch-medium", "scratch-base",
             "bert-tiny", "bert-mini", "bert-small", "bert-medium", "mpnet"]
    encoders = ([e for e in order if any(r["encoder"] == e for r in results)]
                + sorted({r["encoder"] for r in results} - set(order)))
    lrs = sorted({r["lr"] for r in results})
    text = summarize(results, encoders, lrs)
    print(text)
    out = Path(argv[0]) / "summary_merged.txt"
    out.write_text(text)
    print(f"[lr-probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
