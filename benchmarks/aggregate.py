"""Read each method's metrics JSON and print a side-by-side comparison.

Looks at ``benchmarks/results/{kgfm,ultra,motif}.json`` (override with
``--results-dir``) and emits a Markdown table on stdout. If ``--out`` is
given, the table is also written to that path.

Note on protocol differences
----------------------------
- ``kgfm`` uses a *pooled* ranking protocol: each test triple is ranked
  against ``pool_size`` candidate tail texts plus the true tail.
- ``ULTRA`` and ``MOTIF`` (when run via their stock CLIs) use *filtered*
  ranking against all entities in the KG.

These numbers are therefore directionally comparable but not strictly
equivalent. The table flags the protocol per row to keep this honest.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

_DEFAULT_FILES = ["kgfm.json", "ultra.json", "motif.json"]
_METRIC_KEYS = ["mrr", "hits@1", "hits@3", "hits@10", "ndcg", "mr"]


def _norm_metrics(raw: Dict[str, Any]) -> Dict[str, str]:
    """Normalize varied key casings into our common schema."""
    flat: Dict[str, str] = {}
    for k, v in (raw or {}).items():
        kl = k.lower().replace("hit@", "hits@")
        if isinstance(v, (int, float)):
            flat[kl] = f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
        else:
            flat[kl] = str(v)
    return flat


def _row(record: Dict[str, Any]) -> Dict[str, str]:
    metrics = _norm_metrics(record.get("metrics", {}))
    method = record.get("method", "?")
    encoder = record.get("encoder")
    if encoder:
        # Distinguish frozen-LM rows so the table doesn't collapse them onto
        # the fine-tuned cells of the same encoder.
        if record.get("freeze_encoder"):
            encoder = f"{encoder}, frozen"
        method = f"{method} ({encoder})"
    protocol = record.get("protocol", {}).get("type")
    if not protocol:
        protocol = "filtered" if method.lower().startswith(("ultra", "motif")) else "?"
    return {
        "Method": method,
        "Mode": record.get("mode", "trained" if method.lower().startswith("kgfm") else "zero-shot"),
        "Protocol": protocol,
        **{k: metrics.get(k, "—") for k in _METRIC_KEYS},
        "n_eval": str(record.get("metrics", {}).get("n", "—")),
    }


def _markdown_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_no results found_\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r.get(h, "—") for h in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default="benchmarks/results")
    p.add_argument("--files", nargs="*", default=_DEFAULT_FILES,
                   help="Filenames inside --results-dir to aggregate.")
    p.add_argument("--out", default=None,
                   help="Write the rendered Markdown table to this path too.")
    args = p.parse_args()

    rows: List[Dict[str, str]] = []
    for fn in args.files:
        path = os.path.join(args.results_dir, fn)
        if not os.path.isfile(path):
            print(f"[aggregate] skipping missing {path}")
            continue
        with open(path) as f:
            record = json.load(f)
        rows.append(_row(record))

    table = _markdown_table(rows)
    print(table)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(table)
        print(f"[aggregate] wrote {args.out}")


if __name__ == "__main__":
    main()
