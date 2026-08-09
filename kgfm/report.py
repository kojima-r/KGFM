"""`kgfm report` — collect a run directory's results into a table and a report.

Reporting is deliberately separate from any one method: it discovers whatever
is in the run directory rather than being told what ran, so it works the same
whether the run has one kgfm cell or twelve, and whether `kgfm-ultra` /
`kgfm-motif` were run at all — including hours later, into a run that has
since finished. Row order is kgfm cells (sorted), then ULTRA, then MOTIF.

Two outputs, both written into the run directory:

- ``table.md``   — the comparison table alone, for pasting into notes.
- ``report.html`` — the same table plus the training curves, per-method
  parameters, and the run's metadata, in one self-contained page.

Training curves are *parsed back out of the per-cell logs* rather than emitted
by the trainer. That keeps the trainer unchanged and means reports can be
produced for runs that already finished, which is the same "pick up whatever
is there" contract as the rest of this module.

Note on protocol differences
----------------------------
- kgfm rows may be *pooled* (ranked against ``pool_size`` sampled tails) or
  *filtered* (ranked against the full tail vocabulary with (h,r) masking).
- ULTRA / MOTIF always report filtered ranking, averaged over head and tail.

The Protocol and n_eval columns exist to keep that visible; see
benchmarks/README.md before comparing rows across methods.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runs import RunLogger, read_commands, resolve_run_dir

DEFAULT_RESULTS_ROOT = "benchmarks/results/chembl"
_METRIC_KEYS = ["mrr", "hits@1", "hits@3", "hits@10", "ndcg", "mr"]
_BASELINE_FILES = ("ultra.json", "motif.json")

# Emitted by kgfm/train.py:
#   "[step    50] loss=5.5320  rate=9311 ex/s  gnorm=1.2345"
# gnorm is absent when --grad-clip is 0, and in logs written before it existed.
_STEP_RE = re.compile(
    r"^\[step\s+(\d+)\]\s+loss=([-\d.eE+]+)\s+rate=([\d.]+)"
    r"(?:\s*ex/s\s*gnorm=([-\d.eE+]+))?"
)
# "[init] device=cuda:0 world_size=1 ... global_bs=256"
_GLOBAL_BS_RE = re.compile(r"^\[init\].*global_bs=(\d+)")
# "[valid eval @ step 1000] {'MRR': ..., ...}"
_EVAL_RE = re.compile(r"^\[(\w+) eval @ step (\d+)\]\s*(\{.*\})\s*$")
_INIT_RE = re.compile(r"^\[init\] encoder=(\S+) dim=(\d+) params total=([\d,]+)")


@dataclass
class TrainingCurve:
    """What a single cell's log says about how training went.

    ``losses`` is the training loss at each logged step; ``valid_losses`` is
    the same quantity on held-out data at each validation step, so the two can
    share an axis. ``valid_metrics`` holds every other validation number
    (MRR, Hit@k, nDCG) keyed by name.
    """

    cell: str
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    rates: List[float] = field(default_factory=list)
    grad_norms: List[Tuple[int, float]] = field(default_factory=list)
    valid_losses: List[Tuple[int, float]] = field(default_factory=list)
    valid_metrics: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)
    encoder_info: str = ""
    global_batch_size: Optional[int] = None

    @property
    def evals(self) -> List[Tuple[int, float]]:
        """Validation MRR history (kept for convenience)."""
        return self.valid_metrics.get("MRR", [])

    def examples(self, steps: List[int]) -> List[int]:
        """Convert optimizer steps to examples seen.

        Cells with different batch sizes (ngram at 1024 vs transformer at 64)
        are not comparable step-for-step — the same step count is 16x the data.
        Plotting against examples seen puts them on a common x-axis.
        """
        bs = self.global_batch_size or 1
        return [st * bs for st in steps]

    def gap(self) -> List[Tuple[int, float]]:
        """valid loss - train loss at each validation step (generalisation gap).

        The train loss is taken from the nearest logged step at or before the
        validation step, which is the value the run was actually at.
        """
        if not self.valid_losses or not self.steps:
            return []
        out = []
        for vstep, vloss in self.valid_losses:
            prior = [(st, ls) for st, ls in zip(self.steps, self.losses) if st <= vstep]
            if prior:
                out.append((vstep, vloss - prior[-1][1]))
        return out


def parse_training_log(path: Path) -> TrainingCurve:
    """Recover a cell's loss / throughput / validation history from its log.

    Resumed runs append to the same file, so step numbers can repeat or jump;
    the points are kept in the order they were logged rather than sorted, which
    is what actually happened.
    """
    curve = TrainingCurve(cell=path.stem.replace("cell_", ""))
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return curve
    for line in text.splitlines():
        m = _STEP_RE.match(line)
        if m:
            step = int(m.group(1))
            curve.steps.append(step)
            curve.losses.append(float(m.group(2)))
            curve.rates.append(float(m.group(3)))
            if m.group(4):
                curve.grad_norms.append((step, float(m.group(4))))
            continue
        m = _GLOBAL_BS_RE.match(line)
        if m and curve.global_batch_size is None:
            curve.global_batch_size = int(m.group(1))
            continue
        m = _EVAL_RE.match(line)
        if m:
            try:
                metrics = ast.literal_eval(m.group(3))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(metrics, dict):
                continue
            step = int(m.group(2))
            for key, value in metrics.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key == "loss":
                    curve.valid_losses.append((step, float(value)))
                elif key in ("n", "pool", "vocab", "filter_pairs",
                             "tails_outside_vocab"):
                    continue  # bookkeeping, not a metric worth plotting
                else:
                    curve.valid_metrics.setdefault(key, []).append(
                        (step, float(value))
                    )
            continue
        m = _INIT_RE.match(line)
        if m and not curve.encoder_info:
            curve.encoder_info = f"{m.group(1)}, dim={m.group(2)}, {m.group(3)} params"
    return curve


def discover_embeddings(out_dir: Path) -> Dict[str, dict]:
    """`embeddings_<tag>.json` produced by `kgfm viz`, keyed by cell tag."""
    found: Dict[str, dict] = {}
    for path in sorted(out_dir.glob("embeddings_*.json")):
        record = _load_json(path)
        if record:
            found[path.stem.replace("embeddings_", "")] = record
    return found


def discover_curves(out_dir: Path) -> List[TrainingCurve]:
    return [parse_training_log(p) for p in sorted(out_dir.glob("cell_*.log"))]


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


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
    protocol = (record.get("protocol") or {}).get("type")
    if not protocol:
        protocol = "filtered" if method.lower().startswith(("ultra", "motif")) else "?"
    row = {
        "Method": method,
        "Mode": record.get(
            "mode", "trained" if method.lower().startswith("kgfm") else "zero-shot"
        ),
        "Protocol": protocol,
        **{k: metrics.get(k, "—") for k in _METRIC_KEYS},
        "n_eval": str(record.get("metrics", {}).get("n", "—")),
    }
    if record.get("error"):
        # A failed method is more useful in the table than absent from it.
        row["Mode"] = f"{row['Mode']} (failed)"
    return row


def _markdown_table(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_no results found_\n"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r.get(h, "—") for h in headers) + " |")
    return "\n".join(lines) + "\n"


def discover_result_files(out_dir: Path) -> List[Path]:
    """kgfm cells first (sorted), then the baselines, in a stable order."""
    files = sorted(out_dir.glob("kgfm_*.json"))
    for name in _BASELINE_FILES:
        path = out_dir / name
        if path.is_file():
            files.append(path)
    return files


def load_records(out_dir: Path) -> List[Dict[str, Any]]:
    """Every method's JSON in this run, in table order."""
    records = []
    for path in discover_result_files(out_dir):
        record = _load_json(path)
        if record is None:
            print(f"[report] skipping unreadable {path}")
            continue
        records.append(record)
    return records


def build_table(out_dir: Path, out_path: Optional[Path] = None) -> str:
    rows = [_row(r) for r in load_records(out_dir)]
    table = _markdown_table(rows)
    print(table)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table)
        print(f"[report] wrote {out_path}")
    return table


def build_html(out_dir: Path, out_path: Optional[Path] = None,
               backend: str = "auto") -> str:
    """Render the full HTML report for a run directory."""
    from . import report_html

    records = load_records(out_dir)
    commands = read_commands(out_dir)
    page = report_html.render(
        out_dir,
        rows=[_row(r) for r in records],
        records=records,
        curves=discover_curves(out_dir),
        meta=_load_json(out_dir / "meta.json"),
        kg_stats=_load_json(out_dir / "chembl_kg_stats.json"),
        backend=backend,
        embeddings=discover_embeddings(out_dir),
        commands=commands,
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(page)
        print(f"[report] wrote {out_path}")
    return page


def list_runs(results_root: str) -> None:
    """Show every run and which methods have results in it."""
    root = Path(results_root)
    if not root.is_dir():
        raise SystemExit(f"No results directory at {root}.")
    latest = (root / "latest").resolve() if (root / "latest").exists() else None

    runs = sorted(
        (d for d in root.iterdir() if d.is_dir() and not d.is_symlink()),
        reverse=True,
    )
    if not runs:
        print(f"[report] no runs under {root}")
        return
    print(f"{'run':44s} {'methods':28s} table")
    for run_dir in runs:
        found = discover_result_files(run_dir)
        names = []
        for path in found:
            stem = path.stem
            names.append("kgfm" if stem.startswith("kgfm_") else stem)
        summary = ", ".join(dict.fromkeys(names)) or "-"
        marker = " *" if latest and run_dir.resolve() == latest else ""
        has_table = "yes" if (run_dir / "table.md").is_file() else "no"
        print(f"{run_dir.name + marker:44s} {summary:28s} {has_table}")
    if latest:
        print("\n* = latest")


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out-dir", default="latest",
                   help="Run directory to report on. 'latest', a bare "
                        "timestamp, or a path (default: latest).")
    p.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT,
                   help="Where run directories live.")
    p.add_argument("--out", default=None,
                   help="Where to write the table "
                        "(default: <run-dir>/table.md; '-' to only print).")
    p.add_argument("--html-out", default=None,
                   help="Where to write the HTML report "
                        "(default: <run-dir>/report.html).")
    p.add_argument("--no-html", action="store_true",
                   help="Skip the HTML report; write only the Markdown table.")
    p.add_argument("--charts", default="auto",
                   choices=["auto", "plotly", "matplotlib", "svg"],
                   help="Chart backend for the HTML report. 'auto' prefers "
                        "plotly (interactive), then matplotlib, then the "
                        "dependency-free built-in SVG.")
    p.add_argument("--list", action="store_true",
                   help="List the runs found under --results-root and exit.")


def run_from_args(args: argparse.Namespace) -> Optional[str]:
    if args.list:
        list_runs(args.results_root)
        return None
    try:
        out_dir = resolve_run_dir(args.results_root, args.out_dir)
    except FileNotFoundError as exc:
        raise SystemExit(f"{exc}\nTry `kgfm report --list`.") from exc

    if args.out == "-":
        out_path = None
    else:
        out_path = Path(args.out) if args.out else out_dir / "table.md"
    table = build_table(out_dir, out_path)

    if not args.no_html:
        html_path = Path(args.html_out) if args.html_out else out_dir / "report.html"
        build_html(out_dir, html_path, backend=args.charts)
        # Recorded after rendering so the page does not list the very
        # invocation that produced it; the next render picks it up.
        if (out_dir / "meta.json").is_file():
            RunLogger(out_dir).record_command(note="rendered this report")
    return table


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="kgfm report", description=__doc__.splitlines()[0]
    )
    add_arguments(p)
    run_from_args(p.parse_args(argv))


if __name__ == "__main__":
    main()
