#!/usr/bin/env python
"""Per-size learning-rate probe for the scaling study.

WHY THIS EXISTS
A scaling curve is only a statement about *capacity* if every size was trained
at a rate that suits it. One global rate does not do that: `train.default_lr`
hands every random-init encoder the same 1e-4, and the optimal rate for a 4M
model is not the optimal rate for a 110M one — smaller models tolerate (and
need) more. Without this probe the size axis measures "capacity, confounded
with how well the fixed rate happened to fit", which is exactly the confound
that flattened the 2026-08-24 fit.

WHAT IT DOES
Runs `kgfm train` once per (encoder, lr), short, with the *same* cell settings
the real study uses, and reports the best validation loss each reached. The
winners go into a config's `cells: <tag>: lr:` block.

    python benchmark_scaling/lr_probe.py \
        --encoders scratch-tiny,scratch-mini,scratch-small \
        --lrs 3e-5,1e-4,3e-4,1e-3 --gpus 0,1

CAVEAT, and it is a real one: a rate that wins at 3000 steps is not
necessarily the rate that wins at 40000. Short probes favour large rates,
because the run ends before the large-rate curve flattens out. Read the result
as a bracket ("the optimum is around here"), not as a decimal, and prefer the
smaller of two rates that are within noise of each other.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kgfm import envs  # noqa: E402

# `[valid eval @ step 500] {'mrr': ..., 'loss': ...}` — the same line
# kgfm/report.py and kgfm/scaling/points.py parse. If train.py's format
# changes, all three move together.
_EVAL_RE = re.compile(r"\[(\w+) eval @ step (\d+)\]\s*(\{.*\})")


def _csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def run_one(encoder: str, lr: float, gpu: str, args: argparse.Namespace,
            log_dir: Path) -> Dict[str, Any]:
    """Train one (encoder, lr) cell and return its validation history."""
    tag = f"{encoder}_lr{lr:g}"
    log_path = log_dir / f"{tag}.log"
    ckpt_dir = log_dir / "ckpts" / tag
    # envs.kgfm_cmd already passes -u (the child's stdout is a pipe, which
    # Python would otherwise block-buffer), so this does not add its own.
    cmd = list(envs.kgfm_cmd(args.conda_env)) + [
        "train",
        "--encoder", encoder,
        "--head", args.head,
        "--proj-dim", str(args.proj_dim),
        "--lr", repr(lr),
        "--batch-size", str(args.batch_size),
        "--max-steps", str(args.steps),
        "--eval-every", str(args.eval_every),
        "--log-every", str(args.eval_every),
        "--valid-loss-batches", str(args.valid_loss_batches),
        "--eval-n-triples", str(args.eval_n_triples),
        "--eval-pool-size", str(args.eval_n_triples),
        # The probe only needs the validation history; a final test pass would
        # double the cost of the cheap cells for a number nothing reads.
        "--final-n-triples", "0",
        "--encoder-weight-decay", str(args.encoder_weight_decay),
        "--head-weight-decay", str(args.head_weight_decay),
        "--train-list", args.train_list,
        "--valid-list", args.valid_list,
        "--test-list", args.test_list,
        "--ckpt-dir", str(ckpt_dir),
        # One write at the end instead of every 1000 steps: the checkpoints are
        # thrown away below and writing them is a large share of a tiny cell.
        "--ckpt-every", str(args.steps + 1),
        "--seed", str(args.seed),
    ]
    # One probe cell per GPU means two trainers share the box. Left alone,
    # torch sizes its intra-op pool to every core in each process and they
    # spend the difference fighting each other.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               OMP_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads))
    t0 = time.time()
    history: List[Dict[str, Any]] = []
    with log_path.open("w") as fh:
        fh.write("$ " + " ".join(cmd) + f"\n# CUDA_VISIBLE_DEVICES={gpu}\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, text=True,
                                bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            m = _EVAL_RE.search(line)
            if m and m.group(1) == "valid":
                try:
                    metrics = ast.literal_eval(m.group(3))
                except (ValueError, SyntaxError):
                    continue
                if isinstance(metrics, dict) and "loss" in metrics:
                    history.append({"step": int(m.group(2)),
                                    "loss": float(metrics["loss"]),
                                    # eval.py spells it "MRR"; keep the fallback in case that changes.
                                    "mrr": metrics.get("MRR",
                                                       metrics.get("mrr"))})
        rc = proc.wait()

    if not args.keep_ckpts:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    losses = [h["loss"] for h in history]
    return {
        "encoder": encoder, "lr": lr, "gpu": gpu, "returncode": rc,
        "seconds": round(time.time() - t0, 1),
        "log": str(log_path), "history": history,
        "best_loss": min(losses) if losses else None,
        "final_loss": losses[-1] if losses else None,
        # A cell that never left ln(B) collapsed; report it rather than
        # letting it win a comparison by being flat.
        "best_mrr": max((h["mrr"] for h in history if h["mrr"] is not None),
                        default=None),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoders", type=_csv,
                   default=["scratch-tiny", "scratch-mini", "scratch-small",
                            "scratch-medium", "scratch-base"])
    p.add_argument("--lrs", type=_csv, default=["3e-5", "1e-4", "3e-4", "1e-3"])
    p.add_argument("--gpus", type=_csv, default=["0"],
                   help="Physical GPU ids; one job runs per id at a time.")
    p.add_argument("--out-dir", default="benchmark_scaling/results/lr_probe")
    p.add_argument("--conda-env", default="kgfm")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--head", default="linear")
    p.add_argument("--eval-every", type=int, default=600)
    p.add_argument("--valid-loss-batches", type=int, default=8)
    p.add_argument("--eval-n-triples", type=int, default=1000)
    p.add_argument("--encoder-weight-decay", type=float, default=0.01)
    p.add_argument("--head-weight-decay", type=float, default=0.0)
    p.add_argument("--train-list", default="list_chembl/train.txt")
    p.add_argument("--valid-list", default="list_chembl/valid.txt")
    p.add_argument("--test-list", default="list_chembl/test.txt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--select", default="loss", choices=["loss", "mrr"],
                   help="Criterion the pasteable cells: block uses. Leave at "
                        "loss — a learning rate is tuned against what the "
                        "optimizer minimizes. The MRR column is a "
                        "cross-check, not a selector, even though the scaling "
                        "report fits 1-MRR.")
    p.add_argument("--keep-ckpts", action="store_true")
    p.add_argument("--threads", type=int, default=8,
                   help="OMP/MKL threads per concurrently running cell.")
    a = p.parse_args(argv)

    out_dir = Path(a.out_dir)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    jobs = [(e, float(lr)) for e in a.encoders for lr in a.lrs]
    print(f"lr probe: {len(a.encoders)} encoders x {len(a.lrs)} rates = "
          f"{len(jobs)} cells, {a.steps} steps each, on GPUs {a.gpus}",
          flush=True)

    work: "queue.Queue[tuple]" = queue.Queue()
    for job in jobs:
        work.put(job)
    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                encoder, lr = work.get_nowait()
            except queue.Empty:
                return
            print(f"[gpu {gpu}] start {encoder} lr={lr:g}", flush=True)
            res = run_one(encoder, lr, gpu, a, out_dir / "logs")
            with lock:
                results.append(res)
                (out_dir / "lr_probe.json").write_text(
                    json.dumps(results, indent=2))
            print(f"[gpu {gpu}] done  {encoder} lr={lr:g} "
                  f"best_loss={res['best_loss']} rc={res['returncode']} "
                  f"({res['seconds']:.0f}s)", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in a.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    text = summarize(results, a.encoders,
                     [float(x) for x in a.lrs], a.select,
                     a.batch_size)
    print(text)
    (out_dir / "summary.txt").write_text(text)
    return 0


# A cell that never escaped the collapsed fixed point sits at the objective's
# ceiling: every embedding identical -> every logit equal -> loss exactly
# ln(B). Those cells must be excluded from the winner search, not ranked. All
# four of scratch-base's first-grid cells landed there and differed only in the
# 8th decimal, so `min` "chose" a rate on pure float noise.
COLLAPSE_RTOL = 0.005


def is_collapsed(loss: Optional[float], batch_size: int) -> bool:
    """True if ``loss`` is indistinguishable from the ln(B) random-guess line."""
    if loss is None:
        return False
    import math
    ceiling = math.log(batch_size)
    return abs(loss - ceiling) <= COLLAPSE_RTOL * ceiling


def summarize(results: List[Dict[str, Any]], encoders: List[str],
              lrs: List[float], select: str = "loss",
              batch_size: int = 256) -> str:
    """rows=encoder, cols=lr tables of both criteria, plus the winners.

    Both criteria are printed because they can disagree and the disagreement
    matters: the loss is what the learning rate is directly optimizing, while
    `1 - MRR` is the axis the scaling report fits (the in-batch loss is bounded
    by ln(B) and floored by repeated tails, so it compresses differences that
    ranking does not). ``select`` decides which one the pasteable block uses.

    Collapsed cells print as `collapse` and are never selected. If a whole row
    collapsed, the row says so instead of naming a rate — the grid is above
    that model's usable range and the answer is to extend it downward, not to
    pick the least-bad number in it.
    """
    by = {(r["encoder"], r["lr"]): r for r in results}
    w = max([len(e) for e in encoders] + [8])

    def alive(enc: str, lr: float) -> bool:
        r = by.get((enc, lr))
        return bool(r) and not is_collapsed(r.get("best_loss"), batch_size)

    def block(key: str, title: str, better) -> List[str]:
        out = ["", title, ""]
        out.append(" " * w + "".join(f"{lr:>12g}" for lr in lrs) + "     best")
        for enc in encoders:
            cells = []
            for lr in lrs:
                r = by.get((enc, lr))
                v = r.get(key) if r else None
                if v is None:
                    cells.append(f"{'--':>12}")
                elif not alive(enc, lr):
                    cells.append(f"{'collapse':>12}")
                else:
                    cells.append(f"{v:12.4f}")
            cand = [(by[(enc, lr)][key], lr) for lr in lrs
                    if alive(enc, lr) and by[(enc, lr)].get(key) is not None]
            out.append(f"{enc:<{w}}" + "".join(cells)
                       + (f"  {better(cand)[1]:g}" if cand
                          else "  all collapsed"))
        return out

    import math
    out = [f"(collapse = validation loss within {COLLAPSE_RTOL:.1%} of "
           f"ln(B)={math.log(batch_size):.4f}, the random-guess line)"]
    out += block("best_loss", "best validation loss (lower is better)", min)
    out += block("best_mrr", "best validation MRR (higher is better)", max)

    key, better = ("best_loss", min) if select == "loss" else ("best_mrr", max)
    out += ["", f"cells: block for the scaling config (selected by {select})", ""]
    edge = []
    for enc in encoders:
        cand = [(by[(enc, lr)][key], lr) for lr in lrs
                if alive(enc, lr) and by[(enc, lr)].get(key) is not None]
        if not cand:
            out.append(f"  # {enc}: every rate in this grid collapsed — "
                       f"extend the grid downward")
            continue
        win = better(cand)[1]
        out.append(f"  {enc}:\n    lr: {win:g}")
        if win == min(lrs) and len(lrs) > 1:
            edge.append(enc)
    if edge:
        out += ["",
                "# WARNING: these won at the bottom of the grid, so the value "
                "is a bound,",
                "# not an optimum: " + ", ".join(edge),
                "# Extend the grid downward and re-merge before using it."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
