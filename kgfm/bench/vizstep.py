"""Benchmark step: project every trained cell's embedding space to 2D.

Thin wrapper over `kgfm viz` — one projection per checkpoint directory the
sweep produced, written as ``embeddings_<tag>.json`` for `kgfm report` to plot.
Runs in-process: it is just an encode plus a dimensionality reduction, and
doing them one at a time keeps peak GPU memory to a single model.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .. import viz
from ..data import read_file_list
from ..runs import RunLogger
from .config import BenchConfig


def run_step(cfg: BenchConfig, out_dir: Path, logger: RunLogger) -> List[Path]:
    ckpt_dirs = sorted(out_dir.glob("kgfm_ckpts_*"))
    if not ckpt_dirs:
        logger.log("skip viz (no kgfm checkpoints in this run)")
        return []

    files = read_file_list(cfg.test_list)
    written: List[Path] = []
    with logger.step("viz"):
        for ckpt_dir in ckpt_dirs:
            tag = ckpt_dir.name.replace("kgfm_ckpts_", "")
            out_path = out_dir / f"embeddings_{tag}.json"
            if cfg.resume and out_path.is_file():
                print(f"[viz] skip {tag} (resume: {out_path.name} exists)")
                written.append(out_path)
                continue
            # Same preference order the final eval uses, so the plotted space
            # is the one the reported metrics came from.
            ckpt = next(
                (ckpt_dir / n for n in ("best.pt", "final.pt", "last.pt")
                 if (ckpt_dir / n).is_file()), None,
            )
            if ckpt is None:
                print(f"[viz] no checkpoint in {ckpt_dir}")
                continue
            print(f"[viz] {tag}: projecting {ckpt.name}")
            try:
                record = viz.build_projection(
                    str(ckpt), files,
                    reducer=cfg.viz_reducer,
                    max_points=cfg.viz_max_points,
                    seed=cfg.seed,
                )
            except Exception as exc:                        # noqa: BLE001
                # A failed projection must not cost the run its results.
                print(f"[viz] {tag} failed: {type(exc).__name__}: {exc}")
                continue
            import json

            out_path.write_text(json.dumps(record, indent=2))
            print(f"[viz] wrote {out_path}")
            logger.record_command(
                kind="step", tag=tag,
                command=(f"kgfm viz --ckpt {ckpt} --test-list {cfg.test_list} "
                         f"--reducer {cfg.viz_reducer} "
                         f"--max-points {cfg.viz_max_points} --seed {cfg.seed}"),
                note="embedding projection",
            )
            written.append(out_path)
    return written
