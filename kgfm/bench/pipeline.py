"""The kgfm side of the ChEMBL benchmark: prep -> sweep.

This owns the *run* — flags, run directory, meta.json, ordering — and
delegates each step to its module. It covers kgfm only. The baselines are
separate commands (`kgfm-ultra`, `kgfm-motif`) pointed at the same run
directory, and `kgfm report` collects whatever ended up there:

    kgfm bench run --config benchmarks/config_large.yaml   # creates the run dir, trains
    kgfm-ultra --out-dir latest       # zero-shot ULTRA into the same dir
    kgfm-motif --out-dir latest
    kgfm report --out-dir latest      # one table over all of it

Each step can also be invoked on its own against an existing run directory
(`kgfm bench sweep --out-dir ...`), which is how you re-run just the piece
that failed.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from .. import envs
from ..runs import RunLogger, create_run_dir, resolve_run_dir, run_timestamp
from . import prep, sweep, vizstep
from .config import BenchConfig


def _git_info() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        )
        return {"git_rev": rev, "git_dirty_files": len(dirty.splitlines())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_rev": "n/a", "git_dirty_files": 0}


def _gpu_line() -> str:
    if not shutil.which("nvidia-smi"):
        return "n/a"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.splitlines()[0].strip() if out.strip() else "n/a"
    except (OSError, subprocess.CalledProcessError, IndexError):
        return "n/a"


def write_meta(cfg: BenchConfig, out_dir: Path, timestamp: str) -> None:
    """Record how this run was configured. Never overwritten on resume."""
    meta_path = out_dir / "meta.json"
    if cfg.resume and meta_path.is_file():
        return
    env_info = envs.describe(cfg.conda_env)
    meta = {
        "benchmark": "chembl",
        "timestamp_utc": timestamp,
        "host": platform.node(),
        **_git_info(),
        "conda_env": cfg.conda_env,
        "python": env_info.get("python", "n/a"),
        "torch": env_info.get("torch", "n/a"),
        "gpu": _gpu_line(),
        # kgfm-side parameters only; the baselines record their own in
        # ultra.json / motif.json.
        "params": cfg.as_meta(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def open_run(cfg: BenchConfig) -> Path:
    """Create or resolve the run directory this invocation writes into."""
    if cfg.resume:
        return resolve_run_dir(cfg.results_root, cfg.resume)
    return create_run_dir(cfg.results_root, cfg.run_label)


def run(cfg: BenchConfig) -> Path:
    cfg.validate()
    out_dir = open_run(cfg)
    logger = RunLogger(out_dir)
    write_meta(cfg, out_dir, run_timestamp(out_dir))

    logger.record_command(note="kgfm side of the benchmark")
    logger.log(f"==> kgfm bench run{' (RESUME)' if cfg.resume else ''}")
    logger.log(f"    out_dir    = {out_dir}")
    logger.log(f"    config     = {cfg.config_file or '(defaults)'}")
    logger.log(f"    conda_env  = {cfg.conda_env} ({envs.env_python(cfg.conda_env)})")
    logger.log(f"    gpu        = {_gpu_line()}")
    logger.log(f"    caps       = train={cfg.max_train} valid={cfg.max_valid} "
               f"test={cfg.max_test}")
    logger.log(f"    sweep      = encoders={cfg.encoders} freezes={cfg.freezes} "
               f"protocols={cfg.protocols} nproc={cfg.nproc}")
    # One line per cell rather than one line of global settings: with `cells:`
    # in play there is no single batch size or step count to print, and the
    # resolved value is the thing worth recording.
    for tag in cfg.cell_tags():
        cell = cfg.resolve_cell(tag)
        extra = "".join(
            f" {key}={cell[key]}" for key in (
                "proj_dim", "lr", "loss", "weight_decay",
                "encoder_weight_decay", "head_weight_decay",
                "encoder_dropout", "head_dropout",
            ) if cell[key] is not None
        )
        logger.log(f"    cell {tag:<19} max_steps={cell['max_steps']} "
                   f"batch_size={cell['batch_size']} "
                   f"eval_every={cell['eval_every']}{extra}"
                   + ("   <- cells:" if cfg.cells.get(tag) else ""))
    if cfg.skip:
        logger.log(f"    skipping   = {', '.join(cfg.skip)}")
    logger.log("")

    # "skip prep" means reuse the existing KG rather than skip outright — the
    # baselines still need its stats recorded in the run dir.
    prep.run_step(cfg, out_dir, logger, reuse="prep" in cfg.skip)

    if "sweep" in cfg.skip:
        logger.log("skip sweep (--skip sweep)")
    else:
        sweep.run_step(cfg, out_dir, logger)

    if "viz" in cfg.skip:
        logger.log("skip viz (--skip viz)")
    else:
        vizstep.run_step(cfg, out_dir, logger)

    logger.log("")
    logger.log(f"==> done. kgfm results in {out_dir}")
    logger.log(f"    Latest pointer: {Path(cfg.results_root) / 'latest'} -> {out_dir.name}")
    logger.log("    Baselines + table:")
    logger.log(f"      kgfm-ultra --out-dir {out_dir.name}")
    logger.log(f"      kgfm-motif --out-dir {out_dir.name}")
    logger.log(f"      kgfm report --out-dir {out_dir.name}")
    return out_dir
