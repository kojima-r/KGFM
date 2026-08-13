"""`kgfm bench ...` argument wiring.

Scale settings come from a YAML file
(``--config benchmarks/config_large.yaml``), not from code.

Scope is kgfm's own benchmark: prep and the sweep. ULTRA / MOTIF live in
`kgfm-ultra` / `kgfm-motif`, and the table in `kgfm report`.

Every step takes ``--out-dir`` so it can be pointed at an existing run, and
the shared config flags default to ``None`` so ``--config`` can fill them in
without clobbering anything the user typed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from ..heads import HEADS
from ..losses import DEFAULT_LOSS, LOSSES
from ..runs import RunLogger, resolve_run_dir
from . import cell, pipeline, prep, sweep
from .config import STEPS, BenchConfig, available_configs, build_config


def _csv(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]


def _add_shared(p: argparse.ArgumentParser, *, with_out_dir: bool = True) -> None:
    """Flags every bench subcommand understands."""
    if with_out_dir:
        p.add_argument("--out-dir", default=None,
                       help="Run directory to work in. Accepts 'latest', a "
                            "bare timestamp, or a path. Required for the "
                            "individual steps.")
    p.add_argument("--config", default=None, metavar="PATH",
                   help="YAML settings file (see benchmarks/config_*.yaml). A "
                        "bare name resolves to benchmarks/config_<name>.yaml. "
                        "Flags given on the command line override it.")
    p.add_argument("--conda-env", default=None,
                   help="Conda env every child process runs in (default kgfm).")
    p.add_argument("--results-root", default=None,
                   help="Where run directories live "
                        "(default benchmarks/results/chembl).")
    p.add_argument("--kg-dir", default=None,
                   help="Prepared entity-ID KG directory (the baselines read "
                        "this too; pass them the same --kg-dir).")
    p.add_argument("--resume", nargs="?", const="latest", default=None,
                   help="Resume an existing run instead of starting one. "
                        "With no value, resumes 'latest'. Each step skips "
                        "itself when its own output is already present.")


def _add_data(p: argparse.ArgumentParser) -> None:
    p.add_argument("--train-list", default=None)
    p.add_argument("--valid-list", default=None)
    p.add_argument("--test-list", default=None)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-valid", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--strict-transductive", action="store_true", default=None,
                   help="Drop valid/test triples with entities unseen in "
                        "train. ChEMBL shards have near-disjoint entity sets, "
                        "so this leaves almost nothing held out — the default "
                        "is inductive.")


def _add_sweep(p: argparse.ArgumentParser) -> None:
    p.add_argument("--encoders", type=_csv, default=None,
                   help="Comma-separated: ngram,transformer")
    p.add_argument("--heads", type=_csv, default=None,
                   help=f"Comma-separated projection heads to sweep "
                        f"({','.join(HEADS)}). More than one adds a _<head> "
                        f"segment to every cell tag.")
    p.add_argument("--freezes", type=_csv, default=None,
                   help="Comma-separated: off,on. 'on' needs a head with "
                        "parameters (proj-dim, or head=linear/mlp).")
    p.add_argument("--protocols", type=_csv, default=None,
                   help="Comma-separated: pooled,filtered")
    # Everything from here down is a cell-level setting, and passing it on the
    # command line is a *single global override*: it is applied to every cell
    # after (and therefore wins over) any `cells:` block in the config file.
    # To vary one of these per cell, put it under `cells: <tag>:` in the YAML
    # — that is what the two levels are for. `--transformer-batch-size` used
    # to live here and is now `cells: transformer: batch_size:`.
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for EVERY cell. Per-cell values belong in "
                        "the config file under `cells:`.")
    p.add_argument("--proj-dim", type=int, default=None)
    p.add_argument("--per-device-train-batch-size", type=int, default=None)
    p.add_argument("--per-device-eval-batch-size", type=int, default=None)
    p.add_argument("--gradient-accumulation-steps", type=int, default=None)
    p.add_argument("--nproc", type=int, default=None,
                   help="GPUs per kgfm cell; >1 launches the training pass "
                        "under torchrun.")
    p.add_argument("--master-port", type=int, default=None)
    p.add_argument("--log-every", type=int, default=None,
                   help="Steps between loss log lines (these become the "
                        "training curve in `kgfm report`).")
    p.add_argument("--eval-every", type=int, default=None,
                   help="Steps between in-loop validation passes (default "
                        "1000, i.e. never for short runs).")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate for every cell. Default is chosen per "
                        "encoder (1e-3 ngram, 3e-5 transformer fine-tune).")
    p.add_argument("--loss", default=None, choices=list(LOSSES),
                   help="Training objective for every cell (default: "
                        f"{DEFAULT_LOSS}). See kgfm/losses.py.")
    p.add_argument("--loss-temperature", type=float, default=None,
                   help="Softmax sharpness for --loss contrastive.")
    p.add_argument("--weight-decay", type=float, default=None,
                   help="Base weight decay for every cell; the two halves "
                        "below fall back to it.")
    p.add_argument("--encoder-weight-decay", type=float, default=None,
                   help="Weight decay for the text encoder only.")
    p.add_argument("--head-weight-decay", type=float, default=None,
                   help="Weight decay for the projection head only.")
    p.add_argument("--encoder-dropout", type=float, default=None,
                   help="Dropout inside the encoder. Unset keeps the "
                        "pretrained model's own value.")
    p.add_argument("--head-dropout", type=float, default=None,
                   help="Dropout on the encoder output, before the head.")
    p.add_argument("--no-mask-duplicate-tails", dest="mask_duplicate_tails",
                   action="store_const", const=False, default=None,
                   help="Keep duplicate tails as in-batch negatives "
                        "(masking them is the default).")
    p.add_argument("--max-rows-per-file", type=int, default=None,
                   help="Rows to read from each TSV before moving to the next. "
                        "Unset reads to the end — on ChEMBL (10M rows/file) a "
                        "run then trains on ~1 file per dataloader worker.")
    p.add_argument("--valid-loss-batches", type=int, default=None,
                   help="Held-out batches averaged into the validation loss "
                        "plotted against the training loss (0 disables).")
    p.add_argument("--n-eval-triples", type=int, default=None)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--max-filter-tails", type=int, default=None)
    p.add_argument("--max-filter-rows", type=int, default=None)
    p.add_argument("--viz-reducer", default=None,
                   choices=["auto", "pca", "umap"],
                   help="Dimensionality reduction for the embedding plots.")
    p.add_argument("--viz-max-points", type=int, default=None,
                   help="Points per embedding plot (split h/t).")


def build_parser(sub: argparse._SubParsersAction) -> None:
    bench = sub.add_parser(
        "bench", help="kgfm's ChEMBL benchmark (prep + sweep)",
        description=(
            "Run kgfm's side of the ChEMBL benchmark, or one step of it. "
            "The baselines are separate commands (kgfm-ultra, kgfm-motif) "
            "and the comparison table is `kgfm report`."
        ),
    )
    steps = bench.add_subparsers(dest="bench_command", metavar="<step>")

    # --- full pipeline ---
    run_p = steps.add_parser(
        "run", help="prep + sweep into a fresh run directory")
    run_p.add_argument("--run-label", default=None,
                       help="Postfix for the run directory name.")
    run_p.add_argument("--skip", action="append", type=_csv, default=None,
                       metavar="STEP",
                       help=f"Skip a step ({'|'.join(STEPS)}). Repeatable / "
                            "comma-separated. 'prep' means reuse the existing KG.")
    _add_shared(run_p, with_out_dir=False)
    _add_data(run_p)
    _add_sweep(run_p)
    run_p.set_defaults(func=_cmd_run)

    # --- individual steps ---
    prep_p = steps.add_parser("prep", help="build the entity-ID ChEMBL KG")
    _add_shared(prep_p)
    _add_data(prep_p)
    prep_p.set_defaults(func=_cmd_prep)

    sweep_p = steps.add_parser("sweep", help="kgfm sweep over encoders/freezes/protocols")
    _add_shared(sweep_p)
    _add_data(sweep_p)
    _add_sweep(sweep_p)
    sweep_p.set_defaults(func=_cmd_sweep)

    configs_p = steps.add_parser(
        "configs", help="list benchmarks/config_*.yaml and their settings")
    configs_p.set_defaults(func=_cmd_configs)

    # --- one sweep cell (spawned by `sweep`; rarely run by hand) ---
    cell_p = steps.add_parser(
        "cell", help="train+evaluate one kgfm cell (used internally by sweep)")
    cell.add_arguments(cell_p)
    cell_p.set_defaults(func=_cmd_cell)

    bench.set_defaults(func=lambda args: bench.print_help())


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------

_CONFIG_FIELDS = {f for f in BenchConfig().__dict__}


def _config_from(args: argparse.Namespace) -> BenchConfig:
    overrides: Dict[str, Any] = {
        k: v for k, v in vars(args).items() if k in _CONFIG_FIELDS
    }
    if isinstance(getattr(args, "skip", None), list):
        # --skip is `append` of comma-split lists; flatten it.
        overrides["skip"] = [s for group in args.skip for s in group]
    cfg = build_config(overrides, getattr(args, "config", None))
    cfg.validate()
    return cfg


def _step_context(args: argparse.Namespace) -> tuple:
    """Resolve (cfg, out_dir, logger) for a single-step invocation."""
    cfg = _config_from(args)
    target = args.out_dir or cfg.resume or "latest"
    try:
        out_dir = resolve_run_dir(cfg.results_root, target)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\nPass --out-dir, or start a run with `kgfm bench run`."
        ) from exc
    return cfg, out_dir, RunLogger(out_dir)


def _cmd_run(args: argparse.Namespace) -> None:
    pipeline.run(_config_from(args))


def _cmd_prep(args: argparse.Namespace) -> None:
    cfg, out_dir, logger = _step_context(args)
    prep.run_step(cfg, out_dir, logger)


def _cmd_sweep(args: argparse.Namespace) -> None:
    cfg, out_dir, logger = _step_context(args)
    sweep.run_step(cfg, out_dir, logger)


def _cmd_cell(args: argparse.Namespace) -> None:
    cell.run_from_args(args)


def _cmd_configs(args: argparse.Namespace) -> None:
    from .config import CONFIG_DIR, load_config_file

    names = available_configs()
    if not names:
        print(f"No YAML settings files under {CONFIG_DIR}/")
        return
    base = BenchConfig()
    for name in names:
        print(f"[{CONFIG_DIR}/{name}.yaml]")   # name already carries the prefix
        overrides = load_config_file(name)
        # `cells` is a nested mapping; printing it inline is unreadable, and it
        # is the part a reader most wants to see, so give it its own block
        # after the flat settings.
        cells = overrides.pop("cells", {})
        for key, value in overrides.items():
            if key == "config_file":
                continue
            print(f"    {key:26s} {getattr(base, key, None)!r:>24} -> {value!r}")
        for tag, block in cells.items():
            print(f"    cells: {tag}")
            for key, value in sorted(block.items()):
                print(f"        {key:22s} {getattr(base, key, None)!r:>24} -> {value!r}")
        print()
